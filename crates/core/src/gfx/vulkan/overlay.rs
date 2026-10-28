//! The editor's UI, drawn over the finished frame, with liquid glass under its
//! translucent surfaces: a blur of what is beneath, bent and fringed at the
//! edge like a lens and lit along its rim.
//!
//! A menu's glass blurs what is beneath it, and over a panel what is beneath
//! it is more UI — which has to have been drawn, to the image, before anything
//! can sample it. So the editor hands its frame over in phases (see
//! `editor::phases`): the panels, then the viewport's floating controls, then
//! menus and popups, then tooltips. Before a phase with glass, the target is
//! captured — blitted down to half and then quarter resolution — and each
//! surface's rect gets a blur of that copy, cut to its rounded corners, under
//! the translucent fill the phase then draws. A frame with no glass captures
//! nothing, and one with only the viewport's controls captures once.
//!
//! Recorded raw like the rest of the frame (`record.rs`) and submitted after
//! it, behind the frame's semaphore. The barriers are written here rather than
//! compiled: the graph sees the overlay as one `Raw` pass writing the swapchain
//! image, and what happens inside it — the image becoming a blit source and a
//! colour attachment again between phases — is this module's business, the way
//! a mip chain is `texture.rs`'s.
//!
//! The vertex layout, blend state, font swizzle and mesh shaders are ported
//! from egui_winit_vulkano 0.28 (Apache-2.0), which drew the editor before
//! this did and could not draw it in more than one piece.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use egui::epaint::{ImageDelta, Primitive};
use vulkano::command_buffer::{BlitImageInfo, ImageBlit, RenderingAttachmentInfo, RenderingInfo};
use vulkano::format::{Format as VkFormat, FormatFeatures};
use vulkano::image::sampler::Filter;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageAspects, ImageLayout, ImageSubresourceLayers};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::Scissor;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::sync::{AccessFlags, ImageMemoryBarrier, PipelineStages};

use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::ImageUsage;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, Blend, BlockKind, BufferId, Device, Encoder,
    GraphicsPipelineDesc, PipelineId, SamplerDesc, SamplerId, TextureDesc, TextureDimension,
    TextureSwizzle, TextureViewId, VertexAttribute, VertexFormat, VertexStream, Viewport,
};
use crate::gfx::shaders::ShaderId;

use super::context::VkContext;
use super::convert::engine_format;
use super::driver::VkEncoder;
use super::record::{self, Recorder};

/// One frame of the editor, as the renderer draws it.
#[derive(Default)]
pub struct OverlayFrame {
    pub pixels_per_point: f32,
    /// Every change not yet drawn, oldest first.
    pub textures: egui::TexturesDelta,
    /// In draw order. Glass in the first phase is ignored: nothing is beneath
    /// it yet.
    pub phases: Vec<OverlayPhase>,
    /// The transform gizmo's triangles, drawn by their own pass before the
    /// editor's UI rather than by this one — see `vulkan::gizmo`.
    pub gizmo: Vec<super::gizmo::GizmoVertex>,
    /// The meshes the selection outline is drawn round — see `vulkan::outline`.
    pub outline: Vec<super::outline::OutlineItem>,
}

#[derive(Default)]
pub struct OverlayPhase {
    /// Blurred into before any of the phase's meshes draw.
    pub glass: Vec<Glass>,
    pub primitives: Vec<egui::ClippedPrimitive>,
}

/// A translucent surface's outline, in points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glass {
    pub rect: egui::Rect,
    pub radius: f32,
}

/// egui's vertex, field for field.
#[derive(Pod, Zeroable, Vertex, Clone, Copy)]
#[repr(C)]
struct OverlayVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    tex_coords: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    color: [u8; 4],
}

impl OverlayVertex {
    /// This stream as the RHI describes it; a test holds it against the layout
    /// the `Vertex` derive produces.
    const STREAM: VertexStream = VertexStream {
        stride: 20,
        per_instance: false,
        attributes: &[
            VertexAttribute {
                name: "position",
                format: VertexFormat::Float2,
                offset: 0,
            },
            VertexAttribute {
                name: "tex_coords",
                format: VertexFormat::Float2,
                offset: 8,
            },
            VertexAttribute {
                name: "color",
                format: VertexFormat::Unorm8x4,
                offset: 16,
            },
        ],
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FrostPush {
    rect_min: [f32; 2],
    rect_max: [f32; 2],
    texel: [f32; 2],
    radius: f32,
}

impl FrostPush {
    fn words(&self) -> [u32; 7] {
        [
            self.rect_min[0],
            self.rect_min[1],
            self.rect_max[0],
            self.rect_max[1],
            self.texel[0],
            self.texel[1],
            self.radius,
        ]
        .map(f32::to_bits)
    }
}

/// Every phase's meshes, uploaded once: a vertex and an index buffer.
type Geometry = (BufferId, BufferId);

struct Texture {
    view: TextureViewId,
    set: BindGroupId,
}

/// The two downsampled copies of the target a capture writes, sized to it.
struct Backdrop {
    extent: [u32; 2],
    half: TextureViewId,
    quarter: TextureViewId,
    set: BindGroupId,
}

pub(super) struct OverlayPass {
    format: Format,
    pipeline: PipelineId,
    /// `None` where the target cannot be blitted from, which leaves the glass
    /// as plain translucency.
    frost: Option<PipelineId>,
    sampler: SamplerId,
    /// `R8G8_UNORM` swizzled up to premultiplied white where views can swizzle,
    /// at half the size of the `R8G8B8A8_SRGB` it falls back to.
    font_format: Format,
    textures: HashMap<egui::TextureId, Texture>,
    backdrop: Option<Backdrop>,
    /// Reused each frame so a frame that changes little allocates little.
    vertices: Vec<OverlayVertex>,
    indices: Vec<u32>,
    pixels: Vec<u8>,
}

/// The recorder under an encoder, for the barriers, blits and render-pass
/// brackets the RHI has no words for yet.
fn raw<'e>(encoder: &'e mut Encoder<'_>) -> &'e mut Recorder {
    let Encoder::Vulkan(encoder) = encoder;
    encoder.recorder()
}

impl OverlayPass {
    /// `capturable` is whether the target may be a transfer source — a
    /// swapchain has to have been created with that usage, and the surface may
    /// not offer it.
    pub(super) fn new(
        ctx: &VkContext,
        device: &Device,
        target_format: VkFormat,
        capturable: bool,
    ) -> Self {
        let physical = ctx.device.physical_device();
        let blittable = |format: VkFormat, features: FormatFeatures| {
            physical
                .format_properties(format)
                .is_ok_and(|properties| properties.optimal_tiling_features.contains(features))
        };
        let format = engine_format(target_format);
        let frost = (capturable
            && blittable(
                target_format,
                FormatFeatures::BLIT_SRC
                    | FormatFeatures::BLIT_DST
                    | FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR,
            ))
        .then(|| {
            device.create_graphics_pipeline(&GraphicsPipelineDesc {
                blend: Blend::Alpha,
                ..GraphicsPipelineDesc::fullscreen(
                    ShaderId::FullscreenVert,
                    ShaderId::FrostFrag,
                    &[format],
                )
            })
        });

        let swizzle = !ctx.device.enabled_extensions().khr_portability_subset
            || ctx.device.enabled_features().image_view_format_swizzle;
        let font_format = if swizzle
            && blittable(
                VkFormat::R8G8_UNORM,
                FormatFeatures::SAMPLED_IMAGE | FormatFeatures::TRANSFER_DST,
            ) {
            Format::Rg8Unorm
        } else {
            Format::Rgba8Srgb
        };

        Self {
            format,
            // egui's colours are premultiplied, and its target's alpha is
            // coverage accumulated front to back.
            pipeline: device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_streams: &[OverlayVertex::STREAM],
                blend: Blend::PremultipliedCoverage,
                ..GraphicsPipelineDesc::fullscreen(
                    ShaderId::OverlayVert,
                    ShaderId::OverlayFrag,
                    &[format],
                )
            }),
            frost,
            sampler: device.create_sampler(SamplerDesc {
                linear: true,
                mipmap_linear: false,
                repeat: false,
                compare: false,
                anisotropy: None,
                max_lod: Some(0.0),
            }),
            font_format,
            textures: HashMap::new(),
            backdrop: None,
            vertices: Vec::new(),
            indices: Vec::new(),
            pixels: Vec::new(),
        }
    }

    /// Record `frame` over `target`, which arrives and is left in `PresentSrc`
    /// — where the graph leaves the swapchain image, and where presenting
    /// needs it. Takes the frame's texture changes, which are applied once.
    pub(super) fn record(
        &mut self,
        device: &Device,
        recorder: &mut Recorder,
        target: &Arc<ImageView>,
        frame: &mut OverlayFrame,
    ) {
        let Device::Vulkan(driver) = device;
        let mut encoder = Encoder::Vulkan(VkEncoder::new(recorder, driver));
        let encoder = &mut encoder;

        let image = target.image().clone();
        let extent = [image.extent()[0], image.extent()[1]];
        let ppp = frame.pixels_per_point;
        let textures = std::mem::take(&mut frame.textures);

        self.upload_textures(device, encoder, &textures.set);
        let geometry = self.upload_geometry(device, frame);

        raw(encoder).image_barrier(present_to_color(image.clone()));
        let mut rendering = false;
        let mut cursor = (0, 0);
        for (index, phase) in frame.phases.iter().enumerate() {
            let glass: Vec<_> = phase
                .glass
                .iter()
                .filter(|_| index > 0)
                .filter_map(|glass| frost_draw(glass, ppp, extent))
                .collect();
            if let Some(frost) = self.frost
                && !glass.is_empty()
            {
                if rendering {
                    raw(encoder).end_rendering();
                }
                let backdrop = self.backdrop(device, extent);
                capture(device, raw(encoder), &image, backdrop);
                let texel = backdrop_texel(backdrop.extent);
                let set = backdrop.set;
                begin(raw(encoder), target);
                rendering = true;

                encoder.bind_pipeline(frost);
                encoder.set_viewport(viewport(extent));
                encoder.bind_group(0, set);
                for (scissor, push) in glass {
                    encoder.set_scissor(scissor.offset, scissor.extent);
                    encoder.push_constants(&FrostPush { texel, ..push }.words());
                    encoder.draw(3, 1, 0, 0);
                }
            }

            if phase.primitives.is_empty() {
                continue;
            }
            let Some(geometry) = geometry else {
                continue;
            };
            if !rendering {
                begin(raw(encoder), target);
                rendering = true;
            }
            cursor = self.draw_meshes(encoder, &phase.primitives, geometry, cursor, ppp, extent);
        }
        if rendering {
            raw(encoder).end_rendering();
        }
        raw(encoder).image_barrier(color_to_present(image));

        // The sets a recorded draw bound are held by the recording, so a
        // texture freed here outlives every frame still reading it.
        for id in &textures.free {
            if let Some(texture) = self.textures.remove(id) {
                device.destroy_bind_group(texture.set);
                device.destroy_texture(texture.view);
            }
        }
    }

    /// Returns the vertex and index cursors past this phase's meshes.
    fn draw_meshes(
        &self,
        encoder: &mut Encoder,
        primitives: &[egui::ClippedPrimitive],
        (vertices, indices): Geometry,
        (mut vertex, mut index): (u32, u32),
        ppp: f32,
        extent: [u32; 2],
    ) -> (u32, u32) {
        encoder.bind_pipeline(self.pipeline);
        encoder.bind_vertex_buffer(vertices);
        encoder.bind_index_buffer(indices);
        encoder.set_viewport(viewport(extent));
        encoder.push_constants(&[
            (extent[0] as f32 / ppp).to_bits(),
            (extent[1] as f32 / ppp).to_bits(),
            self.format.is_srgb() as u32,
        ]);

        let mut bound_texture = None;
        let mut bound_clip = None;
        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in primitives
        {
            let Primitive::Mesh(mesh) = primitive else {
                continue;
            };
            let (first_vertex, first_index) = (vertex, index);
            vertex += mesh.vertices.len() as u32;
            index += mesh.indices.len() as u32;
            let Some(scissor) = scissor(*clip_rect, ppp, extent) else {
                continue;
            };
            if mesh.indices.is_empty() {
                continue;
            }
            if bound_texture != Some(mesh.texture_id) {
                let Some(texture) = self.textures.get(&mesh.texture_id) else {
                    tracing::warn!(
                        "egui drew with {:?}, which was never uploaded",
                        mesh.texture_id
                    );
                    continue;
                };
                encoder.bind_group(0, texture.set);
                bound_texture = Some(mesh.texture_id);
            }
            if bound_clip != Some(scissor) {
                encoder.set_scissor(scissor.offset, scissor.extent);
                bound_clip = Some(scissor);
            }
            encoder.draw_indexed(
                mesh.indices.len() as u32,
                1,
                first_index,
                first_vertex as i32,
                0,
            );
        }
        (vertex, index)
    }

    /// Every phase's meshes in one vertex and one index buffer, in draw order.
    fn upload_geometry(&mut self, device: &Device, frame: &OverlayFrame) -> Option<Geometry> {
        let meshes = || {
            frame
                .phases
                .iter()
                .flat_map(|phase| &phase.primitives)
                .filter_map(|primitive| match &primitive.primitive {
                    Primitive::Mesh(mesh) => Some(mesh),
                    Primitive::Callback(_) => None,
                })
        };

        self.vertices.clear();
        self.vertices.extend(
            meshes()
                .flat_map(|mesh| &mesh.vertices)
                .map(|from| OverlayVertex {
                    position: [from.pos.x, from.pos.y],
                    tex_coords: [from.uv.x, from.uv.y],
                    color: from.color.to_array(),
                }),
        );
        self.indices.clear();
        self.indices
            .extend(meshes().flat_map(|mesh| &mesh.indices).copied());
        if self.vertices.is_empty() || self.indices.is_empty() {
            return None;
        }

        Some((
            device.upload_vertices(bytemuck::cast_slice(&self.vertices)),
            device.upload_block(BlockKind::Index, bytemuck::cast_slice(&self.indices)),
        ))
    }

    /// Record the frame's texture changes ahead of its draws.
    ///
    /// In the overlay's own command buffer rather than a blocking submit: the
    /// barrier before each copy waits on earlier fragment reads, which covers a
    /// previous frame's overlay still sampling a texture this one rewrites.
    fn upload_textures(
        &mut self,
        device: &Device,
        encoder: &mut Encoder,
        deltas: &[(egui::TextureId, ImageDelta)],
    ) {
        for (id, delta) in deltas {
            let format = match &delta.image {
                egui::ImageData::Color(_) => Format::Rgba8Srgb,
                egui::ImageData::Font(_) => self.font_format,
            };
            let [width, height] = delta.image.size().map(|side| side as u32);
            let texel = if format == Format::Rg8Unorm { 2 } else { 4 };
            self.pixels.clear();
            self.pixels
                .resize(width as usize * height as usize * texel, 0);
            pack(&delta.image, format, &mut self.pixels);
            let staging = device.upload_block(BlockKind::Staging, &self.pixels);

            let (view, offset, initial) = match delta.pos {
                Some([x, y]) => {
                    let Some(texture) = self.textures.get(id) else {
                        tracing::warn!("egui patched {id:?}, which was never uploaded");
                        continue;
                    };
                    (texture.view, [x as u32, y as u32], false)
                }
                None => {
                    // A replaced texture is retired on the device's frame delay,
                    // which is what keeps a frame still sampling it valid.
                    if let Some(old) = self.textures.remove(id) {
                        device.destroy_bind_group(old.set);
                        device.destroy_texture(old.view);
                    }
                    let texture = self.create_texture(device, format, [width, height]);
                    let view = texture.view;
                    self.textures.insert(*id, texture);
                    (view, [0, 0], true)
                }
            };
            encoder.write_texture(view, offset, [width, height], staging, initial);
        }
    }

    fn create_texture(&self, device: &Device, format: Format, extent: [u32; 2]) -> Texture {
        let view = device.create_texture_swizzled(
            &TextureDesc {
                format,
                dimension: TextureDimension::D2,
                extent: [extent[0], extent[1], 1],
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::Sample1,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                memoryless: false,
            },
            if format == Format::Rg8Unorm {
                TextureSwizzle::CoverageToWhite
            } else {
                TextureSwizzle::Identity
            },
        );
        let set = device.create_persistent_bind_group(
            self.pipeline,
            0,
            &[Binding {
                binding: 0,
                resource: BindingResource::SampledTexture {
                    view,
                    sampler: self.sampler,
                },
            }],
        );
        Texture { view, set }
    }

    /// The capture targets for `extent`, rebuilt when the window resizes.
    fn backdrop(&mut self, device: &Device, extent: [u32; 2]) -> &Backdrop {
        if self
            .backdrop
            .as_ref()
            .is_none_or(|backdrop| backdrop.extent != extent)
        {
            if let Some(old) = self.backdrop.take() {
                device.destroy_bind_group(old.set);
                device.destroy_texture(old.half);
                device.destroy_texture(old.quarter);
            }
            let (half_extent, quarter_extent) = backdrop_extents(extent);
            let image = |extent: [u32; 2], usage| {
                device.create_texture(&TextureDesc {
                    format: self.format,
                    dimension: TextureDimension::D2,
                    extent: [extent[0], extent[1], 1],
                    mip_levels: 1,
                    array_layers: 1,
                    samples: SampleCount::Sample1,
                    usage,
                    memoryless: false,
                })
            };
            let half = image(
                half_extent,
                // SAMPLED only because a texture's view needs a usage a view
                // can have; nothing samples this level.
                ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
            );
            let quarter = image(
                quarter_extent,
                ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            );
            let set = device.create_persistent_bind_group(
                self.frost.expect("only frosted frames capture"),
                0,
                &[Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: quarter,
                        sampler: self.sampler,
                    },
                }],
            );
            self.backdrop = Some(Backdrop {
                extent,
                half,
                quarter,
                set,
            });
        }
        self.backdrop.as_ref().expect("built above")
    }
}

/// Copy everything drawn so far into `backdrop`'s quarter-resolution image,
/// through the half one, and hand `target` back to rendering with its
/// contents.
fn capture(device: &Device, recorder: &mut Recorder, target: &Arc<Image>, backdrop: &Backdrop) {
    let Device::Vulkan(driver) = device;
    let half = &driver.vk_view(backdrop.half).image().clone();
    let quarter = &driver.vk_view(backdrop.quarter).image().clone();
    recorder
        .image_barrier(record::color_to_transfer_src(
            target.clone(),
            record::whole_image(target),
        ))
        .image_barrier(reuse_as_transfer_dst(half.clone()))
        .blit_image(downsample(target, half))
        .image_barrier(record::transfer_dst_to_src(
            half.clone(),
            record::whole_image(half),
        ))
        .image_barrier(reuse_as_transfer_dst(quarter.clone()))
        .blit_image(downsample(half, quarter))
        .image_barrier(record::to_shader_read(
            quarter.clone(),
            record::whole_image(quarter),
            ImageLayout::TransferDstOptimal,
        ))
        .image_barrier(ImageMemoryBarrier {
            src_stages: PipelineStages::ALL_TRANSFER,
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            dst_access: AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
            old_layout: ImageLayout::TransferSrcOptimal,
            new_layout: ImageLayout::ColorAttachmentOptimal,
            subresource_range: record::whole_image(target),
            ..ImageMemoryBarrier::image(target.clone())
        });
}

/// A capture image about to be rewritten. Its old contents do not matter, but
/// the previous capture's reads of it — a blit, or the glass sampling it — have
/// to finish first.
fn reuse_as_transfer_dst(image: Arc<Image>) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::FRAGMENT_SHADER | PipelineStages::ALL_TRANSFER,
        ..record::to_transfer_dst(
            image.clone(),
            record::whole_image(&image),
            ImageLayout::Undefined,
        )
    }
}

fn downsample(src: &Arc<Image>, dst: &Arc<Image>) -> BlitImageInfo {
    let corner = |image: &Arc<Image>| [image.extent()[0], image.extent()[1], 1];
    BlitImageInfo {
        regions: [ImageBlit {
            src_subresource: color_layers(),
            src_offsets: [[0, 0, 0], corner(src)],
            dst_subresource: color_layers(),
            dst_offsets: [[0, 0, 0], corner(dst)],
            ..Default::default()
        }]
        .into(),
        filter: Filter::Linear,
        ..BlitImageInfo::images(src.clone(), dst.clone())
    }
}

fn color_layers() -> ImageSubresourceLayers {
    ImageSubresourceLayers {
        aspects: ImageAspects::COLOR,
        mip_level: 0,
        array_layers: 0..1,
    }
}

/// Into rendering, keeping what the frame drew. The semaphore the submission
/// waits on has already made the frame's writes visible.
fn present_to_color(image: Arc<Image>) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
        dst_access: AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
        old_layout: ImageLayout::PresentSrc,
        new_layout: ImageLayout::ColorAttachmentOptimal,
        subresource_range: record::whole_image(&image),
        ..ImageMemoryBarrier::image(image)
    }
}

fn color_to_present(image: Arc<Image>) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
        src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stages: PipelineStages::BOTTOM_OF_PIPE,
        dst_access: AccessFlags::empty(),
        old_layout: ImageLayout::ColorAttachmentOptimal,
        new_layout: ImageLayout::PresentSrc,
        subresource_range: record::whole_image(&image),
        ..ImageMemoryBarrier::image(image)
    }
}

fn begin(recorder: &mut Recorder, target: &Arc<ImageView>) {
    recorder.begin_rendering(RenderingInfo {
        color_attachments: vec![Some(RenderingAttachmentInfo {
            load_op: AttachmentLoadOp::Load,
            store_op: AttachmentStoreOp::Store,
            ..RenderingAttachmentInfo::image_view(target.clone())
        })],
        ..Default::default()
    });
}

fn viewport(extent: [u32; 2]) -> Viewport {
    Viewport {
        origin: [0.0, 0.0],
        extent: [extent[0] as f32, extent[1] as f32],
        depth_range: [0.0, 1.0],
    }
}

/// A clip rect in points as a scissor in pixels, rounded to the nearest pixel
/// and clamped to the target. `None` when nothing of it is on screen.
fn scissor(clip: egui::Rect, ppp: f32, extent: [u32; 2]) -> Option<Scissor> {
    let [width, height] = extent.map(|side| side as f32);
    let min = (clip.min.to_vec2() * ppp).round();
    let max = (clip.max.to_vec2() * ppp).round();
    let (x0, y0) = (min.x.clamp(0.0, width), min.y.clamp(0.0, height));
    let (x1, y1) = (max.x.clamp(x0, width), max.y.clamp(y0, height));
    (x1 > x0 && y1 > y0).then_some(Scissor {
        offset: [x0 as u32, y0 as u32],
        extent: [(x1 - x0) as u32, (y1 - y0) as u32],
    })
}

/// The scissor and outline for one surface's glass. Rounded outward so the
/// blur reaches the surface's antialiased edge; the outline itself stays exact.
fn frost_draw(glass: &Glass, ppp: f32, extent: [u32; 2]) -> Option<(Scissor, FrostPush)> {
    let rect = egui::Rect::from_min_max(
        (glass.rect.min.to_vec2() * ppp).floor().to_pos2(),
        (glass.rect.max.to_vec2() * ppp).ceil().to_pos2(),
    );
    let scissor = scissor(rect, 1.0, extent)?;
    Some((
        scissor,
        FrostPush {
            rect_min: (glass.rect.min.to_vec2() * ppp).into(),
            rect_max: (glass.rect.max.to_vec2() * ppp).into(),
            texel: [0.0; 2],
            radius: glass.radius * ppp,
        },
    ))
}

/// Half and quarter of `extent`, never below one pixel.
fn backdrop_extents(extent: [u32; 2]) -> ([u32; 2], [u32; 2]) {
    let half = extent.map(|side| (side / 2).max(1));
    (half, half.map(|side| (side / 2).max(1)))
}

fn backdrop_texel(extent: [u32; 2]) -> [f32; 2] {
    extent.map(|side| 1.0 / side as f32)
}

/// One texture delta's pixels, packed for `format`.
fn pack(image: &egui::ImageData, format: Format, into: &mut [u8]) {
    match (image, format) {
        (egui::ImageData::Color(image), _) => {
            let bytes = image.pixels.iter().flat_map(|color| color.to_array());
            for (into, from) in into.iter_mut().zip(bytes) {
                *into = from;
            }
        }
        // Linear coverage in red and the same coverage as sRGB alpha in green:
        // egui wants colour linear in the shader and alpha not, and the view's
        // swizzle spreads the two over RRRG.
        (egui::ImageData::Font(image), Format::Rg8Unorm) => {
            let linear = image
                .pixels
                .iter()
                .map(|coverage| (coverage.clamp(0.0, 1.0 - f32::EPSILON) * 256.0) as u8);
            let bytes = linear
                .zip(image.srgba_pixels(None))
                .flat_map(|(linear, srgb)| [linear, srgb.a()]);
            for (into, from) in into.iter_mut().zip(bytes) {
                *into = from;
            }
        }
        (egui::ImageData::Font(image), _) => {
            let bytes = image.srgba_pixels(None).flat_map(|color| color.to_array());
            for (into, from) in into.iter_mut().zip(bytes) {
                *into = from;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::vulkan::driver::VkDriver;
    use vulkano::buffer::BufferUsage;
    use vulkano::image::view::ImageView;
    use vulkano::image::{ImageCreateInfo, ImageUsage as VkImageUsage};
    use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

    fn device(ctx: &Arc<VkContext>) -> Device {
        Device::Vulkan(VkDriver::new(ctx.clone(), 2))
    }

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> egui::Rect {
        egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1))
    }

    #[test]
    fn the_overlay_stream_matches_its_derived_layout() {
        let derived = OverlayVertex::per_vertex();
        assert_eq!(OverlayVertex::STREAM.stride, derived.stride);
        for attribute in OverlayVertex::STREAM.attributes {
            let member = &derived.members[attribute.name];
            assert_eq!(member.offset, attribute.offset, "{}", attribute.name);
        }
        assert_eq!(
            OverlayVertex::STREAM.attributes.len(),
            derived.members.len()
        );
    }

    #[test]
    fn a_clip_rect_scales_rounds_and_clamps_to_the_target() {
        let scissor = scissor(rect(10.2, 4.0, 2000.0, 30.6), 1.5, [800, 600]).expect("on screen");
        assert_eq!(scissor.offset, [15, 6]);
        assert_eq!(scissor.extent, [785, 40]);
    }

    /// egui clips to rects that run off the window or have no area; Vulkan
    /// rejects a scissor with a negative offset, and a draw into nothing is
    /// work.
    #[test]
    fn a_clip_rect_off_screen_or_empty_is_no_scissor() {
        assert_eq!(
            scissor(rect(-50.0, -50.0, -10.0, -10.0), 1.0, [800, 600]),
            None
        );
        assert_eq!(
            scissor(rect(900.0, 10.0, 950.0, 20.0), 1.0, [800, 600]),
            None
        );
        assert_eq!(scissor(rect(10.0, 10.0, 10.0, 20.0), 1.0, [800, 600]), None);
        assert_eq!(scissor(egui::Rect::NOTHING, 1.0, [800, 600]), None);
    }

    /// The scissor covers every pixel the surface touches, including the
    /// partly covered ones on its edge, while the outline the shader cuts to
    /// stays where the surface really is.
    #[test]
    fn glass_rounds_its_scissor_outward_and_keeps_its_outline_exact() {
        let glass = Glass {
            rect: rect(10.3, 20.7, 50.2, 60.5),
            radius: 6.0,
        };
        let (scissor, push) = frost_draw(&glass, 2.0, [800, 600]).expect("on screen");
        assert_eq!(scissor.offset, [20, 41]);
        assert_eq!(scissor.extent, [81, 80]);
        assert_eq!(push.rect_min, [20.6, 41.4]);
        assert_eq!(push.rect_max, [100.4, 121.0]);
        assert_eq!(push.radius, 12.0);
    }

    #[test]
    fn the_backdrop_halves_twice_and_never_vanishes() {
        assert_eq!(backdrop_extents([1920, 1080]), ([960, 540], [480, 270]));
        assert_eq!(backdrop_extents([1001, 3]), ([500, 1], [250, 1]));
        assert_eq!(backdrop_extents([1, 1]), ([1, 1], [1, 1]));
    }

    /// The whole pass on a real device: a hard white-to-black edge, with glass
    /// over the middle of it. Inside the glass the edge is blurred; just
    /// outside it, the same edge is untouched.
    /// Several textures in one frame share the staging buffer, and each copy
    /// must start on a whole texel of its own format. A font patch three
    /// texels wide is six bytes in the swizzled format, which leaves the next
    /// texture's pixels two bytes off a four-byte texel unless the staging
    /// allocation is aligned.
    #[test]
    #[ignore = "needs a GPU"]
    fn textures_after_an_odd_sized_one_upload() {
        let instance = crate::capture::headless_instance();
        let ctx = Arc::new(VkContext::new(&instance, None));
        let device = device(&ctx);
        let format = VkFormat::R8G8B8A8_SRGB;
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format,
                extent: [4, 4, 1],
                usage: VkImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let view = ImageView::new_default(image.clone()).unwrap();

        let delta = |image: egui::ImageData| ImageDelta::full(image, egui::TextureOptions::LINEAR);
        let mut frame = OverlayFrame {
            gizmo: Vec::new(),
            outline: Vec::new(),
            pixels_per_point: 1.0,
            textures: egui::TexturesDelta {
                set: vec![
                    (
                        egui::TextureId::Managed(0),
                        delta(egui::ImageData::Font(egui::FontImage::new([3, 1]))),
                    ),
                    (
                        egui::TextureId::Managed(1),
                        delta(egui::ImageData::Color(Arc::new(egui::ColorImage::new(
                            [1, 1],
                            egui::Color32::WHITE,
                        )))),
                    ),
                ],
                free: Vec::new(),
            },
            phases: Vec::new(),
        };

        let mut pass = OverlayPass::new(&ctx, &device, format, false);
        let mut recorder = Recorder::new(&ctx);
        recorder.image_barrier(ImageMemoryBarrier {
            old_layout: ImageLayout::Undefined,
            ..color_to_present(image)
        });
        pass.record(&device, &mut recorder, &view, &mut frame);
        recorder.submit_and_wait(&ctx);
        assert_eq!(pass.textures.len(), 2);
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn glass_blurs_what_is_beneath_it_and_nothing_else() {
        use vulkano::buffer::{Buffer, BufferCreateInfo};
        use vulkano::command_buffer::CopyImageToBufferInfo;

        const EXTENT: [u32; 2] = [256, 128];
        let instance = crate::capture::headless_instance();
        let ctx = Arc::new(VkContext::new(&instance, None));
        let device = device(&ctx);
        let format = VkFormat::R8G8B8A8_SRGB;
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format,
                extent: [EXTENT[0], EXTENT[1], 1],
                usage: VkImageUsage::COLOR_ATTACHMENT | VkImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let view = ImageView::new_default(image.clone()).unwrap();

        let egui = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(rect(0.0, 0.0, EXTENT[0] as f32, EXTENT[1] as f32)),
            ..Default::default()
        };
        let output = egui.run(input, |ctx| {
            let painter = ctx.layer_painter(egui::LayerId::background());
            painter.rect_filled(rect(0.0, 0.0, 128.0, 128.0), 0.0, egui::Color32::WHITE);
            painter.rect_filled(rect(128.0, 0.0, 256.0, 128.0), 0.0, egui::Color32::BLACK);
        });
        let mut frame = OverlayFrame {
            gizmo: Vec::new(),
            outline: Vec::new(),
            pixels_per_point: 1.0,
            phases: vec![
                OverlayPhase {
                    glass: Vec::new(),
                    primitives: egui.tessellate(output.shapes, 1.0),
                },
                OverlayPhase {
                    glass: vec![Glass {
                        rect: rect(64.0, 32.0, 192.0, 96.0),
                        radius: 8.0,
                    }],
                    primitives: Vec::new(),
                },
            ],
            textures: output.textures_delta,
        };

        let mut pass = OverlayPass::new(&ctx, &device, format, true);
        assert!(pass.frost.is_some(), "this device cannot blit {format:?}");
        let readback = Buffer::new_slice::<u8>(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            u64::from(EXTENT[0] * EXTENT[1] * 4),
        )
        .unwrap();
        let mut recorder = Recorder::new(&ctx);
        recorder.image_barrier(ImageMemoryBarrier {
            old_layout: ImageLayout::Undefined,
            ..color_to_present(image.clone())
        });
        pass.record(&device, &mut recorder, &view, &mut frame);
        recorder
            .image_barrier(ImageMemoryBarrier {
                src_stages: PipelineStages::ALL_COMMANDS,
                src_access: AccessFlags::MEMORY_WRITE,
                dst_stages: PipelineStages::ALL_TRANSFER,
                dst_access: AccessFlags::TRANSFER_READ,
                old_layout: ImageLayout::PresentSrc,
                new_layout: ImageLayout::TransferSrcOptimal,
                subresource_range: record::whole_image(&image),
                ..ImageMemoryBarrier::image(image.clone())
            })
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, readback.clone()));
        recorder.submit_and_wait(&ctx);

        let pixels = readback.read().unwrap();
        let red = |x: u32, y: u32| pixels[((y * EXTENT[0] + x) * 4) as usize];
        assert_eq!(
            (red(127, 16), red(128, 16)),
            (255, 0),
            "the edge above the glass"
        );
        assert_eq!(
            (red(127, 112), red(128, 112)),
            (255, 0),
            "the edge below it"
        );
        let (left, right) = (red(127, 64), red(128, 64));
        assert!(
            (64..=250).contains(&left) && (8..=200).contains(&right) && left > right,
            "the edge under the glass is {left} | {right}, not blurred"
        );
        assert_eq!(
            red(70, 64),
            255,
            "far from the edge, the glass is the colour beneath"
        );
    }

    /// What the glass costs at 1080p: one editor-sized frame drawn with no
    /// glass, with the viewport's controls' (one capture), and with a menu and
    /// a tooltip open over those (three). Timed per submitted-and-waited
    /// frame, so every row carries the same submission overhead and the
    /// differences between rows are the captures and blurs.
    ///
    /// ```text
    /// cargo test --release -p orrin-core --lib glass_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a GPU"]
    fn glass_cost() {
        const EXTENT: [u32; 2] = [1920, 1080];
        const FRAMES: usize = 300;
        let instance = crate::capture::headless_instance();
        let ctx = Arc::new(VkContext::new(&instance, None));
        let device = device(&ctx);
        let format = VkFormat::R8G8B8A8_SRGB;
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format,
                extent: [EXTENT[0], EXTENT[1], 1],
                usage: VkImageUsage::COLOR_ATTACHMENT | VkImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let view = ImageView::new_default(image.clone()).unwrap();

        // A panel's worth of rows down each side, so the base phase has the
        // mesh count of a real editor frame.
        let egui = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(rect(0.0, 0.0, EXTENT[0] as f32, EXTENT[1] as f32)),
            ..Default::default()
        };
        let output = egui.run(input, |ctx| {
            egui::SidePanel::left("left").show(ctx, |ui| {
                for row in 0..60 {
                    ui.label(format!("Cube {row}"));
                }
            });
            egui::SidePanel::right("right").show(ctx, |ui| {
                for row in 0..60 {
                    ui.label(format!("Property {row}"));
                }
            });
        });
        let base = egui.tessellate(output.shapes, 1.0);
        let mut textures = Some(output.textures_delta);

        let mut pass = OverlayPass::new(&ctx, &device, format, true);
        let mut first = true;
        let toolbar = Glass {
            rect: rect(400.0, 40.0, 700.0, 72.0),
            radius: 6.0,
        };
        let popup = Glass {
            rect: rect(60.0, 30.0, 280.0, 260.0),
            radius: 6.0,
        };
        println!(
            "
{:<28} {:>10}",
            "overlay at 1920x1080", "median ms"
        );
        for (name, glass) in [
            ("no glass", vec![]),
            ("viewport controls", vec![(1, toolbar)]),
            (
                "+ a menu and a tooltip",
                vec![(1, toolbar), (2, popup), (3, popup)],
            ),
        ] {
            let mut times = Vec::with_capacity(FRAMES);
            for frame_index in 0..FRAMES + 20 {
                let mut phases: Vec<_> = (0..4).map(|_| OverlayPhase::default()).collect();
                phases[0].primitives = base.clone();
                for (phase, glass) in &glass {
                    phases[*phase].glass.push(*glass);
                }
                let mut frame = OverlayFrame {
                    gizmo: Vec::new(),
                    outline: Vec::new(),
                    pixels_per_point: 1.0,
                    textures: textures.take().unwrap_or_default(),
                    phases,
                };
                let start = std::time::Instant::now();
                let mut recorder = Recorder::new(&ctx);
                if first {
                    recorder.image_barrier(ImageMemoryBarrier {
                        old_layout: ImageLayout::Undefined,
                        ..color_to_present(image.clone())
                    });
                    first = false;
                }
                pass.record(&device, &mut recorder, &view, &mut frame);
                recorder.submit_and_wait(&ctx);
                if frame_index >= 20 {
                    times.push(start.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(f64::total_cmp);
            println!("{name:<28} {:>10.3}", times[FRAMES / 2]);
        }
    }

    /// The swizzled font format carries linear coverage and sRGB alpha; a
    /// half-covered texel is the case where the two differ.
    #[test]
    fn a_font_texel_packs_linear_coverage_beside_srgb_alpha() {
        let image = egui::ImageData::Font(egui::FontImage {
            size: [2, 1],
            pixels: vec![1.0, 0.5],
        });
        let mut bytes = [0; 4];
        pack(&image, Format::Rg8Unorm, &mut bytes);
        assert_eq!(bytes[0], 255);
        assert_eq!(bytes[1], 255);
        assert_eq!(bytes[2], 128);
        assert!(bytes[3] > 128, "alpha {} is not gamma-encoded", bytes[3]);
    }
}
