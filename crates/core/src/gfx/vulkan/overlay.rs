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

use egui::epaint::{ImageDelta, Primitive};
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    BlitImageInfo, BufferImageCopy, CopyBufferToImageInfo, ImageBlit, RenderingAttachmentInfo,
    RenderingInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::{Format, FormatFeatures, NumericFormat};
use vulkano::image::sampler::{
    ComponentMapping, ComponentSwizzle, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{
    Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageUsage,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::sync::{AccessFlags, ImageMemoryBarrier, PipelineStages};

use super::context::VkContext;
use super::record::{self, Arena, Recorder};
use super::rendering;

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
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct OverlayVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    tex_coords: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    color: [u8; 4],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct MeshPush {
    screen_size: [f32; 2],
    output_in_linear_colorspace: i32,
}

#[derive(BufferContents, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct FrostPush {
    rect_min: [f32; 2],
    rect_max: [f32; 2],
    texel: [f32; 2],
    radius: f32,
}

/// Every phase's meshes, uploaded once: a vertex and an index buffer.
type Geometry = (Subbuffer<[OverlayVertex]>, Subbuffer<[u32]>);

struct Texture {
    image: Arc<Image>,
    set: Arc<DescriptorSet>,
}

/// The two downsampled copies of the target a capture writes, sized to it.
struct Backdrop {
    extent: [u32; 2],
    half: Arc<Image>,
    quarter: Arc<Image>,
    set: Arc<DescriptorSet>,
}

pub(super) struct OverlayPass {
    format: Format,
    pipeline: Arc<GraphicsPipeline>,
    /// `None` where the target cannot be blitted from, which leaves the glass
    /// as plain translucency.
    frost: Option<Arc<GraphicsPipeline>>,
    sampler: Arc<Sampler>,
    /// `R8G8_UNORM` swizzled up to premultiplied white where views can swizzle,
    /// at half the size of the `R8G8B8A8_SRGB` it falls back to.
    font_format: Format,
    textures: HashMap<egui::TextureId, Texture>,
    geometry: Arena,
    staging: Arena,
    backdrop: Option<Backdrop>,
}

impl OverlayPass {
    /// `capturable` is whether the target may be a transfer source — a
    /// swapchain has to have been created with that usage, and the surface may
    /// not offer it.
    pub(super) fn new(ctx: &VkContext, format: Format, capturable: bool) -> Self {
        let physical = ctx.device.physical_device();
        let blittable = |format: Format, features: FormatFeatures| {
            physical
                .format_properties(format)
                .is_ok_and(|properties| properties.optimal_tiling_features.contains(features))
        };
        let frost = (capturable
            && blittable(
                format,
                FormatFeatures::BLIT_SRC
                    | FormatFeatures::BLIT_DST
                    | FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR,
            ))
        .then(|| build_frost_pipeline(ctx, format));

        let swizzle = !ctx.device.enabled_extensions().khr_portability_subset
            || ctx.device.enabled_features().image_view_format_swizzle;
        let font_format = if swizzle
            && blittable(
                Format::R8G8_UNORM,
                FormatFeatures::SAMPLED_IMAGE | FormatFeatures::TRANSFER_DST,
            ) {
            Format::R8G8_UNORM
        } else {
            Format::R8G8B8A8_SRGB
        };

        let arena = |buffer_usage| {
            Arena::new(
                ctx.memory_allocator.clone(),
                SubbufferAllocatorCreateInfo {
                    buffer_usage,
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )
        };

        Self {
            format,
            pipeline: build_mesh_pipeline(ctx, format),
            frost,
            sampler: Sampler::new(
                ctx.device.clone(),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::ClampToEdge; 3],
                    ..Default::default()
                },
            )
            .expect("failed to create the overlay sampler"),
            font_format,
            textures: HashMap::new(),
            geometry: arena(BufferUsage::VERTEX_BUFFER | BufferUsage::INDEX_BUFFER),
            staging: arena(BufferUsage::TRANSFER_SRC),
            backdrop: None,
        }
    }

    /// Record `frame` over `target`, which arrives and is left in `PresentSrc`
    /// — where the graph leaves the swapchain image, and where presenting
    /// needs it. Takes the frame's texture changes, which are applied once.
    pub(super) fn record(
        &mut self,
        ctx: &VkContext,
        recorder: &mut Recorder,
        target: &Arc<ImageView>,
        frame: &mut OverlayFrame,
    ) {
        let image = target.image().clone();
        let extent = [image.extent()[0], image.extent()[1]];
        let ppp = frame.pixels_per_point;
        let textures = std::mem::take(&mut frame.textures);

        self.upload_textures(ctx, recorder, &textures.set);
        let geometry = self.upload_geometry(frame);

        recorder.image_barrier(present_to_color(image.clone()));
        let mut rendering = false;
        let mut cursor = (0, 0);
        for (index, phase) in frame.phases.iter().enumerate() {
            let glass: Vec<_> = phase
                .glass
                .iter()
                .filter(|_| index > 0)
                .filter_map(|glass| frost_draw(glass, ppp, extent))
                .collect();
            if let Some(frost) = self.frost.clone()
                && !glass.is_empty()
            {
                if rendering {
                    recorder.end_rendering();
                }
                let backdrop = self.backdrop(ctx, extent);
                capture(recorder, &image, backdrop);
                let texel = backdrop_texel(backdrop.extent);
                let set = backdrop.set.clone();
                begin(recorder, target);
                rendering = true;

                recorder
                    .bind_pipeline_graphics(&frost)
                    .set_viewport(0, &[viewport(extent)])
                    .bind_descriptor_sets(PipelineBindPoint::Graphics, frost.layout(), 0, &[set]);
                for (scissor, push) in glass {
                    recorder
                        .set_scissor(0, &[scissor])
                        .push_constants(frost.layout(), 0, &FrostPush { texel, ..push })
                        .draw(3, 1, 0, 0);
                }
            }

            if phase.primitives.is_empty() {
                continue;
            }
            let Some(geometry) = &geometry else {
                continue;
            };
            if !rendering {
                begin(recorder, target);
                rendering = true;
            }
            cursor = self.draw_meshes(recorder, &phase.primitives, geometry, cursor, ppp, extent);
        }
        if rendering {
            recorder.end_rendering();
        }
        recorder.image_barrier(color_to_present(image));

        // The sets a recorded draw bound are held by the recording, so a
        // texture freed here outlives every frame still reading it.
        for id in &textures.free {
            self.textures.remove(id);
        }
    }

    /// Returns the vertex and index cursors past this phase's meshes.
    fn draw_meshes(
        &self,
        recorder: &mut Recorder,
        primitives: &[egui::ClippedPrimitive],
        (vertices, indices): &Geometry,
        (mut vertex, mut index): (u32, u32),
        ppp: f32,
        extent: [u32; 2],
    ) -> (u32, u32) {
        let layout = self.pipeline.layout();
        recorder
            .bind_pipeline_graphics(&self.pipeline)
            .bind_vertex_buffers(0, vertices.clone())
            .bind_index_buffer(indices.clone())
            .set_viewport(0, &[viewport(extent)])
            .push_constants(
                layout,
                0,
                &MeshPush {
                    screen_size: [extent[0] as f32 / ppp, extent[1] as f32 / ppp],
                    output_in_linear_colorspace: (self.format.numeric_format_color()
                        == Some(NumericFormat::SRGB))
                        as i32,
                },
            );

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
                recorder.bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    layout,
                    0,
                    std::slice::from_ref(&texture.set),
                );
                bound_texture = Some(mesh.texture_id);
            }
            if bound_clip != Some(scissor) {
                recorder.set_scissor(0, &[scissor]);
                bound_clip = Some(scissor);
            }
            recorder.draw_indexed(
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
    fn upload_geometry(&self, frame: &OverlayFrame) -> Option<Geometry> {
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
        let vertex_count: usize = meshes().map(|mesh| mesh.vertices.len()).sum();
        let index_count: usize = meshes().map(|mesh| mesh.indices.len()).sum();
        if vertex_count == 0 || index_count == 0 {
            return None;
        }

        let vertices = self
            .geometry
            .allocate_slice::<OverlayVertex>(vertex_count as u64);
        let indices = self.geometry.allocate_slice::<u32>(index_count as u64);
        {
            let mut write = vertices.write().expect("a fresh suballocation is unshared");
            for (into, from) in write
                .iter_mut()
                .zip(meshes().flat_map(|mesh| &mesh.vertices))
            {
                *into = OverlayVertex {
                    position: [from.pos.x, from.pos.y],
                    tex_coords: [from.uv.x, from.uv.y],
                    color: from.color.to_array(),
                };
            }
        }
        {
            let mut write = indices.write().expect("a fresh suballocation is unshared");
            for (into, from) in write
                .iter_mut()
                .zip(meshes().flat_map(|mesh| &mesh.indices))
            {
                *into = *from;
            }
        }
        Some((vertices, indices))
    }

    /// Record the frame's texture changes ahead of its draws.
    ///
    /// In the overlay's own command buffer rather than a blocking submit: the
    /// barrier before each copy waits on earlier fragment reads, which covers a
    /// previous frame's overlay still sampling a texture this one rewrites.
    fn upload_textures(
        &mut self,
        ctx: &VkContext,
        recorder: &mut Recorder,
        deltas: &[(egui::TextureId, ImageDelta)],
    ) {
        for (id, delta) in deltas {
            let format = match &delta.image {
                egui::ImageData::Color(_) => Format::R8G8B8A8_SRGB,
                egui::ImageData::Font(_) => self.font_format,
            };
            let [width, height] = delta.image.size().map(|side| side as u32);
            let bytes = u64::from(width) * u64::from(height) * format.block_size();
            // Whole words, so each copy starts on a texel of its format: both
            // formats here have texels of two or four bytes, and deltas share
            // one staging arena.
            let stage = self
                .staging
                .allocate_slice::<u32>(bytes.div_ceil(4))
                .reinterpret::<[u8]>()
                .slice(..bytes);
            pack(
                &delta.image,
                format,
                &mut stage.write().expect("a fresh suballocation is unshared"),
            );

            let (image, old_layout, offset) = match delta.pos {
                Some([x, y]) => {
                    let Some(texture) = self.textures.get(id) else {
                        tracing::warn!("egui patched {id:?}, which was never uploaded");
                        continue;
                    };
                    (
                        texture.image.clone(),
                        ImageLayout::ShaderReadOnlyOptimal,
                        [x as u32, y as u32, 0],
                    )
                }
                None => {
                    let texture = self.create_texture(ctx, format, [width, height]);
                    let image = texture.image.clone();
                    self.textures.insert(*id, texture);
                    (image, ImageLayout::Undefined, [0, 0, 0])
                }
            };

            let range = record::whole_image(&image);
            recorder
                .image_barrier(ImageMemoryBarrier {
                    src_stages: PipelineStages::FRAGMENT_SHADER,
                    ..record::to_transfer_dst(image.clone(), range.clone(), old_layout)
                })
                .copy_buffer_to_image(CopyBufferToImageInfo {
                    regions: [BufferImageCopy {
                        image_offset: offset,
                        image_extent: [width, height, 1],
                        image_subresource: color_layers(),
                        ..Default::default()
                    }]
                    .into(),
                    ..CopyBufferToImageInfo::buffer_image(stage, image.clone())
                })
                .image_barrier(record::to_shader_read(
                    image,
                    range,
                    ImageLayout::TransferDstOptimal,
                ));
        }
    }

    fn create_texture(&self, ctx: &VkContext, format: Format, extent: [u32; 2]) -> Texture {
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("failed to allocate an egui texture");
        let component_mapping = match format {
            Format::R8G8_UNORM => ComponentMapping {
                r: ComponentSwizzle::Red,
                g: ComponentSwizzle::Red,
                b: ComponentSwizzle::Red,
                a: ComponentSwizzle::Green,
            },
            _ => ComponentMapping::identity(),
        };
        let view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                component_mapping,
                ..ImageViewCreateInfo::from_image(&image)
            },
        )
        .expect("failed to create an egui texture view");
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                view,
                self.sampler.clone(),
            )],
            [],
        )
        .expect("failed to create an egui texture set");
        Texture { image, set }
    }

    /// The capture targets for `extent`, rebuilt when the window resizes.
    fn backdrop(&mut self, ctx: &VkContext, extent: [u32; 2]) -> &Backdrop {
        if self
            .backdrop
            .as_ref()
            .is_none_or(|backdrop| backdrop.extent != extent)
        {
            let (half_extent, quarter_extent) = backdrop_extents(extent);
            let image = |extent: [u32; 2], usage| {
                Image::new(
                    ctx.memory_allocator.clone(),
                    ImageCreateInfo {
                        format: self.format,
                        extent: [extent[0], extent[1], 1],
                        usage,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .expect("failed to allocate the overlay backdrop")
            };
            let half = image(
                half_extent,
                ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC,
            );
            let quarter = image(
                quarter_extent,
                ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            );
            let frost = self.frost.as_ref().expect("only frosted frames capture");
            let set = DescriptorSet::new(
                ctx.descriptor_set_allocator.clone(),
                frost.layout().set_layouts()[0].clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    ImageView::new_default(quarter.clone())
                        .expect("failed to view the overlay backdrop"),
                    self.sampler.clone(),
                )],
                [],
            )
            .expect("failed to create the backdrop set");
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
fn capture(recorder: &mut Recorder, target: &Arc<Image>, backdrop: &Backdrop) {
    let half = &backdrop.half;
    let quarter = &backdrop.quarter;
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
        offset: [0.0, 0.0],
        extent: [extent[0] as f32, extent[1] as f32],
        depth_range: 0.0..=1.0,
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
        (egui::ImageData::Font(image), Format::R8G8_UNORM) => {
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

fn build_mesh_pipeline(ctx: &VkContext, format: Format) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    let vs = overlay_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = overlay_fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let vertex_input_state = OverlayVertex::per_vertex().definition(&vs).unwrap();
    // egui's colours are premultiplied, and its target's alpha is coverage
    // accumulated front to back.
    let blend = AttachmentBlend {
        src_color_blend_factor: BlendFactor::One,
        src_alpha_blend_factor: BlendFactor::OneMinusDstAlpha,
        dst_alpha_blend_factor: BlendFactor::One,
        ..AttachmentBlend::alpha()
    };
    build_pipeline(ctx, format, [vs, fs], vertex_input_state, blend)
}

fn build_frost_pipeline(ctx: &VkContext, format: Format) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    let vs = fullscreen_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = frost_fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    build_pipeline(
        ctx,
        format,
        [vs, fs],
        VertexInputState::default(),
        AttachmentBlend::alpha(),
    )
}

fn build_pipeline(
    ctx: &VkContext,
    format: Format,
    [vs, fs]: [vulkano::shader::EntryPoint; 2],
    vertex_input_state: VertexInputState,
    blend: AttachmentBlend,
) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    GraphicsPipeline::new(
        device.clone(),
        ctx.pipeline_cache(),
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                1,
                ColorBlendAttachmentState {
                    blend: Some(blend),
                    ..Default::default()
                },
            )),
            dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                .into_iter()
                .collect(),
            subpass: Some(rendering::pipeline_info(&[format], None).into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

mod overlay_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/overlay.vert" }
}

mod overlay_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/overlay.frag" }
}

mod fullscreen_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/fullscreen.vert" }
}

mod frost_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/frost.frag" }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> egui::Rect {
        egui::Rect::from_min_max(egui::pos2(x0, y0), egui::pos2(x1, y1))
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
        let ctx = VkContext::new(&instance, None);
        let format = Format::R8G8B8A8_SRGB;
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format,
                extent: [4, 4, 1],
                usage: ImageUsage::COLOR_ATTACHMENT,
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

        let mut pass = OverlayPass::new(&ctx, format, false);
        let mut recorder = Recorder::new(&ctx);
        recorder.image_barrier(ImageMemoryBarrier {
            old_layout: ImageLayout::Undefined,
            ..color_to_present(image)
        });
        pass.record(&ctx, &mut recorder, &view, &mut frame);
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
        let ctx = VkContext::new(&instance, None);
        let format = Format::R8G8B8A8_SRGB;
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format,
                extent: [EXTENT[0], EXTENT[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
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

        let mut pass = OverlayPass::new(&ctx, format, true);
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
        pass.record(&ctx, &mut recorder, &view, &mut frame);
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
        let ctx = VkContext::new(&instance, None);
        let format = Format::R8G8B8A8_SRGB;
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format,
                extent: [EXTENT[0], EXTENT[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
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

        let mut pass = OverlayPass::new(&ctx, format, true);
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
                pass.record(&ctx, &mut recorder, &view, &mut frame);
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
        pack(&image, Format::R8G8_UNORM, &mut bytes);
        assert_eq!(bytes[0], 255);
        assert_eq!(bytes[1], 255);
        assert_eq!(bytes[2], 128);
        assert!(bytes[3] > 128, "alpha {} is not gamma-encoded", bytes[3]);
    }
}
