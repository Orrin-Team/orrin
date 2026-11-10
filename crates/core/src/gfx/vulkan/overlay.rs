//! The editor's UI, drawn over the finished frame.
//!
//! Recorded on its own encoder and submitted after the frame, behind the
//! frame's semaphore. The graph sees the overlay as one `Raw` pass writing the
//! swapchain image; the encoder's `open_swapchain` and `close_swapchain` move
//! it in and out of the layout the draw needs, so this module names no API
//! object.
//!
//! The vertex layout, blend state, font swizzle and mesh shaders are ported
//! from egui_winit_vulkano 0.28 (Apache-2.0), which drew the editor before
//! this did.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use egui::epaint::{ImageDelta, Primitive};

use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::ImageUsage;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, Blend, BlockKind, BufferId, Device, Encoder, Feature,
    GraphicsPipelineDesc, PipelineId, SamplerDesc, SamplerId, TextureDesc, TextureDimension,
    TextureSwizzle, TextureViewId, VertexAttribute, VertexFormat, VertexStream, Viewport,
};
use crate::gfx::shaders::ShaderId;

/// One frame of the editor, as the renderer draws it.
#[derive(Default)]
pub struct OverlayFrame {
    pub pixels_per_point: f32,
    /// Every change not yet drawn, oldest first.
    pub textures: egui::TexturesDelta,
    /// In draw order.
    pub primitives: Vec<egui::ClippedPrimitive>,
    /// The transform gizmo's triangles, drawn by their own pass before the
    /// editor's UI rather than by this one — see `vulkan::gizmo`.
    pub gizmo: Vec<super::gizmo::GizmoVertex>,
    /// The meshes the selection outline is drawn round — see `vulkan::outline`.
    pub outline: Vec<super::outline::OutlineItem>,
    /// The viewport's draw mode, when it is not plain shaded.
    pub wireframe: Option<super::wireframe::Wireframe>,
    /// What each view showing a material preview asks for this frame; the
    /// primitives paint each with its [`Viewer::texture`](super::material_preview::Viewer::texture).
    pub material_previews: super::material_preview::Subjects,
}

/// Every egui texture the renderer lends the editor rather than one egui
/// uploads, so that no two are ever given the same id.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UserTexture {
    MaterialPreview(super::material_preview::Viewer),
}

impl UserTexture {
    pub fn id(self) -> egui::TextureId {
        use super::material_preview::Viewer;
        egui::TextureId::User(match self {
            Self::MaterialPreview(Viewer::Inspector) => 0,
            Self::MaterialPreview(Viewer::Browser) => 1,
        })
    }
}

/// egui's vertex, field for field.
#[derive(Pod, Zeroable, Clone, Copy)]
#[cfg_attr(test, derive(vulkano::pipeline::graphics::vertex_input::Vertex))]
#[repr(C)]
struct OverlayVertex {
    #[cfg_attr(test, format(R32G32_SFLOAT))]
    position: [f32; 2],
    #[cfg_attr(test, format(R32G32_SFLOAT))]
    tex_coords: [f32; 2],
    #[cfg_attr(test, format(R8G8B8A8_UNORM))]
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

/// The frame's meshes, uploaded once: a vertex and an index buffer.
type Geometry = (BufferId, BufferId);

struct Texture {
    view: TextureViewId,
    set: BindGroupId,
}

pub(super) struct OverlayPass {
    format: Format,
    pipeline: PipelineId,
    sampler: SamplerId,
    /// `R8G8_UNORM` swizzled up to premultiplied white where views can swizzle,
    /// at half the size of the `R8G8B8A8_SRGB` it falls back to.
    font_format: Format,
    textures: HashMap<egui::TextureId, Texture>,
    /// A set for each image the renderer lends egui by a user id, kept for as
    /// long as the image is.
    lent: HashMap<TextureViewId, BindGroupId>,
    /// Reused each frame so a frame that changes little allocates little.
    vertices: Vec<OverlayVertex>,
    indices: Vec<u32>,
    pixels: Vec<u8>,
}

impl OverlayPass {
    pub(super) fn new(device: &Device, format: Format) -> Self {
        let font_format = if device.supports(Feature::CoverageTexture) {
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
            lent: HashMap::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            pixels: Vec::new(),
        }
    }

    /// Record `frame` over `target`, which arrives and is left in `PresentSrc`
    /// — where the graph leaves the swapchain image, and where presenting
    /// needs it. Takes the frame's texture changes, which are applied once.
    /// `lent` are images the renderer drew, sampled where egui names their id.
    pub(super) fn record(
        &mut self,
        device: &Device,
        encoder: &mut Encoder,
        target: TextureViewId,
        frame: &mut OverlayFrame,
        lent: &[(egui::TextureId, TextureViewId)],
    ) {
        let extent = device.view_extent(target);
        let ppp = frame.pixels_per_point;
        let textures = std::mem::take(&mut frame.textures);

        self.upload_textures(device, encoder, &textures.set);
        let geometry = self.upload_geometry(device, frame);
        let lent: Vec<(egui::TextureId, BindGroupId)> = lent
            .iter()
            .map(|&(id, view)| (id, self.lend(device, view)))
            .collect();

        encoder.open_swapchain(target);
        if let Some(geometry) = geometry {
            encoder.begin_rendering(target);
            self.draw_meshes(encoder, &frame.primitives, geometry, ppp, extent, &lent);
            encoder.end_rendering();
        }
        encoder.close_swapchain(target);

        // The sets a recorded draw bound are held by the recording, so a
        // texture freed here outlives every frame still reading it.
        for id in &textures.free {
            if let Some(texture) = self.textures.remove(id) {
                device.destroy_bind_group(texture.set);
                device.destroy_texture(texture.view);
            }
        }
    }

    fn draw_meshes(
        &self,
        encoder: &mut Encoder,
        primitives: &[egui::ClippedPrimitive],
        (vertices, indices): Geometry,
        ppp: f32,
        extent: [u32; 2],
        lent: &[(egui::TextureId, BindGroupId)],
    ) {
        let (mut vertex, mut index) = (0, 0);
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
                let set = self
                    .textures
                    .get(&mesh.texture_id)
                    .map(|t| t.set)
                    .or_else(|| {
                        lent.iter()
                            .find(|(id, _)| *id == mesh.texture_id)
                            .map(|(_, set)| *set)
                    });
                // A lent image not drawn yet paints nothing, which is not a
                // mistake: the view behind it shows through until it is.
                let Some(set) = set else {
                    if matches!(mesh.texture_id, egui::TextureId::Managed(_)) {
                        tracing::warn!(
                            "egui drew with {:?}, which was never uploaded",
                            mesh.texture_id
                        );
                    }
                    continue;
                };
                encoder.bind_group(0, set);
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
    }

    /// The frame's meshes in one vertex and one index buffer, in draw order.
    fn upload_geometry(&mut self, device: &Device, frame: &OverlayFrame) -> Option<Geometry> {
        let meshes = || {
            frame
                .primitives
                .iter()
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

    fn lend(&mut self, device: &Device, view: TextureViewId) -> BindGroupId {
        let (pipeline, sampler) = (self.pipeline, self.sampler);
        *self.lent.entry(view).or_insert_with(|| {
            device.create_persistent_bind_group(
                pipeline,
                0,
                &[Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture { view, sampler },
                }],
            )
        })
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
}

fn viewport(extent: [u32; 2]) -> Viewport {
    Viewport {
        origin: [0.0, 0.0],
        extent: [extent[0] as f32, extent[1] as f32],
        depth_range: [0.0, 1.0],
    }
}

/// A scissor rectangle in framebuffer pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Clip {
    offset: [u32; 2],
    extent: [u32; 2],
}

/// A clip rect in points as a scissor in pixels, rounded to the nearest pixel
/// and clamped to the target. `None` when nothing of it is on screen.
fn scissor(clip: egui::Rect, ppp: f32, extent: [u32; 2]) -> Option<Clip> {
    let [width, height] = extent.map(|side| side as f32);
    let min = (clip.min.to_vec2() * ppp).round();
    let max = (clip.max.to_vec2() * ppp).round();
    let (x0, y0) = (min.x.clamp(0.0, width), min.y.clamp(0.0, height));
    let (x1, y1) = (max.x.clamp(x0, width), max.y.clamp(y0, height));
    (x1 > x0 && y1 > y0).then_some(Clip {
        offset: [x0 as u32, y0 as u32],
        extent: [(x1 - x0) as u32, (y1 - y0) as u32],
    })
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
    /// Two lent textures given one id would paint one in the other's place.
    #[test]
    fn every_lent_texture_has_its_own_id() {
        use super::super::material_preview::Viewer;
        let ids: Vec<_> = Viewer::ALL
            .map(|viewer| UserTexture::MaterialPreview(viewer).id())
            .into();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "{ids:?}");
        assert!(ids.iter().all(|id| matches!(id, egui::TextureId::User(_))));
    }

    use super::*;
    use crate::gfx::vulkan::context::VkContext;
    use crate::gfx::vulkan::driver::color_to_present;
    use crate::gfx::vulkan::driver::{VkDriver, VkEncoder};
    use crate::gfx::vulkan::record::{self, Recorder};
    use std::sync::Arc;
    use vulkano::buffer::BufferUsage;
    use vulkano::format::Format as VkFormat;
    use vulkano::image::view::ImageView;
    use vulkano::image::{Image, ImageLayout};
    use vulkano::image::{ImageCreateInfo, ImageUsage as VkImageUsage};
    use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
    use vulkano::pipeline::graphics::vertex_input::Vertex as _;
    use vulkano::sync::{AccessFlags, ImageMemoryBarrier, PipelineStages};

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
            wireframe: None,
            material_previews: Default::default(),
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
            primitives: Vec::new(),
        };

        let mut pass =
            OverlayPass::new(&device, crate::gfx::vulkan::convert::engine_format(format));
        let mut recorder = Recorder::new(&ctx);
        recorder.image_barrier(ImageMemoryBarrier {
            old_layout: ImageLayout::Undefined,
            ..color_to_present(image)
        });
        {
            let driver = device.vulkan();
            let target = driver.import_view(view.clone());
            let mut encoder = Encoder::Vulkan(VkEncoder::new(&mut recorder, driver));
            pass.record(&device, &mut encoder, target, &mut frame, &[]);
        }
        recorder.submit_and_wait(&ctx);
        assert_eq!(pass.textures.len(), 2);
    }

    /// What the editor's UI cadence rests on (see `editor::cadence`): a frame
    /// whose UI was not rebuilt hands the renderer the *same* `OverlayFrame`
    /// again, after the first draw has already taken its texture deltas. That
    /// second draw has to produce the same image — the glyphs drawn from the
    /// font texture the first one uploaded, not a blank or a stale atlas.
    #[test]
    #[ignore = "needs a GPU"]
    fn a_cached_frame_draws_the_same_image_again() {
        use vulkano::buffer::{Buffer, BufferCreateInfo};
        use vulkano::command_buffer::CopyImageToBufferInfo;

        const EXTENT: [u32; 2] = [256, 64];
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
        // The background covers every pixel, so each draw overwrites the last
        // and the two readbacks are comparable without a clear between them.
        let output = egui.run(input, |ctx| {
            let painter = ctx.layer_painter(egui::LayerId::background());
            painter.rect_filled(
                rect(0.0, 0.0, EXTENT[0] as f32, EXTENT[1] as f32),
                0.0,
                egui::Color32::DARK_BLUE,
            );
            painter.text(
                egui::pos2(8.0, 8.0),
                egui::Align2::LEFT_TOP,
                "Hierarchy",
                egui::FontId::proportional(24.0),
                egui::Color32::WHITE,
            );
        });
        let mut frame = OverlayFrame {
            gizmo: Vec::new(),
            outline: Vec::new(),
            wireframe: None,
            material_previews: Default::default(),
            pixels_per_point: 1.0,
            primitives: egui.tessellate(output.shapes, 1.0),
            textures: output.textures_delta,
        };
        assert!(
            !frame.textures.set.is_empty(),
            "the first frame carries the font atlas"
        );

        let mut pass =
            OverlayPass::new(&device, crate::gfx::vulkan::convert::engine_format(format));
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
        let mut draw = |frame: &mut OverlayFrame| -> Vec<u8> {
            let mut recorder = Recorder::new(&ctx);
            recorder.image_barrier(ImageMemoryBarrier {
                old_layout: ImageLayout::Undefined,
                ..color_to_present(image.clone())
            });
            {
                let driver = device.vulkan();
                let target = driver.import_view(view.clone());
                let mut encoder = Encoder::Vulkan(VkEncoder::new(&mut recorder, driver));
                pass.record(&device, &mut encoder, target, frame, &[]);
            }
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
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                    image.clone(),
                    readback.clone(),
                ));
            recorder.submit_and_wait(&ctx);
            readback.read().unwrap().to_vec()
        };

        let first = draw(&mut frame);
        assert!(
            frame.textures.set.is_empty(),
            "the first draw takes the deltas"
        );
        let second = draw(&mut frame);
        let lit = first.chunks(4).filter(|texel| texel[0] > 200).count();
        assert!(
            lit > 50,
            "the label drew only {lit} light texels; the test is not looking at text"
        );
        assert!(
            first == second,
            "a cached frame drew a different image the second time"
        );
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
