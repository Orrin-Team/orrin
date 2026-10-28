//! The one geometry pass in front of shading: view-space normals, screen-space
//! motion, the specular colour a surface reflects with, and the depth all three
//! are read against.
//!
//! It started as SSAO's private prepass and is shared now because TAA needs the
//! same rasterisation for a different attachment. Everything downstream of it —
//! SSAO, TAA, motion blur, depth of field, screen-space reflections — reads the
//! targets it leaves rather than rasterising the scene again.
//!
//! Reflections are what made it sample material maps rather than only transform
//! vertices. A reflection has to be tinted by the surface's own `f0` and
//! sharpened or spread by its roughness, and neither survives as a per-object
//! constant once a metal-rough map is involved; the normal it reflects about is
//! the mapped one for the same reason, or reflections slide over bumps the lit
//! image clearly has. SSAO reads the better normals too.

use std::sync::Arc;

use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::MemoryTypeFilter;

use crate::gfx::rhi::{
    CullMode, DepthTest, Device, Encoder, GraphicsPipelineDesc, PipelineId, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::{PositionVertex, SurfaceVertex};

use super::PassCtx;
use super::context::VkContext;
use super::cull::Draws;
use super::instances::{GpuObject, InstanceEntry};
use super::record::Arena;
use super::swapchain::DEPTH_FORMAT;
use super::taa::FrameView;

/// Where `prepass.frag` declares the material texture array. One set later than
/// the forward pass's, because this pipeline carries a frame block the other
/// takes from set 0 — the array itself is binding 0 of it either way, which is
/// what [`VkContext::mark_texture_array_partial`] assumes.
const TEXTURE_SET: usize = 3;

pub(super) const NORMAL_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::Rgba8Unorm;

/// `rgb` = the surface's normal-incidence specular colour, `a` = perceptual
/// roughness.
///
/// `f0` rather than base colour and metallic, because `f0` is the only thing
/// downstream actually wants and packing it here means the mix a metal implies
/// is done once, next to the material that decides it. Eight bits costs a
/// dielectric's ~4% about a thousandth in absolute terms, which lands well
/// inside the error the split-sum approximation already carries.
pub(super) const MATERIAL_FORMAT: crate::gfx::format::Format =
    crate::gfx::format::Format::Rgba8Unorm;

/// Signed and float, unlike the normal target: a motion vector is a UV *delta*,
/// so it is negative half the time, and a UNORM encoding would need a bias that
/// costs precision exactly where the vectors are smallest and matter most.
pub(super) const VELOCITY_FORMAT: crate::gfx::format::Format =
    crate::gfx::format::Format::Rg16Float;

/// The per-frame camera block. Shared with the SSAO passes, which reconstruct
/// view-space positions from the same depth this pass wrote and so must use the
/// same — jittered — projection to do it.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct FrameUbo {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    jitter: [f32; 4],
    /// `proj * view`, premultiplied on the CPU — the identical `Mat4` the
    /// forward pass pushes as a push constant.
    ///
    /// Last in the block rather than beside `proj`, and that position is
    /// load-bearing: `ssao.frag` and `contact_shadows.frag` declare only the
    /// first three matrices of this block, which is legal exactly as long as
    /// what they declare stays a prefix of what is uploaded. Appending keeps it
    /// one; inserting would silently hand both of them the wrong matrix.
    ///
    /// It exists so `prepass.vert` can compute `gl_Position` from the same
    /// expression `forward.vert` does. Multiplying `proj * (view * world)` and
    /// `(proj * view) * world` are equal in exact arithmetic and not in floating
    /// point, and the forward pass now depth-tests `EQUAL` against the depth
    /// this pass wrote — so a difference in the last bit is a surface that
    /// vanishes. See the `invariant gl_Position` in both shaders.
    view_proj: [[f32; 4]; 4],
}

pub struct GeometryPrepass {
    pipeline: PipelineId,
    /// The same pass for a `Masked` run: back faces kept, and a fragment shader
    /// that cuts the texels below the material's cutoff away.
    ///
    /// A second pipeline rather than a flag the one shader tests, because a
    /// `discard` anywhere in a shader costs *every* draw through it its early
    /// depth test — and this pass exists to be the depth everything downstream
    /// reads. Two pipelines at startup is the price of keeping that cost on the
    /// foliage alone.
    masked_pipeline: PipelineId,
    /// The material maps are sampled at the same texture coordinates and the
    /// same mip selection the forward pass uses, so the two rasterisations
    /// cannot disagree about what a surface is.
    sampler: Arc<Sampler>,
    uniform_allocator: Arena,
}

/// Set `set` of `pipeline`'s layout, for the builders below that still write
/// their own descriptor sets.
fn layout(
    device: &Device,
    pipeline: PipelineId,
    set: usize,
) -> Arc<vulkano::descriptor_set::layout::DescriptorSetLayout> {
    let Device::Vulkan(driver) = device;
    driver.set_layout(pipeline, set)
}

impl GeometryPrepass {
    pub fn new(ctx: &VkContext, device_rhi: &Device) -> Self {
        let device = &ctx.device;
        let pipeline = |fragment, cull| {
            device_rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex: ShaderId::PrepassVert,
                fragment,
                color_formats: &[NORMAL_FORMAT, VELOCITY_FORMAT, MATERIAL_FORMAT],
                depth_format: Some(DEPTH_FORMAT),
                vertex_streams: &[PositionVertex::STREAM, SurfaceVertex::STREAM],
                cull,
                depth_test: Some(DepthTest { write: true }),
                partial_texture_set: Some(TEXTURE_SET as u32),
            })
        };
        let pipeline_plain = pipeline(ShaderId::PrepassFrag, CullMode::Back);
        // Two-sided for a cutout, matching the forward pass. The two rasterise
        // the same triangles or the depth this pass leaves is not the depth the
        // frame was shaded against.
        let masked_pipeline = pipeline(ShaderId::PrepassFragMasked, CullMode::None);
        let anisotropy = device.enabled_features().sampler_anisotropy.then(|| {
            device
                .physical_device()
                .properties()
                .max_sampler_anisotropy
                .min(16.0)
        });
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                anisotropy,
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .unwrap();
        let uniform_allocator = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Self {
            pipeline: pipeline_plain,
            masked_pipeline,
            sampler,
            uniform_allocator,
        }
    }

    /// Upload the camera block once for every pass in the frame that reads it.
    pub(super) fn begin_frame(&self, view: &FrameView) -> Subbuffer<FrameUbo> {
        let frame = self.uniform_allocator.allocate_sized::<FrameUbo>();
        *frame.write().unwrap() = FrameUbo {
            view: view.view.to_cols_array_2d(),
            proj: view.proj.to_cols_array_2d(),
            inv_proj: view.proj.inverse().to_cols_array_2d(),
            prev_view_proj: view.prev_view_proj.to_cols_array_2d(),
            jitter: [view.jitter.x, view.jitter.y, 0.0, 0.0],
            view_proj: view.view_proj.to_cols_array_2d(),
        };
        frame
    }

    /// The sampler this pass reads material maps through.
    ///
    /// Lent to the shadow pass's cutout pipeline rather than duplicated there:
    /// an alpha test that filtered differently from the two passes it has to
    /// agree with would cut the leaf out at a slightly different place in the
    /// shadow map than in the frame, which reads as the shadow being offset.
    pub(super) fn material_sampler(&self) -> &Arc<Sampler> {
        &self.sampler
    }

    /// The set-1 per-object descriptor set this pass binds.
    pub(super) fn build_object_set(
        &self,
        ctx: &VkContext,
        device: &Device,
        rows: &Subbuffer<[GpuObject]>,
        indices: &Subbuffer<[InstanceEntry]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout(device, self.pipeline, 1),
            [
                WriteDescriptorSet::buffer(0, rows.clone()),
                WriteDescriptorSet::buffer(1, indices.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// The set-2 material table, over the same buffer the forward pass reads.
    ///
    /// Its own set rather than the forward pass's, for the reason the object
    /// sets are kept apart: set compatibility is a property of the layout each
    /// pipeline declares, not of the buffer written into it. The *data* is
    /// shared, which is what keeps the two passes agreeing about a material.
    pub(super) fn build_material_set(
        &self,
        ctx: &VkContext,
        device: &Device,
        materials: &Subbuffer<[super::forward::GpuMaterial]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout(device, self.pipeline, 2),
            [WriteDescriptorSet::buffer(0, materials.clone())],
            [],
        )
        .unwrap()
    }

    /// The set-3 texture array, filled exactly as the forward pass fills its
    /// own: every unused slot points at the white default so no descriptor is
    /// left unwritten.
    pub(super) fn build_texture_set(
        &self,
        ctx: &VkContext,
        device: &Device,
        textures: &[Arc<ImageView>],
    ) -> Arc<DescriptorSet> {
        let default_view = textures[0].clone();
        let texture_array = (0..ctx.texture_array_len(textures.len())).map(|index| {
            textures
                .get(index)
                .cloned()
                .unwrap_or_else(|| default_view.clone())
        });
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout(device, self.pipeline, TEXTURE_SET),
            [
                WriteDescriptorSet::image_view_array(0, 0, texture_array),
                WriteDescriptorSet::sampler(1, self.sampler.clone()),
            ],
            [],
        )
        .unwrap()
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "a pass records from the frame's bindings; a params struct only renames the list"
    )]
    pub(super) fn record(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        renderer: &PassCtx<'_>,
        draws: Draws<'_>,
        extent: [u32; 2],
        frame: Subbuffer<FrameUbo>,
        decals: Subbuffer<super::forward::GpuDecals>,
        object_set: Arc<DescriptorSet>,
        material_set: Arc<DescriptorSet>,
        texture_set: Arc<DescriptorSet>,
    ) {
        let Device::Vulkan(driver) = device;
        let frame_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            layout(device, self.pipeline, 0),
            [
                WriteDescriptorSet::buffer(0, frame),
                // The very buffer object the forward pass's set 0 binds. See
                // `ForwardPass::upload_decals` — one allocation bound twice is
                // what keeps the two passes from stamping different decals onto
                // the same pixel.
                WriteDescriptorSet::buffer(1, decals),
            ],
            [],
        )
        .unwrap();

        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });

        let sets = [
            driver.import_set(frame_set),
            driver.import_set(object_set),
            driver.import_set(material_set),
            driver.import_set(texture_set),
        ];
        let bind_all = |encoder: &mut Encoder, pipeline: PipelineId| {
            encoder.bind_pipeline(pipeline);
            for (set, group) in sets.iter().enumerate() {
                encoder.bind_group(set as u32, *group);
            }
        };

        // Bound once for the pass: a mesh is a span into the arena, so nothing
        // between draws changes them. See `mesh`.
        encoder.bind_geometry(renderer.arena, false);

        let mut bound: Option<bool> = None;

        // One instanced draw per (mesh, material) unit, matching the forward
        // pass. The model and normal matrices come from the shared object
        // buffer, so this pass no longer recomputes an inverse-transpose per
        // item.
        for unit in draws.units() {
            let Some(mesh) = renderer.meshes.get(unit.mesh as usize) else {
                continue;
            };
            // The same question the forward pass asks of the same flag word, so
            // the two passes cut a cutout out at the same place. See
            // `GpuMaterial::is_masked`.
            let wants_masked = renderer
                .materials
                .get(unit.material as usize)
                .is_some_and(super::forward::GpuMaterial::is_masked);
            if bound != Some(wants_masked) {
                bind_all(
                    encoder,
                    if wants_masked {
                        self.masked_pipeline
                    } else {
                        self.pipeline
                    },
                );
                bound = Some(wants_masked);
            }
            encoder.draw_indexed(
                mesh.span.index_count,
                unit.instances,
                mesh.span.first_index,
                mesh.span.vertex_offset,
                unit.object_base,
            );
        }

        // The compute path draws the same geometry as two multi-draws, one per
        // pipeline variant. Empty on the CPU path, and vice versa.
        for region in draws.regions() {
            bind_all(
                encoder,
                if region.masked {
                    self.masked_pipeline
                } else {
                    self.pipeline
                },
            );
            encoder.draw_indexed_indirect(driver.import_buffer(region.commands.into_bytes()));
        }
    }
}
