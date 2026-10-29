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

use crate::gfx::format::SampleCount;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, Blend, BlockKind, BufferId, CullMode, DepthCompare,
    DepthTest, Device, Encoder, GraphicsPipelineDesc, PipelineId, SamplerDesc, SamplerId,
    TextureViewId, Topology, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::{PositionVertex, SurfaceVertex};

use super::PassCtx;
use super::cull::Draws;
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
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
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
    sampler: SamplerId,
}

impl GeometryPrepass {
    pub fn new(device_rhi: &Device) -> Self {
        let pipeline = |fragment, cull| {
            device_rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex: ShaderId::PrepassVert,
                fragment,
                color_formats: &[NORMAL_FORMAT, VELOCITY_FORMAT, MATERIAL_FORMAT],
                depth_format: Some(DEPTH_FORMAT),
                vertex_streams: &[PositionVertex::STREAM, SurfaceVertex::STREAM],
                cull,
                depth_test: Some(DepthTest {
                    write: true,
                    compare: DepthCompare::Less,
                }),
                partial_texture_set: Some(TEXTURE_SET as u32),
                dynamic_depth_bias: false,
                samples: SampleCount::Sample1,
                alpha_to_coverage: false,
                immutable_samplers: &[],
                blend: Blend::None,
                topology: Topology::Triangles,
                mask_extra_attachments: false,
                attachment_blends: &[],
                layout_from: None,
                layout_prefix_from: None,
            })
        };
        let pipeline_plain = pipeline(ShaderId::PrepassFrag, CullMode::Back);
        // Two-sided for a cutout, matching the forward pass. The two rasterise
        // the same triangles or the depth this pass leaves is not the depth the
        // frame was shaded against.
        let masked_pipeline = pipeline(ShaderId::PrepassFragMasked, CullMode::None);
        let sampler = device_rhi.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: true,
            repeat: true,
            compare: false,
            anisotropy: device_rhi.max_anisotropy(),
            max_lod: None,
        });

        Self {
            pipeline: pipeline_plain,
            masked_pipeline,
            sampler,
        }
    }

    /// Upload the camera block once for every pass in the frame that reads it.
    pub(super) fn begin_frame(&self, device: &Device, view: &FrameView) -> BufferId {
        let frame = FrameUbo {
            view: view.view.to_cols_array_2d(),
            proj: view.proj.to_cols_array_2d(),
            inv_proj: view.proj.inverse().to_cols_array_2d(),
            prev_view_proj: view.prev_view_proj.to_cols_array_2d(),
            jitter: [view.jitter.x, view.jitter.y, 0.0, 0.0],
            view_proj: view.view_proj.to_cols_array_2d(),
        };
        device.upload_block(BlockKind::Uniform, bytemuck::bytes_of(&frame))
    }

    pub(super) fn material_sampler(&self) -> SamplerId {
        self.sampler
    }

    /// The set-1 per-object descriptor set this pass binds.
    pub(super) fn build_object_set(
        &self,
        device: &Device,
        rows: BufferId,
        indices: BufferId,
    ) -> BindGroupId {
        device.create_transient_bind_group(
            self.pipeline,
            1,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::StorageBuffer(rows),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(indices),
                },
            ],
        )
    }

    pub(super) fn build_material_set(&self, device: &Device, materials: BufferId) -> BindGroupId {
        device.create_persistent_bind_group(
            self.pipeline,
            2,
            &[Binding {
                binding: 0,
                resource: BindingResource::StorageBuffer(materials),
            }],
        )
    }

    /// The set-3 texture array, filled exactly as the forward pass fills its
    /// own: every unused slot points at the white default so no descriptor is
    /// left unwritten.
    pub(super) fn build_texture_set(
        &self,
        device: &Device,
        textures: &[TextureViewId],
    ) -> BindGroupId {
        device.create_texture_array_group(self.pipeline, TEXTURE_SET as u32, textures, self.sampler)
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
        frame: BufferId,
        decals: BufferId,
        object_set: BindGroupId,
        material_set: BindGroupId,
        texture_set: BindGroupId,
    ) {
        let frame_set = device.create_transient_bind_group(
            self.pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::UniformBuffer(frame),
                },
                // The very buffer object the forward pass's set 0 binds. See
                // `ForwardPass::upload_decals` — one allocation bound twice is
                // what keeps the two passes from stamping different decals onto
                // the same pixel.
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(decals),
                },
            ],
        );

        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });

        let sets = [frame_set, object_set, material_set, texture_set];
        let bind_all = |encoder: &mut Encoder, pipeline: PipelineId| {
            encoder.bind_pipeline(pipeline);
            for (set, group) in sets.iter().enumerate() {
                encoder.bind_group(set as u32, *group);
            }
        };

        // Bound once for the pass: a mesh is a span into the arena, so nothing
        // between draws changes them. See `mesh`.
        encoder.bind_geometry(false);

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
            encoder.draw_indexed_indirect(region.commands, region.range);
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod metal_tests {
    use super::*;
    use crate::gfx::format::Format;

    /// The material array is padded to its full length with the white default,
    /// and binding it declares each texture it holds to Metal once, not once
    /// per slot.
    #[test]
    #[ignore = "needs a Metal device"]
    fn a_padded_texture_array_declares_each_texture_once() {
        let device = Device::Metal(
            crate::gfx::metal::MtlDriver::new().expect("this machine runs the Metal backend"),
        );
        device.begin_frame(0);
        let prepass = GeometryPrepass::new(&device);
        let white = device.create_texture_2d(&[255; 4], [1, 1], Format::Rgba8Unorm);
        let black = device.create_texture_2d(&[0, 0, 0, 255], [1, 1], Format::Rgba8Unorm);
        let group = prepass.build_texture_set(&device, &[white, black, white]);
        assert_eq!(device.metal().declared_uses(group), 2);
    }
}
