//! What the frame traces rays for, and what goes into the scene they are traced
//! against.
//!
//! The acceleration structures are the driver's (`accel.rs`). This module
//! decides what stands in them each frame and owns the passes that trace:
//! today the debug views, which draw the traced scene in place of the frame.
//!
//! # What a ray can hit
//!
//! Everything the opaque queue draws that stands within
//! [`RayTracingSettings::distance`] of the camera, whichever way the camera
//! faces. That is a cull of its own and deliberately not `cull.comp`'s: the
//! frustum is the one volume a ray is traced to get out of. Tested on the host,
//! against the sphere round each mesh's bounds, because the instance list is
//! the host's to write anyway — the device builds from addresses the host
//! resolved.
//!
//! Blended and refractive surfaces are left out. A ray asks what is there, and
//! for those the honest answer is a fraction this has no way to return.
//!
//! An instance carries its row in the per-object buffer as its id, so a hit
//! names the row and the row names the mesh and the material. See
//! `ray_hit.glsl`.

use glam::{Mat4, Vec3};

use crate::gfx::format::Format;
use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, Device, Encoder, GeometryStream, GraphicsPipelineDesc,
    PipelineId, RayInstance, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::{DrawList, RenderItem};
use crate::scene::RayTracingSettings;

use super::forward::{GpuMaterial, GpuMesh};

/// The debug block as the words of its std140 layout: `inv_view_proj`, then
/// `origin` and `mode`.
const DEBUG_WORDS: usize = 16 + 4 + 4;

pub struct RayTracePass {
    debug_pipeline: PipelineId,
    /// Reused each frame, so a frame that traces allocates nothing for the
    /// list once it has seen the scene.
    instances: Vec<RayInstance>,
    debug_uniforms: Option<BufferId>,
}

impl RayTracePass {
    /// Only on a device that traces: the pipeline's shader is one another
    /// device would refuse.
    pub fn new(device: &Device, swapchain_format: Format) -> Self {
        let debug_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc::fullscreen(
            ShaderId::FullscreenVert,
            ShaderId::RayDebugFrag,
            &[swapchain_format],
        ));
        Self {
            debug_pipeline,
            instances: Vec::new(),
            debug_uniforms: None,
        }
    }

    /// Hand the device this frame's traced scene, and resolve the debug view's
    /// camera. Before anything records: the build passes record what this
    /// decides.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_frame(
        &mut self,
        device: &Device,
        settings: &RayTracingSettings,
        draws: DrawList<'_>,
        meshes: &[GpuMesh],
        materials: &[GpuMaterial],
        camera: Vec3,
        unjittered_view_proj: Mat4,
    ) {
        let reach = settings.distance.max(0.0);
        self.instances.clear();
        self.instances.extend((0..draws.len()).filter_map(|i| {
            let item = draws.item(i);
            let mesh = meshes.get(item.mesh.0 as usize)?;
            within(item, mesh, camera, reach).then(|| RayInstance {
                span: mesh.span,
                transform: [
                    item.model.row(0).to_array(),
                    item.model.row(1).to_array(),
                    item.model.row(2).to_array(),
                ],
                id: item.instance,
                opaque: !materials
                    .get(item.material.0 as usize)
                    .is_some_and(GpuMaterial::is_masked),
            })
        }));
        device.prepare_ray_scene(&self.instances);

        let mut words = [0u32; DEBUG_WORDS];
        let floats = unjittered_view_proj
            .inverse()
            .to_cols_array()
            .into_iter()
            .chain([camera.x, camera.y, camera.z, reach]);
        for (word, float) in words.iter_mut().zip(floats) {
            *word = float.to_bits();
        }
        words[20] = settings.debug_view.mode();
        self.debug_uniforms = Some(device.upload_uniform(&words));
    }

    /// One ray per pixel, drawn over the whole target.
    pub(super) fn record_debug(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        extent: [u32; 2],
        object_rows: BufferId,
    ) {
        let uniforms = self
            .debug_uniforms
            .expect("the graph scheduled the ray debug view before begin_frame");
        let scene = device.create_transient_bind_group(
            self.debug_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SceneAccel,
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::UniformBuffer(uniforms),
                },
            ],
        );
        let hit = hit_group(device, self.debug_pipeline, object_rows);

        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });
        encoder.bind_pipeline(self.debug_pipeline);
        encoder.bind_group(0, scene);
        encoder.bind_group(1, hit);
        encoder.draw(3, 1, 0, 0);
    }
}

/// Set 1 of every shader that includes `ray_hit.glsl`: the rows a hit's id
/// indexes and the geometry its triangle is in.
fn hit_group(
    device: &Device,
    pipeline: PipelineId,
    object_rows: BufferId,
) -> crate::gfx::rhi::BindGroupId {
    device.create_bind_group(
        pipeline,
        1,
        &[
            Binding {
                binding: 0,
                resource: BindingResource::StorageBuffer(object_rows),
            },
            Binding {
                binding: 1,
                resource: BindingResource::Geometry(GeometryStream::Positions),
            },
            Binding {
                binding: 2,
                resource: BindingResource::Geometry(GeometryStream::Indices),
            },
        ],
    )
}

/// Whether any of `item` stands within `reach` of `camera`, by the sphere
/// round its mesh's bounds. The item's own bounds are not used: under GPU
/// culling an opaque item carries none.
fn within(item: &RenderItem, mesh: &GpuMesh, camera: Vec3, reach: f32) -> bool {
    let centre = item.model.transform_point3(mesh.bounds.center().into());
    let scale = item
        .model
        .x_axis
        .truncate()
        .length()
        .max(item.model.y_axis.truncate().length())
        .max(item.model.z_axis.truncate().length());
    let radius = mesh.bounds.half_extents().length() * scale;
    centre.distance(camera) - radius <= reach
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Aabb;
    use crate::gfx::rhi::MeshSpan;
    use crate::scene::{MaterialHandle, MeshHandle, ShadowCasting};

    fn item(model: Mat4) -> RenderItem {
        RenderItem {
            model,
            prev_model: model,
            bounds: Aabb::EMPTY,
            mesh: MeshHandle(0),
            material: MaterialHandle(0),
            instance: 0,
            channels: 0,
            shadows: ShadowCasting::default(),
        }
    }

    /// The cull is by distance to the nearest point of the bounds and ignores
    /// where the camera looks, so what is behind it stays traceable, and a
    /// scaled object reaches as far as it is drawn.
    #[test]
    fn an_object_is_traced_by_distance_whichever_way_the_camera_faces() {
        let mesh = GpuMesh {
            span: MeshSpan::default(),
            bounds: Aabb::from_points([Vec3::splat(-1.0), Vec3::splat(1.0)]),
            vertices: 0,
        };
        let camera = Vec3::ZERO;
        let behind = item(Mat4::from_translation(Vec3::new(0.0, 0.0, 40.0)));
        let far = item(Mat4::from_translation(Vec3::new(0.0, 0.0, -80.0)));
        let far_but_large = item(
            Mat4::from_translation(Vec3::new(0.0, 0.0, -80.0))
                * Mat4::from_scale(Vec3::splat(20.0)),
        );
        assert!(within(&behind, &mesh, camera, 50.0));
        assert!(!within(&far, &mesh, camera, 50.0));
        assert!(within(&far_but_large, &mesh, camera, 50.0));
    }
}
