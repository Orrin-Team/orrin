//! What the frame traces rays for, and what goes into the scene they are traced
//! against.
//!
//! The acceleration structures are the driver's (`accel.rs`). This module
//! decides what stands in them each frame, uploads what a hit needs to become a
//! lit surface, and owns the debug views, which draw the traced scene in place
//! of the frame, and the path-traced reference (`path_trace.comp`). A pass that
//! traces for its own reasons — the reflections, when the screen runs out —
//! binds what is prepared here through [`RayFrame`].
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
//! # What a hit reads
//!
//! Three sets, the same for every shader that shades a hit (`ray_shade.glsl`):
//!
//! - The scene's structure and a table with a row per instance: its material,
//!   where its mesh starts in the shared geometry, and its light channels. An
//!   instance carries its row's number as its id. Uploaded each frame and not
//!   read out of the per-object rows, which persist and are rewritten at the
//!   head of the next frame: what a hit reads is this frame's, whatever the
//!   two frames' order on the device.
//! - The frame's own lighting block and light tables, the material table and
//!   the environment cube — the buffers the forward pass binds, so a hit is
//!   lit by what its reflector is — and the world light grid in place of the
//!   view's clusters (`gfx::light_grid`).
//! - The material texture array.
//!
//! A hit reads a material's standard reading, with none of the optional lobes
//! and no custom surface function: `gfx/surface/mod.rs` says why.

use glam::{Mat4, Vec3};

use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::{BufferUsage, ImageUsage};
use crate::gfx::light_grid::LightGrid;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, BlockKind, BufferDesc, BufferId, ComputePipelineDesc,
    Device, Encoder, GeometryStream, GraphicsPipelineDesc, MemoryLocation, PipelineId, RayInstance,
    SamplerId, TextureDesc, TextureDimension, TextureViewId, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::{DrawList, RenderItem, SceneLighting};
use crate::scene::RayTracingSettings;

use super::forward::{GpuMaterial, GpuMesh, LightBlocks};
use super::gi::{GI_SET, GiViews};

/// The debug block as the words of its std140 layout: `inv_view_proj`, then
/// `origin` and `mode`.
const DEBUG_WORDS: usize = 16 + 4 + 4;

/// The world light grid as the device holds it: every cell's range, and the
/// lists they index.
struct GridBuffers {
    cells: BufferId,
    lights: BufferId,
}

/// The set a traced compute pipeline reads the material textures from.
pub(super) const TEXTURE_SET: u32 = 3;

/// The path block as the words of its std140 layout: `inv_view_proj`, then
/// `origin` and `frame`.
const PATH_WORDS: usize = 16 + 4 + 4;

/// The reference path tracer's own state: what it has summed so far, and for
/// which view.
struct PathTracer {
    pipeline: PipelineId,
    /// The material textures, bound for this pipeline's layout and kept until
    /// they change.
    textures: Option<BindGroupId>,
    /// Every path's radiance so far, and the extent it was made at.
    sum: Option<(TextureViewId, [u32; 2])>,
    /// How many paths each pixel of `sum` holds.
    paths: u32,
    /// The view those paths were traced through. A different one starts the
    /// sum again: paths through another camera are of another picture.
    view: Mat4,
    uniforms: Option<BufferId>,
}

pub struct RayTracePass {
    debug_pipeline: PipelineId,
    path: PathTracer,
    /// Reused each frame, with the table beside it, so a frame that traces
    /// allocates nothing for either once it has seen the scene.
    instances: Vec<RayInstance>,
    /// One row per entry of `instances`: material, first index, vertex offset,
    /// light channels.
    table: Vec<[u32; 4]>,
    table_buffer: Option<BufferId>,
    grid: LightGrid,
    /// Kept until the lists change, which is when a light moves or the camera
    /// leaves a cell of the grid's finest cascade.
    grid_buffers: Option<GridBuffers>,
    /// The cascades the lists were binned round, uploaded with the frame.
    grid_header: Option<BufferId>,
    debug_uniforms: Option<BufferId>,
}

/// What a pass that shades a ray's hits binds, for the frame being recorded.
#[derive(Clone, Copy)]
pub(super) struct RayFrame<'a> {
    pub pass: &'a RayTracePass,
    pub lights: LightBlocks,
    pub materials: BufferId,
    pub environment: TextureViewId,
    pub environment_sampler: SamplerId,
    /// The irradiance volumes, or their inert stand-ins.
    pub gi: GiViews,
}

/// Which parts of sets 2 and 4 a traced shader reads. A pipeline's layout
/// holds only the bindings its shader reads, and a set is written for its
/// layout, so each pass names its part rather than every pass binding all.
#[derive(Clone, Copy)]
pub(super) struct RayReads {
    /// The lighting block, the light tables and the world light grid.
    pub lights: bool,
    pub materials: bool,
    pub environment: bool,
    /// The vertex stream a hit's normals, tangents and colours are read from.
    pub surfaces: bool,
    /// The irradiance volumes, set 4.
    pub gi: bool,
}

impl RayReads {
    /// A shader that runs the whole of `ray_shade`.
    pub(super) const SHADING: Self = Self {
        lights: true,
        materials: true,
        environment: true,
        surfaces: true,
        gi: true,
    };
}

impl RayFrame<'_> {
    /// Bind sets 1 and 2 of `pipeline`, whose shader includes `ray_shade.glsl`,
    /// with what `reads` says of set 2, and set 4 where it reads the volumes.
    /// Set 3, the textures, is the pipeline's owner's: it is kept between
    /// frames, and a set belongs to the layout it was made for.
    pub(super) fn bind(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        pipeline: PipelineId,
        reads: RayReads,
    ) {
        let pass = self.pass;
        let grid = pass
            .grid_buffers
            .as_ref()
            .expect("a pass shaded a hit before begin_frame");
        let uniform = |binding, buffer| Binding {
            binding,
            resource: BindingResource::UniformBuffer(buffer),
        };
        let storage = |binding, buffer| Binding {
            binding,
            resource: BindingResource::StorageBuffer(buffer),
        };
        let lights = [
            uniform(0, self.lights.lighting),
            uniform(1, self.lights.points),
            uniform(2, self.lights.spots),
            uniform(
                6,
                pass.grid_header
                    .expect("a pass shaded a hit before begin_frame"),
            ),
            storage(7, grid.cells),
            storage(8, grid.lights),
        ];
        let environment = [
            Binding {
                binding: 4,
                resource: BindingResource::Texture(self.environment),
            },
            Binding {
                binding: 5,
                resource: BindingResource::Sampler(self.environment_sampler),
            },
        ];
        let materials = [storage(3, self.materials)];
        let surfaces = [Binding {
            binding: 9,
            resource: BindingResource::Geometry(GeometryStream::Surfaces),
        }];
        let mut bindings = [lights[0]; 10];
        let mut count = 0;
        let parts: [(bool, &[Binding]); 4] = [
            (reads.lights, &lights),
            (reads.materials, &materials),
            (reads.environment, &environment),
            (reads.surfaces, &surfaces),
        ];
        for binding in parts
            .iter()
            .filter(|(read, _)| *read)
            .flat_map(|(_, part)| part.iter())
        {
            bindings[count] = *binding;
            count += 1;
        }
        let shading = device.create_transient_bind_group(pipeline, 2, &bindings[..count]);
        encoder.bind_group(1, pass.hit_group(device, pipeline));
        encoder.bind_group(2, shading);
        if reads.gi {
            encoder.bind_group(GI_SET, self.gi.group(device, pipeline, GI_SET, 0));
        }
    }
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
        let path = PathTracer {
            pipeline: device.create_compute_pipeline_over_textures(
                &ComputePipelineDesc {
                    shader: ShaderId::PathTraceComp.into(),
                },
                TEXTURE_SET,
            ),
            textures: None,
            sum: None,
            paths: 0,
            view: Mat4::ZERO,
            uniforms: None,
        };
        Self {
            debug_pipeline,
            path,
            instances: Vec::new(),
            table: Vec::new(),
            table_buffer: None,
            grid: LightGrid::default(),
            grid_buffers: None,
            grid_header: None,
            debug_uniforms: None,
        }
    }

    /// Hand the device this frame's traced scene, bin the lights round the
    /// camera, and resolve the debug view's camera. Before anything records:
    /// the build passes record what this decides.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_frame(
        &mut self,
        device: &Device,
        settings: &RayTracingSettings,
        draws: DrawList<'_>,
        meshes: &[GpuMesh],
        materials: &[GpuMaterial],
        lighting: &SceneLighting,
        camera: Vec3,
        unjittered_view_proj: Mat4,
    ) {
        let reach = settings.distance.max(0.0);
        self.instances.clear();
        self.table.clear();
        for i in 0..draws.len() {
            let item = draws.item(i);
            let Some(mesh) = meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            if !within(item, mesh, camera, reach) {
                continue;
            }
            self.instances.push(RayInstance {
                span: mesh.span,
                transform: [
                    item.model.row(0).to_array(),
                    item.model.row(1).to_array(),
                    item.model.row(2).to_array(),
                ],
                id: self.table.len() as u32,
                opaque: !materials
                    .get(item.material.0 as usize)
                    .is_some_and(GpuMaterial::is_masked),
            });
            self.table.push([
                item.material.0,
                mesh.span.first_index,
                mesh.span.vertex_offset as u32,
                item.channels,
            ]);
        }
        device.prepare_ray_scene(&self.instances);
        // Never empty, for the reason no bound buffer here is: a scene of
        // nothing still binds a table, which nothing then indexes.
        if self.table.is_empty() {
            self.table.push([0; 4]);
        }
        self.table_buffer =
            Some(device.upload_block(BlockKind::Storage, bytemuck::cast_slice(&self.table)));

        if self.grid.bin(lighting, camera) || self.grid_buffers.is_none() {
            if let Some(old) = self.grid_buffers.take() {
                device.destroy_buffer(old.cells);
                device.destroy_buffer(old.lights);
            }
            let buffer = |words: &[u32]| {
                let bytes: &[u8] = bytemuck::cast_slice(words);
                device.create_buffer_bytes(
                    BufferDesc {
                        size: bytes.len() as u64,
                        usage: BufferUsage::STORAGE_BUFFER,
                        location: MemoryLocation::GpuOnly,
                    },
                    bytes,
                )
            };
            self.grid_buffers = Some(GridBuffers {
                cells: buffer(bytemuck::cast_slice(&self.grid.ranges)),
                lights: if self.grid.indices.is_empty() {
                    buffer(&[0])
                } else {
                    buffer(&self.grid.indices)
                },
            });
        }
        let mut header = [0u32; 4 * crate::gfx::light_grid::CASCADES];
        for (words, cascade) in header
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&self.grid.cascades)
        {
            let origin = cascade.origin;
            *words = [origin.x, origin.y, origin.z, cascade.cell].map(f32::to_bits);
        }
        self.grid_header = Some(device.upload_uniform(&header));

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

    /// Ready the reference for a frame that draws it: the image its paths are
    /// summed in, at `extent`, and which path of each pixel this frame traces.
    /// Returns that image, for the graph to be handed. After
    /// [`begin_frame`](Self::begin_frame).
    pub(super) fn begin_path(
        &mut self,
        device: &Device,
        extent: [u32; 2],
        camera: Vec3,
        unjittered_view_proj: Mat4,
        textures: &[TextureViewId],
        sampler: SamplerId,
    ) -> TextureViewId {
        let path = &mut self.path;
        if path.textures.is_none() {
            path.textures = Some(device.create_texture_array_group(
                path.pipeline,
                TEXTURE_SET,
                textures,
                sampler,
            ));
        }
        if path.sum.is_none_or(|(_, made_at)| made_at != extent) {
            if let Some((old, _)) = path.sum.take() {
                device.destroy_texture(old);
            }
            let sum = device.create_texture(&TextureDesc {
                format: Format::Rgba32Float,
                dimension: TextureDimension::D2,
                extent: [extent[0], extent[1], 1],
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::Sample1,
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                memoryless: false,
            });
            device.initialise_sampled(sum);
            path.sum = Some((sum, extent));
            path.paths = 0;
        }
        if path.view != unjittered_view_proj {
            path.view = unjittered_view_proj;
            path.paths = 0;
        }

        let mut words = [0u32; PATH_WORDS];
        let floats = unjittered_view_proj
            .inverse()
            .to_cols_array()
            .into_iter()
            .chain([camera.x, camera.y, camera.z, 0.0]);
        for (word, float) in words.iter_mut().zip(floats) {
            *word = float.to_bits();
        }
        words[20] = path.paths;
        path.uniforms = Some(device.upload_uniform(&words));
        path.paths = path.paths.saturating_add(1);
        path.sum.expect("made above").0
    }

    /// Let go of the reference's texture set, which names textures that have
    /// changed.
    pub(super) fn forget_textures(&mut self, device: &Device) {
        if let Some(group) = self.path.textures.take() {
            device.destroy_bind_group(group);
        }
    }

    /// One more path per pixel, into the sum and out as its mean.
    pub(super) fn record_path(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        rays: &RayFrame<'_>,
        sum: TextureViewId,
        mean: TextureViewId,
    ) {
        let path = &self.path;
        let targets = device.create_transient_bind_group(
            path.pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::StorageTexture(sum),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageTexture(mean),
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::UniformBuffer(
                        path.uniforms
                            .expect("the graph scheduled the reference before begin_path"),
                    ),
                },
            ],
        );
        encoder.bind_pipeline(path.pipeline);
        encoder.bind_group(0, targets);
        // Every bounce is followed, so no hit reads the volumes.
        rays.bind(
            device,
            encoder,
            path.pipeline,
            RayReads {
                gi: false,
                ..RayReads::SHADING
            },
        );
        encoder.bind_group(
            TEXTURE_SET,
            path.textures
                .expect("the graph scheduled the reference before begin_path"),
        );
        let extent = device.view_extent(mean);
        encoder.dispatch([extent[0].div_ceil(8), extent[1].div_ceil(8), 1]);
    }

    /// One ray per pixel, drawn over the whole target.
    pub(super) fn record_debug(&self, device: &Device, encoder: &mut Encoder, extent: [u32; 2]) {
        let uniforms = self
            .debug_uniforms
            .expect("the graph scheduled the ray debug view before begin_frame");
        let view = device.create_transient_bind_group(
            self.debug_pipeline,
            0,
            &[Binding {
                binding: 0,
                resource: BindingResource::UniformBuffer(uniforms),
            }],
        );

        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });
        encoder.bind_pipeline(self.debug_pipeline);
        encoder.bind_group(0, view);
        encoder.bind_group(1, self.hit_group(device, self.debug_pipeline));
        encoder.draw(3, 1, 0, 0);
    }

    /// Set 1 of every shader that includes `ray_hit.glsl`: the scene, the
    /// table a hit's id indexes and the geometry its triangle is in.
    fn hit_group(&self, device: &Device, pipeline: PipelineId) -> BindGroupId {
        device.create_transient_bind_group(
            pipeline,
            1,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SceneAccel,
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(
                        self.table_buffer
                            .expect("a pass traced the scene before begin_frame"),
                    ),
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::Geometry(GeometryStream::Positions),
                },
                Binding {
                    binding: 3,
                    resource: BindingResource::Geometry(GeometryStream::Indices),
                },
            ],
        )
    }
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
