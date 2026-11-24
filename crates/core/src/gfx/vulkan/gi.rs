//! Global illumination: irradiance volumes round the camera, traced a cascade
//! a frame, and the world radiance cache that keeps the cost of lighting
//! their rays flat. `gfx/probes.rs` is the probe field's layout; this is what
//! the frame does with it.
//!
//! DDGI-style probes in id Tech 8's cascaded layout (Majercik et al. 2019,
//! Sousa 2025), over a spatial-hash cache (Binder et al. 2019, Gautron 2020)
//! — chosen in #132 over surfels, voxel cone tracing and screen-space history
//! because it traces no ray per pixel, so nothing on screen is noisy, and its
//! cost is set by the probe count rather than by the resolution.
//!
//! # A frame
//!
//! Four dispatches, at the head of the frame, behind the scene's build:
//!
//! 1. `gi_update`: each probe of the grid last frame traced has its tile
//!    blended toward what its rays brought back — a hit's light read out of
//!    the cache by the cell the trace filed it under, a miss's from the sky.
//! 2. `gi_trace`: every probe of the grid this frame updates traces its rays.
//!    Only what each ray met is recorded — where, how far, front or back — and
//!    a front hit is filed in the cache under the cell it landed in, marking
//!    the cell used.
//! 3. `gi_cache_compact`: one look at every cell. Used this frame, it goes on
//!    a list, counted into the next dispatch's arguments as it grows; unused
//!    for [`LIFETIME`] frames, it is emptied, which is how the table follows
//!    the camera with no rebuild; otherwise its light is carried forward.
//! 4. `gi_cache_shade`: each listed cell shaded once, with the hit shading
//!    every traced pass runs, from where the lowest-numbered ray that used it
//!    landed. Its bounce light comes from the volumes the update just wrote,
//!    which is how one bounce becomes many — from the probes a ray from the
//!    cell reaches, since this is the one lookup whose errors are fed back.
//!
//! The light a cell bounces is read from a third atlas, each probe's
//! irradiance over its last few updates alone, not from the one a surface
//! reads. A probe can average hundreds of updates where its rays disagree,
//! and light reaches a probe through the cells at the rate it blends, so a
//! room fed from that atlas would take as many updates to fill with light, or
//! to go dark, for every bounce it has. A surface reads the steady one.
//!
//! Not all of it over the last few updates, though. In an aisle lit through
//! its arches nearly all of a probe's light is the one ray in a few dozen
//! updates that meets a sunlit floor, a hundred times the rest; over two
//! updates that ray lit every cell round the probe for a few frames, the final
//! gather read those cells, and a dark urn's light swung by a third and its
//! wall's by a tenth every hundred frames. So an update's brightest ray, where
//! it is `OUTLYING` times the update's mean, has its part kept in a fourth
//! atlas and blended as long as the irradiance, and the bounce tile is the
//! rest over the last few updates plus that. Each part is a mean of its own
//! with weights fixed before it is seen, so the sum is the light's whole mean;
//! a lamp's brightest ray is a few times the rest of a room it lights, and its
//! light comes and goes at the pace it did. Not the whole irradiance for the
//! bounce, which steadies the aisle as well but leaves a switched lamp's light
//! in a closed room for seconds.
//!
//! The update reads the cache's light and the shading writes it, while the
//! shading reads the volumes and the update writes them: a cycle within the
//! frame. So the cache's light is a pair that swaps each frame, like the
//! occlusion pyramids, and so are the rays: the update takes in last frame's
//! rays from the half last frame's shading wrote, in which every cell they
//! landed in has been shaded. Not this frame's rays from last frame's cells:
//! a cell first used this frame has no light yet, and a probe in a room whose
//! walls are new to the cache would be lit by its windows' sky alone, until a
//! history of hundreds of updates forgot it. A ray the cache could not answer
//! even so — no room in the table, or its hit not found again — takes the
//! mean of its probe's others.
//!
//! # The cache
//!
//! A flat table of [`CELLS`] cells, keyed by a hit's position quantised to a
//! cell, the cell's level of detail and which of six ways the surface faces
//! (`gi_cache.glsl`), with a second hash as checksum and linear probing on a
//! collision. A cell is a quarter of a metre near the camera and doubles with
//! distance. Each is shaded every frame it is used, from wherever a ray landed
//! in it, and holds the mean of its last few shadings: a cell across a shadow's
//! edge would otherwise be lit one frame and dark the next, for every probe
//! ray that reads it at once, and a bad value still lasts only a few frames.
//!
//! The final gather (`vulkan::gi_gather`) reads the cache too, and claims the
//! cells its rays land in that no probe ray has: such a cell is marked used
//! for the next frame, so that frame's list holds it, and is shaded from the
//! gather's ray, left in a table of hits beside the cache.
//!
//! # Memory
//!
//! At sixteen probes a side: the four atlases 79 MB, the two frames' rays
//! 16.8 MB, the cache 8.4 MB, the gather's hits beside it 8.4 MB and its list
//! 1 MB, the probes' state 0.4 MB. Twelve a side takes the atlases to 33 MB
//! and the rays to 7.1 MB. All of it is made when the volumes first go live
//! and let go when they stop.
//!
//! Off — on a device that cannot trace, or with the setting off — every pass
//! that reads the volumes binds inert stand-ins whose block says so, and
//! takes the environment's diffuse series exactly as before.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::{BufferUsage, ImageUsage};
use crate::gfx::probes::{self, CASCADES, MAX_RAYS, Schedule};
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, BlockKind, BufferDesc, BufferId, ComputePipelineDesc,
    Device, Encoder, MemoryLocation, PipelineId, SamplerDesc, SamplerId, TextureDesc,
    TextureDimension, TextureViewId,
};
use crate::gfx::sh::{self, SH9};
use crate::gfx::shaders::ShaderId;
use crate::scene::GiSettings;

use super::gi_gather::{GatherCache, GatherPass};
use super::raytrace::{RayFrame, RayReads, TEXTURE_SET};

/// The set a traced shader reads the volumes at. See `ray_shade.glsl`.
pub(super) const GI_SET: u32 = 4;
/// Every atlas. Half floats, not the shared exponent id ships irradiance in:
/// `E5B9G9R9` is not a format this device family can store to, and a probe's
/// update reads and writes its own texels.
pub(super) const ATLAS_FORMAT: Format = Format::Rgba16Float;
/// Cells in the radiance cache. A power of two: a key is masked into it.
const CELLS: u32 = 1 << 18;
/// Frames a cell outlives its last use: past a full round of the cascades,
/// so a cell only one grid's rays reach survives until they come back.
const LIFETIME: u32 = 4 * CASCADES as u32;
/// The cache's cell nearest the camera, in metres.
const FINEST_CELL: f32 = 0.25;
/// Invocations a workgroup, for the passes with one invocation per ray or
/// per cell. Keep in sync with the shaders, `SHADE_GROUP` in
/// `gi_cache_compact.comp` above all: that is what its count of groups counts.
const GROUP: u32 = 64;

/// The volumes' block as `gi.glsl` declares it, std140.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuGi {
    grid: [u32; 4],
    frame: [u32; 4],
    params: [f32; 4],
    camera: [f32; 4],
    environment: [f32; 4],
    rotation: [[f32; 4]; 3],
    cascades: [[f32; 4]; CASCADES],
    sky: [[f32; 4]; SH9],
}

/// What a shader that reads the volumes binds, as `gi.glsl` lays it out: the
/// block, the probes' state, the two atlases and their sampler.
#[derive(Clone, Copy)]
pub(super) struct GiViews {
    pub block: BufferId,
    pub probes: BufferId,
    pub irradiance: TextureViewId,
    pub distance: TextureViewId,
    pub sampler: SamplerId,
}

impl GiViews {
    /// The five bindings from `first` on.
    pub(super) fn bindings(&self, first: u32) -> [Binding; 5] {
        [
            Binding {
                binding: first,
                resource: BindingResource::UniformBuffer(self.block),
            },
            Binding {
                binding: first + 1,
                resource: BindingResource::StorageBuffer(self.probes),
            },
            Binding {
                binding: first + 2,
                resource: BindingResource::Texture(self.irradiance),
            },
            Binding {
                binding: first + 3,
                resource: BindingResource::Texture(self.distance),
            },
            Binding {
                binding: first + 4,
                resource: BindingResource::Sampler(self.sampler),
            },
        ]
    }

    /// Set `set` of `pipeline`, holding nothing but the five from `first`.
    pub(super) fn group(
        &self,
        device: &Device,
        pipeline: PipelineId,
        set: u32,
        first: u32,
    ) -> BindGroupId {
        device.create_transient_bind_group(pipeline, set, &self.bindings(first))
    }
}

/// The probes, at one size and spacing.
struct Volumes {
    probes: u32,
    spacing: f32,
    extent: [u32; 2],
    irradiance: TextureViewId,
    distance: TextureViewId,
    bounce: TextureViewId,
    outlier: TextureViewId,
    states: BufferId,
    /// What each probe ray met, ping-ponged: the update reads last frame's
    /// while the trace writes this frame's.
    rays: [BufferId; 2],
}

/// The radiance cache.
struct Cache {
    cells: BufferId,
    /// Each cell's light, ping-ponged: one half is read as last frame's while
    /// the other is written.
    radiance: [BufferId; 2],
    list: BufferId,
    args: BufferId,
    /// The final gather's ray each cell it asked for is shaded from.
    hits: BufferId,
}

struct Pipelines {
    trace: PipelineId,
    compact: PipelineId,
    update: PipelineId,
    shade: PipelineId,
}

pub struct GiPass {
    /// `None` on a device that cannot trace, which has the stand-ins alone.
    pipelines: Option<Pipelines>,
    /// The material textures for the two pipelines that read them, the trace
    /// and the shading, kept until they change.
    textures: Option<[BindGroupId; 2]>,
    sampler: SamplerId,
    /// What a frame binds where the volumes are not live: a block that says
    /// so, one probe, and a texel of nothing.
    fallback: GiViews,
    volumes: Option<Volumes>,
    cache: Option<Cache>,
    schedule: Schedule,
    /// This frame's number, from one: a cell's age is counted in these.
    frame: u32,
    rays: u32,
    /// This frame's block, while the volumes are live.
    block: Option<BufferId>,
    /// This frame's block as it was, for next frame's update of its rays.
    traced: Option<GpuGi>,
    /// Last frame's block, uploaded again for this frame's update; `None`
    /// on the first frame the volumes are live, which has no rays to take in.
    previous: Option<BufferId>,
    /// The final gather, which reads and claims the cache.
    pub(super) gather: GatherPass,
}

impl GiPass {
    pub fn new(device: &Device, traces: bool) -> Self {
        let pipelines = traces.then(|| {
            let compute = |shader: ShaderId| {
                device.create_compute_pipeline(&ComputePipelineDesc {
                    shader: shader.into(),
                })
            };
            let over_textures = |shader: ShaderId| {
                device.create_compute_pipeline_over_textures(
                    &ComputePipelineDesc {
                        shader: shader.into(),
                    },
                    TEXTURE_SET,
                )
            };
            Pipelines {
                trace: over_textures(ShaderId::GiTraceComp),
                compact: compute(ShaderId::GiCacheCompactComp),
                update: compute(ShaderId::GiUpdateComp),
                shade: over_textures(ShaderId::GiCacheShadeComp),
            }
        });
        // Linear and clamped, the one level: a tile's border is what keeps
        // a fetch at its edge from blending in the next tile.
        let sampler = device.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(0.0),
        });
        let nothing = device.create_texture_2d(&[0; 4], [1, 1], Format::Rgba8Unorm);
        let fallback = GiViews {
            block: block_buffer(
                device,
                BufferUsage::UNIFORM_BUFFER,
                bytemuck::bytes_of(&GpuGi::zeroed()),
            ),
            probes: block_buffer(device, BufferUsage::STORAGE_BUFFER, &[0; 16]),
            irradiance: nothing,
            distance: nothing,
            sampler,
        };
        Self {
            pipelines,
            textures: None,
            sampler,
            fallback,
            volumes: None,
            cache: None,
            schedule: Schedule::default(),
            frame: 0,
            rays: 0,
            block: None,
            traced: None,
            previous: None,
            gather: GatherPass::new(device),
        }
    }

    /// Bring the volumes up to this frame's settings and camera, and pick the
    /// grid it updates. `live` is whether the frame runs the four passes;
    /// where it does not, everything is let go and the stand-ins are bound.
    /// `gathered` is whether the final gather lights the opaque surfaces.
    /// Before anything records.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_frame(
        &mut self,
        device: &Device,
        settings: &GiSettings,
        live: bool,
        gathered: bool,
        camera: Vec3,
        irradiance: [Vec3; SH9],
        environment_yaw: f32,
    ) {
        if !live || self.pipelines.is_none() {
            self.release(device);
            return;
        }
        let settings = settings.held();
        if self
            .volumes
            .as_ref()
            .is_none_or(|v| v.probes != settings.probes || v.spacing != settings.spacing)
        {
            self.release(device);
            self.volumes = Some(allocate_volumes(device, settings.probes, settings.spacing));
            self.cache = Some(allocate_cache(device));
        }
        let volumes = self.volumes.as_ref().expect("made above");

        let cascades = probes::cascades(camera, volumes.spacing, volumes.probes);
        let update = self.schedule.next(&cascades);
        self.frame = self.frame.wrapping_add(1).max(1);
        self.rays = settings.rays;

        let columns = update.rotation.to_cols_array_2d();
        let sky = sh::radiance(&irradiance);
        let (sin, cos) = environment_yaw.to_radians().sin_cos();
        let block = GpuGi {
            grid: [volumes.probes, settings.rays, 1, update.level as u32],
            frame: [self.frame, LIFETIME, settings.debug_view.mode(), CELLS - 1],
            params: [
                settings.hysteresis,
                FINEST_CELL,
                1.0 / volumes.extent[0] as f32,
                1.0 / volumes.extent[1] as f32,
            ],
            camera: [
                camera.x,
                camera.y,
                camera.z,
                if settings.reflections { 1.0 } else { 0.0 },
            ],
            environment: [
                sin,
                cos,
                if settings.fog { 1.0 } else { 0.0 },
                if gathered { 1.0 } else { 0.0 },
            ],
            rotation: [
                [columns[0][0], columns[0][1], columns[0][2], update.offset],
                [columns[1][0], columns[1][1], columns[1][2], 0.0],
                [columns[2][0], columns[2][1], columns[2][2], 0.0],
            ],
            cascades: cascades
                .map(|c| [c.base.x as f32, c.base.y as f32, c.base.z as f32, c.spacing]),
            sky: sky.map(|c| [c.x, c.y, c.z, 0.0]),
        };
        self.block = Some(device.upload_block(BlockKind::Uniform, bytemuck::bytes_of(&block)));
        self.previous = self
            .traced
            .replace(block)
            .map(|last| device.upload_block(BlockKind::Uniform, bytemuck::bytes_of(&last)));
    }

    /// What a shader that reads the volumes binds this frame: the volumes
    /// where they are live, the stand-ins where they are not.
    pub(super) fn views(&self) -> GiViews {
        match (&self.volumes, self.block) {
            (Some(volumes), Some(block)) => GiViews {
                block,
                probes: volumes.states,
                irradiance: volumes.irradiance,
                distance: volumes.distance,
                sampler: self.sampler,
            },
            _ => self.fallback,
        }
    }

    /// The stand-ins alone, for a draw that stands in no scene.
    pub(super) fn fallback(&self) -> GiViews {
        self.fallback
    }

    /// The four atlases and their extent, for the graph to be handed.
    pub(super) fn atlases(&self) -> Option<([TextureViewId; 4], [u32; 2])> {
        self.volumes
            .as_ref()
            .map(|v| ([v.irradiance, v.distance, v.bounce, v.outlier], v.extent))
    }

    /// Make sure the trace and the shading have the material textures bound,
    /// before a frame that traces records. Kept until the textures change.
    pub(super) fn prepare(
        &mut self,
        device: &Device,
        textures: &[TextureViewId],
        sampler: SamplerId,
    ) {
        if let Some(pipelines) = &self.pipelines
            && self.textures.is_none()
        {
            self.textures = Some([pipelines.trace, pipelines.shade].map(|pipeline| {
                device.create_texture_array_group(pipeline, TEXTURE_SET, textures, sampler)
            }));
        }
    }

    /// Let go of the texture sets, which name textures that have changed.
    pub(super) fn forget_textures(&mut self, device: &Device) {
        for group in self.textures.take().into_iter().flatten() {
            device.destroy_bind_group(group);
        }
        self.gather.forget_textures(device);
    }

    /// The cache as the final gather reads and claims it, this frame.
    pub(super) fn gather_cache(&self) -> GatherCache {
        let (_, _, cache, _) = self.live();
        let (_, radiance) = self.radiance(cache);
        GatherCache {
            cells: cache.cells,
            radiance,
            hits: cache.hits,
        }
    }

    fn release(&mut self, device: &Device) {
        self.block = None;
        self.traced = None;
        self.previous = None;
        if let Some(volumes) = self.volumes.take() {
            device.destroy_texture(volumes.irradiance);
            device.destroy_texture(volumes.distance);
            device.destroy_texture(volumes.bounce);
            device.destroy_texture(volumes.outlier);
            device.destroy_buffer(volumes.states);
            for buffer in volumes.rays {
                device.destroy_buffer(buffer);
            }
        }
        if let Some(cache) = self.cache.take() {
            for buffer in [cache.cells, cache.list, cache.args, cache.hits]
                .into_iter()
                .chain(cache.radiance)
            {
                device.destroy_buffer(buffer);
            }
        }
        // A grid made again starts with no probe in it.
        self.schedule = Schedule::default();
    }

    fn live(&self) -> (&Pipelines, &Volumes, &Cache, BufferId) {
        let message = "the graph scheduled the volumes before begin_frame made them";
        (
            self.pipelines.as_ref().expect(message),
            self.volumes.as_ref().expect(message),
            self.cache.as_ref().expect(message),
            self.block.expect(message),
        )
    }

    /// Last frame's half of the cache's light, and this frame's.
    fn radiance(&self, cache: &Cache) -> (BufferId, BufferId) {
        let this = (self.frame & 1) as usize;
        (cache.radiance[1 - this], cache.radiance[this])
    }

    /// Last frame's rays, and this frame's.
    fn rays(&self, volumes: &Volumes) -> (BufferId, BufferId) {
        let this = (self.frame & 1) as usize;
        (volumes.rays[1 - this], volumes.rays[this])
    }

    fn textures(&self) -> [BindGroupId; 2] {
        self.textures
            .expect("the volumes were recorded before prepare")
    }

    pub(super) fn record_trace(&self, device: &Device, encoder: &mut Encoder, rays: &RayFrame<'_>) {
        let (pipelines, volumes, cache, block) = self.live();
        let (_, traced) = self.rays(volumes);
        let pipeline = pipelines.trace;
        let set = device.create_transient_bind_group(
            pipeline,
            0,
            &[
                storage(0, traced),
                storage(1, cache.cells),
                storage(2, cache.list),
                storage(3, cache.args),
            ],
        );
        // Where each probe stands, and nothing else of the volumes.
        let gi = device.create_transient_bind_group(
            pipeline,
            GI_SET,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::UniformBuffer(block),
                },
                storage(1, volumes.states),
            ],
        );
        encoder.bind_pipeline(pipeline);
        encoder.bind_group(0, set);
        rays.bind(
            device,
            encoder,
            pipeline,
            RayReads {
                lights: false,
                materials: true,
                environment: false,
                surfaces: true,
                gi: false,
            },
        );
        encoder.bind_group(GI_SET, gi);
        encoder.bind_group(TEXTURE_SET, self.textures()[0]);
        let invocations = volumes.probes.pow(3) * self.rays;
        encoder.dispatch([invocations.div_ceil(GROUP), 1, 1]);
    }

    pub(super) fn record_compact(&self, device: &Device, encoder: &mut Encoder) {
        let (pipelines, _, cache, block) = self.live();
        let (history, radiance) = self.radiance(cache);
        let set = device.create_transient_bind_group(
            pipelines.compact,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::UniformBuffer(block),
                },
                storage(1, cache.cells),
                storage(2, history),
                storage(3, radiance),
                storage(4, cache.list),
                storage(5, cache.args),
            ],
        );
        encoder.bind_pipeline(pipelines.compact);
        encoder.bind_group(0, set);
        encoder.dispatch([CELLS / GROUP, 1, 1]);
    }

    /// Last frame's rays taken into the probes it traced them for, by last
    /// frame's block: nothing on the first frame the volumes are live.
    pub(super) fn record_update(&self, device: &Device, encoder: &mut Encoder) {
        let (pipelines, volumes, cache, _) = self.live();
        let Some(block) = self.previous else {
            return;
        };
        let (history, _) = self.radiance(cache);
        let (traced, _) = self.rays(volumes);
        let set = device.create_transient_bind_group(
            pipelines.update,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::UniformBuffer(block),
                },
                storage(1, volumes.states),
                storage(2, traced),
                storage(3, history),
                Binding {
                    binding: 4,
                    resource: BindingResource::StorageTexture(volumes.irradiance),
                },
                Binding {
                    binding: 5,
                    resource: BindingResource::StorageTexture(volumes.distance),
                },
                Binding {
                    binding: 6,
                    resource: BindingResource::StorageTexture(volumes.bounce),
                },
                Binding {
                    binding: 7,
                    resource: BindingResource::StorageTexture(volumes.outlier),
                },
            ],
        );
        encoder.bind_pipeline(pipelines.update);
        encoder.bind_group(0, set);
        encoder.dispatch([volumes.probes.pow(3), 1, 1]);
    }

    pub(super) fn record_shade(&self, device: &Device, encoder: &mut Encoder, rays: &RayFrame<'_>) {
        let (pipelines, volumes, cache, _) = self.live();
        let (_, radiance) = self.radiance(cache);
        let (_, traced) = self.rays(volumes);
        let pipeline = pipelines.shade;
        let set = device.create_transient_bind_group(
            pipeline,
            0,
            &[
                storage(0, cache.cells),
                storage(1, traced),
                storage(2, radiance),
                storage(3, cache.list),
                storage(4, cache.hits),
            ],
        );
        encoder.bind_pipeline(pipeline);
        encoder.bind_group(0, set);
        let rays = RayFrame {
            gi: GiViews {
                irradiance: volumes.bounce,
                ..rays.gi
            },
            ..*rays
        };
        rays.bind(
            device,
            encoder,
            pipeline,
            RayReads {
                environment: false,
                ..RayReads::SHADING
            },
        );
        encoder.bind_group(TEXTURE_SET, self.textures()[1]);
        encoder.dispatch_indirect(cache.args);
    }
}

fn storage(binding: u32, buffer: BufferId) -> Binding {
    Binding {
        binding,
        resource: BindingResource::StorageBuffer(buffer),
    }
}

fn block_buffer(device: &Device, usage: BufferUsage, bytes: &[u8]) -> BufferId {
    device.create_buffer_bytes(
        BufferDesc {
            size: bytes.len() as u64,
            usage,
            location: MemoryLocation::GpuOnly,
        },
        bytes,
    )
}

fn words(device: &Device, usage: BufferUsage, words: &[u32]) -> BufferId {
    device.create_buffer_init(
        BufferDesc {
            size: words.len() as u64 * 4,
            usage,
            location: MemoryLocation::GpuOnly,
        },
        words,
    )
}

/// Every atlas, entered sampled as the frame declares them, and every probe
/// with no state: an update reads a probe's texels only once its state says
/// it has written them.
fn allocate_volumes(device: &Device, probes: u32, spacing: f32) -> Volumes {
    let extent = probes::atlas_extent(probes);
    let atlas = || {
        let view = device.create_texture(&TextureDesc {
            format: ATLAS_FORMAT,
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Sample1,
            usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
            memoryless: false,
        });
        device.initialise_sampled(view);
        view
    };
    let count = probes.pow(3);
    Volumes {
        probes,
        spacing,
        extent,
        irradiance: atlas(),
        distance: atlas(),
        bounce: atlas(),
        outlier: atlas(),
        states: words(
            device,
            BufferUsage::STORAGE_BUFFER,
            &vec![0; (count * CASCADES as u32 * 4) as usize],
        ),
        // A ray is a `vec4` and a `uvec4`.
        rays: [(); 2].map(|_| {
            device.create_buffer(BufferDesc {
                size: u64::from(count * MAX_RAYS) * 32,
                usage: BufferUsage::STORAGE_BUFFER,
                location: MemoryLocation::GpuOnly,
            })
        }),
    }
}

/// Every cell empty, which is a checksum of zero and no ray to shade it from.
fn allocate_cache(device: &Device) -> Cache {
    const NO_RAY: u32 = u32::MAX;
    let cells: Vec<u32> = (0..CELLS).flat_map(|_| [0, 0, NO_RAY, 0]).collect();
    let dark = vec![0; CELLS as usize * 2];
    Cache {
        cells: words(device, BufferUsage::STORAGE_BUFFER, &cells),
        radiance: [
            words(device, BufferUsage::STORAGE_BUFFER, &dark),
            words(device, BufferUsage::STORAGE_BUFFER, &dark),
        ],
        list: device.create_buffer(BufferDesc {
            size: u64::from(CELLS + 1) * 4,
            usage: BufferUsage::STORAGE_BUFFER,
            location: MemoryLocation::GpuOnly,
        }),
        args: words(
            device,
            BufferUsage::STORAGE_BUFFER | BufferUsage::INDIRECT_BUFFER,
            &[0, 1, 1, 0],
        ),
        // A ray is a `vec4` and a `uvec4`, as the probes' are. Read only for
        // a cell the gather has just written its ray for.
        hits: device.create_buffer(BufferDesc {
            size: u64::from(CELLS) * 32,
            usage: BufferUsage::STORAGE_BUFFER,
            location: MemoryLocation::GpuOnly,
        }),
    }
}
