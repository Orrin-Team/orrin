//! Froxel volumetric fog: the air as a grid of cells in front of the camera,
//! each one asked what it scatters and what it blocks.
//!
//! Two dispatches and no full-resolution pass, which is the whole reason the
//! effect fits. `fog_scatter` lights every froxel — a cascade lookup apiece, and
//! the only reason a sunbeam through a window has an edge — and `fog_integrate`
//! marches each column into the running (radiance, transmittance) pair a
//! fragment can apply in one texture fetch. The application itself is a few lines
//! of `shading.glsl` and `skybox.frag`, not a pass: a composite over the frame
//! would move more bytes than the entire simulation above it.
//!
//! The grid is [`FROXEL_SHIFT`]-reduced in screen space and [`FROXEL_SLICES`]
//! deep, so at 1440p it is 160x90x64 — about 920k cells and 7.4 MB a volume.
//! Two volumes and a history is roughly 30 MB of traffic, against the ~60 MB a
//! single full-resolution pass moves. That ratio is the argument for the whole
//! design.
//!
//! The history is **imported**, for the reason the TAA history and the exposure
//! buffer are: a transient is `Undefined` at every frame's start by contract,
//! and a volume that survives a frame boundary is precisely what the depth
//! jitter needs to be averaged by. Two allocations, ping-ponged, so the scatter
//! pass never reads the volume it is writing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use glam::{Mat4, Vec3};
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::gfx::SceneLighting;
use crate::gfx::rhi::{
    Binding, BindingResource, ComputePipelineDesc, Device, Encoder, PipelineId, SamplerDesc,
    SamplerId, TextureViewId,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::FogSettings;

use super::ShadowFrame;
use super::context::VkContext;
use super::convert::vk_format;
use super::forward::GpuCascades;
use super::record::Arena;
use super::taa::FrameView;

/// How far the froxel grid is reduced from the frame in each screen axis.
///
/// Four — one froxel per 16x16 pixels — and that is a budget decision rather
/// than a quality ceiling. Frostbite's 8x8 at 1080p would be 320x180x64 here,
/// four times the cells, and this renderer's whole frame target is 0.5-0.66 ms.
/// The volume is trilinearly filtered and temporally accumulated, so what a
/// coarse grid costs is the sharpness of a shaft's edge, not its presence.
pub(super) const FROXEL_SHIFT: u32 = 4;

/// Depth slices. Keep in sync with `FOG_SLICES` in `shaders/fog.glsl` — the
/// integration walks exactly this many, and the shader sizes its loop off the
/// image rather than off the constant so a mismatch would silently shorten the
/// volume rather than read past it.
pub(super) const FROXEL_SLICES: u32 = 64;

/// `rgb` = in-scattered radiance, `a` = transmittance, in the integrated volume;
/// per-metre scattering and extinction in the scatter volume that feeds it.
///
/// Sixteen-bit float for the reason the frame's colour is: this is photometric
/// radiance, so a sunlit shaft is in the thousands of cd/m² while the shadowed
/// air beside it is in the ones, and a normalised format would band across
/// exactly the gradient the effect exists to draw.
pub(super) const FOG_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::Rgba16Float;

/// Side of the scatter dispatch's workgroup, matching `fog_scatter.comp`.
const SCATTER_TILE: u32 = 4;

/// Side of the integration's workgroup, matching `fog_integrate.comp`. Two
/// dimensions, because that pass runs one invocation per froxel *column*.
const INTEGRATE_TILE: u32 = 8;

/// Length of the depth-jitter sequence, for the reason [`JITTER_PHASES`] governs
/// TAA's: long enough to cover a slice evenly, short enough that the history
/// still remembers the start of it.
const JITTER_PHASES: u64 = 8;

/// Where `fog_scatter.comp` declares the cascade comparison sampler. Bound
/// immutably at pipeline-layout construction, so it has to match the shader by
/// hand rather than being derived from it — the same hand-kept pairing
/// `SHADOW_SAMPLER_BINDING` is for the forward pass.
const SHADOW_SAMPLER_BINDING: u32 = 2;

/// The uniform block both dispatches read, mirroring `GpuFog` in
/// `shaders/fog.glsl`. std140 packs this exactly like the `#[repr(C)]` here
/// because every field is a `mat4` or a `vec4`.
///
/// One block, bound in three places — both dispatches, and set 3 of every shader
/// that applies the result. That is the decal precedent: a medium described in
/// two buffers is a medium two passes can disagree about, and the disagreement
/// would show up as fog that does not match the air it is standing in.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct GpuFog {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    /// rgb = single-scattering albedo, w = extinction at the reference height.
    albedo: [f32; 4],
    /// x = height falloff, y = reference height, z = far distance,
    /// w = anisotropy.
    params: [f32; 4],
    sun_direction: [f32; 4],
    /// rgb = sun colour, w = illuminance in lux.
    sun_color: [f32; 4],
    /// rgb = the ambient luminance the medium sees, cd/m².
    ambient: [f32; 4],
    /// xyz = camera position, w = 1.0 when the froxel volume is live.
    camera_pos: [f32; 4],
    prev_camera_pos: [f32; 4],
    /// x = history feedback, y = depth jitter, z = 1.0 to drop the history.
    temporal: [f32; 4],
    cascades: GpuCascades,
}

pub struct FogPass {
    scatter_pipeline: PipelineId,
    integrate_pipeline: PipelineId,
    /// Linear and clamped: the reprojection lands between froxels in all three
    /// axes, and the clamp is what makes a froxel that has just entered the
    /// frustum fall back to its own measurement rather than a neighbour's.
    linear_clamp: SamplerId,
    /// Nearest, for the integration: it reads exactly the froxel it is
    /// integrating, and a bilinear tap would smear a neighbouring column's
    /// extinction into a transmittance about to be multiplied along a whole ray.
    nearest_clamp: SamplerId,
    /// The same linear clamped sampler, as the shading passes bind it beside the
    /// volume they apply the fog from.
    shading_sampler: Arc<Sampler>,
    uniform_allocator: Arena,
    /// Ping-ponged: `frame & 1` is this frame's scatter target and the other is
    /// the history. Empty until the first frame the effect is enabled for.
    history: Option<[Arc<ImageView>; 2]>,
    /// A 1x1x1 volume of "no in-scatter, full transmittance", bound wherever the
    /// real one is absent so every shader has one path rather than two — the
    /// same trick the 1x1 white ambient-occlusion view plays.
    fallback: Arc<ImageView>,
    extent: [u32; 3],
    frame: u64,
    /// Set whenever the history cannot be trusted — first frame, a resize, or
    /// the effect having been off. The scatter pass then keeps its own
    /// measurement, which is one jittered frame instead of a frame of somewhere
    /// else's air.
    ///
    /// Atomic for the reason `TaaPass::reset` is: the scatter clears it, and
    /// the scatter may be recording on a worker.
    reset: AtomicBool,
    previous_view_proj: Option<Mat4>,
    previous_camera: Vec3,
    /// This frame's block, resolved once in `begin_frame`. Both dispatches and
    /// every shader that applies the fog bind the same allocation, so the medium
    /// cannot differ between the pass that lit it and the pass that reads it.
    uniforms: Option<Subbuffer<GpuFog>>,
}

impl FogPass {
    pub fn new(ctx: &VkContext, device: &Device) -> Self {
        // The comparison sampler has to be *immutable* — part of the layout
        // rather than something written into a descriptor set — because MoltenVK
        // is a portability-subset device and rejects a written one outright. The
        // forward pass patches its own set 3 for exactly this reason; this is the
        // same constraint reached from a compute pipeline.
        let shadow_sampler = device.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: false,
            repeat: false,
            compare: true,
            anisotropy: None,
            max_lod: None,
        });
        let scatter_pipeline = device.create_compute_pipeline_with_samplers(
            &ComputePipelineDesc {
                shader: ShaderId::FogScatterComp,
            },
            &[(0, SHADOW_SAMPLER_BINDING, shadow_sampler)],
        );
        let integrate_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            shader: ShaderId::FogIntegrateComp,
        });

        let sampler = |linear: bool| {
            device.create_sampler(SamplerDesc {
                linear,
                mipmap_linear: false,
                repeat: false,
                compare: false,
                anisotropy: None,
                max_lod: Some(if linear { 1.0 } else { 0.0 }),
            })
        };
        let linear_clamp = sampler(true);
        let nearest_clamp = sampler(false);

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
            scatter_pipeline,
            integrate_pipeline,
            linear_clamp,
            nearest_clamp,
            shading_sampler: Sampler::new(
                ctx.device.clone(),
                SamplerCreateInfo {
                    address_mode: [SamplerAddressMode::ClampToEdge; 3],
                    ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
                },
            )
            .unwrap(),
            uniform_allocator,
            history: None,
            fallback: allocate_fallback(ctx),
            extent: [0; 3],
            frame: 0,
            reset: AtomicBool::new(true),
            previous_view_proj: None,
            previous_camera: Vec3::ZERO,
            uniforms: None,
        }
    }

    /// The froxel grid a frame of this extent gets. Also what the graph declares,
    /// so the dispatch and the allocation cannot disagree.
    pub(super) fn froxel_extent(extent: [u32; 2]) -> [u32; 3] {
        [
            (extent[0] >> FROXEL_SHIFT).max(1),
            (extent[1] >> FROXEL_SHIFT).max(1),
            FROXEL_SLICES,
        ]
    }

    /// Resolve this frame's medium into the block everything binds, and advance
    /// the jitter.
    ///
    /// Called before anything records, like the other per-frame `begin_frame`s,
    /// and called whether or not the effect is on: the shading path binds this
    /// block either way, because the analytic fog past the volume's far plane is
    /// described by the very same numbers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_frame(
        &mut self,
        ctx: &VkContext,
        settings: &FogSettings,
        lighting: &SceneLighting,
        shadows: Option<ShadowFrame<'_>>,
        view: &FrameView,
        camera_pos: Vec3,
        ambient: Vec3,
        extent: [u32; 2],
    ) {
        let live = self.enabled(settings);
        let froxels = Self::froxel_extent(extent);

        if live {
            if self.history.is_none() || self.extent != froxels {
                self.history = Some(allocate_history(ctx, froxels));
                self.extent = froxels;
                self.reset.store(true, Ordering::Relaxed);
            }
            self.frame = self.frame.wrapping_add(1);
        } else {
            // Freed rather than kept: two volumes is real memory, and what they
            // hold is stale the moment a frame renders without them.
            self.history = None;
            self.reset.store(true, Ordering::Relaxed);
        }

        // The shader wants the direction *toward* the sun, matching the lighting
        // block's own convention.
        let to_sun = (-lighting.sun.direction).normalize_or_zero();
        let previous_view_proj = self.previous_view_proj.unwrap_or(view.unjittered_view_proj);

        let uniforms = self.uniform_allocator.allocate_sized::<GpuFog>();
        *uniforms.write().unwrap() = GpuFog {
            // Unjittered, both of them: the volume is reprojected against its own
            // history rather than resolved by TAA, so the raster's subpixel
            // jitter would fight the reprojection for a sixteenth of a froxel.
            inv_view_proj: view.unjittered_view_proj.inverse().to_cols_array_2d(),
            prev_view_proj: previous_view_proj.to_cols_array_2d(),
            albedo: [
                settings.albedo.x,
                settings.albedo.y,
                settings.albedo.z,
                settings.density.max(0.0),
            ],
            params: [
                settings.height_falloff,
                settings.height,
                settings.distance.max(1.0),
                settings.anisotropy.clamp(-0.95, 0.95),
            ],
            sun_direction: [to_sun.x, to_sun.y, to_sun.z, 0.0],
            sun_color: [
                lighting.sun.color.x,
                lighting.sun.color.y,
                lighting.sun.color.z,
                lighting.sun.illuminance,
            ],
            ambient: [ambient.x, ambient.y, ambient.z, 0.0],
            camera_pos: [
                camera_pos.x,
                camera_pos.y,
                camera_pos.z,
                if live { 1.0 } else { 0.0 },
            ],
            prev_camera_pos: [
                self.previous_camera.x,
                self.previous_camera.y,
                self.previous_camera.z,
                0.0,
            ],
            temporal: [
                settings.feedback.clamp(0.0, 0.98),
                jitter(self.frame),
                self.reset.load(Ordering::Relaxed) as u32 as f32,
                0.0,
            ],
            // The same expression `to_gpu_lighting` fills its own copy from, so
            // the froxel that reads a cascade and the surface it lands on cannot
            // be looking at different matrices.
            cascades: GpuCascades::new(shadows),
        };
        self.uniforms = Some(uniforms);

        // Tracked whether or not the effect is on, for the reason TAA tracks its
        // own: otherwise the first frame after it is switched back on reprojects
        // against wherever the camera was when it was switched off.
        self.previous_view_proj = Some(view.unjittered_view_proj);
        self.previous_camera = camera_pos;
    }

    /// Whether the frame runs the two dispatches. Density zero is the off switch
    /// as much as the checkbox is — with no medium there is nothing to march,
    /// and the analytic path returns the same nothing for free.
    pub(super) fn enabled(&self, settings: &FogSettings) -> bool {
        settings.volumetric && settings.density > 0.0
    }

    pub(super) fn uniforms(&self) -> Subbuffer<GpuFog> {
        self.uniforms
            .clone()
            .expect("the frame bound the fog block before begin_frame")
    }

    /// What shaders sample to apply the fog: the integrated volume where the
    /// passes ran, and the inert 1x1x1 stand-in where they did not.
    pub(super) fn volume_or_fallback(&self, volume: Option<Arc<ImageView>>) -> Arc<ImageView> {
        volume.unwrap_or_else(|| self.fallback.clone())
    }

    pub(super) fn sampler(&self) -> Arc<Sampler> {
        self.shading_sampler.clone()
    }

    /// This frame's scatter target, which is next frame's history.
    pub(super) fn scatter_view(&self) -> Arc<ImageView> {
        self.pair()[(self.frame & 1) as usize].clone()
    }

    fn history_view(&self) -> Arc<ImageView> {
        self.pair()[((self.frame + 1) & 1) as usize].clone()
    }

    fn pair(&self) -> &[Arc<ImageView>; 2] {
        self.history
            .as_ref()
            .expect("the graph scheduled the fog scatter with no history allocated")
    }

    pub(super) fn record_scatter(
        &self,
        device: &Device,
        builder: &mut Encoder,
        shadow_maps: TextureViewId,
    ) {
        let Device::Vulkan(driver) = device;
        let target = self.scatter_view();
        let extent = target.image().extent();

        let set = device.create_transient_bind_group(
            self.scatter_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::UniformBuffer(
                        driver.import_buffer(self.uniforms().into_bytes()),
                    ),
                },
                // No write for binding 2: the comparison sampler is immutable in
                // this layout, which is what a portability-subset device requires
                // and what writing one would be rejected for.
                Binding {
                    binding: 1,
                    resource: BindingResource::Texture(shadow_maps),
                },
                Binding {
                    binding: 3,
                    resource: BindingResource::SampledTexture {
                        view: driver.import_view(self.history_view()),
                        sampler: self.linear_clamp,
                    },
                },
                Binding {
                    binding: 4,
                    resource: BindingResource::StorageTexture(driver.import_view(target)),
                },
            ],
        );

        builder.bind_pipeline(self.scatter_pipeline);
        builder.bind_group(0, set);

        // The dispatch covers exactly the volume's extent and the shader discards
        // invocations past `imageSize`, so nothing writes outside it.
        builder.dispatch([
            extent[0].div_ceil(SCATTER_TILE),
            extent[1].div_ceil(SCATTER_TILE),
            extent[2].div_ceil(SCATTER_TILE),
        ]);

        // The volume that just scattered is the history the next frame reads, so
        // whatever made it untrustworthy is over.
        self.reset.store(false, Ordering::Relaxed);
    }

    pub(super) fn record_integrate(
        &self,
        device: &Device,
        builder: &mut Encoder,
        target: TextureViewId,
    ) {
        let Device::Vulkan(driver) = device;
        let extent = device.view_extent(target);
        let extent = [extent[0], extent[1]];

        let set = device.create_transient_bind_group(
            self.integrate_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::UniformBuffer(
                        driver.import_buffer(self.uniforms().into_bytes()),
                    ),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::SampledTexture {
                        view: driver.import_view(self.scatter_view()),
                        sampler: self.nearest_clamp,
                    },
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::StorageTexture(target),
                },
            ],
        );

        builder.bind_pipeline(self.integrate_pipeline);
        builder.bind_group(0, set);

        // One invocation per froxel *column* — the shader loops the depth axis
        // itself — so the dispatch covers the volume's `xy` only, and the shader
        // discards columns past `imageSize`.
        builder.dispatch([
            extent[0].div_ceil(INTEGRATE_TILE),
            extent[1].div_ceil(INTEGRATE_TILE),
            1,
        ]);
    }
}

/// Where inside its slice frame `index` samples, in `[0, 1)`.
///
/// Halton base 2 rather than a fixed ladder, for the reason the frame's own
/// jitter is Halton: the sequence is low-discrepancy at *every* prefix length, so
/// a history only a few frames deep — all a moving camera keeps — is still evenly
/// spread through the slice.
fn jitter(index: u64) -> f32 {
    // From one, because Halton's zeroth element is zero and a frame sampling the
    // slice's front face contributes nothing an unjittered volume would not.
    let mut i = (index % JITTER_PHASES) as u32 + 1;
    let mut fraction = 1.0f32;
    let mut result = 0.0f32;
    while i > 0 {
        fraction /= 2.0;
        result += fraction * (i % 2) as f32;
        i /= 2;
    }
    result
}

fn allocate_history(ctx: &VkContext, extent: [u32; 3]) -> [Arc<ImageView>; 2] {
    std::array::from_fn(|_| {
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim3d,
                format: vk_format(FOG_FORMAT),
                extent,
                // Written as a storage image by the scatter pass, read as a
                // sampled one by the integration this frame and by the scatter
                // pass's next frame.
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("failed to allocate the fog scattering history");
        ImageView::new_default(image).unwrap()
    })
}

/// A volume that says "nothing scattered, nothing absorbed" everywhere.
///
/// Never written, and it does not need to be: the allocation is zeroed, which is
/// zero in-scattered radiance and — in the alpha the shader reads as
/// transmittance — zero as well. So the fallback would *hide* the frame rather
/// than leave it alone, which is why `fog_term` gates the fetch on the block's
/// own "volume is live" flag and this exists only to satisfy the binding.
fn allocate_fallback(ctx: &VkContext) -> Arc<ImageView> {
    let image = Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim3d,
            format: vk_format(FOG_FORMAT),
            extent: [1, 1, 1],
            usage: ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("failed to allocate the fog fallback volume");
    ImageView::new_default(image).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grid has to shrink with the frame and never reach zero, or a window
    /// dragged to a sliver asks Vulkan for a zero-extent image.
    #[test]
    fn the_froxel_grid_follows_the_frame_and_stays_allocatable() {
        assert_eq!(FogPass::froxel_extent([2560, 1440]), [160, 90, 64]);
        assert_eq!(FogPass::froxel_extent([1920, 1080]), [120, 67, 64]);
        assert_eq!(FogPass::froxel_extent([4, 1]), [1, 1, 64]);
    }

    /// Every offset has to land inside its slice, or a froxel is measuring the
    /// next one along and the history averages two different depths together.
    #[test]
    fn the_depth_jitter_stays_within_one_slice() {
        for frame in 0..64u64 {
            let offset = jitter(frame);
            assert!(
                (0.0..1.0).contains(&offset),
                "frame {frame} jittered {offset} outside its slice",
            );
        }
    }

    /// The point of a sequence is that consecutive frames sample different
    /// depths; a constant offset would just shift the whole volume forward.
    #[test]
    fn consecutive_frames_sample_different_depths() {
        for frame in 0..JITTER_PHASES {
            assert_ne!(
                jitter(frame),
                jitter(frame + 1),
                "frame {frame} and the next sampled the same depth",
            );
        }
    }
}
