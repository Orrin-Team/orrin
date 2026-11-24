//! The final gather: per-pixel detail on top of the irradiance volumes, which
//! are a metre apart and cannot give a small object its indirect shadow or
//! the colour it bleeds onto what is beside it. id Tech 8's final gather
//! (Sousa 2025), without its sample reuse.
//!
//! # A frame
//!
//! Seven dispatches between the prepass and the forward pass, all but the
//! last at half or a quarter of the frame's resolution each way:
//!
//! 1. `gi_gather`: one ray per texel, cosine-weighted about the prepass's
//!    normal, from a low-discrepancy sequence with a per-pixel scramble
//!    (`low_discrepancy.glsl`). It shades nothing: what it met is read from
//!    last frame's lit colour where the camera sees it, from the world radiance
//!    cache where it does not, from the volumes where the cache has nothing
//!    yet, and from the sky where it met nothing. Each texel keeps what it
//!    found as two bands of spherical harmonics a colour, so it is a small
//!    irradiance probe the full-resolution normal is read through later.
//! 2. `gi_gather_temporal`: the ray added to a running mean of its surface's,
//!    found by the prepass's motion and dropped on a disocclusion, with a
//!    count of the frames in it (Mattausch et al. 2010) capped at
//!    [`GiSettings::gather_history`] where the rays agree and at up to 1024
//!    where they do not, and restarted where a light is switched on. No ray's
//!    light is held back: in a room lit by a patch of sun a few rays meet,
//!    those few are most of its light.
//! 3. and 4. `gi_gather_blur_x` and `_y`: a Gaussian along each axis weighted
//!    by plane, normal and how far the rays went, wider where the history is
//!    short.
//! 5. and 6. `gi_gather_blur_wide_x` and `_y`: the same again over what that
//!    left, its taps [`WIDE_STRIDE`] texels apart (Dammertz et al. 2010).
//!    One level leaves a room lit by a small bright panel blotched by four
//!    parts in a hundred, and the two leave it by a half of one.
//! 7. `gi_gather_upscale`: four texels to each pixel, weighted by plane and
//!    normal (Sloan et al. 2007), into the image the forward pass reads by
//!    pixel and evaluates with its own shading normal.
//!
//! The order is temporal, then spatial: the history keeps what the rays
//! found, unblurred, so the blur's error is not fed back into it. SVGF (Schied
//! et al. 2017), whose second level this already is, adds the variance that
//! steers it and is the step up if this is not enough; NRD and the vendor
//! denoisers are a dependency or a vendor each.
//!
//! # The cache
//!
//! A ray the camera cannot vouch for reads the world radiance cache
//! (`vulkan::gi`), and a cell the probes' rays never reach — under a small
//! object, in an alcove narrower than the probes — would never be shaded. So
//! the gather claims the cells it reads: a new cell, or one whose light is a
//! few frames old (`RESHADE` in `gi_gather.comp`), is listed for next frame's
//! shading and shaded from the gather ray that asked, which the gather
//! leaves in a table of its own beside the cache. Until then the ray reads the
//! volumes at its hit. One frame of that, for a cell new to the view.
//!
//! # What reads it
//!
//! The opaque forward pass, for its diffuse light in place of the volumes'.
//! Blended and refractive surfaces are not the surface the prepass saw at
//! their pixel, and the fog is not a surface: they keep the volumes. A frame
//! without the gather binds a stand-in and the volumes' block says so, so the
//! forward pass is one pipeline either way.
//!
//! # Memory
//!
//! At half resolution and 1080p: the raw rays are four layers of 960x540 half
//! floats, 16.6 MB, and the two accumulations five, 20.7 MB apiece; the two blurs
//! three, 12.4 MB each, which the second level's two take over as each is
//! done with; and the upscaled harmonics three layers at the full
//! extent, 49.8 MB. A quarter of that at a quarter resolution, bar the last.
//! The table of hits beside the cache is 8.4 MB, made with the cache.

use glam::Vec3;

use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::ImageUsage;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, BufferId, ComputePipelineDesc, Device, Encoder,
    PipelineId, SamplerDesc, SamplerId, TextureDesc, TextureDimension, TextureViewId,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::GiSettings;

use super::raytrace::{RayFrame, RayReads, TEXTURE_SET};
use super::taa::FrameView;

/// Every image the gather writes: two bands of harmonics in each of three
/// layers, one per colour, and for the raw rays and the accumulation a fourth
/// beside them for the count, the distances and the deviation.
pub(super) const GATHER_FORMAT: Format = Format::Rgba16Float;
/// The harmonics' layers.
pub(super) const SH_LAYERS: u32 = 3;
/// The harmonics' and the fourth.
pub(super) const META_LAYERS: u32 = 4;
/// The accumulation's: those four and a fifth, for the test of whether a light
/// was switched.
pub(super) const HISTORY_LAYERS: u32 = 5;
/// Side of every gather pass's workgroup.
const TILE: u32 = 8;
/// How many texels apart the blur's second level takes its taps: about the
/// first level's narrowest deviation and a half, past which what the first
/// left would alias.
const WIDE_STRIDE: u32 = 3;

/// `Gather` in `gi_gather.glsl`, as the words of its std140 layout: three
/// matrices, then `camera`, `previous_camera` and `params`.
const BLOCK_WORDS: usize = 3 * 16 + 3 * 4;
const LIT: u32 = 1;
const HISTORY: u32 = 2;

/// The world radiance cache as the gather reads and claims it.
#[derive(Clone, Copy)]
pub(super) struct GatherCache {
    pub cells: BufferId,
    /// This frame's light, as the cache's shading left it.
    pub radiance: BufferId,
    pub hits: BufferId,
}

/// What the gather reads of the prepass.
#[derive(Clone, Copy)]
pub(super) struct Prepass {
    pub depth: TextureViewId,
    pub normal: TextureViewId,
    pub velocity: TextureViewId,
}

struct Pipelines {
    gather: PipelineId,
    temporal: PipelineId,
    blur: PipelineId,
    upscale: PipelineId,
}

/// The two accumulations, at one extent: `frame & 1` is this frame's, the
/// other last frame's.
struct History {
    views: [TextureViewId; 2],
    extent: [u32; 2],
}

pub struct GatherPass {
    /// Made the first frame the gather runs: a session that never asks for
    /// it pays nothing at start-up.
    pipelines: Option<Pipelines>,
    /// The material textures, for the gather's ray when the cache has nothing.
    textures: Option<BindGroupId>,
    history: Option<History>,
    frame: u32,
    /// Where last frame's camera stood, for what a still surface's distance
    /// from it was.
    previous_camera: Option<Vec3>,
    /// Whether the accumulation holds a frame to build on.
    trusted: bool,
    block: Option<BufferId>,
    linear: SamplerId,
    /// What the forward pass binds where the gather did not run, and the
    /// gather binds for last frame's colour where there is none.
    stand_in: TextureViewId,
    unlit: TextureViewId,
}

impl GatherPass {
    pub(super) fn new(device: &Device) -> Self {
        let linear = device.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(0.0),
        });
        let stand_in = device.create_texture(&TextureDesc {
            format: GATHER_FORMAT,
            dimension: TextureDimension::D2Array,
            extent: [1, 1, 1],
            mip_levels: 1,
            array_layers: SH_LAYERS,
            samples: SampleCount::Sample1,
            usage: ImageUsage::SAMPLED,
            memoryless: false,
        });
        device.initialise_sampled(stand_in);
        let unlit = device.create_texture_2d(&[0; 4], [1, 1], Format::Rgba8Unorm);
        Self {
            pipelines: None,
            textures: None,
            history: None,
            frame: 0,
            previous_camera: None,
            trusted: false,
            block: None,
            linear,
            stand_in,
            unlit,
        }
    }

    /// Ready this frame's gather, at `1 >> shift` of `extent` each way, or let
    /// its history go where `shift` is `None`. `lit` is whether last frame's
    /// lit colour can be read. Before anything records. Returns the two
    /// accumulations, last frame's then this frame's, for the graph to be
    /// handed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_frame(
        &mut self,
        device: &Device,
        settings: &GiSettings,
        shift: Option<u32>,
        view: &FrameView,
        camera: Vec3,
        extent: [u32; 2],
        lit: bool,
    ) -> Option<([TextureViewId; 2], [u32; 2])> {
        let previous_camera = self.previous_camera.replace(camera);
        let Some(shift) = shift else {
            self.release(device);
            return None;
        };
        if self.pipelines.is_none() {
            self.pipelines = Some(Pipelines {
                gather: device.create_compute_pipeline_over_textures(
                    &ComputePipelineDesc {
                        shader: ShaderId::GiGatherComp.into(),
                    },
                    TEXTURE_SET,
                ),
                temporal: compute(device, ShaderId::GiGatherTemporalComp),
                blur: compute(device, ShaderId::GiGatherBlurComp),
                upscale: compute(device, ShaderId::GiGatherUpscaleComp),
            });
        }
        let low = [(extent[0] >> shift).max(1), (extent[1] >> shift).max(1)];
        if self.history.as_ref().is_none_or(|h| h.extent != low) {
            self.release(device);
            self.history = Some(History {
                views: [allocate(device, low), allocate(device, low)],
                extent: low,
            });
        }
        let trusted = std::mem::replace(&mut self.trusted, true);
        self.frame = self.frame.wrapping_add(1);

        let settings = settings.held();
        let matrices = [
            view.view_proj,
            view.view_proj.inverse(),
            view.view.inverse(),
        ];
        let previous = previous_camera.unwrap_or(camera);
        let mut words = [0u32; BLOCK_WORDS];
        let floats = matrices.iter().flat_map(|m| m.to_cols_array()).chain([
            camera.x,
            camera.y,
            camera.z,
            settings.gather_history as f32,
            previous.x,
            previous.y,
            previous.z,
            0.0,
        ]);
        for (word, float) in words.iter_mut().zip(floats) {
            *word = float.to_bits();
        }
        let history_flag = if trusted && previous_camera.is_some() {
            HISTORY
        } else {
            0
        };
        words[BLOCK_WORDS - 4..].copy_from_slice(&[
            shift,
            self.frame,
            if lit { LIT } else { 0 } | history_flag,
            0,
        ]);
        self.block = Some(device.upload_uniform(&words));

        let history = self.history.as_ref().expect("made above");
        let this = (self.frame & 1) as usize;
        Some((
            [history.views[1 - this], history.views[this]],
            history.extent,
        ))
    }

    /// What the forward pass binds where no gather ran this frame.
    pub(super) fn stand_in(&self) -> TextureViewId {
        self.stand_in
    }

    /// Make sure the gather has the material textures bound. Kept until the
    /// textures change.
    pub(super) fn prepare(
        &mut self,
        device: &Device,
        textures: &[TextureViewId],
        sampler: SamplerId,
    ) {
        if let Some(pipelines) = &self.pipelines
            && self.textures.is_none()
        {
            self.textures = Some(device.create_texture_array_group(
                pipelines.gather,
                TEXTURE_SET,
                textures,
                sampler,
            ));
        }
    }

    pub(super) fn forget_textures(&mut self, device: &Device) {
        if let Some(group) = self.textures.take() {
            device.destroy_bind_group(group);
        }
    }

    fn release(&mut self, device: &Device) {
        if let Some(history) = self.history.take() {
            for view in history.views {
                device.destroy_texture(view);
            }
        }
        self.trusted = false;
    }

    fn live(&self) -> (&Pipelines, BufferId) {
        let message = "the graph scheduled the gather before begin_frame readied it";
        (
            self.pipelines.as_ref().expect(message),
            self.block.expect(message),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_gather(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        rays: &RayFrame<'_>,
        prepass: Prepass,
        lit: Option<TextureViewId>,
        cache: GatherCache,
        raw: TextureViewId,
    ) {
        let (pipelines, block) = self.live();
        let pipeline = pipelines.gather;
        let set = device.create_transient_bind_group(
            pipeline,
            0,
            &[
                texture(0, prepass.depth),
                texture(1, prepass.normal),
                texture(2, prepass.velocity),
                Binding {
                    binding: 3,
                    resource: BindingResource::SampledTexture {
                        view: lit.unwrap_or(self.unlit),
                        sampler: self.linear,
                    },
                },
                storage_texture(4, raw),
                uniform(5, block),
                storage(6, cache.cells),
                storage(7, cache.radiance),
                storage(8, cache.hits),
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
                gi: true,
            },
        );
        encoder.bind_group(
            TEXTURE_SET,
            self.textures
                .expect("the gather was recorded before prepare"),
        );
        dispatch_over(device, encoder, raw);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_temporal(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        prepass: Prepass,
        raw: TextureViewId,
        history: TextureViewId,
        accumulated: TextureViewId,
    ) {
        let (pipelines, block) = self.live();
        let set = device.create_transient_bind_group(
            pipelines.temporal,
            0,
            &[
                texture(0, prepass.depth),
                texture(1, prepass.normal),
                texture(2, prepass.velocity),
                texture(3, raw),
                texture(4, history),
                storage_texture(5, accumulated),
                uniform(6, block),
            ],
        );
        encoder.bind_pipeline(pipelines.temporal);
        encoder.bind_group(0, set);
        dispatch_over(device, encoder, accumulated);
    }

    /// One axis of one level of the blur: across where `vertical` is false,
    /// and the second level, its taps `WIDE_STRIDE` apart, where `wide`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_blur(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        prepass: Prepass,
        source: TextureViewId,
        meta: TextureViewId,
        target: TextureViewId,
        vertical: bool,
        wide: bool,
    ) {
        let (pipelines, block) = self.live();
        let set = device.create_transient_bind_group(
            pipelines.blur,
            0,
            &[
                texture(0, prepass.depth),
                texture(1, prepass.normal),
                texture(2, source),
                texture(3, meta),
                storage_texture(4, target),
                uniform(5, block),
            ],
        );
        encoder.bind_pipeline(pipelines.blur);
        encoder.bind_group(0, set);
        let stride = if wide { WIDE_STRIDE } else { 1 };
        encoder.push_constants(&[u32::from(!vertical) * stride, u32::from(vertical) * stride]);
        dispatch_over(device, encoder, target);
    }

    pub(super) fn record_upscale(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        prepass: Prepass,
        gathered: TextureViewId,
        target: TextureViewId,
    ) {
        let (pipelines, block) = self.live();
        let set = device.create_transient_bind_group(
            pipelines.upscale,
            0,
            &[
                texture(0, prepass.depth),
                texture(1, prepass.normal),
                texture(2, gathered),
                storage_texture(3, target),
                uniform(4, block),
            ],
        );
        encoder.bind_pipeline(pipelines.upscale);
        encoder.bind_group(0, set);
        dispatch_over(device, encoder, target);
    }
}

fn compute(device: &Device, shader: ShaderId) -> PipelineId {
    device.create_compute_pipeline(&ComputePipelineDesc {
        shader: shader.into(),
    })
}

/// One accumulation, entered sampled as the frame declares it.
fn allocate(device: &Device, extent: [u32; 2]) -> TextureViewId {
    let view = device.create_texture(&TextureDesc {
        format: GATHER_FORMAT,
        dimension: TextureDimension::D2Array,
        extent: [extent[0], extent[1], 1],
        mip_levels: 1,
        array_layers: HISTORY_LAYERS,
        samples: SampleCount::Sample1,
        usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
        memoryless: false,
    });
    device.initialise_sampled(view);
    view
}

fn dispatch_over(device: &Device, encoder: &mut Encoder, target: TextureViewId) {
    let extent = device.view_extent(target);
    encoder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
}

fn texture(binding: u32, view: TextureViewId) -> Binding {
    Binding {
        binding,
        resource: BindingResource::Texture(view),
    }
}

fn storage_texture(binding: u32, view: TextureViewId) -> Binding {
    Binding {
        binding,
        resource: BindingResource::StorageTexture(view),
    }
}

fn storage(binding: u32, buffer: BufferId) -> Binding {
    Binding {
        binding,
        resource: BindingResource::StorageBuffer(buffer),
    }
}

fn uniform(binding: u32, buffer: BufferId) -> Binding {
    Binding {
        binding,
        resource: BindingResource::UniformBuffer(buffer),
    }
}
