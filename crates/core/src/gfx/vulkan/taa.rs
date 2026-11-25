//! Temporal antialiasing: the camera's subpixel jitter, and the resolve that
//! turns it back into a stable image.
//!
//! Two things live here that do not look alike. The jitter is a property of the
//! *whole frame* — every pass that rasterises geometry must use the same
//! projection or they disagree by a subpixel — so this module produces the
//! [`FrameView`] the frame is drawn with rather than each pass deriving its own
//! matrices. The resolve is an ordinary compute node.
//!
//! The history images are **imported** into the graph, for the reason the
//! exposure buffer is: a transient is `Undefined` at every frame's start by
//! contract, and a history that survives a frame boundary is precisely what this
//! needs. Two allocations, ping-ponged, so the resolve never reads the image it
//! is writing.
//!
//! **The resolve is also the upscaler.** With an [`Upscaling`] mode on, the
//! scene is rendered at the render extent and the history is at the output's;
//! the jitter then walks a longer sequence in render pixels, and the resolve
//! reconstructs each output pixel from the samples round it before it
//! accumulates, keeping in the history's alpha how much it has accumulated
//! (see `taa_resolve.comp`). At native the resolve is the one it always was,
//! down to its format and its shader, which is a second pipeline rather than
//! a branch so that it stays so. The history is dropped on a resize, on a
//! change of mode, and when [`TaaPass::reset`] is asked for at a camera cut.
//!
//! **Where a vendor upscaler goes.** It replaces the resolve node and nothing
//! else, and what it needs at that point is all here: the colour the resolve
//! reads (linear HDR at the render extent), the prepass depth (reversed, a
//! finite far plane) and motion (`now - before` in UV, unjittered, at the
//! render extent), [`FrameView::jitter_pixels`], the reset flag, and the
//! output target. Exposure is the one input the resolve does not read: the
//! frame meters after it, and `upscaler` says why a vendor's node cannot
//! simply declare a read of last frame's adapted value. While a vendor
//! resolves, the jitter and the reset flag here are all of this module the
//! frame uses: no history is allocated, since the vendor keeps its own.

use std::sync::atomic::{AtomicBool, Ordering};

use glam::{Mat4, Vec2};

use crate::gfx::format::Format;
use crate::gfx::format::SampleCount;
use crate::gfx::graph::Extents;
use crate::gfx::graph::ImageUsage;
use crate::gfx::rhi::{
    Binding, BindingResource, ComputePipelineDesc, Device, Encoder, PipelineId, SamplerDesc,
    SamplerId, TextureDesc, TextureDimension, TextureViewId,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::{Camera, TaaSettings, Upscaling};

use super::hdr::{HDR_FORMAT, HDR_WIDE_FORMAT};

/// Side of the resolve's compute workgroup.
const TILE: u32 = 8;

/// What the history is kept in: the packed format at native, and with an
/// upscaling mode the one with an alpha, which holds how much each pixel has
/// accumulated.
pub(super) fn history_format(upscaling: Upscaling) -> Format {
    match upscaling {
        Upscaling::Native => HDR_FORMAT,
        _ => HDR_WIDE_FORMAT,
    }
}

/// The camera matrices the whole frame is drawn with.
///
/// One struct rather than each pass calling `Camera::view_projection` because
/// the jitter has to be identical across the prepass, the forward pass, the
/// skybox and the debug lines. Two of them disagreeing by a subpixel is not a
/// subtle artifact — the sky would sit still while the geometry shook against
/// it.
#[derive(Clone, Copy)]
pub(super) struct FrameView {
    pub view: Mat4,
    /// Jittered when TAA is on, and the plain projection when it is off.
    pub proj: Mat4,
    pub view_proj: Mat4,
    /// Unjittered, and what `prev_view_proj` is compared against: motion vectors
    /// must not carry the jitter, or the resolve would reproject away the very
    /// offsets it exists to accumulate.
    pub unjittered_view_proj: Mat4,
    /// Last frame's `unjittered_view_proj`; equal to this frame's on the first.
    pub prev_view_proj: Mat4,
    /// The NDC offset baked into `proj`, so a shader can remove it.
    pub jitter: Vec2,
    /// The same offset in render pixels: where inside its pixel this frame
    /// sampled, which is the jitter as an upscaler takes it.
    pub jitter_pixels: Vec2,
}

pub struct TaaPass {
    pipeline: PipelineId,
    /// The resolve that also upscales, for a frame with an upscaling mode.
    upscale_pipeline: PipelineId,
    /// Linear and clamped: the history is resampled at a reprojected coordinate
    /// that almost never lands on a texel centre, and the bicubic fetch in the
    /// shader is built out of bilinear taps.
    linear_clamp: SamplerId,
    /// Nearest, because velocity and depth are fetched at exact texels and
    /// interpolating either across a silhouette invents a surface.
    nearest_clamp: SamplerId,
    /// Ping-ponged: `frame & 1` is this frame's target and the other is the
    /// history. Empty until the first frame TAA is enabled for.
    history: Option<[TextureViewId; 2]>,
    extent: [u32; 2],
    /// What the history was accumulated under: a history of another mode is
    /// of other samples, and may not even be of the same format.
    upscaling: Upscaling,
    frame: u64,
    /// Set whenever the history cannot be trusted — first frame, a resize, or
    /// TAA having been off. The resolve then passes the current frame straight
    /// through, which is one aliased frame instead of a frame of garbage.
    ///
    /// Atomic because the resolve clears it, and the resolve may be recording
    /// on a worker: the pass that trusts the history is the one that knows it
    /// can be trusted again, and moving the clear onto the main thread would
    /// mean tracking there whether the pass ran at all.
    reset: AtomicBool,
    previous_view_proj: Option<Mat4>,
    settings: TaaSettings,
}

impl TaaPass {
    pub fn new(device: &Device) -> Self {
        let pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            shader: ShaderId::TaaResolveComp.into(),
        });
        let upscale_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            shader: ShaderId::TaaResolveCompUpscale.into(),
        });
        let sampler = |linear: bool| {
            device.create_sampler(SamplerDesc {
                linear,
                mipmap_linear: false,
                repeat: false,
                compare: false,
                anisotropy: None,
                max_lod: Some(if linear { 1.0 } else { 0.0 }),
                lod_bias: 0.0,
            })
        };
        let linear_clamp = sampler(true);
        let nearest_clamp = sampler(false);

        Self {
            pipeline,
            upscale_pipeline,
            linear_clamp,
            nearest_clamp,
            history: None,
            extent: [0, 0],
            upscaling: Upscaling::Native,
            frame: 0,
            reset: AtomicBool::new(true),
            previous_view_proj: None,
            settings: TaaSettings::default(),
        }
    }

    /// Advance one frame: allocate what the resolve will need, pick this frame's
    /// jitter, and hand back the matrices every rasterising pass must use.
    ///
    /// Called whether or not TAA is enabled, because `prev_view_proj` has to
    /// track the camera regardless — otherwise the first frame after it is
    /// switched on reprojects against wherever the camera was when it was
    /// switched off. `vendor` is a frame a vendor's upscaler resolves: it
    /// jitters as any resolved frame does, and holds no history here.
    pub(super) fn begin_frame(
        &mut self,
        device: &Device,
        settings: &TaaSettings,
        camera: &Camera,
        extents: Extents,
        vendor: bool,
    ) -> FrameView {
        self.settings = *settings;
        let extent = extents.output;

        if settings.resolves() && vendor {
            if self.history.is_some() {
                self.release(device);
                self.reset.store(true, Ordering::Relaxed);
            }
            self.frame = self.frame.wrapping_add(1);
        } else if settings.resolves() {
            if self.history.is_none()
                || self.extent != extent
                || self.upscaling != settings.upscaling
            {
                self.release(device);
                self.history = Some(allocate_history(
                    device,
                    extent,
                    history_format(settings.upscaling),
                ));
                self.extent = extent;
                self.upscaling = settings.upscaling;
                self.reset.store(true, Ordering::Relaxed);
            }
            self.frame = self.frame.wrapping_add(1);
        } else {
            // Freed rather than kept: two full-resolution HDR targets is real
            // memory, and anything they held is stale the moment a frame renders
            // without them.
            self.release(device);
            self.reset.store(true, Ordering::Relaxed);
        }

        let aspect = extent[0] as f32 / extent[1].max(1) as f32;
        let view = camera.view();
        let projection = camera.projection(aspect);
        let unjittered_view_proj = projection * view;

        let phases = settings.upscaling.jitter_phases();
        let (jitter, jitter_pixels) = if settings.resolves() {
            (
                jitter_offset(self.frame, phases, extents.render) * settings.jitter_scale,
                halton_offset(self.frame, phases) * settings.jitter_scale,
            )
        } else {
            (Vec2::ZERO, Vec2::ZERO)
        };
        let proj = jittered(projection, jitter);

        let prev_view_proj = self.previous_view_proj.unwrap_or(unjittered_view_proj);
        self.previous_view_proj = Some(unjittered_view_proj);

        FrameView {
            view,
            proj,
            view_proj: proj * view,
            unjittered_view_proj,
            prev_view_proj,
            jitter,
            jitter_pixels,
        }
    }

    /// Drop the history at the next resolve, for a camera cut: the frame after
    /// it shares nothing with the frames before, and reprojecting them would
    /// ghost the last shot over the first frames of the next.
    pub(super) fn reset(&self) {
        self.reset.store(true, Ordering::Relaxed);
    }

    /// Whether the resolve recording now is to drop its history, for a
    /// vendor's resolve, which keeps that history itself; clears it, as the
    /// engine's own resolve does once it has run.
    pub(super) fn take_reset(&self) -> bool {
        self.reset.swap(false, Ordering::Relaxed)
    }

    fn release(&mut self, device: &Device) {
        for view in self.history.take().into_iter().flatten() {
            device.destroy_texture(view);
        }
    }

    /// This frame's resolve target, which is next frame's history.
    pub(super) fn output_view(&self) -> TextureViewId {
        self.pair()[(self.frame & 1) as usize]
    }

    pub(super) fn history_view(&self) -> TextureViewId {
        self.pair()[((self.frame + 1) & 1) as usize]
    }

    fn pair(&self) -> &[TextureViewId; 2] {
        self.history
            .as_ref()
            .expect("the graph scheduled a TAA resolve with no history allocated")
    }

    pub(super) fn record(
        &self,
        device: &Device,
        builder: &mut Encoder,
        view: &FrameView,
        color: TextureViewId,
        velocity: TextureViewId,
        depth: TextureViewId,
    ) {
        let target = self.output_view();
        let extent = device.view_extent(target);
        let pipeline = match self.upscaling {
            Upscaling::Native => self.pipeline,
            _ => self.upscale_pipeline,
        };

        // The resolve's uniforms: a buffer rather than push constants because
        // two mat4s already fill the guaranteed 128-byte push range on their own.
        // `params`: xy = 1/extent, z = history weight, w != 0 to ignore the
        // history; then the jitter in render pixels, which only the upscaling
        // resolve reads.
        let params = [
            1.0 / extent[0] as f32,
            1.0 / extent[1] as f32,
            self.settings.feedback.clamp(0.0, 0.99),
            self.reset.load(Ordering::Relaxed) as u32 as f32,
            view.jitter_pixels.x,
            view.jitter_pixels.y,
            0.0,
            0.0,
        ];
        let mut words = [0u32; 40];
        let matrices = [
            view.unjittered_view_proj.inverse().to_cols_array_2d(),
            view.prev_view_proj.to_cols_array_2d(),
        ];
        for (word, float) in words
            .iter_mut()
            .zip(matrices.iter().flatten().flatten().chain(&params))
        {
            *word = float.to_bits();
        }
        let uniforms = device.upload_uniform(&words);

        let sampled = |binding, view, sampler| Binding {
            binding,
            resource: BindingResource::SampledTexture { view, sampler },
        };
        let set = device.create_transient_bind_group(
            pipeline,
            0,
            &[
                sampled(0, color, self.nearest_clamp),
                sampled(1, self.history_view(), self.linear_clamp),
                sampled(2, velocity, self.nearest_clamp),
                sampled(3, depth, self.nearest_clamp),
                Binding {
                    binding: 4,
                    resource: BindingResource::StorageTexture(target),
                },
                Binding {
                    binding: 5,
                    resource: BindingResource::UniformBuffer(uniforms),
                },
            ],
        );

        builder.bind_pipeline(pipeline);
        builder.bind_group(0, set);

        // The dispatch covers exactly the target's extent and the shader discards
        // invocations past `imageSize(u_target)`, so nothing writes outside it.
        builder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);

        // The frame that just resolved is the history the next one reads, so
        // whatever made it untrustworthy is over.
        self.reset.store(false, Ordering::Relaxed);
    }
}

/// Where inside the pixel frame `index` samples, in pixels from its centre, of
/// a sequence `phases` long: [`Upscaling::jitter_phases`], so that at a ratio
/// each output pixel is still visited about as often as at native.
///
/// Halton (2,3) rather than a fixed rotated grid: the sequence is
/// low-discrepancy at *every* prefix length, so a history only four frames deep
/// — all a moving camera ever keeps — is still evenly distributed.
fn halton_offset(index: u64, phases: u32) -> Vec2 {
    // From one, because Halton's zeroth element is the origin and a frame with
    // no offset contributes nothing the unjittered image would not have.
    let phase = (index % phases as u64) as u32 + 1;
    Vec2::new(halton(phase, 2) - 0.5, halton(phase, 3) - 0.5)
}

/// [`halton_offset`] as an NDC offset, for a frame rendered at `extent`.
fn jitter_offset(index: u64, phases: u32, extent: [u32; 2]) -> Vec2 {
    // NDC spans [-1, 1] across the frame, so one pixel is 2 / extent of it.
    halton_offset(index, phases) * 2.0 / Vec2::new(extent[0].max(1) as f32, extent[1].max(1) as f32)
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut fraction = 1.0f32;
    let mut result = 0.0f32;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// Offset the projection so the frame rasterises `jitter` NDC units across.
///
/// The third column, not the fourth: clip.xy must move by `jitter * w`, and `w`
/// comes from the view-space z that column multiplies. Adding to the translation
/// column instead would shift the image by a constant in clip space, which is a
/// shear once the perspective divide is done with it.
fn jittered(mut projection: Mat4, jitter: Vec2) -> Mat4 {
    projection.z_axis.x -= jitter.x;
    projection.z_axis.y -= jitter.y;
    projection
}

fn allocate_history(device: &Device, extent: [u32; 2], format: Format) -> [TextureViewId; 2] {
    std::array::from_fn(|_| {
        let texture = device.create_texture(&TextureDesc {
            format,
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Sample1,
            // Written as a storage image by the resolve, read as a sampled
            // one by both the resolve's next frame and everything the frame
            // composites afterwards.
            usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
            memoryless: false,
        });
        // The layout `taa_history` is declared to enter in, from the first
        // frame on; that frame's contents are still ignored (see `reset`).
        device.initialise_sampled(texture);
        texture
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NATIVE_PHASES: u32 = 8;

    /// The first resolve after the history is allocated passes the frame
    /// through untouched, which is the simplest thing a compute pass can be
    /// held to on a real device: whatever went in comes out.
    #[test]
    #[ignore = "needs a Metal device"]
    #[cfg(target_os = "macos")]
    fn the_first_resolve_passes_the_frame_through_on_metal() {
        use crate::gfx::format::{Format, SampleCount};
        use crate::gfx::graph::ImageUsage;
        use crate::gfx::rhi::{TextureDesc, TextureDimension};
        let device = Device::Metal(crate::gfx::metal::MtlDriver::new().expect("a Metal device"));
        device.begin_frame(0);
        let mut pass = TaaPass::new(&device);
        let settings = TaaSettings {
            enabled: true,
            ..TaaSettings::default()
        };
        let camera = Camera::default();
        let view = pass.begin_frame(&device, &settings, &camera, Extents::native([8, 8]), false);

        let floats = |values: &[f32]| -> Vec<u8> { bytemuck::cast_slice(values).to_vec() };
        let color = device.create_texture_2d(
            &floats(&[0.5, 0.25, 0.125, 1.0].repeat(64)),
            [8, 8],
            Format::Rgba32Float,
        );
        let velocity = device.create_texture_2d(&vec![0u8; 8 * 8 * 4], [8, 8], Format::Rg16Float);
        let depth = device.create_texture(&TextureDesc {
            format: Format::D32Float,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Sample1,
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            memoryless: false,
        });
        device.fill_depth(depth, crate::gfx::depth::CLEAR);

        device.record_blocking(|encoder| {
            pass.record(&device, encoder, &view, color, velocity, depth);
        });
        let Device::Metal(driver) = &device else {
            unreachable!()
        };
        let pixels = driver.read_texture(pass.output_view());
        assert!(
            pixels.iter().any(|&byte| byte != 0),
            "the resolve wrote nothing: {pixels:?}"
        );
    }

    /// Every offset has to land inside the pixel, or the frame is sampling a
    /// neighbour and the resolve accumulates a blur rather than a reconstruction.
    #[test]
    fn the_jitter_stays_within_one_pixel() {
        let extent = [1920, 1080];
        let half = Vec2::new(1.0 / extent[0] as f32, 1.0 / extent[1] as f32);
        for frame in 0..64u64 {
            let offset = jitter_offset(frame, NATIVE_PHASES, extent);
            assert!(
                offset.x.abs() <= half.x && offset.y.abs() <= half.y,
                "frame {frame} jittered {offset:?} past half a pixel {half:?}",
            );
        }
    }

    /// The point of a sequence is that consecutive frames sample *different*
    /// places; a constant offset would just shift the image.
    #[test]
    fn consecutive_frames_sample_different_points() {
        let extent = [1920, 1080];
        for frame in 0..NATIVE_PHASES as u64 {
            assert_ne!(
                jitter_offset(frame, NATIVE_PHASES, extent),
                jitter_offset(frame + 1, NATIVE_PHASES, extent),
                "frame {frame} and the next sampled the same point",
            );
        }
    }

    /// The offsets must average out, or the accumulated image sits off-centre
    /// from where the unjittered one would be.
    #[test]
    fn a_full_cycle_is_centred() {
        let extent = [1920, 1080];
        let total: Vec2 = (0..NATIVE_PHASES as u64)
            .map(|frame| jitter_offset(frame, NATIVE_PHASES, extent))
            .sum();
        let mean = total / NATIVE_PHASES as f32;
        assert!(mean.length() < 1e-4, "the cycle is biased by {mean:?}");
    }

    /// The jitter must be a clip-space translation of `jitter * w` and nothing
    /// else: the same world point has to keep its depth and its `w`, or the
    /// depth test would disagree between a jittered and an unjittered frame.
    #[test]
    fn jittering_translates_clip_space_and_leaves_depth_alone() {
        use glam::Vec4;

        let camera = Camera::default();
        let projection = camera.projection(16.0 / 9.0);
        let jitter = Vec2::new(0.0013, -0.0021);
        let point = Vec4::new(1.5, -0.5, -7.0, 1.0);

        let plain = projection * point;
        let shaken = jittered(projection, jitter) * point;

        assert!((shaken.z - plain.z).abs() < 1e-6);
        assert!((shaken.w - plain.w).abs() < 1e-6);
        assert!((shaken.x - (plain.x + jitter.x * plain.w)).abs() < 1e-5);
        assert!((shaken.y - (plain.y + jitter.y * plain.w)).abs() < 1e-5);
    }
}
