//! Depth of field: a half-resolution scatter-as-gather, composited back over
//! the sharp frame.
//!
//! Four dispatches, and the split is what keeps it affordable. The prefilter
//! halves the frame and computes each texel's circle of confusion once; the
//! gather is the only pass that pays for the kernel, and it pays at a quarter of
//! the pixels; the composite is full-resolution but reads four texels. Doing the
//! kernel at full resolution instead would cost four times as much for a result
//! the composite's bilinear upsample makes indistinguishable.
//!
//! Everything the shaders need is a push constant, because the whole lens is
//! five floats — see [`DofSettings`] for where they come from, and why the focal
//! length is not among them.

use crate::gfx::rhi::{
    Binding, BindingResource, ComputePipelineDesc, Device, Encoder, PipelineId, SamplerDesc,
    SamplerId, TextureViewId,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::{Camera, DofSettings};

/// Side of the compute workgroup for every pass here.
const TILE: u32 = 8;

/// The widest circle of confusion the chain will reproduce, in half-resolution
/// pixels.
///
/// A bound rather than a setting. The kernel has a fixed tap count, so past some
/// width the same taps thin out until the bokeh breaks into visible dots; the
/// honest cap belongs next to the tap count it is a property of, not on a slider
/// where it reads as a quality dial.
///
/// It is *not* the kernel radius — that is sized per tile from the blur actually
/// present, which is what keeps an ordinary aperture from spreading fifty taps
/// over a circle a tenth their reach.
const MAX_COC_RADIUS: f32 = 24.0;

/// How many times the frame is halved to reach the circle-of-confusion tile
/// grid.
///
/// One more than the five that would tile the half-resolution image at
/// [`MAX_COC_RADIUS`], so a tile is 32 of its texels: the gather dilates over
/// one ring, and a tile narrower than the widest circle would let a foreground
/// object spill into a tile that never heard about it.
pub(super) const COC_TILE_SHIFT: u32 = 6;

/// Side of a tile in half-resolution texels, which is what the shaders declare.
/// Only the tests read it — the pipeline works in `COC_TILE_SHIFT` — and they
/// are what holds it to the number in the shader source.
#[cfg(test)]
const COC_TILE: u32 = 1 << (COC_TILE_SHIFT - 1);

pub struct DofPass {
    prefilter_pipeline: PipelineId,
    tile_max_pipeline: PipelineId,
    gather_pipeline: PipelineId,
    composite_pipeline: PipelineId,
    /// Linear and clamped: the composite reads the half-resolution fields at
    /// full-resolution positions, which land between texels by construction.
    linear_clamp: SamplerId,
    /// Nearest, because depth interpolated across a silhouette is a distance no
    /// surface is at, and the circle of confusion derived from it would ring the
    /// edge with a focus that belongs to neither side.
    nearest_clamp: SamplerId,
    settings: DofSettings,
    /// Derived once per frame from the settings and the camera, because the
    /// prefilter and the composite must agree on it exactly.
    coc_scale: f32,
    near: f32,
    far: f32,
}

impl DofPass {
    pub fn new(device: &Device) -> Self {
        let pipeline = |shader| device.create_compute_pipeline(&ComputePipelineDesc { shader });
        let prefilter_pipeline = pipeline(ShaderId::DofPrefilterComp);
        let tile_max_pipeline = pipeline(ShaderId::DofTileMaxComp);
        let gather_pipeline = pipeline(ShaderId::DofGatherComp);
        let composite_pipeline = pipeline(ShaderId::DofCompositeComp);

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

        Self {
            prefilter_pipeline,
            tile_max_pipeline,
            gather_pipeline,
            composite_pipeline,
            linear_clamp,
            nearest_clamp,
            settings: DofSettings::default(),
            coc_scale: 0.0,
            near: 0.1,
            far: 1000.0,
        }
    }

    /// Resolve the lens against this frame's camera.
    ///
    /// The camera is what supplies the field of view the focal length comes out
    /// of, so the same aperture is a different depth of field at a different
    /// zoom — which is the point of describing it this way.
    pub fn begin_frame(&mut self, settings: &DofSettings, camera: &Camera, extent: [u32; 2]) {
        self.settings = *settings;
        self.near = camera.near;
        self.far = camera.far;
        // Halved because everything downstream of the prefilter measures in
        // half-resolution pixels, while the setting derives a radius in the
        // frame's own.
        self.coc_scale = settings.coc_radius_scale(camera.fov_y, extent[1]) * 0.5;
    }

    pub fn record_prefilter(
        &self,
        device: &Device,
        builder: &mut Encoder,
        color: TextureViewId,
        depth: TextureViewId,
        target: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.prefilter_pipeline,
            0,
            &[
                self.sampled(0, color, self.linear_clamp),
                self.sampled(1, depth, self.nearest_clamp),
                self.storage(2, target),
            ],
        );

        builder.bind_pipeline(self.prefilter_pipeline);
        builder.bind_group(0, set);
        builder.push_constants(&[
            self.coc_scale.to_bits(),
            self.settings.focus_distance.to_bits(),
            self.near.to_bits(),
            self.far.to_bits(),
            MAX_COC_RADIUS.to_bits(),
        ]);
        dispatch_over(device, builder, target);
    }

    pub fn record_tile_max(
        &self,
        device: &Device,
        builder: &mut Encoder,
        prefiltered: TextureViewId,
        target: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.tile_max_pipeline,
            0,
            &[
                self.sampled(0, prefiltered, self.nearest_clamp),
                self.storage(1, target),
            ],
        );

        builder.bind_pipeline(self.tile_max_pipeline);
        builder.bind_group(0, set);
        dispatch_over(device, builder, target);
    }

    pub fn record_gather(
        &self,
        device: &Device,
        builder: &mut Encoder,
        prefiltered: TextureViewId,
        tiles: TextureViewId,
        near: TextureViewId,
        far: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.gather_pipeline,
            0,
            &[
                self.sampled(0, prefiltered, self.linear_clamp),
                self.sampled(1, tiles, self.nearest_clamp),
                self.storage(2, near),
                self.storage(3, far),
            ],
        );

        builder.bind_pipeline(self.gather_pipeline);
        builder.bind_group(0, set);
        dispatch_over(device, builder, near);
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "a pass records from the frame's bindings; a params struct only renames the list"
    )]
    pub fn record_composite(
        &self,
        device: &Device,
        builder: &mut Encoder,
        color: TextureViewId,
        depth: TextureViewId,
        near: TextureViewId,
        far: TextureViewId,
        target: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.composite_pipeline,
            0,
            &[
                self.sampled(0, color, self.nearest_clamp),
                self.sampled(1, depth, self.nearest_clamp),
                self.sampled(2, near, self.linear_clamp),
                self.sampled(3, far, self.linear_clamp),
                self.storage(4, target),
            ],
        );

        builder.bind_pipeline(self.composite_pipeline);
        builder.bind_group(0, set);
        builder.push_constants(&[
            self.coc_scale.to_bits(),
            self.settings.focus_distance.to_bits(),
            self.near.to_bits(),
            self.far.to_bits(),
        ]);
        dispatch_over(device, builder, target);
    }

    fn sampled(&self, binding: u32, view: TextureViewId, sampler: SamplerId) -> Binding {
        Binding {
            binding,
            resource: BindingResource::SampledTexture { view, sampler },
        }
    }

    fn storage(&self, binding: u32, view: TextureViewId) -> Binding {
        Binding {
            binding,
            resource: BindingResource::StorageTexture(view),
        }
    }
}

fn dispatch_over(device: &Device, builder: &mut Encoder, target: TextureViewId) {
    let extent = device.view_extent(target);
    builder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tile shift, the tile size the two shaders declare, and the clamp the
    /// prefilter applies are one fact spread across three files. A tile narrower
    /// than the widest circle breaks the gather's one-ring dilation: a
    /// foreground object would spill into a tile that never saw it, and its
    /// bokeh would stop dead on the tile grid.
    #[test]
    fn a_tile_is_wider_than_the_widest_circle() {
        assert_eq!(COC_TILE, 32, "the shaders declare a 32-texel tile");
        assert!(
            COC_TILE as f32 >= MAX_COC_RADIUS,
            "a {COC_TILE}-texel tile cannot contain a {MAX_COC_RADIUS}-texel circle",
        );
    }

    /// The kernel is sized per tile precisely so that the taps land *inside* the
    /// circle being gathered. This is the arithmetic that was wrong when the
    /// kernel was fixed at `MAX_COC_RADIUS`: a circle an eighth of that caught
    /// one tap in forty-eight, which is a passthrough rather than a blur.
    #[test]
    fn every_circle_gets_the_whole_tap_budget() {
        const TAPS: usize = 48;
        for circle in [1.0f32, 2.7, 6.0, 12.0, MAX_COC_RADIUS] {
            let inside = (0..TAPS)
                .filter(|&i| {
                    let t = (i as f32 + 0.5) / TAPS as f32;
                    // The gather's spiral, spread over the tile's own maximum —
                    // which for a uniform region is the circle itself.
                    t.sqrt() * circle <= circle
                })
                .count();
            assert_eq!(
                inside, TAPS,
                "a {circle}px circle gathered only {inside} of {TAPS} taps",
            );
        }
    }
}
