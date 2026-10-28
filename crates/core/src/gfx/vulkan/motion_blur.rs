//! Motion blur: a tile-max velocity pyramid, then a reconstruction gather.
//!
//! The two tile passes exist to answer one question cheaply — "is anything near
//! this pixel moving fast?" — because the honest answer is what separates blur
//! that bleeds past a silhouette from blur that stops dead at one. A pixel of
//! still background beside a fast object has no motion of its own, so a gather
//! driven by per-pixel velocity leaves it sharp and the object cut out against
//! it. Driving the gather by the neighbourhood's strongest motion instead, and
//! letting each tap argue for itself, is the reconstruction of McGuire et al.
//! (2012) that both Unreal and HDRP ship.
//!
//! They cost almost nothing: the first pass reads each velocity texel once
//! across the whole frame, and the second runs over an image a two-hundred-and-
//! fifty-sixth the size.

use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, ComputePipelineDesc, Device, Encoder, PipelineId,
    SamplerDesc, SamplerId, TextureViewId,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::{Camera, MotionBlurSettings};

use super::taa::FrameView;

/// Side of the compute workgroup for every pass here.
const TILE: u32 = 8;

/// How many times the frame is halved to reach the tile grid, so a tile is 16
/// pixels across.
///
/// A shift because that is how the graph sizes an image, and 16 rather than the
/// 20 the paper uses for the same reason: the level has to land on exactly the
/// size successive halvings produce, or the tile a pixel computes for itself is
/// not the texel it reads.
pub(super) const TILE_SHIFT: u32 = 4;

/// The longest streak the reconstruction will draw, in pixels.
///
/// Twice a tile, and that factor is load-bearing rather than generous: the
/// gather spreads a vector half its length either side of centre, so a
/// two-tile vector reaches exactly one tile — which is precisely the ring the
/// neighbour-max pass dilates over. Raising this without widening that dilation
/// would let a fast object's blur cross into a tile that never heard about it,
/// and the tile grid would print itself on the frame.
const MAX_BLUR_PIXELS: f32 = (2 << TILE_SHIFT) as f32;

/// The uniforms both the tile pass and the gather bind, as the words of their
/// std140 block: `inv_view_proj`, `prev_view_proj`, then `params` — x = shutter
/// fraction, y = max blur radius in pixels, z = near, w = far. A buffer rather
/// than push constants because two mat4s alone fill the guaranteed 128-byte push
/// range.
const UBO_WORDS: usize = 36;

pub struct MotionBlurPass {
    tile_max_pipeline: PipelineId,
    neighbour_max_pipeline: PipelineId,
    gather_pipeline: PipelineId,
    /// Nearest throughout. Velocity and depth interpolated across a silhouette
    /// describe a surface that is not there, and a tile maximum averaged with
    /// its neighbour is no longer a maximum.
    nearest_clamp: SamplerId,
    settings: MotionBlurSettings,
    near: f32,
    far: f32,
}

impl MotionBlurPass {
    pub fn new(device: &Device) -> Self {
        let pipeline = |shader| device.create_compute_pipeline(&ComputePipelineDesc { shader });
        let tile_max_pipeline = pipeline(ShaderId::MotionBlurTileMaxComp);
        let neighbour_max_pipeline = pipeline(ShaderId::MotionBlurNeighbourMaxComp);
        let gather_pipeline = pipeline(ShaderId::MotionBlurGatherComp);

        let nearest_clamp = device.create_sampler(SamplerDesc {
            linear: false,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(0.0),
        });

        Self {
            tile_max_pipeline,
            neighbour_max_pipeline,
            gather_pipeline,
            nearest_clamp,
            settings: MotionBlurSettings::default(),
            near: 0.1,
            far: 1000.0,
        }
    }

    pub fn begin_frame(&mut self, settings: &MotionBlurSettings, camera: &Camera) {
        self.settings = *settings;
        self.near = camera.near;
        self.far = camera.far;
    }

    /// The uniforms both the tile pass and the gather bind.
    ///
    /// One description of the frame's motion, allocated per pass rather than
    /// shared, because the two record at different points in the frame — but
    /// derived from the same [`FrameView`], so they cannot disagree about where
    /// the sky went.
    fn uniforms(&self, device: &Device, view: &FrameView) -> BufferId {
        let Device::Vulkan(driver) = device;
        // Unjittered, so the sky's reprojection agrees with the motion vectors the
        // prepass wrote — those are unjittered too.
        let matrices = [
            view.unjittered_view_proj.inverse().to_cols_array_2d(),
            view.prev_view_proj.to_cols_array_2d(),
        ];
        let params = [
            self.settings.shutter_fraction(),
            MAX_BLUR_PIXELS,
            self.near,
            self.far,
        ];
        let mut words = [0u32; UBO_WORDS];
        for (word, float) in words
            .iter_mut()
            .zip(matrices.iter().flatten().flatten().chain(&params))
        {
            *word = float.to_bits();
        }
        driver.upload_uniform(&words)
    }

    pub(super) fn record_tile_max(
        &self,
        device: &Device,
        builder: &mut Encoder,
        view: &FrameView,
        velocity: TextureViewId,
        depth: TextureViewId,
        target: TextureViewId,
    ) {
        let sampled = |binding, view| Binding {
            binding,
            resource: BindingResource::SampledTexture {
                view,
                sampler: self.nearest_clamp,
            },
        };
        let set = device.create_bind_group(
            self.tile_max_pipeline,
            0,
            &[
                sampled(0, velocity),
                sampled(1, depth),
                Binding {
                    binding: 2,
                    resource: BindingResource::StorageTexture(target),
                },
                Binding {
                    binding: 3,
                    resource: BindingResource::UniformBuffer(self.uniforms(device, view)),
                },
            ],
        );

        builder.bind_pipeline(self.tile_max_pipeline);
        builder.bind_group(0, set);
        dispatch_over(device, builder, target);
    }

    pub fn record_neighbour_max(
        &self,
        device: &Device,
        builder: &mut Encoder,
        tiles: TextureViewId,
        target: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.neighbour_max_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: tiles,
                        sampler: self.nearest_clamp,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageTexture(target),
                },
            ],
        );

        builder.bind_pipeline(self.neighbour_max_pipeline);
        builder.bind_group(0, set);
        dispatch_over(device, builder, target);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_gather(
        &self,
        device: &Device,
        builder: &mut Encoder,
        view: &FrameView,
        color: TextureViewId,
        velocity: TextureViewId,
        depth: TextureViewId,
        neighbour: TextureViewId,
        target: TextureViewId,
    ) {
        let sampled = |binding, view| Binding {
            binding,
            resource: BindingResource::SampledTexture {
                view,
                sampler: self.nearest_clamp,
            },
        };
        let set = device.create_bind_group(
            self.gather_pipeline,
            0,
            &[
                sampled(0, color),
                sampled(1, velocity),
                sampled(2, depth),
                sampled(3, neighbour),
                Binding {
                    binding: 4,
                    resource: BindingResource::StorageTexture(target),
                },
                Binding {
                    binding: 5,
                    resource: BindingResource::UniformBuffer(self.uniforms(device, view)),
                },
            ],
        );

        builder.bind_pipeline(self.gather_pipeline);
        builder.bind_group(0, set);
        dispatch_over(device, builder, target);
    }
}

fn dispatch_over(device: &Device, builder: &mut Encoder, target: TextureViewId) {
    let extent = device.view_extent(target);
    builder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tile shift, the blur cap and the shader's `TILE` are one fact spread
    /// across three files, and the reconstruction is only seamless while they
    /// agree: the gather reaches half a vector, so the cap has to be exactly the
    /// two tiles the neighbour-max pass dilates one ring over.
    #[test]
    fn the_blur_cap_reaches_exactly_the_dilated_ring() {
        let tile = (1u32 << TILE_SHIFT) as f32;
        assert_eq!(tile, 16.0, "the shaders declare a 16-pixel tile");
        assert_eq!(MAX_BLUR_PIXELS * 0.5, tile);
    }
}
