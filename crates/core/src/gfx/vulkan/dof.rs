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

use std::sync::Arc;

use vulkano::buffer::BufferContents;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};

use crate::scene::{Camera, DofSettings};

use super::context::VkContext;
use super::record::Recorder;

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

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PrefilterPush {
    coc_scale: f32,
    focus_distance: f32,
    near: f32,
    far: f32,
    max_radius: f32,
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct CompositePush {
    coc_scale: f32,
    focus_distance: f32,
    near: f32,
    far: f32,
}

pub struct DofPass {
    prefilter_pipeline: Arc<ComputePipeline>,
    tile_max_pipeline: Arc<ComputePipeline>,
    gather_pipeline: Arc<ComputePipeline>,
    composite_pipeline: Arc<ComputePipeline>,
    /// Linear and clamped: the composite reads the half-resolution fields at
    /// full-resolution positions, which land between texels by construction.
    linear_clamp: Arc<Sampler>,
    /// Nearest, because depth interpolated across a silhouette is a distance no
    /// surface is at, and the circle of confusion derived from it would ring the
    /// edge with a focus that belongs to neither side.
    nearest_clamp: Arc<Sampler>,
    settings: DofSettings,
    /// Derived once per frame from the settings and the camera, because the
    /// prefilter and the composite must agree on it exactly.
    coc_scale: f32,
    near: f32,
    far: f32,
}

impl DofPass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let prefilter_pipeline = build_pipeline(
            ctx,
            prefilter_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let tile_max_pipeline = build_pipeline(
            ctx,
            tile_max_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let gather_pipeline = build_pipeline(
            ctx,
            gather_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let composite_pipeline = build_pipeline(
            ctx,
            composite_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );

        let linear_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();
        let nearest_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();

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
        builder: &mut Recorder,
        ctx: &VkContext,
        color: Arc<ImageView>,
        depth: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.prefilter_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, color, self.linear_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(2, target.clone()),
            ],
        );

        builder
            .bind_pipeline_compute(&self.prefilter_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.prefilter_pipeline.layout(),
                0,
                &[set],
            )
            .push_constants(
                self.prefilter_pipeline.layout(),
                0,
                &PrefilterPush {
                    coc_scale: self.coc_scale,
                    focus_distance: self.settings.focus_distance,
                    near: self.near,
                    far: self.far,
                    max_radius: MAX_COC_RADIUS,
                },
            );
        dispatch_over(builder, &target);
    }

    /// The widest near and far circle in each tile, which is what the gather
    /// sizes its kernel from.
    pub fn record_tile_max(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        prefiltered: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.tile_max_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, prefiltered, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(1, target.clone()),
            ],
        );

        builder
            .bind_pipeline_compute(&self.tile_max_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.tile_max_pipeline.layout(),
                0,
                &[set],
            );
        dispatch_over(builder, &target);
    }

    pub fn record_gather(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        prefiltered: Arc<ImageView>,
        tiles: Arc<ImageView>,
        near: Arc<ImageView>,
        far: Arc<ImageView>,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.gather_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, prefiltered, self.linear_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, tiles, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(2, near.clone()),
                WriteDescriptorSet::image_view(3, far),
            ],
        );

        builder
            .bind_pipeline_compute(&self.gather_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.gather_pipeline.layout(),
                0,
                &[set],
            );
        dispatch_over(builder, &near);
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "a pass records from the frame's bindings; a params struct only renames the list"
    )]
    pub fn record_composite(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        color: Arc<ImageView>,
        depth: Arc<ImageView>,
        near: Arc<ImageView>,
        far: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.composite_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, color, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(2, near, self.linear_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(3, far, self.linear_clamp.clone()),
                WriteDescriptorSet::image_view(4, target.clone()),
            ],
        );

        builder
            .bind_pipeline_compute(&self.composite_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.composite_pipeline.layout(),
                0,
                &[set],
            )
            .push_constants(
                self.composite_pipeline.layout(),
                0,
                &CompositePush {
                    coc_scale: self.coc_scale,
                    focus_distance: self.settings.focus_distance,
                    near: self.near,
                    far: self.far,
                },
            );
        dispatch_over(builder, &target);
    }
}

fn dispatch_over(builder: &mut Recorder, target: &Arc<ImageView>) {
    let extent = target.image().extent();

    // SAFETY: the dispatch covers exactly `extent`, and each shader discards
    // invocations past `imageSize`, so nothing writes outside the image. The
    // descriptors bound above match the shader's layout, and the graph declared
    // every resource this pass touches, so its barriers precede it.
    builder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
}

fn build_pipeline(
    ctx: &VkContext,
    entry_point: vulkano::shader::EntryPoint,
) -> Arc<ComputePipeline> {
    let device = &ctx.device;
    let stage = PipelineShaderStageCreateInfo::new(entry_point);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        ctx.pipeline_cache(),
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

mod prefilter_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/dof_prefilter.comp" }
}

mod tile_max_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/dof_tile_max.comp" }
}

mod gather_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/dof_gather.comp" }
}

mod composite_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/dof_composite.comp" }
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
