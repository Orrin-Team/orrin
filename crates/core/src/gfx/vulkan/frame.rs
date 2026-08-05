//! The engine's frame, declared as a graph.
//!
//! This module is the one place that says what a frame *is*. It takes no
//! `Device` and allocates nothing: it registers resources and passes and hands
//! back the compiled result, so the same function that runs on the GPU each
//! frame is the one CI compiles and asserts the barrier plan of.
//!
//! Adding a pass is a registration here plus a [`PassBody`] arm in `execute`.
//! Nothing computes an execution order, a layout, or a barrier by hand, which is
//! what makes the shadow cascades of Part 1 a loop over `declare` rather than
//! surgery on a fixed pipeline.

use vulkano::format::Format;
use vulkano::image::ImageLayout;

use crate::gfx::graph::{
    Access, Extent, FrameGraph, GraphBuilder, GraphError, ImageDesc, PassId, PassKind, ResourceId,
    compile,
};
use crate::gfx::shadows::MAX_CASCADES;

use super::MSAA_SAMPLES;
use super::bloom::MAX_BLOOM_MIPS;
use super::hdr::HDR_FORMAT;
use super::ssao::{AO_FORMAT, NORMAL_FORMAT};
use super::swapchain::DEPTH_FORMAT;

/// What a frame's structure depends on. A change to any of these recompiles the
/// graph; nothing else does, which is the "recompiled on structure change, not
/// per frame" rule made concrete — the field list *is* the rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameConfig {
    pub color_format: Format,
    pub ssao: bool,
    /// Whether the frame meters its own luminance. Off is a different graph, not
    /// a flag read at record time: the two compute passes are never registered
    /// and the tonemap pass falls back to the manual exposure it is pushed.
    pub auto_exposure: bool,
    /// Levels in the bloom chain, zero for none. Derived from the frame's extent
    /// rather than set, so the number of passes registered cannot disagree with
    /// the number of levels there is room for — the same reason
    /// `shadow_cascades` is sourced from the cascade set.
    pub bloom_mips: u8,
    /// Whether the editor's egui overlay draws over the frame. Off for headless
    /// and export renders.
    pub overlay: bool,
    pub shadow_cascades: u8,
    pub shadow_resolution: u32,
}

/// Which piece of engine code a graph node runs.
///
/// The graph knows a pass by its declarations; this is the other half of the
/// mapping, kept as data so it stays device-free and so the executor's dispatch
/// is exhaustive by the compiler's own reckoning rather than by convention.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PassBody {
    SsaoPrepass,
    SsaoResolve,
    SsaoBlur,
    Forward,
    LuminanceHistogram,
    LuminanceAverage,
    /// Half the frame, exposed and firefly-weighted: the chain's first level.
    BloomPrefilter,
    /// Writes down-chain level `n`, reading level `n - 1`.
    BloomDownsample(u32),
    /// Writes up-chain level `n`, reading the coarser level and down-chain `n`.
    BloomUpsample(u32),
    Tonemap,
    Overlay,
    ShadowCascade(u32),
}

/// Handles the executor needs to bind per-frame resources and to find the
/// graph-owned views a pass draws into.
#[derive(Clone, Copy, Debug)]
pub struct FrameIds {
    pub object_transforms: ResourceId,
    pub swapchain_color: ResourceId,
    pub hdr_color: ResourceId,
    pub msaa_hdr: ResourceId,
    pub msaa_depth: ResourceId,
    /// Written by the metering passes and read by the tonemap pass. Always
    /// declared, because the tonemap pass binds it whether or not anything wrote
    /// it this frame — an import may be read without a writer, which is exactly
    /// the case metering-off produces.
    pub exposure: ResourceId,
    /// `None` when metering is off, along with the two passes that touch it.
    pub histogram: Option<ResourceId>,
    pub bloom: Option<BloomIds>,
    pub ssao: Option<SsaoIds>,
    pub shadows: Option<ResourceId>,
}

/// The two chains bloom needs, each level a separate image.
///
/// Fixed-length arrays rather than `Vec` so `FrameIds` stays `Copy`; only the
/// first `mips` entries of `down` and the first `mips - 1` of `up` are ever
/// populated, and the rest stay `None` so a length mistake is a panic naming the
/// level rather than a read of some other pass's image.
#[derive(Clone, Copy, Debug)]
pub struct BloomIds {
    pub mips: u8,
    pub down: [Option<ResourceId>; MAX_BLOOM_MIPS],
    pub up: [Option<ResourceId>; MAX_BLOOM_MIPS],
}

impl BloomIds {
    pub fn down(&self, level: usize) -> ResourceId {
        self.down[level].expect("bloom down-chain level was never declared")
    }

    pub fn up(&self, level: usize) -> ResourceId {
        self.up[level].expect("bloom up-chain level was never declared")
    }

    /// What the tonemap pass composites: the finest level of the up chain, or —
    /// for a one-level chain, which has no upsample step — the down chain's only
    /// level.
    pub fn result(&self) -> ResourceId {
        if self.mips >= 2 {
            self.up(0)
        } else {
            self.down(0)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SsaoIds {
    pub normal: ResourceId,
    pub depth: ResourceId,
    pub raw_ao: ResourceId,
    pub ao: ResourceId,
}

pub struct Frame {
    pub graph: FrameGraph,
    pub ids: FrameIds,
    /// Indexed by [`PassId`], so a pass's declarations and its body cannot drift.
    pub bodies: Vec<PassBody>,
}

pub fn declare(config: FrameConfig) -> Result<Frame, GraphError> {
    let mut builder = GraphBuilder::new();
    let mut bodies = Vec::new();
    let record = |id: PassId, body: PassBody, bodies: &mut Vec<PassBody>| {
        debug_assert_eq!(id.index(), bodies.len());
        bodies.push(body);
    };

    // Host-written each frame and read by both geometry passes; the per-object
    // inverse-transpose is too expensive to compute twice, so the two passes
    // share one upload and the graph records that they do.
    let object_transforms = builder.import_buffer("object_transforms");

    let swapchain_color = builder.import_image(
        "swapchain_color",
        ImageDesc::new(config.color_format),
        // An acquired image's contents are not ours to keep, and the
        // presentation engine wants it back in `PresentSrc`.
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );

    let shadows = (config.shadow_cascades > 0).then(|| {
        builder.create_image(
            "shadow_cascades",
            ImageDesc::new(DEPTH_FORMAT)
                .extent(Extent::Fixed([config.shadow_resolution; 2]))
                .array_layers(config.shadow_cascades as u32),
        )
    });

    // Resource and pass names live in separate namespaces, so a level's image
    // and the pass that writes it can share a name — and it reads well in the
    // plan, where `bloom_down_2` writes `bloom_down_2`.
    const BLOOM_DOWN_NAMES: [&str; MAX_BLOOM_MIPS] = [
        "bloom_down_0",
        "bloom_down_1",
        "bloom_down_2",
        "bloom_down_3",
        "bloom_down_4",
        "bloom_down_5",
    ];
    const BLOOM_UP_NAMES: [&str; MAX_BLOOM_MIPS] = [
        "bloom_up_0",
        "bloom_up_1",
        "bloom_up_2",
        "bloom_up_3",
        "bloom_up_4",
        "bloom_up_5",
    ];
    // Level 0's downsample is the prefilter, which has its own name; the slot is
    // present so the two tables index alike.
    const BLOOM_DOWN_PASS_NAMES: [&str; MAX_BLOOM_MIPS] = BLOOM_DOWN_NAMES;
    const BLOOM_UP_PASS_NAMES: [&str; MAX_BLOOM_MIPS] = BLOOM_UP_NAMES;

    const CASCADE_PASS_NAMES: [&str; MAX_CASCADES] = [
        "shadow_cascade_0",
        "shadow_cascade_1",
        "shadow_cascade_2",
        "shadow_cascade_3",
    ];

    if let Some(shadows) = shadows {
        for cascade in 0..config.shadow_cascades as u32 {
            let id = builder
                .pass(CASCADE_PASS_NAMES[cascade as usize], PassKind::Inline)
                .access(object_transforms, Access::StorageRead)
                .access(shadows, Access::DepthAttachment)
                .build();
            record(id, PassBody::ShadowCascade(cascade), &mut bodies);
        }
    }

    let ssao = config.ssao.then(|| SsaoIds {
        normal: builder.create_image("ssao_normal", ImageDesc::new(NORMAL_FORMAT)),
        depth: builder.create_image("ssao_depth", ImageDesc::new(DEPTH_FORMAT)),
        raw_ao: builder.create_image("ssao_raw_ao", ImageDesc::new(AO_FORMAT)),
        ao: builder.create_image("ssao_ao", ImageDesc::new(AO_FORMAT)),
    });

    let msaa_hdr =
        builder.create_image("msaa_hdr", ImageDesc::new(HDR_FORMAT).samples(MSAA_SAMPLES));
    let msaa_depth = builder.create_image(
        "msaa_depth",
        ImageDesc::new(DEPTH_FORMAT).samples(MSAA_SAMPLES),
    );
    let hdr_color = builder.create_image("hdr_color", ImageDesc::new(HDR_FORMAT));

    if let Some(ssao) = ssao {
        let id = builder
            .pass("ssao_prepass", PassKind::Inline)
            .access(object_transforms, Access::StorageRead)
            .access(ssao.normal, Access::ColorAttachment)
            .access(ssao.depth, Access::DepthAttachment)
            .build();
        record(id, PassBody::SsaoPrepass, &mut bodies);

        let id = builder
            .pass("ssao_resolve", PassKind::Inline)
            .access(ssao.depth, Access::Sampled)
            .access(ssao.normal, Access::Sampled)
            .access(ssao.raw_ao, Access::ColorAttachment)
            .build();
        record(id, PassBody::SsaoResolve, &mut bodies);

        let id = builder
            .pass("ssao_blur", PassKind::Inline)
            .access(ssao.raw_ao, Access::Sampled)
            .access(ssao.ao, Access::ColorAttachment)
            .build();
        record(id, PassBody::SsaoBlur, &mut bodies);
    }

    let mut forward = builder
        .pass("forward", PassKind::Inline)
        .access(object_transforms, Access::StorageRead);
    if let Some(ssao) = ssao {
        forward = forward.access(ssao.ao, Access::Sampled);
    }
    if let Some(shadows) = shadows {
        forward = forward.access(shadows, Access::Sampled);
    }
    let id = forward
        .access(msaa_hdr, Access::ColorAttachment)
        .access(msaa_depth, Access::DepthAttachment)
        .access(hdr_color, Access::ResolveAttachment)
        .build();
    record(id, PassBody::Forward, &mut bodies);

    // Imported, not created: temporal adaptation is a value that has to outlive
    // the frame, and a transient is `Undefined` at every frame's start by
    // contract. The exposure module owns the allocation; the graph only orders
    // access to it.
    let exposure = builder.import_buffer("exposure");

    let histogram = config.auto_exposure.then(|| {
        let histogram = builder.import_buffer("luminance_histogram");

        let id = builder
            .pass("luminance_histogram", PassKind::Compute)
            .access(hdr_color, Access::Sampled)
            .access(histogram, Access::StorageWrite)
            .build();
        record(id, PassBody::LuminanceHistogram, &mut bodies);

        // One `StorageWrite` covers both halves of what this pass does to each
        // buffer — it reads the bins and zeroes them, reads last frame's adapted
        // luminance and replaces it. Declaring the read separately would be two
        // accesses to one resource in one pass, which the compiler rejects
        // because a pass is a single point in the schedule.
        let id = builder
            .pass("luminance_average", PassKind::Compute)
            .access(histogram, Access::StorageWrite)
            .access(exposure, Access::StorageWrite)
            .build();
        record(id, PassBody::LuminanceAverage, &mut bodies);

        histogram
    });

    let bloom = (config.bloom_mips > 0).then(|| {
        let mips = config.bloom_mips as usize;
        let mut down = [None; MAX_BLOOM_MIPS];
        let mut up = [None; MAX_BLOOM_MIPS];

        // Level `n` is the frame halved `n + 1` times, so level 0 is half the
        // frame. Sized off the frame rather than off the level above it, so a
        // resize moves the whole chain together.
        for level in 0..mips {
            let shift = level as u32 + 1;
            down[level] = Some(builder.create_image(
                BLOOM_DOWN_NAMES[level],
                ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(shift)),
            ));
            if level + 1 < mips {
                up[level] = Some(builder.create_image(
                    BLOOM_UP_NAMES[level],
                    ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(shift)),
                ));
            }
        }

        let id = builder
            .pass("bloom_prefilter", PassKind::Compute)
            .access(hdr_color, Access::Sampled)
            .access(exposure, Access::StorageRead)
            .access(down[0].unwrap(), Access::StorageWrite)
            .build();
        record(id, PassBody::BloomPrefilter, &mut bodies);

        for level in 1..mips {
            let id = builder
                .pass(BLOOM_DOWN_PASS_NAMES[level], PassKind::Compute)
                .access(down[level - 1].unwrap(), Access::Sampled)
                .access(down[level].unwrap(), Access::StorageWrite)
                .build();
            record(id, PassBody::BloomDownsample(level as u32), &mut bodies);
        }

        // Back down the chain, coarsest first. The coarsest step spreads the
        // down chain's last level; every step after it spreads the up-chain
        // level it just produced.
        for level in (0..mips.saturating_sub(1)).rev() {
            let coarse = if level + 2 < mips {
                up[level + 1].unwrap()
            } else {
                down[level + 1].unwrap()
            };
            let id = builder
                .pass(BLOOM_UP_PASS_NAMES[level], PassKind::Compute)
                .access(coarse, Access::Sampled)
                .access(down[level].unwrap(), Access::Sampled)
                .access(up[level].unwrap(), Access::StorageWrite)
                .build();
            record(id, PassBody::BloomUpsample(level as u32), &mut bodies);
        }

        BloomIds {
            mips: config.bloom_mips,
            down,
            up,
        }
    });

    // Reading `exposure` is also what orders this behind the metering: nothing
    // else connects the two, since tonemap and the histogram pass both only read
    // `hdr_color`.
    let mut tonemap = builder
        .pass("tonemap", PassKind::Inline)
        .access(hdr_color, Access::Sampled)
        .access(exposure, Access::StorageRead);
    if let Some(bloom) = bloom {
        tonemap = tonemap.access(bloom.result(), Access::Sampled);
    }
    let id = tonemap
        .access(swapchain_color, Access::ColorAttachment)
        .build();
    record(id, PassBody::Tonemap, &mut bodies);

    if config.overlay {
        let id = builder
            .pass("overlay", PassKind::Raw)
            .access(swapchain_color, Access::ColorAttachment)
            .build();
        record(id, PassBody::Overlay, &mut bodies);
    }

    Ok(Frame {
        graph: compile(builder)?,
        ids: FrameIds {
            object_transforms,
            swapchain_color,
            hdr_color,
            msaa_hdr,
            msaa_depth,
            exposure,
            histogram,
            bloom,
            ssao,
            shadows,
        },
        bodies,
    })
}
