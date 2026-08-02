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
    pub ssao: Option<SsaoIds>,
    pub shadows: Option<ResourceId>,
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

    let id = builder
        .pass("tonemap", PassKind::Inline)
        .access(hdr_color, Access::Sampled)
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
            ssao,
            shadows,
        },
        bodies,
    })
}
