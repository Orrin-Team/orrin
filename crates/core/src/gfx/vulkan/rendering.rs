//! What each pass renders into, without render pass objects or framebuffers.
//!
//! Every graphics pass opens a rendering instance with dynamic rendering, which
//! replaces three things this module now owns between them: each pass's
//! `RenderPass`, its `Framebuffer`, and the positional clear list that had to
//! match the framebuffer's attachment order.
//!
//! Two consequences shape what is here. Load and store ops are decided at record
//! time, so the shadow atlas no longer needs a second render pass differing from
//! the cascades' only in `load_op`. And attachment layouts are stated per
//! attachment, so the read-only prepass depth that `oit.rs`, `refraction.rs` and
//! the one-sample forward pass borrow is one field rather than three hand-built
//! `RenderPassCreateInfo`s.
//!
//! [`VK_KHR_dynamic_rendering`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_dynamic_rendering.html

use std::sync::Arc;

use glam::Vec3;
use vulkano::command_buffer::{
    RenderingAttachmentInfo, RenderingAttachmentResolveInfo, RenderingInfo,
};
use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};

use super::frame::{FrameIds, PassBody};
use super::resources::GraphImages;

/// The formats a pipeline will be used with.
///
/// The whole of what a pipeline needs to know about its target. Each pass module
/// builds its own from the format constants it already declares, so a format
/// still has exactly one definition.
pub(super) fn pipeline_info(
    color: &[Format],
    depth: Option<Format>,
) -> PipelineRenderingCreateInfo {
    PipelineRenderingCreateInfo {
        color_attachment_formats: color.iter().copied().map(Some).collect(),
        depth_attachment_format: depth,
        ..Default::default()
    }
}

/// A colour attachment that is cleared, drawn into, and kept.
fn cleared(view: Arc<ImageView>, clear: ClearValue) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_value: Some(clear),
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// A colour attachment whose previous contents do not matter and which is
/// entirely rewritten — the swapchain image, and the shadow atlas whose live
/// tiles `record_atlas` clears itself.
fn overwritten(view: Arc<ImageView>) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::DontCare,
        store_op: AttachmentStoreOp::Store,
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// A colour attachment drawn over what an earlier pass left in it.
fn drawn_over(view: Arc<ImageView>) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Load,
        store_op: AttachmentStoreOp::Store,
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// A multisampled colour attachment that is cleared, drawn into, resolved into
/// `resolve`, and then discarded.
///
/// `DontCare` on the multisampled image is the point of it: nothing reads the
/// samples after the resolve, so on a tiler they never reach memory at all, and
/// the image carries `TRANSIENT_ATTACHMENT` for that reason.
fn resolved(
    view: Arc<ImageView>,
    resolve: Arc<ImageView>,
    clear: ClearValue,
) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::DontCare,
        clear_value: Some(clear),
        resolve_info: Some(RenderingAttachmentResolveInfo::image_view(resolve)),
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// A depth attachment this pass rasterises for itself: cleared to far, tested
/// and written, and discarded at the end because nothing samples it.
fn depth_scratch(view: Arc<ImageView>) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::DontCare,
        clear_value: Some(1.0.into()),
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// A depth attachment that is cleared, written, and kept for later passes to
/// sample — the geometry prepass's, and each shadow cascade's.
fn depth_written(view: Arc<ImageView>) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_value: Some(1.0.into()),
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// The geometry prepass's depth, borrowed by a later pass that tests against it
/// and never writes it.
///
/// `DepthStencilReadOnlyOptimal` is the layout the barrier plan says it is in,
/// so stating it here keeps this attachment from transitioning an image nobody
/// wrote. Stored rather than `DontCare`: nothing was written, and `DontCare`
/// would license a driver to leave the depth undefined for the passes that read
/// it afterwards — which is most of them.
fn depth_borrowed(view: Arc<ImageView>) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        image_layout: vulkano::image::ImageLayout::DepthStencilReadOnlyOptimal,
        load_op: AttachmentLoadOp::Load,
        store_op: AttachmentStoreOp::Store,
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// What one graphics pass renders into, or `None` for a compute pass or one
/// recorded outside the executor.
///
/// Replaces `PassFramebuffers::build` and `clear_values` together: an attachment
/// carries its own clear, so there is no positional list to keep in step with a
/// framebuffer, and no telling four forward shapes apart by counting
/// attachments.
pub(super) fn rendering_info(
    ids: &FrameIds,
    images: &GraphImages,
    swapchain: &Arc<ImageView>,
    body: PassBody,
    background: Vec3,
) -> Option<RenderingInfo> {
    let color_only = |attachment: RenderingAttachmentInfo| RenderingInfo {
        color_attachments: vec![Some(attachment)],
        ..Default::default()
    };

    Some(match body {
        // Flat +Z in the normal buffer, no motion in the velocity buffer, far in
        // depth. Zero velocity is what the sky and any unrasterised pixel are
        // left with, and the resolve reads that as "reproject with the camera
        // alone" rather than as a stationary surface.
        //
        // A black `f0` at zero roughness in the material target, so a pixel
        // nothing rasterised reflects nothing rather than mirroring the sky at
        // whatever the previous contents implied.
        PassBody::GeometryPrepass => {
            let prepass = ids.prepass.expect("prepass without its resources");
            RenderingInfo {
                color_attachments: vec![
                    Some(cleared(
                        images.view(prepass.normal),
                        [0.5, 0.5, 1.0, 0.0].into(),
                    )),
                    Some(cleared(
                        images.view(prepass.velocity),
                        [0.0, 0.0, 0.0, 0.0].into(),
                    )),
                    Some(cleared(
                        images.view(prepass.material),
                        [0.0, 0.0, 0.0, 0.0].into(),
                    )),
                ],
                depth_attachment: Some(depth_written(images.view(prepass.depth))),
                ..Default::default()
            }
        }
        // 1.0 = fully unoccluded, so an untouched pixel darkens nothing.
        PassBody::SsaoResolve => {
            let ssao = ids.ssao.expect("SSAO pass without SSAO resources");
            color_only(cleared(
                images.view(ssao.raw_ao),
                [1.0, 0.0, 0.0, 0.0].into(),
            ))
        }
        PassBody::SsaoBlur => {
            let ssao = ids.ssao.expect("SSAO pass without SSAO resources");
            color_only(cleared(images.view(ssao.ao), [1.0, 0.0, 0.0, 0.0].into()))
        }
        // 1.0 = fully lit, for the same reason: the pass writes every pixel, so
        // this only decides what a pixel would be if it somehow did not.
        PassBody::ContactShadows => color_only(cleared(
            images.view(
                ids.contact_shadows
                    .expect("contact shadow pass without its mask"),
            ),
            [1.0, 0.0, 0.0, 0.0].into(),
        )),
        // Four render passes and four framebuffers, now two independent
        // questions: does this frame resolve, and does it diffuse.
        //
        // The diffusible clear is the feature's mask — the skybox and debug
        // lines write nothing to that attachment, and a zero radius is exactly
        // "nothing scattered here".
        PassBody::Forward => {
            let color = ClearValue::from([background.x, background.y, background.z, 1.0]);
            let diffusible = ClearValue::from([0.0, 0.0, 0.0, 0.0]);
            let sss = ids.subsurface;
            match ids.msaa {
                Some(msaa) => {
                    let mut color_attachments = vec![Some(resolved(
                        images.view(msaa.hdr),
                        images.view(ids.hdr_color),
                        color,
                    ))];
                    if let Some(sss) = sss {
                        color_attachments.push(Some(resolved(
                            images.view(
                                sss.msaa_diffusible
                                    .expect("a multisampled frame resolves its diffusible"),
                            ),
                            images.view(sss.diffusible),
                            diffusible,
                        )));
                    }
                    RenderingInfo {
                        color_attachments,
                        depth_attachment: Some(depth_scratch(images.view(msaa.depth))),
                        ..Default::default()
                    }
                }
                // One sample: no resolve, so `hdr_color` is the colour
                // attachment itself and is stored; and the depth is the
                // prepass's, borrowed for the `EQUAL` test.
                None => {
                    let mut color_attachments =
                        vec![Some(cleared(images.view(ids.hdr_color), color))];
                    if let Some(sss) = sss {
                        color_attachments
                            .push(Some(cleared(images.view(sss.diffusible), diffusible)));
                    }
                    RenderingInfo {
                        color_attachments,
                        depth_attachment: Some(depth_borrowed(
                            images.view(
                                ids.prepass
                                    .expect("a one-sample forward pass borrows the prepass depth")
                                    .depth,
                            ),
                        )),
                        ..Default::default()
                    }
                }
            }
        }
        // Nothing accumulated, and everything revealed. The second is the one
        // that matters: revealage is a running *product* of `1 - alpha`, so a
        // clear of zero would hide the opaque frame everywhere rather than
        // nowhere.
        PassBody::OitAccumulate => {
            let oit = ids
                .transparency
                .expect("transparency pass without its targets");
            let prepass = ids
                .prepass
                .expect("the graph scheduled transparency with no prepass");
            RenderingInfo {
                color_attachments: vec![
                    Some(cleared(images.view(oit.accum), [0.0, 0.0, 0.0, 0.0].into())),
                    Some(cleared(
                        images.view(oit.reveal),
                        [1.0, 0.0, 0.0, 0.0].into(),
                    )),
                ],
                depth_attachment: Some(depth_borrowed(images.view(prepass.depth))),
                ..Default::default()
            }
        }
        // Nothing covered, so nothing replaced: the composite reads this as the
        // frame's own colour showing through in full. Zero rather than one is
        // what makes it an `over` of nothing instead of a black wall.
        PassBody::RefractionDraw => {
            let refraction = ids.refraction.expect("refraction pass without its targets");
            let prepass = ids
                .prepass
                .expect("the graph scheduled refraction with no prepass");
            RenderingInfo {
                color_attachments: vec![Some(cleared(
                    images.view(refraction.accum),
                    [0.0, 0.0, 0.0, 0.0].into(),
                ))],
                depth_attachment: Some(depth_borrowed(images.view(prepass.depth))),
                ..Default::default()
            }
        }
        // One layer of the cascade array, not the array view: an attachment is a
        // single layer, and handing it the whole array would make every cascade
        // clear and overwrite the same texels without erroring.
        //
        // Entirely re-rendered, so clearing it whole is exactly what it wants: a
        // texel nobody drew into reads as far, and therefore as lit.
        PassBody::ShadowCascade(cascade) => RenderingInfo {
            depth_attachment: Some(depth_written(images.layer_view(
                ids.shadows.expect("shadow pass without a shadow image"),
                cascade,
            ))),
            ..Default::default()
        },
        // Not re-rendered whole: it holds up to 64 tiles of which a frame lights
        // a fraction, so it loads `DontCare` and `record_atlas` clears the ones
        // assigned. That difference — one `load_op` — cost a second render pass
        // object before; here it is this arm.
        PassBody::PunctualShadows => RenderingInfo {
            depth_attachment: Some(RenderingAttachmentInfo {
                load_op: AttachmentLoadOp::DontCare,
                store_op: AttachmentStoreOp::Store,
                ..RenderingAttachmentInfo::image_view(
                    images.view(
                        ids.shadow_atlas
                            .expect("punctual shadow pass without an atlas"),
                    ),
                )
            }),
            ..Default::default()
        },
        // The acquired swapchain image, every pixel of which the tonemap writes.
        PassBody::Tonemap => color_only(overwritten(swapchain.clone())),
        // The tonemapped frame, which the handles are drawn over.
        PassBody::Gizmo => color_only(drawn_over(swapchain.clone())),
        // Dispatches, and the overlay, which records its own command buffer.
        PassBody::Overlay
        | PassBody::OcclusionHiz
        | PassBody::SsrHiz
        | PassBody::SsrSource
        | PassBody::SsrTrace
        | PassBody::SsrResolve
        | PassBody::FogScatter
        | PassBody::FogIntegrate
        | PassBody::SubsurfaceBlurHorizontal
        | PassBody::SubsurfaceBlurVertical
        | PassBody::SubsurfaceComposite
        | PassBody::OitComposite
        | PassBody::RefractionScene
        | PassBody::RefractionComposite
        | PassBody::TaaResolve
        | PassBody::DofPrefilter
        | PassBody::DofTileMax
        | PassBody::DofGather
        | PassBody::DofComposite
        | PassBody::MotionBlurTileMax
        | PassBody::MotionBlurNeighbourMax
        | PassBody::MotionBlurGather
        | PassBody::LuminanceHistogram
        | PassBody::LuminanceAverage
        | PassBody::BloomPrefilter
        | PassBody::BloomDownsample(_)
        | PassBody::BloomUpsample(_)
        | PassBody::CullReset
        | PassBody::Cull => return None,
    })
}
