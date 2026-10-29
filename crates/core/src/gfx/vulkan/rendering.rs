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

use glam::Vec3;
use vulkano::format::Format;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;

use super::frame::{FrameIds, PassBody};
use crate::gfx::images::ImageTable;
use crate::gfx::rhi::{
    ColorAttachment, DepthAttachment, LoadOp, RenderTargets, StoreOp, TextureViewId,
};

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
fn cleared(target: TextureViewId, clear: [f32; 4]) -> ColorAttachment {
    ColorAttachment {
        target,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        clear,
        resolve: None,
    }
}

/// A colour attachment whose previous contents do not matter and which is
/// entirely rewritten — the swapchain image, and the shadow atlas whose live
/// tiles `record_atlas` clears itself.
fn overwritten(target: TextureViewId) -> ColorAttachment {
    ColorAttachment {
        target,
        load: LoadOp::DontCare,
        store: StoreOp::Store,
        clear: [0.0; 4],
        resolve: None,
    }
}

/// A colour attachment drawn over what an earlier pass left in it.
fn drawn_over(target: TextureViewId) -> ColorAttachment {
    ColorAttachment {
        target,
        load: LoadOp::Load,
        store: StoreOp::Store,
        clear: [0.0; 4],
        resolve: None,
    }
}

/// A multisampled colour attachment that is cleared, drawn into, resolved into
/// `resolve`, and then discarded.
///
/// `DontCare` on the multisampled image is the point of it: nothing reads the
/// samples after the resolve, so on a tiler they never reach memory at all, and
/// the image carries `TRANSIENT_ATTACHMENT` for that reason.
fn resolved(target: TextureViewId, resolve: TextureViewId, clear: [f32; 4]) -> ColorAttachment {
    ColorAttachment {
        target,
        load: LoadOp::Clear,
        store: StoreOp::DontCare,
        clear,
        resolve: Some(resolve),
    }
}

/// A depth attachment this pass rasterises for itself: cleared to far, tested
/// and written, and discarded at the end because nothing samples it.
fn depth_scratch(target: TextureViewId) -> DepthAttachment {
    DepthAttachment {
        target,
        load: LoadOp::Clear,
        store: StoreOp::DontCare,
        clear: 1.0,
        read_only: false,
    }
}

/// A depth attachment that is cleared, written, and kept for later passes to
/// sample — the geometry prepass's, and each shadow cascade's.
fn depth_written(target: TextureViewId) -> DepthAttachment {
    DepthAttachment {
        target,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        clear: 1.0,
        read_only: false,
    }
}

/// The geometry prepass's depth, borrowed by a later pass that tests against it
/// and never writes it.
///
/// Stored rather than `DontCare`: nothing was written, and `DontCare` would
/// license a driver to leave the depth undefined for the passes that read it
/// afterwards — which is most of them.
fn depth_borrowed(target: TextureViewId) -> DepthAttachment {
    DepthAttachment {
        target,
        load: LoadOp::Load,
        store: StoreOp::Store,
        clear: 0.0,
        read_only: true,
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
    images: &ImageTable,
    swapchain: TextureViewId,
    body: PassBody,
    background: Vec3,
) -> Option<RenderTargets> {
    let color_only = |attachment: ColorAttachment| RenderTargets {
        colors: vec![attachment],
        depth: None,
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
            RenderTargets {
                colors: vec![
                    cleared(images.texture(prepass.normal), [0.5, 0.5, 1.0, 0.0]),
                    cleared(images.texture(prepass.velocity), [0.0; 4]),
                    cleared(images.texture(prepass.material), [0.0; 4]),
                ],
                depth: Some(depth_written(images.texture(prepass.depth))),
            }
        }
        // 1.0 = fully unoccluded, so an untouched pixel darkens nothing.
        PassBody::SsaoResolve => {
            let ssao = ids.ssao.expect("SSAO pass without SSAO resources");
            color_only(cleared(images.texture(ssao.raw_ao), [1.0, 0.0, 0.0, 0.0]))
        }
        PassBody::SsaoBlur => {
            let ssao = ids.ssao.expect("SSAO pass without SSAO resources");
            color_only(cleared(images.texture(ssao.ao), [1.0, 0.0, 0.0, 0.0]))
        }
        // 1.0 = fully lit, for the same reason: the pass writes every pixel, so
        // this only decides what a pixel would be if it somehow did not.
        PassBody::ContactShadows => color_only(cleared(
            images.texture(
                ids.contact_shadows
                    .expect("contact shadow pass without its mask"),
            ),
            [1.0, 0.0, 0.0, 0.0],
        )),
        // Four render passes and four framebuffers, now two independent
        // questions: does this frame resolve, and does it diffuse.
        //
        // The diffusible clear is the feature's mask — the skybox and debug
        // lines write nothing to that attachment, and a zero radius is exactly
        // "nothing scattered here".
        PassBody::Forward => {
            let color = [background.x, background.y, background.z, 1.0];
            let diffusible = [0.0; 4];
            let sss = ids.subsurface;
            match ids.msaa {
                Some(msaa) => {
                    let mut colors = vec![resolved(
                        images.texture(msaa.hdr),
                        images.texture(ids.hdr_color),
                        color,
                    )];
                    if let Some(sss) = sss {
                        colors.push(resolved(
                            images.texture(
                                sss.msaa_diffusible
                                    .expect("a multisampled frame resolves its diffusible"),
                            ),
                            images.texture(sss.diffusible),
                            diffusible,
                        ));
                    }
                    RenderTargets {
                        colors,
                        depth: Some(depth_scratch(images.texture(msaa.depth))),
                    }
                }
                // One sample: no resolve, so `hdr_color` is the colour
                // attachment itself and is stored; and the depth is the
                // prepass's, borrowed for the `EQUAL` test.
                None => {
                    let mut colors = vec![cleared(images.texture(ids.hdr_color), color)];
                    if let Some(sss) = sss {
                        colors.push(cleared(images.texture(sss.diffusible), diffusible));
                    }
                    RenderTargets {
                        colors,
                        depth: Some(depth_borrowed(
                            images.texture(
                                ids.prepass
                                    .expect("a one-sample forward pass borrows the prepass depth")
                                    .depth,
                            ),
                        )),
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
            RenderTargets {
                colors: vec![
                    cleared(images.texture(oit.accum), [0.0; 4]),
                    cleared(images.texture(oit.reveal), [1.0, 0.0, 0.0, 0.0]),
                ],
                depth: Some(depth_borrowed(images.texture(prepass.depth))),
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
            RenderTargets {
                colors: vec![cleared(images.texture(refraction.accum), [0.0; 4])],
                depth: Some(depth_borrowed(images.texture(prepass.depth))),
            }
        }
        // One layer of the cascade array, not the array view: an attachment is a
        // single layer, and handing it the whole array would make every cascade
        // clear and overwrite the same texels without erroring.
        //
        // Entirely re-rendered, so clearing it whole is exactly what it wants: a
        // texel nobody drew into reads as far, and therefore as lit.
        PassBody::ShadowCascade(cascade) => RenderTargets {
            colors: Vec::new(),
            depth: Some(depth_written(images.layer_texture(
                ids.shadows.expect("shadow pass without a shadow image"),
                cascade,
            ))),
        },
        // Not re-rendered whole: it holds up to 64 tiles of which a frame lights
        // a fraction, so it loads `DontCare` and `record_atlas` clears the ones
        // assigned. That difference — one `load_op` — cost a second render pass
        // object before; here it is this arm.
        PassBody::PunctualShadows => RenderTargets {
            colors: Vec::new(),
            depth: Some(DepthAttachment {
                target: images.texture(
                    ids.shadow_atlas
                        .expect("punctual shadow pass without an atlas"),
                ),
                load: LoadOp::DontCare,
                store: StoreOp::Store,
                clear: 1.0,
                read_only: false,
            }),
        },
        // The acquired swapchain image, every pixel of which the tonemap writes.
        PassBody::Tonemap => color_only(overwritten(swapchain)),
        // Cleared to "nothing outlined".
        PassBody::OutlineMask => color_only(cleared(
            images.texture(
                ids.outline_mask
                    .expect("an outline mask pass without its mask"),
            ),
            [0.0; 4],
        )),
        // The tonemapped frame, which the outline and the handles are drawn over.
        PassBody::Outline | PassBody::Gizmo => color_only(drawn_over(swapchain)),
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
