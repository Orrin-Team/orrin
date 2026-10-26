//! The synchronisation tripwire the render graph exists to make possible (#7).
//!
//! A wrong barrier is the worst bug the renderer can have: it produces correct
//! pixels on the machine that wrote it and a race on someone else's scheduler,
//! and it survives review because nobody reads a barrier and sees it is one flag
//! too loose. The graph derives them instead, which means the derivation itself
//! can be pinned — this file renders the reference frame's plan to text and
//! compares it against a checked-in baseline.
//!
//! **What a failure means.** The plan changed. That is legitimate whenever the
//! frame's structure changed on purpose, and a regression whenever it didn't:
//! a barrier that quietly loosens, a transition that disappears, a pass that
//! moves. Read the diff before regenerating — the baseline is the review, so
//! updating it without reading it spends the whole mechanism.
//!
//! Regenerate with `ORRIN_UPDATE_GOLDEN=1 cargo test -p orrin-core --test
//! render_graph`, and put the diff in the commit.
//!
//! It runs on a GPU-less CI runner because compiling a graph takes no `Device`,
//! which is the property that made the derivation worth doing this way.

use std::fs;
use std::path::PathBuf;

use orrin_core::gfx::vulkan::frame::{FrameConfig, declare};
use vulkano::format::Format;

/// The swapchain format the baseline is written against. Real ones vary by
/// surface; the plan does not depend on it, and pinning it keeps the file from
/// depending on whoever regenerated it.
const COLOR_FORMAT: Format = Format::B8G8R8A8_SRGB;

/// The cascade resolution the baseline is written against. It changes what the
/// images are sized to, not what the plan says, so pinning it keeps the file
/// from depending on whoever regenerated it.
const SHADOW_RESOLUTION: u32 = 2048;

/// The bloom chain length the baseline is written against — what
/// `bloom::mip_count` yields for any frame from 1080p up. Pinned for the same
/// reason as the cascade resolution: the plan should not depend on the window
/// whoever regenerated it happened to have open.
const BLOOM_MIPS: u8 = 6;

fn configs() -> Vec<(&'static str, FrameConfig)> {
    vec![
        // The multisampled path, which is no longer the default and so needs
        // saying explicitly. It is the one shape where the forward pass writes
        // depth: everything else in this file borrows the geometry prepass's
        // read-only, which is what the `EQUAL` test is for. Both variants are
        // here because the pair is exactly what a regression would collapse —
        // an MSAA frame that stopped resolving, or a one-sample frame that
        // started declaring a depth *write* and quietly took the prepass's
        // single-writer guarantee with it.
        (
            "editor frame, MSAA on",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: true,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // MSAA with the diffusion on, which is the five-attachment forward pass:
        // two multisampled colour targets and two resolves. The one-sample frame
        // with diffusion has three attachments and so shares a count with the
        // multisampled frame without it — see `clear_values`, which is why that
        // function reads the sample count rather than counting.
        (
            "editor frame, MSAA and subsurface on",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: true,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: true,
                transparency: false,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        (
            "editor frame, TAA and SSAO on",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // Four cascades write four layers of one image, which the graph tracks
        // as one resource — so consecutive cascades are separated by a
        // write-after-write barrier with no layout transition. That
        // serialisation is the cost of not tracking subresources in v1, and it
        // is in the baseline so that removing it later is a visible diff rather
        // than a silent one.
        (
            "editor frame, four cascades",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // SSAO off is a different graph, not a flag read at record time: the
        // three passes are never registered and the forward pass never declares
        // the read. Both shapes are baselined because both ship. Bloom is off
        // here too, which is the shape where the tonemap pass declares no bloom
        // input and samples a 1x1 black view instead.
        (
            "editor frame, SSAO off",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: false,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: 0,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // The shape that proves the prepass belongs to the frame rather than to
        // SSAO: nothing reads its normal target here, and it still runs, because
        // TAA needs the motion vectors that come off the same rasterisation.
        (
            "editor frame, TAA without SSAO",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: false,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // Metering off is the other shape that ships. Worth baselining for one
        // thing in particular: the tonemap pass still declares its read of
        // `exposure`, an import nothing writes in this configuration. That is
        // legal where reading an unwritten *transient* is not, and it is what
        // lets one tonemap pipeline serve both modes.
        (
            "editor frame, auto exposure off",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: false,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // A window too small for a real chain still gets one level, and that
        // shape has no upsample pass in it at all — so the tonemap pass
        // composites the down chain directly. It is the case where `result()`
        // takes its other branch, and nothing else in the suite reaches it.
        (
            "editor frame, one bloom level",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: 1,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // The froxel fog, with the cascades it exists to read. Worth baselining
        // for two things the rest of the suite cannot show. The scatter pass is
        // an *import* read and written by one pass, which is the shape the
        // reprojection needs and the only one in the frame besides TAA's. And
        // the fog is the one effect whose result a *geometry* pass consumes: the
        // forward pass declares the volume as an input, so the two dispatches
        // land ahead of shading rather than in the composite chain after it, and
        // a diff that moved them would be a diff in what the geometry can see.
        (
            "editor frame, volumetric fog with cascades",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: true,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: true,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // The whole optical chain at once: lens, then shutter, then sensor.
        // Worth baselining as one shape rather than two because the passes are
        // chained through `scene_color` — each rebinds it for the next — so the
        // barriers between them are the thing that would break if the order
        // were ever rearranged.
        (
            "editor frame, depth of field and motion blur",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: true,
                dof: true,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // The other shape that proves the prepass belongs to the frame rather
        // than to any one consumer: SSAO and TAA are both off here, and it still
        // runs, because motion blur needs the velocities and depth of field
        // needs the depth. It is also the only configuration where the chain
        // reads the forward pass's resolve directly instead of a TAA output.
        (
            "editor frame, depth of field and motion blur without TAA",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: false,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: true,
                motion_blur: true,
                dof: true,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // Reflections are the first thing to sit *between* shading and the
        // temporal resolve, so this shape is what pins that the resolve reads
        // the composite's output rather than the forward pass's own. It is also
        // where the depth pyramid appears: one image, one pass writing every
        // level, and the trace sampling it afterwards.
        (
            "editor frame, screen-space reflections",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: true,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // And the same without a resolve behind it, which is the shape where
        // the composite's output *is* what the optical chain and the tonemap
        // read.
        (
            "editor frame, reflections without TAA",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: false,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: true,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // What actually ships in the editor: the march sits between the prepass
        // and shading, and the forward pass declares a read of its mask beside
        // the one it declares of the AO. Two single-channel screen-space inputs
        // to one pass is the shape that would break if either were ever moved
        // behind it.
        (
            "editor frame, contact shadows",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: true,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // And the march on its own, which is the shape that proves it keeps the
        // geometry prepass in the frame by itself: nothing else here reads
        // depth, normals or motion, and the prepass still runs.
        (
            "editor frame, contact shadows alone",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: false,
                ssao_half_res: false,
                contact_shadows: true,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: false,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: 0,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // The punctual atlas: one pass however many lights cast, because a tile
        // is a viewport rather than a node. That is the whole reason it is an
        // atlas, so the plan is where it should be pinned — six faces of eight
        // lights appearing here as forty-eight passes would be the regression.
        (
            "editor frame, punctual shadow atlas",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: true,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 4096,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // And the atlas with no cascades, which is an ordinary scene: an indoor
        // one, lit by lamps with no sun. The sun and the punctual lights are
        // independent, so the frame has to hold together with either half
        // missing.
        (
            "editor frame, punctual shadows without cascades",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: false,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 2048,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // What ships: transparency accumulates after the reflections and before
        // the temporal resolve, and it attaches the prepass depth read-only
        // while five other passes sample the same image. Depth ping-ponging
        // between `DepthStencilReadOnlyOptimal` and `ShaderReadOnlyOptimal` is
        // the visible cost of that, and it belongs in the baseline so that
        // narrowing it later is a diff rather than a claim.
        (
            "editor frame, transparency",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: true,
                ssr: true,
                subsurface: false,
                transparency: true,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: true,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // And transparency on its own, which is the shape that proves it keeps
        // the geometry prepass in the frame by itself: nothing else here wants
        // depth, normals or motion, and the prepass still runs — because the
        // accumulation has nothing else to depth-test against.
        (
            "editor frame, transparency alone",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: false,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: true,
                refraction: false,
                taa: false,
                auto_exposure: false,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: 0,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // Both non-opaque queues at once, which is the shape that pins the
        // order between them: the refraction pass reads the frame the
        // transparency composite produced, so a change that moved it ahead of
        // that composite would show up here as glass refracting a frame with no
        // blended geometry in it.
        (
            "editor frame, transparency and refraction",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: true,
                ssr: true,
                subsurface: false,
                transparency: true,
                refraction: true,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: true,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // And refraction on its own, which proves it keeps the geometry prepass
        // in the frame by itself for the same reason transparency does: the draw
        // has nothing else to depth-test against.
        (
            "editor frame, refraction alone",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: false,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: true,
                taa: false,
                auto_exposure: false,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: 0,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // Subsurface scattering is the first thing to change the *forward pass's
        // own declarations* rather than only adding passes after it: a second
        // colour attachment and a second resolve. So this shape pins two things
        // the others cannot — that the extra attachment is declared, and that
        // everything downstream reads the composite's output instead of the
        // forward pass's resolve. Reflections are on deliberately: a mirror beside
        // a scattering face should show the face as the frame will, which means
        // the trace has to be handed the diffused colour.
        (
            "editor frame, subsurface scattering",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: true,
                ssr: true,
                subsurface: true,
                transparency: false,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // And on its own, which is the shape that proves the diffusion keeps the
        // geometry prepass alive by itself — the blur weights its taps by the
        // depth that pass writes, so with nothing else in the frame wanting depth
        // the prepass still has to run.
        (
            "editor frame, subsurface scattering alone",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: false,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: true,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: false,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: 0,
                overlay: true,
                shadow_cascades: 0,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        (
            "headless frame, no overlay",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: false,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: false,
                refraction: false,
                taa: false,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: false,
                shadow_cascades: 2,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: false,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // The split, on the frame the editor actually runs. Every configuration
        // above is here with one queue, so the pair is what says what the split
        // changed: the pass order, the barriers and the layouts are the same
        // text, and what is added is the three segment headers and the list of
        // resources that had to stop being exclusive to one queue family.
        // The compute-culled frame. Its whole reason for being in the baseline
        // is the pair of buffer barriers a CPU-culled frame has none of: the
        // write-after-write between the two dispatches, and the storage write
        // that every opaque geometry pass then waits for at the draw-indirect
        // stage. Cascades and the atlas are on, because those are the views the
        // dispatch exists to cull for.
        (
            "editor frame, GPU culling",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: true,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: true,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 4096,
                async_compute: false,
                gpu_culling: true,
                occlusion_culling: false,
            },
        ),
        // The same frame with the depth test on, because what it adds is a
        // second reader of the prepass depth and a *texture the cull samples* —
        // and the pyramid the cull samples is last frame's, so the plan has to
        // show the build after the cull rather than before it. A plan that put
        // them the other way round would be reading a pyramid of this frame's
        // depth to decide what this frame draws, which is the cycle the two
        // resources exist to avoid.
        (
            "editor frame, GPU culling with occlusion",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: true,
                contact_shadows: false,
                ssr: false,
                subsurface: false,
                transparency: true,
                refraction: false,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 4096,
                async_compute: false,
                gpu_culling: true,
                occlusion_culling: true,
            },
        ),
        (
            "editor frame, async compute",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: true,
                contact_shadows: true,
                ssr: false,
                subsurface: false,
                transparency: true,
                refraction: true,
                taa: true,
                auto_exposure: true,
                motion_blur: false,
                dof: false,
                volumetric_fog: false,
                bloom_mips: BLOOM_MIPS,
                overlay: true,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 0,
                async_compute: true,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
        // And the frame the split is *for*: every optical stage on, which is
        // where two thirds of the GPU time is dispatches. The interleaved ones —
        // the depth pyramid, the fog grid, the transparency and refraction
        // composites — must stay on the graphics queue here, because each has a
        // draw after it; only the run from the last draw to the tonemap moves.
        (
            "editor frame, async compute with every optical stage",
            FrameConfig {
                color_format: COLOR_FORMAT,
                msaa: false,
                ssao: true,
                ssao_half_res: true,
                contact_shadows: true,
                ssr: true,
                subsurface: true,
                transparency: true,
                refraction: true,
                taa: true,
                auto_exposure: true,
                motion_blur: true,
                dof: true,
                volumetric_fog: true,
                bloom_mips: BLOOM_MIPS,
                overlay: false,
                shadow_cascades: 4,
                shadow_resolution: SHADOW_RESOLUTION,
                shadow_atlas: 2048,
                async_compute: true,
                gpu_culling: false,
                occlusion_culling: false,
            },
        ),
    ]
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/frame_graph.txt")
}

fn render() -> String {
    let mut out = String::new();
    for (label, config) in configs() {
        let frame = declare(config).expect("the reference frame must compile");
        out.push_str(&format!("=== {label} ===\n{}\n", frame.graph));
    }
    out
}

#[test]
fn the_derived_barrier_sequence_matches_the_baseline() {
    let actual = render();
    let path = golden_path();

    if std::env::var_os("ORRIN_UPDATE_GOLDEN").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &actual).unwrap();
        return;
    }

    let expected = fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(
        actual,
        expected,
        "the frame's derived barrier plan no longer matches {}.\n\
         If the frame's structure changed on purpose, regenerate with \
         ORRIN_UPDATE_GOLDEN=1 and commit the diff. If it did not, this is a \
         synchronisation regression: a barrier has loosened or a transition has \
         gone missing.",
        path.display(),
    );
}

/// A pass's declarations and the engine code that runs it are looked up by the
/// same index, so a registration that forgets its body would silently run the
/// wrong pass.
#[test]
fn every_declared_pass_has_a_body() {
    for (label, config) in configs() {
        let frame = declare(config).unwrap();
        assert_eq!(
            frame.bodies.len(),
            frame.graph.pass_count(),
            "{label}: {} passes declared but {} bodies registered",
            frame.graph.pass_count(),
            frame.bodies.len(),
        );
    }
}

/// Nothing in the shipped frame is dead. A culled pass is a pass that recorded
/// work no one reads, which is worth noticing the moment it appears rather than
/// when someone profiles it.
#[test]
fn the_reference_frame_culls_nothing() {
    for (label, config) in configs() {
        let frame = declare(config).unwrap();
        let culled: Vec<&str> = frame
            .graph
            .culled()
            .iter()
            .map(|&id| frame.graph.pass_name(id))
            .collect();
        assert!(culled.is_empty(), "{label}: culled {culled:?}");
    }
}

/// A one-cascade frame is the case that catches a view type derived from the
/// layer count rather than from the declaration: an image of exactly one array
/// layer looks like a plain 2D image to that heuristic, while the forward
/// pipeline's sampler is compiled as a `texture2DArray` regardless. It renders
/// on four cascades and fails on one, so the setting that provokes it is the
/// one nobody drags to on purpose.
#[test]
fn a_single_cascade_map_is_still_declared_as_an_array() {
    for count in 1..=4u8 {
        let frame = declare(FrameConfig {
            color_format: COLOR_FORMAT,
            msaa: false,
            ssao: true,
            ssao_half_res: false,
            contact_shadows: false,
            ssr: false,
            subsurface: false,
            transparency: false,
            refraction: false,
            taa: false,
            auto_exposure: true,
            motion_blur: false,
            dof: false,
            volumetric_fog: false,
            bloom_mips: BLOOM_MIPS,
            overlay: true,
            shadow_cascades: count,
            shadow_resolution: SHADOW_RESOLUTION,
            shadow_atlas: 0,
            async_compute: false,
            gpu_culling: false,
            occlusion_culling: false,
        })
        .unwrap();

        let (_, image) = frame
            .graph
            .transient_images()
            .find(|(id, _)| frame.graph.resource_name(*id) == "shadow_cascades")
            .unwrap_or_else(|| panic!("{count} cascades declared no shadow map"));

        assert_eq!(
            image.desc.array_layers,
            Some(u32::from(count)),
            "{count} cascades must declare a {count}-layer array, not a plain image",
        );
    }
}

/// TAA ping-pongs two allocations, so the image this frame writes as `taa_color`
/// is the one next frame reads as `taa_history`. That only works if the frame
/// leaves `taa_color` in exactly the layout `taa_history` is declared to enter
/// in — otherwise every frame after the first samples an image the plan says is
/// in some other layout, and the resulting read is undefined on hardware that
/// takes the declaration seriously.
///
/// Nothing in the compiler enforces the pairing; it is a property of how the
/// executor binds the two, so it is asserted here rather than derived.
#[test]
fn the_taa_history_leaves_the_frame_where_the_next_one_expects_it() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: true,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: false,
        refraction: false,
        taa: true,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    })
    .unwrap();

    let plan = format!("{}", frame.graph);
    assert!(
        plan.contains("taa_color General->ShaderReadOnlyOptimal"),
        "the resolve's output must end the frame sampled, not left in General:\n{plan}",
    );
    // An import already in its exit layout needs no closing transition, so the
    // absence of one is the assertion: a `taa_color` line among the final
    // barriers would mean the frame is handing the next one a layout it does not
    // expect.
    let closing: Vec<_> = frame
        .graph
        .final_barriers()
        .iter()
        .filter(|barrier| {
            let name = frame.graph.resource_name(barrier.resource);
            name == "taa_color" || name == "taa_history"
        })
        .collect();
    assert!(closing.is_empty(), "{closing:?}");
}

/// The froxel fog ping-pongs two volumes exactly as TAA ping-pongs two images,
/// so the same round-trip has to hold: the allocation this frame writes as
/// `fog_scatter` is the one next frame's scatter pass samples as its history,
/// and the frame therefore has to leave it in the layout it is declared to enter
/// in.
///
/// Nothing in the compiler can check it — the pairing is a property of how the
/// executor binds the pair — which is why it is asserted here.
#[test]
fn the_fog_history_leaves_the_frame_where_the_next_one_expects_it() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: true,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: false,
        refraction: false,
        taa: false,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        volumetric_fog: true,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 4,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    })
    .unwrap();

    let plan = format!("{}", frame.graph);
    // The integration reads it, which is what puts it back into the layout the
    // next frame's history binding expects — so this transition is load-bearing
    // twice over and not merely an ordering artefact.
    assert!(
        plan.contains("fog_scatter General->ShaderReadOnlyOptimal"),
        "the scatter volume must end the frame sampled, not left in General:\n{plan}",
    );
    let closing: Vec<_> = frame
        .graph
        .final_barriers()
        .iter()
        .filter(|barrier| frame.graph.resource_name(barrier.resource) == "fog_scatter")
        .collect();
    assert!(closing.is_empty(), "{closing:?}");
}

/// The fog is the one effect here whose output a *geometry* pass consumes, so
/// both dispatches have to finish before anything shades — and after the
/// cascades, which the scatter pass reads to put the shadow maps into the air.
///
/// Both halves come out of what `declare` says the passes access rather than out
/// of where they were written, which is exactly why a test is worth having: the
/// day the forward pass stops declaring the volume, the fog would still compile
/// and would silently be read a frame late.
#[test]
fn the_fog_volume_is_built_after_the_cascades_and_before_anything_shades() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: true,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: true,
        refraction: true,
        taa: false,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        volumetric_fog: true,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 4,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    })
    .unwrap();

    let order: Vec<&str> = frame
        .graph
        .order()
        .iter()
        .map(|&pass| frame.graph.pass_name(pass))
        .collect();
    let at = |name: &str| {
        order
            .iter()
            .position(|&pass| pass == name)
            .unwrap_or_else(|| panic!("{name} was not scheduled:\n{order:?}"))
    };

    assert!(at("shadow_cascade_3") < at("fog_scatter"), "{order:?}");
    assert!(at("fog_scatter") < at("fog_integrate"), "{order:?}");
    for shading in ["forward", "oit_accumulate", "refraction_draw"] {
        assert!(at("fog_integrate") < at(shading), "{shading}: {order:?}");
    }
}

/// Transparency's two nodes have to land between the reflections and the
/// temporal resolve, and neither half is a preference.
///
/// After the reflections, because the prepass records only opaque surfaces: a
/// ray traces the depth and normals of the world *behind* the glass, so
/// compositing first would put the glass into a source the trace then reflects
/// as though it were a wall. Before the resolve, because the accumulation
/// rasterises with the frame's jitter like everything else, and a subpixel
/// offset nothing averages is a shimmer rather than an antialiased edge.
///
/// `compile` knows none of that — the placement is a consequence of where
/// `declare` rebinds `shaded`, so moving that block would silently reorder the
/// frame. Asserted against the derived schedule for the reason the optical
/// chain's order is.
#[test]
fn transparency_composites_after_the_reflections_and_before_the_resolve() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: true,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: true,
        subsurface: false,
        transparency: true,
        refraction: false,
        taa: true,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    })
    .unwrap();

    let order: Vec<&str> = frame
        .graph
        .order()
        .iter()
        .map(|&id| frame.graph.pass_name(id))
        .collect();
    let at = |name: &str| {
        order
            .iter()
            .position(|&pass| pass == name)
            .unwrap_or_else(|| panic!("{name} is not in the frame: {order:?}"))
    };

    assert!(at("oit_accumulate") < at("oit_composite"), "{order:?}");
    assert!(at("ssr_resolve") < at("oit_composite"), "{order:?}");
    assert!(at("oit_composite") < at("taa_resolve"), "{order:?}");
}

/// The transform gizmo is editor chrome: drawn over the tonemapped frame, so
/// its colours are exact and nothing in the scene can stand in front of a
/// handle, and under the editor's own UI. A frame without the editor has none.
#[test]
fn the_gizmo_draws_between_the_tonemap_and_the_overlay() {
    let config = |overlay, async_compute| FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: true,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: false,
        refraction: false,
        taa: true,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: BLOOM_MIPS,
        overlay,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute,
        gpu_culling: false,
        occlusion_culling: false,
    };
    for async_compute in [false, true] {
        let frame = declare(config(true, async_compute)).unwrap();
        let order: Vec<&str> = frame
            .graph
            .order()
            .iter()
            .map(|&id| frame.graph.pass_name(id))
            .collect();
        let at = |name: &str| {
            order
                .iter()
                .position(|&pass| pass == name)
                .unwrap_or_else(|| panic!("{name} is not in the frame: {order:?}"))
        };
        assert!(at("tonemap") < at("gizmo"), "{order:?}");
        assert!(at("gizmo") < at("overlay"), "{order:?}");
    }

    let headless = declare(config(false, false)).unwrap();
    let names: Vec<&str> = headless
        .graph
        .order()
        .iter()
        .map(|&id| headless.graph.pass_name(id))
        .collect();
    assert!(!names.contains(&"gizmo"), "{names:?}");
}

/// The accumulation pass *attaches* the prepass depth rather than sampling it,
/// and read-only at that. Both halves are load-bearing.
///
/// Attached, because a fixed-function depth test is what makes a transparent
/// surface disappear behind a wall. Read-only, because the prepass must stay the
/// image's only writer — declaring a writing depth attachment would put this
/// pass ahead of every other reader of the same depth, and would fire the
/// write-after-read assertion the schedule is built to make impossible.
///
/// It also has to stay a depth *attachment* and not become a second sampled
/// read: `oit.rs` builds its render pass by hand precisely so the attachment
/// reference says `DepthStencilReadOnlyOptimal`, and that hand-written render
/// pass is only correct while this declaration is what it says it is.
#[test]
fn transparency_attaches_the_prepass_depth_read_only() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: false,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: true,
        refraction: false,
        taa: false,
        auto_exposure: false,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: 0,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    })
    .unwrap();

    let plan = format!("{}", frame.graph);
    assert!(
        plan.contains("prepass_depth DepthAttachmentRead"),
        "the accumulation must attach the prepass depth read-only:\n{plan}",
    );
    // Nothing but the prepass may write it, and the plan says so by listing
    // exactly one `DepthAttachment` line for it.
    assert_eq!(
        plan.matches("prepass_depth DepthAttachment\n").count(),
        1,
        "the geometry prepass must remain the only writer of the depth:\n{plan}",
    );
}

/// The forward pass declares a second colour attachment and a second resolve
/// exactly when the diffusion runs, and nothing downstream reads the resolve it
/// withheld the diffusible light from.
///
/// Both halves are load-bearing, and neither is checkable any other way.
///
/// The attachment, because it is what tells `PassFramebuffers` and the graph the
/// same story. The framebuffer's attachment list and the render pass it is built
/// against are assembled by hand in `resources.rs`; the graph derives the barriers
/// from these declarations. A declaration that went missing would leave the
/// framebuffer binding an image the plan says nothing about — an image created
/// without `COLOR_ATTACHMENT` usage, which is a startup panic, and if it were not,
/// an untracked write.
///
/// The chaining, because the composite is the *only* thing that puts the
/// diffusible radiance back. `forward_sss.frag` routes it out of `hdr_color`
/// deliberately, so any pass still reading `hdr_color` after the composite is
/// reading a frame with the diffuse half of every scattering surface missing —
/// which looks like a lighting bug, not like a mis-wired graph.
#[test]
fn the_diffusion_replaces_the_frame_the_forward_pass_withheld_light_from() {
    let base = FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: false,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: true,
        transparency: false,
        refraction: false,
        taa: true,
        auto_exposure: false,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: 0,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    };

    let plan = format!("{}", declare(base).unwrap().graph);
    // At one sample there is nothing to resolve from: the forward pass writes
    // the diffusible target directly, and declaring it is the whole requirement.
    assert!(
        plan.contains("subsurface_diffusible ColorAttachment"),
        "the forward pass must declare the second target:\n{plan}",
    );
    assert!(
        !plan.contains("msaa_subsurface"),
        "a one-sample frame must not declare a multisampled second target:\n{plan}",
    );
    // Multisampled, the same target arrives by resolve instead, and both halves
    // have to be declared or the framebuffer binds an image the plan is silent
    // about.
    let multisampled = format!(
        "{}",
        declare(FrameConfig { msaa: true, ..base }).unwrap().graph
    );
    assert!(
        multisampled.contains("msaa_subsurface ColorAttachment")
            && multisampled.contains("subsurface_diffusible ResolveAttachment"),
        "a multisampled forward pass must declare the second target and its \
         resolve:\n{multisampled}",
    );
    // The temporal resolve is the next thing in this frame to read the lit
    // colour, so it is where a broken hand-over would show. It must read the
    // composite's output, never the forward pass's own.
    let taa = plan
        .split("pass taa_resolve")
        .nth(1)
        .expect("this frame resolves temporally");
    assert!(
        taa.contains("subsurface_color Sampled"),
        "the temporal resolve must read the composite's output:\n{plan}",
    );
    assert!(
        !taa.contains("hdr_color Sampled"),
        "nothing after the composite may read the frame the diffusible light was \
         withheld from:\n{plan}",
    );

    // And with the pass off, the frame is exactly what it was before any of this
    // existed: no second attachment, and the resolve reads the forward pass
    // directly again.
    let off = format!(
        "{}",
        declare(FrameConfig {
            subsurface: false,
            ..base
        })
        .unwrap()
        .graph
    );
    assert!(
        !off.contains("msaa_subsurface") && !off.contains("subsurface"),
        "a frame with the diffusion off must declare none of it:\n{off}",
    );
}

/// Refraction's three nodes have to land after the transparency composite and
/// before the temporal resolve, and both halves matter for the same reasons
/// transparency's placement does — with one more on top.
///
/// After the transparency composite, because the pyramid a refractive surface
/// samples its background out of is built from whatever `shaded` names at that
/// point: build it earlier and glass would refract a frame with the blended
/// geometry missing from it. Before the resolve, because the draw rasterises
/// with the frame's jitter like every other geometry pass.
///
/// The order inside the block is not a preference either: the pyramid is the
/// draw's input and the draw is the composite's, so a schedule that ran them in
/// any other order would be reading a target nothing had written.
#[test]
fn refraction_composites_after_the_transparency_and_before_the_resolve() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: true,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: true,
        subsurface: false,
        transparency: true,
        refraction: true,
        taa: true,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    })
    .unwrap();

    let order: Vec<&str> = frame
        .graph
        .order()
        .iter()
        .map(|&id| frame.graph.pass_name(id))
        .collect();
    let at = |name: &str| {
        order
            .iter()
            .position(|&pass| pass == name)
            .unwrap_or_else(|| panic!("{name} is not in the frame: {order:?}"))
    };

    assert!(at("oit_composite") < at("refraction_scene"), "{order:?}");
    assert!(at("refraction_scene") < at("refraction_draw"), "{order:?}");
    assert!(
        at("refraction_draw") < at("refraction_composite"),
        "{order:?}"
    );
    assert!(at("refraction_composite") < at("taa_resolve"), "{order:?}");
}

/// The refraction draw attaches the prepass depth read-only, exactly as the
/// transparency accumulation does, and for the same two reasons: a
/// fixed-function depth test is what puts a pane of glass behind a wall, and
/// `refraction.rs` builds its render pass by hand precisely so the attachment
/// reference says `DepthStencilReadOnlyOptimal`.
///
/// With both queues on and MSAA off, the prepass depth is attached read-only by
/// *three* passes — the forward pass joined them — and sampled by others, which
/// is the case the widened write-after-read barrier in `step` exists for. The
/// single-writer assertion below is what says none of the three started writing
/// it: it is the guarantee the whole one-sample forward path rests on, because a
/// forward pass that wrote depth would be racing the passes that sample it.
#[test]
fn refraction_attaches_the_prepass_depth_read_only() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: false,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: true,
        refraction: true,
        taa: false,
        auto_exposure: false,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: 0,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    })
    .unwrap();

    let plan = format!("{}", frame.graph);
    assert_eq!(
        plan.matches("prepass_depth DepthAttachmentRead").count(),
        3,
        "the forward pass and both non-opaque queues must attach the prepass \
         depth read-only:\n{plan}",
    );
    assert_eq!(
        plan.matches("prepass_depth DepthAttachment\n").count(),
        1,
        "the geometry prepass must remain the only writer of the depth:\n{plan}",
    );
}

/// Light meets a lens, then a shutter, then a sensor, and the frame has to run
/// them in that order — it is what Unity and Unreal both settled on, and it is
/// not arbitrary: defocus before the shutter means the streak smears an image
/// the lens already formed, and both before the sensor means a defocused
/// highlight blooms as the wide soft thing it has become rather than as the
/// point it was.
///
/// Nothing in `compile` knows any of that. The order is a consequence of how
/// `declare` chains `scene_color` from one stage to the next, so reordering
/// those blocks would silently reorder the optics — which is why it is asserted
/// against the derived schedule rather than against the source.
#[test]
fn the_optical_chain_runs_lens_then_shutter_then_sensor() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: true,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: false,
        refraction: false,
        taa: true,
        auto_exposure: true,
        motion_blur: true,
        dof: true,
        volumetric_fog: false,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    })
    .unwrap();

    let order: Vec<&str> = frame
        .graph
        .order()
        .iter()
        .map(|&id| frame.graph.pass_name(id))
        .collect();
    let at = |name: &str| {
        order
            .iter()
            .position(|&pass| pass == name)
            .unwrap_or_else(|| panic!("{name} is not in the frame: {order:?}"))
    };

    assert!(at("taa_resolve") < at("dof_prefilter"), "{order:?}");
    assert!(at("dof_composite") < at("motion_blur_gather"), "{order:?}");
    assert!(
        at("motion_blur_gather") < at("bloom_prefilter"),
        "{order:?}"
    );
    assert!(
        at("motion_blur_gather") < at("luminance_histogram"),
        "{order:?}"
    );
    assert!(at("motion_blur_gather") < at("tonemap"), "{order:?}");

    // The dilation has to sit between the tiles and the gather that reads them,
    // or a fast object's blur stops at the tile boundary it started in.
    assert!(
        at("motion_blur_tile_max") < at("motion_blur_neighbour_max"),
        "{order:?}"
    );
    assert!(
        at("motion_blur_neighbour_max") < at("motion_blur_gather"),
        "{order:?}"
    );
}

/// The geometry prepass exists for whoever needs what it writes, and that is now
/// nine separate consumers. Each has to be able to keep it alive on its own — a
/// prepass gated on any subset of them would leave motion blur gathering along
/// velocities nobody rasterised, depth of field focusing on a depth buffer that
/// was never allocated, a reflection trace reading a material target no pass
/// wrote, or the subsurface blur weighting its taps by a depth that does not
/// exist.
/// A named feature and the flag that turns it on.
type Consumer = (&'static str, fn(&mut FrameConfig));

#[test]
fn any_single_consumer_keeps_the_geometry_prepass() {
    let base = FrameConfig {
        color_format: COLOR_FORMAT,
        // On, and it is the premise of the whole test: a one-sample forward pass
        // is itself a consumer of the prepass, so the "nothing wants it" case
        // below can only be reached with MSAA rasterising its own depth.
        msaa: true,
        ssao: false,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: false,
        refraction: false,
        taa: false,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 0,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 0,
        async_compute: false,
        gpu_culling: false,
        occlusion_culling: false,
    };

    let consumers: [Consumer; 10] = [
        ("ssao", |c| c.ssao = true),
        ("contact shadows", |c| c.contact_shadows = true),
        ("taa", |c| c.taa = true),
        ("motion blur", |c| c.motion_blur = true),
        ("depth of field", |c| c.dof = true),
        ("reflections", |c| c.ssr = true),
        ("transparency", |c| c.transparency = true),
        ("refraction", |c| c.refraction = true),
        ("subsurface diffusion", |c| c.subsurface = true),
        // The forward pass itself, whenever it rasterises at one sample: it has
        // no depth buffer of its own then and tests `EQUAL` against this one.
        ("one-sample shading", |c| c.msaa = false),
    ];
    for (label, enable) in consumers {
        let mut config = base;
        enable(&mut config);
        let frame = declare(config).unwrap();
        assert!(
            frame.ids.prepass.is_some(),
            "{label} alone left the frame with no geometry prepass",
        );
    }

    // And nothing wants it when none of them do, so it is genuinely gated
    // rather than always present. `base` has MSAA on for exactly this line:
    // that is the one configuration whose forward pass rasterises its own depth
    // and therefore needs nothing from this pass.
    assert!(declare(base).unwrap().ids.prepass.is_none());
}

/// The swapchain image is acquired undefined and handed back to the presentation
/// engine, so the frame owes a closing transition to `PresentSrc` whatever else
/// it does.
#[test]
fn every_configuration_leaves_the_swapchain_presentable() {
    for (label, config) in configs() {
        let frame = declare(config).unwrap();
        let closing: Vec<_> = frame
            .graph
            .final_barriers()
            .iter()
            .filter(|barrier| {
                frame.graph.resource_name(barrier.resource) == "swapchain_color"
                    && barrier.new_layout == vulkano::image::ImageLayout::PresentSrc
            })
            .collect();
        assert_eq!(
            closing.len(),
            1,
            "{label}: {:?}",
            frame.graph.final_barriers()
        );
    }
}

/// GPU culling puts the visible set behind a compute dispatch, and the whole of
/// its safety is one ordering: nothing may read a draw command, or the instance
/// list that command indexes, before the dispatch that wrote them.
///
/// Asserted from what `declare` says the passes access rather than from where
/// they were written, because the failure this guards against is a geometry pass
/// that quietly stops declaring the read. That pass would then draw whatever
/// counts the buffer happened to hold — last frame's, on a frame that looks
/// right until something moves.
///
/// The reset half is here for the same reason. `cull` only ever *increments* an
/// instance count, so a frame whose counts were not put back to zero first draws
/// the union of every frame since the buffer was allocated.
#[test]
fn nothing_draws_before_the_cull_pass_has_filled_the_commands() {
    let frame = declare(FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: true,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: false,
        refraction: false,
        taa: true,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 4,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 4096,
        async_compute: false,
        gpu_culling: true,
        occlusion_culling: false,
    })
    .unwrap();

    let order: Vec<&str> = frame
        .graph
        .order()
        .iter()
        .map(|&pass| frame.graph.pass_name(pass))
        .collect();
    let at = |name: &str| {
        order
            .iter()
            .position(|&pass| pass == name)
            .unwrap_or_else(|| panic!("{name} was not scheduled:\n{order:?}"))
    };

    assert!(at("cull_reset") < at("cull"), "{order:?}");
    for geometry in [
        "shadow_cascade_0",
        "shadow_cascade_3",
        "punctual_shadows",
        "geometry_prepass",
        "forward",
    ] {
        assert!(at("cull") < at(geometry), "{geometry}: {order:?}");
    }

    let plan = format!("{}", frame.graph);
    // The dependency has to be an *indirect* read specifically. Declared as an
    // ordinary storage read it would be waited for at the wrong stage: the
    // command is fetched by the draw-indirect stage, which runs ahead of every
    // shader stage in the pass that issues it.
    assert!(
        plan.contains("dst DrawIndirect/IndirectCommandRead"),
        "the commands must be waited for at the draw-indirect stage:\n{plan}",
    );
    // The reset and the cull between them are the only writers of the commands,
    // and the cull is the only writer of the list. A third would be a pass
    // filling in draws of its own, which is the shape the compiler exists to
    // make impossible.
    assert_eq!(
        plan.matches("draw_commands StorageWrite\n").count(),
        2,
        "the reset and the cull must be the only writers of the commands:\n{plan}",
    );
    assert_eq!(
        plan.matches("instance_index StorageWrite\n").count(),
        1,
        "the cull must be the only writer of the instance list:\n{plan}",
    );
}

/// Occlusion culling reads a depth pyramid the frame has not built yet, and the
/// whole of its soundness is that this is deliberate: the pyramid the cull
/// samples is the *previous* frame's, because the one over this frame's depth
/// cannot exist until after the passes whose contents the cull decides.
///
/// So the two are separate resources, ping-ponged by the pass that owns them,
/// and the plan has to show it — the build after the cull, and the image the
/// cull samples never written inside the frame at all. Declared as one resource
/// it would be a write-after-read, which this compiler makes impossible by
/// ordering every reader after every writer: `compile` would put the cull after
/// the build, the build after the prepass, and the prepass after the cull, and
/// report the cycle.
#[test]
fn the_occlusion_pyramid_the_cull_reads_is_not_the_one_the_frame_builds() {
    let frame = declare(FrameConfig {
        occlusion_culling: true,
        ..occlusion_base()
    })
    .unwrap();

    let order: Vec<&str> = frame
        .graph
        .order()
        .iter()
        .map(|&pass| frame.graph.pass_name(pass))
        .collect();
    let at = |name: &str| {
        order
            .iter()
            .position(|&pass| pass == name)
            .unwrap_or_else(|| panic!("{name} was not scheduled:\n{order:?}"))
    };

    // Last frame's pyramid, so the cull does not wait for anything this frame
    // rasterises.
    assert!(at("cull") < at("occlusion_hiz"), "{order:?}");
    // And this frame's is reduced from the depth once that depth is complete.
    assert!(at("geometry_prepass") < at("occlusion_hiz"), "{order:?}");

    let plan = format!("{}", frame.graph);
    // Read by the cull and by nothing else, and — the part that matters —
    // written by nothing. A frame that wrote it would be overwriting the
    // pyramid it is in the middle of reading.
    assert_eq!(
        plan.matches("occlusion_history Sampled\n").count(),
        1,
        "the cull must be the only reader of last frame's pyramid:\n{plan}",
    );
    assert_eq!(
        plan.matches("occlusion_history StorageWrite\n").count(),
        0,
        "last frame's pyramid must not be written this frame:\n{plan}",
    );
    assert_eq!(
        plan.matches("occlusion_hiz StorageWrite\n").count(),
        1,
        "the build must be the only writer of this frame's pyramid:\n{plan}",
    );
}

/// Switching the test off declares neither the build nor the read, rather than
/// building a pyramid nothing samples. The flag is structural, and this is what
/// that word buys.
#[test]
fn a_frame_without_occlusion_culling_builds_no_pyramid() {
    let frame = declare(occlusion_base()).unwrap();

    let plan = format!("{}", frame.graph);
    assert!(!plan.contains("occlusion"), "{plan}");
}

/// The pyramid is reduced from the geometry prepass's depth, so a frame that
/// asks for the test needs that pass whatever else it has switched off — the
/// same claim SSAO, TAA and reflections each make on it.
#[test]
fn occlusion_culling_keeps_the_prepass_alive_on_its_own() {
    let frame = declare(FrameConfig {
        msaa: true,
        ssao: false,
        taa: false,
        transparency: false,
        auto_exposure: false,
        occlusion_culling: true,
        ..occlusion_base()
    })
    .unwrap();

    let order: Vec<&str> = frame
        .graph
        .order()
        .iter()
        .map(|&pass| frame.graph.pass_name(pass))
        .collect();
    assert!(order.contains(&"geometry_prepass"), "{order:?}");
    assert!(order.contains(&"occlusion_hiz"), "{order:?}");
}

/// The frame the three tests above vary one field of.
fn occlusion_base() -> FrameConfig {
    FrameConfig {
        color_format: COLOR_FORMAT,
        msaa: false,
        ssao: true,
        ssao_half_res: false,
        contact_shadows: false,
        ssr: false,
        subsurface: false,
        transparency: true,
        refraction: false,
        taa: true,
        auto_exposure: true,
        motion_blur: false,
        dof: false,
        volumetric_fog: false,
        bloom_mips: BLOOM_MIPS,
        overlay: true,
        shadow_cascades: 4,
        shadow_resolution: SHADOW_RESOLUTION,
        shadow_atlas: 4096,
        async_compute: false,
        gpu_culling: true,
        occlusion_culling: false,
    }
}

/// The whole safety condition for sharing an allocation, asserted over every
/// configuration the frame has: two images that share memory are never wanted at
/// the same time.
///
/// A violation is not a wrong pixel somewhere, it is one pass overwriting
/// another's target — so this is checked as a property of all 28 frames rather
/// than as a case, and from the lifetimes the compiler derived rather than from
/// the ones a reader would expect the post chain to have.
#[test]
fn nothing_shares_memory_with_something_still_alive() {
    for (label, config) in configs() {
        let graph = declare(config).unwrap().graph;
        for group in graph.alias_groups() {
            for pair in group.windows(2) {
                let (before, after) = (
                    graph.lifetime(pair[0]).unwrap(),
                    graph.lifetime(pair[1]).unwrap(),
                );
                assert!(
                    before.end <= after.start,
                    "{label}: {} is still live at {:?} when {} takes its memory at {:?}",
                    graph.resource_name(pair[0]),
                    before,
                    graph.resource_name(pair[1]),
                    after,
                );
            }
        }
    }
}

/// The other half of that condition, and the one a `Device` would otherwise have
/// to be asked about: images share a block only when they would have been
/// allocated identically, so the block one of them fits is the block all of them
/// fit.
#[test]
fn everything_sharing_a_block_would_have_been_allocated_alike() {
    for (label, config) in configs() {
        let graph = declare(config).unwrap().graph;
        let image = |id| {
            graph
                .transient_images()
                .find(|(other, _)| *other == id)
                .unwrap_or_else(|| panic!("{label}: an aliased resource is a transient image"))
                .1
        };
        for group in graph.alias_groups() {
            let first = image(group[0]);
            for &member in &group[1..] {
                let member_image = image(member);
                assert_eq!(
                    (first.desc, first.usage, first.concurrent),
                    (
                        member_image.desc,
                        member_image.usage,
                        member_image.concurrent
                    ),
                    "{label}: {} and {} share a block without sharing a shape",
                    graph.resource_name(group[0]),
                    graph.resource_name(member),
                );
                // Lazily-allocated memory is a different memory type, and on the
                // tiler it exists for it is no allocation at all.
                assert!(!member_image.memoryless, "{label}: memoryless and aliased");
            }
        }
    }
}

/// What the frame actually gets out of it, pinned so the feature cannot quietly
/// become a no-op — a shape change upstream that stopped two stages matching
/// would show up here as nothing shared, and nowhere else.
///
/// The count rather than the arrangement, because which stage inherits which
/// target is the greedy fit's business and reshuffles legitimately whenever a
/// pass moves. How many allocations the frame ends up needing is the thing the
/// item was asked for, so it is the thing asserted.
#[test]
fn the_post_chain_hands_its_targets_down() {
    let (label, config) = configs()
        .into_iter()
        .find(|(label, _)| *label == "editor frame, async compute with every optical stage")
        .expect("the every-stage configuration is what shows the sharing off");
    let graph = declare(config).unwrap().graph;
    let groups: Vec<Vec<&str>> = graph
        .alias_groups()
        .iter()
        .map(|group| group.iter().map(|&id| graph.resource_name(id)).collect())
        .collect();
    let shared: usize = groups.iter().map(|group| group.len()).sum();
    let saved = shared - groups.len();
    assert!(
        saved >= 5,
        "{label}: sharing saved {saved} allocations out of {} transients: {groups:?}",
        graph.transient_images().count(),
    );
}
