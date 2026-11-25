//! The engine's frame, declared as a graph.
//!
//! This module is the one place that says what a frame *is*. It takes no
//! `Device` and allocates nothing: it registers resources and passes and hands
//! back the compiled result, so the same function that runs on the GPU each
//! frame is the one CI compiles and asserts the barrier plan of.
//!
//! Adding a pass is a registration here plus a [`PassBody`] arm in `execute`.
//! Nothing computes an execution order, a layout, or a barrier by hand, which is
//! what makes a new pass a registration in `declare` rather than surgery on a
//! fixed pipeline.

use crate::gfx::format::Format;
use crate::gfx::graph::ImageLayout;

use crate::gfx::graph::{
    Access, Extent, FrameGraph, GraphBuilder, GraphError, ImageDesc, PassBuilder, PassId, PassKind,
    ResourceId, compile,
};

use super::MSAA_SAMPLES;
use super::bloom::MAX_BLOOM_MIPS;
use super::contact_shadows::MASK_FORMAT;
use super::dof::COC_TILE_SHIFT;
use super::fog::{FOG_FORMAT, FROXEL_SHIFT, FROXEL_SLICES, LIGHT_SHIFT, LIGHT_STATE_FORMAT};
use super::hdr::{HDR_FORMAT, HDR_WIDE_FORMAT};
use super::motion_blur::TILE_SHIFT;
use super::occlusion::{OCCLUSION_FORMAT, OCCLUSION_LEVELS};
use super::oit::{ACCUM_FORMAT, REVEAL_FORMAT};
use super::outline::OUTLINE_MASK_FORMAT;
use super::prepass::{MATERIAL_FORMAT, NORMAL_FORMAT, VELOCITY_FORMAT};
use super::refraction::{ACCUM_FORMAT as REFRACTION_ACCUM_FORMAT, SCENE_LEVELS};
use super::shadow::SHADOW_FORMAT;
use super::ssao::AO_FORMAT;
use super::ssr::{HIZ_FORMAT, HIZ_LEVELS, RAY_FORMAT, SOURCE_LEVELS};
use super::subsurface::SUBSURFACE_FORMAT;
use super::swapchain::DEPTH_FORMAT;
use crate::gfx::shadow_views::atlas_extent;

/// What a frame's structure depends on. A change to any of these recompiles the
/// graph; nothing else does, which is the "recompiled on structure change, not
/// per frame" rule made concrete — the field list *is* the rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameConfig {
    pub color_format: Format,
    /// Whether the forward pass rasterises at [`MSAA_SAMPLES`] rather than one.
    ///
    /// Off by default, and the default is the point: this frame also runs TAA,
    /// which already resolves geometric edges, so multisampling on top of it
    /// buys the absence of one frame of temporal lag on silhouettes for roughly
    /// a quarter of the frame time. `resources.rs` asks for `LAZILY_ALLOCATED`
    /// memory so that the multisampled targets cost no DRAM at all, which is
    /// true on MoltenVK's tile memory and false on every desktop driver —
    /// AMD and NVIDIA expose no lazily-allocated memory type, so there the pair
    /// is ordinary VRAM paying full write-and-resolve traffic.
    ///
    /// Kept as a field rather than deleted because the evidence is worth being
    /// able to reproduce, and because the MoltenVK path really is nearly free.
    /// It costs a second set of pipelines at startup, which the masked and
    /// subsurface variants already establish as the shape of this pass.
    ///
    /// Structural, and more deeply than most: it decides which forward *render
    /// pass* the frame opens, whether the multisampled targets are declared at
    /// all, and — off — makes the geometry prepass mandatory, because the
    /// forward pass then depth-tests against the depth that pass wrote instead
    /// of rasterising its own.
    pub msaa: bool,
    pub ssao: bool,
    /// Whether the occlusion is resolved at half the frame's extent. Structural
    /// because it is a size rather than a uniform: the two AO images change
    /// extent, so the graph has to reallocate them and the passes that draw into
    /// them have to be told the new viewport.
    pub ssao_half_res: bool,
    /// Whether the frame marches the depth buffer for the shadow band the
    /// cascades cannot resolve. Structural like the rest, and one more consumer
    /// that keeps the geometry prepass alive on its own — it reads the depth and
    /// the normals that pass writes, and its result is what the forward pass
    /// multiplies the sun's term by.
    pub contact_shadows: bool,
    /// Whether the frame reflects off itself. Structural like the rest, and
    /// another consumer that can keep the geometry prepass alive on its own —
    /// it reads every target the prepass writes but motion.
    pub ssr: bool,
    /// Whether the frame diffuses subsurface light across the image. Structural
    /// twice over, and more deeply than the rest: it registers three passes, it is
    /// another consumer that keeps the geometry prepass alive on its own, and it
    /// decides which of the two forward *render passes* the frame opens — the one
    /// that resolves a second colour target or the one that does not.
    ///
    /// That last part is why the shading is a second fragment shader rather than a
    /// branch. Off, the analytic wrap in `shading.glsl` widens to stand in for the
    /// diffusion, so scattering does not disappear with the passes.
    pub subsurface: bool,
    /// Whether the frame resolves against a reprojected history. Structural
    /// twice over: it registers the resolve node, and it is what makes the
    /// geometry prepass exist in a frame that has SSAO switched off — the
    /// prepass is where the motion vectors come from.
    pub taa: bool,
    /// Whether the frame meters its own luminance. Off is a different graph, not
    /// a flag read at record time: the two compute passes are never registered
    /// and the tonemap pass falls back to the manual exposure it is pushed.
    pub auto_exposure: bool,
    /// Whether the frame reconstructs the shutter's exposure from the motion
    /// vectors. Structural for the same reason TAA is, and with the same
    /// consequence: it is one of the things that makes the geometry prepass
    /// exist in a frame with SSAO switched off.
    pub motion_blur: bool,
    /// Whether the frame defocuses what the lens is not focused on. Structural,
    /// and the other consumer that can keep the prepass alive on its own —
    /// depth is all it needs from it.
    pub dof: bool,
    /// Whether the frame draws blended geometry. Structural like the rest: it
    /// registers the accumulation and composite nodes, and it is one more thing
    /// that keeps the geometry prepass alive — the accumulation depth-tests
    /// against the depth that pass writes, because the forward pass's own is a
    /// memoryless attachment that does not survive its render pass.
    pub transparency: bool,
    /// Whether the frame draws refractive geometry. Structural like the rest,
    /// and it keeps the geometry prepass alive for the reason transparency
    /// does — the draw depth-tests against the depth that pass writes.
    ///
    /// Independent of `transparency` rather than folded into it: the two queues
    /// answer to opposite rules about ordering, so a frame may reasonably want
    /// one without the other, and each is its own A/B for "is this a
    /// transparency bug or a refraction bug?".
    pub refraction: bool,
    /// Levels in the bloom chain, zero for none. Derived from the frame's extent
    /// rather than set, so the number of passes registered cannot disagree with
    /// the number of levels there is room for — the same reason
    /// `shadow_atlas` is sourced from the fitted views.
    /// Whether the frame marches the air in front of the camera as a froxel
    /// volume. Structural like the rest: it registers two dispatches and the
    /// volumes they write, and it is the one effect here that changes what the
    /// *geometry* passes read rather than what runs after them — the forward,
    /// transparency and refraction passes all sample the result.
    ///
    /// Off is not "no fog": the same medium is integrated analytically along
    /// each view ray inside `shading.glsl`, which is what the froxels hand over
    /// to past their far plane anyway. What this switches is whether the first
    /// `FogSettings::distance` metres are shadowed.
    pub volumetric_fog: bool,
    pub bloom_mips: u8,
    /// Whether the editor's egui overlay draws over the frame. Off for headless
    /// and export renders.
    pub overlay: bool,
    /// Whether the selection outline draws: the overlay is on *and* something is
    /// selected. Separate from `overlay` because the outline is two full-screen
    /// passes — a mask, then an edge search over it — and with nothing selected
    /// both would run every frame to produce nothing. Toggling it recompiles the
    /// graph, which `GraphImages`' allocation cache makes cheap.
    pub outline: bool,
    /// Whether the transform gizmo draws, for the reason `outline` is separate.
    pub gizmo: bool,
    /// Whether the game drew any in-game UI this frame. Independent of
    /// `overlay`: a shipped game has no editor and still has its HUD, and a
    /// frame whose game drew nothing registers no pass.
    pub ui: bool,
    /// Whether the frame's target goes to a presentation engine (a window) or is
    /// copied out (an offscreen capture). It decides the layout the target ends
    /// in: `PresentSrc` needs the swapchain extension, which an offscreen device
    /// does not enable, and the readback copies from `TransferSrcOptimal`.
    pub presents: bool,
    /// Edge of the shadow atlas in texels, zero for a frame where nothing
    /// casts. Sourced from the fitted views rather than from the setting, so a
    /// frame with the sun's shadows off and every lamp opted out declares no
    /// atlas rather than keeping one nobody reads.
    ///
    /// One number and one pass however many views there are — which of them a
    /// frame draws varies *inside* the pass, which is exactly why it is not
    /// here.
    pub shadow_atlas: u32,
    /// Whether the frame's compute tail may be submitted to a second queue, so
    /// that the *next* frame's graphics head can start before this frame's post
    /// chain has finished.
    ///
    /// Structural, and the reason it is a config field rather than a runtime
    /// check: it changes how many submissions a frame is, which resources have
    /// to be reachable from both queues, and therefore how they are allocated.
    /// The renderer sets it from what the device turned out to have — a device
    /// with no compute-only queue family plans exactly the frame it always did.
    pub async_compute: bool,
    /// Whether the compute cull also drops what the previous frame's depth
    /// already covered.
    ///
    /// Structural: it registers the pyramid build and gives the cull a texture
    /// to read, and it is one more thing that keeps the geometry prepass alive
    /// — the pyramid is reduced from the depth that pass writes, and a frame
    /// with no prepass has no depth to reduce.
    ///
    /// Kept as a flag rather than always on: the visible set it produces is
    /// *smaller* than the frustum's, so the two have to be comparable on one
    /// binary to say what the difference cost and what it drew.
    pub occlusion_culling: bool,
    /// The viewport's wireframe draw mode, when it is not plain shaded. The
    /// editor's, so it needs the overlay too; `None` declares no node.
    pub wireframe: Option<super::wireframe::Wireframe>,
    /// Whether the inspector's material sphere is redrawn this frame: the
    /// editor is showing one *and* something it shows changed. See
    /// [`material_preview`](super::material_preview).
    pub material_preview: bool,
    /// Whether this frame also draws the picture into an image of its own,
    /// for `VulkanRenderer::take_screenshot` to read back, and which picture:
    /// the editor's remote asked for one. `None` is the frame exactly as it is
    /// without the feature — no node, no image, no barrier — and a frame that
    /// asks recompiles the graph twice, as one that redraws the material
    /// sphere does.
    pub screenshot: Option<ShotView>,
    /// Whether the frame builds the acceleration structures a ray is traced
    /// against: the device can, and something in the frame traces. Structural
    /// because it registers the two builds, and those cost a frame that traces
    /// nothing the same as one that does.
    pub ray_tracing: bool,
    /// Whether the traced scene is drawn in place of the frame. Needs
    /// `ray_tracing`.
    pub ray_debug: bool,
    /// Whether the reflections trace the scene for what the screen cannot
    /// answer. Needs `ray_tracing` and `ssr`.
    pub ray_reflections: bool,
    /// Whether the frame's colour is replaced by the path-traced reference.
    /// Needs `ray_tracing`.
    pub ray_path_trace: bool,
    /// Probes along each side of an irradiance volume's cascade, zero where
    /// the volumes are off. Live, they are four dispatches at the head of the
    /// frame and every pass that lights a surface reads what they leave; a
    /// size rather than a flag, because it is the atlases' extent. Needs
    /// `ray_tracing`. Zero is the frame exactly as it was, lit by the
    /// environment's diffuse series alone.
    pub gi_probes: u32,
    /// How many times the final gather's resolution halves the frame's each
    /// way, zero where it does not run. Live, it is five dispatches between
    /// the prepass and the forward pass, and the forward pass reads what they
    /// leave in place of the volumes; a shift rather than a flag, because it
    /// is the extent of their images. Needs `gi_probes`, and keeps the prepass
    /// alive on its own.
    pub gi_gather: u32,
}

/// What a screenshot shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShotView {
    /// The scene alone, tonemapped.
    Scene,
    /// The scene with the game's own UI over it, and none of the editor's
    /// marks: what a player sees.
    Game,
    /// The window as the developer sees it: the game's view with the
    /// wireframe, the outline, the handles and the editor's own UI over it.
    Editor,
}

/// Which piece of engine code a graph node runs.
///
/// The graph knows a pass by its declarations; this is the other half of the
/// mapping, kept as data so it stays device-free and so the executor's dispatch
/// is exhaustive by the compiler's own reckoning rather than by convention.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PassBody {
    /// Stamps every draw command with the geometry its batch draws and an
    /// instance count of zero, which is the value the cull increments from.
    CullReset,
    /// Tests every live instance against every view and appends the ones that
    /// survive to their batch's slice of that view's instance list.
    Cull,
    GeometryPrepass,
    /// Reduces the depth this frame rasterised to a max pyramid, for the *next*
    /// frame's cull to test against.
    OcclusionHiz,
    SsaoResolve,
    SsaoBlur,
    /// The sun's visibility over the short range a cascade texel cannot resolve.
    ContactShadows,
    /// The light the air scatters from everything but the sun, traced from the
    /// air into the world radiance cache where the irradiance volumes are live.
    FogLight,
    /// What every froxel of air scatters and blocks — one cascade lookup apiece,
    /// which is the whole cost of the effect and the whole reason a shaft has an
    /// edge.
    FogScatter,
    /// Each froxel column marched front to back into the running radiance and
    /// transmittance a fragment applies. One invocation per column rather than
    /// per froxel, because slice `n` needs what slice `n - 1` ended with.
    FogIntegrate,
    Forward,
    /// Every level of the reflection trace's min-depth pyramid, in one dispatch.
    SsrHiz,
    /// Every level of the lit frame's pyramid, likewise — what a ray's cone is
    /// sampled out of.
    SsrSource,
    /// One exact reflection ray per half-resolution pixel.
    SsrTrace,
    /// Gathers those rays into each pixel, traces a smooth surface's own, and
    /// swaps the environment's reflection for what they found.
    SsrResolve,
    /// One axis of the subsurface diffusion each, sharing a pipeline and differing
    /// only in the direction they are pushed — the same shape the bloom chain's
    /// levels take.
    SubsurfaceBlurHorizontal,
    SubsurfaceBlurVertical,
    /// Adds the diffused radiance back into the frame the forward pass withheld it
    /// from.
    SubsurfaceComposite,
    /// Every blended surface, into the two targets whose blend equations
    /// commute — which is what makes the draw order irrelevant.
    OitAccumulate,
    /// Divides that accumulation by its own coverage and mixes it over the lit
    /// frame by the transmittance beside it.
    OitComposite,
    /// The lit frame as a mip pyramid, so a rough refraction reads a cone.
    RefractionScene,
    /// Every refractive surface, back to front, into a premultiplied target.
    RefractionDraw,
    /// Puts that target over the lit frame.
    RefractionComposite,
    /// Reprojects the history onto this frame and accumulates into it.
    TaaResolve,
    /// Half the frame, carrying its own circle of confusion.
    DofPrefilter,
    /// The widest near and far circle per tile, which sizes the gather's kernel.
    DofTileMax,
    /// Spreads each half-resolution texel over its circle, into a near and a far
    /// field.
    DofGather,
    /// Puts both fields back over the sharp frame at full resolution.
    DofComposite,
    /// The longest blur vector in each tile.
    MotionBlurTileMax,
    /// Dilates that by one ring, so every pixel a blur can reach knows about it.
    MotionBlurNeighbourMax,
    /// Reconstructs the exposure by walking the dominant blur vector.
    MotionBlurGather,
    LuminanceHistogram,
    LuminanceAverage,
    /// Half the frame, exposed and firefly-weighted: the chain's first level.
    BloomPrefilter,
    /// Writes down-chain level `n`, reading level `n - 1`.
    BloomDownsample(u32),
    /// Writes up-chain level `n`, reading the coarser level and down-chain `n`.
    BloomUpsample(u32),
    Tonemap,
    /// The game's own UI, over the tonemapped frame.
    Ui,
    /// The selected and hovered meshes, flat, into the outline's mask.
    OutlineMask,
    /// The outline round the mask's shapes, over the tonemapped frame.
    Outline,
    /// The viewport's wireframe, over the tonemapped frame or alone.
    Wireframe,
    /// The editor's transform handles, over the tonemapped frame.
    Gizmo,
    /// The inspector's material on a sphere, into targets of its own.
    MaterialPreview,
    /// That sphere supersampled down and tonemapped, for the overlay to sample.
    MaterialPreviewResolve,
    /// The tonemap again, into the screenshot's own image: the scene with
    /// nothing drawn over it.
    Screenshot,
    /// The game's UI again, over that.
    ScreenshotUi,
    /// The editor's marks again, over that, each as its window twin draws.
    ScreenshotWireframe,
    ScreenshotOutline,
    ScreenshotGizmo,
    /// The editor's UI again, from the primitives the window's overlay draws
    /// this frame: an inline pass, unlike its `Raw` twin, so the graph orders
    /// it and derives its barriers.
    ScreenshotOverlay,
    Overlay,
    /// The acceleration structures of meshes uploaded since the last frame
    /// that built any, within the frame's budget, and their compaction.
    RayMeshes,
    /// The scene's acceleration structure, over this frame's instances.
    RayScene,
    /// The traced scene from one ray per pixel, over the tonemapped frame.
    RayDebug,
    /// One more path per pixel of the reference, and its running mean.
    PathTrace,
    /// Every ray of every probe of the grid this frame updates, each hit
    /// filed in the world radiance cache.
    GiTrace,
    /// The cache's cells the rays used, listed for shading; the ones nothing
    /// has used for long enough, emptied.
    GiCacheCompact,
    /// Each probe's tile brought up to date from its rays.
    GiUpdate,
    /// Each listed cell shaded once, for the probes' next update.
    GiCacheShade,
    /// One ray per texel of the final gather, kept as harmonics.
    GiGather,
    /// Those rays into each texel's running mean, reprojected.
    GiGatherTemporal,
    /// One axis of one level of the gather's spatial filter.
    GiGatherBlur {
        vertical: bool,
        wide: bool,
    },
    /// The gather to the frame's resolution, for the forward pass.
    GiGatherUpscale,
    /// The cached layers of the shadow views that redraw theirs this frame:
    /// the casters standing still, into an atlas kept from frame to frame.
    ShadowStatic,
    /// What the frame is shaded from: every shadow view's live tile, brought up
    /// to date from the cached layer and this frame's moving casters. One pass
    /// over one atlas for the sun's views and the lamps' alike — a tile is a
    /// viewport rather than a node, which is what keeps this a single barrier
    /// instead of one per view.
    Shadows,
}

/// Handles the executor needs to bind per-frame resources and to find the
/// graph-owned views a pass draws into.
#[derive(Clone, Copy, Debug)]
pub struct FrameIds {
    pub object_transforms: ResourceId,
    pub instance_index: ResourceId,
    pub swapchain_color: ResourceId,
    pub hdr_color: ResourceId,
    /// What the tonemap, metering and bloom passes read: whichever image the
    /// optical chain left the frame's colour in — `hdr_color` in the barest
    /// frame, and otherwise the last of the TAA resolve, the lens and the
    /// shutter that ran. Named once here so nothing downstream has to ask which
    /// frame it is in, which is what makes inserting a stage a change to one
    /// binding rather than to every consumer.
    pub scene_color: ResourceId,
    /// What the exposure is metered from: the rasterised frame, which
    /// `scene_color` is but where the path-traced reference stands in for it.
    /// The reference is then exposed as the frame it is held against, so a
    /// difference between them is one of light and not of exposure.
    pub metered_color: ResourceId,
    /// The reference's two images, where the frame draws it.
    pub path: Option<PathIds>,
    /// The multisampled pair, in a frame that rasterises at more than one
    /// sample. `None` on the default path, where the forward pass shades
    /// straight into `hdr_color` and borrows the prepass depth.
    pub msaa: Option<MsaaIds>,
    /// Written by the metering passes and read by the tonemap pass. Always
    /// declared, because the tonemap pass binds it whether or not anything wrote
    /// it this frame — an import may be read without a writer, which is exactly
    /// the case metering-off produces.
    pub exposure: ResourceId,
    /// `None` when metering is off, along with the two passes that touch it.
    pub histogram: Option<ResourceId>,
    pub bloom: Option<BloomIds>,
    /// Present whenever anything downstream needs depth, normals or motion —
    /// SSAO, TAA, motion blur, depth of field, or any combination of them.
    pub prepass: Option<PrepassIds>,
    pub ssao: Option<SsaoIds>,
    /// The two pyramids occlusion culling carries across the frame boundary.
    /// `None` when the frame does not test against depth, along with the build.
    pub occlusion: Option<OcclusionIds>,
    /// The sun-visibility mask, when the frame marches for one. One image and no
    /// struct: the pass reads the prepass and writes this, and there is nothing
    /// else to name.
    pub contact_shadows: Option<ResourceId>,
    pub fog: Option<FogIds>,
    pub ssr: Option<SsrIds>,
    /// Present exactly when the frame runs the diffusion — which is also how the
    /// executor knows to bind the forward pass's second pipeline, so the
    /// framebuffer and the pipeline cannot disagree about the attachment count.
    pub subsurface: Option<SubsurfaceIds>,
    pub transparency: Option<TransparencyIds>,
    pub refraction: Option<RefractionIds>,
    pub taa: Option<TaaIds>,
    pub dof: Option<DofIds>,
    pub motion_blur: Option<MotionBlurIds>,
    /// The shadow atlas, when anything casts: what the shading passes and the
    /// fog read.
    pub shadow_atlas: Option<ResourceId>,
    /// The atlas of cached layers beside it, tile for tile.
    pub shadow_static: Option<ResourceId>,
    /// Which pixels the selection outline is drawn around: red where a selected
    /// mesh covers, green where the hovered one does. With the editor only.
    pub outline_mask: Option<ResourceId>,
    /// The wireframe's mode, and the depth it hides edges with when it has one.
    pub wireframe: Option<(super::wireframe::Wireframe, Option<ResourceId>)>,
    pub material_preview: Option<MaterialPreviewIds>,
    /// The screenshot's image, which the renderer owns for the one frame that
    /// draws it.
    pub screenshot: Option<ResourceId>,
    /// What a ray is traced against, in a frame that traces.
    pub ray: Option<RayIds>,
    /// The irradiance volumes and the radiance cache, where they are live.
    pub gi: Option<GiIds>,
    /// The final gather's images, where it runs.
    pub gather: Option<GatherIds>,
}

/// The irradiance volumes and the world radiance cache: every one kept from
/// frame to frame by the pass that owns it, so imported. See `vulkan::gi`.
#[derive(Clone, Copy, Debug)]
pub struct GiIds {
    /// Every probe's octahedral tile of irradiance, and of the distance it
    /// sees and its square: what every pass that lights a surface samples.
    pub irradiance: ResourceId,
    pub distance: ResourceId,
    /// The same irradiance over the last few updates alone, which the cache's
    /// shading reads for the light that bounced.
    pub bounce: ResourceId,
    /// The part of that which came by each update's brightest rays, over as
    /// many updates as `irradiance`: only the update reads it.
    pub outlier: ResourceId,
    /// Each probe's offset, whether it is inside geometry, and the coordinate
    /// it was last written for.
    pub probes: ResourceId,
    /// What each ray this frame traces met, and what each ray last frame
    /// traced met, which this frame's update takes in.
    pub rays: ResourceId,
    pub rays_previous: ResourceId,
    /// The cache's table, and the list of its cells to shade with the
    /// dispatch that covers them.
    pub cells: ResourceId,
    pub list: ResourceId,
    pub args: ResourceId,
    /// Each cell's light as the update reads it — last frame's shading — and
    /// as this frame's shading writes it. The two swap every frame, for the
    /// reason the occlusion pyramids do: see [`OcclusionIds`].
    pub radiance_history: ResourceId,
    pub radiance: ResourceId,
    /// The final gather's ray for each cell it asked to be shaded, where the
    /// gather runs.
    pub gather_hits: Option<ResourceId>,
}

impl GiIds {
    /// Whether `id` is one of the buffers here, which a barrier covers with a
    /// memory barrier rather than by name.
    pub fn holds_buffer(&self, id: ResourceId) -> bool {
        [
            self.probes,
            self.rays,
            self.rays_previous,
            self.cells,
            self.list,
            self.args,
            self.radiance_history,
            self.radiance,
        ]
        .contains(&id)
            || self.gather_hits == Some(id)
    }
}

/// The final gather's images. See `vulkan::gi_gather`.
///
/// The accumulation is **imported** and ping-ponged, for the reason the TAA
/// history is: `accumulated` is this frame's allocation and next frame's
/// `history`, so it leaves in the layout `history` enters in. Everything else
/// is made and spent within the frame.
#[derive(Clone, Copy, Debug)]
pub struct GatherIds {
    /// The rays as they came back: three layers of harmonics and a fourth
    /// for how far they went.
    pub raw: ResourceId,
    pub history: ResourceId,
    pub accumulated: ResourceId,
    /// The spatial filter's two axes, and its second level's.
    pub blurred_x: ResourceId,
    pub blurred: ResourceId,
    pub wide_x: ResourceId,
    pub wide: ResourceId,
    /// The harmonics at the frame's resolution: what the forward pass reads.
    pub irradiance: ResourceId,
}

/// The path-traced reference: every path's radiance so far, which the pass
/// that owns it keeps between frames, and their mean, which the frame shows.
#[derive(Clone, Copy, Debug)]
pub struct PathIds {
    pub sum: ResourceId,
    pub mean: ResourceId,
}

/// The two levels of what a ray is traced against, which the device owns.
#[derive(Clone, Copy, Debug)]
pub struct RayIds {
    /// Every mesh's acceleration structure, as one resource: a frame builds
    /// some of them and the scene's is built over all of them, so one barrier
    /// orders the two however many there are.
    pub meshes: ResourceId,
    /// The scene's, which a tracing shader binds.
    pub scene: ResourceId,
}

/// The inspector's material sphere: shaded into `hdr` against `depth`, then
/// resolved into `output`, which the preview module owns and the overlay
/// samples after the frame.
#[derive(Clone, Copy, Debug)]
pub struct MaterialPreviewIds {
    pub hdr: ResourceId,
    pub depth: ResourceId,
    pub output: ResourceId,
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

/// What the one geometry pass in front of shading leaves behind. Written
/// together because they come off the same rasterisation, and read apart: SSAO
/// wants depth and normals, TAA and motion blur want depth and motion, depth of
/// field wants depth alone, reflections want all but motion.
#[derive(Clone, Copy, Debug)]
pub struct PrepassIds {
    pub normal: ResourceId,
    pub velocity: ResourceId,
    /// `rgb` = the surface's `f0`, `a` = perceptual roughness. What a reflection
    /// is tinted by and how wide its lobe is — the two things a screen-space
    /// trace cannot recover from depth and normals alone.
    pub material: ResourceId,
    pub depth: ResourceId,
}

/// The froxel fog's two volumes.
///
/// `scatter` is **imported** and ping-ponged, for the reason the TAA history is:
/// the scatter pass reads its own previous frame to average the depth jitter,
/// and a transient is `Undefined` at every frame's start by contract. The
/// backing allocations swap each frame, so the volume bound here is the one
/// bound as the history next — which is why it leaves the frame in the layout it
/// is declared to enter in.
///
/// `volume` is an ordinary transient: it is produced and consumed inside one
/// frame, and nothing wants last frame's integral.
#[derive(Clone, Copy, Debug)]
pub struct FogIds {
    /// The light the air scatters from everything but the sun, traced, where
    /// the irradiance volumes are live: imported and ping-ponged as `scatter`
    /// is, for the same reason.
    pub light: Option<ResourceId>,
    /// How far each cell's rays disagree from one frame to the next, which
    /// sets how long its light is averaged: ping-ponged with `light`.
    pub light_state: Option<ResourceId>,
    /// `rgb` = scattered radiance per metre, `a` = extinction per metre.
    pub scatter: ResourceId,
    /// `rgb` = in-scattered radiance up to this slice, `a` = the transmittance
    /// behind it. What every shading pass samples.
    pub volume: ResourceId,
}

#[derive(Clone, Copy, Debug)]
pub struct SsaoIds {
    pub raw_ao: ResourceId,
    pub ao: ResourceId,
}

/// The reflection trace's images, and what it was handed. All transient: an
/// exact ray is the same next frame, so nothing is kept between frames.
#[derive(Clone, Copy, Debug)]
pub struct SsrIds {
    /// The lit frame the rays sample, recorded rather than re-derived for the
    /// reason depth of field records its own: this pass sits in a chain, and
    /// the executor must bind exactly what `declare` said it would read.
    pub source: ResourceId,
    /// The min-depth pyramid, one image with every level — see
    /// [`ImageDesc::mip_levels`](crate::gfx::graph::ImageDesc::mip_levels) for
    /// why a single pass has to write all of them.
    pub hiz: ResourceId,
    /// Half the frame with a pyramid over it: the lit colour, prefiltered, so a
    /// ray from a rough surface reads the blur its lobe would see in one fetch.
    pub source_pyramid: ResourceId,
    /// Half the frame: the radiance each ray found, with its confidence in
    /// alpha.
    pub rays: ResourceId,
    pub output: ResourceId,
    /// Whether the trace and the composite also trace the scene itself, which
    /// is which pair of pipelines the executor binds.
    pub traced: bool,
}

/// The forward pass's multisampled targets, declared only when `FrameConfig::msaa`
/// is on.
///
/// Both are `DontCare`/`DontCare` within the render pass: the colour is resolved
/// into `hdr_color` and the depth is never read again, which is what lets both
/// ask for lazily-allocated memory on a tiler. On a desktop driver that request
/// is silently ignored and they are the most expensive images in the frame.
#[derive(Clone, Copy, Debug)]
pub struct MsaaIds {
    pub hdr: ResourceId,
    pub depth: ResourceId,
}

/// Subsurface scattering's targets: the one the forward pass resolves into, the
/// two the separable blur ping-pongs through, and what the composite was handed.
///
/// `msaa_diffusible` and `diffusible` are a second colour attachment and a second
/// resolve on the forward pass, in the same shape as its first pair. That is the
/// price of the feature and the reason it is structural: a frame without it
/// declares neither.
#[derive(Clone, Copy, Debug)]
pub struct SubsurfaceIds {
    pub source: ResourceId,
    /// The multisampled half of the pair, and `None` for the same reason
    /// [`FrameIds::msaa`] is: at one sample the forward pass writes `diffusible`
    /// directly and there is nothing to resolve from.
    pub msaa_diffusible: Option<ResourceId>,
    /// `rgb` = the radiance that left the surface elsewhere, `a` = the widest
    /// channel's mean free path in metres. The alpha is the only mask the passes
    /// need, so nothing else carries one.
    pub diffusible: ResourceId,
    /// The horizontal axis's result, and the vertical's. Two images rather than
    /// one, because a resource is unversioned: a pass that read and wrote the same
    /// one would make "readers after all writers" point both ways and `compile`
    /// would report a cycle.
    pub blurred_x: ResourceId,
    pub blurred_y: ResourceId,
    pub output: ResourceId,
}

/// Weighted-blended transparency's two targets, and what it was handed.
///
/// `source` is recorded rather than re-derived for the reason depth of field
/// records its own: this pass sits in a chain — it composites over the
/// reflections' output in a frame that has them and the forward pass's target in
/// a frame that does not — and the executor must bind exactly what `declare`
/// said it would read.
#[derive(Clone, Copy, Debug)]
pub struct TransparencyIds {
    pub source: ResourceId,
    /// `rgb` = weighted premultiplied radiance summed, `a` = weighted coverage
    /// summed.
    pub accum: ResourceId,
    /// The product of `1 - alpha` over everything that covered the pixel. One
    /// channel, and cleared to 1.0 rather than 0 because it is a product.
    pub reveal: ResourceId,
    pub output: ResourceId,
}

/// Screen-space refraction's three images, and what it was handed.
///
/// `source` is recorded rather than re-derived for the reason the transparency
/// ids record theirs: this pass sits in a chain, and the executor must bind
/// exactly what `declare` said it would read.
#[derive(Clone, Copy, Debug)]
pub struct RefractionIds {
    pub source: ResourceId,
    /// The lit frame reduced to a mip chain, at full resolution: a clear pane
    /// of glass shows the world behind it as sharply as the frame recorded it,
    /// and only a rough one reads a coarser level.
    pub scene: ResourceId,
    /// `rgb` = premultiplied radiance, `a` = coverage. Cleared to zero, so a
    /// pixel nothing refractive covered leaves the frame as it found it.
    pub accum: ResourceId,
    pub output: ResourceId,
}

/// The two depth pyramids occlusion culling carries across the frame boundary.
///
/// Both are **imported** and ping-ponged, for the reason the TAA history is —
/// a transient is `Undefined` at every frame's start by contract, and what the
/// cull tests against is precisely a survivor of the last one.
///
/// Two resources rather than one, and this is the whole shape of the feature.
/// The cull decides what the prepass draws; the prepass writes the depth; the
/// pyramid reduces that depth. A cull that read a pyramid of *this* frame's
/// depth would close that ring, and `compile` orders every reader after every
/// writer precisely so that it would say so — the cull after the build, the
/// build after the prepass, the prepass after the cull, and a cycle. So the
/// cull reads `history`, which no pass in this frame writes, and the build
/// writes `hiz`, which no pass in this frame reads. The allocations swap.
#[derive(Clone, Copy, Debug)]
pub struct OcclusionIds {
    /// Last frame's pyramid: what the cull samples.
    pub history: ResourceId,
    /// This frame's: what the build writes, and what `history` names next frame.
    pub hiz: ResourceId,
}

/// The two images TAA carries across the frame boundary.
///
/// Both are **imported**, for the reason the exposure buffer is: a transient is
/// `Undefined` at every frame's start by contract, and a history that survives
/// one frame is precisely what this pass needs. The backing allocations are
/// ping-ponged, so the image bound as `output` this frame is the one bound as
/// `history` next — which is why `output` leaves the frame in the layout
/// `history` declares it enters in.
#[derive(Clone, Copy, Debug)]
pub struct TaaIds {
    /// What this frame's colour is at the point the resolve accumulates it —
    /// the forward pass's target, or the reflection composite's where that ran.
    /// Recorded rather than re-derived for the reason depth of field records its
    /// own source: the executor must bind exactly what `declare` said it reads.
    pub source: ResourceId,
    pub history: ResourceId,
    pub output: ResourceId,
}

/// Depth of field's three images, and what it was handed.
///
/// `source` is recorded rather than re-derived because this pass sits in a
/// chain: it reads the TAA resolve's output in a frame that has one and the
/// forward pass's in a frame that does not, and the executor must bind exactly
/// what `declare` said it would read. The same reason the bloom upsample's
/// coarse level is looked up the same way in both places.
#[derive(Clone, Copy, Debug)]
pub struct DofIds {
    pub source: ResourceId,
    /// Half the frame: colour, with the signed circle of confusion in alpha.
    pub prefiltered: ResourceId,
    /// The widest near and far circle per tile. What the gather sizes its kernel
    /// from, so a fixed tap budget lands inside the blur rather than around it.
    pub tile: ResourceId,
    /// The two fields, kept apart because they composite differently — a near
    /// one spills over what it occludes and a far one does not.
    pub near: ResourceId,
    pub far: ResourceId,
    pub output: ResourceId,
}

/// Motion blur's velocity pyramid, and what it was handed.
#[derive(Clone, Copy, Debug)]
pub struct MotionBlurIds {
    pub source: ResourceId,
    /// The longest blur vector per tile, and that dilated by one ring.
    pub tile: ResourceId,
    pub neighbour: ResourceId,
    pub output: ResourceId,
}

pub struct Frame {
    pub graph: FrameGraph,
    pub ids: FrameIds,
    /// Indexed by [`PassId`], so a pass's declarations and its body cannot drift.
    pub bodies: Vec<PassBody>,
}

/// Declares the draw commands on a pass that issues its draws indirectly.
///
/// One function rather than the declaration written out at each of the four
/// sites, so that an opaque geometry pass cannot draw indirectly without being
/// ordered after the cull: it goes through here or it does not compile against
/// the commands at all.
fn indirect(pass: PassBuilder<'_>, commands: ResourceId) -> PassBuilder<'_> {
    pass.access(commands, Access::IndirectRead)
}

/// Declares what a pass that lights a surface reads of the irradiance
/// volumes, where they are live: one function for every such pass, so none
/// can read them without being ordered after their update.
fn reads_gi(pass: PassBuilder<'_>, gi: Option<GiIds>) -> PassBuilder<'_> {
    match gi {
        Some(ids) => pass
            .access(ids.irradiance, Access::Sampled)
            .access(ids.distance, Access::Sampled)
            .access(ids.probes, Access::StorageRead),
        None => pass,
    }
}

/// What a shadow atlas is: one image every view is a tile of, its pages side
/// by side.
pub fn shadow_atlas_desc(resolution: u32) -> ImageDesc {
    ImageDesc::new(SHADOW_FORMAT).extent(Extent::Fixed(atlas_extent(resolution)))
}

pub fn declare(config: FrameConfig) -> Result<Frame, GraphError> {
    let mut builder = GraphBuilder::new();
    // A pass that shades a ray's hit reads the scene itself: every material's
    // textures, the shared geometry, the material table. Those belong to the
    // graphics queue's family alone — the images the two queues share are the
    // frame's own, which is what the compiler can name and make concurrent —
    // so a frame with such a pass among its dispatches is not split, or its
    // tail would read them from a family that does not own them.
    let shades_hits = config.ray_reflections || config.ray_path_trace || config.gi_probes > 0;
    let gathers = config.ray_tracing && config.gi_probes > 0 && config.gi_gather > 0;
    if config.async_compute && !shades_hits {
        builder.request_async_compute();
    }
    let mut bodies = Vec::new();
    let record = |id: PassId, body: PassBody, bodies: &mut Vec<PassBody>| {
        debug_assert_eq!(id.index(), bodies.len());
        bodies.push(body);
    };

    // Read by every geometry pass. The rows persist between frames and are
    // brought up to date by a copy recorded ahead of the schedule — the
    // per-object inverse-transpose is too expensive to compute twice, so the
    // passes share one buffer and the graph records that they do.
    let object_transforms = builder.import_buffer("object_transforms");
    // The opaque draw orders over those rows, which the cull writes afresh
    // every frame. A second resource rather than a share of the first because
    // they have different lifetimes, and the graph's whole claim is to know each
    // resource's exact one. See `vulkan::instances`.
    let instance_index = builder.import_buffer("instance_index");
    // Culling is two dispatches because a count can only be atomically
    // incremented from a known value. The reset stamps each command with the
    // geometry its batch draws and an instance count of zero; the cull tests
    // every live instance against every view and increments the counts it
    // survives, appending its row to that batch's slice.
    //
    // Two graph nodes rather than one dispatch with a barrier in the middle,
    // because a pass may not write its own barrier — the point of the compiler
    // is that the write-after-write between these two is derived rather than
    // remembered.
    // The blended and refractive queues are CPU-culled — a refractive surface
    // has to be drawn back to front, which is a correctness requirement no
    // atomic append can meet — so their orders are written by the host, into a
    // buffer of their own. A second resource rather than a share of the first
    // because it really is a second allocation, and one name over two buffers
    // is precisely the declaration this compiler exists to catch.
    let blended_index = builder.import_buffer("blended_instance_index");

    let occlusion_ids = config.occlusion_culling.then(|| {
        // Entry `ShaderReadOnlyOptimal` states the steady state the ping-pong
        // guarantees: `hiz` leaves every frame in exactly that layout and is
        // the allocation `history` names next frame. On the first frame after an
        // allocation the layout is still true — `initialise_sampled` puts it
        // there — and its contents are what the cull is told to ignore.
        let history = builder.import_image(
            "occlusion_history",
            ImageDesc::new(OCCLUSION_FORMAT)
                .extent(Extent::FrameDiv(1))
                .mip_levels(OCCLUSION_LEVELS),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        // Entered `Undefined` because every texel is written, and left where
        // the next frame wants to find it.
        let hiz = builder.import_image(
            "occlusion_hiz",
            ImageDesc::new(OCCLUSION_FORMAT)
                .extent(Extent::FrameDiv(1))
                .mip_levels(OCCLUSION_LEVELS),
            ImageLayout::Undefined,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        OcclusionIds { history, hiz }
    });

    let draw_commands = {
        // Which rows the dispatch is offered, and which batch each belongs to.
        // Written afresh every frame, like the blended draw orders and unlike
        // the rows themselves — the batch a row belongs to is
        // cached, but the list of rows is not. See `vulkan::cull`.
        let instance_table = builder.import_buffer("instance_table");
        // One entry per (mesh, material) batch: the geometry its draw command
        // names and where its slice of a view's `instance_index` block begins.
        let batch_table = builder.import_buffer("batch_table");
        // One `VkDrawIndexedIndirectCommand` per batch per view. Imported
        // rather than transient because it is the same allocation frame after
        // frame and only its contents change, which is the argument
        // `object_transforms` makes one comment above.
        let draw_commands = builder.import_buffer("draw_commands");

        let id = builder
            .pass("cull_reset", PassKind::Compute)
            .access(batch_table, Access::StorageRead)
            .access(draw_commands, Access::StorageWrite)
            .build();
        record(id, PassBody::CullReset, &mut bodies);

        let mut cull = builder
            .pass("cull", PassKind::Compute)
            .access(object_transforms, Access::StorageRead)
            .access(instance_table, Access::StorageRead)
            .access(batch_table, Access::StorageRead)
            .access(draw_commands, Access::StorageWrite)
            .access(instance_index, Access::StorageWrite);
        if let Some(ids) = occlusion_ids {
            cull = cull.access(ids.history, Access::Sampled);
        }
        let id = cull.build();
        record(id, PassBody::Cull, &mut bodies);

        draw_commands
    };

    // At the head of the frame, so anything after may trace. Two nodes and
    // not one for the reason the cull is two: the scene's structure is built
    // over the meshes', some of which this frame built, and the barrier between
    // a build and a read of what it built is the compiler's to derive.
    let ray = config.ray_tracing.then(|| {
        let meshes = builder.import_acceleration_structure("ray_meshes");
        let scene = builder.import_acceleration_structure("ray_scene");

        let id = builder
            .pass("ray_meshes", PassKind::Compute)
            .access(meshes, Access::AccelerationStructureBuild)
            .build();
        record(id, PassBody::RayMeshes, &mut bodies);

        let id = builder
            .pass("ray_scene", PassKind::Compute)
            .access(meshes, Access::AccelerationStructureInput)
            .access(scene, Access::AccelerationStructureBuild)
            .build();
        record(id, PassBody::RayScene, &mut bodies);

        RayIds { meshes, scene }
    });

    // Right behind the scene's build, because nothing here reads anything the
    // frame rasterises: the volumes are the world's, not the view's, and every
    // pass that lights a surface reads them. Four nodes for four steps that
    // each need the last finished on the device: the probes, the rays, the
    // cache's list of what they hit, and those cells shaded.
    //
    // The order the cache imposes is the one place this frame reads before it
    // writes. The shading reads the volumes, for the light that bounced, and
    // the update writes them; the update reads the cache's light, and the
    // shading writes it. The update reads it for every cell its rays landed in
    // shaded, so it takes in last frame's rays, from the half of the pair last
    // frame's shading wrote, and runs first — the cycle the occlusion pyramids
    // break the same way, with the rays as a pair as well.
    let gi = ray.filter(|_| config.gi_probes > 0).map(|ray| {
        let atlas = || {
            ImageDesc::new(super::gi::ATLAS_FORMAT).extent(Extent::Fixed(
                crate::gfx::probes::atlas_extent(config.gi_probes),
            ))
        };
        // Entered and left sampled, for the reason every history is: the
        // update reads each texel it writes, and everything after samples.
        let irradiance = builder.import_image(
            "gi_irradiance",
            atlas(),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let distance = builder.import_image(
            "gi_distance",
            atlas(),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let bounce = builder.import_image(
            "gi_bounce",
            atlas(),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let outlier = builder.import_image(
            "gi_outlier",
            atlas(),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let ids = GiIds {
            irradiance,
            distance,
            bounce,
            outlier,
            probes: builder.import_buffer("gi_probes"),
            rays: builder.import_buffer("gi_rays"),
            rays_previous: builder.import_buffer("gi_rays_previous"),
            cells: builder.import_buffer("gi_cells"),
            list: builder.import_buffer("gi_list"),
            args: builder.import_buffer("gi_args"),
            radiance_history: builder.import_buffer("gi_radiance_history"),
            radiance: builder.import_buffer("gi_radiance"),
            gather_hits: gathers.then(|| builder.import_buffer("gi_gather_hits")),
        };

        let id = builder
            .pass("gi_update", PassKind::Compute)
            .access(ids.probes, Access::StorageWrite)
            .access(ids.rays_previous, Access::StorageRead)
            .access(ids.radiance_history, Access::StorageRead)
            .access(irradiance, Access::StorageWrite)
            .access(distance, Access::StorageWrite)
            .access(bounce, Access::StorageWrite)
            .access(outlier, Access::StorageWrite)
            .build();
        record(id, PassBody::GiUpdate, &mut bodies);

        // The list and its dispatch are emptied here, ahead of the pass that
        // fills them.
        let id = builder
            .pass("gi_trace", PassKind::Compute)
            .access(ray.scene, Access::AccelerationStructureRead)
            .access(ids.probes, Access::StorageRead)
            .access(ids.rays, Access::StorageWrite)
            .access(ids.cells, Access::StorageWrite)
            .access(ids.list, Access::StorageWrite)
            .access(ids.args, Access::StorageWrite)
            .build();
        record(id, PassBody::GiTrace, &mut bodies);

        let id = builder
            .pass("gi_cache_compact", PassKind::Compute)
            .access(ids.cells, Access::StorageWrite)
            .access(ids.radiance_history, Access::StorageRead)
            .access(ids.radiance, Access::StorageWrite)
            .access(ids.list, Access::StorageWrite)
            .access(ids.args, Access::StorageWrite)
            .build();
        record(id, PassBody::GiCacheCompact, &mut bodies);

        let mut shade = builder
            .pass("gi_cache_shade", PassKind::Compute)
            .access(ray.scene, Access::AccelerationStructureRead)
            .access(ids.args, Access::IndirectRead)
            .access(ids.list, Access::StorageRead)
            .access(ids.cells, Access::StorageWrite)
            .access(ids.rays, Access::StorageRead)
            .access(ids.probes, Access::StorageRead)
            .access(bounce, Access::Sampled)
            .access(distance, Access::Sampled)
            .access(ids.radiance, Access::StorageWrite);
        // Last frame's gather rays, read for the cells they asked for and
        // declared written, for the reason the trace declares the probes'
        // state: the gather writes this frame's after, and a read would be
        // ordered after every write.
        if let Some(hits) = ids.gather_hits {
            shade = shade.access(hits, Access::StorageWrite);
        }
        record(shade.build(), PassBody::GiCacheShade, &mut bodies);

        ids
    });

    let swapchain_color = builder.import_image(
        "swapchain_color",
        ImageDesc::new(config.color_format),
        // An acquired image's contents are not ours to keep, and the
        // presentation engine wants it back in `PresentSrc` — or, offscreen,
        // the readback wants it in the layout it copies from.
        ImageLayout::Undefined,
        if config.presents {
            ImageLayout::PresentSrc
        } else {
            ImageLayout::TransferSrcOptimal
        },
    );

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

    // Geometry rasterised from each light's point of view, needed before
    // anything shades. One node for every view of every light — a tile is a
    // viewport into this attachment, so four sun views and six faces of eight
    // lamps are still one barrier.
    // Imported rather than graph-owned, both of them, because a view that is
    // not redrawn keeps its tile from an earlier frame and a transient is
    // `Undefined` at the start of each. Entered and left
    // `ShaderReadOnlyOptimal`, the layout the frame's readers leave them in, so
    // those texels survive the round trip through the attachment layout. The
    // shadow pass allocates them to exactly this description.
    let shadow_atlases = (config.shadow_atlas > 0).then(|| {
        let mut import = |name| {
            builder.import_image(
                name,
                shadow_atlas_desc(config.shadow_atlas),
                ImageLayout::ShaderReadOnlyOptimal,
                ImageLayout::ShaderReadOnlyOptimal,
            )
        };
        let (cached, atlas) = (import("shadow_static"), import("shadow_atlas"));

        // Geometry rasterised from each light's point of view, needed before
        // anything shades. Two nodes for every view of every light — a tile is
        // a viewport into an attachment, so four sun views and six faces of
        // eight lamps are a barrier per layer rather than per view. The cached
        // layer first, because the live one is brought up to date from it.
        let id = indirect(
            builder
                .pass("shadow_static", PassKind::Inline)
                .access(object_transforms, Access::StorageRead)
                .access(instance_index, Access::StorageRead),
            draw_commands,
        )
        .access(cached, Access::DepthAttachment)
        .build();
        record(id, PassBody::ShadowStatic, &mut bodies);

        let id = indirect(
            builder
                .pass("shadows", PassKind::Inline)
                .access(object_transforms, Access::StorageRead)
                .access(instance_index, Access::StorageRead),
            draw_commands,
        )
        .access(cached, Access::Sampled)
        .access(atlas, Access::DepthAttachment)
        .build();
        record(id, PassBody::Shadows, &mut bodies);

        (atlas, cached)
    });
    let shadow_atlas = shadow_atlases.map(|(atlas, _)| atlas);

    // One prepass serves every consumer rather than one each: they need the same
    // rasterisation, and running it twice to hand each part of the result to a
    // different reader is the cost the shared node exists to avoid. It writes
    // all three targets whichever consumer asked for it — a second pipeline that
    // dropped the normal attachment for a TAA-without-SSAO frame would buy a
    // target's bandwidth at the price of a second render pass to keep in step.
    // Seven readers now want different parts of it: SSAO and contact shadows
    // take depth and normals, TAA and motion blur take depth and motion, depth
    // of field takes depth alone, reflections take all but motion — and
    // transparency and refraction both attach the depth
    // read-only, the two readers that want it as an attachment rather than as a
    // texture.
    //
    // And one more, which is why `!config.msaa` is in the list: at one sample
    // the forward pass does not rasterise a depth buffer of its own at all. It
    // attaches this one read-only and tests `EQUAL` against it, which is what
    // turns the prepass from a cost the forward pass ignores into perfect
    // early-Z with no shading overdraw.
    let prepass = (!config.msaa
        || config.ssao
        || config.contact_shadows
        || config.subsurface
        || config.taa
        || config.motion_blur
        || config.dof
        || config.ssr
        || config.transparency
        || config.refraction
        || config.occlusion_culling
        || gathers
        // What the wireframe hides covered edges with.
        || (config.overlay && config.wireframe == Some(super::wireframe::Wireframe::OverShaded)))
    .then(|| PrepassIds {
        normal: builder.create_image("prepass_normal", ImageDesc::new(NORMAL_FORMAT)),
        velocity: builder.create_image("prepass_velocity", ImageDesc::new(VELOCITY_FORMAT)),
        material: builder.create_image("prepass_material", ImageDesc::new(MATERIAL_FORMAT)),
        depth: builder.create_image("prepass_depth", ImageDesc::new(DEPTH_FORMAT)),
    });

    // Both AO images at one extent, whichever it is: the blur samples the
    // resolve's output texel for texel, and the forward pass reads the result by
    // screen UV through a linear sampler, so the only pass that has to know
    // which extent this is is the one setting the viewport.
    let ao_extent = if config.ssao_half_res {
        Extent::FrameDiv(1)
    } else {
        Extent::Frame
    };
    let ssao = config.ssao.then(|| SsaoIds {
        raw_ao: builder.create_image("ssao_raw_ao", ImageDesc::new(AO_FORMAT).extent(ao_extent)),
        ao: builder.create_image("ssao_ao", ImageDesc::new(AO_FORMAT).extent(ao_extent)),
    });

    let msaa = config.msaa.then(|| MsaaIds {
        hdr: builder.create_image(
            "msaa_hdr",
            ImageDesc::new(HDR_WIDE_FORMAT).samples(MSAA_SAMPLES),
        ),
        depth: builder.create_image(
            "msaa_depth",
            ImageDesc::new(DEPTH_FORMAT).samples(MSAA_SAMPLES),
        ),
    });
    // The packed format, except in a multisampled frame. Two constraints force
    // that exception and either alone would be enough: a render-pass resolve
    // requires the source and destination formats to match, and alpha to
    // coverage reads the first attachment's alpha — which an attachment with no
    // alpha component supplies as 1.0, quietly un-cutting every cutout.
    let color_format = if config.msaa {
        HDR_WIDE_FORMAT
    } else {
        HDR_FORMAT
    };
    let hdr_color = builder.create_image("hdr_color", ImageDesc::new(color_format));

    if let Some(prepass) = prepass {
        let id = indirect(
            builder
                .pass("geometry_prepass", PassKind::Inline)
                .access(object_transforms, Access::StorageRead)
                .access(instance_index, Access::StorageRead),
            draw_commands,
        )
        .access(prepass.normal, Access::ColorAttachment)
        .access(prepass.velocity, Access::ColorAttachment)
        .access(prepass.material, Access::ColorAttachment)
        .access(prepass.depth, Access::DepthAttachment)
        .build();
        record(id, PassBody::GeometryPrepass, &mut bodies);
    }

    // Registered here because this is where the depth it reduces becomes
    // complete, and scheduled here for the same reason — nothing downstream
    // waits on it, so the earliest slot after the prepass is also the one with
    // the most of the frame left to hide it behind.
    //
    // One dispatch writes every level, which is not a choice: a pass per level,
    // each reading the level above out of the same image, is the cycle
    // `ImageDesc::mip_levels` documents. See `hiz_occlusion.comp` for the tile
    // that buys.
    if let Some(ids) = occlusion_ids {
        let prepass = prepass.expect("occlusion culling reduces the geometry prepass's depth");
        let id = builder
            .pass("occlusion_hiz", PassKind::Compute)
            .access(prepass.depth, Access::Sampled)
            .access(ids.hiz, Access::StorageWrite)
            .build();
        record(id, PassBody::OcclusionHiz, &mut bodies);
    }

    if let Some(ssao) = ssao {
        let prepass = prepass.expect("SSAO reads the geometry prepass");
        let id = builder
            .pass("ssao_resolve", PassKind::Inline)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(ssao.raw_ao, Access::ColorAttachment)
            .build();
        record(id, PassBody::SsaoResolve, &mut bodies);

        // The depth as well as the raw term: the blur is bilateral, and at half
        // resolution it is also the upsample. See `ssao_blur.frag`.
        let id = builder
            .pass("ssao_blur", PassKind::Inline)
            .access(ssao.raw_ao, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(ssao.ao, Access::ColorAttachment)
            .build();
        record(id, PassBody::SsaoBlur, &mut bodies);
    }

    // Between the prepass and shading, and it can be nowhere else: it marches
    // the depth that pass wrote, and the mask it leaves is what the forward pass
    // multiplies the sun's term by. Unblurred, unlike the AO beside it — this is
    // a hard visibility term whose noise is a dithered ray origin, and a spatial
    // filter would smear the contact it exists to sharpen. The temporal resolve
    // is what averages it.
    let contact_shadows = config.contact_shadows.then(|| {
        let prepass = prepass.expect("contact shadows read the geometry prepass");
        let mask = builder.create_image("contact_shadow_mask", ImageDesc::new(MASK_FORMAT));

        let id = builder
            .pass("contact_shadows", PassKind::Inline)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(mask, Access::ColorAttachment)
            .build();
        record(id, PassBody::ContactShadows, &mut bodies);

        mask
    });

    // Between the prepass and shading, where SSAO is, and for its reasons: it
    // reads the prepass and nothing the forward pass draws, and the forward
    // pass reads what it leaves by pixel. After the volumes and the cache's
    // shading, whose light its rays read and whose cells they claim.
    let gather = gi.filter(|_| gathers).map(|gi| {
        use super::gi_gather::{GATHER_FORMAT, HISTORY_LAYERS, META_LAYERS, SH_LAYERS};
        let prepass = prepass.expect("the final gather reads the geometry prepass");
        let ray = ray.expect("the final gather traces");
        let hits = gi.gather_hits.expect("declared with the volumes");
        let low = |layers| {
            ImageDesc::new(GATHER_FORMAT)
                .extent(Extent::FrameDiv(config.gi_gather))
                .array_layers(layers)
        };
        let raw = builder.create_image("gi_gather_raw", low(META_LAYERS));
        // Entered and left sampled, for the reason every history is; and the
        // allocation `accumulated` names this frame is `history` next frame,
        // so it leaves where `history` enters. Entered `Undefined` because
        // every texel is written.
        let history = builder.import_image(
            "gi_gather_history",
            low(HISTORY_LAYERS),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let accumulated = builder.import_image(
            "gi_gather_accumulated",
            low(HISTORY_LAYERS),
            ImageLayout::Undefined,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let blurred_x = builder.create_image("gi_gather_blur_x", low(SH_LAYERS));
        let blurred = builder.create_image("gi_gather_blur_y", low(SH_LAYERS));
        let wide_x = builder.create_image("gi_gather_blur_wide_x", low(SH_LAYERS));
        let wide = builder.create_image("gi_gather_blur_wide_y", low(SH_LAYERS));
        let irradiance = builder.create_image(
            "gi_gather_irradiance",
            ImageDesc::new(GATHER_FORMAT).array_layers(SH_LAYERS),
        );

        // The cache's cells are written: a cell a ray lands in is claimed.
        let pass = builder
            .pass("gi_gather", PassKind::Compute)
            .access(ray.scene, Access::AccelerationStructureRead)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled);
        let id = reads_gi(pass, Some(gi))
            .access(gi.cells, Access::StorageWrite)
            .access(gi.radiance, Access::StorageRead)
            .access(hits, Access::StorageWrite)
            .access(raw, Access::StorageWrite)
            .build();
        record(id, PassBody::GiGather, &mut bodies);

        let id = builder
            .pass("gi_gather_temporal", PassKind::Compute)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(prepass.velocity, Access::Sampled)
            .access(raw, Access::Sampled)
            .access(history, Access::Sampled)
            .access(accumulated, Access::StorageWrite)
            .build();
        record(id, PassBody::GiGatherTemporal, &mut bodies);

        // Two images and not one, for the reason the subsurface blur has two:
        // a pass that read and wrote one would be a cycle.
        let id = builder
            .pass("gi_gather_blur_x", PassKind::Compute)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(accumulated, Access::Sampled)
            .access(blurred_x, Access::StorageWrite)
            .build();
        record(
            id,
            PassBody::GiGatherBlur {
                vertical: false,
                wide: false,
            },
            &mut bodies,
        );

        let id = builder
            .pass("gi_gather_blur_y", PassKind::Compute)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(blurred_x, Access::Sampled)
            .access(accumulated, Access::Sampled)
            .access(blurred, Access::StorageWrite)
            .build();
        record(
            id,
            PassBody::GiGatherBlur {
                vertical: true,
                wide: false,
            },
            &mut bodies,
        );

        // The second level, into two images more: neither outlives the first
        // level's, so each takes the other's allocation.
        let id = builder
            .pass("gi_gather_blur_wide_x", PassKind::Compute)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(blurred, Access::Sampled)
            .access(accumulated, Access::Sampled)
            .access(wide_x, Access::StorageWrite)
            .build();
        record(
            id,
            PassBody::GiGatherBlur {
                vertical: false,
                wide: true,
            },
            &mut bodies,
        );

        let id = builder
            .pass("gi_gather_blur_wide_y", PassKind::Compute)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(wide_x, Access::Sampled)
            .access(accumulated, Access::Sampled)
            .access(wide, Access::StorageWrite)
            .build();
        record(
            id,
            PassBody::GiGatherBlur {
                vertical: true,
                wide: true,
            },
            &mut bodies,
        );

        let id = builder
            .pass("gi_gather_upscale", PassKind::Compute)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(wide, Access::Sampled)
            .access(irradiance, Access::StorageWrite)
            .build();
        record(id, PassBody::GiGatherUpscale, &mut bodies);

        GatherIds {
            raw,
            history,
            accumulated,
            blurred_x,
            blurred,
            wide_x,
            wide,
            irradiance,
        }
    });

    // After the cascades, because it reads them, and before every geometry pass,
    // because all three sample the result. It needs no prepass and takes no
    // screen-space input at all — a froxel is a point in the world, not a pixel —
    // which is what lets the whole effect sit here rather than in the composite
    // chain after shading, and is why it fogs blended and refractive surfaces at
    // their own depth instead of at the opaque depth behind them.
    let fog = config.volumetric_fog.then(|| {
        // Entry `ShaderReadOnlyOptimal` states the steady state, which the
        // ping-pong guarantees: this leaves every frame in exactly that layout
        // and is the allocation the *history* names next frame. On the first
        // frame after an allocation the layout is still true — the allocation
        // put it there — and its contents are what the scatter pass ignores.
        let scatter = builder.import_image(
            "fog_scatter",
            ImageDesc::new(FOG_FORMAT)
                .extent(Extent::FrameDiv(FROXEL_SHIFT))
                .depth(FROXEL_SLICES),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let volume = builder.create_image(
            "fog_volume",
            ImageDesc::new(FOG_FORMAT)
                .extent(Extent::FrameDiv(FROXEL_SHIFT))
                .depth(FROXEL_SLICES),
        );

        // The light the air scatters from everything round it, traced where
        // the volumes are live: after the cache's shading and the gather's
        // claims, whose cells its rays read, and with the volumes beside it for
        // a ray that lands where the cache has no light yet.
        let traced = ray.zip(gi).map(|(ray, gi)| {
            let light = builder.import_image(
                "fog_light",
                ImageDesc::new(FOG_FORMAT)
                    .extent(Extent::FrameDiv(FROXEL_SHIFT + LIGHT_SHIFT))
                    .depth(FROXEL_SLICES >> LIGHT_SHIFT),
                ImageLayout::ShaderReadOnlyOptimal,
                ImageLayout::ShaderReadOnlyOptimal,
            );
            let state = builder.import_image(
                "fog_light_state",
                ImageDesc::new(LIGHT_STATE_FORMAT)
                    .extent(Extent::FrameDiv(FROXEL_SHIFT + LIGHT_SHIFT))
                    .depth(FROXEL_SLICES >> LIGHT_SHIFT),
                ImageLayout::ShaderReadOnlyOptimal,
                ImageLayout::ShaderReadOnlyOptimal,
            );
            let pass = builder
                .pass("fog_light", PassKind::Compute)
                .access(ray.scene, Access::AccelerationStructureRead)
                .access(gi.cells, Access::StorageRead)
                .access(gi.radiance, Access::StorageRead);
            let id = reads_gi(pass, Some(gi))
                .access(light, Access::StorageWrite)
                .access(state, Access::StorageWrite)
                .build();
            record(id, PassBody::FogLight, &mut bodies);
            (light, state)
        });
        let light = traced.map(|(light, _)| light);

        // The history is the *other* allocation of `scatter`, which the graph
        // cannot see and does not need to: the executor binds it, and a resource
        // read and written by one pass is one point in the schedule either way.
        let mut scatter_pass = builder.pass("fog_scatter", PassKind::Compute);
        if let Some(atlas) = shadow_atlas {
            scatter_pass = scatter_pass.access(atlas, Access::Sampled);
        }
        if let Some(light) = light {
            scatter_pass = scatter_pass.access(light, Access::Sampled);
        }
        let id = scatter_pass.access(scatter, Access::StorageWrite).build();
        record(id, PassBody::FogScatter, &mut bodies);

        let id = builder
            .pass("fog_integrate", PassKind::Compute)
            .access(scatter, Access::Sampled)
            .access(volume, Access::StorageWrite)
            .build();
        record(id, PassBody::FogIntegrate, &mut bodies);

        FogIds {
            light,
            light_state: traced.map(|(_, state)| state),
            scatter,
            volume,
        }
    });

    // Declared before the pass that writes them, because the forward pass has to
    // declare them as attachments: a second colour target and its resolve, in the
    // same shape as the frame's first pair, and only in a frame that diffuses.
    let subsurface_targets = config.subsurface.then(|| {
        (
            msaa.map(|_| {
                builder.create_image(
                    "msaa_subsurface",
                    ImageDesc::new(SUBSURFACE_FORMAT).samples(MSAA_SAMPLES),
                )
            }),
            builder.create_image("subsurface_diffusible", ImageDesc::new(SUBSURFACE_FORMAT)),
        )
    });

    let mut forward = indirect(
        builder
            .pass("forward", PassKind::Inline)
            .access(object_transforms, Access::StorageRead)
            .access(instance_index, Access::StorageRead),
        draw_commands,
    );
    if let Some(ssao) = ssao {
        forward = forward.access(ssao.ao, Access::Sampled);
    }
    if let Some(mask) = contact_shadows {
        forward = forward.access(mask, Access::Sampled);
    }
    if let Some(atlas) = shadow_atlas {
        forward = forward.access(atlas, Access::Sampled);
    }
    if let Some(fog) = fog {
        forward = forward.access(fog.volume, Access::Sampled);
    }
    forward = reads_gi(forward, gi);
    if let Some(gather) = gather {
        forward = forward.access(gather.irradiance, Access::Sampled);
    }
    forward = match msaa {
        Some(msaa) => forward
            .access(msaa.hdr, Access::ColorAttachment)
            .access(msaa.depth, Access::DepthAttachment)
            .access(hdr_color, Access::ResolveAttachment),
        // One sample, so there is nothing to resolve from: shade straight into
        // the image the resolve used to land in, and attach the depth the
        // prepass wrote read-only. `DepthAttachmentRead` is the declaration that
        // makes it legal for the passes after this one to keep sampling it —
        // the image never leaves `DepthStencilReadOnlyOptimal`, which is the
        // same contract `oit_accumulate` and `refraction_draw` already sign.
        None => forward.access(hdr_color, Access::ColorAttachment).access(
            prepass
                .expect("a one-sample forward pass depth-tests against the prepass")
                .depth,
            Access::DepthAttachmentRead,
        ),
    };
    // Between the first colour attachment and its resolve in the render pass's
    // own declaration order, but the graph does not care about order — only that
    // an attachment declared here is one the framebuffer binds. `PassFramebuffers`
    // is what keeps the two lists in step.
    if let Some((msaa_diffusible, diffusible)) = subsurface_targets {
        forward = match msaa_diffusible {
            Some(msaa_diffusible) => forward
                .access(msaa_diffusible, Access::ColorAttachment)
                .access(diffusible, Access::ResolveAttachment),
            None => forward.access(diffusible, Access::ColorAttachment),
        };
    }
    let id = forward.build();
    record(id, PassBody::Forward, &mut bodies);

    // Immediately after shading and before everything that reads the lit frame.
    // Both halves of that placement are load-bearing: there is nothing to diffuse
    // until the surface has been lit, and the reflections, the blended queue and
    // the refractive queue all sample the frame's colour — a face that scatters
    // should scatter in a mirror beside it too, and the only way to get that for
    // free is to be the frame those passes are handed.
    let subsurface = subsurface_targets.map(|(msaa_diffusible, diffusible)| {
        let prepass = prepass.expect("subsurface diffusion reads the geometry prepass");
        let blurred_x =
            builder.create_image("subsurface_blur_x", ImageDesc::new(SUBSURFACE_FORMAT));
        let blurred_y =
            builder.create_image("subsurface_blur_y", ImageDesc::new(SUBSURFACE_FORMAT));
        let output = builder.create_image("subsurface_color", ImageDesc::new(HDR_FORMAT));

        let id = builder
            .pass("subsurface_blur_x", PassKind::Compute)
            .access(diffusible, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(blurred_x, Access::StorageWrite)
            .build();
        record(id, PassBody::SubsurfaceBlurHorizontal, &mut bodies);

        let id = builder
            .pass("subsurface_blur_y", PassKind::Compute)
            .access(blurred_x, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(blurred_y, Access::StorageWrite)
            .build();
        record(id, PassBody::SubsurfaceBlurVertical, &mut bodies);

        let id = builder
            .pass("subsurface_composite", PassKind::Compute)
            .access(hdr_color, Access::Sampled)
            .access(blurred_y, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::SubsurfaceComposite, &mut bodies);

        SubsurfaceIds {
            source: hdr_color,
            msaa_diffusible,
            diffusible,
            blurred_x,
            blurred_y,
            output,
        }
    });

    // What everything downstream treats as the lit frame. Named once, so inserting
    // a stage here is a change to this binding rather than to every consumer — the
    // same rule `scene_color` follows at the end of the optical chain.
    let lit_color = subsurface.map_or(hdr_color, |ids| ids.output);

    // Between shading and the temporal resolve, and it has to be both: a ray can
    // only sample radiance that has been lit, and the reflection has to be
    // antialiased with the rest of the frame rather than composited over the
    // top of it.
    let ssr = config.ssr.then(|| {
        let prepass = prepass.expect("screen-space reflections read the geometry prepass");
        let source = lit_color;

        // One image carrying the whole pyramid rather than one per level, which
        // is what lets the trace pick a level per step by index. The build
        // writes every level in a single pass because it must: a pass per
        // level, each reading the level above from the same resource, is a cycle
        // by the rule that makes readers follow writers. Based at half the
        // frame: its level 0 would be the depth buffer, which the trace reads.
        let hiz = builder.create_image(
            "ssr_hiz",
            ImageDesc::new(HIZ_FORMAT)
                .extent(Extent::FrameDiv(1))
                .mip_levels(HIZ_LEVELS),
        );
        // The same shape over the lit frame, and what a rough reflection is
        // made of: the level whose texel matches the lobe's width at the hit
        // carries the blur the lobe would see, in one fetch. Based at half the
        // frame, which is the resolution the trace works at anyway.
        let source_pyramid = builder.create_image(
            "ssr_source",
            ImageDesc::new(HDR_FORMAT)
                .extent(Extent::FrameDiv(1))
                .mip_levels(SOURCE_LEVELS),
        );
        // Half the frame, for the reason depth of field's gather is: the cost is
        // the ray, and a glossy surface's blur is wider than a quad, so tracing
        // four times as many buys a sharper version of something the roughness
        // then blurs. A smooth surface traces the rest of its quad in the
        // composite — which is why that pass reads the pyramids too.
        let rays = builder.create_image(
            "ssr_rays",
            ImageDesc::new(RAY_FORMAT).extent(Extent::FrameDiv(1)),
        );
        let output = builder.create_image("ssr_color", ImageDesc::new(HDR_FORMAT));

        let id = builder
            .pass("ssr_hiz", PassKind::Compute)
            .access(prepass.depth, Access::Sampled)
            .access(hiz, Access::StorageWrite)
            .build();
        record(id, PassBody::SsrHiz, &mut bodies);

        let id = builder
            .pass("ssr_source", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(source_pyramid, Access::StorageWrite)
            .build();
        record(id, PassBody::SsrSource, &mut bodies);

        // Both passes that trace a reflection fall back to the scene's
        // structure where the screen has no answer, so both wait on its build.
        let traced = ray.filter(|_| config.ray_reflections);
        let mut trace = builder
            .pass("ssr_trace", PassKind::Compute)
            .access(hiz, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(prepass.material, Access::Sampled)
            .access(source_pyramid, Access::Sampled)
            .access(source, Access::Sampled)
            .access(rays, Access::StorageWrite);
        if let Some(ray) = traced {
            // A hit's bounce light is the volumes'.
            trace = reads_gi(
                trace.access(ray.scene, Access::AccelerationStructureRead),
                gi,
            );
        }
        record(trace.build(), PassBody::SsrTrace, &mut bodies);

        // The gather and the composite in one pass: what the gather makes for a
        // pixel is read by that pixel alone, so handing it over through a
        // full-resolution image would be a dispatch and a write for nothing.
        //
        // Reads `source` again rather than accumulating into it, for the reason
        // the depth-of-field composite does: resources are unversioned, so a
        // pass that both read and wrote the frame's colour would make "readers
        // after all writers" point in two directions and `compile` would report
        // a cycle.
        let mut resolve = builder
            .pass("ssr_resolve", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(rays, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(prepass.material, Access::Sampled)
            .access(hiz, Access::Sampled)
            .access(source_pyramid, Access::Sampled)
            .access(output, Access::StorageWrite);
        if let Some(ray) = traced {
            resolve = resolve.access(ray.scene, Access::AccelerationStructureRead);
        }
        // What the environment's reflection it takes back was scaled by.
        resolve = reads_gi(resolve, gi);
        record(resolve.build(), PassBody::SsrResolve, &mut bodies);

        SsrIds {
            source,
            hiz,
            source_pyramid,
            rays,
            output,
            traced: traced.is_some(),
        }
    });

    // What the temporal resolve accumulates, and what the optical chain starts
    // from in a frame with no resolve: the reflections' output where they ran,
    // and the forward pass's own target where they did not.
    let mut shaded = ssr.map_or(lit_color, |ssr| ssr.output);

    // After the reflections and before the resolve, and both halves matter.
    //
    // After, because the prepass records only opaque surfaces: a reflection
    // traces the depth and normals of the world *behind* the glass, and
    // compositing first would put the glass into a source the trace then
    // reflects as though it were a wall.
    //
    // Before, because the accumulation rasterises with the frame's jitter like
    // everything else, and a subpixel offset nothing averages is a shimmer. The
    // cost is that a moving transparent surface reprojects along the *opaque*
    // motion vectors under it and ghosts; the neighbourhood clamp takes most of
    // it, and the alternative is transparency with no antialiasing at all.
    let transparency = config.transparency.then(|| {
        let prepass = prepass.expect("transparency depth-tests the geometry prepass");
        let source = shaded;

        let accum = builder.create_image("oit_accum", ImageDesc::new(ACCUM_FORMAT));
        let reveal = builder.create_image("oit_reveal", ImageDesc::new(REVEAL_FORMAT));
        let output = builder.create_image("oit_color", ImageDesc::new(HDR_FORMAT));

        // The depth is *attached*, not sampled: a fixed-function depth test is
        // what makes a transparent surface disappear behind a wall, and a test
        // in the shader would need the depth in a second layout and a discard
        // per fragment. `DepthAttachmentRead` is the declaration that keeps it
        // read-only — the prepass is still its only writer, so every other
        // reader of it is unaffected.
        //
        // The same screen-space and shadow inputs the forward pass declares,
        // because it shades with the same `shading.glsl` and therefore samples
        // the same set.
        let mut accumulate = builder
            .pass("oit_accumulate", PassKind::Inline)
            .access(object_transforms, Access::StorageRead)
            .access(blended_index, Access::StorageRead);
        if let Some(ssao) = ssao {
            accumulate = accumulate.access(ssao.ao, Access::Sampled);
        }
        if let Some(mask) = contact_shadows {
            accumulate = accumulate.access(mask, Access::Sampled);
        }
        if let Some(atlas) = shadow_atlas {
            accumulate = accumulate.access(atlas, Access::Sampled);
        }
        if let Some(fog) = fog {
            accumulate = accumulate.access(fog.volume, Access::Sampled);
        }
        accumulate = reads_gi(accumulate, gi);
        let id = accumulate
            .access(accum, Access::ColorAttachment)
            .access(reveal, Access::ColorAttachment)
            .access(prepass.depth, Access::DepthAttachmentRead)
            .build();
        record(id, PassBody::OitAccumulate, &mut bodies);

        // Reads `source` again rather than accumulating into it, for the reason
        // the reflection and depth-of-field composites do: resources are
        // unversioned, so a pass that both read and wrote the frame's colour
        // would make "readers after all writers" point in two directions and
        // `compile` would report a cycle.
        let id = builder
            .pass("oit_composite", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(accum, Access::Sampled)
            .access(reveal, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::OitComposite, &mut bodies);

        TransparencyIds {
            source,
            accum,
            reveal,
            output,
        }
    });
    if let Some(transparency) = transparency {
        shaded = transparency.output;
    }

    // After the blended queue, so glass refracts what that queue composited, and
    // before the temporal resolve, so it rasterises with the frame's jitter like
    // every other geometry pass.
    //
    // The ordering between the two queues is a choice with no correct answer:
    // neither writes depth, so nothing sorts one against the other. Refraction
    // last means a blended surface *behind* glass is correctly refracted through
    // it, and one in front of glass is not — the commoner case wins, and the
    // uncommon one is what a screen-space technique cannot represent at all.
    let refraction = config.refraction.then(|| {
        let prepass = prepass.expect("refraction depth-tests the geometry prepass");
        let source = shaded;

        // Half the frame, the same base the reflection trace's pyramid of this
        // shape uses. Building it is the most expensive thing this feature does,
        // and every consumer of it is a surface rough enough to be averaging
        // over a cone anyway — the one consumer that is not, a clear pane, reads
        // `source` directly through the draw's second binding below.
        let scene = builder.create_image(
            "refraction_scene",
            ImageDesc::new(HDR_FORMAT)
                .extent(Extent::FrameDiv(1))
                .mip_levels(SCENE_LEVELS),
        );
        let accum =
            builder.create_image("refraction_accum", ImageDesc::new(REFRACTION_ACCUM_FORMAT));
        let output = builder.create_image("refraction_color", ImageDesc::new(HDR_FORMAT));

        let id = builder
            .pass("refraction_scene", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(scene, Access::StorageWrite)
            .build();
        record(id, PassBody::RefractionScene, &mut bodies);

        // The depth is *attached*, not sampled, for the reason the transparency
        // accumulation attaches it: a fixed-function depth test is what makes a
        // refractive surface disappear behind a wall. `DepthAttachmentRead` is
        // the declaration that keeps it read-only.
        //
        // The same screen-space and shadow inputs the forward pass declares,
        // because it shades with the same `shading.glsl` — plus the pyramid
        // above, which is the one input no other geometry pass has.
        let mut draw = builder
            .pass("refraction_draw", PassKind::Inline)
            .access(object_transforms, Access::StorageRead)
            .access(blended_index, Access::StorageRead)
            .access(scene, Access::Sampled)
            .access(source, Access::Sampled);
        if let Some(ssao) = ssao {
            draw = draw.access(ssao.ao, Access::Sampled);
        }
        if let Some(mask) = contact_shadows {
            draw = draw.access(mask, Access::Sampled);
        }
        if let Some(atlas) = shadow_atlas {
            draw = draw.access(atlas, Access::Sampled);
        }
        if let Some(fog) = fog {
            draw = draw.access(fog.volume, Access::Sampled);
        }
        draw = reads_gi(draw, gi);
        let id = draw
            .access(accum, Access::ColorAttachment)
            .access(prepass.depth, Access::DepthAttachmentRead)
            .build();
        record(id, PassBody::RefractionDraw, &mut bodies);

        // Reads `source` again rather than accumulating into it, for the reason
        // every other composite in this frame does: resources are unversioned,
        // so a pass that both read and wrote the frame's colour would make
        // "readers after all writers" point in two directions and `compile`
        // would report a cycle.
        let id = builder
            .pass("refraction_composite", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(accum, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::RefractionComposite, &mut bodies);

        RefractionIds {
            source,
            scene,
            accum,
            output,
        }
    });
    if let Some(refraction) = refraction {
        shaded = refraction.output;
    }

    let taa = config.taa.then(|| {
        let prepass = prepass.expect("TAA reads the geometry prepass");
        // Entry `ShaderReadOnlyOptimal` states the steady state, which the
        // ping-pong guarantees: `output` below leaves every frame in exactly
        // that layout, and it is the allocation `history` names next frame. On
        // the first frame after an allocation the layout is still true — the
        // allocation put it there — and its contents are what `reset` ignores.
        let history = builder.import_image(
            "taa_history",
            ImageDesc::new(HDR_FORMAT),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        // Entered `Undefined` because every texel is written, and left where the
        // next frame wants to find it.
        let output = builder.import_image(
            "taa_color",
            ImageDesc::new(HDR_FORMAT),
            ImageLayout::Undefined,
            ImageLayout::ShaderReadOnlyOptimal,
        );

        let id = builder
            .pass("taa_resolve", PassKind::Compute)
            .access(shaded, Access::Sampled)
            .access(prepass.velocity, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(history, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::TaaResolve, &mut bodies);

        TaaIds {
            source: shaded,
            history,
            output,
        }
    });

    // Everything past shading reads this rather than `hdr_color`, so inserting
    // the resolve is a change to one binding rather than to every consumer.
    // Each optical stage below rebinds it in turn, which is what makes the
    // chain's order a property of this function alone.
    let mut scene_color = taa.map_or(shaded, |taa| taa.output);

    // Lens before shutter before sensor, which is the order light actually
    // meets them and the order Unity and Unreal both settled on. Defocus first,
    // so what the shutter smears is already a lens image; both before bloom and
    // metering, so a defocused highlight blooms as the wide soft thing it has
    // become rather than as the point it was.
    let dof = config.dof.then(|| {
        let prepass = prepass.expect("depth of field reads the geometry prepass");
        let source = scene_color;

        // Half the frame for everything but the composite. The gather is the
        // only pass that pays for the kernel, and paying for it at a quarter of
        // the pixels is what makes a 48-tap disc affordable at all; the
        // composite's bilinear upsample puts back more detail than the extra
        // resolution would have carried, because the field it is upsampling is
        // by definition out of focus.
        // The wide format through the whole chain bar its output, and every
        // stage needs it for a different reason. The prefilter packs a *signed*
        // circle of confusion into alpha; the gather writes premultiplied
        // radiance whose coverage weight the composite divides by; and the tile
        // image carries two channels of signed maxima. None of the three
        // survives a format with no alpha and no negatives.
        let prefiltered = builder.create_image(
            "dof_prefiltered",
            ImageDesc::new(HDR_WIDE_FORMAT).extent(Extent::FrameDiv(1)),
        );
        let near = builder.create_image(
            "dof_near",
            ImageDesc::new(HDR_WIDE_FORMAT).extent(Extent::FrameDiv(1)),
        );
        let far = builder.create_image(
            "dof_far",
            ImageDesc::new(HDR_WIDE_FORMAT).extent(Extent::FrameDiv(1)),
        );
        // Two channels of maxima, for the reason the motion blur tiles are:
        // `R16G16_SFLOAT` is only a guaranteed storage format behind an optional
        // device feature, and at one texel per 4096 the unused half is not worth
        // a feature flag.
        let tile = builder.create_image(
            "dof_tile",
            ImageDesc::new(HDR_WIDE_FORMAT).extent(Extent::FrameDiv(COC_TILE_SHIFT)),
        );
        let output = builder.create_image("dof_color", ImageDesc::new(HDR_FORMAT));

        let id = builder
            .pass("dof_prefilter", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(prefiltered, Access::StorageWrite)
            .build();
        record(id, PassBody::DofPrefilter, &mut bodies);

        // Between the prefilter and the gather for the same reason motion blur's
        // tile pass sits between the velocities and its gather: a fixed tap
        // budget has to be spread over the blur that is actually there, not over
        // the widest one the lens could produce.
        let id = builder
            .pass("dof_tile_max", PassKind::Compute)
            .access(prefiltered, Access::Sampled)
            .access(tile, Access::StorageWrite)
            .build();
        record(id, PassBody::DofTileMax, &mut bodies);

        // One dispatch writing both fields rather than two over the same tiles:
        // they are gathered with different kernels but read the same maxima.
        let id = builder
            .pass("dof_gather", PassKind::Compute)
            .access(prefiltered, Access::Sampled)
            .access(tile, Access::Sampled)
            .access(near, Access::StorageWrite)
            .access(far, Access::StorageWrite)
            .build();
        record(id, PassBody::DofGather, &mut bodies);

        // Reads `source` again rather than accumulating into it, and has to:
        // resources are unversioned, so a pass that both read and wrote the
        // frame's colour would make "readers after all writers" point in two
        // directions at once and `compile` would report a cycle.
        let id = builder
            .pass("dof_composite", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(near, Access::Sampled)
            .access(far, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::DofComposite, &mut bodies);

        DofIds {
            source,
            prefiltered,
            tile,
            near,
            far,
            output,
        }
    });
    if let Some(dof) = dof {
        scene_color = dof.output;
    }

    let motion_blur = config.motion_blur.then(|| {
        let prepass = prepass.expect("motion blur reads the geometry prepass");
        let source = scene_color;

        // `FrameDiv(TILE_SHIFT)` and not a fixed size, because a tile has to be
        // exactly what successive halvings produce: the gather derives its tile
        // from its own pixel coordinate, and a level sized any other way is a
        // texel off from the one it means to read at odd extents.
        //
        // Both carry a two-component *signed* vector, which is the other reason
        // they stay on the wide format: a velocity points in both directions and
        // the packed one is unsigned. A storage
        // image is only guaranteed to support `R16G16_SFLOAT` behind an optional
        // device feature, while `R16G16B16A16_SFLOAT` is always available — and
        // at one texel per 256 the two unused channels are not worth a feature
        // flag on the device.
        let tile = builder.create_image(
            "motion_blur_tile",
            ImageDesc::new(HDR_WIDE_FORMAT).extent(Extent::FrameDiv(TILE_SHIFT)),
        );
        let neighbour = builder.create_image(
            "motion_blur_neighbour",
            ImageDesc::new(HDR_WIDE_FORMAT).extent(Extent::FrameDiv(TILE_SHIFT)),
        );
        let output = builder.create_image("motion_blur_color", ImageDesc::new(HDR_FORMAT));

        // Depth as well as velocity, because where nothing was rasterised the
        // prepass wrote no motion and the sky still sweeps when the camera
        // turns. Both this pass and the gather recover it the same way TAA does,
        // by reprojecting the far plane.
        let id = builder
            .pass("motion_blur_tile_max", PassKind::Compute)
            .access(prepass.velocity, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(tile, Access::StorageWrite)
            .build();
        record(id, PassBody::MotionBlurTileMax, &mut bodies);

        let id = builder
            .pass("motion_blur_neighbour_max", PassKind::Compute)
            .access(tile, Access::Sampled)
            .access(neighbour, Access::StorageWrite)
            .build();
        record(id, PassBody::MotionBlurNeighbourMax, &mut bodies);

        let id = builder
            .pass("motion_blur_gather", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(prepass.velocity, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(neighbour, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::MotionBlurGather, &mut bodies);

        MotionBlurIds {
            source,
            tile,
            neighbour,
            output,
        }
    });
    if let Some(motion_blur) = motion_blur {
        scene_color = motion_blur.output;
    }

    // The reference, where it is asked for, in place of everything above: the
    // frame from here on tonemaps the mean of its paths. It declares the frame
    // it replaces as read, though it reads none of it, so the frame's structure
    // is the same with the reference on as off — every pass still runs, and no
    // history is dropped for having looked at the reference — and so the
    // exposure it is shown at is metered from the frame it is held against.
    let metered_color = scene_color;
    let path = ray.filter(|_| config.ray_path_trace).map(|ray| {
        // Kept between frames, so imported, and entered and left sampled for
        // the reason every history is. See `occlusion_history`.
        let sum = builder.import_image(
            "path_sum",
            ImageDesc::new(Format::Rgba32Float),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let mean = builder.create_image("path_mean", ImageDesc::new(HDR_WIDE_FORMAT));
        let id = builder
            .pass("path_trace", PassKind::Compute)
            .access(ray.scene, Access::AccelerationStructureRead)
            .access(scene_color, Access::Sampled)
            .access(sum, Access::StorageWrite)
            .access(mean, Access::StorageWrite)
            .build();
        record(id, PassBody::PathTrace, &mut bodies);
        PathIds { sum, mean }
    });
    if let Some(path) = path {
        scene_color = path.mean;
    }

    // Imported, not created: temporal adaptation is a value that has to outlive
    // the frame, and a transient is `Undefined` at every frame's start by
    // contract. The exposure module owns the allocation; the graph only orders
    // access to it.
    let exposure = builder.import_buffer("exposure");

    let histogram = config.auto_exposure.then(|| {
        let histogram = builder.import_buffer("luminance_histogram");

        let id = builder
            .pass("luminance_histogram", PassKind::Compute)
            .access(metered_color, Access::Sampled)
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
            .access(scene_color, Access::Sampled)
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
        .access(scene_color, Access::Sampled)
        .access(exposure, Access::StorageRead);
    if let Some(bloom) = bloom {
        tonemap = tonemap.access(bloom.result(), Access::Sampled);
    }
    let id = tonemap
        .access(swapchain_color, Access::ColorAttachment)
        .build();
    record(id, PassBody::Tonemap, &mut bodies);

    // Over every pixel the tonemap wrote, and under everything drawn after it:
    // the traced scene replaces the picture, not the editor's marks on it.
    if let (true, Some(ray)) = (config.ray_debug, ray) {
        let id = builder
            .pass("ray_debug", PassKind::Inline)
            .access(ray.scene, Access::AccelerationStructureRead)
            .access(object_transforms, Access::StorageRead)
            .access(swapchain_color, Access::ColorAttachment)
            .build();
        record(id, PassBody::RayDebug, &mut bodies);
    }

    // First onto the tonemapped image, so the game's UI and every other mark the
    // editor makes stay on top of it. Over the shaded frame it reads the prepass
    // depth to hide covered edges — read-only, so the prepass stays its only
    // writer; alone it clears the frame and draws every edge. See
    // `vulkan::wireframe`.
    let wireframe = config
        .overlay
        .then_some(config.wireframe)
        .flatten()
        .map(|mode| {
            let depth = match mode {
                super::wireframe::Wireframe::OverShaded => prepass.as_ref().map(|p| p.depth),
                super::wireframe::Wireframe::Alone => None,
            };
            let mut pass = builder
                .pass("wireframe", PassKind::Inline)
                .access(swapchain_color, Access::ColorAttachment);
            if let Some(depth) = depth {
                pass = pass.access(depth, Access::DepthAttachmentRead);
            }
            record(pass.build(), PassBody::Wireframe, &mut bodies);
            (mode, depth)
        });

    // Straight onto the tonemapped image, so the game's colours are the ones it
    // asked for rather than the scene's exposure of them; first among the
    // swapchain's writers after the wireframe, so every mark the editor makes
    // stays on top.
    if config.ui {
        let id = builder
            .pass("ui", PassKind::Inline)
            .access(swapchain_color, Access::ColorAttachment)
            .build();
        record(id, PassBody::Ui, &mut bodies);
    }

    let outline_mask = (config.overlay && config.outline).then(|| {
        // Its own target rather than the depth or stencil of the scene: the
        // outline is the editor's, drawn after the frame is finished, and the
        // frame's targets are resolved and gone by then. See `vulkan::outline`.
        let mask = builder.create_image("outline_mask", ImageDesc::new(OUTLINE_MASK_FORMAT));
        let id = builder
            .pass("outline_mask", PassKind::Inline)
            .access(mask, Access::ColorAttachment)
            .build();
        record(id, PassBody::OutlineMask, &mut bodies);
        let id = builder
            .pass("outline", PassKind::Inline)
            .access(mask, Access::Sampled)
            .access(swapchain_color, Access::ColorAttachment)
            .build();
        record(id, PassBody::Outline, &mut bodies);
        mask
    });

    // Its own node, after the tonemap: see `vulkan::gizmo` for why not lines in
    // the forward pass. Ordered before the overlay by registration, the two
    // being writers of the same image.
    if config.overlay && config.gizmo {
        let id = builder
            .pass("gizmo", PassKind::Inline)
            .access(swapchain_color, Access::ColorAttachment)
            .build();
        record(id, PassBody::Gizmo, &mut bodies);
    }

    // Apart from everything above: its own targets at its own size, sharing no
    // resource with the scene, so the scene's plan is the same with it or
    // without it. The output is imported because the overlay samples it after
    // the frame, and on later frames that leave it alone — so it leaves
    // sampled, and is entered `Undefined` because the resolve covers it.
    let material_preview = config.material_preview.then(|| {
        use super::material_preview::{OUTPUT_FORMAT, SIZE, SUPERSAMPLE};
        let shaded = Extent::Fixed([SIZE * SUPERSAMPLE; 2]);
        let hdr = builder.create_image(
            "material_preview_hdr",
            ImageDesc::new(HDR_WIDE_FORMAT).extent(shaded),
        );
        let depth = builder.create_image(
            "material_preview_depth",
            ImageDesc::new(DEPTH_FORMAT).extent(shaded),
        );
        let id = builder
            .pass("material_preview", PassKind::Inline)
            .access(hdr, Access::ColorAttachment)
            .access(depth, Access::DepthAttachment)
            .build();
        record(id, PassBody::MaterialPreview, &mut bodies);

        let output = builder.import_image(
            "material_preview",
            ImageDesc::new(OUTPUT_FORMAT).extent(Extent::Fixed([SIZE; 2])),
            ImageLayout::Undefined,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let id = builder
            .pass("material_preview_resolve", PassKind::Inline)
            .access(hdr, Access::Sampled)
            .access(output, Access::ColorAttachment)
            .build();
        record(id, PassBody::MaterialPreviewResolve, &mut bodies);
        MaterialPreviewIds { hdr, depth, output }
    });

    // The picture a screenshot asks for, drawn a second time into an image of
    // the window's format so each twin's own pipeline draws it: the tonemap,
    // then what the view puts over it, in the order the window's writers draw
    // and reading only what they read, so the frame's other images keep their
    // usage. It ends sampled because the readback leaves it that way.
    let screenshot = config.screenshot.map(|view| {
        let image = builder.import_image(
            "screenshot",
            ImageDesc::new(config.color_format),
            ImageLayout::Undefined,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        let mut pass = builder
            .pass("screenshot", PassKind::Inline)
            .access(scene_color, Access::Sampled)
            .access(exposure, Access::StorageRead);
        if let Some(bloom) = bloom {
            pass = pass.access(bloom.result(), Access::Sampled);
        }
        let id = pass.access(image, Access::ColorAttachment).build();
        record(id, PassBody::Screenshot, &mut bodies);
        let editor = view == ShotView::Editor;
        let mut over = |name: &'static str, body: PassBody, extra: Option<(ResourceId, Access)>| {
            let mut pass = builder
                .pass(name, PassKind::Inline)
                .access(image, Access::ColorAttachment);
            if let Some((resource, access)) = extra {
                pass = pass.access(resource, access);
            }
            record(pass.build(), body, &mut bodies);
        };
        if editor && let Some((_, depth)) = wireframe {
            over(
                "screenshot_wireframe",
                PassBody::ScreenshotWireframe,
                depth.map(|depth| (depth, Access::DepthAttachmentRead)),
            );
        }
        if view != ShotView::Scene && config.ui {
            over("screenshot_ui", PassBody::ScreenshotUi, None);
        }
        if editor && let Some(mask) = outline_mask {
            over(
                "screenshot_outline",
                PassBody::ScreenshotOutline,
                Some((mask, Access::Sampled)),
            );
        }
        if editor && config.overlay && config.gizmo {
            over("screenshot_gizmo", PassBody::ScreenshotGizmo, None);
        }
        // The window's overlay samples the material preview after the frame;
        // this copy of it samples it inside, after the preview's resolve.
        if editor && config.overlay && config.presents {
            over(
                "screenshot_overlay",
                PassBody::ScreenshotOverlay,
                material_preview
                    .as_ref()
                    .map(|preview| (preview.output, Access::Sampled)),
            );
        }
        image
    });

    // Only into a window: offscreen there is no egui to draw and the executor
    // skips the node, but its barriers would still name `PresentSrc`.
    if config.overlay && config.presents {
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
            instance_index,
            swapchain_color,
            hdr_color,
            scene_color,
            metered_color,
            path,
            msaa,
            exposure,
            histogram,
            bloom,
            prepass,
            ssao,
            occlusion: occlusion_ids,
            contact_shadows,
            fog,
            ssr,
            subsurface,
            transparency,
            refraction,
            taa,
            dof,
            motion_blur,
            shadow_atlas,
            shadow_static: shadow_atlases.map(|(_, cached)| cached),
            outline_mask,
            wireframe,
            material_preview,
            screenshot,
            ray,
            gi,
            gather,
        },
        bodies,
    })
}
