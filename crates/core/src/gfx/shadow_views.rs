//! Shadow views: every map a frame's shadows are read from, as one list in one
//! atlas.
//!
//! A cascade, a sun level and a face of a point or spot light are the same
//! thing to everything downstream of the fit: a matrix to draw and sample with,
//! a tile to draw into, a volume that says what can cast into it, and the size
//! of one of its texels. So they are one type, [`ShadowView`], and what culls,
//! draws and filters a shadow is written once against it.
//!
//! One atlas rather than an image per light or a layer per cascade, and that is
//! what makes this affordable in a graph whose resources are unversioned: a
//! tile is a viewport into the same attachment, so every view of every light is
//! a single pass with a single barrier. Passes over the layers of a shared
//! array are serialised by write-after-write, which is what four cascades used
//! to pay. A tile also has its own size, where a layer has the array's.
//!
//! # Two layers, and what a frame does to them
//!
//! Each view has a cached layer — the casters standing still
//! ([`shadow_casters`](super::shadow_casters)), kept from frame to frame in an
//! atlas of its own — and the live layer the frame is shaded from, which is the
//! cached one with the moving casters drawn over it. So shading samples one
//! map, as it always did, and what the cache changes is only how that map is
//! brought up to date.
//!
//! What it is brought up to date *by* is the texels that changed, never the
//! tile. Putting a live tile back whole from its cached layer touches every
//! texel of it, and on a tiling GPU that costs more than the draws it saves:
//! about 0.02 ms a megapixel on an M5 Pro, against 0.04 ms to clear a 2048
//! tile by its load op and draw a hundred casters into it from nothing. So a
//! view remembers the rectangles its moving casters wrote, and the next frame
//! puts back those rectangles and no others.
//!
//! The same number is why a sun level moves in steps rather than following the
//! camera texel by texel ([`LevelWindow`]), and why the sun's views are levels
//! unless a scene asks otherwise: a view fitted to the frustum is a new matrix
//! every time the camera moves, and a new matrix is a tile cleared and redrawn
//! whole.
//!
//! A view that cannot be kept — a light that is moving, a cached layer waiting
//! its turn — is drawn the way it always was, every caster straight into the
//! live layer. So the cache never decides whether a shadow is right, only what
//! it costs; `tests/shadow_cache.rs` holds it to that on a GPU.
//!
//! # What a light may say about its own views
//!
//! A light carries [`LightShadows`], and each field
//! of it lands on the views here rather than in a pass. The filter width, the
//! sharpness and the two biases travel with a view, because two lamps in one
//! atlas are two different answers; `INHERIT` in any of them is the scene's
//! own ([`ViewDefaults`]).
//!
//! A lamp's tile is sized by how much of the screen its reach covers, in
//! powers of two, and a tier is held until the size wanted has left it by a
//! margin on either side. Without the margin a lamp at a tier's boundary
//! changes tile every few frames as the camera breathes, and a new tile is a
//! cached layer thrown away.
//!
//! The update mode is which casters a view takes and when: `Always` keeps what
//! stands still and draws what moves over it every frame, `OnChange` draws
//! only what stands still, so a moving caster casts nothing, and `OnDemand`
//! draws everything in reach once and keeps that picture until the light's
//! revision is bumped. None of them can leave a light's own movement undrawn:
//! a new matrix is a new view whatever the mode says.
//!
//! A light's channel mask is part of what a view *is*. It decides which
//! casters the cull takes for the view, so a mask that changes is a cached
//! layer holding depth from casters that are no longer its own, and the view
//! is redrawn.
//!
//! Nothing here takes a `Device`, for the reason the render graph does not: the
//! allocation, the matrices and the plan are the part that can be wrong in a
//! way pixels only hint at, so they are the part CI can assert.

use glam::{IVec2, Mat4, Vec3, Vec4};

use super::SceneLighting;
use super::punctual::{self, LightKind, MAX_SHADOW_LIGHTS};
use super::shadow_casters::{CasterTracker, FLAG_DYNAMIC, FLAG_STATIC};
use super::shadows::{
    CascadeConfig, LevelWindow, MAX_CASCADES, MAX_SWEEP_PLANES, SweepBounds, sun_views, swept_slice,
};
use crate::geom::{Aabb, Frustum};
use crate::scene::{Camera, LightShadows, ShadowUpdate};

/// Where one view lives in the atlas, in texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AtlasTile {
    pub offset: [u32; 2],
    pub size: u32,
}

impl AtlasTile {
    /// The tile as a UV rectangle of the whole atlas: `xy` offset, `zw` scale.
    ///
    /// What the shader multiplies a view's NDC into, and the reason a view needs
    /// no knowledge of the atlas beyond these four numbers.
    pub fn uv_rect(&self, resolution: u32) -> [f32; 4] {
        let resolution = resolution.max(1) as f32;
        [
            self.offset[0] as f32 / resolution,
            self.offset[1] as f32 / resolution,
            self.size as f32 / resolution,
            self.size as f32 / resolution,
        ]
    }
}

/// Hands out square power-of-two tiles of a square power-of-two atlas.
///
/// A buddy allocator over the atlas's quadtree: a block is either whole, or
/// split into its four quarters. That is what lets views of different sizes
/// share the image without a packing problem — a tile of any size lands on a
/// boundary of its own size, so no two can overlap by part.
#[derive(Clone, Debug)]
pub struct TileAllocator {
    resolution: u32,
    /// Free blocks by level: level 0 is the whole atlas and each level below
    /// halves the edge.
    free: Vec<Vec<[u32; 2]>>,
}

/// The smallest tile the allocator hands out.
const MIN_TILE: u32 = 16;

impl TileAllocator {
    /// An empty atlas `resolution` texels across, rounded down to a power of
    /// two.
    pub fn new(resolution: u32) -> Self {
        let resolution = floor_power_of_two(resolution.max(MIN_TILE));
        let levels = (resolution / MIN_TILE).trailing_zeros() as usize + 1;
        let mut free = vec![Vec::new(); levels];
        free[0].push([0, 0]);
        Self { resolution, free }
    }

    /// Empty the atlas again, at `resolution`, keeping what was allocated to
    /// track it.
    pub fn reset(&mut self, resolution: u32) {
        let resolution = floor_power_of_two(resolution.max(MIN_TILE));
        let levels = (resolution / MIN_TILE).trailing_zeros() as usize + 1;
        self.resolution = resolution;
        self.free.iter_mut().for_each(Vec::clear);
        self.free.resize_with(levels, Vec::new);
        self.free[0].push([0, 0]);
    }

    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    fn level(&self, size: u32) -> Option<usize> {
        let size = floor_power_of_two(size.max(MIN_TILE));
        (size <= self.resolution).then(|| (self.resolution / size).trailing_zeros() as usize)
    }

    /// A tile `size` texels across, rounded down to a power of two, or `None`
    /// when no block that large is free.
    pub fn allocate(&mut self, size: u32) -> Option<AtlasTile> {
        let level = self.level(size)?;
        let from = (0..=level).rev().find(|&l| !self.free[l].is_empty())?;
        // The lowest block rather than the last freed, so what a frame is
        // handed depends on what is allocated and not on the order it got so.
        let at = (0..self.free[from].len())
            .min_by_key(|&i| (self.free[from][i][1], self.free[from][i][0]))
            .expect("the level was found non-empty");
        let offset = self.free[from].swap_remove(at);
        for split in from + 1..=level {
            let half = self.resolution >> split;
            self.free[split].extend([
                [offset[0] + half, offset[1]],
                [offset[0], offset[1] + half],
                [offset[0] + half, offset[1] + half],
            ]);
        }
        Some(AtlasTile {
            offset,
            size: self.resolution >> level,
        })
    }

    /// Give a tile back, joining it with its three siblings when all four are
    /// free so the block they were split from can be handed out whole again.
    pub fn release(&mut self, tile: AtlasTile) {
        let Some(mut level) = self.level(tile.size) else {
            return;
        };
        let mut offset = tile.offset;
        while level > 0 {
            let size = self.resolution >> level;
            let parent = [offset[0] & !(2 * size - 1), offset[1] & !(2 * size - 1)];
            let siblings = [
                parent,
                [parent[0] + size, parent[1]],
                [parent[0], parent[1] + size],
                [parent[0] + size, parent[1] + size],
            ];
            let free = &mut self.free[level];
            if !siblings.iter().all(|s| *s == offset || free.contains(s)) {
                break;
            }
            free.retain(|block| !siblings.contains(block));
            offset = parent;
            level -= 1;
        }
        self.free[level].push(offset);
    }
}

/// The largest power of two that is at most `value`, and at least one.
pub fn floor_power_of_two(value: u32) -> u32 {
    1 << (31 - value.max(1).leading_zeros())
}

/// A rectangle of the atlas, in texels: `min` inclusive, `max` exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasRect {
    pub min: [u32; 2],
    pub max: [u32; 2],
}

impl AtlasRect {
    fn of(tile: AtlasTile) -> Self {
        Self {
            min: tile.offset,
            max: [tile.offset[0] + tile.size, tile.offset[1] + tile.size],
        }
    }

    fn texels(&self) -> u64 {
        u64::from(self.max[0] - self.min[0]) * u64::from(self.max[1] - self.min[1])
    }

    fn union(&self, other: &AtlasRect) -> AtlasRect {
        AtlasRect {
            min: [self.min[0].min(other.min[0]), self.min[1].min(other.min[1])],
            max: [self.max[0].max(other.max[0]), self.max[1].max(other.max[1])],
        }
    }
}

/// What can cast into a view, which is what the cull dispatch tests.
///
/// Two shapes rather than six planes for everything, because the sun's views
/// are not a frustum test. See `casts_into` in `cull.comp`.
#[derive(Clone, Copy, Debug)]
pub enum CasterVolume {
    /// A face of a punctual light: a closed frustum with its apex at the light,
    /// so anything that can occlude light travelling into it is inside it.
    Frustum(Mat4),
    /// A sun view, or a strip of one: anything that reaches `bounds`, a box in
    /// `light_view`'s space, when swept toward the light — so it has no near
    /// plane, and an object between the sun and the box is exactly what casts
    /// into it.
    Sweep {
        light_view: Mat4,
        bounds: SweepBounds,
    },
    /// The part of a sun view the camera can see, swept toward the light, as
    /// the planes whose positive side is inside. For what a frame draws and
    /// throws away; see [`swept_slice`].
    Hull([Vec4; MAX_SWEEP_PLANES], usize),
}

impl CasterVolume {
    /// Whether a caster with these world bounds may reach the view. Conservative
    /// the way the dispatch's own tests are: true means "may".
    fn reached_by(&self, bounds: &Aabb) -> bool {
        match self {
            CasterVolume::Frustum(view_proj) => {
                Frustum::from_view_projection(*view_proj).intersects(bounds)
            }
            CasterVolume::Sweep {
                light_view,
                bounds: sweep,
            } => {
                if !bounds.is_valid() {
                    return true;
                }
                let lit = bounds.transformed(light_view);
                lit.min.x <= sweep.max.x
                    && lit.max.x >= sweep.min.x
                    && lit.min.y <= sweep.max.y
                    && lit.max.y >= sweep.min.y
                    && lit.max.z >= sweep.min.z
            }
            CasterVolume::Hull(planes, count) => {
                if !bounds.is_valid() {
                    return true;
                }
                let (center, extents) = (bounds.center(), bounds.half_extents());
                planes[..*count].iter().all(|plane| {
                    let normal = glam::Vec3A::from(plane.truncate());
                    normal.dot(center) + plane.w + extents.dot(normal.abs()) >= 0.0
                })
            }
        }
    }
}

/// Whose view it is, which decides the depth bias it is drawn with and the
/// penumbra it is filtered to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKind {
    Sun,
    Punctual,
}

/// One map: a cascade, a sun level, or a face of a point or spot light.
#[derive(Clone, Copy, Debug)]
pub struct ShadowView {
    /// What the view is sampled with, and drawn with wherever it is drawn
    /// whole.
    pub view_proj: Mat4,
    pub tile: AtlasTile,
    pub volume: CasterVolume,
    /// World size of one texel where the view's clip-space `w` is one: the
    /// whole of an orthographic sun view, and a metre along the axis of a
    /// perspective face. The filter multiplies it by a lookup's own `w`, which
    /// is what makes one formula serve both.
    pub texel_scale: f32,
    pub kind: ViewKind,
    /// The light channels it takes casters from: its light's.
    pub channels: u32,
    /// The width of its penumbra in metres, and how much of it is visible.
    pub filtering: [f32; 2],
    /// The constant and slope depth bias it is drawn with.
    pub bias: [f32; 2],
}

/// A punctual light the atlas found room for.
#[derive(Clone, Copy, Debug)]
pub struct ShadowCaster {
    pub kind: LightKind,
    /// Index into [`SceneLighting::point_lights`] or `spot_lights`, depending on
    /// `kind`.
    pub light: usize,
    /// This caster's views, contiguous in [`ShadowViews::views`]: six for a
    /// point, one for a spot.
    pub first_view: usize,
    pub view_count: usize,
    /// What the faces' near plane was.
    pub near: f32,
}

/// How the atlas is cut up for the punctual lights and what their frustums are
/// clipped to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasConfig {
    /// Edge of the whole atlas in texels. Frame *structure*: changing it
    /// recompiles the graph and reallocates the image.
    pub resolution: u32,
    /// Whether point and spot lights get tiles at all.
    pub punctual: bool,
    /// Edge of one face's tile.
    pub tile_size: u32,
    /// Near plane of every punctual frustum, in metres.
    pub near: f32,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            resolution: 4096,
            punctual: true,
            tile_size: 512,
            near: 0.05,
        }
    }
}

/// What one frame's shadows are read from.
///
/// No views means nothing casts, which is what makes the graph drop the shadow
/// passes and their atlas entirely rather than keep images for no reader.
#[derive(Clone, Debug, Default)]
pub struct ShadowViews {
    /// Edge of the atlas the tiles are cut from.
    pub resolution: u32,
    /// The sun's views first, nearest first, then each punctual caster's.
    pub views: Vec<ShadowView>,
    /// How many of `views` are the sun's.
    pub sun: usize,
    /// The radial distance each sun view answers out to. Slots past `sun` hold
    /// the last one, so a scan for the first split a fragment is nearer than
    /// can never land on a view that is not there.
    pub splits: [f32; MAX_CASCADES],
    pub casters: Vec<ShadowCaster>,
}

impl ShadowViews {
    /// The caster for a light of `kind` at `index`, if it got tiles.
    ///
    /// A linear scan because the list is at most [`MAX_SHADOW_LIGHTS`] long, and
    /// a parallel lookup table indexed by light would be a second thing to keep
    /// in step with the first.
    pub fn caster(&self, kind: LightKind, index: usize) -> Option<&ShadowCaster> {
        self.casters
            .iter()
            .find(|caster| caster.kind == kind && caster.light == index)
    }
}

/// Which of a view's two layers a draw lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// The cached layer: the casters that are standing still, kept from frame
    /// to frame.
    Static,
    /// What the frame is shaded from.
    Live,
}

/// Which casters a draw takes, as the mask the cull dispatch tests an
/// instance's [flags](super::shadow_casters) against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Casters {
    All,
    Static,
    Dynamic,
}

impl Casters {
    pub fn mask(self) -> u32 {
        match self {
            Casters::All => FLAG_STATIC | FLAG_DYNAMIC,
            Casters::Static => FLAG_STATIC,
            Casters::Dynamic => FLAG_DYNAMIC,
        }
    }
}

/// One draw of casters into a tile. Each is a view of the cull dispatch's own.
#[derive(Clone, Copy, Debug)]
pub struct CasterDraw {
    /// The shadow view this draws for.
    pub view: usize,
    pub layer: Layer,
    pub casters: Casters,
    pub volume: CasterVolume,
    /// What it is drawn with: the view's own matrix, or for a strip of a level
    /// the matrix that lands the strip where its cells are kept.
    pub view_proj: Mat4,
    /// The tile, which the viewport covers whole.
    pub tile: AtlasTile,
    /// The part of the tile the draw may write.
    pub scissor: AtlasRect,
    /// The size below which a caster is not worth drawing here: a texel of the
    /// view, as a length in metres and a further length per metre of depth
    /// into a perspective one. A caster smaller than a texel shadows nothing
    /// the filter can resolve.
    pub cutoff: [f32; 2],
}

/// A rectangle of the live layer put back to what the cached layer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Restore {
    pub rect: AtlasRect,
    pub tile: AtlasTile,
    /// Where in the cached tile the live tile's first texel is kept. A level's
    /// cached texels sit at their cells' places modulo the tile, so its window
    /// starts wherever the camera has carried it; every other view's is zero.
    pub wrap: [u32; 2],
}

/// What happened to a view this frame, for the debug view and the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewWork {
    /// Nothing: no caster moved in it, and the tile holds what it held.
    Held,
    /// The rectangles its moving casters touched were put back and the moving
    /// casters drawn again.
    Composited,
    /// A level followed the camera: the strip it moved onto was drawn into the
    /// cached layer.
    Scrolled,
    /// The cached layer was redrawn whole.
    Refreshed,
    /// Drawn straight into the live layer with every caster, because its
    /// matrix moved or its cached layer is waiting its turn to be redrawn.
    Direct,
}

/// One frame's shadows: the views to shade from, and the work that brings the
/// atlas up to date with them.
#[derive(Clone, Debug, Default)]
pub struct ShadowPlan {
    pub views: ShadowViews,
    /// One per view, in `views`' order.
    pub work: Vec<ViewWork>,
    /// Caster draws, the cached layer's first.
    pub draws: Vec<CasterDraw>,
    /// Rectangles of the cached layer cleared to the far plane before it is
    /// drawn into.
    pub static_fills: Vec<AtlasRect>,
    /// And of the live layer.
    pub live_fills: Vec<AtlasRect>,
    pub restores: Vec<Restore>,
}

impl ShadowPlan {
    fn clear(&mut self) {
        self.views.views.clear();
        self.views.casters.clear();
        self.views.sun = 0;
        self.views.resolution = 0;
        self.views.splits = [0.0; MAX_CASCADES];
        self.work.clear();
        self.draws.clear();
        self.static_fills.clear();
        self.live_fills.clear();
        self.restores.clear();
    }

    /// Whether the cached layer has anything drawn into it this frame.
    pub fn draws_static(&self) -> bool {
        !self.static_fills.is_empty()
    }

    /// Whether the live layer has.
    pub fn draws_live(&self) -> bool {
        !self.live_fills.is_empty()
            || !self.restores.is_empty()
            || self.draws.iter().any(|draw| draw.layer == Layer::Live)
    }
}

/// What a frame's shadows are planned from.
pub struct ShadowInputs<'a> {
    pub camera: &'a Camera,
    pub aspect: f32,
    pub lighting: &'a SceneLighting,
    /// The sun's cascades, or `None` when it casts no shadow.
    pub cascades: Option<&'a CascadeConfig>,
    pub atlas: &'a AtlasConfig,
    /// Which casters are standing still, and what changed about that.
    pub casters: &'a CasterTracker,
    /// Height of the frame in pixels, which is what a light's reach is
    /// measured against to size its tiles.
    pub viewport_height: f32,
    /// The sun's penumbra width and sharpness, and its constant and slope bias.
    pub sun: ViewDefaults,
    /// The same for a lamp that does not say otherwise.
    pub lamps: ViewDefaults,
}

/// How a view is filtered and biased where nothing more specific says.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewDefaults {
    pub filtering: [f32; 2],
    pub bias: [f32; 2],
}

/// Which view a tile belongs to, from one frame to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewKey {
    Sun(u8),
    Face { light: u64, face: u8 },
}

/// What a view's live tile holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Live {
    /// Nothing that can be used.
    Unknown,
    /// Every caster, drawn straight in.
    Everything,
    /// The cached layer, with the moving casters over it.
    Composite,
}

/// What the atlas holds for one view.
#[derive(Debug)]
struct Entry {
    key: ViewKey,
    tile: AtlasTile,
    /// The frame it was last planned.
    seen: u32,
    /// The matrix its tile was last drawn or kept for, and the channels and
    /// the revision it was drawn with.
    view_proj: Mat4,
    channels: u32,
    revision: u32,
    /// Whether the cached layer holds the static casters for `view_proj` — or,
    /// for a level, for the cells of `window`.
    cached: bool,
    live: Live,
    /// A level's window, and the cell its cached tile's first texel holds: the
    /// cached layer keeps cell `c` at `(c - base) mod cells`, so a window that
    /// has moved finds the cells it still shares where it left them.
    window: Option<LevelWindow>,
    base: IVec2,
    /// What the moving casters wrote in the live tile last frame.
    touched: Vec<AtlasRect>,
}

impl Entry {
    fn wrap(&self) -> [u32; 2] {
        match &self.window {
            Some(window) => {
                let n = window.cells as i32;
                let wrap = (window.min_cell - self.base).rem_euclid(IVec2::splat(n));
                [wrap.x as u32, wrap.y as u32]
            }
            None => [0, 0],
        }
    }
}

/// How many texels of cached layer a frame redraws whole. A view past the
/// budget waits a frame, drawn directly meanwhile — so what the budget spreads
/// is the cost of *starting* to cache, never a shadow's correctness. A little
/// over one 2048 tile, so the largest view always fits.
const REFRESH_BUDGET: u64 = 5_000_000;

/// How many rectangles a view restores one by one. Past it they are restored as
/// the one rectangle around them all.
const MAX_TOUCHED: usize = 48;

/// How much a light that already has tiles outranks one that does not. A light
/// trading its tiles with a neighbour as the camera drifts between them costs
/// both their cached layers every time.
const INCUMBENT_BONUS: f32 = 1.5;

/// What a light that reaches nothing in view is worth beside one that does.
const OUT_OF_VIEW: f32 = 1e-6;

/// How many texels across a caster has to be for a view to draw it.
const SMALL_CASTER_TEXELS: f32 = 1.0;

/// Plans each frame's shadow views and the work that brings the atlas up to
/// date with them, and remembers what the atlas holds from the frames before.
///
/// Each view has two layers: a cached one holding the casters that are
/// standing still, and the live one the frame is shaded from, which is the
/// cached layer with the moving casters drawn over it. Per frame a view is in
/// one of the [`ViewWork`] states, and which one is decided here from what
/// changed — so that a still scene draws nothing, a scene with a few moving
/// things restores and redraws only what they touched, and anything this
/// cannot cache is drawn the way it always was.
pub struct ShadowPlanner {
    tiles: TileAllocator,
    entries: Vec<Entry>,
    frame: u32,
    /// The sun as the levels were last placed for it. A level's grid is laid
    /// along the light, so a sun that has turned at all has no windows to keep.
    sun: Option<(Vec3, [u32; MAX_CASCADES])>,
    ranked: Vec<(LightKind, usize, f32)>,
    /// Entries taken out while their view is planned and put back after, so a
    /// frame allocates no list for them.
    spare: Vec<Vec<AtlasRect>>,
}

impl Default for ShadowPlanner {
    fn default() -> Self {
        Self {
            tiles: TileAllocator::new(MIN_TILE),
            entries: Vec::new(),
            frame: 0,
            sun: None,
            ranked: Vec::new(),
            spare: Vec::new(),
        }
    }
}

/// The smallest tile a lamp's face is given.
const MIN_LAMP_TILE: u32 = 64;

/// How far past the tile it holds a lamp's want has to go before it is given
/// the next size up, and how far under half before it is given the next down.
/// Between the two it keeps its tile: a light whose reach sits at a boundary
/// would otherwise change size, and be redrawn whole, with every step the
/// camera took.
const TILE_GROW: f32 = 1.15;
const TILE_SHRINK: f32 = 0.5 * 0.85;

/// What the planner reads of a point or spot light.
struct Lamp {
    id: u64,
    position: Vec3,
    /// How far its shadows reach.
    reach: f32,
    /// The tangent of half the angle a face spans, at most one.
    spread: f32,
    channels: u32,
    shadows: LightShadows,
}

impl Lamp {
    fn of(lighting: &SceneLighting, kind: LightKind, index: usize) -> Self {
        let reach = |range: f32, shadows: &LightShadows| {
            if shadows.range > 0.0 {
                shadows.range.min(range)
            } else {
                range
            }
        };
        match kind {
            LightKind::Point => {
                let light = &lighting.point_lights[index];
                Self {
                    id: light.id,
                    position: light.position,
                    reach: reach(light.range, &light.shadows),
                    spread: 1.0,
                    channels: light.channels,
                    shadows: light.shadows,
                }
            }
            LightKind::Spot => {
                let light = &lighting.spot_lights[index];
                let cos = light.outer_cos.clamp(1e-3, 1.0);
                Self {
                    id: light.id,
                    position: light.position,
                    reach: reach(light.range, &light.shadows),
                    spread: ((1.0 - cos * cos).sqrt() / cos).min(1.0),
                    channels: light.channels,
                    shadows: light.shadows,
                }
            }
        }
    }

    /// The edge of the tile each of its faces gets: a texel for each pixel the
    /// far end of its reach covers on screen, in powers of two, up to the
    /// scene's tile or the light's own cap — and `held`, the tile it has, for
    /// as long as that is near enough.
    ///
    /// A face spans `2 * reach * spread` metres at the far end of its reach,
    /// which from the camera's distance is that many pixels across: more
    /// texels than that are texels nothing on screen can tell apart.
    fn tile(&self, inputs: &ShadowInputs<'_>, held: Option<u32>) -> u32 {
        let largest = if self.shadows.resolution > 0 {
            self.shadows.resolution
        } else {
            inputs.atlas.tile_size
        };
        let largest = floor_power_of_two(largest.max(MIN_LAMP_TILE));
        let distance = self.position.distance(inputs.camera.position);
        let focal = inputs.viewport_height * 0.5 / (inputs.camera.fov_y * 0.5).tan();
        // Inside its reach the far end is all around the camera, and no tile
        // is too large.
        let wanted = if distance <= self.reach {
            largest as f32
        } else {
            (2.0 * self.reach * self.spread * focal / distance).min(largest as f32)
        };
        let size = match held {
            Some(held)
                if wanted <= held as f32 * TILE_GROW && wanted > held as f32 * TILE_SHRINK =>
            {
                held
            }
            _ => (wanted.max(1.0) as u32).next_power_of_two(),
        };
        size.clamp(MIN_LAMP_TILE, largest)
    }
}

/// What one view asks of the planner this frame.
struct Wanted {
    key: ViewKey,
    size: u32,
    kind: ViewKind,
    view_proj: Mat4,
    volume: CasterVolume,
    texel_scale: f32,
    window: Option<LevelWindow>,
    /// The light's rotation, for a level's strips.
    rotation: Mat4,
    /// What can cast into the part of the view a frame can see, for the layers
    /// it draws and throws away. The view's own volume where that is no
    /// tighter.
    visible: CasterVolume,
    /// Corners of the only part of the view the frame can read, where that is
    /// known to be less than all of it.
    read: Option<[Vec3; 8]>,
    /// Whether the light reaches anything in view. One that does not is not
    /// worked on: nothing reads its tiles until it does.
    in_view: bool,
    channels: u32,
    filtering: [f32; 2],
    bias: [f32; 2],
    update: ShadowUpdate,
    revision: u32,
}

impl ShadowPlanner {
    /// Forget what the atlas holds, because the images behind it were replaced.
    pub fn invalidate(&mut self) {
        for entry in self.entries.drain(..) {
            self.spare.push(entry.touched);
        }
        self.tiles.reset(self.tiles.resolution());
        self.sun = None;
    }

    /// Plan a frame into `plan`, which the caller keeps between frames so that
    /// its lists are allocated once. `None` is a frame in which nothing casts.
    pub fn plan(&mut self, inputs: Option<&ShadowInputs<'_>>, plan: &mut ShadowPlan) {
        plan.clear();
        self.frame += 1;
        let Some(inputs) = inputs else {
            self.invalidate();
            return;
        };

        if self.tiles.resolution() != floor_power_of_two(inputs.atlas.resolution.max(MIN_TILE)) {
            self.invalidate();
            self.tiles.reset(inputs.atlas.resolution);
        }
        plan.views.resolution = self.tiles.resolution();

        self.plan_sun(inputs, plan);
        self.plan_lamps(inputs, plan);

        // Whatever was not asked for again gives its tile back.
        let frame = self.frame;
        let (tiles, spare) = (&mut self.tiles, &mut self.spare);
        self.entries.retain_mut(|entry| {
            if entry.seen == frame {
                return true;
            }
            tiles.release(entry.tile);
            spare.push(std::mem::take(&mut entry.touched));
            false
        });

        if plan.views.views.is_empty() {
            plan.views.resolution = 0;
        }
        // The cached layer's draws first, so the pass that records them takes a
        // prefix and the dispatch's numbering is the list's order either way.
        plan.draws
            .sort_unstable_by_key(|draw| (draw.layer == Layer::Live, draw.view));
    }

    fn plan_sun(&mut self, inputs: &ShadowInputs<'_>, plan: &mut ShadowPlan) {
        let Some(config) = inputs.cascades else {
            self.sun = None;
            return;
        };
        let direction = inputs
            .lighting
            .sun
            .direction
            .try_normalize()
            .unwrap_or(Vec3::NEG_Y);
        let count = config.count.clamp(1, MAX_CASCADES);
        let mut sizes = [0; MAX_CASCADES];
        for (index, size) in sizes.iter_mut().enumerate().take(count) {
            *size = config.resolution_of(index);
        }

        // The sun's tiles are placed before any lamp's, in an empty atlas, so
        // that a lamp can never be what a cascade does not fit beside. A change
        // to which tiles the sun wants therefore starts the atlas over.
        if self.sun.is_none_or(|(_, held)| held != sizes) {
            self.invalidate();
        }
        let turned = self.sun.is_none_or(|(held, _)| held != direction);
        self.sun = Some((direction, sizes));

        let mut previous = [None; MAX_CASCADES];
        if !turned {
            for entry in &self.entries {
                if let ViewKey::Sun(index) = entry.key {
                    previous[index as usize] = entry.window;
                }
            }
        }
        let set = sun_views(inputs.camera, inputs.aspect, direction, config, &previous);
        let rotation = set.cascades[0].light_view;

        let mut budget = REFRESH_BUDGET;
        for (index, cascade) in set.cascades[..set.count].iter().enumerate() {
            let volume = CasterVolume::Sweep {
                light_view: cascade.light_view,
                bounds: cascade.bounds,
            };
            let slice = swept_slice(inputs.camera, inputs.aspect, direction, &set.splits, index);
            let wanted = Wanted {
                key: ViewKey::Sun(index as u8),
                size: sizes[index],
                kind: ViewKind::Sun,
                view_proj: cascade.view_proj,
                volume,
                texel_scale: cascade.texel_world_size,
                window: cascade.window,
                rotation,
                visible: slice.map_or(volume, |slice| {
                    CasterVolume::Hull(slice.planes, slice.count)
                }),
                read: slice.map(|slice| slice.corners),
                in_view: true,
                channels: inputs.lighting.sun.channels,
                filtering: inputs.sun.filtering,
                bias: inputs.sun.bias,
                update: ShadowUpdate::Always,
                revision: 0,
            };
            // A cascade is fitted to its tile's texel count, so a tile of any
            // other size would shimmer. One that does not fit ends the sun's
            // views there, and the views before it answer out to their own
            // splits.
            if !self.plan_view(&wanted, inputs.casters, &mut budget, plan) {
                break;
            }
            plan.views.splits[index] = set.splits[index];
        }
        plan.views.sun = plan.views.views.len();
        let last = plan.views.splits[plan.views.sun.saturating_sub(1)];
        for split in &mut plan.views.splits[plan.views.sun..] {
            *split = last;
        }
    }

    fn plan_lamps(&mut self, inputs: &ShadowInputs<'_>, plan: &mut ShadowPlan) {
        if !inputs.atlas.punctual {
            return;
        }
        let lighting = inputs.lighting;
        let id = |kind, light: usize| match kind {
            LightKind::Point => lighting.point_lights[light].id,
            LightKind::Spot => lighting.spot_lights[light].id,
        };

        // A light whose reach is outside the view lights no pixel of the frame,
        // so nothing samples its shadow: the clusters that bin it for shading
        // are cut from this same frustum.
        let frustum = inputs.camera.frustum(inputs.aspect);
        let reaches_view = |kind, light: usize| {
            let (position, range) = match kind {
                LightKind::Point => {
                    let light = &lighting.point_lights[light];
                    (light.position, light.range)
                }
                LightKind::Spot => {
                    let light = &lighting.spot_lights[light];
                    (light.position, light.range)
                }
            };
            frustum.intersects(&Aabb {
                min: (position - Vec3::splat(range)).into(),
                max: (position + Vec3::splat(range)).into(),
            })
        };

        let mut ranked = std::mem::take(&mut self.ranked);
        punctual::rank(lighting, inputs.camera.position, &mut ranked);
        let holds = |entries: &[Entry], light: u64| {
            entries.iter().any(
                |entry| matches!(entry.key, ViewKey::Face { light: held, .. } if held == light),
            )
        };
        for candidate in &mut ranked {
            let held = holds(&self.entries, id(candidate.0, candidate.1));
            if held {
                candidate.2 *= INCUMBENT_BONUS;
            }
            if !reaches_view(candidate.0, candidate.1) {
                // Below every light that does, and not worth new tiles: it
                // keeps the ones it has for as long as nothing in view wants
                // them, so that turning back to it finds its cached layers.
                candidate.2 = if held {
                    candidate.2 * OUT_OF_VIEW
                } else {
                    -1.0
                };
            }
        }
        ranked.retain(|candidate| candidate.2 >= 0.0);
        ranked.sort_by(|a, b| b.2.total_cmp(&a.2));

        // Which lights the atlas has room for, decided before any of them is
        // placed: a light that ranks below the cut gives its tiles back first,
        // so that one newly ranked above it finds them free.
        let side = u64::from(self.tiles.resolution());
        let sun: u64 = plan
            .views
            .views
            .iter()
            .map(|view| AtlasRect::of(view.tile).texels())
            .sum();
        let entries = &self.entries;
        let tile_of = |kind, light: usize| {
            let lamp = Lamp::of(lighting, kind, light);
            let held = entries
                .iter()
                .find(|entry| {
                    entry.key
                        == ViewKey::Face {
                            light: lamp.id,
                            face: 0,
                        }
                })
                .map(|entry| entry.tile.size);
            lamp.tile(inputs, held)
        };
        let mut room = (side * side).saturating_sub(sun);
        let mut casting = 0;
        ranked.retain(|&(kind, light, _)| {
            let tile = u64::from(tile_of(kind, light));
            let needed = punctual::face_count(kind) as u64 * tile * tile;
            // Skipped rather than stopped: a spot still fits in the one tile a
            // point light could not use, and leaving it empty helps nobody.
            let fits = casting < MAX_SHADOW_LIGHTS && needed <= room;
            if fits {
                room -= needed;
                casting += 1;
            }
            fits
        });
        let (tiles, spare) = (&mut self.tiles, &mut self.spare);
        self.entries.retain_mut(|entry| {
            let ViewKey::Face { light, .. } = entry.key else {
                return true;
            };
            let kept = ranked
                .iter()
                .any(|&(kind, index, _)| id(kind, index) == light);
            if !kept {
                tiles.release(entry.tile);
                spare.push(std::mem::take(&mut entry.touched));
            }
            kept
        });

        // What the lamps may redraw is their own allowance rather than what the
        // sun left of its: sharing one would let a sun that scrolls every frame
        // keep every lamp drawn directly for as long as the camera moved.
        let mut budget = REFRESH_BUDGET;
        for &(kind, light, _) in &ranked {
            let needed = punctual::face_count(kind);
            let first_view = plan.views.views.len();
            let in_view = reaches_view(kind, light);
            let lamp = Lamp::of(lighting, kind, light);
            let held = self
                .entries
                .iter()
                .find(|entry| {
                    entry.key
                        == ViewKey::Face {
                            light: lamp.id,
                            face: 0,
                        }
                })
                .map(|entry| entry.tile.size);
            let size = lamp.tile(inputs, held);
            let faces = punctual::faces(lighting, kind, light, inputs.atlas.near, size);
            let or = |value: f32, default: f32| if value >= 0.0 { value } else { default };
            let mut placed = 0;
            for face in 0..needed {
                let wanted = Wanted {
                    key: ViewKey::Face {
                        light: id(kind, light),
                        face: face as u8,
                    },
                    size,
                    kind: ViewKind::Punctual,
                    view_proj: faces.view_proj[face],
                    volume: CasterVolume::Frustum(faces.view_proj[face]),
                    texel_scale: faces.texel_per_metre,
                    window: None,
                    rotation: Mat4::IDENTITY,
                    visible: CasterVolume::Frustum(faces.view_proj[face]),
                    read: None,
                    in_view,
                    channels: lamp.channels,
                    filtering: [
                        or(lamp.shadows.softness, inputs.lamps.filtering[0]),
                        or(lamp.shadows.sharpness, inputs.lamps.filtering[1])
                            .min(crate::scene::MAX_SHADOW_SHARPNESS),
                    ],
                    bias: [
                        or(lamp.shadows.constant_bias, inputs.lamps.bias[0]),
                        or(lamp.shadows.slope_bias, inputs.lamps.bias[1]),
                    ],
                    update: lamp.shadows.update,
                    revision: lamp.shadows.revision,
                };
                if !self.plan_view(&wanted, inputs.casters, &mut budget, plan) {
                    break;
                }
                placed += 1;
            }
            // Room by area is not room by shape. A light whose faces did not
            // all land gives back the ones that did.
            if placed < needed {
                self.unplan(first_view, plan);
                continue;
            }
            plan.views.casters.push(ShadowCaster {
                kind,
                light,
                first_view,
                view_count: needed,
                near: faces.near,
            });
        }
        self.ranked = ranked;
    }

    /// Take back the views planned from `first` on, and their tiles.
    fn unplan(&mut self, first: usize, plan: &mut ShadowPlan) {
        for view in plan.views.views.drain(first..) {
            let tile = AtlasRect::of(view.tile);
            if let Some(at) = self.entries.iter().position(|e| e.tile == view.tile) {
                let entry = self.entries.swap_remove(at);
                self.tiles.release(entry.tile);
                self.spare.push(entry.touched);
            }
            // The fills and restores carry no view, only where they land, and a
            // tile belongs to one view.
            let elsewhere = |rect: &AtlasRect| {
                rect.min[0] < tile.min[0]
                    || rect.min[1] < tile.min[1]
                    || rect.max[0] > tile.max[0]
                    || rect.max[1] > tile.max[1]
            };
            plan.static_fills.retain(elsewhere);
            plan.live_fills.retain(elsewhere);
            plan.restores.retain(|restore| elsewhere(&restore.rect));
        }
        plan.work.truncate(first);
        plan.draws.retain(|draw| draw.view < first);
    }

    /// Plan one view: find or make its entry, decide what this frame does to
    /// it, and append that to the plan. False when the atlas has no tile for
    /// it.
    fn plan_view(
        &mut self,
        wanted: &Wanted,
        casters: &CasterTracker,
        budget: &mut u64,
        plan: &mut ShadowPlan,
    ) -> bool {
        let found = self
            .entries
            .iter()
            .position(|entry| entry.key == wanted.key && entry.tile.size == wanted.size);
        let index = match found {
            Some(index) => index,
            None => {
                // An entry of another size is this view's old tile.
                if let Some(old) = self.entries.iter().position(|e| e.key == wanted.key) {
                    let old = self.entries.swap_remove(old);
                    self.tiles.release(old.tile);
                    self.spare.push(old.touched);
                }
                let Some(tile) = self
                    .tiles
                    .allocate(wanted.size)
                    .filter(|tile| tile.size == wanted.size)
                else {
                    return false;
                };
                self.entries.push(Entry {
                    key: wanted.key,
                    tile,
                    seen: 0,
                    view_proj: Mat4::ZERO,
                    channels: 0,
                    revision: 0,
                    cached: false,
                    live: Live::Unknown,
                    window: None,
                    base: IVec2::ZERO,
                    touched: self.spare.pop().unwrap_or_default(),
                });
                self.entries.len() - 1
            }
        };
        let frame = self.frame;
        let entry = &mut self.entries[index];
        entry.seen = frame;
        let tile = entry.tile;
        let whole = AtlasRect::of(tile);
        let view = plan.views.views.len();
        plan.views.views.push(ShadowView {
            view_proj: wanted.view_proj,
            tile,
            volume: wanted.volume,
            texel_scale: wanted.texel_scale,
            kind: wanted.kind,
            channels: wanted.channels,
            filtering: wanted.filtering,
            bias: wanted.bias,
        });
        // Which casters each kind of draw takes. A light that updates on
        // demand keeps a picture of everything in reach when it was asked; one
        // that updates on change never draws what moves at all.
        let (kept, direct) = match wanted.update {
            ShadowUpdate::Always => (Casters::Static, Casters::All),
            ShadowUpdate::OnChange => (Casters::Static, Casters::Static),
            ShadowUpdate::OnDemand => (Casters::All, Casters::All),
        };
        let cutoff = match wanted.kind {
            ViewKind::Sun => [wanted.texel_scale * SMALL_CASTER_TEXELS, 0.0],
            ViewKind::Punctual => [0.0, wanted.texel_scale * SMALL_CASTER_TEXELS],
        };
        // A layer the frame throws away takes only what can shadow what the
        // camera sees. The cached layer has to be right wherever the camera
        // turns next, and takes everything that reaches the view.
        let draw = |layer, casters, scissor| CasterDraw {
            view,
            layer,
            casters,
            volume: if layer == Layer::Live {
                wanted.visible
            } else {
                wanted.volume
            },
            view_proj: wanted.view_proj,
            tile,
            scissor,
            cutoff,
        };

        // Nothing reads a tile whose light reaches nothing in view, so nothing
        // is done to it. What it holds goes stale meanwhile, and is put right
        // the frame the light is back.
        if !wanted.in_view {
            if entry.cached
                && casters.changed().is_none_or(|changed| {
                    changed
                        .iter()
                        .any(|bounds| wanted.volume.reached_by(bounds))
                })
            {
                entry.cached = false;
            }
            entry.live = Live::Unknown;
            plan.work.push(ViewWork::Held);
            return true;
        }

        // What moved between the matrix the tile holds and the one asked for.
        // Other channels are other casters, which is another picture as surely
        // as another matrix is.
        let same = entry.view_proj == wanted.view_proj && entry.channels == wanted.channels;
        let slid = match (&entry.window, &wanted.window) {
            (Some(old), Some(new)) if !same && entry.cached => {
                let by = (new.min_cell - old.min_cell).abs();
                (old.cells == new.cells
                    && old.texel == new.texel
                    && old.z_eye == new.z_eye
                    && old.depth_range == new.depth_range
                    && by.max_element() < new.cells as i32)
                    .then_some(*old)
            }
            _ => None,
        };
        if !same && slid.is_none() {
            entry.cached = false;
        }
        if entry.revision != wanted.revision {
            entry.cached = false;
        }
        // A caster that joined or left the static set inside the view has made
        // its cached layer wrong, wherever in it the caster stands — unless
        // the layer is a picture taken on demand, which nothing but a demand
        // retakes.
        if entry.cached
            && wanted.update != ShadowUpdate::OnDemand
            && casters.changed().is_none_or(|changed| {
                changed
                    .iter()
                    .any(|bounds| wanted.volume.reached_by(bounds))
            })
        {
            entry.cached = false;
        }
        entry.view_proj = wanted.view_proj;
        entry.channels = wanted.channels;
        entry.revision = wanted.revision;

        if !entry.cached {
            let cost = whole.texels();
            // Only once the matrix has held for a frame: a light that moves
            // every frame would redraw a cached layer it never gets to reuse.
            if same && cost <= *budget {
                *budget -= cost;
                entry.cached = true;
                entry.window = wanted.window;
                entry.base = wanted.window.map_or(IVec2::ZERO, |window| window.min_cell);
                plan.static_fills.push(whole);
                plan.draws.push(draw(Layer::Static, kept, whole));
                plan.restores.push(Restore {
                    rect: whole,
                    tile,
                    wrap: [0, 0],
                });
                plan.work.push(ViewWork::Refreshed);
            } else {
                // Only as much of the tile as the frame can read. Clearing a
                // tile is a touch of every texel in it, and a view drawn
                // directly is cleared every frame it stays so.
                let read = wanted
                    .read
                    .and_then(|corners| {
                        read_rect(&corners, &wanted.view_proj, tile, wanted.texel_scale)
                    })
                    .unwrap_or(whole);
                entry.window = wanted.window;
                entry.live = Live::Everything;
                entry.touched.clear();
                plan.live_fills.push(read);
                plan.draws.push(draw(Layer::Live, direct, read));
                plan.work.push(ViewWork::Direct);
                return true;
            }
        } else if let (Some(old), Some(new)) = (slid, wanted.window) {
            entry.window = Some(new);
            strips(
                &old,
                &new,
                entry.base,
                tile,
                wanted.rotation,
                |volume, view_proj, scissor| {
                    plan.static_fills.push(scissor);
                    plan.draws.push(CasterDraw {
                        view,
                        layer: Layer::Static,
                        casters: kept,
                        volume,
                        view_proj,
                        tile,
                        scissor,
                        cutoff,
                    });
                },
            );
            // The live tile is laid out by the window, so a window that moved
            // has moved every texel of it.
            plan.restores.push(Restore {
                rect: whole,
                tile,
                wrap: entry.wrap(),
            });
            plan.work.push(ViewWork::Scrolled);
        } else if entry.live != Live::Composite {
            plan.restores.push(Restore {
                rect: whole,
                tile,
                wrap: entry.wrap(),
            });
            plan.work.push(ViewWork::Composited);
        } else {
            let wrap = entry.wrap();
            plan.restores.extend(
                entry
                    .touched
                    .iter()
                    .map(|&rect| Restore { rect, tile, wrap }),
            );
            plan.work.push(if entry.touched.is_empty() {
                ViewWork::Held
            } else {
                ViewWork::Composited
            });
        }

        // The moving casters, over a live tile that now holds the cached layer
        // wherever they are about to draw.
        entry.live = Live::Composite;
        entry.touched.clear();
        if wanted.update != ShadowUpdate::Always {
            return true;
        }
        match casters.dynamic() {
            Some(dynamic) => {
                for bounds in dynamic {
                    if !wanted.visible.reached_by(bounds) {
                        continue;
                    }
                    match footprint(bounds, &wanted.view_proj, tile) {
                        Some(rect) if entry.touched.len() < MAX_TOUCHED => entry.touched.push(rect),
                        Some(rect) => {
                            let around = entry.touched.iter().fold(rect, |a, b| a.union(b));
                            entry.touched.clear();
                            entry.touched.push(around);
                        }
                        None => {}
                    }
                }
            }
            None => entry.touched.push(whole),
        }
        if !entry.touched.is_empty() {
            plan.draws.push(draw(Layer::Live, Casters::Dynamic, whole));
            if plan.work[view] == ViewWork::Held {
                plan.work[view] = ViewWork::Composited;
            }
        }
        true
    }
}

/// The rectangle of `tile` a caster with these world bounds can write through
/// `view_proj`, padded by a texel for the rasteriser's rounding, or `None` when
/// it misses the tile.
fn footprint(bounds: &Aabb, view_proj: &Mat4, tile: AtlasTile) -> Option<AtlasRect> {
    let whole = AtlasRect::of(tile);
    if !bounds.is_valid() {
        return Some(whole);
    }
    let (min, max) = (Vec3::from(bounds.min), Vec3::from(bounds.max));
    let mut low = glam::Vec2::splat(f32::INFINITY);
    let mut high = glam::Vec2::splat(f32::NEG_INFINITY);
    for corner in 0..8 {
        let point = Vec3::new(
            if corner & 1 == 0 { min.x } else { max.x },
            if corner & 2 == 0 { min.y } else { max.y },
            if corner & 4 == 0 { min.z } else { max.z },
        );
        let clip = *view_proj * point.extend(1.0);
        // A corner at or behind the eye has no place on the tile, and the box
        // it belongs to may be anywhere on it.
        if clip.w <= 1e-6 {
            return Some(whole);
        }
        let at = (clip.truncate().truncate() / clip.w) * 0.5 + 0.5;
        low = low.min(at);
        high = high.max(at);
    }
    let size = tile.size as f32;
    let low = (low * size - 1.0).floor().max(glam::Vec2::ZERO);
    let high = (high * size + 1.0).ceil().min(glam::Vec2::splat(size));
    if low.x >= high.x || low.y >= high.y {
        return None;
    }
    Some(AtlasRect {
        min: [tile.offset[0] + low.x as u32, tile.offset[1] + low.y as u32],
        max: [
            tile.offset[0] + high.x as u32,
            tile.offset[1] + high.y as u32,
        ],
    })
}

/// How far past what the camera can see a lookup reaches: the filter's kernel
/// and the normal offset in texels, and in metres the thickness a scattering
/// surface looks behind itself by.
const READ_MARGIN_TEXELS: f32 = 8.0;
const READ_MARGIN_METRES: f32 = 0.25;

/// The rectangle of `tile` the frame can read, given the corners of the part
/// of the view it reads from.
fn read_rect(
    corners: &[Vec3; 8],
    view_proj: &Mat4,
    tile: AtlasTile,
    texel: f32,
) -> Option<AtlasRect> {
    let margin = READ_MARGIN_TEXELS + READ_MARGIN_METRES / texel.max(1e-6);
    let mut low = glam::Vec2::splat(f32::INFINITY);
    let mut high = glam::Vec2::splat(f32::NEG_INFINITY);
    for corner in corners {
        let at = view_proj.project_point3(*corner).truncate() * 0.5 + 0.5;
        low = low.min(at);
        high = high.max(at);
    }
    let size = tile.size as f32;
    let low = (low * size - margin).floor().max(glam::Vec2::ZERO);
    let high = (high * size + margin).ceil().min(glam::Vec2::splat(size));
    (low.x < high.x && low.y < high.y).then(|| AtlasRect {
        min: [tile.offset[0] + low.x as u32, tile.offset[1] + low.y as u32],
        max: [
            tile.offset[0] + high.x as u32,
            tile.offset[1] + high.y as u32,
        ],
    })
}

/// The strips a level's window moved onto, as draws into its cached tile.
///
/// The cached tile keeps cell `c` at `(c - base) mod cells`, so the cells the
/// old and new windows share are already in place and only the new ones are
/// drawn: a strip the window's whole height where it moved along the first
/// axis, and one across what is left where it moved along the second. A strip
/// that crosses the tile's edge is drawn in two pieces, each with the matrix
/// that lands its cells where they are kept.
fn strips(
    old: &LevelWindow,
    new: &LevelWindow,
    base: IVec2,
    tile: AtlasTile,
    rotation: Mat4,
    mut emit: impl FnMut(CasterVolume, Mat4, AtlasRect),
) {
    let n = new.cells as i32;
    let (old_min, new_min) = (old.min_cell, new.min_cell);
    let (old_max, new_max) = (old_min + n, new_min + n);
    // The columns the window gained, then the rows, the rows only across the
    // columns it kept so that the corner is drawn once.
    let gained = |old_min: i32, old_max: i32, new_min: i32, new_max: i32| {
        if new_min > old_min {
            old_max.max(new_min)..new_max
        } else {
            new_min..old_min.min(new_max)
        }
    };
    let columns = gained(old_min.x, old_max.x, new_min.x, new_max.x);
    let rows = gained(old_min.y, old_max.y, new_min.y, new_max.y);
    let kept = old_min.x.max(new_min.x)..old_max.x.min(new_max.x);
    let rects = [(columns, new_min.y..new_max.y), (kept, rows)];
    for (xs, ys) in rects {
        if xs.is_empty() || ys.is_empty() {
            continue;
        }
        // Cut where the cells' kept places wrap round the tile.
        for (x0, x1) in pieces(xs.clone(), base.x, n) {
            for (y0, y1) in pieces(ys.clone(), base.y, n) {
                let page = IVec2::new(
                    base.x + (x0 - base.x).div_euclid(n) * n,
                    base.y + (y0 - base.y).div_euclid(n) * n,
                );
                let scissor = AtlasRect {
                    min: [
                        tile.offset[0] + (x0 - page.x) as u32,
                        tile.offset[1] + (y0 - page.y) as u32,
                    ],
                    max: [
                        tile.offset[0] + (x1 - page.x) as u32,
                        tile.offset[1] + (y1 - page.y) as u32,
                    ],
                };
                emit(
                    CasterVolume::Sweep {
                        light_view: rotation,
                        bounds: new.sweep(IVec2::new(x0, y0), IVec2::new(x1, y1)),
                    },
                    new.page_view_proj(rotation, page),
                    scissor,
                );
            }
        }
    }
}

/// A run of cells cut at the multiples of `n` from `base`: at most two pieces
/// for a run shorter than `n`.
fn pieces(cells: std::ops::Range<i32>, base: i32, n: i32) -> impl Iterator<Item = (i32, i32)> {
    let cut = base + ((cells.start - base).div_euclid(n) + 1) * n;
    let first = (cells.start, cells.end.min(cut));
    let second = (cut < cells.end).then_some((cut, cells.end));
    std::iter::once(first).chain(second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::shadow_casters::STATIC_AFTER;
    use crate::gfx::{
        DrawList, MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS, PointLight, RenderItem, SpotLight,
    };
    use crate::scene::{MaterialHandle, MeshHandle};
    use glam::Vec3A;

    fn point(id: u64, position: Vec3) -> PointLight {
        PointLight {
            id,
            position,
            color: Vec3::ONE,
            candela: 1.0,
            range: 10.0,
            casts_shadows: true,
            channels: crate::scene::ALL_CHANNELS,
            shadows: Default::default(),
        }
    }

    fn spot(id: u64, position: Vec3) -> SpotLight {
        SpotLight {
            id,
            position,
            direction: Vec3::NEG_Y,
            color: Vec3::ONE,
            candela: 1.0,
            range: 10.0,
            inner_cos: 0.9,
            outer_cos: 0.7,
            casts_shadows: true,
            channels: crate::scene::ALL_CHANNELS,
            shadows: Default::default(),
        }
    }

    fn cascade_config() -> CascadeConfig {
        CascadeConfig {
            count: 4,
            max_distance: 100.0,
            lambda: 0.75,
            resolution: 2048,
            far_resolution: 1024,
            pullback: 50.0,
            fitted: 0,
        }
    }

    fn caster(instance: u32, at: Vec3) -> RenderItem {
        RenderItem {
            model: Mat4::from_translation(at),
            prev_model: Mat4::from_translation(at),
            bounds: Aabb::EMPTY,
            mesh: MeshHandle(0),
            material: MaterialHandle(0),
            instance,
            channels: crate::scene::ALL_CHANNELS,
            shadows: Default::default(),
        }
    }

    /// A scene, a planner and the tracker feeding it, stepped a frame at a
    /// time the way the renderer steps them.
    struct Rig {
        planner: ShadowPlanner,
        tracker: CasterTracker,
        plan: ShadowPlan,
        lighting: SceneLighting,
        camera: Camera,
        sun: Option<CascadeConfig>,
        atlas: AtlasConfig,
        /// Where each caster stands.
        casters: Vec<Vec3>,
    }

    impl Rig {
        fn new(points: Vec<PointLight>, spots: Vec<SpotLight>) -> Self {
            let mut lighting = SceneLighting {
                point_lights: points,
                spot_lights: spots,
                ..SceneLighting::default()
            };
            lighting.sun.direction = Vec3::new(-0.4, -1.0, -0.3);
            Self {
                planner: ShadowPlanner::default(),
                tracker: CasterTracker::default(),
                plan: ShadowPlan::default(),
                lighting,
                camera: Camera::default(),
                sun: Some(cascade_config()),
                atlas: AtlasConfig::default(),
                casters: Vec::new(),
            }
        }

        fn frame(&mut self) -> &ShadowPlan {
            let items: Vec<RenderItem> = self
                .casters
                .iter()
                .enumerate()
                .map(|(index, at)| caster(index as u32, *at))
                .collect();
            let order: Vec<u32> = (0..items.len() as u32).collect();
            self.tracker.update(DrawList::new(&items, &order), |_| {
                Some(Aabb {
                    min: Vec3A::splat(-0.5),
                    max: Vec3A::splat(0.5),
                })
            });
            self.planner.plan(
                Some(&ShadowInputs {
                    camera: &self.camera,
                    aspect: 16.0 / 9.0,
                    lighting: &self.lighting,
                    cascades: self.sun.as_ref(),
                    atlas: &self.atlas,
                    casters: &self.tracker,
                    viewport_height: 1080.0,
                    sun: ViewDefaults::default(),
                    lamps: ViewDefaults::default(),
                }),
                &mut self.plan,
            );
            &self.plan
        }

        /// Run until every caster has settled and every cached layer has had
        /// its turn to be drawn.
        fn settle(&mut self) {
            for _ in 0..STATIC_AFTER as usize + 16 {
                self.frame();
            }
        }
    }

    fn overlap(a: AtlasTile, b: AtlasTile) -> bool {
        let apart = |axis: usize| {
            a.offset[axis] + a.size <= b.offset[axis] || b.offset[axis] + b.size <= a.offset[axis]
        };
        !(apart(0) || apart(1))
    }

    /// No two views may land on the same texels, whatever their sizes. This is
    /// the one mistake the allocator can make that renders perfectly and
    /// shadows wrongly: the tiles draw fine, and one light reads another's
    /// depth. And every tile has to be inside the image — an offset past the
    /// edge is a viewport the driver will happily accept and clip to nothing.
    #[test]
    fn no_two_views_share_a_texel_and_every_tile_is_inside_the_atlas() {
        let points = (0..MAX_POINT_LIGHTS)
            .map(|i| point(i as u64, Vec3::X * i as f32))
            .collect();
        let spots = (0..MAX_SPOT_LIGHTS)
            .map(|i| spot(1000 + i as u64, Vec3::Y * i as f32))
            .collect();
        let mut rig = Rig::new(points, spots);
        let views = &rig.frame().views;

        assert_eq!(views.sun, 4);
        assert!(views.views.len() > views.sun, "no lamp got a tile");
        for (i, a) in views.views.iter().enumerate() {
            assert!(a.tile.offset[0] + a.tile.size <= views.resolution);
            assert!(a.tile.offset[1] + a.tile.size <= views.resolution);
            for b in &views.views[i + 1..] {
                assert!(!overlap(a.tile, b.tile), "{:?} and {:?}", a.tile, b.tile);
            }
        }
    }

    /// The shape of the thing: the sun's views lead, a point takes six tiles and
    /// a spot one, and each caster's views are contiguous so its range is a
    /// slice.
    #[test]
    fn the_suns_views_lead_and_each_light_owns_a_run() {
        let mut rig = Rig::new(vec![point(1, Vec3::ZERO)], vec![spot(2, Vec3::X)]);
        let config = cascade_config();
        let views = &rig.frame().views;

        assert_eq!(views.views.len(), 4 + 7);
        assert!(views.views[..4].iter().all(|v| v.kind == ViewKind::Sun));
        let point = views.caster(LightKind::Point, 0).unwrap();
        let spot = views.caster(LightKind::Spot, 0).unwrap();
        assert_eq!((point.first_view, point.view_count), (4, 6));
        assert_eq!((spot.first_view, spot.view_count), (10, 1));
        for (index, view) in views.views[..4].iter().enumerate() {
            assert_eq!(view.tile.size, config.resolution_of(index));
        }
    }

    /// A point light that does not fit must not stop a spot that does. Leaving
    /// the last tile empty because the light in front of it wanted six is a
    /// shadow nobody gets for no saving.
    #[test]
    fn a_light_too_big_for_the_gap_is_skipped_not_final() {
        let mut rig = Rig::new(vec![point(1, Vec3::ZERO)], vec![spot(2, Vec3::ZERO)]);
        rig.lighting.point_lights[0].candela = 100.0;
        rig.sun = None;
        // Four tiles. The point light needs six.
        rig.atlas = AtlasConfig {
            resolution: 1024,
            tile_size: 512,
            ..AtlasConfig::default()
        };
        let views = &rig.frame().views;

        assert!(views.caster(LightKind::Point, 0).is_none());
        assert_eq!(
            views.caster(LightKind::Spot, 0).map(|c| c.view_count),
            Some(1),
            "the spot lost its tile to a light that never took one"
        );
        assert_eq!(views.views.len(), 1);
    }

    /// Nothing casting is the shape the graph reads as "declare no atlas at
    /// all", so it has to be reachable rather than approximated by an atlas of
    /// empty tiles.
    #[test]
    fn a_scene_with_no_casters_has_no_atlas() {
        let mut rig = Rig::new(vec![point(1, Vec3::ZERO)], Vec::new());
        rig.lighting.point_lights[0].casts_shadows = false;
        rig.sun = None;
        let views = &rig.frame().views;
        assert!(views.views.is_empty());
        assert_eq!(views.resolution, 0);
    }

    /// The tile rect is what the shader multiplies its NDC into, so it has to
    /// address the same texels the viewport wrote.
    #[test]
    fn a_tile_maps_to_its_own_corner_of_the_atlas() {
        let tile = AtlasTile {
            offset: [512, 1024],
            size: 512,
        };
        let [x, y, w, h] = tile.uv_rect(2048);
        assert_eq!((x, y, w, h), (0.25, 0.5, 0.25, 0.25));
    }

    /// A released tile is whole again: an atlas filled with small tiles and
    /// emptied has to hand out the one tile that spans it.
    #[test]
    fn released_tiles_join_back_into_the_block_they_were_split_from() {
        let mut tiles = TileAllocator::new(1024);
        let small: Vec<_> = (0..16).map(|_| tiles.allocate(256).unwrap()).collect();
        assert!(tiles.allocate(256).is_none(), "the atlas holds sixteen");
        for tile in small {
            tiles.release(tile);
        }
        assert_eq!(
            tiles.allocate(1024),
            Some(AtlasTile {
                offset: [0, 0],
                size: 1024
            })
        );
    }

    /// What the cache is for. Once every caster has settled and every cached
    /// layer has been drawn, a scene in which nothing moves asks for no shadow
    /// work at all: no draw, no clear, no restore.
    #[test]
    fn a_still_scene_draws_no_shadow_geometry() {
        let mut rig = Rig::new(vec![point(1, Vec3::new(0.0, 3.0, 0.0))], Vec::new());
        rig.casters = vec![Vec3::ZERO, Vec3::X * 2.0];
        rig.settle();
        let plan = rig.frame();

        assert!(plan.draws.is_empty(), "{:?}", plan.draws);
        assert!(!plan.draws_static() && !plan.draws_live());
        assert!(plan.work.iter().all(|work| *work == ViewWork::Held));
    }

    /// And with something moving, the frame pays for that thing alone: its
    /// own draw into the views it reaches, and the texels it touched last
    /// frame put back — never a cached layer redrawn, never a whole tile.
    #[test]
    fn a_moving_caster_redraws_only_what_it_touched() {
        let mut rig = Rig::new(vec![point(1, Vec3::new(0.0, 3.0, 0.0))], Vec::new());
        rig.casters = vec![Vec3::ZERO, Vec3::X * 2.0];
        rig.settle();
        // The first step is the frame the caster leaves the cached layers it
        // was baked into, which redraws them; that takes its turns. What is
        // asserted is the frames after.
        for step in 1..24 {
            rig.casters[1] = Vec3::X * (2.0 + 0.01 * step as f32);
            let plan = rig.frame();
            if step < 16 {
                continue;
            }

            assert!(!plan.draws_static(), "a cached layer was redrawn");
            assert!(plan.live_fills.is_empty(), "a tile was cleared whole");
            assert!(!plan.draws.is_empty(), "the moving caster was not drawn");
            assert!(
                plan.draws
                    .iter()
                    .all(|draw| draw.layer == Layer::Live && draw.casters == Casters::Dynamic)
            );
            assert!(
                !plan.restores.is_empty(),
                "what it touched was not put back"
            );
            for restore in &plan.restores {
                assert!(
                    restore.rect.texels() < AtlasRect::of(restore.tile).texels() / 4,
                    "a whole tile was restored for one small caster: {restore:?}",
                );
            }
        }
    }

    /// A caster leaving the static set makes the cached layers it was baked
    /// into wrong — and no others. A lamp too far away to reach it keeps every
    /// tile it has.
    #[test]
    fn a_caster_that_starts_moving_uncaches_the_views_it_stood_in_and_no_others() {
        let near = point(1, Vec3::new(0.0, 3.0, 0.0));
        let far = point(2, Vec3::new(0.0, 3.0, -40.0));
        let mut rig = Rig::new(vec![near, far], Vec::new());
        rig.sun = None;
        rig.casters = vec![Vec3::ZERO];
        rig.settle();

        rig.casters[0] = Vec3::X * 0.5;
        let plan = rig.frame().clone();
        let views = |light: usize| {
            let caster = plan.views.caster(LightKind::Point, light).unwrap();
            caster.first_view..caster.first_view + caster.view_count
        };
        assert!(
            plan.work[views(0)]
                .iter()
                .any(|work| matches!(work, ViewWork::Refreshed | ViewWork::Direct)),
            "{:?}",
            &plan.work[views(0)]
        );
        assert!(
            plan.work[views(1)]
                .iter()
                .all(|work| *work == ViewWork::Held),
            "{:?}",
            &plan.work[views(1)]
        );
    }

    /// A light that moves has no cached layer worth drawing: every frame is a
    /// different matrix. It is drawn directly while it moves, and cached the
    /// frame after it comes to rest.
    #[test]
    fn a_moving_light_is_drawn_directly_until_it_rests() {
        let mut rig = Rig::new(Vec::new(), vec![spot(1, Vec3::new(0.0, 5.0, 0.0))]);
        rig.sun = None;
        rig.casters = vec![Vec3::ZERO];
        rig.settle();

        for step in 1..4 {
            rig.lighting.spot_lights[0].position.x = 0.1 * step as f32;
            let plan = rig.frame();
            assert_eq!(plan.work, [ViewWork::Direct]);
            assert!(
                !plan.draws_static(),
                "a layer was cached for one frame's use"
            );
        }
        assert_eq!(rig.frame().work, [ViewWork::Refreshed]);
        assert_eq!(rig.frame().work, [ViewWork::Held]);
    }

    /// A light whose reach is out of view lights nothing on screen, so nothing
    /// is done for it however much moves around it — and what its tiles hold
    /// by then is not trusted the frame it comes back.
    #[test]
    fn a_light_that_reaches_nothing_in_view_is_left_alone_until_it_does() {
        // Twenty metres ahead, so that turning round puts all of its reach
        // behind the camera.
        let under = Vec3::new(0.0, 0.0, -20.0);
        let mut rig = Rig::new(Vec::new(), vec![spot(1, under + Vec3::Y * 5.0)]);
        rig.sun = None;
        rig.casters = vec![under, under + Vec3::X];
        rig.settle();

        // Face away, and keep a caster moving under the light.
        let ahead = rig.camera.target;
        rig.camera.target = rig.camera.position + (rig.camera.position - ahead);
        for step in 1..12 {
            rig.casters[1] = under + Vec3::X * (1.0 + 0.05 * step as f32);
            let plan = rig.frame();
            assert_eq!(plan.views.views.len(), 1, "it lost the tile it had");
            assert!(!plan.draws_static() && !plan.draws_live(), "step {step}");
        }

        rig.camera.target = ahead;
        let plan = rig.frame();
        assert!(
            plan.draws_live(),
            "it came back into view and was not redrawn"
        );
        let whole = AtlasRect::of(plan.views.views[0].tile);
        assert!(
            plan.live_fills.contains(&whole) || plan.restores.iter().any(|r| r.rect == whole),
            "a tile that sat out {:?} was trusted as it was",
            plan.work
        );
    }

    /// A lamp's tile is sized by how much of the screen its reach covers, so a
    /// light across the level does not hold the texels of one overhead — and it
    /// keeps the tile it has while the camera hovers at the distance where one
    /// size gives way to the next, because every change of size is the light
    /// redrawn whole.
    #[test]
    fn a_lamps_tile_follows_its_reach_on_screen_and_holds_at_a_boundary() {
        let mut rig = Rig::new(Vec::new(), vec![spot(1, Vec3::new(0.0, 5.0, -30.0))]);
        rig.sun = None;
        let size = |rig: &mut Rig, distance: f32| {
            rig.camera.position = Vec3::new(0.0, 5.0, -30.0 + distance);
            rig.camera.target = Vec3::new(0.0, 5.0, -30.0);
            rig.frame().views.views[0].tile.size
        };

        let near = size(&mut rig, 5.0);
        assert_eq!(
            near, rig.atlas.tile_size,
            "inside its reach it takes the largest"
        );
        let mut sizes = vec![near];
        let mut shrank_at = 0.0;
        for step in 0..200 {
            let distance = 12.0 + step as f32;
            let next = size(&mut rig, distance);
            if next != *sizes.last().unwrap() {
                sizes.push(next);
                shrank_at = distance;
            }
        }
        assert!(sizes.len() >= 3, "it never shrank: {sizes:?}");
        assert!(sizes.windows(2).all(|pair| pair[1] < pair[0]), "{sizes:?}");
        assert!(sizes.iter().all(|size| size.is_power_of_two()));

        // Back and forth across the distance it last shrank at.
        let mut last = size(&mut rig, shrank_at);
        for step in 0..40 {
            let wobble = if step % 2 == 0 { 0.97 } else { 1.03 };
            let now = size(&mut rig, shrank_at * wobble);
            assert_eq!(now, last, "a tile that changes size at a boundary");
            last = now;
        }
    }

    /// The three ways a light can be told to keep its shadow. One that updates
    /// on change never draws what moves, however it moves; one that updates on
    /// demand draws nothing at all until it is asked, and then everything.
    #[test]
    fn a_light_draws_no_more_often_than_its_update_mode_says() {
        use crate::scene::ShadowUpdate;
        for update in [ShadowUpdate::OnChange, ShadowUpdate::OnDemand] {
            let mut lamp = spot(1, Vec3::new(0.0, 5.0, 0.0));
            lamp.shadows.update = update;
            let mut rig = Rig::new(Vec::new(), vec![lamp]);
            rig.sun = None;
            rig.casters = vec![Vec3::ZERO, Vec3::X];
            rig.settle();

            // Something moving under it.
            for step in 1..12 {
                rig.casters[1] = Vec3::X * (1.0 + 0.05 * step as f32);
                let plan = rig.frame();
                assert!(
                    plan.draws
                        .iter()
                        .all(|draw| draw.casters != Casters::Dynamic),
                    "{update:?} drew what moves"
                );
                if update == ShadowUpdate::OnDemand {
                    assert!(!plan.draws_static() && !plan.draws_live(), "step {step}");
                }
            }

            if update == ShadowUpdate::OnDemand {
                rig.lighting.spot_lights[0].shadows.revision += 1;
                let plan = rig.frame();
                assert!(plan.draws_static(), "it was asked and did not redraw");
                assert!(
                    plan.draws.iter().all(|draw| draw.casters == Casters::All),
                    "a picture on demand is of everything in reach"
                );
                assert_eq!(rig.frame().work, [ViewWork::Held]);
            }
        }
    }

    /// A light's channels say whose shadows it takes, so changing them changes
    /// what its tiles should hold as surely as moving it does.
    #[test]
    fn a_light_whose_channels_change_is_redrawn() {
        let mut rig = Rig::new(Vec::new(), vec![spot(1, Vec3::new(0.0, 5.0, 0.0))]);
        rig.sun = None;
        rig.casters = vec![Vec3::ZERO];
        rig.settle();
        assert_eq!(rig.frame().work, [ViewWork::Held]);

        rig.lighting.spot_lights[0].channels = 0b10;
        let plan = rig.frame();
        assert_eq!(plan.views.views[0].channels, 0b10);
        assert!(plan.draws_live(), "{:?}", plan.work);
    }

    /// The budget spreads the cost of starting to cache, and only that: a view
    /// still waiting for its cached layer is drawn directly, with every
    /// caster, so nothing is ever missing a shadow because of it.
    #[test]
    fn a_view_waiting_for_its_cached_layer_is_drawn_directly() {
        let points = (0..4)
            .map(|i| point(i, Vec3::new(i as f32 * 3.0, 3.0, 0.0)))
            .collect();
        let mut rig = Rig::new(points, Vec::new());
        rig.sun = None;
        rig.atlas.tile_size = 1024;
        rig.casters = vec![Vec3::ZERO];
        // Past the first frame, in which every view is new and drawn directly.
        rig.frame();
        let plan = rig.frame();
        let refreshed = plan
            .work
            .iter()
            .filter(|work| **work == ViewWork::Refreshed)
            .count();
        let direct = plan
            .work
            .iter()
            .filter(|work| **work == ViewWork::Direct)
            .count();
        assert!(refreshed > 0 && direct > 0, "{:?}", plan.work);
        assert!(refreshed as u64 * 1024 * 1024 <= REFRESH_BUDGET);
        assert_eq!(refreshed + direct, plan.work.len());
        // And they all get there.
        rig.settle();
        assert!(rig.frame().work.iter().all(|work| *work == ViewWork::Held));
    }

    /// Where a draw puts a world point, as a texel of the atlas, if it draws it
    /// at all.
    fn drawn_at(draw: &CasterDraw, point: Vec3) -> Option<[u32; 2]> {
        let ndc = draw.view_proj.project_point3(point);
        let at = (ndc.truncate() * 0.5 + 0.5) * draw.tile.size as f32;
        let texel = [
            draw.tile.offset[0] as i64 + at.x.floor() as i64,
            draw.tile.offset[1] as i64 + at.y.floor() as i64,
        ];
        let inside = (0..2).all(|axis| {
            texel[axis] >= i64::from(draw.scissor.min[axis])
                && texel[axis] < i64::from(draw.scissor.max[axis])
        });
        inside.then_some([texel[0] as u32, texel[1] as u32])
    }

    /// The whole of scrolling in one property. As the camera walks, a level's
    /// cached layer is drawn a strip at a time into wherever those cells are
    /// kept, and the live tile is restored from it through the wrap. So for
    /// any point the level covers, the cached texel the restore reads for it
    /// has to be the texel the last draw that covered it wrote — through
    /// however many moves, across the tile's edge and back.
    #[test]
    fn a_scrolled_level_restores_each_cell_from_where_it_was_drawn() {
        let mut rig = Rig::new(Vec::new(), Vec::new());
        rig.casters = vec![Vec3::ZERO];
        rig.settle();
        let level = 1;

        // Sample points the level covers throughout, near where the walk goes.
        let samples: Vec<Vec3> = (0..40)
            .map(|i| {
                let t = i as f32;
                rig.camera.position + Vec3::new((t * 1.7).sin() * 4.0, 0.0, (t * 0.9).cos() * 4.0)
            })
            .collect();
        let mut written: Vec<Option<[u32; 2]>> = vec![None; samples.len()];
        let mut scrolled = 0;
        // A cached layer drawn whole writes every sample; start from one.
        rig.planner.invalidate();
        let heading = Vec3::new(1.0, 0.0, 0.6).normalize();
        for step in 0..400 {
            // Out, back past the start, and out again, so the window crosses
            // its tile's edge in both directions.
            let phase = (step as f32 / 400.0 * std::f32::consts::TAU).sin();
            let moved = heading * phase * 9.0;
            rig.camera.target += moved - (rig.camera.position - samples_origin(&samples));
            rig.camera.position = samples_origin(&samples) + moved;
            let plan = rig.frame().clone();
            scrolled += usize::from(plan.work[level] == ViewWork::Scrolled);
            for draw in plan
                .draws
                .iter()
                .filter(|draw| draw.view == level && draw.layer == Layer::Static)
            {
                for (sample, texel) in samples.iter().zip(&mut written) {
                    if let Some(at) = drawn_at(draw, *sample) {
                        *texel = Some(at);
                    }
                }
            }
            if plan.work[level] == ViewWork::Direct {
                written.fill(None);
                continue;
            }

            // What the restore reads for each sample's live texel.
            let view = &plan.views.views[level];
            let wrap = rig
                .planner
                .entries
                .iter()
                .find(|entry| entry.key == ViewKey::Sun(level as u8))
                .unwrap()
                .wrap();
            for (sample, texel) in samples.iter().zip(&written) {
                let ndc = view.view_proj.project_point3(*sample);
                let live = ((ndc.truncate() * 0.5 + 0.5) * view.tile.size as f32).floor();
                let size = view.tile.size;
                let read = [
                    view.tile.offset[0] + (live.x as u32 + wrap[0]) % size,
                    view.tile.offset[1] + (live.y as u32 + wrap[1]) % size,
                ];
                assert_eq!(Some(read), *texel, "step {step}: {sample}");
            }
        }
        assert!(
            scrolled >= 4,
            "the walk only scrolled the level {scrolled} times"
        );
    }

    fn samples_origin(samples: &[Vec3]) -> Vec3 {
        samples.iter().sum::<Vec3>() / samples.len() as f32
    }
}
