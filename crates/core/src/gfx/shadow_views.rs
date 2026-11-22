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
//! Nothing here takes a `Device`, for the reason the render graph does not: the
//! allocation and the matrices are the part that can be wrong in a way pixels
//! only hint at, so they are the part CI can assert.

use glam::Mat4;

use super::SceneLighting;
use super::punctual::{self, LightKind, MAX_SHADOW_LIGHTS};
use super::shadows::{CascadeConfig, CascadeSet, MAX_CASCADES, cascades};
use crate::scene::Camera;

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

/// What can cast into a view, which is what the cull dispatch tests.
///
/// Two shapes rather than six planes for everything, because the sun's views
/// are not a frustum test. See `casts_into` in `cull.comp`.
#[derive(Clone, Copy, Debug)]
pub enum CasterVolume {
    /// A face of a punctual light: a closed frustum with its apex at the light,
    /// so anything that can occlude light travelling into it is inside it.
    Frustum(Mat4),
    /// A sun view, which keeps anything that reaches its box when swept toward
    /// the light — so it has no near plane, and an object between the sun and
    /// the box is exactly what casts into it.
    Sweep {
        light_view: Mat4,
        half_extent: f32,
        depth_range: f32,
    },
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
    /// What the view is drawn with and sampled with.
    pub view_proj: Mat4,
    pub tile: AtlasTile,
    pub volume: CasterVolume,
    /// World size of one texel where the view's clip-space `w` is one: the
    /// whole of an orthographic sun view, and a metre along the axis of a
    /// perspective face. The filter multiplies it by a lookup's own `w`, which
    /// is what makes one formula serve both.
    pub texel_scale: f32,
    pub kind: ViewKind,
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

/// What one frame's shadows resolved to.
///
/// No views means nothing casts, which is what makes the graph drop the shadow
/// pass and its atlas entirely rather than keep an image for no reader.
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

/// What laying a frame out reuses from the frame before, so that it allocates
/// nothing once the lists have grown to the scene.
#[derive(Debug)]
struct Scratch {
    tiles: TileAllocator,
    ranked: Vec<(LightKind, usize, f32)>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            tiles: TileAllocator::new(MIN_TILE),
            ranked: Vec::new(),
        }
    }
}

/// Lay a frame's views out in the atlas: the sun's `set` first, then the
/// punctual lights worth a tile, nearest the camera first.
///
/// The sun goes first because it lights every pixel a lamp does not, so a full
/// atlas costs a lamp its shadow before it costs the sun a cascade.
pub fn fit(
    sun: Option<(&CascadeSet, &CascadeConfig)>,
    lighting: &SceneLighting,
    camera: glam::Vec3,
    config: &AtlasConfig,
) -> ShadowViews {
    let mut out = ShadowViews::default();
    fit_into(
        &mut Scratch::default(),
        &mut out,
        sun,
        lighting,
        camera,
        config,
    );
    out
}

fn fit_into(
    scratch: &mut Scratch,
    out: &mut ShadowViews,
    sun: Option<(&CascadeSet, &CascadeConfig)>,
    lighting: &SceneLighting,
    camera: glam::Vec3,
    config: &AtlasConfig,
) {
    let Scratch { tiles, ranked } = scratch;
    tiles.reset(config.resolution);
    out.resolution = tiles.resolution();
    out.views.clear();
    out.casters.clear();
    out.sun = 0;
    out.splits = [0.0; MAX_CASCADES];

    if let Some((set, cascade_config)) = sun {
        for (index, cascade) in set.cascades[..set.count].iter().enumerate() {
            // A cascade is fitted to its tile's texel count, so a tile of any
            // other size would shimmer. One that does not fit ends the sun's
            // views there, and the views before it answer out to their own
            // splits.
            let size = cascade_config.resolution_of(index);
            let Some(tile) = tiles.allocate(size).filter(|tile| tile.size == size) else {
                break;
            };
            out.views.push(ShadowView {
                view_proj: cascade.view_proj,
                tile,
                volume: CasterVolume::Sweep {
                    light_view: cascade.light_view,
                    half_extent: cascade.half_extent,
                    depth_range: cascade.depth_range,
                },
                texel_scale: cascade.texel_world_size,
                kind: ViewKind::Sun,
            });
            out.splits[index] = set.splits[index];
        }
        out.sun = out.views.len();
        let last = out.splits[out.sun.saturating_sub(1)];
        for split in &mut out.splits[out.sun..] {
            *split = last;
        }
    }

    if config.punctual {
        punctual::rank(lighting, camera, ranked);
        for &(kind, light, _) in ranked.iter() {
            if out.casters.len() >= MAX_SHADOW_LIGHTS {
                break;
            }
            let needed = punctual::face_count(kind);
            let first_view = out.views.len();
            let mut placed = [AtlasTile::default(); 6];
            let mut count = 0;
            while count < needed {
                match tiles.allocate(config.tile_size) {
                    Some(tile) => {
                        placed[count] = tile;
                        count += 1;
                    }
                    None => break,
                }
            }
            // Skipped rather than stopped: a spot still fits in the one tile a
            // point light could not use, and leaving it empty helps nobody.
            if count < needed {
                for tile in &placed[..count] {
                    tiles.release(*tile);
                }
                continue;
            }

            let faces = punctual::faces(lighting, kind, light, config.near, placed[0].size);
            for (face, tile) in placed[..needed].iter().enumerate() {
                out.views.push(ShadowView {
                    view_proj: faces.view_proj[face],
                    tile: *tile,
                    volume: CasterVolume::Frustum(faces.view_proj[face]),
                    texel_scale: faces.texel_per_metre,
                    kind: ViewKind::Punctual,
                });
            }
            out.casters.push(ShadowCaster {
                kind,
                light,
                first_view,
                view_count: needed,
                near: faces.near,
            });
        }
    }

    if out.views.is_empty() {
        out.resolution = 0;
    }
}

/// What a frame's shadows are planned from.
pub struct ShadowInputs<'a> {
    pub camera: &'a Camera,
    pub aspect: f32,
    pub lighting: &'a SceneLighting,
    /// The sun's cascades, or `None` when it casts no shadow.
    pub sun: Option<&'a CascadeConfig>,
    pub atlas: &'a AtlasConfig,
}

/// Plans each frame's shadow views, keeping the lists it works in so that a
/// frame allocates nothing for them.
#[derive(Default)]
pub struct ShadowPlanner {
    scratch: Scratch,
}

impl ShadowPlanner {
    /// Plan a frame into `views`, which the caller keeps between frames for the
    /// reason this keeps its own lists. `None` is a frame in which nothing
    /// casts.
    pub fn plan(&mut self, inputs: Option<&ShadowInputs<'_>>, views: &mut ShadowViews) {
        let Some(inputs) = inputs else {
            views.views.clear();
            views.casters.clear();
            views.sun = 0;
            views.resolution = 0;
            return;
        };
        let set = inputs.sun.map(|config| {
            cascades(
                inputs.camera,
                inputs.aspect,
                inputs.lighting.sun.direction,
                config,
            )
        });
        fit_into(
            &mut self.scratch,
            views,
            set.as_ref().zip(inputs.sun),
            inputs.lighting,
            inputs.camera.position,
            inputs.atlas,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::{MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS, PointLight, SpotLight};
    use glam::Vec3;

    fn point(position: Vec3) -> PointLight {
        PointLight {
            position,
            color: Vec3::ONE,
            candela: 1.0,
            range: 10.0,
            casts_shadows: true,
        }
    }

    fn spot(position: Vec3) -> SpotLight {
        SpotLight {
            position,
            direction: Vec3::NEG_Y,
            color: Vec3::ONE,
            candela: 1.0,
            range: 10.0,
            inner_cos: 0.9,
            outer_cos: 0.7,
            casts_shadows: true,
        }
    }

    fn lighting(points: Vec<PointLight>, spots: Vec<SpotLight>) -> SceneLighting {
        SceneLighting {
            point_lights: points,
            spot_lights: spots,
            ..SceneLighting::default()
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
        }
    }

    fn sun_set() -> CascadeSet {
        cascades(
            &Camera::default(),
            16.0 / 9.0,
            Vec3::new(-0.4, -1.0, -0.3),
            &cascade_config(),
        )
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
        let points: Vec<_> = (0..MAX_POINT_LIGHTS)
            .map(|i| point(Vec3::X * i as f32))
            .collect();
        let spots: Vec<_> = (0..MAX_SPOT_LIGHTS)
            .map(|i| spot(Vec3::Y * i as f32))
            .collect();
        let set = sun_set();
        let config = cascade_config();
        let views = fit(
            Some((&set, &config)),
            &lighting(points, spots),
            Vec3::ZERO,
            &AtlasConfig::default(),
        );

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
        let set = sun_set();
        let config = cascade_config();
        let views = fit(
            Some((&set, &config)),
            &lighting(vec![point(Vec3::ZERO)], vec![spot(Vec3::X)]),
            Vec3::ZERO,
            &AtlasConfig::default(),
        );

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
    /// shadow nobody gets for no saving — and the tiles it took before it ran
    /// out have to come back, or the spot finds none either.
    #[test]
    fn a_light_too_big_for_the_gap_is_skipped_not_final() {
        // Four tiles. The point light needs six.
        let config = AtlasConfig {
            resolution: 1024,
            tile_size: 512,
            ..AtlasConfig::default()
        };
        let mut scene = lighting(vec![point(Vec3::ZERO)], vec![spot(Vec3::ZERO)]);
        scene.point_lights[0].candela = 100.0;
        let views = fit(None, &scene, Vec3::ZERO, &config);

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
        let mut scene = lighting(vec![point(Vec3::ZERO)], Vec::new());
        scene.point_lights[0].casts_shadows = false;
        let views = fit(None, &scene, Vec3::ZERO, &AtlasConfig::default());
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
}
