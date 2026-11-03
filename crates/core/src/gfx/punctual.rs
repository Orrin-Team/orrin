//! The punctual shadow atlas: which lights get tiles, where those tiles are,
//! and what matrix each one is drawn with.
//!
//! A point light is six frustums and a spot light is one, so the natural shape
//! is not one image per light but one image with a tile per *face*. That is what
//! makes this affordable in a graph whose resources are unversioned: every face
//! is a viewport into the same attachment, so the whole thing is a single pass
//! with a single barrier however many lights cast. Six passes per light over a
//! shared array image would be serialised by write-after-write, which is exactly
//! what the cascades already pay for four.
//!
//! Nothing here takes a `Device`, for the reason the render graph does not: the
//! allocation and the matrices are the part that can be wrong in a way pixels
//! only hint at, so they are the part CI can assert.

use glam::{Mat4, Vec3};

use super::{MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS, SceneLighting};

/// How many lights may cast at once, whatever the atlas has room for.
///
/// A second budget above the tile count because the two limit different things:
/// tiles bound the memory, this bounds the *draws*, and a scene of small lights
/// in a large atlas would otherwise re-record the caster list forty-eight times.
pub const MAX_SHADOW_LIGHTS: usize = 8;

/// The most faces those lights can ask for, which is all of them being points.
pub const MAX_ATLAS_FACES: usize = 6 * MAX_SHADOW_LIGHTS;

/// The cube's faces, in the order the shader's major-axis test returns.
///
/// The order is a contract with `forward.frag`: it picks a face from the
/// dominant axis of light-to-fragment and indexes this table, so a permutation
/// here lights the wrong tile. The matrices themselves carry every other
/// convention — the shader projects with the same one the face was rendered
/// with, so nothing else has to agree.
const FACE_DIRECTIONS: [Vec3; 6] = [
    Vec3::X,
    Vec3::NEG_X,
    Vec3::Y,
    Vec3::NEG_Y,
    Vec3::Z,
    Vec3::NEG_Z,
];

/// How much wider than its 90° share each cube face is rendered, in texels of
/// its own tile.
///
/// A face rendered at exactly 90° fills its tile edge to edge, so the outermost
/// percentage-closer tap of a fragment near a cube seam has nothing of its own
/// to read and gets clamped back onto the last real texel. Widening the frustum
/// by a texel and a half puts real geometry under those taps instead. It costs
/// that fraction of the tile's resolution, which at 512 is under a percent.
const FACE_BORDER_TEXELS: f32 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    Point,
    Spot,
}

/// Where one face lives in the atlas, in texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AtlasTile {
    pub offset: [u32; 2],
    pub size: u32,
}

impl AtlasTile {
    /// The tile as a UV rectangle of the whole atlas: `xy` offset, `zw` scale.
    ///
    /// What the shader multiplies a face's NDC into, and the reason a face needs
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

/// One rendered frustum: the matrix it is drawn and sampled with, and the tile
/// it lands in.
#[derive(Clone, Copy, Debug)]
pub struct ShadowFace {
    pub view_proj: Mat4,
    pub tile: AtlasTile,
}

/// A light the atlas found room for.
#[derive(Clone, Copy, Debug)]
pub struct ShadowCaster {
    pub kind: LightKind,
    /// Index into [`SceneLighting::point_lights`] or `spot_lights`, depending on
    /// `kind`.
    pub light: usize,
    /// This caster's faces, contiguous in [`ShadowAtlas::faces`]: six for a
    /// point, one for a spot.
    pub first_face: usize,
    pub face_count: usize,
    /// The sphere the caster cull tests against — the light's reach, since
    /// nothing outside it can shadow anything the light touches. Per light
    /// rather than per face: culling six frustums separately would buy a
    /// fraction of the draws back at six times the test, and a point light's
    /// reach is small enough that the whole set usually goes into every face.
    pub center: Vec3,
    pub radius: f32,
    /// What the faces' near plane was, which the shader needs to turn a depth
    /// comparison back into a distance for its bias.
    pub near: f32,
}

/// How the atlas is cut up and what its frustums are clipped to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasConfig {
    /// Edge of the whole atlas in texels. Frame *structure*: changing it
    /// recompiles the graph and reallocates the image.
    pub resolution: u32,
    /// Edge of one face's tile. The number of tiles follows from the two, so a
    /// light's share of the atlas is a division rather than a packing problem.
    pub tile_size: u32,
    /// Near plane of every punctual frustum, in metres. It cannot be derived
    /// from the light's range the way the far plane is: precision near the light
    /// is what a shadow's contact looks like, and a light with a hundred-metre
    /// reach still stands centimetres from what it lights.
    pub near: f32,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            resolution: 4096,
            tile_size: 512,
            near: 0.05,
        }
    }
}

impl AtlasConfig {
    /// Tiles across, and therefore `columns * columns` in total.
    pub fn columns(&self) -> u32 {
        (self.resolution / self.tile_size.max(1)).max(1)
    }

    pub fn capacity(&self) -> usize {
        let columns = self.columns() as usize;
        columns * columns
    }
}

/// What one frame's punctual shadows resolved to.
///
/// A `faces` of zero means nothing casts, which is what makes the graph drop the
/// atlas pass entirely rather than clear a 32 MB image for no reader — the same
/// way a cascade count of zero drops the cascade passes.
#[derive(Clone, Debug, Default)]
pub struct ShadowAtlas {
    pub resolution: u32,
    pub tile_size: u32,
    pub faces: Vec<ShadowFace>,
    pub casters: Vec<ShadowCaster>,
}

impl ShadowAtlas {
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

/// How much a light is worth a tile, given where the camera is.
///
/// Distance to the light's *reach* rather than to the light itself, so standing
/// inside a large dim light scores above squinting at a bright one across the
/// level — which is what a viewer would say about whose shadows they can see.
///
/// Candela rather than the authored lumens, so the comparison is between what the
/// two lights actually put out in a direction: a spot's reflector concentrates
/// its power, and ranking a 3 000 lumen beam against a 3 000 lumen bulb by their
/// labels would rate them equal when one is an order of magnitude brighter.
fn importance(position: Vec3, range: f32, candela: f32, camera: Vec3) -> f32 {
    let to_surface = (position.distance(camera) - range).max(0.0);
    candela.max(0.0) / (1.0 + to_surface * to_surface)
}

/// Which faces a kind of light needs.
fn face_count(kind: LightKind) -> usize {
    match kind {
        LightKind::Point => 6,
        LightKind::Spot => 1,
    }
}

/// Vulkan's clip space, matching [`Camera::projection_range`] exactly.
///
/// The Y flip is not cosmetic: it is what makes a triangle's winding come out
/// the same here as in the forward pass, and therefore what lets the shadow
/// pipeline cull back faces rather than silently culling front ones.
fn perspective(fov_y: f32, near: f32, far: f32) -> Mat4 {
    let mut proj = Mat4::perspective_rh(fov_y, 1.0, near, far);
    proj.y_axis.y *= -1.0;
    proj
}

/// The field of view one tile is rendered at, widened by
/// [`FACE_BORDER_TEXELS`] so a filter tap at the tile's edge has real depth
/// under it.
fn bordered_fov(half_angle: f32, tile_size: u32) -> f32 {
    let extent = half_angle.tan();
    let border = 2.0 * FACE_BORDER_TEXELS / tile_size.max(1) as f32;
    2.0 * (extent * (1.0 + border)).atan()
}

/// Build one cube face's matrix about `position`.
fn point_face(position: Vec3, face: usize, near: f32, far: f32, tile_size: u32) -> Mat4 {
    let forward = FACE_DIRECTIONS[face];
    // Any up that is not the forward will do — the shader projects with this
    // very matrix, so the choice is invisible to it. It only has to be
    // non-degenerate, which the poles are the sole exception to.
    let up = if forward.y.abs() > 0.5 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(position, position + forward, up);
    perspective(
        bordered_fov(std::f32::consts::FRAC_PI_4, tile_size),
        near,
        far,
    ) * view
}

/// Assign tiles to the lights worth them, and build the matrices to draw them.
///
/// `camera` is the eye position, which is the only thing the importance sort
/// needs — the frustum is deliberately not consulted, because a light behind the
/// camera still casts shadows into what is in front of it.
pub fn fit(lighting: &SceneLighting, camera: Vec3, config: &AtlasConfig) -> ShadowAtlas {
    let mut atlas = ShadowAtlas {
        resolution: config.resolution,
        tile_size: config.tile_size,
        faces: Vec::new(),
        casters: Vec::new(),
    };

    // (kind, index, importance). Built as one list so the two kinds compete for
    // the same tiles: a spot nobody can see should lose to a point light
    // overhead, and ranking them separately could not express that.
    let mut candidates: Vec<(LightKind, usize, f32)> = Vec::new();
    for (index, light) in lighting
        .point_lights
        .iter()
        .take(MAX_POINT_LIGHTS)
        .enumerate()
    {
        if light.casts_shadows && light.candela > 0.0 && light.range > 0.0 {
            candidates.push((
                LightKind::Point,
                index,
                importance(light.position, light.range, light.candela, camera),
            ));
        }
    }
    for (index, light) in lighting
        .spot_lights
        .iter()
        .take(MAX_SPOT_LIGHTS)
        .enumerate()
    {
        if light.casts_shadows && light.candela > 0.0 && light.range > 0.0 {
            candidates.push((
                LightKind::Spot,
                index,
                importance(light.position, light.range, light.candela, camera),
            ));
        }
    }

    // Descending, and stable so equal scores keep their extraction order rather
    // than swapping between frames — a light trading tiles with its twin every
    // frame is a flicker nobody would attribute to a sort.
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2));

    let columns = config.columns();
    let capacity = config.capacity();
    let mut next_tile = 0usize;

    for (kind, index, _) in candidates {
        if atlas.casters.len() >= MAX_SHADOW_LIGHTS {
            break;
        }
        let needed = face_count(kind);
        // Skipped rather than stopped: a spot still fits in the one tile a point
        // light could not use, and leaving it empty helps nobody.
        if next_tile + needed > capacity {
            continue;
        }

        let first_face = atlas.faces.len();
        let (center, radius, near) = match kind {
            LightKind::Point => {
                let light = &lighting.point_lights[index];
                let near = config.near.clamp(1e-3, light.range * 0.5);
                for face in 0..6 {
                    atlas.faces.push(ShadowFace {
                        view_proj: point_face(
                            light.position,
                            face,
                            near,
                            light.range,
                            config.tile_size,
                        ),
                        tile: tile_at(next_tile + face, columns, config.tile_size),
                    });
                }
                (light.position, light.range, near)
            }
            LightKind::Spot => {
                let light = &lighting.spot_lights[index];
                let near = config.near.clamp(1e-3, light.range * 0.5);
                // Back from the cosine the shader compares against, so the
                // frustum and the falloff cannot describe different cones.
                let half_angle = light.outer_cos.clamp(-1.0, 1.0).acos();
                let up = if light.direction.y.abs() > 0.99 {
                    Vec3::Z
                } else {
                    Vec3::Y
                };
                let view = Mat4::look_at_rh(light.position, light.position + light.direction, up);
                atlas.faces.push(ShadowFace {
                    view_proj: perspective(
                        bordered_fov(half_angle, config.tile_size),
                        near,
                        light.range,
                    ) * view,
                    tile: tile_at(next_tile, columns, config.tile_size),
                });
                (light.position, light.range, near)
            }
        };

        atlas.casters.push(ShadowCaster {
            kind,
            light: index,
            first_face,
            face_count: needed,
            center,
            radius,
            near,
        });
        next_tile += needed;
    }

    atlas
}

/// Where tile `index` sits, filling the atlas row by row.
fn tile_at(index: usize, columns: u32, tile_size: u32) -> AtlasTile {
    let columns = columns.max(1) as usize;
    AtlasTile {
        offset: [
            ((index % columns) as u32) * tile_size,
            ((index / columns) as u32) * tile_size,
        ],
        size: tile_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::{PointLight, SpotLight};

    fn point(position: Vec3, candela: f32, casts: bool) -> PointLight {
        PointLight {
            position,
            color: Vec3::ONE,
            candela,
            range: 10.0,
            casts_shadows: casts,
        }
    }

    fn spot(position: Vec3, candela: f32, casts: bool) -> SpotLight {
        SpotLight {
            position,
            direction: Vec3::NEG_Y,
            color: Vec3::ONE,
            candela,
            range: 10.0,
            inner_cos: 0.9,
            outer_cos: 0.7,
            casts_shadows: casts,
        }
    }

    fn lighting(points: Vec<PointLight>, spots: Vec<SpotLight>) -> SceneLighting {
        SceneLighting {
            point_lights: points,
            spot_lights: spots,
            ..SceneLighting::default()
        }
    }

    /// The shape of the thing: six tiles for a point, one for a spot, and every
    /// face contiguous so a caster's range in `faces` is a slice.
    #[test]
    fn a_point_takes_six_tiles_and_a_spot_takes_one() {
        let atlas = fit(
            &lighting(
                vec![point(Vec3::ZERO, 1.0, true)],
                vec![spot(Vec3::X, 1.0, true)],
            ),
            Vec3::ZERO,
            &AtlasConfig::default(),
        );

        assert_eq!(atlas.faces.len(), 7);
        assert_eq!(atlas.casters.len(), 2);
        let point = atlas.caster(LightKind::Point, 0).unwrap();
        let spot = atlas.caster(LightKind::Spot, 0).unwrap();
        assert_eq!((point.first_face, point.face_count), (0, 6));
        assert_eq!((spot.first_face, spot.face_count), (6, 1));
    }

    /// No two faces may land on the same texels. This is the one mistake the
    /// allocator can make that renders perfectly and shadows wrongly: the tiles
    /// draw fine, and one light reads another's depth.
    #[test]
    fn no_two_faces_share_a_tile() {
        let points: Vec<_> = (0..MAX_POINT_LIGHTS)
            .map(|i| point(Vec3::X * i as f32, 1.0, true))
            .collect();
        let spots: Vec<_> = (0..MAX_SPOT_LIGHTS)
            .map(|i| spot(Vec3::Y * i as f32, 1.0, true))
            .collect();
        let atlas = fit(
            &lighting(points, spots),
            Vec3::ZERO,
            &AtlasConfig::default(),
        );

        let mut seen = std::collections::HashSet::new();
        for face in &atlas.faces {
            assert!(
                seen.insert(face.tile.offset),
                "two faces both claim {:?}",
                face.tile.offset
            );
        }
    }

    /// And every tile has to be inside the image. An offset past the edge is a
    /// viewport the driver will happily accept and clip to nothing.
    #[test]
    fn every_tile_lands_inside_the_atlas() {
        let points: Vec<_> = (0..MAX_POINT_LIGHTS)
            .map(|i| point(Vec3::X * i as f32, 1.0, true))
            .collect();
        let config = AtlasConfig {
            resolution: 2048,
            tile_size: 512,
            ..AtlasConfig::default()
        };
        let atlas = fit(&lighting(points, Vec::new()), Vec3::ZERO, &config);

        for face in &atlas.faces {
            assert!(face.tile.offset[0] + face.tile.size <= config.resolution);
            assert!(face.tile.offset[1] + face.tile.size <= config.resolution);
        }
    }

    /// A light that says it does not cast never reaches the allocator, whatever
    /// its importance — that is the whole point of the switch.
    #[test]
    fn a_light_that_opts_out_gets_nothing() {
        let atlas = fit(
            &lighting(
                vec![point(Vec3::ZERO, 100.0, false), point(Vec3::X, 1.0, true)],
                Vec::new(),
            ),
            Vec3::ZERO,
            &AtlasConfig::default(),
        );

        assert_eq!(atlas.casters.len(), 1);
        assert!(atlas.caster(LightKind::Point, 0).is_none());
        assert!(atlas.caster(LightKind::Point, 1).is_some());
    }

    /// When more lights ask than fit, the ones nearest the camera win. The
    /// alternative — first come, first served — makes which light casts depend
    /// on entity spawn order, which is not a thing a viewer can see or an author
    /// can control.
    #[test]
    fn the_budget_goes_to_the_nearest_lights() {
        // Four tiles: room for exactly four spots and no point light at all.
        let config = AtlasConfig {
            resolution: 1024,
            tile_size: 512,
            ..AtlasConfig::default()
        };
        let spots = vec![
            spot(Vec3::X * 100.0, 1.0, true),
            spot(Vec3::X * 1.0, 1.0, true),
            spot(Vec3::X * 50.0, 1.0, true),
            spot(Vec3::X * 2.0, 1.0, true),
            spot(Vec3::X * 200.0, 1.0, true),
        ];
        let atlas = fit(&lighting(Vec::new(), spots), Vec3::ZERO, &config);

        assert_eq!(atlas.casters.len(), 4);
        let chosen: Vec<usize> = atlas.casters.iter().map(|c| c.light).collect();
        assert!(chosen.contains(&1) && chosen.contains(&3));
        assert!(!chosen.contains(&4), "the farthest light took a tile");
    }

    /// A point light that does not fit must not stop a spot that does. Leaving
    /// the last tile empty because the light in front of it wanted six is a
    /// shadow nobody gets for no saving.
    #[test]
    fn a_light_too_big_for_the_gap_is_skipped_not_final() {
        // Four tiles. The point light (brightest, so sorted first) needs six.
        let config = AtlasConfig {
            resolution: 1024,
            tile_size: 512,
            ..AtlasConfig::default()
        };
        let atlas = fit(
            &lighting(
                vec![point(Vec3::ZERO, 100.0, true)],
                vec![spot(Vec3::ZERO, 1.0, true)],
            ),
            Vec3::ZERO,
            &config,
        );

        assert!(atlas.caster(LightKind::Point, 0).is_none());
        assert!(
            atlas.caster(LightKind::Spot, 0).is_some(),
            "the spot lost its tile to a light that never took one"
        );
    }

    /// Nothing casting is the shape the graph reads as "declare no atlas at
    /// all", so it has to be reachable rather than approximated by an atlas of
    /// empty tiles.
    #[test]
    fn a_scene_with_no_casters_produces_no_faces() {
        let atlas = fit(
            &lighting(vec![point(Vec3::ZERO, 1.0, false)], Vec::new()),
            Vec3::ZERO,
            &AtlasConfig::default(),
        );
        assert!(atlas.faces.is_empty());
        assert!(atlas.casters.is_empty());
    }

    /// Every face's matrix has to put its own light's surroundings on screen.
    /// Six faces at 90° cover the sphere, so a point one unit away along each
    /// axis must land inside the frustum that faces it — and this is what
    /// catches a Y flip or a handedness applied in the wrong place.
    #[test]
    fn each_cube_face_projects_what_it_faces() {
        let atlas = fit(
            &lighting(vec![point(Vec3::ZERO, 1.0, true)], Vec::new()),
            Vec3::ZERO,
            &AtlasConfig::default(),
        );

        for (face, direction) in FACE_DIRECTIONS.iter().enumerate() {
            let target = *direction * 2.0;
            let clip = atlas.faces[face].view_proj * target.extend(1.0);
            assert!(clip.w > 0.0, "face {face} put its own axis behind it");
            let ndc = clip.truncate() / clip.w;
            assert!(
                ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0,
                "face {face} projected its own axis off-tile: {ndc:?}",
            );
            assert!(
                (0.0..=1.0).contains(&ndc.z),
                "face {face} projected its own axis outside the depth range: {ndc:?}",
            );
        }
    }

    /// And no face may claim what another one owns, or the shader's major-axis
    /// pick would be a coin toss between two tiles holding different depths.
    #[test]
    fn the_cube_faces_do_not_overlap() {
        let atlas = fit(
            &lighting(vec![point(Vec3::ZERO, 1.0, true)], Vec::new()),
            Vec3::ZERO,
            &AtlasConfig::default(),
        );

        for (face, direction) in FACE_DIRECTIONS.iter().enumerate() {
            let target = *direction * 2.0;
            let inside: Vec<usize> = (0..6)
                .filter(|&other| {
                    let clip = atlas.faces[other].view_proj * target.extend(1.0);
                    if clip.w <= 0.0 {
                        return false;
                    }
                    let ndc = clip.truncate() / clip.w;
                    ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0
                })
                .collect();
            assert_eq!(inside, vec![face], "face {face}'s axis is on {inside:?}");
        }
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
}
