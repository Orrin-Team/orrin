//! Point and spot lights as shadow casters: which of them are worth a place in
//! the atlas, and what matrix each of their faces is drawn with.
//!
//! A point light is six frustums and a spot light is one, so the unit here is
//! the *face*. Where a face lands in the atlas is [`shadow_views`]' business,
//! beside the sun's views, because they share one image and one pass.
//!
//! Nothing here takes a `Device`, for the reason the render graph does not: the
//! matrices are the part that can be wrong in a way pixels only hint at, so
//! they are the part CI can assert.
//!
//! [`shadow_views`]: super::shadow_views

use glam::{Mat4, Vec3};

use super::{MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS, SceneLighting};

/// How many lights may cast at once, whatever the atlas has room for.
///
/// A second budget above the tile count because the two limit different things:
/// tiles bound the memory, this bounds the *draws*, and a scene of small lights
/// in a large atlas would otherwise re-record the caster list forty-eight times.
pub const MAX_SHADOW_LIGHTS: usize = 8;

/// The most faces those lights can ask for, which is all of them being points.
pub const MAX_SHADOW_FACES: usize = 6 * MAX_SHADOW_LIGHTS;

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
/// filter taps of a fragment near a cube seam have nothing of their own to read
/// and get clamped back onto the last real texel. Widening the frustum by the
/// kernel's reach puts real geometry under those taps instead: four texels, the
/// widest the tent gets (`SHADOW_KERNEL_MAX` in `shadow_filter.glsl`). It costs
/// that fraction of the tile's resolution, which at 512 is a percent and a
/// half.
const FACE_BORDER_TEXELS: f32 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightKind {
    Point,
    Spot,
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
pub fn face_count(kind: LightKind) -> usize {
    match kind {
        LightKind::Point => 6,
        LightKind::Spot => 1,
    }
}

/// Vulkan's clip space, matching [`Camera::projection_range`]'s Y flip but not
/// its depth: a shadow map stays forward-Z, for the reason [`crate::gfx::depth`]
/// gives.
///
/// The Y flip is not cosmetic: it is what makes a triangle's winding come out
/// the same here as in the forward pass, and therefore what lets the shadow
/// pipeline cull back faces rather than silently culling front ones.
fn perspective(fov_y: f32, near: f32, far: f32) -> Mat4 {
    let mut proj = Mat4::perspective_rh(fov_y, 1.0, near, far);
    proj.y_axis.y *= -1.0;
    proj
}

/// The tangent of half the field of view one tile is rendered at: the face's
/// own, widened by [`FACE_BORDER_TEXELS`] so a filter tap at the tile's edge has
/// real depth under it.
fn bordered_extent(half_angle: f32, tile_size: u32) -> f32 {
    let border = 2.0 * FACE_BORDER_TEXELS / tile_size.max(1) as f32;
    half_angle.tan() * (1.0 + border)
}

/// Build one cube face's matrix about `position`.
fn point_face(position: Vec3, face: usize, near: f32, far: f32, extent: f32) -> Mat4 {
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
    perspective(2.0 * extent.atan(), near, far) * view
}

/// The lights that ask for shadows, most worth a tile first.
///
/// `camera` is the eye position, which is the only thing the ranking needs —
/// the frustum is deliberately not consulted, because a light behind the camera
/// still casts shadows into what is in front of it.
///
/// Into `candidates`, which a caller keeps between frames so that ranking
/// allocates nothing; the third field is the score it was sorted by.
pub fn rank(lighting: &SceneLighting, camera: Vec3, candidates: &mut Vec<(LightKind, usize, f32)>) {
    // One list so the two kinds compete for the same tiles: a spot nobody can
    // see should lose to a point light overhead, and ranking them separately
    // could not express that.
    candidates.clear();
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
}

/// One light's faces: the matrices they are drawn and sampled with.
#[derive(Clone, Copy, Debug)]
pub struct Faces {
    pub view_proj: [Mat4; 6],
    /// Six for a point, one for a spot.
    pub count: usize,
    /// What the faces' near plane was.
    pub near: f32,
    /// World size of one texel of a `tile_size` tile, a metre along the face's
    /// axis from the light. A perspective texel grows with that depth, so this
    /// is what the filter multiplies by it.
    pub texel_per_metre: f32,
}

/// Build the faces of the light of `kind` at `index`, for tiles `tile_size`
/// texels across. `near` is the near plane asked for, in metres; it cannot be
/// derived from the light's range the way the far plane is, because precision
/// near the light is what a shadow's contact looks like.
pub fn faces(
    lighting: &SceneLighting,
    kind: LightKind,
    index: usize,
    near: f32,
    tile_size: u32,
) -> Faces {
    let mut view_proj = [Mat4::IDENTITY; 6];
    let (extent, near) = match kind {
        LightKind::Point => {
            let light = &lighting.point_lights[index];
            let near = near.clamp(1e-3, light.range * 0.5);
            let extent = bordered_extent(std::f32::consts::FRAC_PI_4, tile_size);
            for (face, slot) in view_proj.iter_mut().enumerate() {
                *slot = point_face(light.position, face, near, light.range, extent);
            }
            (extent, near)
        }
        LightKind::Spot => {
            let light = &lighting.spot_lights[index];
            let near = near.clamp(1e-3, light.range * 0.5);
            // Back from the cosine the shader compares against, so the
            // frustum and the falloff cannot describe different cones.
            let half_angle = light.outer_cos.clamp(-1.0, 1.0).acos();
            let extent = bordered_extent(half_angle, tile_size);
            let up = if light.direction.y.abs() > 0.99 {
                Vec3::Z
            } else {
                Vec3::Y
            };
            let view = Mat4::look_at_rh(light.position, light.position + light.direction, up);
            view_proj[0] = perspective(2.0 * extent.atan(), near, light.range) * view;
            (extent, near)
        }
    };
    Faces {
        view_proj,
        count: face_count(kind),
        near,
        texel_per_metre: 2.0 * extent / tile_size.max(1) as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::{PointLight, SpotLight};

    fn point(position: Vec3, candela: f32, casts: bool) -> PointLight {
        PointLight {
            id: 0,
            position,
            color: Vec3::ONE,
            candela,
            range: 10.0,
            casts_shadows: casts,
        }
    }

    fn spot(position: Vec3, candela: f32, casts: bool) -> SpotLight {
        SpotLight {
            id: 0,
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

    fn ranked(lighting: &SceneLighting) -> Vec<(LightKind, usize)> {
        let mut candidates = Vec::new();
        rank(lighting, Vec3::ZERO, &mut candidates);
        candidates
            .into_iter()
            .map(|(kind, index, _)| (kind, index))
            .collect()
    }

    /// A light that says it does not cast never reaches the allocator, whatever
    /// its importance — that is the whole point of the switch.
    #[test]
    fn a_light_that_opts_out_is_not_ranked() {
        let ranked = ranked(&lighting(
            vec![point(Vec3::ZERO, 100.0, false), point(Vec3::X, 1.0, true)],
            Vec::new(),
        ));
        assert_eq!(ranked, vec![(LightKind::Point, 1)]);
    }

    /// When more lights ask than fit, the ones nearest the camera win. The
    /// alternative — first come, first served — makes which light casts depend
    /// on entity spawn order, which is not a thing a viewer can see or an author
    /// can control.
    #[test]
    fn the_nearest_lights_rank_first() {
        let spots = vec![
            spot(Vec3::X * 100.0, 1.0, true),
            spot(Vec3::X * 1.0, 1.0, true),
            spot(Vec3::X * 50.0, 1.0, true),
            spot(Vec3::X * 2.0, 1.0, true),
            spot(Vec3::X * 200.0, 1.0, true),
        ];
        let order: Vec<usize> = ranked(&lighting(Vec::new(), spots))
            .into_iter()
            .map(|(_, index)| index)
            .collect();
        assert_eq!(order, vec![1, 3, 2, 0, 4]);
    }

    fn cube(position: Vec3) -> Faces {
        faces(
            &lighting(vec![point(position, 1.0, true)], Vec::new()),
            LightKind::Point,
            0,
            0.05,
            512,
        )
    }

    /// Every face's matrix has to put its own light's surroundings on screen.
    /// Six faces at 90° cover the sphere, so a point one unit away along each
    /// axis must land inside the frustum that faces it — and this is what
    /// catches a Y flip or a handedness applied in the wrong place.
    #[test]
    fn each_cube_face_projects_what_it_faces() {
        let faces = cube(Vec3::ZERO);
        for (face, direction) in FACE_DIRECTIONS.iter().enumerate() {
            let target = *direction * 2.0;
            let clip = faces.view_proj[face] * target.extend(1.0);
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
        let faces = cube(Vec3::ZERO);
        for (face, direction) in FACE_DIRECTIONS.iter().enumerate() {
            let target = *direction * 2.0;
            let inside: Vec<usize> = (0..6)
                .filter(|&other| {
                    let clip = faces.view_proj[other] * target.extend(1.0);
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

    /// The filter sizes its kernel by this number times a fragment's depth in
    /// the face, so it has to be what a texel really spans there: a metre
    /// along the axis, one texel across the tile's middle.
    #[test]
    fn a_face_texel_is_as_wide_as_the_matrix_makes_it() {
        let faces = cube(Vec3::ZERO);
        let at = |x: f32| {
            let clip = faces.view_proj[0] * Vec3::new(1.0, 0.0, x).extend(1.0);
            (clip.x / clip.w) * 0.5 * 512.0
        };
        let texels = (at(faces.texel_per_metre) - at(0.0)).abs();
        assert!((texels - 1.0).abs() < 1e-3, "{texels} texels");
    }
}
