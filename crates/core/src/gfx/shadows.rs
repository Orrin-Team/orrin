use glam::{Mat4, Vec3};

use crate::scene::Camera;

pub const MAX_CASCADES: usize = 4;
pub const OVERLAP: f32 = 0.005;

/// Vulkan NDC: x,y in [-1,1], z in [0,1]. First four are the near plane.
const NDC_CORNERS: [Vec3; 8] = [
    Vec3::new(-1.0, -1.0, 0.0),
    Vec3::new( 1.0, -1.0, 0.0),
    Vec3::new(-1.0,  1.0, 0.0),
    Vec3::new( 1.0,  1.0, 0.0),
    Vec3::new(-1.0, -1.0, 1.0),
    Vec3::new( 1.0, -1.0, 1.0),
    Vec3::new(-1.0,  1.0, 1.0),
    Vec3::new( 1.0,  1.0, 1.0),
];

pub struct CascadeConfig { 
    pub count: usize, 
    pub max_distance: f32, 
    pub lambda: f32, 
    pub resolution: u32,
    pub pullback: f32,
}

#[derive(Clone, Copy, Default)]
pub struct Cascade {
    pub view_proj: Mat4,      // render + sample
    pub light_view: Mat4,     // cull: world -> light space
    pub half_extent: f32,     // the snapped sphere radius, r
    pub depth_range: f32,     // 0 .. 2r + pullback
    pub split_distance: f32,
    pub texel_world_size: f32,
}

#[derive(Clone, Copy)]
pub struct CascadeSet {
    pub cascades: [Cascade; MAX_CASCADES],
    pub splits: [f32; MAX_CASCADES],
    pub count: usize,
}

pub fn split_distances(near: f32, config: &CascadeConfig) -> [f32; MAX_CASCADES] {
    // Every input here reaches a slider, so none of them can be trusted to be
    // sane. A zero `near` sends the logarithmic term to infinity, and a
    // `max_distance` at or below `near` makes the splits non-monotonic, which
    // the shader's first-hit cascade scan reads as a valid partition.
    let count = config.count.clamp(1, MAX_CASCADES);
    let lambda = config.lambda.clamp(0.0, 1.0);
    let near = near.max(1e-3);
    let far = config.max_distance.max(near + 1e-3);

    // Unused slots hold the far distance rather than zero: the shader picks the
    // first split the fragment is nearer than, so a slot that can never win is
    // inert, while a zero would be picked and index a cascade with no matrix.
    let mut splits = [far; MAX_CASCADES];

    for i in 0..count {
        let t = (i + 1) as f32 / count as f32;
        let log_split = near * (far / near).powf(t);
        let uniform_split = near + t * (far - near);
        splits[i] = lambda * log_split + (1.0 - lambda) * uniform_split;
    }

    // Exact, not what `powf` returns at t = 1: the fit uses this as the last
    // cascade's far plane and the shader compares against it, so an ulp of
    // disagreement is an unshadowed ring at the edge of the last cascade.
    splits[count - 1] = far;

    splits
}

pub fn sub_frustum_corners(camera: &Camera, aspect: f32, near: f32, far: f32) -> [Vec3; 8] {
    let inv = (camera.projection_range(aspect, near, far) * camera.view()).inverse();
    NDC_CORNERS.map(|ndc| inv.project_point3(ndc))
}

pub fn fit_cascade(
    corners: &[Vec3; 8],
    light_dir: Vec3,
    split_distance: f32,
    config: &CascadeConfig,
) -> Cascade {
    let center = corners.iter().sum::<Vec3>() / 8.0;
    let mut radius = corners.iter().map(|c| (*c - center).length()).fold(0.0f32, f32::max);
    radius = (radius * 16.0).ceil() / 16.0; // Quantising the radius because of float noise

    // Unit length is load-bearing, not cosmetic: `eye` below is placed by
    // scaling this vector, so a non-unit direction moves the near plane and
    // clips the far corners out of the map with no error anywhere.
    let light_dir = light_dir.normalize();

    let up = if light_dir.dot(Vec3::Y).abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let rot = Mat4::look_at_rh(Vec3::ZERO, light_dir, up); // Rotation only, eye at origin

    let texel = 2.0 * radius / config.resolution as f32;
    let mut c_ls = rot.transform_point3(center);
    c_ls.x = (c_ls.x / texel).floor() * texel;
    c_ls.y = (c_ls.y / texel).floor() * texel;

    let snapped_center = rot.inverse().transform_point3(c_ls);
    let eye = snapped_center - light_dir * (radius + config.pullback);
    let light_view = Mat4::look_at_rh(eye, snapped_center, up);

    let mut ortho = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, 2.0*radius + config.pullback);
    ortho.y_axis.y *= -1.0;

    let view_proj = ortho * light_view;
    let half_extent = radius;
    let depth_range = 2.0*radius + config.pullback;
    
    Cascade { 
        view_proj, 
        light_view, 
        half_extent,
        depth_range,
        split_distance,
        texel_world_size: texel 
    }
}

pub fn cascades(camera: &Camera, aspect: f32, light_dir: Vec3, config: &CascadeConfig) -> CascadeSet {
    let count = config.count.clamp(1, MAX_CASCADES);
    // Unit length is not cosmetic here: `fit_cascade` scales the near-plane
    // pullback by this vector, so an inspector sun of (0,-2,0) would silently
    // double it. A zero sun has no direction to normalize and would reach
    // `look_at_rh` as NaNs, so it falls back to straight down.
    let light_dir = light_dir.try_normalize().unwrap_or(Vec3::NEG_Y);
    let splits = split_distances(camera.near, config);

    let mut cascades = [Cascade::default(); MAX_CASCADES];
    for i in 0..count {
        let near = if i == 0 { camera.near } else { splits[i - 1] * (1.0 - OVERLAP) };
        let corners = sub_frustum_corners(camera, aspect, near, splits[i]);
        cascades[i] = fit_cascade(&corners, light_dir, splits[i], config);
    }

    CascadeSet { cascades, splits, count }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASPECT: f32 = 16.0 / 9.0;

    /// An ordinary afternoon sun: down, and off to one side so nothing lines up
    /// with a world axis by accident.
    ///
    /// Normalized, because `fit_cascade` takes this as a unit vector — it scales
    /// the near-plane pullback by it. Only `cascades` normalizes on the way in,
    /// so a test calling the fit directly has to do it here.
    fn sun() -> Vec3 {
        Vec3::new(-0.4, -1.0, -0.3).normalize()
    }

    fn config() -> CascadeConfig {
        CascadeConfig {
            count: 4,
            max_distance: 100.0,
            lambda: 0.75,
            resolution: 2048,
            pullback: 50.0,
        }
    }

    /// The light-space frame the fit snaps against: rotation only, anchored at
    /// the world origin. Deliberately duplicated rather than shared with
    /// `fit_cascade` — a test that calls the implementation's own helper cannot
    /// notice the implementation changing it.
    fn light_rot(light_dir: Vec3) -> Mat4 {
        let d = light_dir.normalize();
        let up = if d.dot(Vec3::Y).abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        Mat4::look_at_rh(Vec3::ZERO, d, up)
    }

    /// Recover the cascade box's center from the matrices the fit returned.
    /// `light_view` puts the eye at the origin, and the eye sits `r + pullback`
    /// back along the light direction.
    fn snapped_center(cascade: &Cascade, light_dir: Vec3, config: &CascadeConfig) -> Vec3 {
        let eye = cascade.light_view.inverse().transform_point3(Vec3::ZERO);
        eye + light_dir.normalize() * (cascade.half_extent + config.pullback)
    }

    fn fit_slice(camera: &Camera, near: f32, far: f32, config: &CascadeConfig) -> Cascade {
        let corners = sub_frustum_corners(camera, ASPECT, near, far);
        fit_cascade(&corners, sun(), far, config)
    }

    /// True if `point` is inside the Vulkan NDC cube, with `slack` allowed on
    /// x and y.
    fn inside_ndc(point: Vec3, slack: f32) -> bool {
        point.x.abs() <= 1.0 + slack
            && point.y.abs() <= 1.0 + slack
            && point.z >= -1e-4
            && point.z <= 1.0 + 1e-4
    }

    // --- split_distances ---

    #[test]
    fn splits_increase_and_end_exactly_at_max_distance() {
        let config = config();
        let splits = split_distances(0.1, &config);
        for pair in splits[..config.count].windows(2) {
            assert!(pair[1] > pair[0], "splits must increase: {splits:?}");
        }
        assert!(splits[0] > 0.1);
        // Exact, not approximate: the shader compares against this value and the
        // fit uses it as a far plane, so an ulp of drift is a visible seam.
        assert_eq!(splits[config.count - 1], config.max_distance);
    }

    /// lambda = 0 collapses to plain linear interpolation, which is a closed
    /// form the test can write independently. Together with the logarithmic case
    /// this pins the blend direction — a swapped `lambda`/`1 - lambda` produces
    /// splits that still increase and still look plausible in the editor.
    #[test]
    fn lambda_zero_is_the_uniform_partition() {
        let config = CascadeConfig {
            lambda: 0.0,
            ..config()
        };
        let (near, far) = (0.1, config.max_distance);
        let splits = split_distances(near, &config);
        for i in 0..config.count - 1 {
            let t = (i + 1) as f32 / config.count as f32;
            let expected = near + t * (far - near);
            assert!(
                (splits[i] - expected).abs() < 1e-3,
                "cascade {i}: {} != {expected}",
                splits[i],
            );
        }
    }

    #[test]
    fn lambda_one_is_the_logarithmic_partition() {
        let config = CascadeConfig {
            lambda: 1.0,
            ..config()
        };
        let (near, far) = (0.1, config.max_distance);
        let splits = split_distances(near, &config);
        for i in 0..config.count - 1 {
            let t = (i + 1) as f32 / config.count as f32;
            let expected = near * (far / near).powf(t);
            assert!(
                (splits[i] - expected).abs() < 1e-3,
                "cascade {i}: {} != {expected}",
                splits[i],
            );
        }
    }

    /// Slots past `count` must be unreachable by the shader's first-hit scan.
    /// Holding `max_distance` makes them inert; a zero would win every
    /// comparison and index a cascade with no matrix behind it.
    #[test]
    fn unused_slots_hold_the_far_distance() {
        let config = CascadeConfig { count: 2, ..config() };
        let splits = split_distances(0.1, &config);
        for &split in &splits[config.count..] {
            assert_eq!(split, config.max_distance);
        }
    }

    /// Every one of these arrives from an editor slider, so none of them may
    /// panic or produce a non-finite partition.
    #[test]
    fn degenerate_configs_stay_finite() {
        let cases = [
            ("zero cascades", CascadeConfig { count: 0, ..config() }),
            ("too many cascades", CascadeConfig { count: 99, ..config() }),
            ("negative lambda", CascadeConfig { lambda: -5.0, ..config() }),
            ("lambda past one", CascadeConfig { lambda: 7.0, ..config() }),
            ("max distance below near", CascadeConfig { max_distance: 0.05, ..config() }),
        ];
        for (label, config) in cases {
            for (i, split) in split_distances(0.1, &config).iter().enumerate() {
                assert!(split.is_finite(), "{label}: split {i} is {split}");
            }
        }
        for split in split_distances(0.0, &config()) {
            assert!(split.is_finite(), "zero near plane produced {split}");
        }
    }

    // --- sub_frustum_corners ---

    /// The round trip through the matrix the slice was built from. Catches a
    /// wrong NDC z range (the [-1,1] OpenGL convention) immediately.
    #[test]
    fn corners_round_trip_into_the_ndc_cube() {
        let camera = Camera::default();
        let (near, far) = (5.0, 25.0);
        let view_proj = camera.projection_range(ASPECT, near, far) * camera.view();
        for corner in sub_frustum_corners(&camera, ASPECT, near, far) {
            let ndc = view_proj.project_point3(corner);
            assert!(inside_ndc(ndc, 1e-3), "{corner} -> {ndc}");
        }
    }

    /// The perspective divide is the easy half of this function to forget:
    /// `transform_point3` produces corners that look right near the view axis
    /// and drift badly at the edges, which this catches as a wrong plane depth.
    #[test]
    fn the_two_corner_groups_lie_on_the_slice_planes() {
        let camera = Camera::default();
        let (near, far) = (5.0, 25.0);
        let forward = (camera.target - camera.position).normalize();
        let corners = sub_frustum_corners(&camera, ASPECT, near, far);

        for (i, corner) in corners.iter().enumerate() {
            let depth = (*corner - camera.position).dot(forward);
            let expected = if i < 4 { near } else { far };
            assert!(
                (depth - expected).abs() < 1e-2,
                "corner {i} sits at depth {depth}, expected {expected}",
            );
        }
    }

    #[test]
    fn a_wider_aspect_widens_only_the_horizontal_extent() {
        let camera = Camera::default();
        let spread = |aspect: f32| {
            let corners = sub_frustum_corners(&camera, aspect, 5.0, 25.0);
            // Measure in view space, where x is horizontal by construction.
            let view = camera.view();
            let local: Vec<Vec3> = corners
                .iter()
                .map(|c| view.transform_point3(*c))
                .collect();
            let x = local.iter().map(|p| p.x.abs()).fold(0.0f32, f32::max);
            let y = local.iter().map(|p| p.y.abs()).fold(0.0f32, f32::max);
            (x, y)
        };
        let (narrow_x, narrow_y) = spread(1.0);
        let (wide_x, wide_y) = spread(2.0);
        assert!(wide_x > narrow_x * 1.5, "{wide_x} vs {narrow_x}");
        assert!((wide_y - narrow_y).abs() < 1e-2, "{wide_y} vs {narrow_y}");
    }

    /// The sphere fit takes the corner mean as its center, which is only a sane
    /// choice because a symmetric frustum's mean sits on the view axis.
    #[test]
    fn the_corner_mean_lies_on_the_view_axis() {
        let camera = Camera::default();
        let forward = (camera.target - camera.position).normalize();
        let mean = sub_frustum_corners(&camera, ASPECT, 5.0, 25.0)
            .iter()
            .sum::<Vec3>()
            / 8.0;
        let offset = mean - camera.position;
        assert!(
            offset.cross(forward).length() < 1e-2,
            "mean {mean} is off the view axis",
        );
    }

    // --- fit_cascade ---

    /// Snapping moves the box by up to one texel, so coverage is exact only to
    /// within that. One texel is `2 / resolution` in NDC units.
    #[test]
    fn the_fit_covers_every_corner_to_within_a_texel() {
        let camera = Camera::default();
        let config = config();
        let corners = sub_frustum_corners(&camera, ASPECT, 5.0, 25.0);
        let cascade = fit_cascade(&corners, sun(), 25.0, &config);
        let slack = 2.0 / config.resolution as f32;
        for corner in corners {
            let ndc = cascade.view_proj.project_point3(corner);
            assert!(inside_ndc(ndc, slack), "{corner} -> {ndc}");
        }
    }

    /// The property the whole sphere-fit exists for. A tight box fitted to the
    /// frustum corners would change size as the camera turns, and every shadow
    /// edge would crawl. Tolerance is the radius quantisation step, since a true
    /// radius sitting on a 1/16 boundary can round either way.
    #[test]
    fn rotating_the_camera_in_place_does_not_change_the_extent() {
        let base = Camera::default();
        let config = config();
        let mut min = f32::INFINITY;
        let mut max = 0.0f32;
        for step in 0..16 {
            let angle = step as f32 * std::f32::consts::TAU / 16.0;
            let camera = Camera {
                target: base.position + Vec3::new(angle.cos(), -0.2, angle.sin()),
                ..base
            };
            let extent = fit_slice(&camera, 5.0, 25.0, &config).half_extent;
            min = min.min(extent);
            max = max.max(extent);
        }
        assert!(
            max - min <= 1.0 / 16.0 + 1e-3,
            "extent swings from {min} to {max} as the camera turns",
        );
    }

    /// If this fails while the extent test passes, the snap is happening in a
    /// frame that moves with the camera — the shadows will shimmer exactly as
    /// much as if nothing were snapped at all.
    #[test]
    fn the_cascade_center_lands_on_the_texel_grid() {
        let camera = Camera::default();
        let config = config();
        let cascade = fit_slice(&camera, 5.0, 25.0, &config);
        let center = light_rot(sun()).transform_point3(snapped_center(&cascade, sun(), &config));
        for (axis, value) in [("x", center.x), ("y", center.y)] {
            let texels = value / cascade.texel_world_size;
            assert!(
                (texels - texels.round()).abs() < 1e-2,
                "{axis} center is {texels} texels, not a whole number",
            );
        }
    }

    #[test]
    fn moving_the_camera_one_texel_moves_the_grid_one_texel() {
        let base = Camera::default();
        let config = config();
        let rot = light_rot(sun());
        let before = fit_slice(&base, 5.0, 25.0, &config);
        let texel = before.texel_world_size;

        // The world-space direction of light-space +X, so the move is purely
        // along one axis of the grid being snapped against.
        let axis = rot.inverse().transform_vector3(Vec3::X).normalize();
        let moved = Camera {
            position: base.position + axis * texel,
            target: base.target + axis * texel,
            ..base
        };
        let after = fit_slice(&moved, 5.0, 25.0, &config);

        let delta = rot.transform_point3(snapped_center(&after, sun(), &config))
            - rot.transform_point3(snapped_center(&before, sun(), &config));
        assert!(
            (delta.x - texel).abs() < texel * 0.1,
            "moved {} along the grid, expected {texel}",
            delta.x,
        );
        assert!(delta.y.abs() < texel * 0.1, "grid drifted {} in y", delta.y);
    }

    /// The fit places its eye by scaling the light vector, so a caller that
    /// skips normalizing would silently move the near plane and clip the far
    /// corners out of the map. Nothing errors, so only this catches it.
    #[test]
    fn the_fit_normalizes_its_light_direction() {
        let camera = Camera::default();
        let config = config();
        let corners = sub_frustum_corners(&camera, ASPECT, 5.0, 25.0);
        let unit = fit_cascade(&corners, sun(), 25.0, &config);
        let scaled = fit_cascade(&corners, sun() * 3.0, 25.0, &config);
        assert!(
            unit.view_proj.abs_diff_eq(scaled.view_proj, 1e-3),
            "the fit depends on the light vector's length",
        );
    }

    /// A sun pointing straight down is a default scene setting, not an edge
    /// case: without the `up` guard `look_at_rh` returns NaNs that reach the
    /// shader as a black screen.
    #[test]
    fn a_light_parallel_to_up_produces_finite_matrices() {
        let camera = Camera::default();
        let config = config();
        let corners = sub_frustum_corners(&camera, ASPECT, 5.0, 25.0);
        for light in [Vec3::NEG_Y, Vec3::Y] {
            let cascade = fit_cascade(&corners, light, 25.0, &config);
            assert!(cascade.view_proj.is_finite(), "{light} view_proj");
            assert!(cascade.light_view.is_finite(), "{light} light_view");
        }
    }

    // --- cascades ---

    #[test]
    fn the_cascade_count_is_clamped_to_the_supported_range() {
        let camera = Camera::default();
        for (requested, expected) in [(0, 1), (1, 1), (4, 4), (99, MAX_CASCADES)] {
            let config = CascadeConfig {
                count: requested,
                ..config()
            };
            let set = cascades(&camera, ASPECT, sun(), &config);
            assert_eq!(set.count, expected, "requested {requested}");
        }
    }

    #[test]
    fn the_set_carries_the_same_splits_the_shader_selects_on() {
        let camera = Camera::default();
        let config = config();
        let set = cascades(&camera, ASPECT, sun(), &config);
        assert_eq!(set.splits, split_distances(camera.near, &config));
        for i in 0..set.count {
            assert_eq!(set.cascades[i].split_distance, set.splits[i]);
        }
    }

    #[test]
    fn cascades_grow_outward() {
        let camera = Camera::default();
        let set = cascades(&camera, ASPECT, sun(), &config());
        for i in 1..set.count {
            assert!(
                set.cascades[i].half_extent > set.cascades[i - 1].half_extent,
                "cascade {i} is not larger than {}",
                i - 1,
            );
        }
    }

    /// Each cascade must cover its own selection range — the slice the shader
    /// will actually route to it. The fitted slice is wider still (the blend
    /// overlap extends its near plane), so this holds without the test needing
    /// to know the overlap width.
    #[test]
    fn every_cascade_covers_the_slice_it_is_selected_for() {
        let camera = Camera::default();
        let config = config();
        let set = cascades(&camera, ASPECT, sun(), &config);
        let slack = 2.0 / config.resolution as f32;
        for i in 0..set.count {
            let near = if i == 0 { camera.near } else { set.splits[i - 1] };
            for corner in sub_frustum_corners(&camera, ASPECT, near, set.splits[i]) {
                let ndc = set.cascades[i].view_proj.project_point3(corner);
                assert!(inside_ndc(ndc, slack), "cascade {i}: {corner} -> {ndc}");
            }
        }
    }

    /// `fit_cascade` uses the light direction as a length as well as a
    /// direction — it scales the near-plane pullback — so an unnormalized sun
    /// from the inspector must not change the result.
    #[test]
    fn the_light_direction_is_normalized_before_use() {
        let camera = Camera::default();
        let config = config();
        let unit = cascades(&camera, ASPECT, sun().normalize(), &config);
        let scaled = cascades(&camera, ASPECT, sun().normalize() * 7.5, &config);
        for i in 0..unit.count {
            assert!(
                unit.cascades[i]
                    .view_proj
                    .abs_diff_eq(scaled.cascades[i].view_proj, 1e-3),
                "cascade {i} depends on the light vector's length",
            );
        }
    }

    #[test]
    fn a_zero_light_direction_produces_finite_matrices() {
        let camera = Camera::default();
        let set = cascades(&camera, ASPECT, Vec3::ZERO, &config());
        for i in 0..set.count {
            assert!(set.cascades[i].view_proj.is_finite(), "cascade {i}");
            assert!(set.cascades[i].light_view.is_finite(), "cascade {i}");
        }
    }
}