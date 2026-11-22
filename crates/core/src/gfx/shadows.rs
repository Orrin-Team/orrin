use glam::{IVec2, Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::scene::Camera;

pub const MAX_CASCADES: usize = 4;
pub const OVERLAP: f32 = 0.005;

/// Vulkan NDC: x,y in [-1,1], z in [0,1]. First four are the near plane.
const NDC_CORNERS: [Vec3; 8] = {
    use crate::gfx::depth::{FAR, NEAR};
    [
        Vec3::new(-1.0, -1.0, NEAR),
        Vec3::new(1.0, -1.0, NEAR),
        Vec3::new(-1.0, 1.0, NEAR),
        Vec3::new(1.0, 1.0, NEAR),
        Vec3::new(-1.0, -1.0, FAR),
        Vec3::new(1.0, -1.0, FAR),
        Vec3::new(-1.0, 1.0, FAR),
        Vec3::new(1.0, 1.0, FAR),
    ]
};

/// How far past its split distance a level's square reaches, in its own texels:
/// the filter kernel's reach and the normal offset a lookup is moved by before
/// it is projected.
const LEVEL_MARGIN_TEXELS: f32 = 8.0;

/// How far a level's square moves when it moves, as a fraction of its edge.
///
/// A level does not follow the camera texel by texel. Every move slides what
/// its tile holds, and sliding a tile is a touch of every texel in it, which on
/// a tiling GPU costs more than drawing the casters does — so it moves rarely
/// and by a strip at a time, and is that much larger than the disc it answers
/// for to leave the camera room to walk about inside it.
const LEVEL_STEP: f32 = 1.0 / 16.0;

/// How far from the middle of its square the camera gets, in steps, before the
/// square follows. Past half a step, so a camera pacing across one boundary
/// does not move the square on every crossing.
const LEVEL_SLACK_STEPS: f32 = 0.75;

/// The least a level's depth range is padded by at each end, in metres. The
/// range is anchored rather than centred on the camera for the reason the
/// square is: depth already drawn is only reusable while the plane it was
/// measured from stays put.
const LEVEL_DEPTH_STEP: f32 = 16.0;

pub struct CascadeConfig {
    pub count: usize,
    pub max_distance: f32,
    pub lambda: f32,
    /// Edge of the tiles of the sun's first two views, in texels.
    pub resolution: u32,
    /// Edge of the tiles of the views past those, which answer for a distance
    /// where a texel is fewer pixels.
    pub far_resolution: u32,
    pub pullback: f32,
    /// How many of the sun's views, nearest first, are [fitted to the view
    /// frustum](fit_cascade) rather than [placed around the camera](LevelWindow).
    pub fitted: usize,
}

/// How many of the sun's views take the larger tile.
const NEAR_VIEWS: usize = 2;

impl CascadeConfig {
    /// Edge of the tile sun view `index` is drawn into.
    pub fn resolution_of(&self, index: usize) -> u32 {
        if index < NEAR_VIEWS {
            self.resolution
        } else {
            self.far_resolution
        }
    }
}

/// What can cast into one of the sun's views, as a box in the space of its
/// `light_view`: anything that reaches the box when swept toward the light. No
/// upper `z`, because an object between the sun and the box is exactly what
/// casts into it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SweepBounds {
    /// Least `x` and `y`, and the far plane's `z`.
    pub min: Vec3,
    /// Greatest `x` and `y`.
    pub max: Vec2,
}

/// A level: a square of the sun's view centred near the camera, on a grid that
/// does not move.
///
/// Where [`fit_cascade`] bounds the slice of the frustum a cascade answers for,
/// a level bounds the whole disc the shader's radial selection can route to it,
/// in view or not. That makes it independent of where the camera is looking —
/// turning never moves it — and it is smaller than the sphere around a wide
/// frustum's far corners besides.
///
/// The grid is the level's own texels, counted from the light-space origin:
/// cell `(i, j)` covers `x` in `[i, i + 1) * texel` and `y` in
/// `(-(j + 1), -j] * texel`, the second axis running down because a tile's rows
/// do. A window is `cells` of them each way from `min_cell`, and a depth range
/// hung from `z_eye`. Because the grid is fixed, a window that has moved shares
/// most of its cells with where it was, and what was drawn for them is still
/// right: only the strip it moved onto is new.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelWindow {
    pub min_cell: IVec2,
    pub cells: u32,
    /// World size of one cell.
    pub texel: f32,
    /// Light-space `z` of the plane depth is measured from, the end of the
    /// range nearest the light.
    pub z_eye: f32,
    pub depth_range: f32,
}

impl LevelWindow {
    /// Where a level answering for everything within `radius` of the camera
    /// sits, given where it sat. `camera` is in light space.
    ///
    /// `previous` is kept while it still covers the disc, with room to spare:
    /// that is what makes a level's matrix the same frame after frame while the
    /// camera moves about inside it.
    pub fn place(
        camera: Vec3,
        radius: f32,
        resolution: u32,
        pullback: f32,
        previous: Option<&LevelWindow>,
    ) -> Self {
        let cells = resolution.max(64);
        let n = cells as f32;
        let step = (n * LEVEL_STEP).round().max(1.0);
        let slack = step * LEVEL_SLACK_STEPS;
        // Solved so that the square still reaches `radius` with the camera a
        // slack off centre and the margin taken off it: the texel is a fraction
        // of the extent it is a margin of.
        let half_extent = radius / (1.0 - 2.0 * (slack + LEVEL_MARGIN_TEXELS) / n);
        let texel = 2.0 * half_extent / n;
        let depth_step = radius.max(LEVEL_DEPTH_STEP);
        let depth_range = 2.0 * (radius + depth_step) + pullback;

        let previous = previous.filter(|old| {
            old.cells == cells && old.texel == texel && old.depth_range == depth_range
        });

        let at = Vec2::new(camera.x, -camera.y) / texel;
        let half = (cells / 2) as i32;
        let min_cell = previous
            .map(|old| old.min_cell)
            .filter(|min| {
                let off = at - (min.as_vec2() + Vec2::splat(half as f32));
                off.abs().max_element() <= slack
            })
            .unwrap_or_else(|| ((at / step).round() * step).as_ivec2() - IVec2::splat(half));

        // The light looks down -z, so the end nearest it is the greatest `z`.
        let needed = camera.z + radius + pullback;
        let z_eye = previous
            .map(|old| old.z_eye)
            .filter(|eye| *eye >= needed && *eye - depth_range <= camera.z - radius)
            .unwrap_or(needed + depth_step);

        Self {
            min_cell,
            cells,
            texel,
            z_eye,
            depth_range,
        }
    }

    /// Half the square's edge, in metres.
    pub fn half_extent(&self) -> f32 {
        0.5 * self.cells as f32 * self.texel
    }

    /// The projection that lands the `cells`-wide square of cells from `min`
    /// on a whole tile, cell for texel. Light space in, so it goes after the
    /// light's rotation.
    fn projection(&self, min: IVec2) -> Mat4 {
        let size = self.cells as f32 * self.texel;
        let left = min.x as f32 * self.texel;
        let top = -(min.y as f32) * self.texel;
        let mut ortho = Mat4::orthographic_rh(
            left,
            left + size,
            top - size,
            top,
            -self.z_eye,
            -self.z_eye + self.depth_range,
        );
        // The flip the camera's projection has, which is what makes a
        // triangle's winding come out the same here as in the forward pass —
        // and what makes a tile's rows run down the grid's second axis. Both
        // terms of the row, because this box is not centred on the origin.
        ortho.y_axis.y *= -1.0;
        ortho.w_axis.y *= -1.0;
        ortho
    }

    /// What the level is sampled with: the window on its tile.
    pub fn view_proj(&self, rotation: Mat4) -> Mat4 {
        self.projection(self.min_cell) * rotation
    }

    /// What a strip of the level is drawn with into its cached layer, whose
    /// texels are kept at their cells' places modulo the tile rather than the
    /// window's: the square of cells from `page`, which is the window slid by
    /// whole tiles so the strip lands where its cells are kept.
    pub fn page_view_proj(&self, rotation: Mat4, page: IVec2) -> Mat4 {
        self.projection(page) * rotation
    }

    /// What can cast into the cells from `min` up to `max`.
    pub fn sweep(&self, min: IVec2, max: IVec2) -> SweepBounds {
        SweepBounds {
            min: Vec3::new(
                min.x as f32 * self.texel,
                -(max.y as f32) * self.texel,
                self.z_eye - self.depth_range,
            ),
            max: Vec2::new(max.x as f32 * self.texel, -(min.y as f32) * self.texel),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Cascade {
    pub view_proj: Mat4,  // render + sample
    pub light_view: Mat4, // cull: world -> light space
    pub half_extent: f32, // the snapped sphere radius, r
    pub depth_range: f32, // 0 .. 2r + pullback
    pub split_distance: f32,
    pub texel_world_size: f32,
    /// What casts into it, in `light_view`'s space.
    pub bounds: SweepBounds,
    /// The window, for a view that is a level rather than fitted to the
    /// frustum.
    pub window: Option<LevelWindow>,
}

/// A `count` of zero means shadows are off — every other field is then inert,
/// which is what `Default` produces.
#[derive(Clone, Copy, Default)]
pub struct CascadeSet {
    pub cascades: [Cascade; MAX_CASCADES],
    pub splits: [f32; MAX_CASCADES],
    pub count: usize,
    /// `|R|` for the rotation every cascade's `light_view` shares: the element-
    /// wise absolute value of world-to-light-space, with no translation.
    ///
    /// A property of the *set* rather than of a cascade, because it is the same
    /// matrix in all of them — see [`light_rotation`]. Multiplying a box's half-
    /// extents by it gives that box's half-extent in light space, which is what
    /// the culling sweep tests against every cascade in turn. Hoisted here so
    /// the sweep derives it once per entity instead of once per entity per
    /// cascade, which was the same answer computed four times.
    ///
    /// Inert when `count` is zero, like every other field.
    pub abs_light_rotation: Mat3,
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

    for (i, split) in splits.iter_mut().take(count).enumerate() {
        let t = (i + 1) as f32 / count as f32;
        let log_split = near * (far / near).powf(t);
        let uniform_split = near + t * (far - near);
        *split = lambda * log_split + (1.0 - lambda) * uniform_split;
    }

    // Exact, not what `powf` returns at t = 1: the fit uses this as the last
    // cascade's far plane and the shader compares against it, so an ulp of
    // disagreement is an unshadowed ring at the edge of the last cascade.
    splits[count - 1] = far;

    splits
}

/// The up vector the light-space frame is built from: +Y, except where the
/// light is near enough to vertical that +Y is degenerate as an up.
///
/// One function rather than two copies, because [`light_rotation`] claiming to
/// reproduce `fit_cascade`'s frame is only true while they agree about this.
fn light_up(light_dir: Vec3) -> Vec3 {
    if light_dir.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    }
}

/// The rotation every cascade's `light_view` shares: world space into light
/// space, with the eye at the origin.
///
/// `look_at_rh`'s upper 3x3 is a function of the view direction and `up` alone —
/// the eye contributes only the translation — and every cascade looks along the
/// same `light_dir` with the same `up`. Only where each one is *standing*
/// differs, which is why the culling sweep can transform a centre per cascade
/// but an extent only once.
///
/// `light_dir` must be unit length, as everywhere else in this module.
pub fn light_rotation(light_dir: Vec3) -> Mat3 {
    Mat3::from_mat4(Mat4::look_at_rh(Vec3::ZERO, light_dir, light_up(light_dir)))
}

/// Element-wise `|m|`. Along each light-space axis the farthest corner of a
/// rotated box picks the sign of every term, which is what makes the absolute
/// value the right thing to multiply half-extents by — the same trick
/// [`Aabb::transformed`](crate::geom::Aabb::transformed) uses.
fn abs_mat3(m: Mat3) -> Mat3 {
    Mat3::from_cols(m.x_axis.abs(), m.y_axis.abs(), m.z_axis.abs())
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
    let mut radius = corners
        .iter()
        .map(|c| (*c - center).length())
        .fold(0.0f32, f32::max);
    radius = (radius * 16.0).ceil() / 16.0; // Quantising the radius because of float noise

    // Unit length is load-bearing, not cosmetic: `eye` below is placed by
    // scaling this vector, so a non-unit direction moves the near plane and
    // clips the far corners out of the map with no error anywhere.
    let light_dir = light_dir.normalize();

    let up = light_up(light_dir);
    let rot = Mat4::look_at_rh(Vec3::ZERO, light_dir, up); // Rotation only, eye at origin

    let texel = 2.0 * radius / config.resolution as f32;
    let mut c_ls = rot.transform_point3(center);
    c_ls.x = (c_ls.x / texel).floor() * texel;
    c_ls.y = (c_ls.y / texel).floor() * texel;

    let snapped_center = rot.inverse().transform_point3(c_ls);
    let eye = snapped_center - light_dir * (radius + config.pullback);
    let light_view = Mat4::look_at_rh(eye, snapped_center, up);

    let mut ortho = Mat4::orthographic_rh(
        -radius,
        radius,
        -radius,
        radius,
        0.0,
        2.0 * radius + config.pullback,
    );
    ortho.y_axis.y *= -1.0;

    let view_proj = ortho * light_view;
    let half_extent = radius;
    let depth_range = 2.0 * radius + config.pullback;

    Cascade {
        view_proj,
        light_view,
        half_extent,
        depth_range,
        split_distance,
        texel_world_size: texel,
        bounds: SweepBounds {
            min: Vec3::new(-half_extent, -half_extent, -depth_range),
            max: Vec2::splat(half_extent),
        },
        window: None,
    }
}

/// The sun's view for a level's window.
fn level(window: LevelWindow, light_dir: Vec3, split_distance: f32) -> Cascade {
    let rotation = Mat4::look_at_rh(Vec3::ZERO, light_dir, light_up(light_dir));
    Cascade {
        view_proj: window.view_proj(rotation),
        light_view: rotation,
        half_extent: window.half_extent(),
        depth_range: window.depth_range,
        split_distance,
        texel_world_size: window.texel,
        bounds: window.sweep(
            window.min_cell,
            window.min_cell + IVec2::splat(window.cells as i32),
        ),
        window: Some(window),
    }
}

/// The corners of the slice of the view cascade `index` answers for: from the
/// previous split, pulled in by the blend overlap, to its own.
fn slice_corners(
    camera: &Camera,
    aspect: f32,
    splits: &[f32; MAX_CASCADES],
    index: usize,
) -> [Vec3; 8] {
    let near = if index == 0 {
        camera.near
    } else {
        splits[index - 1] * (1.0 - OVERLAP)
    };
    sub_frustum_corners(camera, aspect, near, splits[index])
}

/// The most planes [`swept_slice`] returns: a frustum's outline seen from any
/// direction has at most six edges, and one more plane closes the far end.
pub const MAX_SWEEP_PLANES: usize = 8;

/// A slice of the view frustum swept toward the light. See [`swept_slice`].
#[derive(Clone, Copy, Debug)]
pub struct SweptSlice {
    pub planes: [Vec4; MAX_SWEEP_PLANES],
    /// How many of `planes` are set.
    pub count: usize,
    /// The slice's own corners, which bound where in the sun's view the camera
    /// can be reading.
    pub corners: [Vec3; 8],
}

/// What can cast into the part of a sun view the camera can see: the slice of
/// the view frustum that view `index` answers for, swept toward the light.
///
/// As planes whose positive side is inside, in world space: one through each
/// edge of the slice's outline as the light sees it, and one behind the slice's
/// farthest point from the light. Open toward the light, like every sun
/// volume. Tighter than the view's own box wherever the camera is not looking
/// at all of it — and a level's box is the whole disc around the camera, of
/// which a frustum is a wedge.
///
/// For the layers a frame draws and throws away. What a cached layer holds has
/// to be right wherever the camera turns next, so it is culled against the box
/// and never against this.
///
/// `None` when the outline does not come out as a polygon, which is a slice
/// seen exactly end on.
pub fn swept_slice(
    camera: &Camera,
    aspect: f32,
    light_dir: Vec3,
    splits: &[f32; MAX_CASCADES],
    index: usize,
) -> Option<SweptSlice> {
    let light_dir = light_dir.try_normalize().unwrap_or(Vec3::NEG_Y);
    // The view selects by distance from the eye, and a fragment that far along
    // a ray to the frustum's corner is nearer than that in depth: the slice
    // starts where the corner ray reaches the previous split.
    let corner = (camera.fov_y * 0.5).tan() * (1.0 + aspect * aspect).sqrt();
    let near = if index == 0 {
        camera.near
    } else {
        splits[index - 1] * (1.0 - OVERLAP) / (1.0 + corner * corner).sqrt()
    };
    let corners = sub_frustum_corners(camera, aspect, near.max(camera.near), splits[index]);
    let rotation = Mat4::look_at_rh(Vec3::ZERO, light_dir, light_up(light_dir));
    let lit = corners.map(|corner| rotation.transform_point3(corner));

    // The outline: the convex hull of the corners as the light sees them, by
    // Andrew's monotone chain, counter-clockwise.
    let mut points = lit.map(|point| point.truncate());
    points.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let mut hull = [Vec2::ZERO; 16];
    let mut count = 0;
    for pass in 0..2 {
        let start = count;
        for step in 0..points.len() {
            let point = if pass == 0 {
                points[step]
            } else {
                points[points.len() - 1 - step]
            };
            while count >= start + 2 && cross(hull[count - 2], hull[count - 1], point) <= 1e-6 {
                count -= 1;
            }
            hull[count] = point;
            count += 1;
        }
        // Each chain ends on the point the other begins with.
        count -= 1;
    }
    if !(3..MAX_SWEEP_PLANES).contains(&count) {
        return None;
    }

    let to_world = Mat3::from_mat4(rotation).transpose();
    let mut planes = [Vec4::ZERO; MAX_SWEEP_PLANES];
    for edge in 0..count {
        let (a, b) = (hull[edge], hull[(edge + 1) % count]);
        // Inside is to the left of a counter-clockwise edge.
        let normal = Vec2::new(a.y - b.y, b.x - a.x).normalize_or_zero();
        planes[edge] = (to_world * normal.extend(0.0)).extend(-normal.dot(a));
    }
    // The light looks down -z, so the slice's far end is its least z.
    let far = lit
        .iter()
        .map(|point| point.z)
        .fold(f32::INFINITY, f32::min);
    planes[count] = (to_world * Vec3::Z).extend(-far);
    Some(SweptSlice {
        planes,
        count: count + 1,
        corners,
    })
}

pub fn cascades(
    camera: &Camera,
    aspect: f32,
    light_dir: Vec3,
    config: &CascadeConfig,
) -> CascadeSet {
    sun_views(camera, aspect, light_dir, config, &[None; MAX_CASCADES])
}

/// The sun's views for a frame: the first [`CascadeConfig::fitted`] fitted to
/// their slices of the view, the rest placed around the camera as levels.
///
/// `previous` is where each level's window sat last frame, which it keeps while
/// it can — see [`LevelWindow::place`]. A caller that passes none gets every
/// level centred afresh.
pub fn sun_views(
    camera: &Camera,
    aspect: f32,
    light_dir: Vec3,
    config: &CascadeConfig,
    previous: &[Option<LevelWindow>; MAX_CASCADES],
) -> CascadeSet {
    let count = config.count.clamp(1, MAX_CASCADES);
    // Unit length is not cosmetic here: `fit_cascade` scales the near-plane
    // pullback by this vector, so an inspector sun of (0,-2,0) would silently
    // double it. A zero sun has no direction to normalize and would reach
    // `look_at_rh` as NaNs, so it falls back to straight down.
    let light_dir = light_dir.try_normalize().unwrap_or(Vec3::NEG_Y);
    let splits = split_distances(camera.near, config);
    let rotation = Mat4::look_at_rh(Vec3::ZERO, light_dir, light_up(light_dir));
    let camera_ls = rotation.transform_point3(camera.position);

    let mut cascades = [Cascade::default(); MAX_CASCADES];
    for i in 0..count {
        cascades[i] = if i < config.fitted {
            let corners = slice_corners(camera, aspect, &splits, i);
            fit_cascade(&corners, light_dir, splits[i], config)
        } else {
            let window = LevelWindow::place(
                camera_ls,
                splits[i],
                config.resolution_of(i),
                config.pullback,
                previous[i].as_ref(),
            );
            level(window, light_dir, splits[i])
        };
    }

    // The sweep culls against a rotation it is told the cascades share. If a
    // future change to the fit ever breaks that — a per-cascade up vector, a
    // stabilisation pass that rotates one of them — this is where it says so,
    // rather than the culling quietly answering for the wrong box.
    #[cfg(debug_assertions)]
    {
        let shared = light_rotation(light_dir);
        for (index, cascade) in cascades[..count].iter().enumerate() {
            assert!(
                Mat3::from_mat4(cascade.light_view).abs_diff_eq(shared, 1e-4),
                "cascade {index} does not share the set's light rotation; \
                 `CascadeSet::abs_light_rotation` is no longer valid for it",
            );
        }
    }

    CascadeSet {
        cascades,
        splits,
        count,
        abs_light_rotation: abs_mat3(light_rotation(light_dir)),
    }
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
            far_resolution: 1024,
            pullback: 50.0,
            fitted: 2,
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
        for (i, &split) in splits.iter().take(config.count - 1).enumerate() {
            let t = (i + 1) as f32 / config.count as f32;
            let expected = near + t * (far - near);
            assert!(
                (split - expected).abs() < 1e-3,
                "cascade {i}: {split} != {expected}",
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
        for (i, &split) in splits.iter().take(config.count - 1).enumerate() {
            let t = (i + 1) as f32 / config.count as f32;
            let expected = near * (far / near).powf(t);
            assert!(
                (split - expected).abs() < 1e-3,
                "cascade {i}: {split} != {expected}",
            );
        }
    }

    /// Slots past `count` must be unreachable by the shader's first-hit scan.
    /// Holding `max_distance` makes them inert; a zero would win every
    /// comparison and index a cascade with no matrix behind it.
    #[test]
    fn unused_slots_hold_the_far_distance() {
        let config = CascadeConfig {
            count: 2,
            ..config()
        };
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
            (
                "zero cascades",
                CascadeConfig {
                    count: 0,
                    ..config()
                },
            ),
            (
                "too many cascades",
                CascadeConfig {
                    count: 99,
                    ..config()
                },
            ),
            (
                "negative lambda",
                CascadeConfig {
                    lambda: -5.0,
                    ..config()
                },
            ),
            (
                "lambda past one",
                CascadeConfig {
                    lambda: 7.0,
                    ..config()
                },
            ),
            (
                "max distance below near",
                CascadeConfig {
                    max_distance: 0.05,
                    ..config()
                },
            ),
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
            let local: Vec<Vec3> = corners.iter().map(|c| view.transform_point3(*c)).collect();
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

    /// The light-space position of a world point, as the level grid counts it.
    fn in_light_space(point: Vec3) -> Vec3 {
        light_rot(sun()).transform_point3(point)
    }

    /// What a level is for: everything within its radius of the camera is
    /// inside its square, in every direction, so no turn of the camera can
    /// carry the view out of it — and it stays so while the camera walks about
    /// inside the slack the window was placed with, without the window moving.
    #[test]
    fn a_level_holds_its_disc_while_the_camera_moves_inside_its_slack() {
        let start = Vec3::new(13.7, 2.0, -41.3);
        let placed = LevelWindow::place(in_light_space(start), 32.0, 1024, 50.0, None);
        let rotation = light_rot(sun());
        let step = placed.cells as f32 * LEVEL_STEP * placed.texel;

        for travelled in [
            Vec3::ZERO,
            Vec3::X * step * 0.2,
            Vec3::new(-0.2, 0.3, 0.2) * step,
        ] {
            let camera = start + travelled;
            let window =
                LevelWindow::place(in_light_space(camera), 32.0, 1024, 50.0, Some(&placed));
            assert_eq!(window, placed, "the window moved after {travelled}");
            for direction in [
                Vec3::X,
                Vec3::NEG_X,
                Vec3::Y,
                Vec3::NEG_Y,
                Vec3::Z,
                Vec3::NEG_Z,
                Vec3::new(1.0, 1.0, 1.0).normalize(),
                Vec3::new(-1.0, 0.3, -1.0).normalize(),
            ] {
                let ndc = window
                    .view_proj(rotation)
                    .project_point3(camera + direction * 32.0);
                assert!(inside_ndc(ndc, 0.0), "{travelled}: {direction} -> {ndc}");
            }
        }
    }

    /// And when the camera does leave the slack, the window follows by whole
    /// cells of a grid that has not moved: the cells the two windows share are
    /// the same cells, which is what lets their texels be kept.
    #[test]
    fn a_level_that_moves_stays_on_its_grid() {
        let start = Vec3::new(13.7, 2.0, -41.3);
        let placed = LevelWindow::place(in_light_space(start), 32.0, 1024, 50.0, None);
        let rotation = light_rot(sun());
        let across = rotation.inverse().transform_vector3(Vec3::X);
        let far = start + across * placed.half_extent() * 0.5;
        let moved = LevelWindow::place(in_light_space(far), 32.0, 1024, 50.0, Some(&placed));

        assert_ne!(moved.min_cell, placed.min_cell, "the window never followed");
        assert_eq!(moved.texel, placed.texel);
        assert_eq!(
            moved.z_eye, placed.z_eye,
            "a sideways move re-anchored depth"
        );
        // A point's texel under each window differs by exactly the cells the
        // window moved, so it is the same cell of the same grid.
        let point = start + Vec3::new(3.0, 0.0, -2.0);
        let texel = |window: &LevelWindow| {
            let ndc = window.view_proj(rotation).project_point3(point);
            (ndc.truncate() * 0.5 + 0.5) * window.cells as f32
        };
        let slid = texel(&placed) - texel(&moved);
        let cells = (moved.min_cell - placed.min_cell).as_vec2();
        assert!(slid.abs_diff_eq(cells, 1e-2), "{slid} against {cells}");
    }

    /// What the sweep is for and what it must never do. A caster standing in
    /// the slice, or anywhere between it and the sun, is inside every plane;
    /// one behind the camera is not, though the level's own box holds it.
    #[test]
    fn a_swept_slice_keeps_what_shadows_the_view_and_drops_what_is_behind_it() {
        let camera = Camera::default();
        let config = config();
        let splits = split_distances(camera.near, &config);
        let forward = (camera.target - camera.position).normalize();
        let inside = |index: usize, point: Vec3| {
            let slice = swept_slice(&camera, ASPECT, sun(), &splits, index).unwrap();
            slice.planes[..slice.count]
                .iter()
                .all(|plane| plane.truncate().dot(point) + plane.w >= -1e-3)
        };

        for index in 0..config.count {
            let near = if index == 0 {
                camera.near
            } else {
                splits[index - 1]
            };
            let middle = camera.position + forward * 0.5 * (near + splits[index]);
            assert!(inside(index, middle), "view {index} lost its own middle");
            assert!(
                inside(index, middle - sun().normalize() * 40.0),
                "view {index} lost a caster between it and the sun"
            );
            assert!(
                !inside(index, camera.position - forward * splits[index]),
                "view {index} kept a caster behind the camera"
            );
        }
        // Every corner of every slice, which is where a plane facing the wrong
        // way shows first.
        for index in 0..config.count {
            for corner in slice_corners(&camera, ASPECT, &splits, index) {
                assert!(inside(index, corner), "view {index} lost {corner}");
            }
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
        for i in 0..set.count.min(config.fitted) {
            let near = if i == 0 {
                camera.near
            } else {
                set.splits[i - 1]
            };
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
        for (i, cascade) in set.cascades.iter().take(set.count).enumerate() {
            assert!(cascade.view_proj.is_finite(), "cascade {i}");
            assert!(cascade.light_view.is_finite(), "cascade {i}");
        }
    }
}
