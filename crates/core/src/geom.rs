//! Bounding volumes and the frustum test, shared by collision and rendering.
//!
//! Culling lives above the render backend — extraction runs it before any
//! renderer type is in reach — so the types it needs are here rather than in
//! `gfx`. Shadow cascades reuse the same `(frustum, bounds)` test with a light's
//! matrix in place of the camera's.

use glam::{Mat3, Mat4, Vec3, Vec4};

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// The identity for [`union`](Self::union) and the start of any
    /// accumulation: inverted, so the first point folded in becomes the box
    /// exactly. Not a valid box on its own — see [`is_valid`](Self::is_valid).
    pub const EMPTY: Self = Self {
        min: Vec3::INFINITY,
        max: Vec3::NEG_INFINITY,
    };

    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        points.into_iter().fold(Self::EMPTY, |bounds, point| Aabb {
            min: bounds.min.min(point),
            max: bounds.max.max(point),
        })
    }

    /// False for [`EMPTY`](Self::EMPTY) and for anything non-finite. Callers
    /// that cannot measure a volume must treat it as visible rather than cull
    /// it, so this is the guard the frustum test consults first.
    pub fn is_valid(&self) -> bool {
        self.min.is_finite() && self.max.is_finite() && self.min.cmple(self.max).all()
    }

    pub fn overlaps(&self, other: &Aabb) -> bool {
        (self.min.cmple(other.max) & other.min.cmple(self.max)).all()
    }

    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Squared distance from `point` to the nearest point of this box, and zero
    /// for a point inside it.
    ///
    /// Squared because every caller compares it against another of the same, and
    /// front-to-back ordering only needs the comparison. An invalid box has no
    /// nearest point, so it sorts to the back rather than producing a negative
    /// clamp of `min - max`.
    pub fn distance_squared_to(&self, point: Vec3) -> f32 {
        if !self.is_valid() {
            return f32::INFINITY;
        }
        let outside = (self.min - point).max(point - self.max).max(Vec3::ZERO);
        outside.length_squared()
    }

    /// The axis-aligned box enclosing this one after `model` — the same
    /// `abs(M) * half_extents` trick the collision side uses for a rotated box:
    /// along each world axis the farthest corner picks the sign of every term,
    /// which is the element-wise absolute value.
    ///
    /// Enclosing, not exact: a rotated box grows. Re-deriving it from the source
    /// vertices every frame would be tighter and far more expensive.
    pub fn transformed(&self, model: &Mat4) -> Aabb {
        if !self.is_valid() {
            return *self;
        }
        let center = model.transform_point3(self.center());
        let m = Mat3::from_mat4(*model);
        let abs = Mat3::from_cols(m.x_axis.abs(), m.y_axis.abs(), m.z_axis.abs());
        let extents = abs * self.half_extents();
        Aabb {
            min: center - extents,
            max: center + extents,
        }
    }
}

/// Six inward-facing clip planes, each `xyz` a unit normal and `w` the offset:
/// a point is inside when `dot(n, p) + w >= 0` for all six.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Gribb-Hartmann: the clip volume's planes are sums and differences of the
    /// rows of whatever matrix maps world space into it. That the matrix is a
    /// camera's is never assumed, so a shadow cascade's orthographic
    /// light-space matrix goes through this unchanged.
    ///
    /// Vulkan's clip volume is `-w <= x, y <= w` but `0 <= z <= w`, so the near
    /// plane is row 2 alone rather than OpenGL's `row3 + row2`. The projection's
    /// flipped Y (see [`Camera::projection`](crate::scene::Camera::projection))
    /// only swaps which of these is the top and which the bottom; as an
    /// unordered set of half-spaces the result is the same.
    pub fn from_view_projection(view_proj: Mat4) -> Self {
        let (r0, r1, r2, r3) = (
            view_proj.row(0),
            view_proj.row(1),
            view_proj.row(2),
            view_proj.row(3),
        );
        Self {
            planes: [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map(normalize_plane),
        }
    }

    /// Conservative: true means "may be visible". A box straddling two planes
    /// near a corner can pass every plane individually while lying outside the
    /// volume, which costs a draw, never a missing object.
    pub fn intersects(&self, bounds: &Aabb) -> bool {
        // Unmeasurable bounds are drawn, not culled.
        if !bounds.is_valid() {
            return true;
        }
        let center = bounds.center();
        let extents = bounds.half_extents();
        self.planes.iter().all(|plane| {
            let normal = plane.truncate();
            // `extents · |n|` is the box's support along the plane normal: the
            // signed distance from the center to its farthest corner. Adding it
            // to the center's distance tests the corner most likely to be
            // inside, so the box is out only when even that one is behind.
            let center_distance = normal.dot(center) + plane.w;
            center_distance + extents.dot(normal.abs()) >= 0.0
        })
    }
}

/// Scale a plane to a unit normal. The inside/outside sign is unchanged by any
/// positive scale, so this matters only for the plane's `w` to be a true
/// distance — which cascade selection will want.
fn normalize_plane(plane: Vec4) -> Vec4 {
    let length = plane.truncate().length();
    if length > 0.0 { plane / length } else { plane }
}

#[cfg(test)]
mod tests {
    use super::{Aabb, Frustum};
    use glam::{Mat4, Quat, Vec3};

    fn unit_cube() -> Aabb {
        Aabb {
            min: Vec3::splat(-0.5),
            max: Vec3::splat(0.5),
        }
    }

    /// A camera at +Z looking at the origin, matching `Camera::default`'s
    /// right-handed, Y-flipped Vulkan projection.
    fn camera_view_projection() -> Mat4 {
        let mut proj = Mat4::perspective_rh(60f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        proj.y_axis.y *= -1.0;
        proj * Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
    }

    #[test]
    fn bounds_of_points_enclose_them() {
        let bounds = Aabb::from_points([
            Vec3::new(-1.0, 0.0, 2.0),
            Vec3::new(3.0, -4.0, 0.0),
            Vec3::ZERO,
        ]);
        assert_eq!(bounds.min, Vec3::new(-1.0, -4.0, 0.0));
        assert_eq!(bounds.max, Vec3::new(3.0, 0.0, 2.0));
    }

    #[test]
    fn bounds_of_nothing_are_invalid() {
        assert!(!Aabb::from_points([]).is_valid());
        assert!(!Aabb::EMPTY.is_valid());
        assert!(unit_cube().is_valid());
    }

    #[test]
    fn a_transform_moves_and_scales_bounds() {
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 2.0, 2.0),
            Quat::IDENTITY,
            Vec3::new(10.0, 0.0, 0.0),
        );
        let bounds = unit_cube().transformed(&model);
        assert!((bounds.min - Vec3::new(9.0, -1.0, -1.0)).length() < 1e-5);
        assert!((bounds.max - Vec3::new(11.0, 1.0, 1.0)).length() < 1e-5);
    }

    /// A 45° turn makes the enclosing box wider by √2 on the rotated axes, and
    /// the box must still contain the geometry — an under-estimate here would
    /// cull objects that are on screen.
    #[test]
    fn a_rotation_grows_the_enclosing_box() {
        let model = Mat4::from_quat(Quat::from_rotation_y(45f32.to_radians()));
        let bounds = unit_cube().transformed(&model);
        let half = 0.5 * 2f32.sqrt();
        assert!((bounds.max.x - half).abs() < 1e-5, "{bounds:?}");
        assert!((bounds.max.z - half).abs() < 1e-5, "{bounds:?}");
        assert!((bounds.max.y - 0.5).abs() < 1e-5, "{bounds:?}");
    }

    #[test]
    fn an_object_in_front_of_the_camera_is_visible() {
        let frustum = Frustum::from_view_projection(camera_view_projection());
        assert!(frustum.intersects(&unit_cube()));
    }

    #[test]
    fn objects_outside_every_plane_are_culled() {
        let frustum = Frustum::from_view_projection(camera_view_projection());
        for offset in [
            Vec3::new(0.0, 0.0, 50.0),   // behind the camera
            Vec3::new(0.0, 0.0, -200.0), // past the far plane
            Vec3::new(100.0, 0.0, 0.0),  // off to the right
            Vec3::new(-100.0, 0.0, 0.0),
            Vec3::new(0.0, 100.0, 0.0), // above
            Vec3::new(0.0, -100.0, 0.0),
        ] {
            let bounds = unit_cube().transformed(&Mat4::from_translation(offset));
            assert!(!frustum.intersects(&bounds), "should be culled: {offset}");
        }
    }

    /// The near plane is the one Vulkan's `0 <= z` clip range makes easy to get
    /// wrong (the OpenGL form would place it a full frustum-depth behind).
    #[test]
    fn the_near_plane_sits_at_the_camera() {
        let frustum = Frustum::from_view_projection(camera_view_projection());
        // Camera is at z = 5 looking down -Z with near = 0.1: a small box just
        // in front of it is in, the same box just behind it is out.
        let tiny = Aabb {
            min: Vec3::splat(-0.01),
            max: Vec3::splat(0.01),
        };
        assert!(
            frustum
                .intersects(&tiny.transformed(&Mat4::from_translation(Vec3::new(0.0, 0.0, 4.0))))
        );
        assert!(
            !frustum
                .intersects(&tiny.transformed(&Mat4::from_translation(Vec3::new(0.0, 0.0, 5.5))))
        );
    }

    /// An object big enough to span the whole view is visible even though every
    /// one of its corners is outside the frustum.
    #[test]
    fn an_enclosing_box_is_visible() {
        let frustum = Frustum::from_view_projection(camera_view_projection());
        let huge = Aabb {
            min: Vec3::splat(-1000.0),
            max: Vec3::splat(1000.0),
        };
        assert!(frustum.intersects(&huge));
    }

    #[test]
    fn unmeasurable_bounds_are_never_culled() {
        let frustum = Frustum::from_view_projection(camera_view_projection());
        assert!(frustum.intersects(&Aabb::EMPTY));
    }

    /// The same test has to work for the orthographic light-space matrices the
    /// shadow cascades will hand it, not just perspective ones.
    #[test]
    fn an_orthographic_matrix_culls_too() {
        let light = Mat4::orthographic_rh(-2.0, 2.0, -2.0, 2.0, 0.1, 20.0)
            * Mat4::look_at_rh(Vec3::new(0.0, 10.0, 0.0), Vec3::ZERO, Vec3::Z);
        let frustum = Frustum::from_view_projection(light);
        assert!(frustum.intersects(&unit_cube()));
        let outside = unit_cube().transformed(&Mat4::from_translation(Vec3::new(9.0, 0.0, 0.0)));
        assert!(!frustum.intersects(&outside));
    }
}
