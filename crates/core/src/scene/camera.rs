use glam::{Mat4, Quat, Vec3};

use crate::geom::Frustum;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 1.5, 4.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Camera {
    #[inline]
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    #[inline]
    pub fn projection_range(&self, aspect: f32, near: f32, far: f32) -> Mat4 {
        let mut proj = Mat4::perspective_rh(self.fov_y, aspect, near, far);
        proj.y_axis.y *= -1.0;
        proj
    }

    // Right-handed perspective for Vulkan's [0,1] depth range, with Y flipped for
    // its +Y-down clip space. Standard forward-Z: near maps to 0, far to 1, paired
    // with a LESS depth test and a depth clear of 1.0. (Not reverse-Z — that would
    // mean swapping near/far here and flipping the compare op + clear renderer-wide.)
    #[inline]
    pub fn projection(&self, aspect: f32) -> Mat4 {
        self.projection_range(aspect, self.near, self.far)
    }

    #[inline]
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }

    /// The six clip planes of this camera's volume, for culling. `aspect` must
    /// be the one the frame is drawn with, or the side planes will not match
    /// what ends up on screen.
    #[inline]
    pub fn frustum(&self, aspect: f32) -> Frustum {
        Frustum::from_view_projection(self.view_projection(aspect))
    }

    /// Which way the camera faces, as a rotation of -Z (forward) and +Y (up):
    /// the form a script holds an orientation in.
    pub fn rotation(&self) -> Quat {
        let forward = (self.target - self.position).normalize_or(Vec3::NEG_Z);
        let right = forward.cross(self.up).normalize_or(Vec3::X);
        let up = right.cross(forward);
        Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward)).normalize()
    }

    /// Stand at `position` facing `rotation`. The point looked at keeps its
    /// distance, because the editor orbits and frames around it.
    pub fn place(&mut self, position: Vec3, rotation: Quat) {
        let distance = self.target.distance(self.position).max(1e-3);
        let rotation = rotation.normalize();
        self.position = position;
        self.target = position + rotation * Vec3::NEG_Z * distance;
        self.up = rotation * Vec3::Y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        a.distance(b) < 1e-4
    }

    #[test]
    fn the_rotation_faces_the_target_with_the_up_kept_up() {
        let camera = Camera::default();
        let rotation = camera.rotation();
        assert!(close(
            rotation * Vec3::NEG_Z,
            (camera.target - camera.position).normalize()
        ));
        assert!((rotation * Vec3::Y).dot(Vec3::Y) > 0.9, "upright");
    }

    #[test]
    fn a_placed_camera_reads_back_where_and_how_it_was_placed() {
        let mut camera = Camera::default();
        let distance = camera.target.distance(camera.position);
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, 0.6, -0.3, 0.0);
        let position = Vec3::new(2.0, 1.0, -3.0);

        camera.place(position, rotation);

        assert_eq!(camera.position, position);
        assert!(close(
            camera.rotation() * Vec3::NEG_Z,
            rotation * Vec3::NEG_Z
        ));
        assert!(close(camera.up, rotation * Vec3::Y));
        assert!((camera.target.distance(position) - distance).abs() < 1e-4);
    }
}
