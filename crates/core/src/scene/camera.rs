use glam::{Mat4, Vec3};

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
}
