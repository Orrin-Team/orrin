use glam::{Mat4, Vec3};

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

    // Right-handed perspective for Vulkan's [0,1] depth range, with Y flipped for
    // its +Y-down clip space. Standard forward-Z: near maps to 0, far to 1, paired
    // with a LESS depth test and a depth clear of 1.0. (Not reverse-Z — that would
    // mean swapping near/far here and flipping the compare op + clear renderer-wide.)
    #[inline]
    pub fn projection(&self, aspect: f32) -> Mat4 {
        let mut proj = Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far);
        proj.y_axis.y *= -1.0; // flip Y for Vulkan clip space
        proj
    }

    #[inline]
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }
}
