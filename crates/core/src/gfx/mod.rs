pub mod vulkan;

use crate::scene::{Camera, CpuMesh, HdrSettings, MaterialHandle, MeshHandle, SsaoSettings};
use glam::{Mat4, Vec3};
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex as VertexTrait;

#[derive(BufferContents, VertexTrait, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    /// Object-space tangent (+U texture direction) in `xyz`; `w` is the
    /// bitangent handedness (±1) used to rebuild the TBN basis for normal maps.
    #[format(R32G32B32A32_SFLOAT)]
    pub tangent: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct RenderItem {
    pub model: Mat4,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

pub const MAX_POINT_LIGHTS: usize = 16;

/// Size of the shader's bound texture array (set 2). Keep at or below the
/// device's `maxPerStageDescriptorSampledImages` (≥16 guaranteed; MoltenVK
/// allows far more).
pub const MAX_TEXTURES: usize = 64;

/// The `u32` is the texture's index in the shader's array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    /// The direction the light *travels* (e.g. roughly downward for a sun).
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
}

#[derive(Clone, Debug)]
pub struct SceneLighting {
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub sun: DirectionalLight,
    /// Anything past [`MAX_POINT_LIGHTS`] is ignored.
    pub point_lights: Vec<PointLight>,
    /// Blinn-Phong specular exponent. Higher = smaller, sharper highlight.
    pub shininess: f32,
    pub specular_strength: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32,
    pub emissive: Vec3,
    pub albedo_texture: Option<TextureHandle>,
    pub normal_texture: Option<TextureHandle>,
    pub metallic_roughness_texture: Option<TextureHandle>,
    pub emissive_texture: Option<TextureHandle>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Vec3::splat(0.8),
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            emissive: Vec3::ZERO,
            albedo_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
        }
    }
}

impl Default for SceneLighting {
    fn default() -> Self {
        Self {
            ambient_color: Vec3::new(0.6, 0.7, 1.0),
            ambient_intensity: 0.15,
            sun: DirectionalLight {
                direction: Vec3::new(-0.4, -1.0, -0.6).normalize(),
                color: Vec3::new(1.0, 0.97, 0.92),
                intensity: 1.0,
            },
            point_lights: Vec::new(),
            shininess: 32.0,
            specular_strength: 0.4,
        }
    }
}

// The seam between the engine and a concrete graphics API: implement for other
// backends (wgpu, D3D12) without touching scene/app code.
pub trait RenderBackend {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle;
    fn load_material(&mut self, material: &Material) -> MaterialHandle;
    fn load_texture(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        srgb: bool,
    ) -> TextureHandle;
    fn resize(&mut self, extent: [u32; 2]);
    fn render(
        &mut self,
        items: &[RenderItem],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        hdr: &HdrSettings,
    );
}
