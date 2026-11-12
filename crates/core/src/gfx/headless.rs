//! A [`RenderBackend`] that uploads nothing.
//!
//! Startup is more than Vulkan: meshes are generated, textures are decoded and
//! synthesized, and the scene graph is built, all on the CPU and all before the
//! first frame. That half is what decays as dependencies accumulate, and it is
//! the half a machine with no GPU can still measure — which is what lets the
//! cold-start guard in `tests/cold_start.rs` run in CI.
//!
//! Handles are issued in upload order, exactly as a real backend does, so the
//! [`Assets`](crate::scene::Assets) table and [`MeshBounds`] a headless run
//! produces are indistinguishable from a live one's.

use crate::geom::Aabb;
use crate::gfx::surface::{
    Ignored, ParamBuffer, ParamTable, ParamValue, SurfaceArtifact, SurfaceShaderError,
    SurfaceShaderHandle, check_artifact,
};
use crate::gfx::vulkan::MAX_SURFACE_SHADERS;
use crate::gfx::{
    DecalInstance, DrawList, Material, RenderBackend, SceneLighting, TexelFormat, TextureHandle,
    TextureLevels,
};
use crate::scene::{
    BloomSettings, Camera, ContactShadowSettings, CpuMesh, DofSettings, EnvironmentSettings,
    FogSettings, HdrSettings, MaterialHandle, MeshHandle, MotionBlurSettings, RefractionSettings,
    SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings, TransparencySettings,
};

/// Counts uploads and derives mesh bounds; does no GPU work of any kind.
pub struct HeadlessBackend {
    mesh_bounds: Vec<Aabb>,
    /// Each material as last loaded or updated, by handle.
    materials: Vec<Material>,
    material_updates: u32,
    /// The texel of each 1x1 texture as last loaded or swapped in, and whether
    /// it was sRGB, by handle; `None` for any larger one.
    textures: Vec<Option<([u8; 4], bool)>>,
    mesh_replacements: u32,
    texture_replacements: u32,
    bc: bool,
    /// The format and level count of the last texture swapped in.
    last_texture: Option<(TexelFormat, usize)>,
    /// Surface shaders are bookkept exactly as the renderer bookkeeps them —
    /// tables, attachments, packing — with nothing built.
    surfaces: ParamBuffer,
    surface_uploads: u32,
}

impl Default for HeadlessBackend {
    fn default() -> Self {
        Self {
            mesh_bounds: Vec::new(),
            materials: Vec::new(),
            material_updates: 0,
            textures: Vec::new(),
            mesh_replacements: 0,
            texture_replacements: 0,
            bc: false,
            last_texture: None,
            surfaces: ParamBuffer::new(|texture| texture as u32),
            surface_uploads: 0,
        }
    }
}

impl HeadlessBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// One that reports BC support, as a desktop GPU does.
    pub fn with_bc() -> Self {
        Self {
            bc: true,
            ..Self::default()
        }
    }

    /// How many meshes, materials, and textures a run uploaded — enough for a
    /// benchmark to assert it measured a real scene rather than an empty one.
    pub fn upload_counts(&self) -> (usize, u32, u32) {
        (
            self.mesh_bounds.len(),
            self.materials.len() as u32,
            self.textures.len() as u32,
        )
    }

    /// `handle` as it was last loaded or updated.
    pub fn material(&self, handle: MaterialHandle) -> Option<Material> {
        self.materials.get(handle.0 as usize).copied()
    }

    /// The texel `handle` holds and whether it is sRGB, if it is 1x1.
    pub fn texel(&self, handle: TextureHandle) -> Option<([u8; 4], bool)> {
        self.textures.get(handle.0 as usize).copied().flatten()
    }

    /// How many in-place material edits reached this backend.
    pub fn material_updates(&self) -> u32 {
        self.material_updates
    }

    /// How many meshes were swapped in place, as a reimport does.
    pub fn mesh_replacements(&self) -> u32 {
        self.mesh_replacements
    }

    pub fn texture_replacements(&self) -> u32 {
        self.texture_replacements
    }

    /// The format and level count of the last texture swapped in.
    pub fn last_texture(&self) -> Option<(TexelFormat, usize)> {
        self.last_texture
    }

    /// The surface parameters as the renderer would upload them. A default
    /// texture packs as its [`DefaultTexture`](crate::gfx::surface::DefaultTexture)
    /// discriminant here.
    pub fn surface_buffer(&self) -> &ParamBuffer {
        &self.surfaces
    }

    /// How many frames uploaded the surface parameters, which the renderer
    /// does on exactly these frames.
    pub fn surface_uploads(&self) -> u32 {
        self.surface_uploads
    }
}

impl RenderBackend for HeadlessBackend {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle {
        let handle = MeshHandle(self.mesh_bounds.len() as u32);
        // Derived here for the same reason the real backend derives it at
        // upload: culling reads it out of the world, never out of a backend.
        self.mesh_bounds.push(mesh.bounds());
        handle
    }

    fn replace_mesh(&mut self, handle: MeshHandle, mesh: &CpuMesh) {
        if let Some(bounds) = self.mesh_bounds.get_mut(handle.0 as usize) {
            *bounds = mesh.bounds();
            self.mesh_replacements += 1;
        }
    }

    fn mesh_bounds(&self, mesh: MeshHandle) -> Option<Aabb> {
        self.mesh_bounds.get(mesh.0 as usize).copied()
    }

    fn load_material(&mut self, material: &Material) -> MaterialHandle {
        let handle = MaterialHandle(self.materials.len() as u32);
        self.materials.push(*material);
        handle
    }

    fn update_material(&mut self, handle: MaterialHandle, material: &Material) {
        if let Some(slot) = self.materials.get_mut(handle.0 as usize) {
            *slot = *material;
        }
        self.material_updates += 1;
    }

    fn load_texture(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        srgb: bool,
    ) -> TextureHandle {
        let handle = TextureHandle(self.textures.len() as u32);
        let texel = (width == 1 && height == 1)
            .then(|| Some((pixels.try_into().ok()?, srgb)))
            .flatten();
        self.textures.push(texel);
        handle
    }

    fn replace_texture(&mut self, handle: TextureHandle, texture: &TextureLevels) {
        if let Some(slot) = self.textures.get_mut(handle.0 as usize) {
            self.texture_replacements += 1;
            self.last_texture = Some((texture.format, texture.levels.len()));
            let srgb = match texture.format {
                TexelFormat::Rgba8Srgb => Some(true),
                TexelFormat::Rgba8Unorm => Some(false),
                _ => None,
            };
            *slot = (texture.width == 1 && texture.height == 1)
                .then(|| Some((texture.levels.first()?.as_slice().try_into().ok()?, srgb?)))
                .flatten();
        }
    }

    fn samples_bc(&self) -> bool {
        self.bc
    }

    fn load_environment(&mut self, _pixels: &[f32], _width: u32, _height: u32) {}

    fn load_surface_shader(
        &mut self,
        artifact: &SurfaceArtifact,
    ) -> Result<SurfaceShaderHandle, SurfaceShaderError> {
        check_artifact(artifact)?;
        if self.surfaces.loaded() >= MAX_SURFACE_SHADERS {
            return Err(SurfaceShaderError::TooMany);
        }
        Ok(self.surfaces.load(artifact.params.clone()))
    }

    fn replace_surface_shader(
        &mut self,
        shader: SurfaceShaderHandle,
        artifact: &SurfaceArtifact,
    ) -> Result<Vec<(MaterialHandle, Ignored)>, SurfaceShaderError> {
        if self.surfaces.table(shader).is_none() {
            return Err(SurfaceShaderError::UnknownShader(shader));
        }
        check_artifact(artifact)?;
        self.surfaces.replace(shader, artifact.params.clone())
    }

    fn set_material_surface(
        &mut self,
        material: MaterialHandle,
        surface: Option<(SurfaceShaderHandle, &[(String, ParamValue)])>,
    ) -> Result<Vec<Ignored>, SurfaceShaderError> {
        if material.0 as usize >= self.materials.len() {
            return Err(SurfaceShaderError::UnknownMaterial(material));
        }
        self.surfaces.attach(material, surface)
    }

    fn surface_params(&self, shader: SurfaceShaderHandle) -> Option<&ParamTable> {
        self.surfaces.table(shader)
    }

    fn resize(&mut self, _extent: [u32; 2]) {}

    fn render(
        &mut self,
        _draws: DrawList<'_>,
        _transparent: DrawList<'_>,
        _refractive: DrawList<'_>,
        _decals: &[DecalInstance],
        _lighting: &SceneLighting,
        _camera: &Camera,
        _ssao: &SsaoSettings,
        _contact_shadows: &ContactShadowSettings,
        _ssr: &SsrSettings,
        _subsurface: &SubsurfaceSettings,
        _transparency: &TransparencySettings,
        _refraction: &RefractionSettings,
        _taa: &TaaSettings,
        _motion_blur: &MotionBlurSettings,
        _dof: &DofSettings,
        _bloom: &BloomSettings,
        _hdr: &HdrSettings,
        _environment: &EnvironmentSettings,
        _fog: &FogSettings,
        _dt: f32,
    ) {
        if self.surfaces.loaded() > 0 && self.surfaces.take_changed().is_some() {
            self.surface_uploads += 1;
        }
    }
}
