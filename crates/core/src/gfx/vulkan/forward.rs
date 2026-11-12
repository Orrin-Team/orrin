use glam::Vec3;

use crate::geom::Aabb;
use crate::gfx::clusters::{CLUSTERS, DepthSlicing, LightClusters};
use crate::gfx::punctual::{LightKind, MAX_ATLAS_FACES, ShadowAtlas};
use crate::gfx::sh::SH9;
use crate::gfx::shadows::MAX_CASCADES;
use crate::gfx::{
    BlendMode, DecalInstance, MAX_DECALS, MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS, Material,
    PositionVertex, SceneLighting, SurfaceVertex,
};
use crate::scene::{Camera, EnvironmentSettings};

use super::cull::{Draws, Surface};
use super::subsurface::SUBSURFACE_FORMAT;
use super::surface::{Surfaces, Twin};
use super::swapchain::DEPTH_FORMAT;
use super::taa::FrameView;
use super::{MSAA_SAMPLES, ShadowFrame, VulkanRenderer};
use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::BufferUsage;
use crate::gfx::rhi::MeshSpan;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, Blend, BlockKind, BufferDesc, BufferId, CullMode,
    DepthCompare, DepthTest, Device, Encoder, GraphicsPipelineDesc, MemoryLocation, PipelineId,
    SamplerDesc, SamplerId, ShaderModuleId, ShaderRef, TextureViewId, Topology, Viewport,
};
use crate::gfx::shaders::ShaderId;

pub struct GpuMesh {
    /// Where this mesh's indices and vertices sit in the shared buffers every
    /// pass binds once. See [`mesh`](super::mesh).
    pub span: MeshSpan,
    /// Object-space bounds, derived at upload because that is the last place the
    /// vertex data exists on the CPU. Culling reads them through
    /// [`RenderBackend::mesh_bounds`](crate::gfx::RenderBackend::mesh_bounds).
    pub bounds: Aabb,
}

/// Where `forward.frag` declares the cascade comparison sampler. It is bound
/// immutably at pipeline-layout construction, so these have to match the shader
/// by hand rather than being derived from it.
///
/// Shared with the refraction pass, which builds a layout of its own — the
/// immutable sampler is part of the set layout, so a pipeline that patched it
/// differently would no longer be set-compatible with this one and could not be
/// handed the same five descriptor sets.
pub(super) const SHADOW_SET: usize = 3;
pub(super) const SHADOW_SAMPLER_BINDING: u32 = 2;

/// Where `shading.glsl` declares the material texture array, and by hand for the
/// same reason [`SHADOW_SET`] is: what is patched into the layout after
/// reflection cannot be derived from it. Binding 0 of the set, which is what
/// [`VkContext::mark_texture_array_partial`] assumes.
pub(super) const TEXTURE_SET: usize = 2;

/// Default texture indices, matching the order `VulkanRenderer::new` seeds them.
const WHITE_TEXTURE: u32 = 0;
const FLAT_NORMAL_TEXTURE: u32 = 1;

/// Feature bits in [`GpuMaterial::flags`], mirrored by `shaders/material.glsl`.
///
/// The whole point of the word: a draw is one batch and a batch is one
/// (mesh, material) pair, so every invocation of a draw reads the same flags —
/// a draw either takes a lobe's branch or does not, and the cost of a feature a
/// material never asked for is one coherent test. Set from whether the block was
/// actually authored rather than from a separate toggle, so a material cannot
/// claim a lobe and supply nothing to it.
pub(crate) mod material_flags {
    pub const CLEARCOAT: u32 = 1 << 0;
    pub const SHEEN: u32 = 1 << 1;
    pub const ANISOTROPY: u32 = 1 << 2;
    pub const TRANSMISSION: u32 = 1 << 3;
    pub const SUBSURFACE: u32 = 1 << 4;
    pub const PARALLAX: u32 = 1 << 5;
    /// Unlike every bit above it, this one selects a *pipeline* as well as a
    /// branch: the passes read it off `BlendMode` on the CPU to choose between
    /// the plain and the alpha-testing variant. It rides here too so the shader
    /// can ask the same question about the pixel it is on — the 1-sample passes
    /// need it to know whether to `discard`.
    pub const MASKED: u32 = 1 << 6;
    /// Every optional lobe: what a material must use none of to draw with the
    /// flat pipeline, whose shader is compiled without them. See
    /// `ForwardPipelines`.
    pub const LOBES: u32 = CLEARCOAT | SHEEN | ANISOTROPY | TRANSMISSION | SUBSURFACE | PARALLAX;
}

/// Ceiling on a material's parallax step counts. Keep in sync with
/// `PARALLAX_STEP_LIMIT` in `shaders/parallax.glsl`, which is what actually
/// bounds the loop: a material asking for more would pay for the extra
/// iterations in register pressure and get none of them.
const PARALLAX_STEP_LIMIT: u32 = 64;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub(crate) struct GpuMaterial {
    /// `rgb` = albedo, `a` = opacity.
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4], // metallic, roughness, reflectance, ior
    /// Indices into the set-2 texture array: [albedo, normal, metal-rough, emissive].
    tex_indices: [u32; 4],
    clearcoat: [f32; 4],    // strength, roughness
    sheen: [f32; 4],        // rgb = colour, a = roughness
    anisotropy: [f32; 4],   // strength, cos(rotation), sin(rotation)
    transmission: [f32; 4], // transmission, thickness, attenuation distance
    /// `rgb` = what the volume absorbs over `transmission[2]`.
    attenuation: [f32; 4],
    subsurface: [f32; 4], // rgb = scattering tint, a = forward-scatter power
    /// `rgb` = per-channel mean free path in metres. Separate from the block
    /// above because the shader wants the tint and the distances at different
    /// points — the tint weights what comes back, the distances decide how far
    /// across the image it is allowed to come back from.
    subsurface_radius: [f32; 4],
    /// Depth of the height field in metres, then the step counts the march is
    /// allowed head-on and edge-on.
    parallax: [f32; 4],
    /// `x` = the alpha a `Masked` fragment must reach to survive.
    ///
    /// A block of its own rather than a spare lane in one of the ones above, for
    /// the reason every block here has one: what a cutout is has nothing to do
    /// with a height field or a coat, and a scalar parked in a neighbour's
    /// padding is a comment away from being lost. Sixteen bytes a material.
    alpha: [f32; 4],
    /// [clearcoat, clearcoat normal, sheen, anisotropy].
    tex_indices_ext: [u32; 4],
    /// [transmission, feature flags, subsurface, height]. The flags ride here
    /// rather than in a float field so the shader can test them without
    /// `floatBitsToUint`.
    tex_flags: [u32; 4],
    /// `x` = the first word of this material's parameters in a surface shader's
    /// parameter buffer (`gfx::surface`). Only a custom surface reads it.
    surface: [u32; 4],
}

impl GpuMaterial {
    /// Whether a run drawing with this material needs the alpha-testing
    /// pipeline.
    ///
    /// Asked of the *uploaded* flag word rather than of a second CPU-side table,
    /// so the pipeline a draw is recorded with and the branch its shader takes
    /// are reading one bit. A `MaterialBlends` that disagreed with the material
    /// it was registered beside could then only cost a queue, never a pipeline.
    pub(super) fn is_masked(&self) -> bool {
        self.tex_flags[1] & material_flags::MASKED != 0
    }

    /// Whether any optional lobe is on, which is what sends a material to the
    /// full forward pipeline rather than the flat one. Read off the uploaded
    /// flag word for the reason `is_masked` is.
    pub(super) fn is_lobed(&self) -> bool {
        self.tex_flags[1] & material_flags::LOBES != 0
    }

    /// Point the record at its block in the surface parameters. Returns
    /// whether that moved it.
    pub(super) fn set_surface_base(&mut self, base: u32) -> bool {
        std::mem::replace(&mut self.surface[0], base) != base
    }

    /// The class a batch of this material draws in when no surface shader
    /// reads it.
    pub(super) fn built_in_variant(&self) -> super::cull::Variant {
        super::cull::Variant {
            masked: self.is_masked(),
            surface: if self.is_lobed() {
                super::cull::Surface::Lobed
            } else {
                super::cull::Surface::Flat
            },
        }
    }
}

/// Pack the engine's [`SceneLighting`] into the std140 layout the shader expects.
/// The point lights as the shader's storage buffer holds them, into `out`.
/// Never empty: a buffer binding of zero bytes is invalid, and a frame with no
/// lights still binds one, whose row no cluster ever names.
fn gpu_point_lights(lighting: &SceneLighting, atlas: &ShadowAtlas, out: &mut Vec<GpuPointLight>) {
    out.clear();
    for (index, light) in lighting
        .point_lights
        .iter()
        .take(MAX_POINT_LIGHTS)
        .enumerate()
    {
        out.push(GpuPointLight {
            position: [
                light.position.x,
                light.position.y,
                light.position.z,
                light.range.max(1e-4),
            ],
            color: [light.color.x, light.color.y, light.color.z, light.candela],
            // A light that asked for tiles and did not get them keeps its -1 and
            // shades unshadowed, which is the whole behaviour of the budget: too
            // many casters costs shadows, never correctness.
            shadow: match atlas.caster(LightKind::Point, index) {
                Some(caster) => [caster.first_face as f32, caster.near, 0.0, 0.0],
                None => [-1.0, 0.0, 0.0, 0.0],
            },
        });
    }
    if out.is_empty() {
        out.push(GpuPointLight::ZERO);
    }
}

/// [`gpu_point_lights`] for the spots.
fn gpu_spot_lights(lighting: &SceneLighting, atlas: &ShadowAtlas, out: &mut Vec<GpuSpotLight>) {
    out.clear();
    for (index, light) in lighting
        .spot_lights
        .iter()
        .take(MAX_SPOT_LIGHTS)
        .enumerate()
    {
        let face = atlas
            .caster(LightKind::Spot, index)
            .map_or([-1.0, 0.0], |caster| {
                [caster.first_face as f32, caster.near]
            });
        out.push(GpuSpotLight {
            position: [
                light.position.x,
                light.position.y,
                light.position.z,
                light.range.max(1e-4),
            ],
            direction: [
                light.direction.x,
                light.direction.y,
                light.direction.z,
                light.outer_cos,
            ],
            color: [light.color.x, light.color.y, light.color.z, light.candela],
            params: [light.inner_cos, face[0], face[1], 0.0],
        });
    }
    if out.is_empty() {
        out.push(GpuSpotLight::ZERO);
    }
}

/// The cluster lists as the shader's `LightClusters` block reads them: the
/// fixed array of ranges, then the indices — at least one, for the reason the
/// light buffers are never empty.
fn gpu_clusters(clusters: &LightClusters, out: &mut Vec<u32>) {
    out.clear();
    if clusters.ranges.len() == CLUSTERS {
        out.extend(clusters.ranges.iter().flatten());
    } else {
        out.resize(CLUSTERS * 4, 0);
    }
    out.extend(&clusters.indices);
    out.push(0);
}

#[allow(clippy::too_many_arguments)]
fn to_gpu_lighting(
    lighting: &SceneLighting,
    camera: &Camera,
    extent: [u32; 2],
    shadows: Option<ShadowFrame<'_>>,
    irradiance: [Vec3; SH9],
    environment_yaw: f32,
    env_specular: Vec3,
    surface_time: f32,
) -> GpuLighting {
    let camera_pos = camera.position;
    let (w, h) = (extent[0] as f32, extent[1] as f32);
    let count = lighting.point_lights.len().min(MAX_POINT_LIGHTS);
    let spot_count = lighting.spot_lights.len().min(MAX_SPOT_LIGHTS);
    let slicing = DepthSlicing::new(camera.near, camera.far);
    let forward = (camera.target - camera.position).normalize_or(Vec3::NEG_Z);

    // The shader wants the direction *toward* the light, so negate.
    let to_sun = (-lighting.sun.direction).normalize_or_zero();

    GpuLighting {
        camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
        ambient: [
            lighting.ambient_color.x,
            lighting.ambient_color.y,
            lighting.ambient_color.z,
            lighting.ambient_nits,
        ],
        sun_direction: [to_sun.x, to_sun.y, to_sun.z, 0.0],
        sun_color: [
            lighting.sun.color.x,
            lighting.sun.color.y,
            lighting.sun.color.z,
            lighting.sun.illuminance,
        ],
        params: [
            count as f32,
            surface_time,
            lighting.specular_strength,
            spot_count as f32,
        ],
        viewport: [w, h, 1.0 / w, 1.0 / h],
        cascades: GpuCascades::new(shadows),
        clusters: [slicing.scale, slicing.bias, 0.0, 0.0],
        view_forward: [forward.x, forward.y, forward.z, 0.0],
        environment: {
            let (sin, cos) = environment_yaw.to_radians().sin_cos();
            [sin, cos, 0.0, 0.0]
        },
        env_specular: [env_specular.x, env_specular.y, env_specular.z, 0.0],
        irradiance: irradiance.map(|c| [c.x, c.y, c.z, 0.0]),
    }
}

/// Everything it takes to read the cascaded shadow maps, mirroring `Cascades` in
/// `shaders/cascades.glsl`.
///
/// Its own struct because two uniform blocks carry it: the lighting block every
/// surface shades from, and the fog block the froxel scatter pass dispatches
/// with. A dispatch has no lighting set bound and the alternative was declaring
/// the whole `Lighting` block — point lights, spot lights, irradiance — in a
/// shader that wants four fields of it. One constructor fills both, so the two
/// copies are made together or not at all.
///
/// std140 gives a struct of a `mat4` array and three `vec4`s exactly the offsets
/// it gave these fields written out flat, which is why nesting them changed no
/// layout.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub(super) struct GpuCascades {
    /// Per-cascade light view-projection. std140 lays a `mat4` out as four
    /// `vec4`s with no padding between them, which is exactly what this is.
    view_proj: [[[f32; 4]; 4]; MAX_CASCADES],
    /// Split distances, as radial distance from the camera.
    splits: [f32; MAX_CASCADES],
    /// World size of one shadow texel in each cascade, for the normal-offset
    /// bias. It differs per cascade because each fits a different-sized box to
    /// the same number of texels.
    texel_sizes: [f32; MAX_CASCADES],
    /// x = cascade count, y = blend overlap fraction, z = strength,
    /// w = 1.0 to tint by cascade index.
    params: [f32; 4],
}

impl GpuCascades {
    pub(super) fn new(shadows: Option<ShadowFrame<'_>>) -> Self {
        let mut view_proj = [[[0.0f32; 4]; 4]; MAX_CASCADES];
        let mut splits = [0.0f32; MAX_CASCADES];
        let mut texel_sizes = [0.0f32; MAX_CASCADES];
        // A zero count is what makes every shadow lookup return "lit"; the
        // arrays above are then never indexed.
        let params = match shadows {
            Some(shadows) => {
                let live = &shadows.cascades.cascades[..shadows.cascades.count];
                for (slot, cascade) in view_proj.iter_mut().zip(live) {
                    *slot = cascade.view_proj.to_cols_array_2d();
                }
                for (index, cascade) in live.iter().enumerate() {
                    splits[index] = cascade.split_distance;
                    texel_sizes[index] = cascade.texel_world_size;
                }
                [
                    shadows.cascades.count as f32,
                    crate::gfx::shadows::OVERLAP,
                    shadows.settings.strength,
                    if shadows.settings.debug_cascades {
                        1.0
                    } else {
                        0.0
                    },
                ]
            }
            None => [0.0; 4],
        };

        Self {
            view_proj,
            splits,
            texel_sizes,
            params,
        }
    }
}

/// GPU mirror of a [`PointLight`](crate::gfx::PointLight): three `vec4`s, so
/// std430 lays the shader's array out exactly like a slice of these.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct GpuPointLight {
    /// xyz = world position, w = range.
    position: [f32; 4],
    /// rgb = color, w = luminous intensity in candela.
    color: [f32; 4],
    /// x = index of this light's first atlas face, negative for a light the
    /// atlas had no room for; y = the near plane its faces were rendered with,
    /// which the shader needs to bias the comparison against.
    shadow: [f32; 4],
}

impl GpuPointLight {
    const ZERO: Self = Self {
        position: [0.0; 4],
        color: [0.0; 4],
        shadow: [-1.0, 0.0, 0.0, 0.0],
    };
}

/// GPU mirror of a [`SpotLight`](crate::gfx::SpotLight): four `vec4`s, as std430
/// wants them.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct GpuSpotLight {
    /// xyz = world position, w = range.
    position: [f32; 4],
    /// xyz = cone axis, w = cosine of the outer half angle.
    direction: [f32; 4],
    /// rgb = color, w = luminous intensity in candela, the reflector already
    /// divided out.
    color: [f32; 4],
    /// x = cosine of the inner half angle, y = atlas face index (negative for
    /// none), z = near plane.
    params: [f32; 4],
}

impl GpuSpotLight {
    const ZERO: Self = Self {
        position: [0.0; 4],
        // A `w` of 1.0 is a cone of zero width, so an unused slot lights
        // nothing even if the count were ever wrong.
        direction: [0.0, 0.0, -1.0, 1.0],
        color: [0.0; 4],
        params: [1.0, -1.0, 0.0, 0.0],
    };
}

/// GPU layout of the per-frame lighting uniform buffer (set 0, binding 0).
/// Every field is a `vec4` so the Rust `#[repr(C)]` layout matches std140 with
/// no hidden padding.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct GpuLighting {
    /// xyz = camera world position.
    camera_pos: [f32; 4],
    /// rgb = ambient color, w = ambient luminance in cd/m².
    ambient: [f32; 4],
    /// xyz = normalized direction toward the sun.
    sun_direction: [f32; 4],
    /// rgb = sun color, w = illuminance in lux on a surface facing it.
    sun_color: [f32; 4],
    /// x = point light count, y = what a surface shader's `i.time` reads,
    /// z = specular strength, w = spot light count. Only `y` is read.
    params: [f32; 4],
    /// x=w, y=h, z=1/w, w=1/h
    viewport: [f32; 4],
    cascades: GpuCascades,
    /// x = the depth slicing's scale, y = its bias. See `gfx::clusters`.
    clusters: [f32; 4],
    /// xyz = the camera's forward axis, which a fragment's view depth — and so
    /// its cluster's slice — is measured along.
    view_forward: [f32; 4],
    /// x = sin(environment yaw), y = cos(environment yaw). The same rotation
    /// the skybox samples through, so the sky and what it lights agree.
    environment: [f32; 4],
    /// rgb = what sampled environment radiance is multiplied by. Carries the
    /// scene's flat ambient when no environment is loaded, which is what makes
    /// the 1x1 white fallback cube behave as a uniform environment.
    env_specular: [f32; 4],
    /// Diffuse irradiance as nine spherical-harmonic coefficients, already
    /// convolved with the cosine lobe and divided by pi — see `gfx::sh`. `vec4`
    /// rather than `vec3` because std140 pads an array element to 16 bytes
    /// either way, so the padding may as well be visible on both sides.
    irradiance: [[f32; 4]; SH9],
}

/// Feature bits in [`GpuDecal::tex`]`[3]`, mirrored by `shaders/decals.glsl`.
///
/// Both say whether a block was *authored*, exactly as `material_flags` does,
/// and both exist because the neutral value is not available: a decal with no
/// normal map would decode the flat-normal default to its own projection axis
/// and tilt every surface it touched to face the projector, and there is no
/// roughness that means "leave what was there".
pub(crate) mod decal_flags {
    pub const NORMAL: u32 = 1 << 0;
    pub const SURFACE: u32 = 1 << 1;
}

/// One decal as the shader reads it. 160 bytes; std430 packs this exactly like
/// the `#[repr(C)]` here because every field is 16 bytes or a `mat4`.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub(crate) struct GpuDecal {
    world_to_decal: [[f32; 4]; 4],
    /// `rgb` = tint, `a` = opacity.
    base_color: [f32; 4],
    /// metallic, roughness, normal strength, cos(fade angle).
    params: [f32; 4],
    /// [albedo, normal, metal-rough, feature flags]. The flags ride in a `uint`
    /// lane for the reason `GpuMaterial`'s do.
    tex: [u32; 4],
    /// The decal's own axes in world space: `u`, `v` and the projection axis.
    /// Three `vec4`s rather than a `mat3`, because std430 lays a `mat3` out as
    /// three `vec4`s anyway and this way both sides can see that it does.
    axis_u: [f32; 4],
    axis_v: [f32; 4],
    axis_n: [f32; 4],
}

impl GpuDecal {
    /// A slot no fragment will ever be inside: the identity puts the box at the
    /// origin, and a zero opacity would weight it away even there. Written into
    /// the tail of the block so nothing is left undefined, though the count is
    /// what actually stops the loop.
    const ZERO: Self = Self {
        world_to_decal: [[0.0; 4]; 4],
        base_color: [0.0; 4],
        params: [0.0; 4],
        tex: [0; 4],
        axis_u: [0.0; 4],
        axis_v: [0.0; 4],
        axis_n: [0.0; 4],
    };
}

/// The whole decal block: how many are live, then a fixed array of them.
///
/// One buffer with the count inside it rather than a count passed alongside,
/// and that is the point. The forward pass reads its frame data out of the
/// lighting uniform and the prepass reads its own out of `FrameUbo`; a count
/// added to both would be two places to keep in step, and a frame where they
/// disagreed would light a decal the prepass had not written a normal for. Here
/// there is one array and one length, in one allocation, bound to both.
///
/// Fixed-length rather than a runtime-sized array, which costs 2.5 KB a frame
/// whether the scene has decals or not and buys one buffer that both passes
/// can bind with no length to agree on either.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub(crate) struct GpuDecals {
    /// `x` = how many of `decals` are live. A `uvec4` because std430 aligns the
    /// struct that follows it to sixteen bytes regardless, so the padding may as
    /// well be visible on both sides.
    count: [u32; 4],
    decals: [GpuDecal; MAX_DECALS],
}

/// One atlas face as the shader reads it: the matrix that rendered it, and the
/// slice of the atlas it landed in.
///
/// The matrix is stored rather than rebuilt from the light's position and a face
/// convention, which the `vector_to_depth` trick would allow. Storing it means
/// the shader projects with *the same* matrix the tile was rasterised with, so
/// the two cannot disagree about a near plane, a border, or a handedness — the
/// only thing left to get right is which face, and that is one comparison.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct GpuShadowFace {
    view_proj: [[f32; 4]; 4],
    /// xy = atlas UV offset, zw = atlas UV scale.
    rect: [f32; 4],
}

impl GpuShadowFace {
    const ZERO: Self = Self {
        view_proj: [[0.0; 4]; 4],
        rect: [0.0; 4],
    };
}

/// What set 0 binds.
struct LightingSet {
    lighting: BufferId,
    decals: BufferId,
    points: BufferId,
    spots: BufferId,
    clusters: BufferId,
}

/// What set 3 binds.
struct SurfaceSet {
    ao: TextureViewId,
    shadow: TextureViewId,
    env_cube: TextureViewId,
    env_sampler: SamplerId,
    contact_shadow: TextureViewId,
    atlas: TextureViewId,
    shadow_faces: BufferId,
    fog_volume: TextureViewId,
    fog_sampler: SamplerId,
    fog: BufferId,
}

/// The views a frame binds when a term has nothing to say — white occlusion,
/// lit shadows, still air — and the environment, which the preview does want.
pub(super) struct NeutralViews {
    pub ao: TextureViewId,
    pub contact_shadow: TextureViewId,
    pub shadow: TextureViewId,
    pub atlas: TextureViewId,
    pub env_cube: TextureViewId,
    pub env_sampler: SamplerId,
    pub fog_volume: TextureViewId,
    pub fog_sampler: SamplerId,
}

/// The blocks a frame binds when there is nothing to light with — no punctual
/// light, decal, cluster, shadow face or fog — which the material preview
/// binds on every redraw and therefore holds rather than uploads.
pub(super) struct NeutralBlocks {
    decals: BufferId,
    points: BufferId,
    spots: BufferId,
    clusters: BufferId,
    shadow_faces: BufferId,
    fog: BufferId,
}

/// The lighting block the material preview shades with. Kept as the block so
/// what decides whether the preview is redrawn is exactly what it is drawn
/// with — less a surface shader's time, which is the frame's it is redrawn in.
pub(super) struct PreviewLighting(GpuLighting);

impl PreviewLighting {
    pub(super) fn new(
        lighting: &SceneLighting,
        camera: &Camera,
        extent: [u32; 2],
        irradiance: [Vec3; SH9],
        environment_yaw: f32,
        env_specular: Vec3,
    ) -> Self {
        Self(to_gpu_lighting(
            lighting,
            camera,
            extent,
            None,
            irradiance,
            environment_yaw,
            env_specular,
            0.0,
        ))
    }

    pub(super) fn bytes(&self) -> &[u8] {
        bytemuck::bytes_of(&self.0)
    }
}

/// The five descriptor sets a pass shading into the lit frame binds, in bind
/// order.
///
/// Both such passes bind the same five — the transparency pass draws through
/// this pass's own pipeline layout, so it can and must. Bundling them is what
/// keeps that from being a comment somebody has to keep true.
pub(super) struct ForwardSets {
    groups: [BindGroupId; 5],
}

impl ForwardSets {
    pub(super) fn groups(&self) -> [BindGroupId; 5] {
        self.groups
    }

    /// The same five with the object set swapped, for a pass that indexes
    /// through a different instance buffer.
    ///
    /// Under GPU culling the opaque queue reads what the dispatch compacted and
    /// the blended queues read what the host wrote, and the two are different
    /// allocations. Rebuilding the other four sets to say so would be four
    /// descriptor writes a frame for a difference in one of them.
    pub(super) fn with_object_set(&self, object_set: BindGroupId) -> ForwardSets {
        let mut groups = self.groups;
        groups[4] = object_set;
        ForwardSets { groups }
    }
}

/// The four opaque pipelines one forward render pass needs.
///
/// Two axes, and neither can be collapsed. `subsurface` is a property of the
/// *frame* — it decides which render pass is open and therefore how many colour
/// attachments a pipeline must declare — while `masked` is a property of a
/// *run*, so both of those have to exist at once whichever frame it is.
/// Every forward pipeline one sample count needs, in the four built-in variants
/// a batch can be (see `cull::Variant`), each with and without the subsurface
/// target.
///
/// The `flat` ones run `shading.glsl` compiled with `ORRIN_FLAT_SURFACE`, which
/// takes the optional lobes out — clear coat, sheen, anisotropy, transmission,
/// subsurface, parallax — for the materials that use none of them, which is
/// most. Not to save the lobes' work: behind their flag tests that is only a
/// branch, and measured at ~0.025 ms. It is that their code being there at all
/// cost every material a quarter of the pass (forward 0.77 -> 0.58 ms on the
/// demo, M5 Pro, with the lobes compiled out), as register pressure and the
/// occupancy it costs. One source still, as a define, so the two cannot drift.
///
/// These are the built-in surfaces' classes only. A surface shader is a class
/// per mask of its own (`cull::Surface::Custom`), always compiled against every
/// lobe, and drawn by its twins of the lobed pipelines here, which
/// `ForwardPass::build_twin` makes on first use and `surface` keeps — so the
/// pipelines that exist for the whole session are still these sixteen.
struct ForwardPipelines {
    plain: PipelineId,
    subsurface: PipelineId,
    /// Alpha-testing, for the `Masked` runs inside the same render pass. Back
    /// faces are kept, because a cutout sheet has no back to cull, and the cut
    /// itself is taken by whichever mechanism the sample count affords: alpha to
    /// coverage across four samples, a `discard` at one.
    masked: PipelineId,
    masked_subsurface: PipelineId,
    flat: PipelineId,
    flat_subsurface: PipelineId,
    masked_flat: PipelineId,
    masked_flat_subsurface: PipelineId,
}

impl ForwardPipelines {
    /// The pipeline a batch of `variant` draws with in a frame that does or
    /// does not diffuse subsurface light. A surface shader's batches draw with
    /// the lobed one when it has no twin to draw them, which is the standard
    /// material it was compiled against.
    fn for_variant(&self, variant: super::cull::Variant, subsurface: bool) -> PipelineId {
        let lobed = variant.surface != super::cull::Surface::Flat;
        match (variant.masked, lobed, subsurface) {
            (false, false, false) => self.flat,
            (false, false, true) => self.flat_subsurface,
            (false, true, false) => self.plain,
            (false, true, true) => self.subsurface,
            (true, false, false) => self.masked_flat,
            (true, false, true) => self.masked_flat_subsurface,
            (true, true, false) => self.masked,
            (true, true, true) => self.masked_subsurface,
        }
    }
}

/// The four attachment shapes the lit frame is rasterised into.
///
/// Two independent questions, and they are independent: does this frame resolve
/// multisamples, and does it write a second target for subsurface diffusion.
///
/// - **Resolving** changes the colour format as well as the sample count. The
///   multisampled pair writes `msaa_color_format` and resolves into `hdr_color`,
///   which the graph declares at that same wider format precisely so the resolve
///   is legal (`frame.rs`); the one-sample pair writes `hdr_color` at the
///   narrower packed format directly.
/// - **Diffusing** appends `SUBSURFACE_FORMAT` as a second colour target. The
///   skybox and the debug lines rasterise into these same shapes and write
///   nothing to that attachment, which is what `GraphicsPipelineDesc::mask_extra_attachments` is for.
///
/// The depth format is the same in all four; what differs is what the pass does
/// with it, and that is now a record-time decision rather than a property baked
/// into a pass object. At more than one sample the pass rasterises its own depth
/// and discards it; at one sample it borrows the geometry prepass's read-only
/// and tests `EQUAL`. See [`rendering`](super::rendering).
#[derive(Clone)]
pub struct ForwardTargets {
    color_format: Format,
    msaa_color_format: Format,
}

impl ForwardTargets {
    /// The colour formats of one of the four shapes, for a pass that builds its
    /// pipeline through the RHI rather than from the description above.
    pub fn colors(&self, msaa: bool, subsurface: bool) -> Vec<Format> {
        let first = if msaa {
            self.msaa_color_format
        } else {
            self.color_format
        };
        if subsurface {
            vec![first, SUBSURFACE_FORMAT]
        } else {
            vec![first]
        }
    }

    fn new(color_format: Format, msaa_color_format: Format) -> Self {
        Self {
            color_format,
            msaa_color_format,
        }
    }
}

/// The light and cluster rows, rebuilt every frame into buffers kept for their
/// allocations. Behind a lock only because `begin_frame` takes `&self`; it is
/// taken once a frame, on the thread that records it.
#[derive(Default)]
struct LightScratch {
    points: Vec<GpuPointLight>,
    spots: Vec<GpuSpotLight>,
    clusters: Vec<u32>,
}

pub struct ForwardPass {
    /// The four shapes this pass rasterises into, and the only thing a pipeline
    /// needs to know about any of them.
    ///
    /// Four render pass objects until dynamic rendering: two written by
    /// `single_pass_renderpass!` and two built by hand, because the macro could
    /// not express a depth attachment referenced in
    /// `DepthStencilReadOnlyOptimal`. They are four *descriptions* now, all of
    /// them made the same way, and the read-only depth is a field on the
    /// attachment at record time rather than a property of a pass object — see
    /// [`rendering`](super::rendering).
    ///
    /// Still four, and still all made up front: which one a frame uses comes out
    /// of `FrameConfig`, and the skybox and the debug lines rasterise alongside
    /// this pass, so all three modules build one pipeline per shape at startup.
    pub targets: ForwardTargets,
    /// Indexed by frame: `multisampled` when `FrameConfig::msaa` is on, `single`
    /// otherwise. Eight pipelines at startup rather than four — see
    /// [`ForwardPipelines`] for the two axes, and `single_render_pass` for the
    /// third. All eight are built from the same shader modules and the same
    /// immutable shadow sampler, so they are set compatible and the executor's
    /// five descriptor sets bind to any of them. A surface shader's twins of
    /// them are built by [`build_twin`](Self::build_twin) from the same
    /// [`pipeline_desc`] and kept with the shader, in `surface`.
    multisampled: ForwardPipelines,
    single: ForwardPipelines,
    /// Per-frame storage for the decal block. Owned here rather than by a decal
    /// module because there is no decal *pass* to own it — the block is frame
    /// data that two existing passes read, which is exactly what the shadow face
    /// table above is.
    sampler: SamplerId,
    ao_sampler: SamplerId,
    light_scratch: std::sync::Mutex<LightScratch>,
}

impl ForwardPass {
    /// `color_format` is what the one-sample pair writes — the frame's packed
    /// colour format. `msaa_color_format` is the wider one the multisampled pair
    /// needs: a resolve requires both images to agree, and alpha to coverage
    /// needs an alpha channel to read. See `hdr::HDR_WIDE_FORMAT`.
    pub fn new(rhi: &Device, color_format: Format, msaa_color_format: Format) -> Self {
        let targets = ForwardTargets::new(color_format, msaa_color_format);

        // One sampler for all eight pipelines. Vulkan compares immutable samplers
        // by *identity*, so two calls producing identical samplers would give two
        // set-3 layouts that are not compatible. The executor binds one set of
        // five descriptor sets for whichever opaque variant ran, so every
        // pipeline has to be built around the same sampler object — the same rule
        // `refraction.rs` obeys by lifting this layout instead of deriving it.
        let shadow_sampler = rhi.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: false,
            repeat: false,
            compare: true,
            anisotropy: None,
            max_lod: None,
        });
        let samplers = [(SHADOW_SET as u32, SHADOW_SAMPLER_BINDING, shadow_sampler)];
        let pipeline = |colors: &[Format], fragment: ShaderId, cull_back: bool, samples| {
            rhi.create_graphics_pipeline(&pipeline_desc(
                colors,
                fragment.into(),
                cull_back,
                samples,
                &samplers,
            ))
        };
        let msaa = MSAA_SAMPLES;
        let one = SampleCount::Sample1;
        let sss = |format: Format| [format, SUBSURFACE_FORMAT];
        let multisampled = ForwardPipelines {
            plain: pipeline(&[msaa_color_format], ShaderId::ForwardFrag, true, msaa),
            subsurface: pipeline(
                &sss(msaa_color_format),
                ShaderId::ForwardSssFrag,
                true,
                msaa,
            ),
            masked: pipeline(&[msaa_color_format], ShaderId::ForwardFrag, false, msaa),
            masked_subsurface: pipeline(
                &sss(msaa_color_format),
                ShaderId::ForwardSssFrag,
                false,
                msaa,
            ),
            flat: pipeline(&[msaa_color_format], ShaderId::ForwardFragFlat, true, msaa),
            flat_subsurface: pipeline(
                &sss(msaa_color_format),
                ShaderId::ForwardSssFragFlat,
                true,
                msaa,
            ),
            masked_flat: pipeline(&[msaa_color_format], ShaderId::ForwardFragFlat, false, msaa),
            masked_flat_subsurface: pipeline(
                &sss(msaa_color_format),
                ShaderId::ForwardSssFragFlat,
                false,
                msaa,
            ),
        };
        // The masked variants take the alpha-testing shaders rather than the
        // plain ones: there is no coverage to spend at one sample, so the cut has
        // to be a `discard`. Everything else about them is unchanged.
        let single = ForwardPipelines {
            plain: pipeline(&[color_format], ShaderId::ForwardFrag, true, one),
            subsurface: pipeline(&sss(color_format), ShaderId::ForwardSssFrag, true, one),
            masked: pipeline(&[color_format], ShaderId::ForwardFragAlphaTest, false, one),
            masked_subsurface: pipeline(
                &sss(color_format),
                ShaderId::ForwardSssFragAlphaTest,
                false,
                one,
            ),
            flat: pipeline(&[color_format], ShaderId::ForwardFragFlat, true, one),
            flat_subsurface: pipeline(&sss(color_format), ShaderId::ForwardSssFragFlat, true, one),
            masked_flat: pipeline(
                &[color_format],
                ShaderId::ForwardFragAlphaTestFlat,
                false,
                one,
            ),
            masked_flat_subsurface: pipeline(
                &sss(color_format),
                ShaderId::ForwardSssFragAlphaTestFlat,
                false,
                one,
            ),
        };

        // Material textures are the only ones with a mip chain to sample; the
        // AO and tonemap inputs are screen-space targets read at 1:1.
        let sampler = rhi.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: true,
            repeat: true,
            compare: false,
            anisotropy: rhi.max_anisotropy(),
            max_lod: None,
        });

        let ao_sampler = rhi.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(1.0),
        });

        Self {
            targets,
            multisampled,
            single,
            sampler,
            ao_sampler,
            light_scratch: std::sync::Mutex::new(LightScratch::default()),
        }
    }

    /// The quartet a frame draws with, given whether it rasterises multisampled.
    fn pipelines(&self, msaa: bool) -> &ForwardPipelines {
        if msaa {
            &self.multisampled
        } else {
            &self.single
        }
    }

    /// A surface shader's twin of the pipeline a `masked` run draws with in
    /// this attachment shape: [`pipeline_desc`] with `fragment` in it, and the
    /// built-in's five sets lifted rather than derived, for the reason the
    /// refraction pass lifts them — set 3 holds the shadow sampler immutably.
    pub(super) fn build_twin(
        &self,
        device: &Device,
        msaa: bool,
        subsurface: bool,
        masked: bool,
        fragment: ShaderModuleId,
    ) -> Result<PipelineId, String> {
        let colors = self.targets.colors(msaa, subsurface);
        let built_in = self.pipelines(msaa).for_variant(
            super::cull::Variant {
                masked,
                surface: super::cull::Surface::Lobed,
            },
            subsurface,
        );
        device.try_create_graphics_pipeline(&GraphicsPipelineDesc {
            layout_prefix_from: Some((built_in, 5)),
            ..pipeline_desc(
                &colors,
                fragment.into(),
                !masked,
                if msaa {
                    MSAA_SAMPLES
                } else {
                    SampleCount::Sample1
                },
                &[],
            )
        })
    }

    /// The layout the transparency pass builds its own pipeline with. Shared
    /// rather than derived a second time, so the two pipelines cannot disagree
    /// about a binding — see `oit.rs`.
    /// The pipeline whose layout the blended-geometry passes are handed the
    /// descriptor sets of.
    pub(super) fn layout_source(&self) -> PipelineId {
        self.single.plain
    }

    /// The set-4 per-object descriptor set for this frame's object buffer.
    ///
    /// Built once per frame by the executor rather than once per pass: the
    /// buffer changes every frame, so nothing here can be cached across frames,
    /// but nothing needs to be rebuilt within one either.
    pub(super) fn build_object_set(
        &self,
        device: &Device,
        rows: BufferId,
        indices: BufferId,
    ) -> BindGroupId {
        device.create_transient_bind_group(
            self.single.plain,
            4,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::StorageBuffer(rows),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(indices),
                },
            ],
        )
    }

    /// Build the set-1 material descriptor set over the shared table. Cached by
    /// the renderer and only rebuilt when the material table changes.
    pub fn build_material_set(&self, device: &Device, materials: BufferId) -> BindGroupId {
        device.create_persistent_bind_group(
            self.single.plain,
            1,
            &[Binding {
                binding: 0,
                resource: BindingResource::StorageBuffer(materials),
            }],
        )
    }

    pub fn build_texture_set(&self, device: &Device, textures: &[TextureViewId]) -> BindGroupId {
        device.create_texture_array_group(
            self.single.plain,
            TEXTURE_SET as u32,
            textures,
            self.sampler,
        )
    }

    /// Pack this frame's decals into the block both geometry passes bind.
    ///
    /// Uploaded unconditionally, including for a frame with no decals in it —
    /// 2.5 KB of streaming write against a binding that has to be bound either
    /// way, and a zero count is what the shader's frame-uniform branch tests.
    /// The alternative is a "no decals" buffer kept alive as a second path
    /// through both passes, which is more state to get wrong than the write
    /// costs.
    pub(super) fn upload_decals(&self, device: &Device, decals: &[DecalInstance]) -> BufferId {
        let mut block = GpuDecals {
            count: [0; 4],
            decals: [GpuDecal::ZERO; MAX_DECALS],
        };
        {
            let count = decals.len().min(MAX_DECALS);
            block.count = [count as u32, 0, 0, 0];
            for (slot, decal) in block.decals.iter_mut().zip(decals.iter().take(count)) {
                let mut flags = 0u32;
                if decal.normal.is_some() {
                    flags |= decal_flags::NORMAL;
                }
                if decal.affects_surface {
                    flags |= decal_flags::SURFACE;
                }
                *slot = GpuDecal {
                    world_to_decal: decal.world_to_decal.to_cols_array_2d(),
                    base_color: [
                        decal.base_color.x,
                        decal.base_color.y,
                        decal.base_color.z,
                        decal.opacity,
                    ],
                    params: [
                        decal.metallic,
                        decal.roughness,
                        decal.normal_strength,
                        decal.angle_cos,
                    ],
                    tex: [
                        decal.albedo.map_or(WHITE_TEXTURE, |h| h.0),
                        decal.normal.map_or(FLAT_NORMAL_TEXTURE, |h| h.0),
                        decal.metallic_roughness.map_or(WHITE_TEXTURE, |h| h.0),
                        flags,
                    ],
                    axis_u: [
                        decal.axes.x_axis.x,
                        decal.axes.x_axis.y,
                        decal.axes.x_axis.z,
                        0.0,
                    ],
                    axis_v: [
                        decal.axes.y_axis.x,
                        decal.axes.y_axis.y,
                        decal.axes.y_axis.z,
                        0.0,
                    ],
                    axis_n: [
                        decal.axes.z_axis.x,
                        decal.axes.z_axis.y,
                        decal.axes.z_axis.z,
                        0.0,
                    ],
                };
            }
        }
        device.upload_block(BlockKind::Storage, bytemuck::bytes_of(&block))
    }

    /// Build the five descriptor sets both passes into the lit frame bind, and
    /// upload the per-frame blocks two of them point at.
    ///
    /// Once per frame rather than once per pass, and *before* the executor walks
    /// the schedule rather than inside the forward pass's body: the transparency
    /// pass binds the same five, and where the compiler chose to put it relative
    /// to this one is not something either pass may depend on.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_frame(
        &self,
        renderer: &VulkanRenderer,
        device: &Device,
        lighting: &SceneLighting,
        camera: &Camera,
        extent: [u32; 2],
        ao_view: TextureViewId,
        contact_shadow_view: TextureViewId,
        shadow_view: TextureViewId,
        atlas_view: TextureViewId,
        env_cube: TextureViewId,
        env_cube_sampler: SamplerId,
        shadows: Option<ShadowFrame<'_>>,
        atlas: &ShadowAtlas,
        material_set: BindGroupId,
        texture_set: BindGroupId,
        object_set: BindGroupId,
        decals: BufferId,
        environment: &EnvironmentSettings,
        fog: BufferId,
        fog_volume: TextureViewId,
        fog_sampler: SamplerId,
        surface_time: f32,
    ) -> ForwardSets {
        // Both halves fall back to the scene's flat ambient when nothing is
        // loaded — the diffuse as a band-0-only series, the specular as a tint
        // on a white cube. Two descriptions of the same uniform environment,
        // which is what keeps them from disagreeing.
        let ambient = lighting.ambient_color * lighting.ambient_nits;
        let irradiance = renderer.environment.irradiance(ambient, environment);
        let env_specular = renderer.environment.specular_tint(ambient, environment);
        let lighting_block = to_gpu_lighting(
            lighting,
            camera,
            extent,
            shadows,
            irradiance,
            environment.yaw,
            env_specular,
            surface_time,
        );
        let lighting_buffer =
            device.upload_block(BlockKind::Uniform, bytemuck::bytes_of(&lighting_block));
        let (point_buffer, spot_buffer, cluster_buffer) = {
            let mut scratch = self.light_scratch.lock().expect("light scratch poisoned");
            let LightScratch {
                points,
                spots,
                clusters,
            } = &mut *scratch;
            gpu_point_lights(lighting, atlas, points);
            gpu_spot_lights(lighting, atlas, spots);
            points.resize(MAX_POINT_LIGHTS, GpuPointLight::ZERO);
            spots.resize(MAX_SPOT_LIGHTS, GpuSpotLight::ZERO);
            gpu_clusters(&renderer.light_clusters, clusters);
            (
                device.upload_block(BlockKind::Uniform, bytemuck::cast_slice(points)),
                device.upload_block(BlockKind::Uniform, bytemuck::cast_slice(spots)),
                device.upload_block(BlockKind::Storage, bytemuck::cast_slice(clusters)),
            )
        };

        // Always at least one entry: a frame where nothing punctual casts still
        // has to bind something for the descriptor. Every light's face index is
        // -1 in that frame, so the row is never read.
        let face_count = atlas.faces.len().clamp(1, MAX_ATLAS_FACES);
        let mut shadow_faces = vec![GpuShadowFace::ZERO; face_count];
        for (row, face) in shadow_faces.iter_mut().zip(&atlas.faces) {
            *row = GpuShadowFace {
                view_proj: face.view_proj.to_cols_array_2d(),
                rect: face.tile.uv_rect(atlas.resolution),
            };
        }
        let shadow_faces =
            device.upload_block(BlockKind::Storage, bytemuck::cast_slice(&shadow_faces));

        self.assemble(
            device,
            LightingSet {
                lighting: lighting_buffer,
                decals,
                points: point_buffer,
                spots: spot_buffer,
                clusters: cluster_buffer,
            },
            SurfaceSet {
                ao: ao_view,
                shadow: shadow_view,
                env_cube,
                env_sampler: env_cube_sampler,
                contact_shadow: contact_shadow_view,
                atlas: atlas_view,
                shadow_faces,
                fog_volume,
                fog_sampler,
                fog,
            },
            material_set,
            texture_set,
            object_set,
        )
    }

    /// Sets 0 and 3 over what they bind, with the three shared ones beside
    /// them. The one place either set's bindings are listed, so the frame's
    /// sets and the material preview's cannot disagree about a slot.
    fn assemble(
        &self,
        device: &Device,
        set0: LightingSet,
        set3: SurfaceSet,
        material_set: BindGroupId,
        texture_set: BindGroupId,
        object_set: BindGroupId,
    ) -> ForwardSets {
        let lighting_set = device.create_transient_bind_group(
            self.single.plain,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::UniformBuffer(set0.lighting),
                },
                // The same buffer object the prepass binds. Not a copy: what a
                // decal does to a normal, an `f0` and a roughness has to be the
                // same in both passes or `ssr_resolve.comp` subtracts an
                // environment term belonging to the surface as it was before the
                // stamp landed — the identical argument the parallax march makes
                // about *where* a map is sampled.
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(set0.decals),
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::UniformBuffer(set0.points),
                },
                Binding {
                    binding: 3,
                    resource: BindingResource::UniformBuffer(set0.spots),
                },
                Binding {
                    binding: 4,
                    resource: BindingResource::StorageBuffer(set0.clusters),
                },
            ],
        );

        // Set 3 is the screen-space and shadow inputs. The cascades are kept as a
        // separate image and comparison sampler rather than a combined one, for
        // the same reason the texture array is: Metal caps samplers per stage far
        // lower than sampled images.
        let sampled = |binding, view: TextureViewId| Binding {
            binding,
            resource: BindingResource::SampledTexture {
                view,
                sampler: self.ao_sampler,
            },
        };
        let texture = |binding, view: TextureViewId| Binding {
            binding,
            resource: BindingResource::Texture(view),
        };
        let ao_set = device.create_transient_bind_group(
            self.single.plain,
            3,
            &[
                sampled(0, set3.ao),
                texture(1, set3.shadow),
                texture(3, set3.env_cube),
                Binding {
                    binding: 4,
                    resource: BindingResource::Sampler(set3.env_sampler),
                },
                sampled(5, set3.contact_shadow),
                // The atlas reads through the cascades' comparison sampler at
                // binding 2 — same conventions, same `Less` against a map
                // cleared to 1.0 — so it needs no sampler of its own. Which
                // matters: Metal caps samplers per stage far below sampled
                // images, and this shader is already at five.
                texture(6, set3.atlas),
                Binding {
                    binding: 7,
                    resource: BindingResource::StorageBuffer(set3.shadow_faces),
                },
                // The fog volume and the medium that produced it. Both are bound
                // in every frame — the volume as a 1x1x1 stand-in when the two
                // dispatches did not run — because the analytic term past the
                // froxels reads the same block, so there is one path in the
                // shader rather than a second pipeline.
                Binding {
                    binding: 8,
                    resource: BindingResource::SampledTexture {
                        view: set3.fog_volume,
                        sampler: set3.fog_sampler,
                    },
                },
                Binding {
                    binding: 9,
                    resource: BindingResource::UniformBuffer(set3.fog),
                },
            ],
        );

        ForwardSets {
            groups: [lighting_set, material_set, texture_set, ao_set, object_set],
        }
    }

    /// The five sets the material preview draws its sphere with: the frame's
    /// material table and textures, and everything else a sphere standing in
    /// no scene should see — no punctual light, no decal, no shadow, no fog,
    /// and no screen-space term, since the frame's screen is not its screen.
    pub(super) fn neutral_blocks(&self, device: &Device) -> NeutralBlocks {
        let block = |usage, bytes: &[u8]| {
            device.create_buffer_bytes(
                BufferDesc {
                    size: bytes.len() as u64,
                    usage,
                    location: MemoryLocation::GpuOnly,
                },
                bytes,
            )
        };
        let (uniform, storage) = (BufferUsage::UNIFORM_BUFFER, BufferUsage::STORAGE_BUFFER);
        let no_decals = GpuDecals {
            count: [0; 4],
            decals: [GpuDecal::ZERO; MAX_DECALS],
        };
        NeutralBlocks {
            decals: block(storage, bytemuck::bytes_of(&no_decals)),
            points: block(
                uniform,
                bytemuck::cast_slice(&[GpuPointLight::ZERO; MAX_POINT_LIGHTS]),
            ),
            spots: block(
                uniform,
                bytemuck::cast_slice(&[GpuSpotLight::ZERO; MAX_SPOT_LIGHTS]),
            ),
            clusters: block(storage, bytemuck::cast_slice(&[0u32; CLUSTERS * 4 + 1])),
            shadow_faces: block(storage, bytemuck::bytes_of(&GpuShadowFace::ZERO)),
            fog: block(
                uniform,
                bytemuck::bytes_of(&<super::fog::GpuFog as bytemuck::Zeroable>::zeroed()),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn preview_sets(
        &self,
        device: &Device,
        lighting: &PreviewLighting,
        neutral: NeutralViews,
        blocks: &NeutralBlocks,
        material_set: BindGroupId,
        texture_set: BindGroupId,
        object_set: BindGroupId,
        surface_time: f32,
    ) -> ForwardSets {
        let mut block = lighting.0;
        block.params[1] = surface_time;
        self.assemble(
            device,
            LightingSet {
                lighting: device.upload_block(BlockKind::Uniform, bytemuck::bytes_of(&block)),
                decals: blocks.decals,
                points: blocks.points,
                spots: blocks.spots,
                clusters: blocks.clusters,
            },
            SurfaceSet {
                ao: neutral.ao,
                shadow: neutral.shadow,
                env_cube: neutral.env_cube,
                env_sampler: neutral.env_sampler,
                contact_shadow: neutral.contact_shadow,
                atlas: neutral.atlas,
                shadow_faces: blocks.shadow_faces,
                fog_volume: neutral.fog_volume,
                fog_sampler: neutral.fog_sampler,
                fog: blocks.fog,
            },
            material_set,
            texture_set,
            object_set,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw(
        &self,
        builder: &mut Encoder,
        draws: Draws<'_>,
        view: &FrameView,
        extent: [u32; 2],
        sets: &ForwardSets,
        subsurface: bool,
        // Whether the frame rasterises multisampled. The graph's answer, like
        // `subsurface`: the executor reads it off the frame's ids, so the
        // pipeline bound here cannot disagree with the render pass the
        // framebuffer already opened around it.
        msaa: bool,
        surfaces: Surfaces<'_>,
    ) {
        // The jittered one, from the frame's shared view: every pass that
        // rasterises geometry has to agree on it to a subpixel.
        let view_proj = view.view_proj;
        let pipelines = self.pipelines(msaa);
        // The one value a geometry draw still pushes. The fat per-object matrices
        // are in the set-4 storage buffer, and what used to sit beside
        // `view_proj` here — the run's material and its first object row — moved
        // into the instance list and into `firstInstance`, because a multi-draw
        // has no per-draw push. See [`instances`](super::instances) and
        // [`cull`](super::cull). `view_proj` replaced a pre-multiplied `mvp` when
        // draws became instanced: a run covers many models, so the model half has
        // to be applied in the shader. The transparency and refraction passes push
        // the same range through the same layout.
        let mut words = [0u32; 16];
        for (word, value) in words
            .iter_mut()
            .zip(view_proj.to_cols_array_2d().iter().flatten())
        {
            *word = value.to_bits();
        }
        // The descriptor sets are rebound with the incoming pipeline. Every
        // layout's first five sets are built from the same stages and the same
        // immutable shadow sampler, or lifted from one that was by a surface
        // shader's twin, so they are set compatible and this is a formality —
        // but it is the formality that keeps it true if one ever stops being.
        let groups = sets.groups();
        let bind_all = |builder: &mut Encoder, pipeline: PipelineId| {
            builder.bind_pipeline(pipeline);
            for (set, group) in groups.iter().enumerate() {
                builder.bind_group(set as u32, *group);
            }
            builder.push_constants(&words);
        };

        builder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });

        // Every draw below names its geometry through `firstIndex` and
        // `vertexOffset` into the arena, so the three buffers are bound once
        // for the pass rather than once per mesh. See `mesh`.
        builder.bind_geometry(false);

        // One multi-draw per pipeline class, over the commands the dispatch
        // wrote: the recording cost scales with neither the entity count nor
        // the distinct mesh/material pairs, only with the classes — four, and
        // two more for each surface shader a scene draws.
        for region in draws.regions() {
            let twin = match region.variant.surface {
                Surface::Custom(slot) => surfaces.for_slot(
                    slot,
                    Twin::Forward {
                        msaa,
                        subsurface,
                        masked: region.variant.masked,
                    },
                ),
                _ => None,
            };
            match twin {
                Some((pipeline, group)) => {
                    bind_all(builder, pipeline);
                    builder.bind_group(crate::gfx::surface::PARAMS_BINDING.0, group);
                }
                None => bind_all(builder, pipelines.for_variant(region.variant, subsurface)),
            }
            builder.draw_indexed_indirect(region.commands, region.range);
        }
    }
}

/// One forward pipeline: the state every built-in one and every surface
/// shader's twin of it are built from, so the two cannot differ in anything but
/// the fragment shader. `samplers` is the shadow sampler a built-in bakes into
/// set 3; a twin passes none and lifts the set instead.
fn pipeline_desc<'a>(
    colors: &'a [Format],
    fragment: ShaderRef,
    // Inverted from the obvious sense, and named for what it
    // selects: back faces are culled for ordinary opaque
    // geometry and kept for `BlendMode::Masked`, because a
    // cutout sheet has no back to cull.
    cull_back: bool,
    // How many samples the render pass rasterises at, and —
    // because the two are the same decision — where the depth
    // being tested came from.
    //
    // At more than one sample this pass rasterises its own
    // depth buffer: the test is `Less` and the pass writes,
    // and a masked run spends the extra samples on the
    // cutout's edge through alpha to coverage.
    //
    // At one sample there is no second depth buffer. The
    // geometry prepass's is attached read-only and the test
    // is `EQUAL` against it, which is the whole point: every
    // fragment that is not the front-most surface is killed
    // before it shades, so the pass has perfect early-Z and
    // zero shading overdraw. It depends on the two vertex
    // shaders agreeing on `gl_Position` to the last bit — see
    // the `invariant` declaration in each.
    samples: SampleCount,
    samplers: &'a [(u32, u32, SamplerId)],
) -> GraphicsPipelineDesc<'a> {
    GraphicsPipelineDesc {
        vertex: ShaderId::ForwardVert.into(),
        fragment,
        color_formats: colors,
        depth_format: Some(DEPTH_FORMAT),
        vertex_streams: &[PositionVertex::STREAM, SurfaceVertex::STREAM],
        cull: if cull_back {
            CullMode::Back
        } else {
            CullMode::None
        },
        depth_test: Some(if samples == SampleCount::Sample1 {
            DepthTest {
                write: false,
                compare: DepthCompare::Equal,
            }
        } else {
            DepthTest {
                write: true,
                compare: crate::gfx::depth::NEARER,
            }
        }),
        // The material textures, of which a scene writes only the ones it
        // loaded. Set on the layout because it is the *layout* the short
        // write is legal against — and this layout is also the one the
        // transparency and refraction pipelines lift, so both inherit it.
        partial_texture_set: Some(TEXTURE_SET as u32),
        dynamic_depth_bias: false,
        samples,
        // Only where there are samples to spend. At one sample the cutout
        // is cut by the `discard` in the shader this variant was built
        // from instead, along the same line.
        alpha_to_coverage: !cull_back && samples != SampleCount::Sample1,
        immutable_samplers: samplers,
        blend: Blend::None,
        topology: Topology::Triangles,
        mask_extra_attachments: false,
        attachment_blends: &[],
        layout_from: None,
        layout_prefix_from: None,
    }
}

pub(super) fn to_gpu_material(m: &Material) -> GpuMaterial {
    // Derived from what was authored, never from a separate switch: a lobe is on
    // exactly when it would change a pixel. `clearcoat_texture` alone turns the
    // coat on because a map that modulates a zero would otherwise be silently
    // dead, and `transmission` only counts on the queue that owns a pass able to
    // fetch what is behind the surface.
    let mut flags = 0u32;
    if m.clearcoat > 0.0 || m.clearcoat_texture.is_some() {
        flags |= material_flags::CLEARCOAT;
    }
    if m.sheen_color != Vec3::ZERO || m.sheen_texture.is_some() {
        flags |= material_flags::SHEEN;
    }
    if m.anisotropy != 0.0 || m.anisotropy_texture.is_some() {
        flags |= material_flags::ANISOTROPY;
    }
    if m.blend == BlendMode::Transmissive
        && (m.transmission > 0.0 || m.transmission_texture.is_some())
    {
        flags |= material_flags::TRANSMISSION;
    }
    // Deliberately *not* gated on the blend mode, unlike transmission above. The
    // screen-space diffusion only reaches the opaque queue, but the analytic half
    // of scattering is a lighting term like any other, so a blended leaf and an
    // opaque one scatter by the same rule — which is the whole reason
    // `shading.glsl` is one file.
    if m.subsurface_color != Vec3::ZERO || m.subsurface_texture.is_some() {
        flags |= material_flags::SUBSURFACE;
    }
    // The one lobe whose map is not optional, and the exception proves the rule
    // the others follow: a clear coat with no map is a coat of uniform strength,
    // while a height field with no heights is a plane. So this asks for both, and
    // a depth of zero is the same plane said the other way.
    if m.height_texture.is_some() && m.parallax_depth > 0.0 {
        flags |= material_flags::PARALLAX;
    }
    // Read straight off the blend mode rather than derived from whether an alpha
    // was authored, which is the one place this file's usual rule does not
    // apply: every other flag above answers "would this change a pixel", and a
    // cutout's answer is "yes, by removing it". A material whose albedo map has
    // no alpha channel samples 1.0 and survives the test everywhere, which is
    // the same no-op the default textures give the lobes.
    if m.blend == BlendMode::Masked {
        flags |= material_flags::MASKED;
    }

    // Beer-Lambert wants an extinction coefficient per unit distance, and an
    // infinite attenuation distance is the "absorbs nothing" case the shader
    // would otherwise reach by dividing by infinity. Zero is that same case
    // expressed as a coefficient, so both collapse to one path there.
    // The march interpolates between these by how head-on the surface is, so a
    // maximum below the minimum would spend *more* samples the less they are
    // needed. Ordered here rather than in the shader, and the upper bound is
    // taken from the already-clamped minimum because `clamp` panics on a range
    // that runs backwards — a material asking for a thousand steps would
    // otherwise supply one.
    let parallax_min = m.parallax_min_steps.clamp(1, PARALLAX_STEP_LIMIT);
    let parallax_max = m
        .parallax_max_steps
        .clamp(parallax_min, PARALLAX_STEP_LIMIT);

    let attenuation_distance = if m.attenuation_distance.is_finite() {
        m.attenuation_distance.max(0.0)
    } else {
        0.0
    };

    // Missing maps fall back to the default textures, which make the sample a
    // no-op (white = ×1, flat normal = unchanged geometric normal).
    GpuMaterial {
        // Opacity rides in `w` because that is where a base colour's alpha
        // belongs and because it costs nothing: the field was already a `vec4`
        // for std430's sake. The opaque pipeline ignores it.
        base_color: [
            m.base_color.x,
            m.base_color.y,
            m.base_color.z,
            m.alpha.clamp(0.0, 1.0),
        ],
        emissive: [m.emissive.x, m.emissive.y, m.emissive.z, 0.0],
        params: [m.metallic, m.roughness, m.reflectance, m.ior.max(1.0)],
        tex_indices: [
            m.albedo_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.normal_texture.map_or(FLAT_NORMAL_TEXTURE, |h| h.0),
            m.metallic_roughness_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.emissive_texture.map_or(WHITE_TEXTURE, |h| h.0),
        ],
        clearcoat: [m.clearcoat, m.clearcoat_roughness, 0.0, 0.0],
        sheen: [
            m.sheen_color.x,
            m.sheen_color.y,
            m.sheen_color.z,
            m.sheen_roughness,
        ],
        // The rotation resolves to its sine and cosine here rather than in the
        // shader: it is per material, not per fragment, and a transcendental per
        // pixel to rotate a constant frame is the kind of cost that never shows
        // up in a profile as itself.
        anisotropy: [
            m.anisotropy.clamp(-1.0, 1.0),
            m.anisotropy_rotation.cos(),
            m.anisotropy_rotation.sin(),
            0.0,
        ],
        transmission: [
            m.transmission.clamp(0.0, 1.0),
            m.thickness.max(0.0),
            attenuation_distance,
            0.0,
        ],
        attenuation: [
            m.attenuation_color.x,
            m.attenuation_color.y,
            m.attenuation_color.z,
            0.0,
        ],
        subsurface: [
            m.subsurface_color.x,
            m.subsurface_color.y,
            m.subsurface_color.z,
            // Floored: the lobe is `exp2(power * (cos - 1))`, and a zero power is
            // a lobe that is one in every direction — forward scattering that
            // does not fall off is a uniform glow rather than a halo.
            m.subsurface_forward_scatter.max(1e-3),
        ],
        // Clamped away from zero rather than allowed to reach it: every consumer
        // divides by a mean free path, and a channel that scatters over no
        // distance at all is the "no scattering" case already expressed by a
        // black `subsurface_color`.
        subsurface_radius: [
            m.subsurface_radius.x.max(1e-6),
            m.subsurface_radius.y.max(1e-6),
            m.subsurface_radius.z.max(1e-6),
            0.0,
        ],
        parallax: [
            m.parallax_depth.max(0.0),
            parallax_min as f32,
            parallax_max as f32,
            0.0,
        ],
        // Clamped just inside the unit range at both ends. A cutoff of exactly
        // zero would keep a fully transparent texel — the test is "reaches" —
        // and one of exactly one would cut every texel of an opaque map away,
        // which reads as the mesh having failed to load rather than as a badly
        // authored number.
        alpha: [m.alpha_cutoff.clamp(1e-3, 1.0 - 1e-3), 0.0, 0.0, 0.0],
        tex_indices_ext: [
            m.clearcoat_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.clearcoat_normal_texture
                .map_or(FLAT_NORMAL_TEXTURE, |h| h.0),
            m.sheen_texture.map_or(WHITE_TEXTURE, |h| h.0),
            // The direction half of this map is signed and decoded as
            // `rg * 2 - 1`, so the neutral fill is the flat normal's (0.5, 0.5)
            // rather than white — which would read as a 45-degree rotation.
            m.anisotropy_texture.map_or(FLAT_NORMAL_TEXTURE, |h| h.0),
        ],
        tex_flags: [
            m.transmission_texture.map_or(WHITE_TEXTURE, |h| h.0),
            flags,
            m.subsurface_texture.map_or(WHITE_TEXTURE, |h| h.0),
            // White is the top of the field everywhere, so an unauthored map is a
            // flat surface rather than one displaced by its full depth — the
            // march reads the same no-op out of it that every other default
            // texture gives its lobe.
            m.height_texture.map_or(WHITE_TEXTURE, |h| h.0),
        ],
        surface: [0; 4],
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuMaterial, material_flags, to_gpu_material};
    use crate::gfx::{BlendMode, Material};

    /// The bit the three opaque passes pick a pipeline from, and the value the
    /// shader tests, are the same word — so this is the one place the two can be
    /// checked against each other without a GPU. A material that answered
    /// `is_masked` differently from what it uploaded would draw through the
    /// alpha-to-coverage pipeline and then never test its alpha, or the reverse.
    #[test]
    fn only_a_masked_material_asks_for_the_alpha_testing_pipeline() {
        for (blend, expected) in [
            (BlendMode::Opaque, false),
            (BlendMode::Masked, true),
            (BlendMode::Blend, false),
            (BlendMode::Transmissive, false),
        ] {
            let gpu = to_gpu_material(&Material {
                blend,
                ..Material::default()
            });
            assert_eq!(gpu.is_masked(), expected, "{blend:?}");
            assert_eq!(
                gpu.tex_flags[1] & material_flags::MASKED != 0,
                expected,
                "{blend:?}: the flag the shader reads disagrees with is_masked"
            );
            assert_eq!(blend.is_masked(), expected, "{blend:?}");
        }
    }

    /// A cutout is opaque geometry: it writes depth, it is in the prepass, and it
    /// is in every caster list. Only the two queues that write no depth leave.
    #[test]
    fn a_cutout_stays_in_the_opaque_queue() {
        assert!(BlendMode::Masked.is_opaque());
        assert!(BlendMode::Opaque.is_opaque());
        assert!(!BlendMode::Blend.is_opaque());
        assert!(!BlendMode::Transmissive.is_opaque());
    }

    /// Both ends of the range are degenerate and neither is worth shipping as a
    /// mystery: at zero every texel survives including the fully transparent
    /// ones, and at one every texel of an opaque map is cut away, which reads as
    /// the mesh having failed to load rather than as a badly authored number.
    #[test]
    fn the_cutoff_is_clamped_inside_the_unit_range() {
        let cutoff = |alpha_cutoff| {
            to_gpu_material(&Material {
                blend: BlendMode::Masked,
                alpha_cutoff,
                ..Material::default()
            })
            .alpha[0]
        };
        assert!(cutoff(0.0) > 0.0);
        assert!(cutoff(1.0) < 1.0);
        assert!(cutoff(-5.0) > 0.0);
        assert_eq!(cutoff(0.5), 0.5);
    }

    /// std430 derives an array's stride from its element, so every pass steps
    /// through the table at the pitch of the one mirror in
    /// `shaders/material.glsl` — a mirror one block short reads every material
    /// past the first out of the middle of its neighbour. No shader can notice,
    /// which is why the size is pinned here.
    #[test]
    fn the_material_block_is_the_size_material_glsl_declares() {
        // 16 vec4-sized blocks: base colour, emissive, params, texture indices,
        // the five lobes' blocks, the two subsurface blocks, parallax, alpha,
        // the extended indices, the flags, and the surface shader's word.
        assert_eq!(std::mem::size_of::<GpuMaterial>(), 16 * 16);
    }
}
