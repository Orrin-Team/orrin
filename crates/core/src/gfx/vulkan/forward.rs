use std::sync::Arc;

use glam::{Mat4, Vec3};
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device as VkDevice;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::PipelineLayout;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};

use crate::geom::Aabb;
use crate::gfx::punctual::{LightKind, MAX_ATLAS_FACES, ShadowAtlas};
use crate::gfx::sh::SH9;
use crate::gfx::shadows::MAX_CASCADES;
use crate::gfx::{
    BlendMode, DecalInstance, MAX_DECALS, MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS, Material,
    PositionVertex, SceneLighting, SurfaceVertex,
};
use crate::scene::{Camera, EnvironmentSettings};

use super::context::VkContext;
use super::convert::vk_format;
use super::cull::Draws;
use super::fog::GpuFog;
use super::instances::{GpuObject, InstanceEntry};
use super::mesh::MeshSpan;
use super::record::Arena;
use super::rendering;
use super::subsurface::SUBSURFACE_FORMAT;
use super::swapchain::DEPTH_FORMAT;
use super::taa::FrameView;
use super::{MSAA_SAMPLES, PassCtx, ShadowFrame, VulkanRenderer};
use crate::gfx::format::{Format, SampleCount};
use crate::gfx::rhi::{
    CullMode, DepthCompare, DepthTest, Device, Encoder, GraphicsPipelineDesc, PipelineId,
    SamplerDesc, Viewport,
};
use crate::gfx::shaders::ShaderId;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;

pub struct GpuMesh {
    /// Where this mesh's indices and vertices sit in the shared buffers every
    /// pass binds once. See [`mesh`](super::mesh).
    pub span: MeshSpan,
    /// Object-space bounds, derived at upload because that is the last place the
    /// vertex data exists on the CPU. Culling reads them through
    /// [`RenderBackend::mesh_bounds`](crate::gfx::RenderBackend::mesh_bounds).
    pub bounds: Aabb,
}

/// The one value a geometry draw still pushes. The fat per-object matrices are
/// in the set-4 storage buffer, and what used to sit beside `view_proj` here —
/// the run's material and its first object row — moved into the instance list
/// and into `firstInstance`, because a multi-draw has no per-draw push. See
/// [`instances`](super::instances) and [`cull`](super::cull).
///
/// `view_proj` replaced a pre-multiplied `mvp` when draws became instanced: a
/// run covers many models, so the model half has to be applied in the shader.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct PushConstants {
    view_proj: [[f32; 4]; 4],
}

impl PushConstants {
    /// Also what the transparency pass pushes: it draws the same geometry
    /// through the same pipeline layout, so it pushes the same range.
    pub(super) fn new(view_proj: Mat4) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
        }
    }
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

/// Feature bits in [`GpuMaterial::flags`], mirrored by `shading.glsl`.
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
}

/// Ceiling on a material's parallax step counts. Keep in sync with
/// `PARALLAX_STEP_LIMIT` in `shaders/parallax.glsl`, which is what actually
/// bounds the loop: a material asking for more would pay for the extra
/// iterations in register pressure and get none of them.
const PARALLAX_STEP_LIMIT: u32 = 64;

#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
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
}

/// Pack the engine's [`SceneLighting`] into the std140 layout the shader expects.
#[allow(clippy::too_many_arguments)]
fn to_gpu_lighting(
    lighting: &SceneLighting,
    camera_pos: Vec3,
    extent: [u32; 2],
    shadows: Option<ShadowFrame<'_>>,
    atlas: &ShadowAtlas,
    irradiance: [Vec3; SH9],
    environment_yaw: f32,
    env_specular: Vec3,
) -> GpuLighting {
    let (w, h) = (extent[0] as f32, extent[1] as f32);
    let count = lighting.point_lights.len().min(MAX_POINT_LIGHTS);
    let mut point_lights = [GpuPointLight::ZERO; MAX_POINT_LIGHTS];
    for (index, (slot, light)) in point_lights
        .iter_mut()
        .zip(lighting.point_lights.iter().take(count))
        .enumerate()
    {
        slot.position = [
            light.position.x,
            light.position.y,
            light.position.z,
            light.range.max(1e-4),
        ];
        slot.color = [light.color.x, light.color.y, light.color.z, light.candela];
        // A light that asked for tiles and did not get them keeps its -1 and
        // shades unshadowed, which is the whole behaviour of the budget: too
        // many casters costs shadows, never correctness.
        slot.shadow = match atlas.caster(LightKind::Point, index) {
            Some(caster) => [caster.first_face as f32, caster.near, 0.0, 0.0],
            None => [-1.0, 0.0, 0.0, 0.0],
        };
    }

    let spot_count = lighting.spot_lights.len().min(MAX_SPOT_LIGHTS);
    let mut spot_lights = [GpuSpotLight::ZERO; MAX_SPOT_LIGHTS];
    for (index, (slot, light)) in spot_lights
        .iter_mut()
        .zip(lighting.spot_lights.iter().take(spot_count))
        .enumerate()
    {
        slot.position = [
            light.position.x,
            light.position.y,
            light.position.z,
            light.range.max(1e-4),
        ];
        slot.direction = [
            light.direction.x,
            light.direction.y,
            light.direction.z,
            light.outer_cos,
        ];
        slot.color = [light.color.x, light.color.y, light.color.z, light.candela];
        let face = atlas
            .caster(LightKind::Spot, index)
            .map_or([-1.0, 0.0], |caster| {
                [caster.first_face as f32, caster.near]
            });
        slot.params = [light.inner_cos, face[0], face[1], 0.0];
    }

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
            lighting.shininess,
            lighting.specular_strength,
            spot_count as f32,
        ],
        viewport: [w, h, 1.0 / w, 1.0 / h],
        cascades: GpuCascades::new(shadows),
        point_lights,
        spot_lights,
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
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
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

/// GPU mirror of a [`PointLight`](crate::gfx::PointLight), padded to std140
/// (three `vec4`s).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
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

/// GPU mirror of a [`SpotLight`](crate::gfx::SpotLight), padded to std140.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
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
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
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
    /// x = point light count, y = shininess, z = specular strength,
    /// w = spot light count.
    params: [f32; 4],
    /// x=w, y=h, z=1/w, w=1/h
    viewport: [f32; 4],
    cascades: GpuCascades,
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
    spot_lights: [GpuSpotLight; MAX_SPOT_LIGHTS],
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
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
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
/// whether the scene has decals or not and buys a `Subbuffer<GpuDecals>` that
/// both passes can bind with no length to agree on either.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
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
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
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

/// The five descriptor sets a pass shading into the lit frame binds, in bind
/// order.
///
/// Both such passes bind the same five — the transparency pass draws through
/// this pass's own pipeline layout, so it can and must. Bundling them is what
/// keeps that from being a comment somebody has to keep true.
pub(super) struct ForwardSets {
    sets: Vec<Arc<DescriptorSet>>,
}

impl ForwardSets {
    pub(super) fn as_vec(&self) -> Vec<Arc<DescriptorSet>> {
        self.sets.clone()
    }

    /// The same five with the object set swapped, for a pass that indexes
    /// through a different instance buffer.
    ///
    /// Under GPU culling the opaque queue reads what the dispatch compacted and
    /// the blended queues read what the host wrote, and the two are different
    /// allocations. Rebuilding the other four sets to say so would be four
    /// descriptor writes a frame for a difference in one of them.
    pub(super) fn with_object_set(&self, object_set: Arc<DescriptorSet>) -> ForwardSets {
        let mut sets = self.sets.clone();
        *sets.last_mut().expect("the object set is the last of five") = object_set;
        ForwardSets { sets }
    }
}

/// The four opaque pipelines one forward render pass needs.
///
/// Two axes, and neither can be collapsed. `subsurface` is a property of the
/// *frame* — it decides which render pass is open and therefore how many colour
/// attachments a pipeline must declare — while `masked` is a property of a
/// *run*, so both of those have to exist at once whichever frame it is.
struct ForwardPipelines {
    plain: PipelineId,
    subsurface: PipelineId,
    /// Alpha-testing, for the `Masked` runs inside the same render pass. Back
    /// faces are kept, because a cutout sheet has no back to cull, and the cut
    /// itself is taken by whichever mechanism the sample count affords: alpha to
    /// coverage across four samples, a `discard` at one.
    masked: PipelineId,
    masked_subsurface: PipelineId,
}

impl ForwardPipelines {
    /// The pair a frame draws with, given whether it diffuses subsurface light.
    fn for_frame(&self, subsurface: bool) -> (PipelineId, PipelineId) {
        if subsurface {
            (self.subsurface, self.masked_subsurface)
        } else {
            (self.plain, self.masked)
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
///   nothing to that attachment, which is what `co_tenant_blend_states` is for.
///
/// The depth format is the same in all four; what differs is what the pass does
/// with it, and that is now a record-time decision rather than a property baked
/// into a pass object. At more than one sample the pass rasterises its own depth
/// and discards it; at one sample it borrows the geometry prepass's read-only
/// and tests `EQUAL`. See [`rendering`](super::rendering).
#[derive(Clone)]
pub struct ForwardTargets {
    pub multisampled: PipelineRenderingCreateInfo,
    pub multisampled_subsurface: PipelineRenderingCreateInfo,
    pub single: PipelineRenderingCreateInfo,
    pub single_subsurface: PipelineRenderingCreateInfo,
}

impl ForwardTargets {
    fn new(color_format: Format, msaa_color_format: Format) -> Self {
        let (color_format, msaa_color_format) =
            (vk_format(color_format), vk_format(msaa_color_format));
        let depth = Some(vk_format(DEPTH_FORMAT));
        Self {
            multisampled: rendering::pipeline_info(&[msaa_color_format], depth),
            multisampled_subsurface: rendering::pipeline_info(
                &[msaa_color_format, vk_format(SUBSURFACE_FORMAT)],
                depth,
            ),
            single: rendering::pipeline_info(&[color_format], depth),
            single_subsurface: rendering::pipeline_info(
                &[color_format, vk_format(SUBSURFACE_FORMAT)],
                depth,
            ),
        }
    }
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
    /// five descriptor sets bind to any of them.
    multisampled: ForwardPipelines,
    single: ForwardPipelines,
    uniform_buffer_allocator: Arena,
    /// Per-frame storage for the atlas face table. A storage buffer rather than
    /// more of the lighting uniform: forty-eight matrices is three kilobytes,
    /// and the guaranteed uniform-buffer range is sixteen.
    shadow_face_allocator: Arena,
    /// Per-frame storage for the decal block. Owned here rather than by a decal
    /// module because there is no decal *pass* to own it — the block is frame
    /// data that two existing passes read, which is exactly what the shadow face
    /// table above is.
    decal_allocator: Arena,
    sampler: Arc<Sampler>,
    ao_sampler: Arc<Sampler>,
}

fn layout(
    device: &Device,
    pipeline: PipelineId,
    set: usize,
) -> Arc<vulkano::descriptor_set::layout::DescriptorSetLayout> {
    let Device::Vulkan(driver) = device;
    driver.set_layout(pipeline, set)
}

impl ForwardPass {
    /// `color_format` is what the one-sample pair writes — the frame's packed
    /// colour format. `msaa_color_format` is the wider one the multisampled pair
    /// needs: a resolve requires both images to agree, and alpha to coverage
    /// needs an alpha channel to read. See `hdr::HDR_WIDE_FORMAT`.
    pub fn new(
        ctx: &VkContext,
        rhi: &Device,
        color_format: Format,
        msaa_color_format: Format,
    ) -> Self {
        let device = &ctx.device;
        let memory_allocator = &ctx.memory_allocator;
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
        let pipeline = |colors: &[Format],
                        fragment: ShaderId,
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
                        samples: SampleCount| {
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex: ShaderId::ForwardVert,
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
                        compare: DepthCompare::Less,
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
                immutable_samplers: &samplers,
                blend_alpha: false,
            })
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
        };

        let uniform_buffer_allocator = Arena::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let shadow_face_allocator = Arena::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let decal_allocator = Arena::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        // Material textures are the only ones with a mip chain to sample; the
        // AO and tonemap inputs are screen-space targets read at 1:1.
        let anisotropy = device.enabled_features().sampler_anisotropy.then(|| {
            device
                .physical_device()
                .properties()
                .max_sampler_anisotropy
                .min(16.0)
        });

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                anisotropy,
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .unwrap();

        let ao_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        Self {
            targets,
            multisampled,
            single,
            uniform_buffer_allocator,
            shadow_face_allocator,
            decal_allocator,
            sampler,
            ao_sampler,
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

    /// The layout the transparency pass builds its own pipeline with. Shared
    /// rather than derived a second time, so the two pipelines cannot disagree
    /// about a binding — see `oit.rs`.
    pub(super) fn pipeline_layout(&self, device: &Device) -> Arc<PipelineLayout> {
        let Device::Vulkan(driver) = device;
        driver.pipeline_layout(self.single.plain)
    }

    /// The set-4 per-object descriptor set for this frame's object buffer.
    ///
    /// Built once per frame by the executor rather than once per pass: the
    /// buffer changes every frame, so nothing here can be cached across frames,
    /// but nothing needs to be rebuilt within one either.
    pub(super) fn build_object_set(
        &self,
        ctx: &VkContext,
        device: &Device,
        rows: &Subbuffer<[GpuObject]>,
        indices: &Subbuffer<[InstanceEntry]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout(device, self.single.plain, 4),
            [
                WriteDescriptorSet::buffer(0, rows.clone()),
                WriteDescriptorSet::buffer(1, indices.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// Build the set-1 material descriptor set over the shared table. Cached by
    /// the renderer and only rebuilt when the material table changes.
    pub fn build_material_set(
        &self,
        ctx: &VkContext,
        device: &Device,
        buffer: &Subbuffer<[GpuMaterial]>,
    ) -> Arc<DescriptorSet> {
        let buffer = buffer.clone();
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout(device, self.single.plain, 1),
            [WriteDescriptorSet::buffer(0, buffer)],
            [],
        )
        .unwrap()
    }

    /// Build the set-2 texture array + sampler descriptor set. Cached by the
    /// renderer and only rebuilt when a texture is added.
    pub fn build_texture_set(
        &self,
        ctx: &VkContext,
        device: &Device,
        textures: &[Arc<ImageView>],
    ) -> Arc<DescriptorSet> {
        let default_view = textures[0].clone();
        let texture_array = (0..ctx.texture_array_len(textures.len())).map(|i| {
            textures
                .get(i)
                .cloned()
                .unwrap_or_else(|| default_view.clone())
        });
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout(device, self.single.plain, TEXTURE_SET),
            [
                WriteDescriptorSet::image_view_array(0, 0, texture_array),
                WriteDescriptorSet::sampler(1, self.sampler.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// Pack this frame's decals into the block both geometry passes bind.
    ///
    /// Uploaded unconditionally, including for a frame with no decals in it —
    /// 2.5 KB of streaming write against a descriptor that has to be bound
    /// either way, and a zero count is what the shader's frame-uniform branch
    /// tests. The alternative is a "no decals" buffer kept alive as a second
    /// path through both passes, which is more state to get wrong than the
    /// write costs.
    pub(super) fn upload_decals(&self, decals: &[DecalInstance]) -> Subbuffer<GpuDecals> {
        let buffer = self.decal_allocator.allocate_sized::<GpuDecals>();
        {
            let mut block = buffer.write().unwrap();
            let count = decals.len().min(MAX_DECALS);
            block.count = [count as u32, 0, 0, 0];
            block.decals = [GpuDecal::ZERO; MAX_DECALS];
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
        buffer
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
        ao_view: Arc<ImageView>,
        contact_shadow_view: Arc<ImageView>,
        shadow_view: Arc<ImageView>,
        atlas_view: Arc<ImageView>,
        shadows: Option<ShadowFrame<'_>>,
        atlas: &ShadowAtlas,
        material_set: Arc<DescriptorSet>,
        texture_set: Arc<DescriptorSet>,
        object_set: Arc<DescriptorSet>,
        decals: Subbuffer<GpuDecals>,
        environment: &EnvironmentSettings,
        fog: Subbuffer<GpuFog>,
        fog_volume: Arc<ImageView>,
        fog_sampler: Arc<Sampler>,
    ) -> ForwardSets {
        let lighting_buffer = self
            .uniform_buffer_allocator
            .allocate_sized::<GpuLighting>();
        // Both halves fall back to the scene's flat ambient when nothing is
        // loaded — the diffuse as a band-0-only series, the specular as a tint
        // on a white cube. Two descriptions of the same uniform environment,
        // which is what keeps them from disagreeing.
        let ambient = lighting.ambient_color * lighting.ambient_nits;
        let irradiance = renderer.environment.irradiance(ambient, environment);
        let env_specular = renderer.environment.specular_tint(ambient, environment);
        *lighting_buffer.write().unwrap() = to_gpu_lighting(
            lighting,
            camera.position,
            extent,
            shadows,
            atlas,
            irradiance,
            environment.yaw,
            env_specular,
        );

        // Always at least one entry: `allocate_slice` rejects a length of zero,
        // and a frame where nothing punctual casts still has to bind something
        // for the descriptor. Every light's face index is -1 in that frame, so
        // the row is never read.
        let face_count = atlas.faces.len().clamp(1, MAX_ATLAS_FACES);
        let shadow_faces = self
            .shadow_face_allocator
            .allocate_slice::<GpuShadowFace>(face_count as u64);
        {
            let mut rows = shadow_faces.write().unwrap();
            rows.fill(GpuShadowFace::ZERO);
            for (row, face) in rows.iter_mut().zip(&atlas.faces) {
                *row = GpuShadowFace {
                    view_proj: face.view_proj.to_cols_array_2d(),
                    rect: face.tile.uv_rect(atlas.resolution),
                };
            }
        }

        let lighting_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            layout(device, self.single.plain, 0),
            [
                WriteDescriptorSet::buffer(0, lighting_buffer),
                // The same buffer object the prepass binds. Not a copy: what a
                // decal does to a normal, an `f0` and a roughness has to be the
                // same in both passes or `ssr_resolve.comp` subtracts an
                // environment term belonging to the surface as it was before the
                // stamp landed — the identical argument the parallax march makes
                // about *where* a map is sampled.
                WriteDescriptorSet::buffer(1, decals),
            ],
            [],
        )
        .unwrap();

        // Set 3 is the screen-space and shadow inputs. The cascades are kept as
        // a separate image and comparison sampler rather than a combined one,
        // for the same reason the texture array is: Metal caps samplers per
        // stage far lower than sampled images.
        let ao_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            layout(device, self.single.plain, 3),
            [
                WriteDescriptorSet::image_view_sampler(0, ao_view, self.ao_sampler.clone()),
                WriteDescriptorSet::image_view(1, shadow_view),
                WriteDescriptorSet::image_view(3, renderer.environment.specular_view()),
                WriteDescriptorSet::sampler(4, renderer.environment.sampler()),
                WriteDescriptorSet::image_view_sampler(
                    5,
                    contact_shadow_view,
                    self.ao_sampler.clone(),
                ),
                // The atlas reads through the cascades' comparison sampler at
                // binding 2 — same conventions, same `Less` against a map
                // cleared to 1.0 — so it needs no sampler of its own. Which
                // matters: Metal caps samplers per stage far below sampled
                // images, and this shader is already at five.
                WriteDescriptorSet::image_view(6, atlas_view),
                WriteDescriptorSet::buffer(7, shadow_faces),
                // The fog volume and the medium that produced it. Both are bound
                // in every frame — the volume as a 1x1x1 stand-in when the two
                // dispatches did not run — because the analytic term past the
                // froxels reads the same block, so there is one path in the
                // shader rather than a second pipeline.
                WriteDescriptorSet::image_view_sampler(8, fog_volume, fog_sampler),
                WriteDescriptorSet::buffer(9, fog),
            ],
            [],
        )
        .unwrap();

        ForwardSets {
            sets: vec![lighting_set, material_set, texture_set, ao_set, object_set],
        }
    }

    /// Draw the opaque geometry. `sets` is what
    /// [`begin_frame`](Self::begin_frame) built.
    /// `subsurface` picks the variant that splits the diffusible radiance into a
    /// second target. It is the graph's answer, not this pass's — the executor
    /// reads it off the frame's ids, so the pipeline bound here and the
    /// framebuffer already bound around it cannot disagree about how many
    /// attachments there are.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw(
        &self,
        device: &Device,
        builder: &mut Encoder,
        renderer: &PassCtx<'_>,
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
    ) {
        let Device::Vulkan(driver) = device;
        // The jittered one, from the frame's shared view: every pass that
        // rasterises geometry has to agree on it to a subpixel.
        let view_proj = view.view_proj;
        let (plain, masked) = self.pipelines(msaa).for_frame(subsurface);
        let mut words = [0u32; 16];
        for (word, value) in words
            .iter_mut()
            .zip(view_proj.to_cols_array_2d().iter().flatten())
        {
            *word = value.to_bits();
        }
        // The descriptor sets are rebound with the incoming pipeline. All four
        // layouts are built from the same stages and the same immutable shadow
        // sampler, so they are set compatible and this is a formality — but it
        // is the formality that keeps it true if one of them ever stops being.
        let groups: Vec<_> = sets
            .as_vec()
            .into_iter()
            .map(|set| driver.import_set(set))
            .collect();
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
        builder.bind_geometry(renderer.arena, false);

        // Nothing is bound yet, so the first run always binds. `None` rather
        // than "the plain one" so that a frame of nothing but foliage does not
        // begin by binding a pipeline it never draws with.
        let mut bound: Option<bool> = None;

        // A unit is one (mesh, material) pair, so each is one instanced draw:
        // the recording cost stops scaling with entity count and starts scaling
        // with distinct mesh/material pairs. Empty on the compute path, where
        // the regions below are what this pass records instead.
        for unit in draws.units() {
            let Some(mesh) = renderer.meshes.get(unit.mesh as usize) else {
                continue;
            };
            // A unit is one material, so this is one lookup per draw rather than
            // per item. The opaque order is grouped by (mesh, material), which
            // means cutouts are not a contiguous tail and this can flip more
            // than once — a pipeline bind per draw in the worst case, against a
            // list whose length is distinct mesh/material pairs.
            let wants_masked = renderer
                .materials
                .get(unit.material as usize)
                .is_some_and(GpuMaterial::is_masked);
            if bound != Some(wants_masked) {
                bind_all(builder, if wants_masked { masked } else { plain });
                bound = Some(wants_masked);
            }
            builder.draw_indexed(
                mesh.span.index_count,
                unit.instances,
                mesh.span.first_index,
                mesh.span.vertex_offset,
                unit.object_base,
            );
        }

        // The compute path's answer to the same question: one multi-draw per
        // pipeline variant, over commands a dispatch wrote. Empty on the CPU
        // path.
        for region in draws.regions() {
            bind_all(builder, if region.masked { masked } else { plain });
            builder.draw_indexed_indirect(driver.import_buffer(region.commands.into_bytes()));
        }
    }
}

/// The material table both geometry passes read.
///
/// One buffer rather than one per pipeline: the prepass writes the `f0` and
/// roughness the forward pass shades with, and two uploads of the same table are
/// two ways for one material to be described differently.
pub(super) fn material_buffer(
    ctx: &VkContext,
    materials: &[GpuMaterial],
) -> Subbuffer<[GpuMaterial]> {
    Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        materials.iter().copied(),
    )
    .expect("failed to allocate material buffer")
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
    }
}

/// Blend states for a pipeline sharing the forward render pass but writing only
/// its colour target — the skybox and the debug lines.
///
/// The subsurface variant of that render pass has a second colour attachment
/// those two shaders know nothing about, and Vulkan requires a blend state per
/// attachment regardless. Leaving the extra one at its default would let them
/// write whatever their fragment shader happened to leave in an output it never
/// declared, which is undefined — and the sky covers most of the frame, so the
/// undefined value would be sitting in the scatter target under every blur tap
/// near a silhouette. An empty write mask is the whole fix, and unlike disabling
/// the write it needs no device feature: the attachment keeps its clear, and a
/// clear of zero is exactly "nothing scattered here".
pub(super) fn co_tenant_blend_states(target: &PipelineRenderingCreateInfo) -> ColorBlendState {
    let mut state = ColorBlendState::with_attachment_states(
        target.color_attachment_formats.len() as u32,
        ColorBlendAttachmentState::default(),
    );
    for attachment in state.attachments.iter_mut().skip(1) {
        attachment.color_write_mask =
            vulkano::pipeline::graphics::color_blend::ColorComponents::empty();
    }
    state
}

/// The vertex shader both passes into the lit frame rasterise with. A blended
/// surface is the same geometry read from the same per-object rows, and
/// `shading.glsl` reads the same varyings out of it whichever fragment shader
/// includes it.
pub(super) fn vertex_shader(device: &Arc<VkDevice>) -> vulkano::shader::EntryPoint {
    vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/forward.vert",
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

    /// std430 derives an array's stride from its element, so the mirrors in
    /// `prepass.frag` and `shadow.frag` step through the table at whatever pitch
    /// this struct is — a mirror one block short reads every material past the
    /// first out of the middle of its neighbour. Nothing in either shader can
    /// notice, which is why the size is pinned here.
    #[test]
    fn the_material_block_is_the_size_both_shader_mirrors_declare() {
        // 15 vec4-sized blocks: base colour, emissive, params, texture indices,
        // the five lobes' blocks, the two subsurface blocks, parallax, alpha,
        // the extended indices, and the flags.
        assert_eq!(std::mem::size_of::<GpuMaterial>(), 15 * 16);
    }
}
