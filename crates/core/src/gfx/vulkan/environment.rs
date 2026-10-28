//! The environment cubemap and the skybox that draws it.
//!
//! The bake deliberately does not go through the render graph. It runs once at
//! load time, its barrier chain is a straight line, and its output outlives
//! every frame — none of which the graph derives anything useful for. What the
//! frame sees is only the finished cube, which after the bake sits in
//! `ShaderReadOnlyOptimal` and never changes layout again.
//!
//! The skybox is a draw inside the forward render pass rather than a graph node
//! of its own, for the same reason the debug lines are: `msaa_hdr` is declared
//! `store_op: DontCare` and allocated lazily, so on a tile GPU it never reaches
//! DRAM. A separate pass would have to `Store` it and `Load` it back, which
//! trades the whole point of that allocation for a node in the schedule.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};

use crate::gfx::format::SampleCount;
use crate::gfx::graph::ImageUsage;
use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, DepthCompare, DepthTest, Device, Encoder,
    GraphicsPipelineDesc, PipelineId, SamplerDesc, SamplerId, TextureDesc, TextureDimension,
    TextureViewId, Viewport,
};
use crate::gfx::sh::{self, SH9};
use crate::gfx::shaders::ShaderId;
use crate::scene::EnvironmentSettings;

use super::MSAA_SAMPLES;
use super::forward::ForwardTargets;
use super::hdr::HDR_WIDE_FORMAT;
use super::taa::FrameView;

/// The environment carries the same radiance the scene does, so it uses the
/// same format the forward target does.
// The wide format, not the packed one the frame's colour uses. This is baked
// once when an environment is loaded rather than written every frame, so there
// is no per-frame bandwidth to win here — and the prefilter accumulates weighted
// taps through the alpha channel the packed format does not have.
pub const CUBE_FORMAT: crate::gfx::format::Format = HDR_WIDE_FORMAT;

/// Edge length of one cube face. Enough for a background at ordinary fields of
/// view; not enough for sharp mirror reflections, which is what would drive
/// raising it.
pub const FACE_SIZE: u32 = 512;

/// Levels in the prefiltered specular chain: 512 down to 16, roughness 0 to 1.
/// Not the full pyramid — past six levels the lobe is wider than the level has
/// texels to describe it, and the extra bake time buys nothing.
///
/// Keep in sync with `SPECULAR_MIPS` in `forward.frag`, which converts a
/// material's roughness into a level of this chain.
pub const SPECULAR_MIPS: u32 = 6;

/// Per-face `(forward, right, up)`, in cube layer order: +X, -X, +Y, -Y, +Z, -Z.
///
/// Derived from the cube face selection table in the Vulkan spec, inverted: for
/// face coordinates `(u, v)` in `[-1,1]` with `v` increasing *down* the face
/// image, the direction is `forward + u * right + v * up`. Two consequences
/// that look like mistakes and are not — the ±Y faces do not share the other
/// four's handedness, and `up` points along -Y for four of the six, because the
/// convention is left-handed and its `t` axis runs down the image.
///
/// A single transposed or mirrored face shows up only as a discontinuity at a
/// face edge, and would silently corrupt every irradiance and prefilter result
/// derived from the cube later. Changing anything here needs the round trip
/// checked: `direction_for(face, uv)` composed with a hardware cube sample must
/// be the identity.
const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
    ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
];

#[derive(Pod, Zeroable, Clone, Copy)]
#[repr(C)]
struct FacePush {
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
}

#[derive(Pod, Zeroable, Clone, Copy)]
#[repr(C)]
struct PrefilterPush {
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    /// x = perceptual roughness, y = source face size in texels.
    params: [f32; 4],
}

pub struct EnvironmentPass {
    bake_pipeline: PipelineId,
    prefilter_pipeline: PipelineId,
    equirect_sampler: SamplerId,
    /// The skybox against each of the forward pass's four target shapes,
    /// indexed `[msaa][subsurface]`. All four live for the session, for the
    /// reason [`ForwardTargets`](super::forward::ForwardTargets) documents.
    skybox_pipelines: [[PipelineId; 2]; 2],
    /// The cube sampler, as the RHI knows it, for the skybox draw.
    skybox_sampler: SamplerId,
    /// Bound in place of the prefiltered chain when nothing is loaded.
    fallback_cube: TextureViewId,
    /// The prefiltered specular chain. `None` until an environment is loaded,
    /// in which case the skybox is simply not recorded and the forward shader
    /// binds `fallback_cube` instead.
    cube: Option<TextureViewId>,
    /// Diffuse irradiance, projected from the equirect source on the CPU before
    /// it was ever uploaded. Held beside the cube because the two are derived
    /// from the same source and would be wrong to have disagree.
    irradiance: Option<[Vec3; SH9]>,
    /// The source's own sky luminance, in whatever units it carried, measured at
    /// bake. Everything sampled from this environment is scaled by the ratio
    /// between it and the cd/m² the scene asked for, so it is the other half of a
    /// calibration and belongs beside the maps it calibrates.
    measured_sky: f32,
}

impl EnvironmentPass {
    /// `targets` are the forward pass's: the skybox rasterises into the same
    /// attachments, after the geometry.
    pub fn new(rhi: &Device, targets: &ForwardTargets) -> Self {
        let bake = |fragment| {
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc::fullscreen(
                ShaderId::FullscreenVert,
                fragment,
                &[CUBE_FORMAT],
            ))
        };
        let bake_pipeline = bake(ShaderId::EquirectToCubeFrag);
        let prefilter_pipeline = bake(ShaderId::PrefilterCubeFrag);
        // Passed in rather than read back off a render pass object: a description
        // carries formats, not a sample count, and a pipeline whose count
        // disagrees with the attachments it is used with is invalid. The two
        // travel together from `ForwardTargets` for that reason.
        let skybox = |msaa: bool, subsurface: bool| {
            rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                // The triangle sits exactly on the far plane, so the test has to
                // accept equality or the sky is rejected by the depth clear it is
                // supposed to fill. Writing depth would make it occlude the debug
                // lines drawn after it.
                depth_test: Some(DepthTest {
                    write: false,
                    compare: DepthCompare::LessOrEqual,
                }),
                depth_format: Some(super::swapchain::DEPTH_FORMAT),
                samples: if msaa {
                    MSAA_SAMPLES
                } else {
                    crate::gfx::format::SampleCount::Sample1
                },
                // Masked past the first attachment: this shader declares one
                // output, and the subsurface target shape has two. The sky covers
                // most of the frame, so an undeclared output left unmasked would
                // be undefined data sitting under every blur tap near a
                // silhouette — see `GraphicsPipelineDesc::mask_extra_attachments`.
                mask_extra_attachments: true,
                ..GraphicsPipelineDesc::fullscreen(
                    ShaderId::SkyboxVert,
                    ShaderId::SkyboxFrag,
                    &targets.colors(msaa, subsurface),
                )
            })
        };
        let skybox_pipelines = [
            [skybox(false, false), skybox(false, true)],
            [skybox(true, false), skybox(true, true)],
        ];
        let skybox_sampler = rhi.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: true,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: None,
        });

        // Repeat in u so the seam wraps, clamp in v so the poles do not fold
        // across to the opposite hemisphere.
        let equirect_sampler = rhi.create_sampler_wrap_u(SamplerDesc {
            linear: true,
            mipmap_linear: false,
            repeat: true,
            compare: false,
            anisotropy: None,
            max_lod: Some(0.0),
        });

        Self {
            bake_pipeline,
            prefilter_pipeline,
            equirect_sampler,
            skybox_pipelines,
            skybox_sampler,
            fallback_cube: rhi.create_texture_cube(
                bytemuck::cast_slice(&[1.0f32; 24]),
                1,
                crate::gfx::format::Format::Rgba32Float,
            ),
            cube: None,
            irradiance: None,
            measured_sky: 0.0,
        }
    }

    /// The prefiltered specular chain, or the white fallback when nothing is
    /// loaded. Always a cube, so the forward pipeline's `textureCube` binding
    /// is satisfied without a second shader path.
    pub fn specular_view(&self) -> TextureViewId {
        self.cube.unwrap_or(self.fallback_cube)
    }

    /// Mipmapped, because the prefiltered roughness chain lands in the levels,
    /// and clamped on every axis.
    pub fn sampler(&self) -> SamplerId {
        self.skybox_sampler
    }

    /// What the sampled environment radiance is multiplied by. With an
    /// environment that is its calibration to cd/m²; without one it is the
    /// scene's flat ambient, which turns the 1x1 white fallback into a uniform
    /// environment of that colour — the same environment the band-0-only
    /// irradiance describes, so the diffuse and specular halves agree.
    pub fn specular_tint(&self, ambient: Vec3, settings: &EnvironmentSettings) -> Vec3 {
        match self.cube {
            Some(_) => Vec3::splat(settings.calibration(self.measured_sky)),
            None => ambient,
        }
    }

    /// The nine coefficients the forward shader evaluates, calibrated to cd/m².
    ///
    /// The same factor `specular_tint` returns, because the two halves describe
    /// one sky: calibrating the diffuse probe and the specular chain differently
    /// would make a rough surface and a smooth one disagree about how bright the
    /// world is.
    ///
    /// With no environment loaded this is `ambient` expressed as a band-0-only
    /// series, which evaluates to exactly `ambient` for every normal. So the
    /// flat ambient term is not a second path in the shader — it is the same
    /// path with nothing above the constant, the way an SSAO-less frame samples
    /// a 1x1 white image rather than branching.
    pub fn irradiance(&self, ambient: Vec3, settings: &EnvironmentSettings) -> [Vec3; SH9] {
        match self.irradiance {
            Some(coefficients) => {
                let calibration = settings.calibration(self.measured_sky);
                coefficients.map(|c| c * calibration)
            }
            None => sh::from_constant(ambient),
        }
    }

    /// Project an equirectangular source into a cubemap, replacing whatever was
    /// loaded before. `pixels` is tightly packed RGBA f32, row-major from the
    /// top-left, `width` by `height`.
    ///
    /// Blocks until the bake completes: it is a load-time operation, and the
    /// alternative is a half-written cube visible to the first frame.
    pub fn set_source(&mut self, device: &Device, pixels: &[f32], extent: [u32; 2]) {
        let expected = extent[0] as usize * extent[1] as usize * 4;
        assert_eq!(
            pixels.len(),
            expected,
            "equirect source is {} floats, expected {expected} for {}x{} RGBA",
            pixels.len(),
            extent[0],
            extent[1],
        );

        // Projected from the source rather than read back from the cube: the
        // pixels are already here, and a readback would be the only GPU-to-CPU
        // transfer anywhere in the engine.
        self.irradiance = Some(sh::project_equirect(pixels, extent));
        // Measured here rather than at load: the generated placeholder sky never
        // passes through `load_hdri`, and both sources have to be calibrated the
        // same way or the demo scene is the one scene the mechanism does not
        // cover.
        self.measured_sky = sh::sky_luminance(pixels, extent);

        let equirect = device.create_texture_2d(
            bytemuck::cast_slice(pixels),
            extent,
            crate::gfx::format::Format::Rgba32Float,
        );
        let cube = bake(
            device,
            self.bake_pipeline,
            self.prefilter_pipeline,
            self.equirect_sampler,
            self.skybox_sampler,
            equirect,
        );
        device.destroy_texture(equirect);
        if let Some(old) = self.cube.replace(cube) {
            device.destroy_texture(old);
        }
    }

    /// Record the skybox. Called at the end of the forward pass body, after the
    /// geometry, so the depth test rejects it everywhere something was drawn.
    #[allow(
        clippy::too_many_arguments,
        reason = "a pass records from the frame's bindings; a params struct only renames the list"
    )]
    pub fn record_skybox(
        &self,
        device: &Device,
        builder: &mut Encoder,
        view: &FrameView,
        extent: [u32; 2],
        settings: &EnvironmentSettings,
        subsurface: bool,
        msaa: bool,
        fog: BufferId,
        fog_volume: TextureViewId,
        fog_sampler: SamplerId,
    ) {
        let Some(cube) = self.cube else {
            return;
        };
        if !settings.show_skybox {
            return;
        }
        // Whichever render pass the executor opened around this call.
        let pipeline = self.skybox_pipelines[msaa as usize][subsurface as usize];

        let set = device.create_transient_bind_group(
            pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::Texture(cube),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::Sampler(self.skybox_sampler),
                },
                // The sky is fogged out of the same volume the geometry is, or
                // the horizon is a seam — see `skybox.frag`.
                Binding {
                    binding: 2,
                    resource: BindingResource::SampledTexture {
                        view: fog_volume,
                        sampler: fog_sampler,
                    },
                },
                Binding {
                    binding: 3,
                    resource: BindingResource::UniformBuffer(fog),
                },
            ],
        );

        // Translation stripped: what the vertex shader unprojects is then a
        // direction, so the ray needs no camera-relative correction and holds
        // its precision arbitrarily far from the origin.
        let view_rotation = Mat4::from_mat3(Mat3::from_mat4(view.view));
        // The jittered projection, like every other pass in this render pass:
        // a sky that ignored the jitter would sit still while the geometry
        // shook against it, and TAA would resolve a seam along every silhouette.
        let inverse = (view.proj * view_rotation).inverse();
        // Rotating the ray rotates the environment, so the yaw costs nothing in
        // the shader.
        let matrix = Mat4::from_rotation_y(settings.yaw.to_radians()) * inverse;

        let params = [
            settings.calibration(self.measured_sky),
            1.0 / extent[0] as f32,
            1.0 / extent[1] as f32,
            0.0,
        ];
        let mut words = [0u32; 20];
        for (word, float) in words
            .iter_mut()
            .zip(matrix.to_cols_array_2d().iter().flatten().chain(&params))
        {
            *word = float.to_bits();
        }

        builder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });
        builder.bind_pipeline(pipeline);
        builder.bind_group(0, set);
        builder.push_constants(&words);
        builder.draw(3, 1, 0, 0);
    }
}

/// Draw all six faces of one mip level, with per-face push constants.
fn render_level<P: Pod>(
    device: &Device,
    encoder: &mut Encoder,
    pipeline: PipelineId,
    set: crate::gfx::rhi::BindGroupId,
    cube: TextureViewId,
    mip: u32,
    push: impl Fn(usize) -> P,
) {
    let size = (FACE_SIZE >> mip).max(1) as f32;
    for face in 0..6u32 {
        let target = device.face_view(cube, face, mip);
        encoder.begin_rendering_fresh(target);
        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [size, size],
            depth_range: [0.0, 1.0],
        });
        encoder.bind_pipeline(pipeline);
        encoder.bind_group(0, set);
        let pushed = push(face as usize);
        encoder.push_constants(bytemuck::cast_slice(std::slice::from_ref(&pushed)));
        encoder.draw(3, 1, 0, 0);
        encoder.end_rendering();
        device.destroy_texture(target);
    }
}

fn create_cube(device: &Device, mip_levels: u32, usage: ImageUsage) -> TextureViewId {
    device.create_texture(&TextureDesc {
        format: CUBE_FORMAT,
        dimension: TextureDimension::Cube,
        extent: [FACE_SIZE, FACE_SIZE, 1],
        mip_levels,
        array_layers: 6,
        samples: SampleCount::Sample1,
        usage,
        memoryless: false,
    })
}

/// Project the equirect into a cubemap, then prefilter that into the specular
/// chain the forward shader samples.
///
/// Two images, not one: the prefilter reads a box-filtered pyramid of the
/// source while it writes the GGX-filtered pyramid of the result, and an image
/// cannot be both without splitting the work per subresource. The source is
/// dropped on return, so only the prefiltered chain is retained.
fn bake(
    device: &Device,
    project: PipelineId,
    prefilter: PipelineId,
    equirect_sampler: SamplerId,
    cube_sampler: SamplerId,
    equirect: TextureViewId,
) -> TextureViewId {
    let source = create_cube(
        device,
        FACE_SIZE.ilog2() + 1,
        ImageUsage::COLOR_ATTACHMENT
            | ImageUsage::SAMPLED
            | ImageUsage::TRANSFER_SRC
            | ImageUsage::TRANSFER_DST,
    );
    let specular = create_cube(
        device,
        SPECULAR_MIPS,
        ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
    );

    let project_set = device.create_transient_bind_group(
        project,
        0,
        &[Binding {
            binding: 0,
            resource: BindingResource::SampledTexture {
                view: equirect,
                sampler: equirect_sampler,
            },
        }],
    );

    // Split so the pyramid is complete before anything reads it: the prefilter
    // samples the source as a *cube* while the blit chain writes it level by
    // level, and the two views want it in different layouts.
    device.record_blocking(|encoder| {
        render_level(device, encoder, project, project_set, source, 0, |face| {
            let (forward, right, up) = FACES[face];
            FacePush {
                forward: [forward[0], forward[1], forward[2], 0.0],
                right: [right[0], right[1], right[2], 0.0],
                up: [up[0], up[1], up[2], 0.0],
            }
        });
        encoder.generate_cube_mips(source);
    });

    let prefilter_set = device.create_transient_bind_group(
        prefilter,
        0,
        &[
            Binding {
                binding: 0,
                resource: BindingResource::Texture(source),
            },
            Binding {
                binding: 1,
                resource: BindingResource::Sampler(cube_sampler),
            },
        ],
    );

    device.record_blocking(|encoder| {
        for mip in 0..SPECULAR_MIPS {
            // Level 0 is roughness 0 and the rest walk up to 1. The shader
            // samples this chain at `roughness * (SPECULAR_MIPS - 1)`, so the
            // mapping is stated in exactly these two places and nowhere else.
            let roughness = mip as f32 / (SPECULAR_MIPS - 1) as f32;
            render_level(
                device,
                encoder,
                prefilter,
                prefilter_set,
                specular,
                mip,
                |face| {
                    let (forward, right, up) = FACES[face];
                    PrefilterPush {
                        forward: [forward[0], forward[1], forward[2], 0.0],
                        right: [right[0], right[1], right[2], 0.0],
                        up: [up[0], up[1], up[2], 0.0],
                        params: [roughness, FACE_SIZE as f32, 0.0, 0.0],
                    }
                },
            );
        }
        encoder.publish_render_target(specular);
    });

    device.destroy_texture(source);
    specular
}

#[cfg(test)]
mod tests {
    use super::FACES;

    /// The cube face selection rules from the Vulkan spec: given a direction,
    /// which layer samples it and at what face coordinates. This is what the
    /// hardware does on a `textureCube` fetch, written out so the bake's
    /// direction table can be checked against it without a GPU.
    fn select_face(dir: [f32; 3]) -> (usize, f32, f32) {
        let [x, y, z] = dir;
        let (face, ma, sc, tc) = if x.abs() >= y.abs() && x.abs() >= z.abs() {
            if x > 0.0 {
                (0, x, -z, -y)
            } else {
                (1, -x, z, -y)
            }
        } else if y.abs() >= z.abs() {
            if y > 0.0 {
                (2, y, x, z)
            } else {
                (3, -y, x, -z)
            }
        } else if z > 0.0 {
            (4, z, x, -y)
        } else {
            (5, -z, -x, -y)
        };
        (face, sc / ma, tc / ma)
    }

    /// The bake writes face `f` by projecting `forward + u * right + v * up`.
    /// If the hardware would not fetch that direction from face `f` at exactly
    /// `(u, v)`, the cube is transposed, mirrored, or on the wrong layer — a
    /// defect that shows up only as a discontinuity at a face edge, and that
    /// silently corrupts every irradiance and prefilter result derived from the
    /// cube afterwards.
    #[test]
    fn every_face_basis_round_trips_through_hardware_cube_selection() {
        // Off-centre and asymmetric on purpose: (0,0) round-trips even for a
        // transposed basis, and a symmetric pair hides a mirrored one.
        let samples = [
            (0.0, 0.0),
            (0.5, 0.25),
            (-0.75, 0.5),
            (0.9, -0.6),
            (-0.3, -0.95),
        ];

        for (face, (forward, right, up)) in FACES.iter().enumerate() {
            for (u, v) in samples {
                let dir = [
                    forward[0] + u * right[0] + v * up[0],
                    forward[1] + u * right[1] + v * up[1],
                    forward[2] + u * right[2] + v * up[2],
                ];
                let (got_face, got_u, got_v) = select_face(dir);

                assert_eq!(
                    got_face, face,
                    "face {face} at ({u}, {v}) is sampled from layer {got_face}",
                );
                assert!(
                    (got_u - u).abs() < 1e-5 && (got_v - v).abs() < 1e-5,
                    "face {face}: wrote ({u}, {v}), hardware samples ({got_u}, {got_v})",
                );
            }
        }
    }
}
