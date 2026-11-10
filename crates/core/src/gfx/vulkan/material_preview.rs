//! The inspector's material, drawn on a sphere by the renderer itself.
//!
//! The sphere is shaded by `forward.vert` and `forward.frag` through the
//! forward pipeline's own layout, against the frame's own material table and
//! texture array, so what the inspector shows is the material as the frame
//! draws it — every lobe, the normal and height maps, the cutout — rather than
//! a second description of it that could drift. The CPU thumbnails in
//! `gfx::thumbnail` remain for lists, where a picture per row has to cost
//! nothing; this is the one large view, of the one material being edited.
//!
//! **What it is lit by.** The scene's sky, ambient and environment yaw, so a
//! metal reflects the world it will be seen in, and a studio key light of its
//! own, so a material in a night scene is still visible. Everything else a
//! frame binds is bound neutral — no punctual light, decal, shadow, fog or
//! screen-space term — because the sphere stands in no scene and the frame's
//! screen-space targets describe a different screen. Exposure is metered at the
//! sphere ([`exposure`]) rather than borrowed from the viewport, whose adapted
//! value describes whatever the camera happens to be looking at.
//!
//! **What it costs.** Two passes on their own targets, sharing nothing with the
//! scene, so the scene's barrier plan is identical with them or without them.
//! They are in the graph only on frames where something the sphere shows has
//! changed — the material, a texture under it, the environment, the light
//! ([`Schedule`]) — so a preview left open costs the frame nothing, and
//! dragging a slider costs one 512² draw and one 256² resolve a frame.
//!
//! **Why three output images.** The overlay samples the result in a submission
//! of its own after the frame, which the next frame does not wait for; on the
//! async-compute path it is also drawn a frame late, over the frame before.
//! A write therefore goes to the image that neither of the last two writes
//! went to, and the overlay is handed the image its own frame wrote. With the
//! overlay's run-ahead bound of two, that image is never one still being read.
//!
//! Blended and transmissive materials are shaded as their surface alone: what
//! shows through glass is a queue this preview does not run.

use std::f32::consts::PI;

use glam::{Mat4, Vec3};

use super::cull::Variant;
use super::forward::{ForwardPass, ForwardSets};
use super::hdr::HDR_WIDE_FORMAT;
use super::swapchain::DEPTH_FORMAT;
use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::ImageUsage;
use crate::gfx::images::ImageEntry;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, CullMode, DepthTest, Device, Encoder,
    GraphicsPipelineDesc, MeshSpan, PipelineId, SamplerDesc, SamplerId, TextureDesc,
    TextureDimension, TextureViewId, Topology, Viewport,
};
use crate::gfx::sh::{self, SH9};
use crate::gfx::shaders::ShaderId;
use crate::gfx::{DirectionalLight, PositionVertex, SceneLighting, SurfaceVertex};
use crate::scene::{Camera, CpuMesh, HdrSettings, MaterialHandle};

/// The edge of the image the inspector shows, in pixels.
pub const SIZE: u32 = 256;
/// Samples per output pixel along each axis: the sphere is shaded at
/// `SIZE * SUPERSAMPLE` and averaged after tonemapping, which is what gives a
/// one-sample render an antialiased silhouette. `SUPERSAMPLE` in
/// `material_preview.frag`.
pub const SUPERSAMPLE: u32 = 2;
pub const OUTPUT_FORMAT: Format = Format::Rgba8Srgb;
/// The egui texture the inspector paints the sphere with.
pub const TEXTURE: egui::TextureId = egui::TextureId::User(0);

/// The studio key's illuminance, about a clear midday sun's: the same order as
/// a calibrated sky's irradiance, so neither swamps the other on the sphere.
pub(super) const KEY_LUX: f32 = 20_000.0;

/// The engine's built-in sphere: unit diameter, like every primitive.
const RADIUS: f32 = 0.5;
const FOV_Y_DEGREES: f32 = 30.0;
/// How much of the view's half-height the silhouette takes.
const FILL: f32 = 0.84;
/// Toward the key light, before normalising: above the camera and to its left.
const TO_KEY: Vec3 = Vec3::new(-0.55, 0.65, 0.75);
/// What an incident-light meter is calibrated to (ISO 2720's C for a flat
/// receptor). It puts a white diffuse surface at `C / (1.2 * 100 * pi)`, about
/// two thirds, after exposure.
const METER_CALIBRATION: f32 = 250.0;
const REC709_LUMINANCE: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);
/// Enough segments that the silhouette is round at 512 pixels.
const SECTORS: u32 = 96;
const STACKS: u32 = 48;
const OUTPUTS: usize = 3;

pub(super) fn camera() -> Camera {
    let fov_y = FOV_Y_DEGREES.to_radians();
    let distance = RADIUS / (FILL * fov_y / 2.0).sin();
    Camera {
        position: Vec3::new(0.0, 0.25, 1.0).normalize() * distance,
        target: Vec3::ZERO,
        up: Vec3::Y,
        fov_y,
        near: (distance - RADIUS) * 0.5,
        far: distance + 2.0 * RADIUS,
    }
}

pub(super) fn lighting(scene: &SceneLighting) -> SceneLighting {
    SceneLighting {
        sun: DirectionalLight {
            direction: -TO_KEY.normalize(),
            color: Vec3::ONE,
            illuminance: KEY_LUX,
        },
        point_lights: Vec::new(),
        spot_lights: Vec::new(),
        ..scene.clone()
    }
}

/// The exposure an incident-light meter at the sphere, facing the camera,
/// would set. The sky is read the way `sh_irradiance` reads it, through the
/// environment's yaw, so the meter and the sphere agree about which way the
/// bright half of a sky faces.
pub(super) fn exposure(lighting: &SceneLighting, irradiance: &[Vec3; SH9], yaw: f32) -> f32 {
    let to_camera = camera().position.normalize();
    let key = lighting.sun.illuminance * (-lighting.sun.direction).dot(to_camera).max(0.0);
    let (s, c) = yaw.to_radians().sin_cos();
    let in_environment = Vec3::new(
        c * to_camera.x - s * to_camera.z,
        to_camera.y,
        s * to_camera.x + c * to_camera.z,
    );
    let sky = PI * sh::evaluate(irradiance, in_environment).dot(REC709_LUMINANCE);
    let incident = (key + sky).max(f32::MIN_POSITIVE);
    HdrSettings::exposure_from_ev100((incident * 100.0 / METER_CALIBRATION).log2())
}

/// Everything the sphere's pixels depend on. `generation` stands for what the
/// rest cannot see: a material's parameters, a texture's texels and the
/// environment's cube change in place under handles that do not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Key<'a> {
    pub material: MaterialHandle,
    pub generation: u64,
    pub lighting: &'a [u8],
}

/// What the preview last drew, kept as owned bytes so asking costs nothing.
#[derive(Default)]
pub(super) struct Schedule {
    drawn: Option<(MaterialHandle, u64)>,
    lighting: Vec<u8>,
}

impl Schedule {
    pub(super) fn due(&self, key: Option<&Key<'_>>) -> bool {
        key.is_some_and(|key| {
            self.drawn != Some((key.material, key.generation)) || self.lighting != key.lighting
        })
    }

    pub(super) fn drawn(&mut self, key: &Key<'_>) {
        self.drawn = Some((key.material, key.generation));
        self.lighting.clear();
        self.lighting.extend_from_slice(key.lighting);
    }
}

/// One redraw: the sets the sphere binds, and how it is tonemapped.
pub(super) struct PreviewFrame {
    pub sets: ForwardSets,
    pub variant: Variant,
    pub exposure: f32,
}

pub(super) struct MaterialPreviewPass {
    /// By `Variant`: [plain, lobed, masked, masked and lobed].
    pipelines: [PipelineId; 4],
    resolve: PipelineId,
    nearest: SamplerId,
    sphere: MeshSpan,
    outputs: [TextureViewId; OUTPUTS],
    /// The output the overlay samples, and the one written before it.
    shown: usize,
    previous: usize,
    pub(super) schedule: Schedule,
}

impl MaterialPreviewPass {
    pub(super) fn new(device: &Device, forward: &ForwardPass) -> Self {
        let pipeline = |fragment, cull| {
            device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_streams: &[PositionVertex::STREAM, SurfaceVertex::STREAM],
                cull,
                depth_test: Some(DepthTest {
                    write: true,
                    compare: crate::gfx::depth::NEARER,
                }),
                depth_format: Some(DEPTH_FORMAT),
                topology: Topology::Triangles,
                // The forward pipeline's layout, so the frame's material and
                // texture sets bind here unchanged — see `oit.rs`.
                layout_from: Some(forward.layout_source()),
                ..GraphicsPipelineDesc::fullscreen(
                    ShaderId::ForwardVert,
                    fragment,
                    &[HDR_WIDE_FORMAT],
                )
            })
        };
        let pipelines = [
            pipeline(ShaderId::ForwardFragFlat, CullMode::Back),
            pipeline(ShaderId::ForwardFrag, CullMode::Back),
            pipeline(ShaderId::ForwardFragAlphaTestFlat, CullMode::None),
            pipeline(ShaderId::ForwardFragAlphaTest, CullMode::None),
        ];
        let resolve = device.create_graphics_pipeline(&GraphicsPipelineDesc::fullscreen(
            ShaderId::FullscreenVert,
            ShaderId::MaterialPreviewFrag,
            &[OUTPUT_FORMAT],
        ));
        let nearest = device.create_sampler(SamplerDesc {
            linear: false,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(0.0),
        });
        let mesh = CpuMesh::sphere(SECTORS, STACKS);
        let (sphere, _) = device.upload_mesh(&mesh.vertices, &mesh.indices);
        let outputs = std::array::from_fn(|_| {
            let view = device.create_texture(&TextureDesc {
                format: OUTPUT_FORMAT,
                dimension: TextureDimension::D2,
                extent: [SIZE, SIZE, 1],
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::Sample1,
                // Transfer as well, for `VulkanRenderer::read_material_preview`.
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::SAMPLED
                    | ImageUsage::TRANSFER_SRC,
                memoryless: false,
            });
            device.initialise_sampled(view);
            view
        });
        Self {
            pipelines,
            resolve,
            nearest,
            sphere,
            outputs,
            shown: 0,
            previous: OUTPUTS - 1,
            schedule: Schedule::default(),
        }
    }

    /// Move on to the output neither of the last two redraws wrote, and name
    /// it for the frame's resolve to write. See the module docs.
    pub(super) fn advance(&mut self) -> ImageEntry {
        let next = (0..OUTPUTS)
            .find(|slot| *slot != self.shown && *slot != self.previous)
            .expect("three outputs leave one free");
        self.previous = self.shown;
        self.shown = next;
        ImageEntry {
            whole: self.outputs[next],
            levels: Vec::new(),
            layers: Vec::new(),
            extent: [SIZE; 2],
            mip_levels: 1,
        }
    }

    /// What the overlay samples: the last output a frame wrote.
    pub(super) fn shown(&self) -> TextureViewId {
        self.outputs[self.shown]
    }

    pub(super) fn record_sphere(&self, encoder: &mut Encoder, frame: &PreviewFrame) {
        let camera = camera();
        let view_proj = camera.projection(1.0) * camera.view();
        let mut words = [0u32; 16];
        for (word, value) in words
            .iter_mut()
            .zip(view_proj.to_cols_array_2d().iter().flatten())
        {
            *word = value.to_bits();
        }
        let side = (SIZE * SUPERSAMPLE) as f32;
        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [side, side],
            depth_range: [0.0, 1.0],
        });
        let variant = frame.variant;
        encoder.bind_pipeline(self.pipelines[variant.masked as usize * 2 + variant.lobed as usize]);
        for (set, group) in frame.sets.groups().iter().enumerate() {
            encoder.bind_group(set as u32, *group);
        }
        encoder.push_constants(&words);
        encoder.bind_geometry(false);
        encoder.draw_indexed(
            self.sphere.index_count,
            1,
            self.sphere.first_index,
            self.sphere.vertex_offset,
            0,
        );
    }

    pub(super) fn record_resolve(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        shaded: TextureViewId,
        exposure: f32,
    ) {
        let set: BindGroupId = device.create_bind_group(
            self.resolve,
            0,
            &[Binding {
                binding: 0,
                resource: BindingResource::SampledTexture {
                    view: shaded,
                    sampler: self.nearest,
                },
            }],
        );
        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [SIZE as f32; 2],
            depth_range: [0.0, 1.0],
        });
        encoder.bind_pipeline(self.resolve);
        encoder.bind_group(0, set);
        encoder.push_constants(&[exposure.to_bits()]);
        encoder.draw(3, 1, 0, 0);
    }
}

/// The sphere's one object row, at the origin, and its one draw-order entry.
pub(super) fn object_rows(material: MaterialHandle) -> (super::instances::GpuObject, [u32; 2]) {
    (
        super::instances::GpuObject::new(&Mat4::IDENTITY, &Mat4::IDENTITY),
        [0, material.0],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::PointLight;

    const SPHERE_RADIUS: f32 = 0.5;

    /// The sphere's silhouette as a fraction of the half-height of the view:
    /// big enough to read a highlight, small enough that the edge where
    /// Fresnel and the rim live is never cut off.
    #[test]
    fn the_sphere_fills_most_of_the_preview_and_none_of_it_is_cut_off() {
        let camera = camera();
        assert_eq!(camera.target, Vec3::ZERO);
        let distance = camera.position.length();
        let silhouette = (SPHERE_RADIUS / distance).asin();
        let fill = silhouette / (camera.fov_y / 2.0);
        assert!((0.75..0.92).contains(&fill), "fills {fill}");
        assert!(camera.near < distance - SPHERE_RADIUS);
        assert!(camera.far > distance + SPHERE_RADIUS);
    }

    /// A studio key: on the side the camera sees and from above, so the
    /// terminator falls across the sphere rather than behind it, in white so
    /// the albedo reads as authored. The sky and the ambient are the scene's —
    /// a metal reflects the environment it will be seen in — and nothing
    /// punctual from the scene reaches a sphere that is not in it.
    #[test]
    fn the_key_light_comes_from_above_the_camera_and_the_sky_is_the_scenes() {
        let scene = SceneLighting {
            ambient_color: Vec3::new(0.2, 0.3, 0.9),
            ambient_nits: 42.0,
            sun: DirectionalLight {
                direction: Vec3::Y,
                color: Vec3::new(1.0, 0.2, 0.1),
                illuminance: 3.0,
            },
            point_lights: vec![PointLight {
                position: Vec3::ZERO,
                color: Vec3::ONE,
                candela: 1_000.0,
                range: 10.0,
                casts_shadows: false,
            }],
            ..SceneLighting::default()
        };
        let rig = lighting(&scene);
        assert_eq!(rig.ambient_color, scene.ambient_color);
        assert_eq!(rig.ambient_nits, scene.ambient_nits);
        assert!(rig.point_lights.is_empty() && rig.spot_lights.is_empty());

        let to_light = -rig.sun.direction;
        let to_camera = camera().position.normalize();
        assert!(to_light.dot(to_camera) > 0.5, "{to_light}");
        assert!(to_light.y > 0.3, "{to_light}");
        assert_eq!(rig.sun.color, Vec3::ONE);
        assert_eq!(rig.sun.illuminance, KEY_LUX);
    }

    /// An incident-light meter: whatever the sky, a white diffuse surface
    /// facing the camera lands at one exposed value, so a material is judged
    /// against the same white in every scene. The value is what a meter
    /// calibrated at C = 250 puts it at — 250 / (1.2 * 100 * pi).
    #[test]
    fn a_white_diffuse_surface_facing_the_camera_is_exposed_alike_under_any_sky() {
        let to_camera = camera().position.normalize();
        for sky in [0.0, 1.0, 600.0, 8_000.0] {
            let rig = lighting(&SceneLighting::default());
            let irradiance = sh::from_constant(Vec3::splat(sky));
            let incident = rig.sun.illuminance * (-rig.sun.direction).dot(to_camera).max(0.0)
                + std::f32::consts::PI * sky;
            let white = incident / std::f32::consts::PI;
            let exposed = white * exposure(&rig, &irradiance, 37.0);
            let expected = 250.0 / (1.2 * 100.0 * std::f32::consts::PI);
            assert!(
                (exposed - expected).abs() < 1e-3,
                "under a sky of {sky} nits: {exposed}"
            );
        }
    }

    /// The meter reads the sky the way the shader does — rotated by the
    /// environment's yaw — so turning a lopsided sky changes the exposure the
    /// way it changes what lights the sphere.
    #[test]
    fn the_meter_reads_the_sky_through_the_environments_yaw() {
        let rig = lighting(&SceneLighting::default());
        let mut lopsided = [Vec3::ZERO; SH9];
        lopsided[0] = Vec3::splat(400.0);
        lopsided[2] = Vec3::splat(300.0);
        let facing = exposure(&rig, &lopsided, 0.0);
        let turned = exposure(&rig, &lopsided, 180.0);
        assert!(
            (facing - turned).abs() > 1e-9,
            "{facing} and {turned} read one sky"
        );
    }

    fn key(material: u32, generation: u64, lighting: &[u8]) -> Key<'_> {
        Key {
            material: MaterialHandle(material),
            generation,
            lighting,
        }
    }

    /// Redrawn when anything it shows changed — the material, a texture under
    /// it, or the light — and otherwise kept, so a preview left open costs the
    /// frame nothing.
    #[test]
    fn the_preview_is_redrawn_only_when_what_it_shows_changed() {
        let mut schedule = Schedule::default();
        let (lit, relit) = ([0u8; 4], [9u8; 4]);
        assert!(!schedule.due(None));
        assert!(schedule.due(Some(&key(1, 0, &lit))));
        schedule.drawn(&key(1, 0, &lit));
        assert!(!schedule.due(Some(&key(1, 0, &lit))));
        assert!(!schedule.due(None));
        assert!(schedule.due(Some(&key(2, 0, &lit))));
        assert!(schedule.due(Some(&key(1, 1, &lit))));
        assert!(schedule.due(Some(&key(1, 0, &relit))));
    }
}
