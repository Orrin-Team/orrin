//! The inspector's material, drawn on a sphere by the renderer itself.
//!
//! The sphere is shaded by `forward.vert` and `forward.frag` through the
//! forward pipeline's own layout, against the frame's own material table and
//! texture array, so what the inspector shows is the material as the frame
//! draws it — every lobe, the normal and height maps, the cutout — rather than
//! a second description of it that could drift. The CPU thumbnails in
//! `gfx::thumbnail` remain for lists, where a picture per row has to cost
//! nothing; this is the one large view, of the one material being edited. A
//! material read through a surface shader is drawn by that shader's twin of
//! the sphere's pipeline (`surface`), which is what makes this the view a
//! shader is previewed in while it is written.
//!
//! **What it is lit by.** The scene's sky, ambient and environment yaw, so a
//! metal reflects the world it will be seen in, and a studio key light of its
//! own at a fixed ratio above that sky ([`KEY_TO_SKY`]), so the sphere always
//! has a lit and a shadowed side and an emitter is judged against the scene's
//! level of light rather than a midday sun's. Everything else a
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
//! dragging a slider or the sphere costs one 640² draw and one 320² resolve a
//! frame.
//!
//! **Who it is for.** Each editor view that shows one — the inspector's
//! Material section, the Import tab's material, the Shader tab's — is a
//! [`Viewer`] with images
//! and a schedule of its own, so two views can show two materials at once. A
//! frame redraws at most one of them, taking turns when both are due, which
//! keeps the passes in the graph to one pair however many views there are.
//!
//! **Orbiting.** Dragging a view moves the camera round the sphere, and the key
//! light with it, as a studio light on the camera would: what changes is the
//! sky the surface reflects and the side of its textures in view, never which
//! way the terminator falls.
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
use super::forward::{ForwardPass, ForwardSets, NeutralBlocks};
use super::hdr::HDR_WIDE_FORMAT;
use super::overlay::UserTexture;
use super::swapchain::DEPTH_FORMAT;
use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::ImageUsage;
use crate::gfx::images::ImageEntry;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, CullMode, DepthTest, Device, Encoder,
    GraphicsPipelineDesc, MeshSpan, PipelineId, SamplerDesc, SamplerId, ShaderModuleId, ShaderRef,
    TextureDesc, TextureDimension, TextureViewId, Topology, Viewport,
};
use crate::gfx::sh::{self, SH9};
use crate::gfx::shaders::ShaderId;
use crate::gfx::{DirectionalLight, PositionVertex, SceneLighting, SurfaceVertex};
use crate::scene::{Camera, CpuMesh, HdrSettings, MaterialHandle};

/// The edge of the image a view shows, in pixels: the inspector's largest,
/// 160 points, on a 2x display. A view sizes itself with [`display_side`], so
/// the picture is never shown larger than it was drawn.
pub const SIZE: u32 = 320;
/// Samples per output pixel along each axis: the sphere is shaded at
/// `SIZE * SUPERSAMPLE` and averaged after tonemapping, which is what gives a
/// one-sample render an antialiased silhouette. `SUPERSAMPLE` in
/// `material_preview.frag`.
pub const SUPERSAMPLE: u32 = 2;
pub const OUTPUT_FORMAT: Format = Format::Rgba8Srgb;

/// An editor view with a preview of its own.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Viewer {
    Inspector,
    Browser,
    Shader,
}

impl Viewer {
    pub const ALL: [Self; VIEWERS] = [Self::Inspector, Self::Browser, Self::Shader];

    /// The egui texture the view paints its sphere with.
    pub fn texture(self) -> egui::TextureId {
        UserTexture::MaterialPreview(self).id()
    }
}

const VIEWERS: usize = 3;

/// Where the camera stands round the sphere, in degrees: `yaw` about the
/// vertical and `pitch` above the view it starts from.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Orbit {
    pub yaw: f32,
    pub pitch: f32,
}

impl Orbit {
    /// Turned by a drag of `delta` points: the surface follows the pointer.
    pub fn dragged(self, delta: [f32; 2]) -> Self {
        const DEGREES_PER_POINT: f32 = 0.6;
        let lowest = -MAX_ELEVATION_DEGREES - ELEVATION_DEGREES;
        let highest = MAX_ELEVATION_DEGREES - ELEVATION_DEGREES;
        Self {
            yaw: (self.yaw - delta[0] * DEGREES_PER_POINT).rem_euclid(360.0),
            pitch: (self.pitch + delta[1] * DEGREES_PER_POINT).clamp(lowest, highest),
        }
    }
}

/// What a view asks to be shown this frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Subject {
    pub material: MaterialHandle,
    pub orbit: Orbit,
}

/// One request per [`Viewer`], indexed by it.
pub type Subjects = [Option<Subject>; VIEWERS];

/// Each view's picture as the overlay samples it.
pub(super) type Lent = [Option<(egui::TextureId, TextureViewId)>; VIEWERS];

/// The side, in points, to show a preview wanted at `points`: no larger than
/// [`SIZE`] pixels at this display's density.
pub fn display_side(points: f32, pixels_per_point: f32) -> f32 {
    points.min(SIZE as f32 / pixels_per_point)
}

/// The studio key's illuminance over the sky's, on a surface facing the key:
/// a lit side four times the shadowed one, the classic studio ratio, so the
/// sphere always reads as a sphere. Tied to the sky rather than fixed, so the
/// rig sits at the scene's level of light and an emitter is seen against what
/// it will be seen against there: a fixed midday key made a 600-nit panel look
/// dimmer than paper under a dusk sky it outshines.
const KEY_TO_SKY: f32 = 3.0;
/// What the key keeps under a black sky, where the ratio would leave nothing.
const MIN_KEY_LUX: f32 = 1.0;

/// The engine's built-in sphere: unit diameter, like every primitive.
const RADIUS: f32 = 0.5;
const FOV_Y_DEGREES: f32 = 30.0;
/// How far above the sphere's equator the camera starts, and how far an orbit
/// may take it: short of the poles, where "up" stops meaning anything.
const ELEVATION_DEGREES: f32 = 14.0;
const MAX_ELEVATION_DEGREES: f32 = 75.0;
/// How much of the view's half-height the silhouette takes.
const FILL: f32 = 0.84;
/// Toward the key light in the camera's frame — right, up, back toward the
/// camera — before normalising: above the camera and to its left.
const TO_KEY: Vec3 = Vec3::new(-0.55, 0.65, 0.75);
/// What an incident-light meter is calibrated to (ISO 2720's C for a flat
/// receptor). It puts a white diffuse surface at `C / (1.2 * 100 * pi)`, about
/// two thirds, after exposure.
const METER_CALIBRATION: f32 = 250.0;
const REC709_LUMINANCE: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);
/// Enough segments that the silhouette is round at 640 pixels.
const SECTORS: u32 = 96;
const STACKS: u32 = 48;
const OUTPUTS: usize = 3;

fn toward_camera(orbit: Orbit) -> Vec3 {
    let elevation = (ELEVATION_DEGREES + orbit.pitch)
        .clamp(-MAX_ELEVATION_DEGREES, MAX_ELEVATION_DEGREES)
        .to_radians();
    let (sin_yaw, cos_yaw) = orbit.yaw.to_radians().sin_cos();
    Vec3::new(
        sin_yaw * elevation.cos(),
        elevation.sin(),
        cos_yaw * elevation.cos(),
    )
}

fn toward_key(orbit: Orbit) -> Vec3 {
    let back = toward_camera(orbit);
    let right = Vec3::Y.cross(back).normalize();
    let up = back.cross(right);
    (right * TO_KEY.x + up * TO_KEY.y + back * TO_KEY.z).normalize()
}

pub(super) fn camera(orbit: Orbit) -> Camera {
    let fov_y = FOV_Y_DEGREES.to_radians();
    let distance = RADIUS / (FILL * fov_y / 2.0).sin();
    Camera {
        position: toward_camera(orbit) * distance,
        target: Vec3::ZERO,
        up: Vec3::Y,
        fov_y,
        near: (distance - RADIUS) * 0.5,
        far: distance + 2.0 * RADIUS,
    }
}

/// The sky's illuminance on a surface facing `direction`, read the way
/// `sh_irradiance` reads it — through the environment's yaw — so the rig and
/// the shader agree about which way the bright half of a sky faces.
fn sky_illuminance(irradiance: &[Vec3; SH9], yaw: f32, direction: Vec3) -> f32 {
    let (s, c) = yaw.to_radians().sin_cos();
    let in_environment = Vec3::new(
        c * direction.x - s * direction.z,
        direction.y,
        s * direction.x + c * direction.z,
    );
    PI * sh::evaluate(irradiance, in_environment).dot(REC709_LUMINANCE)
}

pub(super) fn lighting(
    scene: &SceneLighting,
    irradiance: &[Vec3; SH9],
    yaw: f32,
    orbit: Orbit,
) -> SceneLighting {
    let to_key = toward_key(orbit);
    let key = KEY_TO_SKY * sky_illuminance(irradiance, yaw, to_key);
    SceneLighting {
        sun: DirectionalLight {
            direction: -to_key,
            color: Vec3::ONE,
            illuminance: key.max(MIN_KEY_LUX),
        },
        point_lights: Vec::new(),
        spot_lights: Vec::new(),
        ..scene.clone()
    }
}

/// The exposure an incident-light meter at the sphere, facing the camera,
/// would set.
pub(super) fn exposure(
    lighting: &SceneLighting,
    irradiance: &[Vec3; SH9],
    yaw: f32,
    orbit: Orbit,
) -> f32 {
    let to_camera = toward_camera(orbit);
    let key = lighting.sun.illuminance * (-lighting.sun.direction).dot(to_camera).max(0.0);
    let sky = sky_illuminance(irradiance, yaw, to_camera);
    let incident = (key + sky).max(f32::MIN_POSITIVE);
    HdrSettings::exposure_from_ev100((incident * 100.0 / METER_CALIBRATION).log2())
}

/// Everything the sphere's pixels depend on. `generation` stands for what the
/// rest cannot see: a material's parameters, a texture's texels and the
/// environment's cube change in place under handles that do not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Key<'a> {
    pub material: MaterialHandle,
    pub orbit: Orbit,
    pub generation: u64,
    pub lighting: &'a [u8],
}

/// What the preview last drew, kept as owned bytes so asking costs nothing.
#[derive(Default)]
pub(super) struct Schedule {
    drawn: Option<(MaterialHandle, Orbit, u64)>,
    lighting: Vec<u8>,
}

impl Schedule {
    pub(super) fn due(&self, key: Option<&Key<'_>>) -> bool {
        key.is_some_and(|key| {
            self.drawn != Some((key.material, key.orbit, key.generation))
                || self.lighting != key.lighting
        })
    }

    pub(super) fn drawn(&mut self, key: &Key<'_>) {
        self.drawn = Some((key.material, key.orbit, key.generation));
        self.lighting.clear();
        self.lighting.extend_from_slice(key.lighting);
    }
}

/// The view to redraw this frame among those `due`, starting from `turn` so
/// that two views both due take turns rather than one starving the other.
pub(super) fn next_due(due: [bool; VIEWERS], turn: usize) -> Option<Viewer> {
    (0..VIEWERS)
        .map(|step| (turn + step) % VIEWERS)
        .find(|&index| due[index])
        .map(|index| Viewer::ALL[index])
}

/// The sphere's pipeline, built-in or a surface shader's twin of it.
fn sphere_desc(fragment: ShaderRef, cull: CullMode) -> GraphicsPipelineDesc<'static> {
    GraphicsPipelineDesc {
        vertex_streams: &[PositionVertex::STREAM, SurfaceVertex::STREAM],
        cull,
        depth_test: Some(DepthTest {
            write: true,
            compare: crate::gfx::depth::NEARER,
        }),
        depth_format: Some(DEPTH_FORMAT),
        topology: Topology::Triangles,
        ..GraphicsPipelineDesc::fullscreen(ShaderId::ForwardVert, fragment, &[HDR_WIDE_FORMAT])
    }
}

/// A surface shader's twin of the sphere's plain or cutout pipeline: the
/// forward layout's five sets lifted, and set 5 for the parameters. A free
/// function because the preview pass is built on first use and a twin may be
/// asked for first.
pub(super) fn build_twin(
    device: &Device,
    forward: &ForwardPass,
    masked: bool,
    fragment: ShaderModuleId,
) -> Result<PipelineId, String> {
    device.try_create_graphics_pipeline(&GraphicsPipelineDesc {
        layout_prefix_from: Some((forward.layout_source(), 5)),
        ..sphere_desc(
            fragment.into(),
            if masked {
                CullMode::None
            } else {
                CullMode::Back
            },
        )
    })
}

/// One redraw: the sets the sphere binds, where it is seen from, and how it
/// is tonemapped.
pub(super) struct PreviewFrame {
    pub sets: ForwardSets,
    pub variant: Variant,
    /// A surface shader's twin and the set 5 it binds, when one reads the
    /// material; the variant's built-in pipeline otherwise.
    pub surface: Option<(PipelineId, BindGroupId)>,
    pub view_proj: Mat4,
    pub exposure: f32,
}

/// A view's images, created the first time it is drawn, and what it last drew.
#[derive(Default)]
struct Slot {
    outputs: Option<[TextureViewId; OUTPUTS]>,
    /// The output the overlay samples, and the one written before it.
    shown: usize,
    previous: usize,
    schedule: Schedule,
}

pub(super) struct MaterialPreviewPass {
    /// By `Variant`: [plain, lobed, masked, masked and lobed].
    pipelines: [PipelineId; 4],
    resolve: PipelineId,
    nearest: SamplerId,
    sphere: MeshSpan,
    neutral: NeutralBlocks,
    slots: [Slot; VIEWERS],
    /// The view first in line when more than one is due.
    turn: usize,
}

impl MaterialPreviewPass {
    pub(super) fn new(device: &Device, forward: &ForwardPass) -> Self {
        let pipeline = |fragment: ShaderId, cull| {
            device.create_graphics_pipeline(&GraphicsPipelineDesc {
                // The forward pipeline's layout, so the frame's material and
                // texture sets bind here unchanged — see `oit.rs`.
                layout_from: Some(forward.layout_source()),
                ..sphere_desc(fragment.into(), cull)
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
        Self {
            pipelines,
            resolve,
            nearest,
            sphere,
            neutral: forward.neutral_blocks(device),
            slots: Default::default(),
            turn: 0,
        }
    }

    pub(super) fn neutral(&self) -> &NeutralBlocks {
        &self.neutral
    }

    pub(super) fn due(&self, viewer: Viewer, key: &Key<'_>) -> bool {
        self.slots[viewer as usize].schedule.due(Some(key))
    }

    /// Which of the views `due` this frame redraws, passing the turn on.
    pub(super) fn choose(&mut self, due: [bool; VIEWERS]) -> Option<Viewer> {
        let viewer = next_due(due, self.turn)?;
        self.turn = viewer as usize + 1;
        Some(viewer)
    }

    /// Count `key` as drawn for `viewer`, move on to the output neither of its
    /// last two redraws wrote, and name that for the frame's resolve. See the
    /// module docs.
    pub(super) fn advance(&mut self, device: &Device, viewer: Viewer, key: &Key<'_>) -> ImageEntry {
        let slot = &mut self.slots[viewer as usize];
        slot.schedule.drawn(key);
        let outputs = *slot.outputs.get_or_insert_with(|| {
            std::array::from_fn(|_| {
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
            })
        });
        let next = (0..OUTPUTS)
            .find(|index| *index != slot.shown && *index != slot.previous)
            .expect("three outputs leave one free");
        slot.previous = slot.shown;
        slot.shown = next;
        ImageEntry {
            whole: outputs[next],
            levels: Vec::new(),
            layers: Vec::new(),
            extent: [SIZE; 2],
            mip_levels: 1,
        }
    }

    /// What the overlay samples for `viewer`: the last output a frame wrote,
    /// or nothing before its first redraw.
    pub(super) fn shown(&self, viewer: Viewer) -> Option<TextureViewId> {
        let slot = &self.slots[viewer as usize];
        Some(slot.outputs?[slot.shown])
    }

    /// Every view's picture, as the overlay is lent them.
    pub(super) fn lent(&self) -> Lent {
        Viewer::ALL.map(|viewer| Some((viewer.texture(), self.shown(viewer)?)))
    }

    pub(super) fn record_sphere(&self, encoder: &mut Encoder, frame: &PreviewFrame) {
        let view_proj = frame.view_proj;
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
        let lobed = variant.surface != super::cull::Surface::Flat;
        let built_in = self.pipelines[variant.masked as usize * 2 + lobed as usize];
        encoder.bind_pipeline(frame.surface.map_or(built_in, |(pipeline, _)| pipeline));
        for (set, group) in frame.sets.groups().iter().enumerate() {
            encoder.bind_group(set as u32, *group);
        }
        if let Some((_, group)) = frame.surface {
            encoder.bind_group(crate::gfx::surface::PARAMS_BINDING.0, group);
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
        let camera = camera(Orbit::default());
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
        let sky = sh::from_constant(Vec3::splat(50.0));
        let rig = lighting(&scene, &sky, 0.0, Orbit::default());
        assert_eq!(rig.ambient_color, scene.ambient_color);
        assert_eq!(rig.ambient_nits, scene.ambient_nits);
        assert!(rig.point_lights.is_empty() && rig.spot_lights.is_empty());

        let to_light = -rig.sun.direction;
        let to_camera = camera(Orbit::default()).position.normalize();
        assert!(to_light.dot(to_camera) > 0.5, "{to_light}");
        assert!(to_light.y > 0.3, "{to_light}");
        assert_eq!(rig.sun.color, Vec3::ONE);
    }

    /// The key is a fixed ratio above the sky it shares the sphere with, so
    /// the sphere always has a lit side and a shadowed one, and the whole rig
    /// sits at the scene's level of light. A black sky still gets a key.
    #[test]
    fn the_key_stands_a_fixed_ratio_above_the_sky() {
        for nits in [1.0, 30.0, 6_000.0] {
            let sky = sh::from_constant(Vec3::splat(nits));
            let rig = lighting(&SceneLighting::default(), &sky, 0.0, Orbit::default());
            let ratio = rig.sun.illuminance / (PI * nits);
            assert!((ratio - KEY_TO_SKY).abs() < 1e-3, "{nits} nits: {ratio}");
        }
        let black = sh::from_constant(Vec3::ZERO);
        let rig = lighting(&SceneLighting::default(), &black, 0.0, Orbit::default());
        assert_eq!(rig.sun.illuminance, MIN_KEY_LUX);
    }

    /// What an emitter looks like depends on the light around it. Under the
    /// default dusk sky a 600-nit panel outshines a lit white surface, as it
    /// does in such a scene; under a daylight sky it does not.
    #[test]
    fn an_emitter_glows_against_a_dim_sky_and_not_against_a_bright_one() {
        let exposed = |sky_nits: f32, nits: f32| {
            let sky = sh::from_constant(Vec3::splat(sky_nits));
            let rig = lighting(&SceneLighting::default(), &sky, 0.0, Orbit::default());
            nits * exposure(&rig, &sky, 0.0, Orbit::default())
        };
        let white = 250.0 / (1.2 * 100.0 * PI);
        assert!(
            exposed(30.0, 600.0) > 2.0 * white,
            "{}",
            exposed(30.0, 600.0)
        );
        assert!(
            exposed(6_000.0, 600.0) < white,
            "{}",
            exposed(6_000.0, 600.0)
        );
    }

    /// An incident-light meter: whatever the sky, a white diffuse surface
    /// facing the camera lands at one exposed value, so a material is judged
    /// against the same white in every scene. The value is what a meter
    /// calibrated at C = 250 puts it at — 250 / (1.2 * 100 * pi).
    #[test]
    fn a_white_diffuse_surface_facing_the_camera_is_exposed_alike_under_any_sky() {
        let to_camera = camera(Orbit::default()).position.normalize();
        for sky in [0.0, 1.0, 600.0, 8_000.0] {
            let irradiance = sh::from_constant(Vec3::splat(sky));
            let rig = lighting(
                &SceneLighting::default(),
                &irradiance,
                37.0,
                Orbit::default(),
            );
            let incident = rig.sun.illuminance * (-rig.sun.direction).dot(to_camera).max(0.0)
                + std::f32::consts::PI * sky;
            let white = incident / std::f32::consts::PI;
            let exposed = white * exposure(&rig, &irradiance, 37.0, Orbit::default());
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
        let rig = lighting(
            &SceneLighting::default(),
            &sh::from_constant(Vec3::splat(400.0)),
            0.0,
            Orbit::default(),
        );
        let mut lopsided = [Vec3::ZERO; SH9];
        lopsided[0] = Vec3::splat(400.0);
        lopsided[2] = Vec3::splat(300.0);
        let facing = exposure(&rig, &lopsided, 0.0, Orbit::default());
        let turned = exposure(&rig, &lopsided, 180.0, Orbit::default());
        assert!(
            (facing - turned).abs() > 1e-9,
            "{facing} and {turned} read one sky"
        );
    }

    fn key(material: u32, generation: u64, lighting: &[u8]) -> Key<'_> {
        Key {
            material: MaterialHandle(material),
            orbit: Orbit::default(),
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
        let turned = Key {
            orbit: Orbit {
                yaw: 30.0,
                pitch: 0.0,
            },
            ..key(1, 0, &lit)
        };
        assert!(schedule.due(Some(&turned)));
    }

    /// The camera orbits at the distance that frames the sphere, so a turn
    /// never crops it, and the key turns with the camera: the terminator falls
    /// where it did, and what moves is the sky in the reflection.
    #[test]
    fn orbiting_moves_the_camera_round_the_sphere_and_the_key_with_it() {
        let rest = Orbit::default();
        let sky = sh::from_constant(Vec3::splat(30.0));
        let to_light = |orbit| {
            -lighting(&SceneLighting::default(), &sky, 0.0, orbit)
                .sun
                .direction
        };
        for orbit in [
            Orbit {
                yaw: 90.0,
                pitch: 0.0,
            },
            Orbit {
                yaw: 210.0,
                pitch: 40.0,
            },
            Orbit {
                yaw: 0.0,
                pitch: -50.0,
            },
        ] {
            let (from, turned) = (camera(rest), camera(orbit));
            assert!((from.position.length() - turned.position.length()).abs() < 1e-5);
            assert!(turned.position.distance(from.position) > 0.1, "{orbit:?}");
            let lit = |orbit, camera: &Camera| to_light(orbit).dot(camera.position.normalize());
            assert!(
                (lit(rest, &from) - lit(orbit, &turned)).abs() < 1e-5,
                "{orbit:?}: the key keeps its angle to the camera"
            );
        }
    }

    /// A drag turns the surface the way the pointer moves, and stops short of
    /// the poles however far it goes.
    #[test]
    fn a_drag_orbits_and_never_reaches_a_pole() {
        let right = Orbit::default().dragged([10.0, 0.0]);
        assert!(
            toward_camera(right).x < 0.0,
            "dragging right swings the camera left"
        );
        let over = Orbit::default().dragged([0.0, 10_000.0]);
        let elevation = toward_camera(over).y.asin().to_degrees();
        assert!(
            (elevation - MAX_ELEVATION_DEGREES).abs() < 1e-3,
            "{elevation}"
        );
        let back = over.dragged([0.0, -1.0]);
        assert!(
            toward_camera(back).y < toward_camera(over).y,
            "no slack to unwind past the pole"
        );
        assert!((0.0..360.0).contains(&Orbit::default().dragged([1e4, 0.0]).yaw));
    }

    /// Two views due in the same frames take turns; one alone has every frame.
    #[test]
    fn views_due_together_take_turns() {
        assert_eq!(next_due([false, false, false], 0), None);
        assert_eq!(next_due([false, true, false], 0), Some(Viewer::Browser));
        assert_eq!(next_due([true, false, false], 1), Some(Viewer::Inspector));
        assert_eq!(next_due([true, true, false], 0), Some(Viewer::Inspector));
        assert_eq!(next_due([true, true, false], 1), Some(Viewer::Browser));
        assert_eq!(next_due([true, true, false], 2), Some(Viewer::Inspector));
        assert_eq!(next_due([true, false, true], 1), Some(Viewer::Shader));
        assert_eq!(next_due([true, true, true], 3), Some(Viewer::Inspector));
    }

    /// A view is never shown larger than the picture was drawn, at any
    /// display density.
    #[test]
    fn a_view_is_sized_to_the_pixels_drawn() {
        assert_eq!(display_side(160.0, 2.0), 160.0);
        assert_eq!(display_side(160.0, 1.0), 160.0);
        assert_eq!(display_side(160.0, 3.0), SIZE as f32 / 3.0);
        assert_eq!(display_side(96.0, 3.0), 96.0);
    }
}
