//! The courtyard: the same materials as the rig, arranged as somewhere.
//!
//! Where [`build_default_scene`](super::build_default_scene) separates variables,
//! this composes them, and the two goals genuinely conflict — which is why this
//! is a second scene rather than a restyling of the first. A readout wants a flat
//! floor, an even light and one feature per object; a photograph wants a low sun,
//! an occluder in front of it, and every feature contributing to one image at
//! once. Neither scene can be graded against the other.
//!
//! # What the frame is built around
//!
//! One decision drives the rest: the sun is **low and the camera looks into it**.
//! That is the lighting a volumetric medium is visible in — the Henyey-Greenstein
//! term at `g = 0.75` is about four times its isotropic value at the 26° this
//! camera sits off the sun, and a quarter of it at 90° — and it is the lighting
//! the transmitted subsurface lobe is visible in, because light that crossed a
//! body reaches the eye only when the body is between the eye and the light. So
//! the shafts through the gateway and the glow through the wax are the same
//! choice, not two.
//!
//! Everything else about the layout falls out of one consequence of that choice,
//! and it took a render to take seriously: **in contre-jour the only surface that
//! is both lit and visible is the ground.** A face is visible if it points at the
//! camera and lit if it points at the sun, and here those directions are 150°
//! apart, so the set of normals satisfying both is a thin lune around
//! `forward × sunward` — which, with a camera looking level and a sun 22° up, is
//! very nearly straight up. Hence: a low camera, a floor that fills a third of
//! the frame, and walls kept to four metres so the sun clears them and *lands*
//! on that floor instead of shadowing all sixteen metres of it. The first version
//! of this scene had seven metre walls and a 14° sun, and its courtyard was
//! uniformly sky-lit grey with a blown sky above it.
//!
//! The rest of the frame is what contre-jour is actually good at: silhouettes,
//! rim light, translucency, and the raked relief of the paving.
//!
//! # The second camera
//!
//! What this lighting cannot do is show a wall. `showcase-materials` therefore
//! stands with the sun behind it, looking along the one surface built to be seen
//! that way — the brick pier, whose face points into the sun. That is where the
//! parallax march and the decal sitting in it are legible. It is deliberately not
//! a prettier angle on the same shot.

use std::f32::consts::FRAC_PI_2;

use glam::{Quat, Vec3};

use orrin_ecs::World;

use super::library::{WALL_TILE, load_library};
use super::textures::{hash01, sky_equirect};
use super::{
    spawn_decal, spawn_directional_light, spawn_mesh, spawn_point_light, spawn_spot_light,
};
use crate::gfx::RenderBackend;
use crate::scene::{
    Camera, Decal, DofSettings, EnvironmentSettings, FogSettings, HdrSettings, MaterialHandle,
    MeshHandle, Transform,
};

/// Where the sunlight *travels*: low, heading `+x` and `+z`, so the sun itself is
/// behind the courtyard's far-left corner and everything the camera faces is
/// backlit.
///
/// 22.3° of elevation, and the elevation is the load-bearing number rather than
/// the azimuth. It sets how much floor the walls shadow — `height / tan(22.3°)`
/// is 9.8 m for a four metre wall, so a sixteen metre courtyard is lit across
/// rather more than half its width. At the 14° this started at, the same wall
/// shadowed twenty-eight metres and there was no lit ground anywhere in frame.
///
/// Normalised at use, so editing one component tilts the sun rather than silently
/// rescaling it.
const SUN: Vec3 = Vec3::new(0.798, -0.379, 0.469);

/// The floor's extent in `x` and `z`. The courtyard is open on its `+z` side,
/// which is where the camera stands.
const FLOOR: [[f32; 2]; 2] = [[-8.0, -9.0], [8.0, 7.0]];

/// Edge of one flagstone, in metres — and so the metres of wall the rock maps are
/// stretched over, since each stone is one quad carrying one tile. Two metres puts
/// the 1K maps at 512 texels a metre, and keeps the marched relief a legible
/// three centimetres rather than a ripple across a floor-sized tile.
const STONE: f32 = 2.0;

/// Wall thickness. Load-bearing for the look rather than the physics: it is what
/// gives the gateway a jamb for the sun to rake across, and a wall with none
/// reads as paper the moment a shaft passes through it.
const THICK: f32 = 0.5;

/// The gateway's opening in the left wall, as `z` bounds and a head height. The
/// sun enters here, so these three numbers decide where the beam lands: at 13.9°
/// through a 3.4 m head, the pool is about eleven metres downwind, which is what
/// puts it in the middle of the courtyard instead of against the wall it came
/// through.
const GATE: [f32; 2] = [-6.4, -3.2];
const GATE_HEAD: f32 = 3.2;

/// Everything drawn here, resolved once. Fetched up front because the spawn
/// helpers take `&mut World` and the handles live in a resource inside it — one
/// borrow at the top is simpler than a `resource::<Assets>()` at every site.
struct Kit {
    cube: MeshHandle,
    quad: MeshHandle,
    sphere: MeshHandle,
    stone: MaterialHandle,
    flagstone: MaterialHandle,
    brick: MaterialHandle,
    foliage: MaterialHandle,
    glass: MaterialHandle,
    crystal: MaterialHandle,
    frosted: MaterialHandle,
    marble: MaterialHandle,
    wax: MaterialHandle,
    ember: MaterialHandle,
}

pub fn build_showcase_scene(world: &mut World, backend: &mut impl RenderBackend) {
    let (assets, mesh_bounds, material_blends, asset_previews) = load_library(backend);

    let kit = Kit {
        // The neutral primitives throughout. The rig's cube and sphere carry
        // `normal * 0.5 + 0.5` in vertex colour, which `read_surface` multiplies
        // into albedo — a readout of which way a face points, and a wall that is
        // lavender on one side and mint on top.
        cube: assets.mesh("cube_plain").unwrap(),
        quad: assets.mesh("plane_plain").unwrap(),
        sphere: assets.mesh("sphere_plain").unwrap(),
        stone: assets.material("stone").unwrap(),
        flagstone: assets.material("rock").unwrap(),
        brick: assets.material("brick").unwrap(),
        foliage: assets.material("foliage").unwrap(),
        glass: assets.material("glass").unwrap(),
        crystal: assets.material("crystal").unwrap(),
        frosted: assets.material("frosted").unwrap(),
        marble: assets.material("marble").unwrap(),
        wax: assets.material("wax").unwrap(),
        ember: assets.material("ember").unwrap(),
    };

    let scorch_albedo = assets.texture("scorch_albedo").unwrap();
    let scorch_normal = assets.texture("scorch_normal").unwrap();
    let scorch_rough = assets.texture("scorch_rough").unwrap();

    world.insert_resource(assets);
    world.insert_resource(mesh_bounds);
    world.insert_resource(material_blends);
    world.insert_resource(asset_previews);

    paving(world, &kit);
    walls(world, &kit);
    gateway(world, &kit);
    arcade(world, &kit);
    pier(world, &kit);
    facing(world, &kit);
    artefacts(world, &kit);
    overgrowth(world, &kit);
    lights(world, &kit, backend);

    // Two on the flagstones and one on the brick pier, which is the pairing the
    // rig makes for the same reason: a decal on a marched height field is the
    // sharpest check that the prepass and the forward pass agree about where a
    // surface *is*, and one on open ground is the only place its own crater
    // catches a light.
    let mark = Decal {
        albedo: Some(scorch_albedo),
        normal: Some(scorch_normal),
        metallic_roughness: Some(scorch_rough),
        affects_surface: true,
        angle: 55.0,
        ..Decal::default()
    };
    // In the sunbeam's own pool and beside the fire — one lit by the sun and one
    // by a 20 000 lumen fixture, which is the whole argument for projecting a
    // decal instead of drawing it over the frame.
    for (index, (x, z, size, opacity)) in [(2.0f32, 2.6f32, 4.0f32, 0.85f32), (-5.0, 0.6, 2.8, 1.0)]
        .into_iter()
        .enumerate()
    {
        spawn_decal(
            world,
            format!("Scorch {index}"),
            Transform {
                translation: Vec3::new(x, 0.0, z),
                rotation: Quat::from_rotation_x(-FRAC_PI_2),
                scale: Vec3::new(size, size, size * 0.4),
            },
            Decal { opacity, ..mark },
        );
    }
    spawn_decal(
        world,
        "Scorch (pier)",
        Transform {
            // Three tenths of a metre off the brick, with a box 0.8 deep along
            // its axis, so it reaches the panel from in front of it. Forward is
            // `-z`, and a quarter turn clockwise about `y` aims that at `+x`,
            // which is into the pier's lit face.
            translation: Vec3::new(0.98, 1.5, -2.4),
            rotation: Quat::from_rotation_y(-FRAC_PI_2),
            scale: Vec3::new(1.5, 1.5, 0.8),
        },
        Decal {
            opacity: 0.7,
            ..mark
        },
    );

    // The air, and the reason the sun is where it is.
    //
    // 0.008 per metre, which is much thinner than it sounds like it should be and
    // was arrived at by looking: at 0.025 the courtyard was a milky wash with the
    // walls barely in it. Two things compound at this camera that do not compound
    // in the rig. The froxels and the analytic tail integrate the *same* ray, so
    // the twenty metres to the far wall are all of them; and the phase function
    // is pointed almost straight at the lens, where `g = 0.75` multiplies the
    // inscatter several times over — the anisotropy that makes a shaft visible
    // makes a haze opaque at the same density.
    //
    // The falloff is where the sky comes from, which is not obvious and cost a
    // render to find. Density alone cannot open the sky up: the analytic tail
    // integrates to the far plane, so *any* density saturates over a kilometre and
    // the whole upper half of the frame becomes inscattered sunlight rather than
    // sky. What bounds an upward ray is the scale height — at `1 / 0.10` = 10 m a
    // ray leaving at 20° crosses about 30 m of medium and comes out at a
    // transmittance the sky is still visible through, while a froxel at the two or
    // three metres a shaft is legible at has lost almost none of its density.
    world.insert_resource(FogSettings {
        density: 0.014,
        height_falloff: 0.10,
        anisotropy: 0.75,
        volumetric: true,
        ..FogSettings::default()
    });

    // A stop and a third down. Metering is doing exactly what it should and the
    // scene is the awkward case for it: contre-jour means most of the frame is
    // shadow at a few tens of cd/m², so the measured average sits far below the
    // sunlit paving and the sky, and both clip. This is the ±EV dial, which is the
    // control a photographer reaches for in the same situation and for the same
    // reason — it is not a correction to the metering.
    world.insert_resource(HdrSettings {
        exposure_compensation: -1.0,
        ..HdrSettings::default()
    });

    // Deliberately off, and worth saying why in a scene that exists to look
    // good. At this field of view the lens is a 29 mm, and 29 mm at f/2.8
    // focused eight metres out is hyperfocal — everything from four metres to
    // the horizon is inside the circle of confusion, so the four defocus passes
    // would render an image indistinguishable from the one without them. Defocus
    // is a close-subject effect; a camera two metres from the fire is where to
    // turn it on.
    world.insert_resource(DofSettings::default());

    // Where the shot is composed from: standing at the open side, looking across
    // the paving with the sun over the far-left corner.
    //
    // Aimed **51° off the sun**, and that number is the one thing here that was
    // chosen rather than found. Closer in is a brighter medium — the phase function
    // at 26° is four times isotropic against 0.8 at 51° — but it also puts the
    // disc in shot, and the disc is 40 000 cd/m² against sunlit stone at 1 900, so
    // it has to be hidden behind something. Two versions of this scene were
    // composed around an occluder for it: one behind a seven metre wall that
    // shadowed the whole courtyard, one behind an arch that pinned the camera six
    // metres from the artefacts and made them fill the frame. Aiming past the sun
    // instead costs a factor of five in the air's brightness and buys the camera
    // its freedom back, which is the better trade — a shaft is *contrast* between
    // lit and shadowed air, and that ratio is the sun-to-sky ratio at any phase
    // angle. The fog's density carries the difference.
    //
    // 45° rather than the default 60°. A wider frame is not more of the courtyard
    // here, it is more of the two walls immediately beside the lens — and it would
    // bring the sun back into shot.
    world.insert_resource(Camera {
        position: Vec3::new(4.4, 1.75, 9.5),
        target: Vec3::new(0.65, 1.0, -4.75),
        fov_y: 45f32.to_radians(),
        ..Camera::default()
    });
}

/// The flagstones: one quad per stone, each carrying one full tile of the rock
/// maps and marching its own height field.
///
/// A quad per stone rather than one big one, and the reason is the parallax. The
/// march reads a depth in metres through a Jacobian measured from the pixel's own
/// UV-versus-world footprint, so a tile stretched across a sixteen metre floor
/// keeps its three centimetres of relief and spreads them over a frequency
/// nothing can see. At two metres a joint is a joint. It also puts a real seam
/// every two metres, which is what a paved floor has.
fn paving(world: &mut World, kit: &Kit) {
    let ground = |v: f32| (v / STONE).round() as i32;
    let mut index = 0;
    for ix in ground(FLOOR[0][0])..ground(FLOOR[1][0]) {
        for iz in ground(FLOOR[0][1])..ground(FLOOR[1][1]) {
            let x = (ix as f32 + 0.5) * STONE;
            let z = (iz as f32 + 0.5) * STONE;
            let seed = (ix.wrapping_mul(73_856_093) ^ iz.wrapping_mul(19_349_663)) as u32;

            spawn_mesh(
                world,
                format!("Flagstone {index}"),
                Transform {
                    // A centimetre of settle either way. Enough that the raking
                    // sun catches the lip of a stone standing proud of its
                    // neighbour, which is most of what makes a floor read as
                    // laid rather than printed.
                    translation: Vec3::new(x, (hash01(seed) - 0.5) * 0.024, z),
                    // Quarter turns only, so the tile still meets itself at every
                    // seam while no two neighbours show the same face of the map.
                    rotation: Quat::from_rotation_y(
                        (hash01(seed ^ 0x9e37_79b9) * 4.0).floor() * FRAC_PI_2,
                    ),
                    scale: Vec3::splat(STONE),
                },
                kit.quad,
                kit.flagstone,
            );
            index += 1;
        }
    }

    // What the stones are laid on. Nothing sees it directly; it is there so the
    // settle above cannot open a slit to the skybox at a grazing angle, which at
    // this camera height is most of the floor.
    let size = [FLOOR[1][0] - FLOOR[0][0], FLOOR[1][1] - FLOOR[0][1]];
    slab(
        world,
        kit,
        "Substrate",
        Vec3::new(
            (FLOOR[0][0] + FLOOR[1][0]) * 0.5,
            -0.35,
            (FLOOR[0][1] + FLOOR[1][1]) * 0.5,
        ),
        Vec3::new(size[0], 0.6, size[1]),
    );
}

/// Three sides of a ruin, in segments, stepping down toward the open side.
///
/// The heights are the composition, and four metres is the whole argument of the
/// scene made concrete: it is what lets a 22° sun clear the wall and light the
/// floor the camera is looking across. Taller walls frame the shot better and
/// leave nothing in it lit. The segments nearer the camera fall to 2.6 and 2.0 so
/// there is a low edge with foliage on it to silhouette against the sky.
fn walls(world: &mut World, kit: &Kit) {
    // (from, to, height) along z, at x = -8.
    for (index, (from, to, height)) in [
        (FLOOR[0][1], GATE[0], 4.0f32),
        (GATE[1], 1.0, 4.0),
        (1.0, 4.0, 2.6),
        (4.0, 7.0, 2.0),
    ]
    .into_iter()
    .enumerate()
    {
        slab(
            world,
            kit,
            format!("West Wall {index}"),
            Vec3::new(FLOOR[0][0] - THICK * 0.5, height * 0.5, (from + to) * 0.5),
            Vec3::new(THICK, height, to - from),
        );
    }

    // The far wall, with a collapsed span the light and the air come through.
    for (index, (from, to, height)) in [(FLOOR[0][0], 2.5f32, 4.5f32), (4.5, FLOOR[1][0], 3.2)]
        .into_iter()
        .enumerate()
    {
        slab(
            world,
            kit,
            format!("North Wall {index}"),
            Vec3::new((from + to) * 0.5, height * 0.5, FLOOR[0][1] - THICK * 0.5),
            Vec3::new(to - from, height, THICK),
        );
    }

    // Sized in whole brick tiles rather than to taste — see `facing`, which clads
    // this wall's inner face. Four metres of length is one panel and two or four of
    // height is one or two courses of them, so the cladding covers the stone
    // exactly instead of leaving a band of bare wall above a floating rectangle.
    for (index, (from, to, height)) in [(FLOOR[0][1], -5.0f32, 4.0f32), (-5.0, -1.0, 2.0)]
        .into_iter()
        .enumerate()
    {
        slab(
            world,
            kit,
            format!("East Wall {index}"),
            Vec3::new(FLOOR[1][0] + THICK * 0.5, height * 0.5, (from + to) * 0.5),
            Vec3::new(THICK, height, to - from),
        );
    }
}

/// The opening the sun comes through, and the two things that make it read as one:
/// a lintel over it, and jambs thick enough to have a lit face and a dark one.
fn gateway(world: &mut World, kit: &Kit) {
    slab(
        world,
        kit,
        "Gate Lintel",
        Vec3::new(
            FLOOR[0][0] - THICK * 0.5,
            (GATE_HEAD + 4.0) * 0.5,
            (GATE[0] + GATE[1]) * 0.5,
        ),
        Vec3::new(THICK, 4.0 - GATE_HEAD, GATE[1] - GATE[0]),
    );
    // Two steps worn into the threshold. They also catch the beam before it
    // reaches the floor, which is what tells the eye the light is coming through
    // an opening rather than from off to the left.
    for (index, (inset, height)) in [(0.0f32, 0.24f32), (0.55, 0.12)].into_iter().enumerate() {
        slab(
            world,
            kit,
            format!("Threshold {index}"),
            Vec3::new(
                FLOOR[0][0] + inset + 0.3,
                height * 0.5,
                (GATE[0] + GATE[1]) * 0.5,
            ),
            Vec3::new(0.6, height, GATE[1] - GATE[0] + 0.6),
        );
    }
}

/// Brick facing on the east wall's inner face, which is the only wall surface the
/// hero camera sees *lit* — its normal is `-x`, and `-x` is where the sun is.
///
/// So this is the same masonry the pier carries, put where the beauty shot can
/// reach it. The pier's face is aimed into the sun as well, but it points away
/// from a camera standing at `+x`, which is where a camera looking anywhere near
/// the sun has to stand. Two brick surfaces rather than one moved, because they
/// answer different questions: this one is masonry in a photograph, and the pier
/// is the parallax march at a measured obliquity.
fn facing(world: &mut World, kit: &Kit) {
    for (index, (z, y)) in [(-7.0f32, 1.0f32), (-7.0, 3.0), (-3.0, 1.0)]
        .into_iter()
        .enumerate()
    {
        spawn_mesh(
            world,
            format!("East Facing {index}"),
            Transform {
                translation: Vec3::new(FLOOR[1][0] - 0.02, y, z),
                rotation: Quat::from_rotation_y(-FRAC_PI_2) * Quat::from_rotation_x(FRAC_PI_2),
                scale: Vec3::new(WALL_TILE[0], 1.0, WALL_TILE[1]),
            },
            kit.quad,
            kit.brick,
        );
    }
}

/// A free-standing arch across the middle of the courtyard, and it is in the scene
/// for one reason: it is what stands between the hero camera and the sun.
///
/// The alternative was a taller wall, and that is the trade the whole layout turns
/// on — a wall high enough to hide a 22° disc from a lens 1.15 m off the ground
/// fourteen metres away is seven and a half metres, and a wall that tall shadows
/// the entire courtyard. An arch hides the disc with half a metre of stone at
/// eight metres' range instead, and pays for itself twice over: its lintel throws
/// a hard-edged band across the lit paving, and the opening under it is a second
/// beam. The numbers below are not free parameters — the lintel is placed on the
/// line from the camera to the sun.
fn arcade(world: &mut World, kit: &Kit) {
    for (index, x) in [-4.2f32, 0.6].into_iter().enumerate() {
        slab(
            world,
            kit,
            format!("Arch Pier {index}"),
            Vec3::new(x, 2.1, 1.9),
            Vec3::new(0.7, 4.2, 0.7),
        );
    }
    slab(
        world,
        kit,
        "Arch Lintel",
        Vec3::new(-1.8, 4.55, 1.9),
        Vec3::new(6.2, 0.7, 0.7),
    );
}

/// A standing pier of brick-faced masonry, and the one surface in the courtyard
/// built to be *looked at* rather than looked past.
///
/// Its face points `-x`, which is the only orientation the sun reaches at this
/// hour — the walls facing the camera are all backlit. So this is where the
/// parallax march, the decal on top of it, and the `showcase-materials` capture
/// all live. It earns its place a second way: four and a half metres of occluder
/// standing in open ground is what turns the beam from the gateway into a shaft
/// with an edge.
fn pier(world: &mut World, kit: &Kit) {
    let face = 1.3;
    slab(
        world,
        kit,
        "Pier",
        Vec3::new(face + 0.3, 2.25, -3.0),
        Vec3::new(0.6, 4.5, 4.0),
    );

    // Two panels stacked up the face, each the four-by-two metres the brick maps
    // were generated for — so a block comes out the half metre it was authored
    // as rather than whatever the pier's aspect would make of it.
    for (index, y) in [1.0f32, 3.0].into_iter().enumerate() {
        spawn_mesh(
            world,
            format!("Pier Facing {index}"),
            Transform {
                // Two centimetres proud of the stone behind it. Far enough that
                // nothing z-fights at this depth range, close enough that the
                // ambient occlusion reads the gap as the recess of a facing
                // rather than as a floating panel.
                translation: Vec3::new(face - 0.02, y, -3.0),
                // Stood on edge, then turned so its normal is `-x`: that lands
                // `u` along `+z` and `v` up, which is the way round the courses
                // were drawn.
                rotation: Quat::from_rotation_y(-FRAC_PI_2) * Quat::from_rotation_x(FRAC_PI_2),
                scale: Vec3::new(WALL_TILE[0], 1.0, WALL_TILE[1]),
            },
            kit.quad,
            kit.brick,
        );
    }
}

/// What the courtyard is for: four bodies of the same stone, each authored to a
/// different transport model, standing between the camera and the sun.
///
/// That placement is the point. A marble sphere lit from the front is a grey
/// sphere; lit from behind, the light that crossed a few millimetres of it comes
/// out on the side facing the camera, which is the only view the transmitted
/// lobe exists for. The two transmissive ones need the same thing for a different
/// reason — what they refract is the frame behind them, and behind them is a
/// sunlit gateway.
fn artefacts(world: &mut World, kit: &Kit) {
    // (name, material, plinth footprint and height, sphere diameter)
    //
    // Every position here is inside the beam, and that is a computed thing rather
    // than an eyeballed one, and it is the *arch's* beam rather than the gateway's:
    // the gateway lights the far corner, twelve metres out, where a sphere is forty
    // pixels of silhouette. The arch throws its light toward the camera instead.
    //
    // Its lit wedge marches with depth — a metre further from the arch is 1.7 m
    // further along `+x` — so the spacing rule is the one that falls out of a low
    // sun: two bodies at the same `z` never shadow each other, and one downwind of
    // another almost always does. These are therefore spread mostly across the
    // beam and only a little along it. The first arrangement had them in the far
    // corner, and the one before that had all four a metre outside the light.
    for (name, material, base, height, diameter) in [
        (
            "Marble",
            kit.marble,
            Vec3::new(1.0, 0.0, 1.2),
            1.7f32,
            0.9f32,
        ),
        ("Crystal", kit.crystal, Vec3::new(3.0, 0.0, 1.2), 0.5, 1.05),
        ("Wax", kit.wax, Vec3::new(5.0, 0.0, 1.4), 1.1, 0.95),
    ] {
        slab(
            world,
            kit,
            format!("{name} Plinth"),
            base + Vec3::new(0.0, height * 0.5, 0.0),
            Vec3::new(diameter + 0.15, height, diameter + 0.15),
        );
        spawn_mesh(
            world,
            name,
            Transform {
                translation: base + Vec3::new(0.0, height + diameter * 0.5, 0.0),
                scale: Vec3::splat(diameter),
                ..Default::default()
            },
            kit.sphere,
            material,
        );
    }

    // Toppled, resting on the flagstones at the near end of the beam. Its
    // background is the paving rather than the sky, which is what makes a rough
    // transmissive surface legible at all: a frosted sphere reads from the *blur*
    // of what is behind it, and sky is uniform whether it was blurred or not.
    spawn_mesh(
        world,
        "Frosted",
        Transform {
            translation: Vec3::new(1.8, 0.6, 3.4),
            scale: Vec3::splat(1.2),
            ..Default::default()
        },
        kit.sphere,
        kit.frosted,
    );

    // Two panes of glazing leaning into each other in the sunbeam, and the
    // readout the rig's three panes make: weighted-blended transparency
    // commutes, so where they cross there must be no seam and no flicker as the
    // camera moves.
    //
    // Standing them in the light rather than against the far wall, which is where
    // they were first put and where a pane of glass is a dark rectangle — the
    // wall behind it is in shadow, so there is nothing for it to transmit or
    // reflect. There is a second thing to notice here that only sunlight shows:
    // the panes cast no shadow at all, because blended geometry never reaches a
    // caster list.
    for (index, (x, z, yaw, lean)) in [(1.6f32, 4.4f32, 14.0f32, 13.0f32), (2.3, 5.0, -22.0, -11.0)]
        .into_iter()
        .enumerate()
    {
        spawn_mesh(
            world,
            format!("Glazing {index}"),
            Transform {
                translation: Vec3::new(x, 1.15, z),
                rotation: Quat::from_rotation_y(yaw.to_radians())
                    * Quat::from_rotation_x(lean.to_radians()),
                scale: Vec3::new(1.4, 1.9, 0.07),
            },
            kit.cube,
            kit.glass,
        );
    }
}

/// Cutout foliage, and the two jobs it does here.
///
/// Along the wall bases it is what a ruin has instead of a clean skirting. On the
/// wall *tops* it is the thing worth building the low segments for: a leaf card
/// against a bright sky, backlit, resolving its edge across the four MSAA samples
/// the frame is already paying for — and lit through, since the leaf material
/// carries a transmitted lobe as well as a cutout.
///
/// Crossed pairs rather than single quads, exactly as the rig spawns them: two
/// cutouts intersecting at ninety degrees write depth against each other, so an
/// ordering mistake shows immediately as one card erasing the other. This is the
/// opaque queue, and nothing in it is sorted.
fn overgrowth(world: &mut World, kit: &Kit) {
    // (x, y, z, scale, yaw)
    //
    // A card is 1.2–1.7 m across, which is a sprig of something growing out of a
    // wall base. The first pass had them at 2.2–3.0 to match the rig's, where a
    // card stands alone on open ground and is a *specimen*; at that size in a
    // courtyard they read as kelp, and two of them were close enough to the hero
    // camera to be the frame.
    for (index, (x, y, z, scale, yaw)) in [
        // Against the far wall, in its own shadow.
        (-7.0f32, 0.0f32, -8.1f32, 1.5f32, 15.0f32),
        (-4.4, 0.0, -8.3, 1.3, -40.0),
        (0.4, 0.0, -8.2, 1.7, 25.0),
        (2.0, 0.0, -8.1, 1.4, -15.0),
        (5.6, 0.0, -8.2, 1.5, 60.0),
        // Either side of the pier, where the shafts pass.
        (2.6, 0.0, -1.0, 1.6, 35.0),
        (2.8, 0.0, -6.5, 1.3, -25.0),
        // The threshold of the gateway, backlit by the beam itself — the one place
        // a leaf is lit from behind and the transmitted lobe is the whole of what
        // reaches the camera.
        (-7.1, 0.0, -3.0, 1.2, 10.0),
        (-7.2, 0.0, -6.5, 1.4, -30.0),
        (-6.6, 0.0, -4.9, 1.3, 45.0),
        // On the low wall tops, against the sky, which is where a cutout's edge is
        // resolved against the greatest contrast the frame has.
        (-8.25, 2.0, 5.2, 1.4, 20.0),
        (-8.25, 2.6, 2.6, 1.6, -20.0),
        (8.25, 2.6, -1.4, 1.5, 40.0),
        // Out in the paving, where a joint has opened — one of them, small and off
        // to the side. Three of these stood in the open floor at first and they
        // were the frame's clutter: a card five metres from the lens with a 1.3 m
        // span is two hundred pixels of bright green across the middle of the shot,
        // and there is nothing behind it worth that.
        (-2.6, 0.0, 5.4, 1.0, -10.0),
        (-6.0, 0.0, 6.2, 1.4, 15.0),
    ]
    .into_iter()
    .enumerate()
    {
        for (half, turn) in [(0usize, 0.0f32), (1, 90.0)] {
            spawn_mesh(
                world,
                format!("Overgrowth {index}.{half}"),
                Transform {
                    // The card's own half height above where it stands, so it
                    // sits on the surface rather than through it.
                    translation: Vec3::new(x, y + scale * 0.5, z),
                    rotation: Quat::from_rotation_y((yaw + turn).to_radians())
                        * Quat::from_rotation_x(FRAC_PI_2),
                    scale: Vec3::splat(scale),
                },
                kit.quad,
                kit.foliage,
            );
        }
    }
}

/// The sun, the sky it belongs to, and the two fixtures that survive being beside
/// it.
///
/// The photometry is the whole difference between this scene's late afternoon and
/// the rig's twilight, and it is a real trade rather than a preference. Under a
/// 100 000 lux noon sun a fire is a rounding error; the rig answers by dropping
/// the sun to 250 lux so a lamp, a bulb and a neon tube are all within a few
/// stops of it, at the cost of a picture nobody would photograph.
///
/// Here the sun is 22 000 lux against a 70 cd/m² sky, and the *ratio* is the
/// number that was tuned rather than either level. At 6 000 against 150 the
/// courtyard came out flat: an 80 lux hemisphere against a 6 000 lux sun is
/// twelve to one, which is an overcast day, and in it every wall the camera faces
/// — all of them, at this sun angle — was lit by a uniform sky, where a normal
/// map has no direction to shade against and rock reads as grey card. Eighty to
/// one is what makes the same maps show relief.
///
/// The fixtures are then scaled to what stands up in the sun's *shadows* rather
/// than in its light, which is the only place a fixture can compete outdoors:
/// sky-lit stone in shadow sits near 30 cd/m², and 20 000 lumens of fire two
/// metres away puts about 50 on the flagstones. Comparable, so the fire reads —
/// and nothing had to lie about its units to get there.
fn lights(world: &mut World, kit: &Kit, backend: &mut impl RenderBackend) {
    let sun = SUN.normalize();
    spawn_directional_light(world, "Sun", sun, Vec3::new(1.0, 0.72, 0.45), 22_000.0);

    // Fed the direction *toward* the sun, so the disc in the sky and the light
    // casting the shadows agree about where it is.
    const SKY: [u32; 2] = [1024, 512];
    backend.load_environment(&sky_equirect(SKY[0], SKY[1], -sun), SKY[0], SKY[1]);
    world.insert_resource(EnvironmentSettings {
        // What this sky *is*, in cd/m², rather than a multiplier on whatever the
        // generator happened to emit: a sky an hour before sunset, dim enough
        // that its shadows have contrast in them and the fire below is not
        // competing with the ambient.
        sky_luminance: 70.0,
        ..EnvironmentSettings::default()
    });

    // The fire: an emissive body and a fixture at the same place, deliberately as
    // two entities. An emissive surface is a surface that is bright — it lights
    // nothing — and a light is not geometry, so it cannot appear in a reflection
    // or be occluded. Wanting both is wanting both objects.
    let fire = Vec3::new(-5.0, 0.0, 0.6);
    slab(
        world,
        kit,
        "Brazier",
        fire + Vec3::new(0.0, 0.18, 0.0),
        Vec3::new(1.5, 0.36, 1.5),
    );
    spawn_mesh(
        world,
        "Fire",
        Transform {
            translation: fire + Vec3::new(0.0, 0.52, 0.0),
            scale: Vec3::splat(0.85),
            ..Default::default()
        },
        kit.sphere,
        kit.ember,
    );
    spawn_point_light(
        world,
        "Fire Light",
        fire + Vec3::new(0.0, 0.75, 0.0),
        Vec3::new(1.0, 0.55, 0.25),
        20_000.0,
        14.0,
    );

    // A sconce high on the east wall, aimed down across the paving. It is the one
    // light in the scene whose shadow has a hard edge from a single small source,
    // which is what the punctual atlas is for — and it lands on the half of the
    // courtyard the sun has already left.
    spawn_spot_light(
        world,
        "Sconce",
        Vec3::new(7.4, 4.2, -2.6),
        Vec3::new(-0.55, -1.0, 0.3).normalize(),
        Vec3::new(1.0, 0.84, 0.62),
        9_000.0,
        18.0,
        20.0,
        30.0,
    );
}

/// A block of the courtyard's structural stone: walls, lintels, plinths, kerbs.
///
/// `size` is metres on each axis and `center` is the middle of the block, so a
/// wall standing on the paving is `height * 0.5` up. The material carries the
/// rock maps without the height map — see the library — because a slab's narrow
/// faces would march a field stretched across their thickness, and a cut face of
/// masonry has no relief to march.
fn slab(world: &mut World, kit: &Kit, name: impl Into<String>, center: Vec3, size: Vec3) {
    spawn_mesh(
        world,
        name,
        Transform {
            translation: center,
            scale: size,
            ..Default::default()
        },
        kit.cube,
        kit.stone,
    );
}
