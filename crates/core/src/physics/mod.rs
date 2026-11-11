//! Rigid-body dynamics, on Jolt.
//!
//! Every entity with a [`Collider`] becomes a Jolt body, of one of three kinds:
//!
//! - **Dynamic**: a [`RigidBody`] that is not kinematic, on a transform root.
//!   Jolt owns its pose and this module writes it back into the
//!   [`LocalTransform`].
//! - **Kinematic**: a kinematic rigid body, or any rigid body with a
//!   transformed parent. Driven to wherever its transform says with
//!   `move_kinematic` rather than a teleport, so a lift carries what stands on
//!   it instead of tunnelling up through it. A parented body is kinematic for
//!   the reason it cannot be dynamic: Jolt's pose is world-space, the
//!   transform is parent-relative, and a parent/child link is a rigid
//!   attachment.
//! - **Static**: a collider with no rigid body. Jolt keeps statics in a
//!   broadphase tree of their own and never steps them, which is what lets a
//!   level have thousands of walls. Moving one — a script, `Spin`, the gizmo
//!   — makes it kinematic for the rest of the session, so a door nobody
//!   declared still pushes what it swings into. Nothing pushes a body without a
//!   rigid body, and two that nothing has moved never report each other:
//!   Unity's rules, both.
//!
//! **The step is fixed and the transform is interpolated.** Jolt is tuned for a
//! fixed step, and the frame here runs at whatever the display allows, so the
//! write-back blends the last two steps by how far the frame is into the next
//! one. A dynamic body's transform is compared against exactly what was last
//! written to it — bit for bit, which is why it is read from `LocalTransform`
//! rather than decomposed out of `WorldTransform` — and anything else is an edit
//! (a script, the gizmo) and teleports the body, keeping its velocity.
//!
//! **Events** land in [`CollisionState::events`], canonically ordered, normal
//! from `a` to `b`; `contacts` says how Jolt's callbacks become them. A step
//! finds contacts before it moves anything, so an overlap a kinematic body is
//! moved into is reported the step after the one that moved it.
//!
//! **Game code can run before each step, through a callback.** [`run_with`]
//! calls its caller back ahead of every step the frame takes, with the step's
//! length: the seam a script host's `OnFixedUpdate` goes through, so this
//! module never learns what is on the other side of it. The callback is
//! handed the world as a step would see it. The simulation is back in
//! [`PhysicsState`], so a push or a velocity read reaches the body. Every
//! dynamic body's transform — local, and world, since a dynamic body is a root
//! — is the last step's pose rather than the frame's interpolated one; its
//! children are not, and wait for the frame's propagation. A
//! [`ForceMode::Force`] or `Acceleration` asked for there is held for that one
//! step rather than for the frame (see `control`). A dynamic body's position or
//! rotation written there teleports it before the step. Anything else that
//! rebuilds what Jolt holds — a body added, removed or reshaped, a kinematic or
//! static collider moved — waits for the next frame's `sync`, as it would from
//! anywhere: that is a sweep of every collider, so it is once a frame. With no
//! callback none of this runs; with one it is two passes over the dynamic
//! bodies a step.
//!
//! **The scene's [`PhysicsSettings`]** are read every frame. Gravity and the step
//! rate apply at once; the layer matrix is baked into Jolt's broadphase filter,
//! so changing it rebuilds the world, and every pair touching at that moment
//! exits rather than vanishing. Layers map to two Jolt object layers each, one
//! per broadphase tree, because statics never test each other and a static
//! tree is what lets Jolt skip asking.
//!
//! **One thread.** The job system is Jolt's single-threaded one: its thread
//! pool would be a second pool beside [`threads`](crate::threads), which exists
//! to be the only one. Parallel steps want a job system over that pool instead,
//! which joltc does not offer yet.
//!
//! The simulation lives for one play session. Stop throws the fork away and
//! [`reset`] drops the Jolt world with it, so the next Play starts at rest from
//! the scene rather than from last session's velocities.

use std::sync::{Arc, Mutex};

use glam::{Quat, Vec3};
use jolt::{
    Activation, AllowedDofs, BodyCreationSettings, BodyId, BroadPhaseLayer, JobSystem, LayerConfig,
    MotionQuality, MotionType, ObjectLayer, OverrideMassProperties, PhysicsSystem,
    PhysicsSystemSettings, TempAllocator,
};
use orrin_ecs::{Entity, FxBuildHasher, FxHashMap, World};

mod contacts;
mod control;
mod events;
pub mod outline;
mod shapes;

pub use control::{ForceMode, RaycastHit, add_force, add_torque, raycast, set_velocity, velocity};
pub use events::{CollisionEvent, CollisionEventKind, CollisionState, Contact, pair_key};

use crate::scene::{
    Collider, LocalTransform, LogBuffer, LogLevel, MeshBounds, MeshHandle, PHYSICS_LAYERS,
    PhysicsSettings, RigidBody, Time, WorldTransform, is_transform_root,
};

/// The step at [`PhysicsSettings`]' default 60 Hz.
pub const STEP: f32 = 1.0 / 60.0;

/// The smallest dimension a shape is built with. Jolt rejects a degenerate
/// shape, and a zero scale is something the inspector can produce.
const MIN_EXTENT: f32 = 1e-3;

/// What a body is to Jolt right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    Static,
    Kinematic,
    Dynamic,
}

type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

#[derive(Default)]
pub struct PhysicsState {
    simulation: Option<Simulation>,
    /// Jolt refused to build a world. Said once, then physics stays off rather
    /// than retrying every frame.
    failed: bool,
    /// What has been said about which entity, so a condition that holds every
    /// frame is reported once rather than buried under itself.
    told: FxHashSet<(Entity, Notice)>,
    /// Pushes on bodies the next run will create; see `control`.
    pending: Vec<(Entity, control::Push)>,
    /// How long a Force or Acceleration asked for now is held: the last
    /// frame's delta, or one step while [`run_with`]'s callback runs.
    held_for: f32,
}

/// Something about an entity's body worth one line in the log.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Notice {
    ParentedBody,
    MeshOnRigidBody,
    NoShape(shapes::Problem),
}

impl Notice {
    fn message(&self, entity: Entity) -> String {
        match self {
            Self::ParentedBody => format!(
                "{entity:?}: a rigid body with a transformed parent follows the parent \
                 instead of being simulated — a parent/child link is a rigid attachment; \
                 unparent it to let it fall"
            ),
            Self::MeshOnRigidBody => format!(
                "{entity:?}: a rigid body cannot collide as a triangle mesh, which has no \
                 inside to be pushed out of; it collides as the mesh's convex hull — set \
                 its collider to Convex Hull to say so"
            ),
            Self::NoShape(shapes::Problem::NoMesh) => format!(
                "{entity:?}: its collider is read from its mesh, and it has none with \
                 triangles registered; it is left out of physics until it does"
            ),
            Self::NoShape(shapes::Problem::Jolt(error)) => {
                format!(
                    "{entity:?}: Jolt rejected its collider ({error}); it is left out of physics"
                )
            }
        }
    }
}

struct Simulation {
    system: PhysicsSystem,
    temp: TempAllocator,
    jobs: JobSystem,
    /// The matrix the broadphase filter was built from.
    layers: Vec<u32>,
    gravity: Vec3,
    bodies: FxHashMap<Entity, Tracked>,
    entities: FxHashMap<BodyId, Entity>,
    /// Filled by the contact listener during `update`, drained after it.
    changes: Arc<Mutex<Vec<contacts::Change>>>,
    pairs: contacts::Pairs,
    shapes: shapes::Cache,
    accumulator: f32,
}

/// What an entity asks to be. A static that has been moved is still
/// `Static` here; [`Tracked::moved`] is what changed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Dynamic,
    Kinematic,
    Static,
}

/// What a body was built from. Any difference rebuilds it.
#[derive(Clone, Copy, Debug)]
struct Spec {
    kind: Kind,
    collider: Collider,
    rigid_body: Option<RigidBody>,
    scale: Vec3,
    /// Which copy of the mesh's triangles, by address: a reimport replaces it.
    geometry: Option<usize>,
}

impl Spec {
    /// Scale is compared with a tolerance because a kinematic or static body's
    /// comes out of a matrix decomposition, whose low bits move as the rotation
    /// does.
    fn matches(&self, other: &Spec) -> bool {
        self.kind == other.kind
            && self.collider == other.collider
            && self.rigid_body == other.rigid_body
            && self.geometry == other.geometry
            && self.scale.abs_diff_eq(other.scale, 1e-4)
    }
}

type Pose = (Vec3, Quat);

struct Tracked {
    id: BodyId,
    spec: Spec,
    /// A static something has moved, and is kinematic from then on.
    moved: bool,
    /// Dynamic: the pose at the last two steps, blended for the write-back.
    previous: Pose,
    current: Pose,
    /// Dynamic: what was last written into the transform. Otherwise: where the
    /// transform says the body should be.
    pose: Pose,
    seen: bool,
}

impl Tracked {
    /// An edit to a dynamic body's transform: the body jumps there and keeps
    /// its velocity.
    fn teleport(&mut self, bodies: &jolt::BodyInterface<'_>, pose: Pose) {
        bodies.set_position_and_rotation(
            self.id,
            to_jolt(pose.0),
            to_jolt_quat(pose.1),
            Activation::Activate,
        );
        self.previous = pose;
        self.current = pose;
        self.pose = pose;
    }

    fn motion(&self) -> Motion {
        match self.spec.kind {
            Kind::Dynamic => Motion::Dynamic,
            Kind::Kinematic => Motion::Kinematic,
            Kind::Static if self.moved => Motion::Kinematic,
            Kind::Static => Motion::Static,
        }
    }
}

/// The Jolt object layer for collision layer `layer`, in the static or the
/// moving broadphase tree.
fn object_layer(layer: u32, moving: bool) -> ObjectLayer {
    let layer = layer.min(PHYSICS_LAYERS - 1);
    ObjectLayer((layer * 2 + moving as u32) as u16)
}

impl Simulation {
    fn new(settings: &PhysicsSettings) -> jolt::Result<Self> {
        let mut layers = LayerConfig::new();
        for index in 0..PHYSICS_LAYERS * 2 {
            let added = layers.add_object_layer(BroadPhaseLayer((index % 2) as u8));
            debug_assert_eq!(added.0 as u32, index);
        }
        for a in 0..PHYSICS_LAYERS * 2 {
            for b in a..PHYSICS_LAYERS * 2 {
                let both_static = a % 2 == 0 && b % 2 == 0;
                if !both_static && settings.collides(a / 2, b / 2) {
                    layers.enable_collision(ObjectLayer(a as u16), ObjectLayer(b as u16));
                }
            }
        }
        let mut system = PhysicsSystem::new(&PhysicsSystemSettings::default(), &layers)?;
        let changes = Arc::new(Mutex::new(Vec::new()));
        system.set_contact_listener(contacts::Recorder(changes.clone()));
        Ok(Self {
            system,
            temp: TempAllocator::default(),
            jobs: JobSystem::single_threaded(),
            layers: settings.layer_collisions.clone(),
            gravity: Vec3::NAN,
            bodies: FxHashMap::default(),
            entities: FxHashMap::default(),
            changes,
            pairs: contacts::Pairs::default(),
            shapes: shapes::Cache::default(),
            accumulator: 0.0,
        })
    }

    fn create(
        &mut self,
        spec: Spec,
        pose: Pose,
        geometry: Option<&Arc<crate::scene::MeshGeometry>>,
    ) -> Result<BodyId, shapes::Problem> {
        let shape = shapes::build(
            spec.collider.shape,
            spec.scale,
            geometry,
            spec.kind == Kind::Dynamic,
            &mut self.shapes,
        )?;
        let motion = match spec.kind {
            Kind::Dynamic => MotionType::Dynamic,
            Kind::Kinematic => MotionType::Kinematic,
            Kind::Static => MotionType::Static,
        };
        let layer = object_layer(spec.collider.layer, spec.kind != Kind::Static);
        let mut settings =
            BodyCreationSettings::new(shape, to_jolt(pose.0), to_jolt_quat(pose.1), motion, layer);
        settings.is_sensor = spec.collider.is_trigger;
        settings.friction = spec.collider.friction;
        settings.restitution = spec.collider.restitution;
        // A kinematic body is how a moved collider reports what it moves into,
        // and Jolt only has kinematics test statics and each other when asked.
        settings.collide_kinematic_vs_non_dynamic = true;
        match spec.kind {
            Kind::Dynamic => {
                let body = spec.rigid_body.unwrap_or_default();
                settings.gravity_factor = body.gravity_scale;
                settings.linear_damping = body.linear_damping.max(0.0);
                settings.angular_damping = body.angular_damping.max(0.0);
                settings.override_mass_properties = OverrideMassProperties::CalculateInertia {
                    mass: body.mass.max(MIN_EXTENT),
                };
                if body.continuous {
                    settings.motion_quality = MotionQuality::LinearCast;
                }
                if body.lock_rotation {
                    settings.allowed_dofs = AllowedDofs::TRANSLATION_X
                        | AllowedDofs::TRANSLATION_Y
                        | AllowedDofs::TRANSLATION_Z;
                }
            }
            // A body standing still still has to notice what it is moved into
            // next, and a sleeping kinematic body tests nothing.
            Kind::Kinematic => settings.allow_sleeping = false,
            // Created able to become kinematic, which Jolt otherwise refuses.
            Kind::Static => settings.allow_dynamic_or_kinematic = true,
        }
        self.system
            .body_interface()
            .create_and_add_body(&settings, Activation::Activate)
            .map_err(|error| shapes::Problem::Jolt(format!("{error:?}")))
    }

    fn destroy(&mut self, entity: Entity, id: BodyId, events: &mut Vec<CollisionEvent>) {
        self.system.destroy_body(id);
        self.entities.remove(&id);
        self.pairs.leave(entity, events);
    }

    /// A static something moved: into the moving tree, and kinematic.
    fn promote(&mut self, tracked: &mut Tracked) {
        let bodies = self.system.body_interface();
        let layer = object_layer(tracked.spec.collider.layer, true);
        if bodies.set_object_layer(tracked.id, layer).is_ok()
            && bodies
                .set_motion_type(tracked.id, MotionType::Kinematic, Activation::Activate)
                .is_ok()
        {
            tracked.moved = true;
        }
    }
}

/// What [`run_with`] calls before each step: the world, and the step's length.
pub type BeforeStep<'a> = dyn FnMut(&mut World, f32) + 'a;

/// Step the simulation by a frame's `delta` and write the dynamic bodies'
/// poses back. Reads world transforms, so it runs after propagation.
pub fn run(world: &mut World, delta: f32) {
    run_with(world, delta, None);
}

/// [`run`], calling `before_step` with the step's length ahead of every step
/// the frame takes — none, or several. What the world looks like to it is in
/// the module docs.
pub fn run_with(world: &mut World, delta: f32, before_step: Option<&mut BeforeStep<'_>>) {
    let settings = scene_settings(world);
    let (simulation, mut pending) = {
        let mut state = world.resource_mut::<PhysicsState>();
        if state.failed {
            return;
        }
        state.held_for = delta;
        (state.simulation.take(), std::mem::take(&mut state.pending))
    };
    let (mut events, report_stays) = {
        let mut collisions = world.resource_mut::<CollisionState>();
        (
            std::mem::take(&mut collisions.events),
            collisions.report_stays,
        )
    };
    events.clear();

    let simulation = match simulation {
        Some(simulation) if simulation.layers == settings.layer_collisions => Ok(simulation),
        Some(mut stale) => {
            stale.pairs.leave_all(&mut events);
            Simulation::new(&settings)
        }
        None => Simulation::new(&settings),
    };
    let mut simulation = match simulation {
        Ok(simulation) => simulation,
        Err(error) => {
            tracing::error!("Jolt could not build a physics world ({error:?}); physics is off");
            world.resource_mut::<PhysicsState>().failed = true;
            return;
        }
    };
    if simulation.gravity != settings.gravity {
        simulation.gravity = settings.gravity;
        simulation.system.set_gravity(to_jolt(settings.gravity));
        // A body asleep under the old gravity would otherwise stay put under
        // the new one until something touched it.
        let bodies = simulation.system.body_interface();
        for tracked in simulation.bodies.values() {
            if tracked.spec.kind == Kind::Dynamic {
                bodies.activate_body(tracked.id);
            }
        }
    }

    let mut notices = Vec::new();
    sync(world, &mut simulation, &mut events, &mut notices);
    control::apply_pending(&simulation, &mut pending);
    // Handed back before the steps, so a push a callback asks of a body that
    // does not exist yet is kept for the next run.
    world.resource_mut::<PhysicsState>().pending = pending;
    let stepped = step(
        world,
        simulation,
        &settings,
        delta,
        &mut events,
        report_stays,
        before_step,
    );
    let Some(mut simulation) = stepped else {
        world.resource_mut::<CollisionState>().events = events;
        return;
    };
    write_back(world, &mut simulation, settings.step());

    world.resource_mut::<CollisionState>().events = events;
    let mut state = world.resource_mut::<PhysicsState>();
    state.simulation = Some(simulation);
    notices.retain(|told| state.told.insert(told.clone()));
    drop(state);
    if !notices.is_empty() {
        tell(world, notices);
    }
}

/// Drop the simulation, so the next [`run`] rebuilds it from the world as it
/// then is.
pub fn reset(world: &World) {
    let mut state = world.resource_mut::<PhysicsState>();
    state.simulation = None;
    state.pending.clear();
}

/// What `entity`'s body is to Jolt, or `None` while it has none — before the
/// first run, outside play, or when its collider could not be built.
pub fn motion(world: &World, entity: Entity) -> Option<Motion> {
    let state = world.get_resource::<PhysicsState>()?;
    let tracked = state.simulation.as_ref()?.bodies.get(&entity)?;
    Some(tracked.motion())
}

/// The first entity's [`PhysicsSettings`], or the defaults.
fn scene_settings(world: &World) -> PhysicsSettings {
    let mut found = None;
    world.query::<&PhysicsSettings>().find(|_, settings| {
        found = Some(settings.clone());
        true
    });
    found.unwrap_or_default()
}

/// Bring the bodies in line with the world: create, rebuild and destroy them,
/// pick up edits to dynamic transforms, and aim the rest at theirs.
// Out of line on purpose: inlined into `run_with` alongside the step callback's
// machinery, the frame of 1 000 static colliders in the physics bench measured
// 7% slower.
#[inline(never)]
fn sync(
    world: &World,
    simulation: &mut Simulation,
    events: &mut Vec<CollisionEvent>,
    notices: &mut Vec<(Entity, Notice)>,
) {
    for tracked in simulation.bodies.values_mut() {
        tracked.seen = false;
    }

    let mut wanted = Vec::new();
    world.query::<(&WorldTransform, &Collider)>().for_each(
        |entity, (world_transform, collider)| {
            wanted.push((entity, world_transform.0, *collider));
        },
    );
    let meshes = world.get_resource::<MeshBounds>();

    let mut created = 0;
    for (entity, matrix, collider) in wanted {
        let rigid_body = world.get::<RigidBody>(entity).map(|body| *body);
        let root = is_transform_root(world, entity);
        let kind = match rigid_body {
            Some(body) if !body.kinematic && root => Kind::Dynamic,
            Some(body) => {
                if !body.kinematic {
                    notices.push((entity, Notice::ParentedBody));
                }
                Kind::Kinematic
            }
            None => Kind::Static,
        };
        let geometry = if collider.shape.uses_mesh() {
            let handle = world.get::<MeshHandle>(entity).map(|handle| *handle);
            handle.and_then(|handle| meshes.as_ref()?.geometry(handle).cloned())
        } else {
            None
        };
        if kind == Kind::Dynamic && collider.shape == crate::scene::ColliderShape::Mesh {
            notices.push((entity, Notice::MeshOnRigidBody));
        }

        let (scale, pose) = if kind == Kind::Dynamic {
            let local = world
                .get::<LocalTransform>(entity)
                .map_or_else(Default::default, |l| l.0);
            (local.scale, (local.translation, local.rotation))
        } else {
            let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
            (scale, (translation, rotation))
        };
        let spec = Spec {
            kind,
            collider,
            rigid_body,
            scale,
            geometry: geometry.as_ref().map(|g| Arc::as_ptr(g) as usize),
        };

        let rebuild = match simulation.bodies.get(&entity) {
            Some(tracked) if tracked.spec.matches(&spec) => None,
            Some(tracked) => Some(Some(tracked.id)),
            None => Some(None),
        };
        if let Some(stale) = rebuild {
            // A dynamic body rebuilt as a dynamic body is the same body moving:
            // it keeps its velocity, and — unless the transform was edited —
            // the pose the simulation had, not the interpolated one behind it.
            let mut carried = None;
            let mut pose = pose;
            if let Some(id) = stale {
                let old = simulation
                    .bodies
                    .remove(&entity)
                    .expect("stale names a body");
                if old.spec.kind == Kind::Dynamic && spec.kind == Kind::Dynamic {
                    let bodies = simulation.system.body_interface();
                    carried = Some((bodies.linear_velocity(id), bodies.angular_velocity(id)));
                    if old.pose == pose {
                        pose = old.current;
                    }
                }
                simulation.destroy(entity, id, events);
            }
            let id = match simulation.create(spec, pose, geometry.as_ref()) {
                Ok(id) => id,
                Err(problem) => {
                    notices.push((entity, Notice::NoShape(problem)));
                    continue;
                }
            };
            if let Some((linear, angular)) = carried {
                let bodies = simulation.system.body_interface();
                bodies.set_linear_and_angular_velocity(id, linear, angular);
            }
            simulation.entities.insert(id, entity);
            created += 1;
            simulation.bodies.insert(
                entity,
                Tracked {
                    id,
                    spec,
                    moved: false,
                    previous: pose,
                    current: pose,
                    pose,
                    seen: true,
                },
            );
            continue;
        }

        let mut tracked = simulation.bodies.remove(&entity).unwrap();
        tracked.seen = true;
        if tracked.pose != pose {
            match tracked.spec.kind {
                Kind::Dynamic => tracked.teleport(&simulation.system.body_interface(), pose),
                Kind::Static if !tracked.moved => simulation.promote(&mut tracked),
                _ => {}
            }
        }
        tracked.pose = pose;
        simulation.bodies.insert(entity, tracked);
    }

    let gone: Vec<(Entity, BodyId)> = simulation
        .bodies
        .iter()
        .filter(|(_, tracked)| !tracked.seen)
        .map(|(&entity, tracked)| (entity, tracked.id))
        .collect();
    for (entity, id) in gone {
        simulation.bodies.remove(&entity);
        simulation.destroy(entity, id, events);
    }
    // Jolt inserts new bodies into its broadphase tree cheaply and unbalanced,
    // and asks to be told after a large batch. Without this a freshly built
    // pile of 5 000 took 110 ms a step rather than 4.4; a handful dropped in by
    // a script is what the cheap insert is for.
    if created >= 64 {
        simulation.system.optimize_broad_phase();
    }
}

/// Take the frame's steps. `None` when a callback reset the simulation under
/// it, which ends the run there.
fn step(
    world: &mut World,
    mut simulation: Simulation,
    settings: &PhysicsSettings,
    delta: f32,
    events: &mut Vec<CollisionEvent>,
    report_stays: bool,
    mut before_step: Option<&mut BeforeStep<'_>>,
) -> Option<Simulation> {
    let step = settings.step();
    simulation.accumulator += delta;
    let mut steps = 0;
    while simulation.accumulator >= step {
        if steps == settings.max_steps_per_frame.max(1) {
            simulation.accumulator %= step;
            break;
        }
        if let Some(before_step) = before_step.as_deref_mut() {
            simulation = call_before_step(world, simulation, step, delta, before_step)?;
        }
        {
            let bodies = simulation.system.body_interface();
            for tracked in simulation.bodies.values_mut() {
                match tracked.motion() {
                    // Asked every step, not once a frame: after the first step of
                    // a frame the body is already there, so the rest ask for zero
                    // velocity instead of carrying it past its target.
                    Motion::Kinematic => bodies.move_kinematic(
                        tracked.id,
                        to_jolt(tracked.pose.0),
                        to_jolt_quat(tracked.pose.1),
                        step,
                    ),
                    Motion::Dynamic => tracked.previous = tracked.current,
                    Motion::Static => {}
                }
            }
        }

        let Simulation {
            system, temp, jobs, ..
        } = &mut simulation;
        if let Err(error) = system.update(step, 1, temp, jobs) {
            tracing::warn!("physics step overflowed Jolt's limits: {error}");
        }

        let bodies = simulation.system.body_interface();
        for tracked in simulation.bodies.values_mut() {
            if tracked.spec.kind == Kind::Dynamic {
                let (position, rotation) = bodies.position_and_rotation(tracked.id);
                tracked.current = (from_jolt(position), from_jolt_quat(rotation));
            }
        }
        let mut changes = simulation.changes.lock().unwrap();
        simulation
            .pairs
            .apply(&mut changes, &simulation.entities, &bodies, events);
        drop(changes);
        simulation.pairs.settle(&bodies, events, report_stays);

        simulation.accumulator -= step;
        steps += 1;
    }
    Some(simulation)
}

/// Hand the world to `before_step` as it stands between two steps — see the
/// module docs — and take back whatever it moved.
fn call_before_step(
    world: &mut World,
    mut simulation: Simulation,
    step: f32,
    delta: f32,
    before_step: &mut BeforeStep<'_>,
) -> Option<Simulation> {
    for (&entity, tracked) in simulation.bodies.iter_mut() {
        if tracked.spec.kind != Kind::Dynamic || tracked.pose == tracked.current {
            continue;
        }
        let (translation, rotation) = tracked.current;
        if let Some(mut local) = world.get_mut::<LocalTransform>(entity) {
            local.translation = translation;
            local.rotation = rotation;
        }
        if let Some(mut placed) = world.get_mut::<WorldTransform>(entity) {
            placed.0 = glam::Mat4::from_scale_rotation_translation(
                tracked.spec.scale,
                rotation,
                translation,
            );
        }
        tracked.pose = tracked.current;
    }

    {
        let mut state = world.resource_mut::<PhysicsState>();
        state.simulation = Some(simulation);
        state.held_for = step;
    }
    before_step(world, step);
    let mut simulation = {
        let mut state = world.resource_mut::<PhysicsState>();
        state.held_for = delta;
        state.simulation.take()?
    };

    let Simulation { system, bodies, .. } = &mut simulation;
    let interface = system.body_interface();
    for (&entity, tracked) in bodies.iter_mut() {
        if tracked.spec.kind != Kind::Dynamic {
            continue;
        }
        let Some(local) = world.get::<LocalTransform>(entity) else {
            continue;
        };
        let pose = (local.translation, local.rotation);
        drop(local);
        if pose != tracked.pose {
            tracked.teleport(&interface, pose);
        }
    }
    Some(simulation)
}

/// Blend each dynamic body between its last two steps by how far the frame
/// is into the next one.
fn write_back(world: &World, simulation: &mut Simulation, step: f32) {
    let alpha = (simulation.accumulator / step).clamp(0.0, 1.0);
    for (&entity, tracked) in simulation.bodies.iter_mut() {
        if tracked.spec.kind != Kind::Dynamic {
            continue;
        }
        let pose = (
            tracked.previous.0.lerp(tracked.current.0, alpha),
            tracked.previous.1.slerp(tracked.current.1, alpha),
        );
        if let Some(mut local) = world.get_mut::<LocalTransform>(entity) {
            local.translation = pose.0;
            local.rotation = pose.1;
        }
        tracked.pose = pose;
    }
}

fn tell(world: &World, notices: Vec<(Entity, Notice)>) {
    let Some(mut log) = world.get_resource_mut::<LogBuffer>() else {
        return;
    };
    let frame = world
        .get_resource::<Time>()
        .map(|time| time.frame_count())
        .unwrap_or(0);
    for (entity, notice) in notices {
        log.push(LogLevel::Warning, notice.message(entity), frame);
    }
}

fn to_jolt(v: Vec3) -> jolt::Vec3 {
    jolt::Vec3::new(v.x, v.y, v.z)
}

fn from_jolt(v: jolt::Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

fn to_jolt_quat(q: Quat) -> jolt::Quat {
    jolt::Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

fn from_jolt_quat(q: jolt::Quat) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

#[cfg(test)]
mod tests;
