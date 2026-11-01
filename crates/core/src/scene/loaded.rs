//! Which scenes are loaded, and loading more of them while the game runs.
//!
//! One scene is *active*: the one the editor opened, or the last one a game
//! loaded in its place. Others can be loaded beside it, *additively*, and
//! unloaded again. A project may also name a *persistent* scene, which is
//! loaded additively when the game starts and survives every load after it —
//! where a game keeps what Unity would keep with `DontDestroyOnLoad`. So every
//! entity belongs to a real scene document, and nothing exists outside one.
//!
//! Membership is a component, [`InScene`], carried by the entities of every
//! scene but the active one. An entity without it belongs to the scene of its
//! nearest ancestor with one — something a script parents under a scene's
//! entity has been put in that scene — and otherwise to the active scene. That
//! is the rule that needs no bookkeeping where most entities are made: the active scene's own load, the editor's spawns and every entity a
//! script makes at runtime are all simply the active scene's, and a load in
//! its place takes them with it. The inverse — tagging the active scene and
//! leaving the others bare — would need every one of those paths to tag.
//!
//! [`InScene`] is not registered: which scenes happen to be loaded beside
//! each other is this session's business, and a saved file would otherwise
//! name a handle that means nothing the next time it is opened.
//!
//! A [`SceneHandle`] names a loaded scene for as long as it stays loaded, and
//! never again after: handles are never reused, so a script holding one for a
//! scene that has gone is told so instead of unloading whatever came next.
//!
//! The functions here change the world at once. Script calls do not reach
//! them directly — a load despawns entities, and with them script instances,
//! which may only happen outside a dispatch window — so `scripting` queues
//! them and applies them after the tick, as it does every structural change.

use orrin_ecs::{Entity, FxBuildHasher, World};
use orrin_registry::{AssetId, Registry, SceneDocument};

use super::{LoadIssue, Parent};

type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

/// What a scene file's name ends in.
pub const EXTENSION: &str = "orrin";

/// A loaded scene, as long as it stays loaded. Never zero, so zero can mean
/// "no scene" across the script ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneHandle(std::num::NonZeroU32);

impl SceneHandle {
    pub fn get(self) -> u32 {
        self.0.get()
    }

    pub fn from_raw(raw: u32) -> Option<Self> {
        std::num::NonZeroU32::new(raw).map(Self)
    }
}

/// The loaded scene an entity belongs to, on the entities of every scene but
/// the active one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InScene(pub SceneHandle);

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedScene {
    pub handle: SceneHandle,
    /// The scene file's asset, when it is one. The editor's scene outside a
    /// project has none.
    pub asset: Option<AssetId>,
    /// What the scene is called where it is listed: its file's name.
    pub name: String,
    /// Kept through every [`load`].
    pub persistent: bool,
}

/// Every loaded scene, the active one first.
pub struct LoadedScenes {
    scenes: Vec<LoadedScene>,
    next: u32,
}

impl Default for LoadedScenes {
    fn default() -> Self {
        let mut loaded = Self {
            scenes: Vec::new(),
            next: 1,
        };
        let handle = loaded.reserve();
        loaded.scenes.push(LoadedScene {
            handle,
            asset: None,
            name: String::new(),
            persistent: false,
        });
        loaded
    }
}

impl LoadedScenes {
    pub fn active(&self) -> &LoadedScene {
        &self.scenes[0]
    }

    /// The active scene first, then the others in the order they loaded.
    pub fn iter(&self) -> impl Iterator<Item = &LoadedScene> {
        self.scenes.iter()
    }

    pub fn get(&self, handle: SceneHandle) -> Option<&LoadedScene> {
        self.scenes.iter().find(|scene| scene.handle == handle)
    }

    /// A handle for a scene about to be loaded additively. Handed out before
    /// the load, so a script gets one back from the call that queues it.
    pub fn reserve(&mut self) -> SceneHandle {
        let handle = SceneHandle::from_raw(self.next).expect("starts at one");
        self.next += 1;
        handle
    }

    /// The editor opened a scene, or put one back after Play: it alone is
    /// loaded now, under a new handle.
    pub fn reset(&mut self, name: impl Into<String>, asset: Option<AssetId>) {
        let handle = self.reserve();
        self.scenes.clear();
        self.scenes.push(LoadedScene {
            handle,
            asset,
            name: name.into(),
            persistent: false,
        });
    }
}

/// The loaded scene `entity` belongs to: see the module docs.
pub fn scene_of(world: &World, entity: Entity) -> SceneHandle {
    let mut cursor = entity;
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        if let Some(member) = world.get::<InScene>(cursor) {
            return member.0;
        }
        match world.get::<Parent>(cursor).map(|parent| parent.get()) {
            Some(parent) if world.is_alive(parent) => cursor = parent,
            _ => break,
        }
    }
    active(world)
}

fn active(world: &World) -> SceneHandle {
    world
        .get_resource::<LoadedScenes>()
        .map_or(SceneHandle(std::num::NonZeroU32::MIN), |loaded| {
            loaded.active().handle
        })
}

fn ensure(world: &mut World) {
    if world.get_resource::<LoadedScenes>().is_none() {
        world.insert_resource(LoadedScenes::default());
    }
}

/// Load `document` as the active scene, in place of the one that was and of
/// every additive scene but the persistent ones.
pub fn load(
    world: &mut World,
    registry: &Registry,
    document: &SceneDocument,
    name: impl Into<String>,
    asset: Option<AssetId>,
) -> Vec<LoadIssue> {
    ensure(world);
    let kept: Vec<SceneHandle> = world
        .resource::<LoadedScenes>()
        .iter()
        .filter(|scene| scene.persistent)
        .map(|scene| scene.handle)
        .collect();
    let doomed: Vec<Entity> = world
        .entities()
        .filter(|&entity| !kept.contains(&scene_of(world, entity)))
        .collect();
    for entity in doomed {
        world.despawn(entity);
    }

    let issues = super::instantiate(document, world, registry);
    let mut loaded = world.resource_mut::<LoadedScenes>();
    let handle = loaded.reserve();
    loaded.scenes.retain(|scene| scene.persistent);
    loaded.scenes.insert(
        0,
        LoadedScene {
            handle,
            asset,
            name: name.into(),
            persistent: false,
        },
    );
    issues
}

/// Load `document` beside what is loaded, under `handle` from
/// [`LoadedScenes::reserve`].
///
/// What it spawned is told apart from what was there by the entities alive
/// before it, which costs a set the size of the world once per load — the
/// alternative, threading a tag through `instantiate` and the prefab
/// expansion inside it, would put the knowledge of scenes into every spawn.
pub fn load_additive(
    world: &mut World,
    registry: &Registry,
    handle: SceneHandle,
    document: &SceneDocument,
    name: impl Into<String>,
    asset: Option<AssetId>,
    persistent: bool,
) -> Vec<LoadIssue> {
    ensure(world);
    let before: FxHashSet<Entity> = world.entities().collect();
    let issues = super::instantiate(document, world, registry);
    let fresh: Vec<Entity> = world
        .entities()
        .filter(|entity| !before.contains(entity))
        .collect();
    for entity in fresh {
        world.insert(entity, InScene(handle));
    }
    world
        .resource_mut::<LoadedScenes>()
        .scenes
        .push(LoadedScene {
            handle,
            asset,
            name: name.into(),
            persistent,
        });
    issues
}

/// The scene file asset `id` names, parsed, and the name it is listed under.
/// Every failure names the scene and why, for the console.
pub fn read(world: &World, id: AssetId) -> Result<(SceneDocument, String), String> {
    let Some(db) = world.get_resource::<crate::asset_db::AssetDatabase>() else {
        return Err(format!(
            "scene {id}: there is no asset database to find it in; scenes are loaded by \
             reference only inside a project"
        ));
    };
    let Some(entry) = db.get(id) else {
        return Err(format!("scene {id}: no asset in the project has this id"));
    };
    if entry.kind != crate::asset_db::AssetKind::Scene {
        return Err(format!(
            "{}: is a {}, not a scene",
            entry.path,
            entry.kind.label()
        ));
    }
    let path = db.dir().join(&entry.path);
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("{}: could not read it: {error}", entry.path))?;
    let document =
        orrin_registry::parse(&text).map_err(|error| format!("{}: {error}", entry.path))?;
    Ok((document, name_of(&entry.path)))
}

/// The scene asset at `path`, when the project's database indexes one there.
pub fn asset_at(world: &World, path: &std::path::Path) -> Option<AssetId> {
    let db = world.get_resource::<crate::asset_db::AssetDatabase>()?;
    let path = std::fs::canonicalize(path).ok()?;
    let dir = std::fs::canonicalize(db.dir()).ok()?;
    let rel: Vec<String> = path
        .strip_prefix(dir)
        .ok()?
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    db.find(&rel.join("/"))
        .filter(|entry| entry.kind == crate::asset_db::AssetKind::Scene)
        .map(|entry| entry.id)
}

/// What a scene at `path` is listed as: its file's name without the extension.
pub fn name_of(path: &str) -> String {
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    file.strip_suffix(&format!(".{EXTENSION}"))
        .unwrap_or(file)
        .to_owned()
}

/// Despawn the scene `handle` names, and everything parented into it.
pub fn unload(world: &mut World, handle: SceneHandle) -> Result<(), String> {
    ensure(world);
    {
        let loaded = world.resource::<LoadedScenes>();
        if loaded.active().handle == handle {
            return Err(format!(
                "`{}` is the active scene and cannot be unloaded; load another scene in its \
                 place instead",
                loaded.active().name
            ));
        }
        if loaded.get(handle).is_none() {
            return Err(format!(
                "scene {} is not loaded; it was unloaded already, or a load replaced it",
                handle.get()
            ));
        }
    }
    let doomed: Vec<Entity> = world
        .entities()
        .filter(|&entity| scene_of(world, entity) == handle)
        .collect();
    for entity in doomed {
        world.despawn(entity);
    }
    world
        .resource_mut::<LoadedScenes>()
        .scenes
        .retain(|scene| scene.handle != handle);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::render_settings::{self, RenderSettings};
    use crate::scene::{Name, register_components};

    fn session() -> (World, Registry) {
        let mut world = World::new();
        world.insert_resource(LoadedScenes::default());
        let mut registry = Registry::new();
        register_components(&mut registry);
        (world, registry)
    }

    fn named(world: &mut World, name: &str) -> Entity {
        let entity = world.spawn();
        world.insert(entity, Name::new(name));
        entity
    }

    /// A scene file holding one entity of each name.
    fn scene(registry: &Registry, names: &[&str]) -> SceneDocument {
        let mut world = World::new();
        for name in names {
            named(&mut world, name);
        }
        crate::scene::to_document(&mut world, registry)
    }

    fn alive(world: &World) -> Vec<String> {
        let mut names: Vec<String> = world
            .entities()
            .filter_map(|entity| world.get::<Name>(entity).map(|name| name.0.clone()))
            .collect();
        names.sort();
        names
    }

    fn listed(world: &World) -> Vec<String> {
        world
            .resource::<LoadedScenes>()
            .iter()
            .map(|scene| scene.name.clone())
            .collect()
    }

    fn additive(
        world: &mut World,
        registry: &Registry,
        name: &str,
        document: &SceneDocument,
        persistent: bool,
    ) -> SceneHandle {
        let handle = world.resource_mut::<LoadedScenes>().reserve();
        load_additive(world, registry, handle, document, name, None, persistent);
        handle
    }

    /// The menu-to-level case: what was playing goes, the new level comes, and
    /// the persistent scene's music player is still there.
    #[test]
    fn a_load_replaces_the_active_scene_and_keeps_the_persistent_one() {
        let (mut world, registry) = session();
        named(&mut world, "Menu");
        let persistent = scene(&registry, &["Music"]);
        additive(&mut world, &registry, "persistent", &persistent, true);

        load(
            &mut world,
            &registry,
            &scene(&registry, &["Level"]),
            "level",
            None,
        );
        assert_eq!(alive(&world), ["Level", "Music"]);
        assert_eq!(listed(&world), ["level", "persistent"]);
    }

    /// A load unloads the gameplay scenes loaded beside the old one, not only
    /// the old one itself.
    #[test]
    fn a_load_takes_the_additive_scenes_with_it() {
        let (mut world, registry) = session();
        named(&mut world, "Hub");
        additive(
            &mut world,
            &registry,
            "shop",
            &scene(&registry, &["Shopkeeper"]),
            false,
        );

        load(
            &mut world,
            &registry,
            &scene(&registry, &["Dungeon"]),
            "dungeon",
            None,
        );
        assert_eq!(alive(&world), ["Dungeon"]);
        assert_eq!(listed(&world), ["dungeon"]);
    }

    #[test]
    fn an_additive_load_keeps_what_is_loaded() {
        let (mut world, registry) = session();
        let hub = named(&mut world, "Hub");
        let shop = additive(
            &mut world,
            &registry,
            "shop",
            &scene(&registry, &["Shopkeeper"]),
            false,
        );

        assert_eq!(alive(&world), ["Hub", "Shopkeeper"]);
        let keeper = world
            .entities()
            .find(|&e| world.get::<Name>(e).is_some_and(|n| n.0 == "Shopkeeper"))
            .unwrap();
        assert_eq!(scene_of(&world, keeper), shop);
        assert_eq!(
            scene_of(&world, hub),
            world.resource::<LoadedScenes>().active().handle
        );
    }

    #[test]
    fn an_unload_takes_only_its_own_scene() {
        let (mut world, registry) = session();
        named(&mut world, "Hub");
        let shop = additive(
            &mut world,
            &registry,
            "shop",
            &scene(&registry, &["Shopkeeper"]),
            false,
        );
        additive(
            &mut world,
            &registry,
            "bank",
            &scene(&registry, &["Teller"]),
            false,
        );

        unload(&mut world, shop).unwrap();
        assert_eq!(alive(&world), ["Hub", "Teller"]);
        assert_eq!(listed(&world), ["", "bank"]);
    }

    /// A script that parents something it spawned under a scene's entity has
    /// put it in that scene; unloading the scene must not leave it orphaned.
    #[test]
    fn an_unload_takes_what_was_parented_into_the_scene() {
        let (mut world, registry) = session();
        let shop = additive(
            &mut world,
            &registry,
            "shop",
            &scene(&registry, &["Shopkeeper"]),
            false,
        );
        let keeper = world.entities().next().unwrap();
        let coin = named(&mut world, "Coin");
        crate::scene::reparent(&mut world, coin, Some(keeper), false).unwrap();

        unload(&mut world, shop).unwrap();
        assert!(alive(&world).is_empty(), "{:?}", alive(&world));
    }

    /// There is always an active scene. Replacing it is a load.
    #[test]
    fn the_active_scene_cannot_be_unloaded() {
        let (mut world, _) = session();
        named(&mut world, "Hub");
        let active = world.resource::<LoadedScenes>().active().handle;
        let error = unload(&mut world, active).unwrap_err();
        assert!(error.contains("active"), "{error}");
        assert_eq!(alive(&world), ["Hub"]);
    }

    /// A handle outlives its scene in a script's field. Using it then must say
    /// so, not unload whatever was loaded after.
    #[test]
    fn a_stale_handle_is_refused() {
        let (mut world, registry) = session();
        let shop = additive(
            &mut world,
            &registry,
            "shop",
            &scene(&registry, &["Shopkeeper"]),
            false,
        );
        unload(&mut world, shop).unwrap();
        additive(
            &mut world,
            &registry,
            "bank",
            &scene(&registry, &["Teller"]),
            false,
        );

        assert!(unload(&mut world, shop).is_err());
        assert_eq!(alive(&world), ["Teller"]);
    }

    /// A script's reference is an id; the file it names is found through the
    /// asset database, and anything else under that id is refused by name.
    #[test]
    fn a_scene_is_read_by_its_asset_id() {
        let (mut world, registry) = session();
        let text = {
            let mut source = World::new();
            named(&mut source, "Gate");
            crate::scene::save(&mut source, &registry)
        };
        let dir = crate::asset_db::tests::Dir::new("loaded-read");
        dir.write("levels/castle.orrin", text.as_bytes());
        dir.write("rock.png", b"");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let castle = db
            .find("levels/castle.orrin")
            .expect("a scene is an asset")
            .id;
        let rock = db.find("rock.png").unwrap().id;
        world.insert_resource(db);

        let (document, name) = read(&world, castle).unwrap();
        assert_eq!(name, "castle");
        load(&mut world, &registry, &document, name, Some(castle));
        assert_eq!(alive(&world), ["Gate"]);
        assert_eq!(
            world.resource::<LoadedScenes>().active().asset,
            Some(castle)
        );

        let error = read(&world, rock).unwrap_err();
        assert!(
            error.contains("rock.png") && error.contains("not a scene"),
            "{error}"
        );
    }

    /// The active scene says how the frame is drawn; the persistent scene's
    /// settings stand in only when the active one has none.
    #[test]
    fn the_active_scenes_render_settings_win() {
        let (mut world, registry) = session();
        let mut persistent = World::new();
        let entity = persistent.spawn();
        let mut settings = RenderSettings::default();
        settings.fog.density = 0.1;
        persistent.insert(entity, settings);
        let persistent = crate::scene::to_document(&mut persistent, &registry);
        additive(&mut world, &registry, "persistent", &persistent, true);

        let fallback = render_settings::source(&world).unwrap();
        assert_eq!(
            world.get::<RenderSettings>(fallback).unwrap().fog.density,
            0.1
        );

        let own = world.spawn();
        let mut settings = RenderSettings::default();
        settings.fog.density = 0.2;
        world.insert(own, settings);
        assert_eq!(render_settings::source(&world), Some(own));
    }
}
