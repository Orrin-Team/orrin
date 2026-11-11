//! The storybook's runner: builds the open story into its world with the
//! script host, and runs that world a frame at a time in a window of its own.
//!
//! The model it serves — which story is open, its controls — is
//! `ui::story::Storybook`, a resource of the editor's world, so the editor
//! never needs the host. This is the half that does.
//!
//! # A window, not a viewport
//!
//! A story is drawn in a second OS window (`gfx::vulkan::ui_window`), so the
//! scene's viewport keeps showing the game's own UI while a story is open, and
//! a story can be looked at at the size of a phone or of a second monitor. The
//! story's screen is that window: its size is the [`GameScreen`] the story is
//! laid out on, and its pointer and keys are an [`InputState`] the runner keeps
//! across rebuilds, fed only by that window's events. The window follows the
//! model by [`window_change`], once a frame.
//!
//! # Lent, not shared
//!
//! A story world has no asset database or prefab index of its own: the project
//! has one of each, and a story looking up a font has to see the project's.
//! [`StoryRunner`] moves them out of the editor's world into the story's for
//! the length of a build or a frame, and back, so there is never a second copy
//! to drift and nothing to keep in sync. Named actions stay with the game: they
//! are bound to the editor window's input, which the story's window is not.
//!
//! # Rebuilt, never patched
//!
//! A story is built again from nothing whenever a control moves or the scripts
//! reload, by the same code that built it the first time. Its world is dropped
//! before the game's assembly is swapped: a behaviour attached in a story holds
//! a handle into that assembly, and one survivor pins the old load context
//! (see the reload's commit-after-teardown rule).

use orrin_ecs::{Entity, World};
use orrin_registry::Registry;

use crate::asset_db::AssetDatabase;
use crate::scene::input::InputState;
use crate::scene::prefab::Prefabs;
use crate::scene::{GameScreen, Time, UiCanvas};
use crate::scripting::Scripting;
use crate::ui::story::{self, StoryStatus, Storybook};

/// The open story's world, if one is open, and its window's input.
pub struct StoryRunner {
    open: Option<Open>,
    input: InputState,
    extent: [u32; 2],
}

impl Default for StoryRunner {
    fn default() -> Self {
        Self {
            open: None,
            input: InputState::new(),
            extent: [1280, 720],
        }
    }
}

/// What the story window has to do for the window to show the story the
/// model has open, given the story it shows now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowChange {
    Keep,
    Open,
    /// Another story, in the window already open.
    Retitle,
    Close,
}

pub fn window_change(wanted: Option<&str>, shown: Option<&str>) -> WindowChange {
    match (wanted, shown) {
        (None, None) => WindowChange::Keep,
        (Some(_), None) => WindowChange::Open,
        (None, Some(_)) => WindowChange::Close,
        (Some(wanted), Some(shown)) if wanted == shown => WindowChange::Keep,
        (Some(_), Some(_)) => WindowChange::Retitle,
    }
}

struct Open {
    name: String,
    world: World,
}

impl StoryRunner {
    /// Drop the story's world, and every behaviour in it. Before a reload,
    /// and when the window closes.
    pub fn close(&mut self) {
        self.open = None;
    }

    /// An event from the story's window.
    pub fn on_window_event(&mut self, event: &winit::event::WindowEvent) {
        if let winit::event::WindowEvent::Resized(size) = event {
            self.set_extent([size.width, size.height]);
        }
        self.input.on_window_event(event, false);
    }

    /// The window's size in physical pixels, which the story is laid out on.
    pub fn set_extent(&mut self, extent: [u32; 2]) {
        self.extent = extent.map(|side| side.max(1));
    }

    /// Clear the frame's pressed and released edges, once the story's scripts
    /// have seen them.
    pub fn end_frame(&mut self) {
        self.input.end_frame();
    }

    /// Read the game's stories again, and rebuild the open one: after the host
    /// boots and after every reload.
    pub fn refresh(&mut self, main: &World, scripting: &Scripting) {
        let mut book = main.resource_mut::<Storybook>();
        book.available = true;
        book.stories = scripting.story_names();
        book.rebuild |= book.open.is_some();
    }

    /// Whether a story is open, built or not.
    pub fn is_open(&self) -> bool {
        self.open.is_some()
    }

    /// Build the story the editor asks for if it is not the one standing, then
    /// run its world one frame of `delta` seconds: pointer, scripts, tweens,
    /// layout.
    pub fn update(
        &mut self,
        main: &mut World,
        scripting: &Scripting,
        registry: &Registry,
        delta: f32,
    ) {
        let (wanted, rebuild) = {
            let book = main.resource::<Storybook>();
            (book.open.clone(), book.rebuild)
        };
        let Some(name) = wanted else {
            self.open = None;
            main.resource_mut::<Storybook>().status = StoryStatus::Closed;
            return;
        };
        if rebuild || self.open.as_ref().is_none_or(|open| open.name != name) {
            self.build(main, scripting, registry, name);
        }
        let extent = self.extent;
        let Some(open) = &mut self.open else {
            return;
        };
        with_lent(main, &mut open.world, &mut self.input, extent, |world| {
            world.resource_mut::<Time>().update(delta, 1.0);
            let size = world.resource::<GameScreen>().size;
            world
                .resource_mut::<UiCanvas>()
                .begin(size.map(|side| side.round() as u32));
            crate::ui::events::update(world);
            scripting.attach_pending(world, registry);
            scripting.tick(world, registry, delta);
            crate::ui::tween::advance(world, delta);
            crate::ui::update(world, extent);
        });
    }

    /// Run `paint` against the open story's world, with the project's assets
    /// lent to it. False, and `paint` not run, when no story is open.
    pub fn paint(&mut self, main: &mut World, paint: impl FnOnce(&World)) -> bool {
        let Some(open) = &mut self.open else {
            return false;
        };
        with_lent(
            main,
            &mut open.world,
            &mut self.input,
            self.extent,
            |world| paint(world),
        );
        true
    }

    fn build(
        &mut self,
        main: &mut World,
        scripting: &Scripting,
        registry: &Registry,
        name: String,
    ) {
        // The old world first, so its behaviours are gone before the new
        // build's are created.
        self.open = None;
        let mut world = story::world();
        let canvas = story::stage(&mut world);
        let args = std::mem::take(&mut main.resource_mut::<Storybook>().args);
        let (built, args) = with_lent(main, &mut world, &mut self.input, self.extent, |world| {
            scripting.build_story(world, registry, &name, canvas, args)
        });
        let mut book = main.resource_mut::<Storybook>();
        book.args = args;
        book.rebuild = false;
        book.status = match built {
            orrin_script::StoryBuild::Built => StoryStatus::Built,
            orrin_script::StoryBuild::Faulted => StoryStatus::Faulted,
            orrin_script::StoryBuild::Missing => StoryStatus::Missing,
        };
        drop(book);
        self.open = Some(Open { name, world });
    }
}

/// Build story `name` with no editor around it, onto a screen of `size`: what
/// the storybook's golden test records. The world is returned laid out.
pub fn build_headless(
    scripting: &Scripting,
    registry: &Registry,
    name: &str,
    size: [u32; 2],
) -> (orrin_script::StoryBuild, World, Entity) {
    let mut world = story::world();
    world.insert_resource(GameScreen::window(size));
    let canvas = story::stage(&mut world);
    let (built, _) = scripting.build_story(
        &mut world,
        registry,
        name,
        canvas,
        story::StoryArgs::default(),
    );
    crate::ui::update(&mut world, size);
    (built, world, canvas)
}

/// Run `body` on `story` with the project's resources moved into it from
/// `main`, and moved back after, and with the story window's `input` and a
/// screen of its `extent`.
fn with_lent<R>(
    main: &mut World,
    story: &mut World,
    input: &mut InputState,
    extent: [u32; 2],
    body: impl FnOnce(&mut World) -> R,
) -> R {
    fn give<T: 'static>(from: &mut World, to: &mut World) {
        if let Some(resource) = from.remove_resource::<T>() {
            to.insert_resource(resource);
        }
    }
    fn lend_all(from: &mut World, to: &mut World) {
        give::<AssetDatabase>(from, to);
        give::<Prefabs>(from, to);
    }
    story.insert_resource(GameScreen::window(extent));
    story.insert_resource(std::mem::take(input));
    lend_all(main, story);
    let result = body(story);
    lend_all(story, main);
    *input = story.remove_resource::<InputState>().unwrap_or_default();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_project_is_lent_and_the_story_has_its_own_pointer_and_screen() {
        let mut main = World::new();
        main.insert_resource(Prefabs::default());
        let mut main_input = InputState::new();
        main_input.place_cursor((5.0, 5.0));
        main.insert_resource(main_input);
        main.insert_resource(GameScreen::window([1920, 1080]));
        let mut story = story::world();
        let mut input = InputState::new();
        input.place_cursor((40.0, 30.0));
        let seen = with_lent(&mut main, &mut story, &mut input, [800, 600], |story| {
            (
                story.get_resource::<Prefabs>().is_some(),
                story.resource::<InputState>().cursor(),
                story.get_resource::<AssetDatabase>().is_some(),
                story.resource::<GameScreen>().size,
            )
        });
        assert_eq!(seen, (true, (40.0, 30.0), false, [800.0, 600.0]));
        assert!(main.get_resource::<Prefabs>().is_some());
        assert!(story.get_resource::<Prefabs>().is_none());
        assert_eq!(main.resource::<InputState>().cursor(), (5.0, 5.0));
        assert_eq!(input.cursor(), (40.0, 30.0), "the story's input comes back");
        assert!(
            main.get_resource::<AssetDatabase>().is_none(),
            "nothing appears that was not there"
        );
        assert_eq!(main.resource::<GameScreen>().size, [1920.0, 1080.0]);
    }

    #[test]
    fn the_window_follows_the_open_story() {
        use WindowChange::*;
        assert_eq!(window_change(None, None), Keep);
        assert_eq!(window_change(Some("A"), None), Open);
        assert_eq!(window_change(Some("A"), Some("A")), Keep);
        assert_eq!(window_change(Some("B"), Some("A")), Retitle);
        assert_eq!(window_change(None, Some("A")), Close);
    }
}
