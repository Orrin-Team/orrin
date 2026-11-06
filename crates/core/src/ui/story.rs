//! The storybook's model: which stories the game declares, which one is open,
//! and the controls it was built with — everything the editor's Storybook tab
//! reads and writes, with no script host in sight.
//!
//! # A story is a world
//!
//! A story is a C# method marked `[Story]` that builds UI. It is built into a
//! [`World`] of its own, made by [`world`] and staged by [`stage`], never into
//! the scene: opening one cannot dirty, save into, or be undone out of the
//! scene being edited, and the scene's own UI cannot be under it. It is drawn
//! in a window of its own, laid out on that window's size. The world
//! carries exactly the resources the UI path and the script ABI read, so a
//! story's behaviours tick, take the pointer and start tweens there as they
//! would in a game. The project-wide ones — the asset database and the
//! prefabs — are lent to it from the editor's world for each frame rather than
//! copied, and its input is its window's, which is why nothing here owns one.
//!
//! # Controls are asked for, not declared
//!
//! A story asks for a control by name each time it builds — `ctx.Float("Width",
//! 160, 80, 400)` — and gets back what the Storybook holds for that name. So the
//! story's code is the only list of its controls, and a story edited to add one
//! shows it on the next build. [`StoryArgs::rebuild`] is the seam: what was set
//! before a build is held, each ask during it takes the held value (clamped to
//! the range it asks with now), and a control the new code no longer asks for
//! is dropped when the build ends.

use orrin_ecs::{Entity, World};

use super::{Canvas, Image, RectTransform, UiTree};
use crate::scene::{Name, Time, UiCanvas, reparent};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StoryValue {
    Float { value: f32, min: f32, max: f32 },
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq)]
pub struct StoryControl {
    pub name: String,
    pub value: StoryValue,
}

/// The open story's controls: a resource of the story's world while it builds,
/// and of the editor's in the [`Storybook`] between builds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StoryArgs {
    controls: Vec<StoryControl>,
    held: Vec<StoryControl>,
}

impl StoryArgs {
    /// Start a build: every control becomes a held value, waiting to be asked
    /// for again.
    pub fn rebuild(self) -> Self {
        Self {
            controls: Vec::new(),
            held: self.controls,
        }
    }

    pub fn float(&mut self, name: &str, default: f32, min: f32, max: f32) -> f32 {
        let (min, max) = if min <= max { (min, max) } else { (max, min) };
        let fit = |value: f32| {
            if value.is_nan() {
                min
            } else {
                value.clamp(min, max)
            }
        };
        let value = self.ask(name, |held| match held {
            Some(StoryValue::Float { value, .. }) => StoryValue::Float {
                value: fit(value),
                min,
                max,
            },
            _ => StoryValue::Float {
                value: fit(default),
                min,
                max,
            },
        });
        match value {
            StoryValue::Float { value, .. } => value,
            StoryValue::Bool(_) => fit(default),
        }
    }

    pub fn bool(&mut self, name: &str, default: bool) -> bool {
        let value = self.ask(name, |held| match held {
            Some(StoryValue::Bool(value)) => StoryValue::Bool(value),
            _ => StoryValue::Bool(default),
        });
        match value {
            StoryValue::Bool(value) => value,
            StoryValue::Float { .. } => default,
        }
    }

    /// The control `name` as this build already declared it, or as `declare`
    /// makes it from what was held.
    fn ask(
        &mut self,
        name: &str,
        declare: impl FnOnce(Option<StoryValue>) -> StoryValue,
    ) -> StoryValue {
        if let Some(control) = self.controls.iter().find(|control| control.name == name) {
            return control.value;
        }
        let held = self
            .held
            .iter()
            .find(|control| control.name == name)
            .map(|control| control.value);
        let value = declare(held);
        self.controls.push(StoryControl {
            name: name.to_owned(),
            value,
        });
        value
    }

    pub fn controls(&self) -> &[StoryControl] {
        &self.controls
    }

    pub fn controls_mut(&mut self) -> &mut [StoryControl] {
        &mut self.controls
    }
}

/// How the open story's last build went.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StoryStatus {
    #[default]
    Closed,
    Built,
    /// It threw; the console has why.
    Faulted,
    /// The game no longer declares it — renamed, or deleted, since the list was
    /// read.
    Missing,
}

/// The storybook as the editor sees it: a resource of the editor's world.
#[derive(Debug, Default)]
pub struct Storybook {
    /// Every story the game declares, by name, in name order.
    pub stories: Vec<String>,
    /// The story to show. Setting it opens that story on the next frame.
    pub open: Option<String>,
    pub args: StoryArgs,
    /// Build the open story again on the next frame: a control moved, or the
    /// scripts reloaded.
    pub rebuild: bool,
    pub status: StoryStatus,
    /// Whether there is a script host to build with at all.
    pub available: bool,
}

impl Storybook {
    /// Whether a story is open, and so has a window.
    pub fn showing(&self) -> bool {
        self.open.is_some()
    }
}

/// The colour behind every story, so a story is judged against one neutral
/// ground rather than whatever the scene's camera is looking at.
pub const BACKDROP: [f32; 4] = [0.105, 0.11, 0.12, 1.0];

/// An empty world with what a story's UI, its scripts and its tweens read.
pub fn world() -> World {
    let mut world = World::new();
    world.insert_resource(UiTree::default());
    world.insert_resource(UiCanvas::default());
    world.insert_resource(super::events::UiPointer::default());
    world.insert_resource(super::events::UiEvents::default());
    world.insert_resource(super::tween::UiTweens::default());
    world.insert_resource(Time::new());
    world.insert_resource(StoryArgs::default());
    world
}

/// Put the backdrop and an empty root canvas into a fresh story `world`,
/// returning the canvas the story builds onto.
pub fn stage(world: &mut World) -> Entity {
    let ground = world.spawn();
    world.insert(ground, Name::new("Backdrop"));
    world.insert(
        ground,
        Canvas {
            sort_order: i32::MIN,
            ..Canvas::default()
        },
    );
    let fill = world.spawn();
    world.insert(fill, Name::new("Backdrop fill"));
    world.insert(fill, RectTransform::stretched());
    world.insert(
        fill,
        Image {
            color: BACKDROP,
            raycast_target: false,
            ..Image::default()
        },
    );
    reparent(world, fill, Some(ground), false).expect("a new entity cannot cycle");

    let canvas = world.spawn();
    world.insert(canvas, Name::new("Story"));
    world.insert(canvas, Canvas::default());
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::GameScreen;
    use crate::ui::tween::UiTweens;

    #[test]
    fn a_control_is_its_default_until_it_is_set() {
        let mut args = StoryArgs::default();
        assert_eq!(args.float("Width", 160.0, 80.0, 400.0), 160.0);
        assert!(!args.bool("Disabled", false));
        assert_eq!(args.controls().len(), 2);
        assert_eq!(
            args.float("Width", 999.0, 0.0, 1.0),
            160.0,
            "a second ask in the same build is the first's control"
        );
        assert_eq!(args.controls().len(), 2);
    }

    #[test]
    fn a_set_value_survives_a_rebuild_clamped_to_the_range_asked_for_now() {
        let mut args = StoryArgs::default();
        args.float("Width", 160.0, 80.0, 400.0);
        args.bool("Disabled", false);
        for control in args.controls_mut() {
            control.value = match control.value {
                StoryValue::Float { min, max, .. } => StoryValue::Float {
                    value: 300.0,
                    min,
                    max,
                },
                StoryValue::Bool(_) => StoryValue::Bool(true),
            };
        }
        let mut args = args.rebuild();
        assert_eq!(args.float("Width", 160.0, 80.0, 250.0), 250.0);
        assert!(args.bool("Disabled", false));
    }

    #[test]
    fn a_control_the_story_stops_asking_for_is_dropped() {
        let mut args = StoryArgs::default();
        args.float("Width", 1.0, 0.0, 2.0);
        args.bool("Old", true);
        let mut args = args.rebuild();
        args.float("Width", 1.0, 0.0, 2.0);
        let args = args.rebuild();
        assert!(
            args.controls().is_empty(),
            "nothing is declared until asked"
        );
        let mut args = args;
        assert!(!args.bool("Old", false), "dropped two builds ago");
    }

    /// A control renamed from a slider to a checkbox is a new control, not a
    /// float read as a bool.
    #[test]
    fn a_held_value_of_another_kind_is_not_reused() {
        let mut args = StoryArgs::default();
        args.float("Flag", 0.75, 0.0, 1.0);
        let mut args = args.rebuild();
        assert!(args.bool("Flag", true));
    }

    #[test]
    fn a_backwards_or_unusable_range_still_gives_a_value_inside_it() {
        let mut args = StoryArgs::default();
        assert_eq!(args.float("A", 5.0, 10.0, 0.0), 5.0);
        let a = args.float("B", f32::NAN, 0.0, 1.0);
        assert_eq!(a, 0.0);
        assert!(matches!(
            args.controls()[0].value,
            StoryValue::Float {
                min: 0.0,
                max: 10.0,
                ..
            }
        ));
    }

    #[test]
    fn a_staged_world_lays_out_a_backdrop_under_the_story_canvas() {
        let mut world = world();
        assert!(world.get_resource::<UiTweens>().is_some());
        assert!(world.get_resource::<StoryArgs>().is_some());
        let canvas = stage(&mut world);
        assert!(world.get::<Canvas>(canvas).is_some());
        world.insert_resource(GameScreen::window([640, 480]));
        crate::ui::update(&mut world, [640, 480]);
        let tree = world.resource::<UiTree>();
        let backdrop = tree.layout.elements();
        assert_eq!(backdrop.len(), 1, "the backdrop, and an empty story canvas");
        let rect = backdrop[0].rect;
        assert_eq!((rect.width, rect.height), (640.0, 480.0));
        assert!(
            !world
                .get::<Image>(backdrop[0].entity)
                .unwrap()
                .raycast_target,
            "the backdrop never takes the pointer from the story"
        );
    }
}
