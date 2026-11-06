//! Every story the demo game declares builds, and lays out exactly as
//! recorded — once as built, and again after two seconds of frames have run
//! its scripts and tweens. The recordings are `golden/stories/*.txt`.
//!
//! A story is the smallest piece of UI someone cares enough about to look at
//! alone, which makes it the right unit for a snapshot: a layout or tween
//! change shows up as the stories it moved. The recording is rects, colours
//! and strings rounded to a tenth, not draw commands, so it reads as a review
//! and does not churn on the last bit of a sine.
//!
//! Regenerate with `ORRIN_UPDATE_GOLDEN=1 cargo test -p orrin-core --test
//! stories`, and read the diff first.
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and skips without them, as
//! `scripted_scene` does. One test, because CoreCLR boots once per process.

#![cfg(feature = "scripting")]

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use orrin_core::scene::{Time, register_components};
use orrin_core::scripting::Scripting;
use orrin_core::storybook::build_headless;
use orrin_core::ui::{Gradient, Image, Text, UiTree};
use orrin_ecs::World;
use orrin_registry::Registry;
use orrin_script::StoryBuild;

const SCREEN: [u32; 2] = [1280, 720];
const FRAME: f32 = 1.0 / 60.0;

fn assemblies() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = Scripting::find_bindings_dir(&root.join("scripting/Orrin"))?;
    let game = Scripting::find_game_assembly(&root.join("scripting/DemoGame"), "DemoGame")?;
    Some((bindings, game))
}

fn golden(name: &str) -> PathBuf {
    let slug: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/stories")
        .join(format!("{slug}.txt"))
}

fn rgba(color: [f32; 4]) -> String {
    color.map(|c| format!("{c:.2}")).join(" ")
}

/// The laid-out tree as text: each element's rectangle, then what it draws.
fn record(world: &World, out: &mut String) {
    let tree = world.resource::<UiTree>();
    for (index, element) in tree.layout.elements().iter().enumerate() {
        let r = element.rect;
        let parent = element.parent.map_or("-".to_owned(), |p| p.to_string());
        write!(
            out,
            "  {index} in {parent}: {:.1} {:.1} {:.1}x{:.1}",
            r.x, r.y, r.width, r.height
        )
        .unwrap();
        if let Some(image) = world.get::<Image>(element.entity) {
            write!(out, " image({})", rgba(image.color)).unwrap();
        }
        if let Some(gradient) = world.get::<Gradient>(element.entity) {
            write!(out, " gradient({:.1} deg)", gradient.rotation).unwrap();
        }
        if let Some(text) = world.get::<Text>(element.entity) {
            write!(out, " text({:?} {:.1})", text.text, text.size).unwrap();
        }
        out.push('\n');
    }
    writeln!(out, "  {} commands", tree.canvas.commands().len()).unwrap();
}

#[test]
fn every_story_builds_and_lays_out_as_recorded() {
    let Some((bindings, game)) = assemblies() else {
        eprintln!("skipping: build scripting/Orrin and scripting/DemoGame first");
        return;
    };
    let scripting = Scripting::boot(&bindings, &game).expect("CoreCLR failed to boot");
    let mut registry = Registry::new();
    register_components(&mut registry);
    scripting.register_components(&mut registry);

    let names = scripting.story_names();
    assert_eq!(
        names,
        ["Buttons/Primary", "Cards/Gradient", "Feedback/Toast"],
        "the demo's stories, in name order"
    );
    let (missing, ..) = build_headless(&scripting, &registry, "No/Such story", SCREEN);
    assert_eq!(missing, StoryBuild::Missing);

    let update = std::env::var_os("ORRIN_UPDATE_GOLDEN").is_some();
    let mut stale = Vec::new();
    for name in &names {
        let (built, mut world, _) = build_headless(&scripting, &registry, name, SCREEN);
        assert_eq!(built, StoryBuild::Built, "{name}");

        let mut actual = format!("{name} at {}x{}\n\nbuilt\n", SCREEN[0], SCREEN[1]);
        record(&world, &mut actual);
        for _ in 0..120 {
            world.resource_mut::<Time>().update(FRAME);
            scripting.attach_pending(&mut world, &registry);
            scripting.tick(&mut world, &registry, FRAME);
            orrin_core::ui::tween::advance(&world, FRAME);
            orrin_core::ui::update(&mut world, SCREEN);
        }
        actual.push_str("\nafter two seconds\n");
        record(&world, &mut actual);

        let path = golden(name);
        if update {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, &actual).unwrap();
        } else if fs::read_to_string(&path).unwrap_or_default() != actual {
            stale.push(format!("{}:\n{actual}", path.display()));
        }
    }
    assert!(
        stale.is_empty(),
        "stories no longer lay out as recorded. If the change is intended, \
         regenerate with ORRIN_UPDATE_GOLDEN=1 and read the diff.\n\n{}",
        stale.join("\n")
    );
}
