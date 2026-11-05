use std::ops::Range;

use orrin_ecs::{Entity, World};

use super::{Button, Layout, UiTree};
use crate::scene::{InputState, Parent};

/// What happened to the pointer over an element. Lock-step with C#
/// `Orrin.PointerEventKind`: append, never renumber.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum PointerKind {
    Enter,
    Exit,
    Down,
    Up,
    Click,
}

/// One event, and who it goes to.
#[derive(Clone, Debug, PartialEq)]
pub struct PointerEvent {
    pub kind: PointerKind,
    /// The element the pointer hit; for `Up` and `Click`, the one it went
    /// down on.
    pub target: Entity,
    /// Physical pixels from the window's top left.
    pub position: [f32; 2],
    /// Where the button went down, for `Down`, `Up` and `Click`.
    pub press_position: [f32; 2],
    /// Into [`UiEvents::chain`].
    chain: Range<u32>,
}

/// This tick's pointer events, in the order they happened. A world resource,
/// drained by the script tick.
#[derive(Default)]
pub struct UiEvents {
    events: Vec<PointerEvent>,
    /// Every event's receivers end to end, so a frame of events is two kept
    /// allocations rather than one per event.
    chains: Vec<Entity>,
}

impl UiEvents {
    pub fn events(&self) -> &[PointerEvent] {
        &self.events
    }

    /// Who `event` is offered to, first to last: the target and then each
    /// ancestor for an event that bubbles, the one element for enter and exit.
    pub fn chain(&self, event: &PointerEvent) -> &[Entity] {
        &self.chains[event.chain.start as usize..event.chain.end as usize]
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn clear(&mut self) {
        self.events.clear();
        self.chains.clear();
    }
}

/// Offer each event to its chain in turn, stopping at the first receiver
/// `send` says handled it — uGUI's bubbling, where an element with no handler
/// passes the event up to its parent.
pub fn deliver(events: &UiEvents, mut send: impl FnMut(Entity, &PointerEvent) -> bool) {
    for event in &events.events {
        for &entity in events.chain(event) {
            if send(entity, event) {
                break;
            }
        }
    }
}

/// The pointer as the UI last saw it: a world resource.
#[derive(Default)]
pub struct UiPointer {
    /// The topmost raycast target under the pointer and every element above
    /// it, deepest first: everything the pointer is inside.
    inside: Vec<Entity>,
    /// Last tick's `inside`, kept for its allocation.
    previous: Vec<Entity>,
    /// The element the left button went down on, while it is held.
    pressed: Option<Entity>,
    press_position: [f32; 2],
    /// Whether a raycast target is under the pointer at all.
    over: bool,
}

impl UiPointer {
    /// Whether the pointer is over a retained element that stops it.
    pub fn is_over_ui(&self) -> bool {
        self.over
    }

    pub fn is_inside(&self, entity: Entity) -> bool {
        self.inside.contains(&entity)
    }

    pub fn pressed(&self) -> Option<Entity> {
        self.pressed
    }

    /// Forget everything: edit mode has no pointer state.
    pub fn reset(&mut self) {
        self.inside.clear();
        self.pressed = None;
        self.over = false;
    }
}

/// The left button and where the pointer is, this tick.
#[derive(Clone, Copy, Debug, Default)]
pub struct PointerInput {
    pub position: [f32; 2],
    pub pressed: bool,
    pub released: bool,
}

/// Hit-test this tick's pointer against what the last frame drew and queue
/// what changed. Run before the script tick, in a ticking game only.
pub fn update(world: &mut World) {
    let input = world
        .get_resource::<InputState>()
        .map_or_else(PointerInput::default, |input| PointerInput {
            position: input.cursor().into(),
            pressed: input.mouse_button_pressed(0),
            released: input.mouse_button_released(0),
        });
    if world.get_resource::<UiPointer>().is_none() {
        world.insert_resource(UiPointer::default());
    }
    if world.get_resource::<UiEvents>().is_none() {
        world.insert_resource(UiEvents::default());
    }
    let tree = world.get_resource::<UiTree>();
    let mut pointer = world.resource_mut::<UiPointer>();
    let mut events = world.resource_mut::<UiEvents>();
    events.clear();
    if let Some(tree) = tree {
        step(world, &tree.layout, &mut pointer, input, &mut events);
    }
}

/// [`update`], given what it reads.
pub fn step(
    world: &World,
    layout: &Layout,
    pointer: &mut UiPointer,
    input: PointerInput,
    out: &mut UiEvents,
) {
    let position = input.position;
    let hit = layout.hit(position, true);
    pointer.over = hit.is_some();
    // A disabled button still stops the pointer, but nothing at or under it
    // takes part.
    let live = hit.filter(|&index| !blocked(world, layout, index));

    std::mem::swap(&mut pointer.inside, &mut pointer.previous);
    let UiPointer {
        inside, previous, ..
    } = pointer;
    inside.clear();
    let mut cursor = live.map(|index| index as u32);
    while let Some(index) = cursor {
        let element = &layout.elements()[index as usize];
        inside.push(element.entity);
        cursor = element.parent;
    }
    let single = |out: &mut UiEvents, kind, entity| {
        let start = out.chains.len() as u32;
        out.chains.push(entity);
        out.events.push(PointerEvent {
            kind,
            target: entity,
            position,
            press_position: position,
            chain: start..start + 1,
        });
    };
    for &left in previous.iter().filter(|e| !inside.contains(e)) {
        if world.is_alive(left) {
            single(out, PointerKind::Exit, left);
        }
    }
    for &entered in inside.iter().filter(|e| !previous.contains(e)) {
        single(out, PointerKind::Enter, entered);
    }

    let target = live.map(|index| layout.elements()[index].entity);
    if input.pressed
        && let Some(target) = target
    {
        pointer.pressed = Some(target);
        pointer.press_position = position;
        bubbling(world, out, PointerKind::Down, target, position, position);
    }
    if input.released
        && let Some(pressed) = pointer.pressed.take()
        && world.is_alive(pressed)
    {
        let from = pointer.press_position;
        bubbling(world, out, PointerKind::Up, pressed, position, from);
        if target == Some(pressed) {
            bubbling(world, out, PointerKind::Click, pressed, position, from);
        }
    }
}

/// Queue an event for `target` and then each of its ancestors in turn.
fn bubbling(
    world: &World,
    out: &mut UiEvents,
    kind: PointerKind,
    target: Entity,
    position: [f32; 2],
    press_position: [f32; 2],
) {
    let start = out.chains.len() as u32;
    let mut cursor = Some(target);
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        let Some(entity) = cursor else {
            break;
        };
        out.chains.push(entity);
        cursor = world.get::<Parent>(entity).map(|parent| parent.get());
    }
    out.events.push(PointerEvent {
        kind,
        target,
        position,
        press_position,
        chain: start..out.chains.len() as u32,
    });
}

/// Whether element `index` is, or is under, a button that is not interactable.
pub fn blocked(world: &World, layout: &Layout, index: usize) -> bool {
    let mut cursor = Some(index as u32);
    while let Some(at) = cursor {
        let element = &layout.elements()[at as usize];
        if world
            .get::<Button>(element.entity)
            .is_some_and(|button| !button.interactable)
        {
            return true;
        }
        cursor = element.parent;
    }
    false
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::*;
    use crate::scene::{Hierarchy, ensure_current, reparent};
    use crate::ui::{Canvas, Image, RectTransform, layout};

    struct Ui {
        world: World,
        canvas: Entity,
        layout: Layout,
        pointer: UiPointer,
        events: UiEvents,
    }

    fn at(x: f32, y: f32, width: f32, height: f32) -> RectTransform {
        RectTransform {
            anchor_min: Vec2::ZERO,
            anchor_max: Vec2::ZERO,
            pivot: Vec2::ZERO,
            anchored_position: Vec2::new(x, y),
            size_delta: Vec2::new(width, height),
            scale: Vec2::ONE,
        }
    }

    impl Ui {
        fn new() -> Self {
            let mut world = World::new();
            let canvas = world.spawn();
            world.insert(canvas, Canvas::default());
            Self {
                world,
                canvas,
                layout: Layout::default(),
                pointer: UiPointer::default(),
                events: UiEvents::default(),
            }
        }

        /// An element with an image that stops the pointer.
        fn target(&mut self, parent: Option<Entity>, rect: RectTransform) -> Entity {
            let entity = self.world.spawn();
            self.world.insert(entity, rect);
            self.world.insert(entity, Image::default());
            reparent(
                &mut self.world,
                entity,
                Some(parent.unwrap_or(self.canvas)),
                false,
            )
            .unwrap();
            entity
        }

        fn frame(&mut self, input: PointerInput) -> Vec<(PointerKind, Entity, Vec<Entity>)> {
            ensure_current(&mut self.world);
            layout::resolve(
                &self.world,
                &self.world.resource::<Hierarchy>(),
                [1000.0, 1000.0],
                &mut self.layout,
            );
            self.events.clear();
            step(
                &self.world,
                &self.layout,
                &mut self.pointer,
                input,
                &mut self.events,
            );
            self.events
                .events()
                .iter()
                .map(|e| (e.kind, e.target, self.events.chain(e).to_vec()))
                .collect()
        }

        fn hover(&mut self, x: f32, y: f32) -> Vec<(PointerKind, Entity, Vec<Entity>)> {
            self.frame(PointerInput {
                position: [x, y],
                ..PointerInput::default()
            })
        }

        fn press(&mut self, x: f32, y: f32) -> Vec<(PointerKind, Entity, Vec<Entity>)> {
            self.frame(PointerInput {
                position: [x, y],
                pressed: true,
                released: false,
            })
        }

        fn release(&mut self, x: f32, y: f32) -> Vec<(PointerKind, Entity, Vec<Entity>)> {
            self.frame(PointerInput {
                position: [x, y],
                pressed: false,
                released: true,
            })
        }
    }

    fn kinds(events: &[(PointerKind, Entity, Vec<Entity>)]) -> Vec<(PointerKind, Entity)> {
        events
            .iter()
            .map(|(kind, target, _)| (*kind, *target))
            .collect()
    }

    #[test]
    fn entering_and_leaving_tells_each_element_once_deepest_first() {
        let mut ui = Ui::new();
        let parent = ui.target(None, at(0.0, 0.0, 400.0, 400.0));
        let child = ui.target(Some(parent), at(0.0, 0.0, 100.0, 100.0));
        use PointerKind::*;

        assert_eq!(
            kinds(&ui.hover(50.0, 50.0)),
            [(Enter, child), (Enter, parent)]
        );
        assert!(ui.hover(51.0, 50.0).is_empty(), "nothing changed");
        assert_eq!(kinds(&ui.hover(300.0, 300.0)), [(Exit, child)]);
        assert_eq!(kinds(&ui.hover(900.0, 900.0)), [(Exit, parent)]);
        let enter = ui.hover(50.0, 50.0);
        assert!(
            enter.iter().all(|(_, target, chain)| chain == &[*target]),
            "enter and exit do not bubble: {enter:?}"
        );
    }

    #[test]
    fn a_click_needs_the_press_and_the_release_on_the_same_element() {
        let mut ui = Ui::new();
        let a = ui.target(None, at(0.0, 0.0, 100.0, 100.0));
        let b = ui.target(None, at(200.0, 0.0, 100.0, 100.0));
        use PointerKind::*;

        ui.hover(50.0, 50.0);
        assert_eq!(kinds(&ui.press(50.0, 50.0)), [(Down, a)]);
        assert_eq!(ui.pointer.pressed(), Some(a));
        assert_eq!(kinds(&ui.release(50.0, 50.0)), [(Up, a), (Click, a)]);
        assert_eq!(ui.pointer.pressed(), None);

        ui.press(50.0, 50.0);
        let dragged_off = kinds(&ui.release(250.0, 50.0));
        assert_eq!(
            dragged_off,
            [(Exit, a), (Enter, b), (Up, a)],
            "the release goes to what was pressed, and is no click"
        );

        // Pressed and let go within one tick is still a click.
        let quick = kinds(&ui.frame(PointerInput {
            position: [250.0, 50.0],
            pressed: true,
            released: true,
        }));
        assert_eq!(quick, [(Down, b), (Up, b), (Click, b)]);
    }

    #[test]
    fn the_topmost_target_takes_the_press_and_a_non_target_lets_it_through() {
        let mut ui = Ui::new();
        let back = ui.target(None, at(0.0, 0.0, 500.0, 500.0));
        let front = ui.target(None, at(0.0, 0.0, 100.0, 100.0));
        let glass = ui.target(None, at(0.0, 0.0, 50.0, 50.0));
        ui.world.get_mut::<Image>(glass).unwrap().raycast_target = false;

        let down = |events: Vec<(PointerKind, Entity, Vec<Entity>)>| {
            events
                .into_iter()
                .find(|(kind, ..)| *kind == PointerKind::Down)
                .map(|(_, target, _)| target)
        };
        assert_eq!(down(ui.press(25.0, 25.0)), Some(front));
        ui.release(25.0, 25.0);
        assert_eq!(down(ui.press(300.0, 300.0)), Some(back));
        assert!(ui.pointer.is_over_ui());
        ui.release(300.0, 300.0);
        ui.hover(900.0, 900.0);
        assert!(!ui.pointer.is_over_ui());
    }

    #[test]
    fn a_press_bubbles_from_its_target_up_the_hierarchy() {
        let mut ui = Ui::new();
        let panel = ui.target(None, at(0.0, 0.0, 400.0, 400.0));
        let button = ui.target(Some(panel), at(0.0, 0.0, 100.0, 100.0));
        let events = ui.press(10.0, 10.0);
        let (_, _, chain) = events
            .iter()
            .find(|(kind, ..)| *kind == PointerKind::Down)
            .unwrap();
        assert_eq!(chain, &[button, panel, ui.canvas]);
    }

    #[test]
    fn delivery_stops_at_the_first_receiver_that_handles_it() {
        let mut ui = Ui::new();
        let panel = ui.target(None, at(0.0, 0.0, 400.0, 400.0));
        let button = ui.target(Some(panel), at(0.0, 0.0, 100.0, 100.0));
        ui.press(10.0, 10.0);
        let mut offered = Vec::new();
        deliver(&ui.events, |entity, event| {
            offered.push((event.kind, entity));
            entity == panel
        });
        use PointerKind::*;
        assert_eq!(
            offered,
            [
                (Enter, button),
                (Enter, panel),
                (Down, button),
                (Down, panel)
            ],
            "the canvas is never offered the press the panel took"
        );
    }

    #[test]
    fn a_disabled_button_and_everything_under_it_get_nothing_but_still_stop_the_pointer() {
        let mut ui = Ui::new();
        let back = ui.target(None, at(0.0, 0.0, 500.0, 500.0));
        let button = ui.target(None, at(0.0, 0.0, 100.0, 100.0));
        ui.world.insert(
            button,
            Button {
                interactable: false,
                ..Button::default()
            },
        );
        let icon = ui.target(Some(button), at(0.0, 0.0, 20.0, 20.0));

        for (x, y) in [(50.0, 50.0), (10.0, 10.0)] {
            assert!(ui.hover(x, y).is_empty());
            assert!(ui.press(x, y).is_empty(), "nothing reaches {back:?} either");
            assert!(ui.release(x, y).is_empty());
            assert!(ui.pointer.is_over_ui());
        }
        assert!(!ui.pointer.is_inside(icon));

        ui.world.get_mut::<Button>(button).unwrap().interactable = true;
        assert_eq!(
            kinds(&ui.press(50.0, 50.0)),
            [(PointerKind::Enter, button), (PointerKind::Down, button)]
        );
    }

    #[test]
    fn a_released_press_on_a_despawned_element_is_forgotten() {
        let mut ui = Ui::new();
        let a = ui.target(None, at(0.0, 0.0, 100.0, 100.0));
        ui.press(50.0, 50.0);
        ui.world.despawn(a);
        let events = kinds(&ui.release(50.0, 50.0));
        assert!(
            events.iter().all(|(kind, _)| *kind == PointerKind::Exit),
            "{events:?}"
        );
        assert_eq!(ui.pointer.pressed(), None);
    }
}
