//! Tweens: a UI element's field eased from one value to another over time —
//! Roblox's `TweenService` and Godot's `Tween`, run by the engine rather than
//! by game code.
//!
//! # Engine-side, against named fields
//!
//! A tween is data in [`UiTweens`], and [`advance`] writes every live one into
//! its element once a frame. Game code starts one with a call and is told when
//! it finishes; it does not run it. That keeps the per-frame cost a loop over a
//! `Vec` with no managed call in it, and lets the storybook play a story's
//! tweens with no script ticking at all.
//!
//! What a tween can drive is a [`TweenProperty`]: a closed list of the retained
//! UI's numeric fields, each up to four floats, rather than a path into the
//! registry. A path would reach every reflected field, at the price of a string
//! lookup and a boxed value per tween per frame; the list costs a `match`, and
//! a field joins it by adding a variant.
//!
//! # Timing
//!
//! A tween waits `delay` seconds once, then runs `duration` seconds forward —
//! and, if it `reverses`, the same back — then repeats that cycle `repeat` more
//! times, or forever for a negative count. Time past the end of a cycle carries
//! into the next, so a looping tween does not drift against the clock. The last
//! frame writes the exact end value rather than an eased approximation of it.
//!
//! The start value is read from the element when the delay ends, unless the
//! tween names one, which is written on the first frame instead — so a fade-in
//! queued behind a delay is invisible while it waits rather than flashing at
//! full opacity first.
//!
//! # One tween per field
//!
//! Starting a tween on a field that already has one replaces it, mid-flight,
//! from wherever it got to: the rule Roblox has, and the one that makes a hover
//! animation safe to start on every enter and exit. A replaced or cancelled
//! tween never reports completion; neither does one whose element was despawned
//! or lost the component it was driving, which simply stops.
//!
//! Colours are eased in the numbers they are authored in — sRGB-encoded,
//! straight alpha — which is what a colour picker shows halfway between two
//! swatches and what a CSS transition does.

use glam::Vec2;
use orrin_ecs::{Entity, World};

use super::{Corner, Gradient, Image, RectTransform, ScrollFrame, Shadow, Stroke, Text};

/// The shape of an ease's curve. Lock-step with C# `Orrin.EaseStyle`: append,
/// never renumber.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EaseStyle {
    #[default]
    Linear,
    Sine,
    Quad,
    Cubic,
    Quart,
    Quint,
    Expo,
    Circ,
    /// Pulls back past the start before it moves.
    Back,
    /// Overshoots and rings like a spring.
    Elastic,
    Bounce,
}

impl EaseStyle {
    pub const ALL: [Self; 11] = [
        Self::Linear,
        Self::Sine,
        Self::Quad,
        Self::Cubic,
        Self::Quart,
        Self::Quint,
        Self::Expo,
        Self::Circ,
        Self::Back,
        Self::Elastic,
        Self::Bounce,
    ];

    pub fn from_u32(value: u32) -> Option<Self> {
        Self::ALL.get(value as usize).copied()
    }
}

/// Which end of the curve the style shapes. Lock-step with C#
/// `Orrin.EaseDirection`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EaseDirection {
    In,
    #[default]
    Out,
    InOut,
}

impl EaseDirection {
    pub const ALL: [Self; 3] = [Self::In, Self::Out, Self::InOut];

    pub fn from_u32(value: u32) -> Option<Self> {
        Self::ALL.get(value as usize).copied()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ease {
    pub style: EaseStyle,
    pub direction: EaseDirection,
}

impl Ease {
    pub const LINEAR: Self = Self::new(EaseStyle::Linear, EaseDirection::In);

    pub const fn new(style: EaseStyle, direction: EaseDirection) -> Self {
        Self { style, direction }
    }

    /// Progress `t` in 0..=1 to eased progress, 0 at 0 and 1 at 1.
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        let ease_in = |t| ease_in(self.style, t);
        match self.direction {
            EaseDirection::In => ease_in(t),
            EaseDirection::Out => 1.0 - ease_in(1.0 - t),
            EaseDirection::InOut if t < 0.5 => ease_in(2.0 * t) / 2.0,
            EaseDirection::InOut => 1.0 - ease_in(2.0 - 2.0 * t) / 2.0,
        }
    }
}

/// Every style as its In curve; Out and InOut are derived from it, which is
/// how Penner's equations relate and what keeps the three from drifting.
fn ease_in(style: EaseStyle, t: f32) -> f32 {
    use std::f32::consts::{FRAC_PI_2, TAU};
    const BACK: f32 = 1.70158;
    match style {
        EaseStyle::Linear => t,
        EaseStyle::Sine => 1.0 - (t * FRAC_PI_2).cos(),
        EaseStyle::Quad => t * t,
        EaseStyle::Cubic => t * t * t,
        EaseStyle::Quart => t * t * t * t,
        EaseStyle::Quint => t * t * t * t * t,
        // The exponential never reaches zero on its own; 2^-10 is a visible
        // step on a long slide.
        EaseStyle::Expo if t <= 0.0 => 0.0,
        EaseStyle::Expo => (10.0 * t - 10.0).exp2(),
        EaseStyle::Circ => 1.0 - (1.0 - t * t).max(0.0).sqrt(),
        EaseStyle::Back => (BACK + 1.0) * t * t * t - BACK * t * t,
        EaseStyle::Elastic if t <= 0.0 || t >= 1.0 => t.clamp(0.0, 1.0),
        EaseStyle::Elastic => -(10.0 * t - 10.0).exp2() * ((10.0 * t - 10.75) * (TAU / 3.0)).sin(),
        EaseStyle::Bounce => 1.0 - bounce_out(1.0 - t),
    }
}

fn bounce_out(t: f32) -> f32 {
    const N: f32 = 7.5625;
    const D: f32 = 2.75;
    if t < 1.0 / D {
        N * t * t
    } else if t < 2.0 / D {
        let t = t - 1.5 / D;
        N * t * t + 0.75
    } else if t < 2.5 / D {
        let t = t - 2.25 / D;
        N * t * t + 0.9375
    } else {
        let t = t - 2.625 / D;
        N * t * t + 0.984375
    }
}

/// A field a tween can drive. Lock-step with C# `Orrin.TweenProperty`: append,
/// never renumber.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TweenProperty {
    /// `RectTransform::anchored_position`.
    Position,
    /// `RectTransform::size_delta`.
    Size,
    /// `RectTransform::scale`.
    Scale,
    ImageColor,
    TextColor,
    TextSize,
    CornerRadius,
    StrokeColor,
    StrokeThickness,
    ShadowColor,
    ShadowOffset,
    ShadowBlur,
    GradientStart,
    GradientEnd,
    GradientRotation,
    ScrollPosition,
}

impl TweenProperty {
    pub const ALL: [Self; 16] = [
        Self::Position,
        Self::Size,
        Self::Scale,
        Self::ImageColor,
        Self::TextColor,
        Self::TextSize,
        Self::CornerRadius,
        Self::StrokeColor,
        Self::StrokeThickness,
        Self::ShadowColor,
        Self::ShadowOffset,
        Self::ShadowBlur,
        Self::GradientStart,
        Self::GradientEnd,
        Self::GradientRotation,
        Self::ScrollPosition,
    ];

    pub fn from_u32(value: u32) -> Option<Self> {
        Self::ALL.get(value as usize).copied()
    }

    /// The field's value on `entity`, in its first one, two or four lanes.
    pub fn read(self, world: &World, entity: Entity) -> Option<[f32; 4]> {
        let two = |v: Vec2| [v.x, v.y, 0.0, 0.0];
        let one = |v: f32| [v, 0.0, 0.0, 0.0];
        Some(match self {
            Self::Position => two(world.get::<RectTransform>(entity)?.anchored_position),
            Self::Size => two(world.get::<RectTransform>(entity)?.size_delta),
            Self::Scale => two(world.get::<RectTransform>(entity)?.scale),
            Self::ImageColor => world.get::<Image>(entity)?.color,
            Self::TextColor => world.get::<Text>(entity)?.color,
            Self::TextSize => one(world.get::<Text>(entity)?.size),
            Self::CornerRadius => one(world.get::<Corner>(entity)?.radius),
            Self::StrokeColor => world.get::<Stroke>(entity)?.color,
            Self::StrokeThickness => one(world.get::<Stroke>(entity)?.thickness),
            Self::ShadowColor => world.get::<Shadow>(entity)?.color,
            Self::ShadowOffset => two(world.get::<Shadow>(entity)?.offset),
            Self::ShadowBlur => one(world.get::<Shadow>(entity)?.blur),
            Self::GradientStart => world.get::<Gradient>(entity)?.start,
            Self::GradientEnd => world.get::<Gradient>(entity)?.end,
            Self::GradientRotation => one(world.get::<Gradient>(entity)?.rotation),
            Self::ScrollPosition => two(world.get::<ScrollFrame>(entity)?.scroll_position),
        })
    }

    /// Write `value`'s lanes into the field. False when `entity` has no
    /// component carrying it.
    pub fn write(self, world: &World, entity: Entity, value: [f32; 4]) -> bool {
        fn set<C: 'static>(world: &World, entity: Entity, field: impl FnOnce(&mut C)) -> bool {
            world
                .get_mut::<C>(entity)
                .map(|mut c| field(&mut c))
                .is_some()
        }
        let [x, y, ..] = value;
        let two = Vec2::new(x, y);
        match self {
            Self::Position => set(world, entity, |c: &mut RectTransform| {
                c.anchored_position = two
            }),
            Self::Size => set(world, entity, |c: &mut RectTransform| c.size_delta = two),
            Self::Scale => set(world, entity, |c: &mut RectTransform| c.scale = two),
            Self::ImageColor => set(world, entity, |c: &mut Image| c.color = value),
            Self::TextColor => set(world, entity, |c: &mut Text| c.color = value),
            Self::TextSize => set(world, entity, |c: &mut Text| c.size = x),
            Self::CornerRadius => set(world, entity, |c: &mut Corner| c.radius = x),
            Self::StrokeColor => set(world, entity, |c: &mut Stroke| c.color = value),
            Self::StrokeThickness => set(world, entity, |c: &mut Stroke| c.thickness = x),
            Self::ShadowColor => set(world, entity, |c: &mut Shadow| c.color = value),
            Self::ShadowOffset => set(world, entity, |c: &mut Shadow| c.offset = two),
            Self::ShadowBlur => set(world, entity, |c: &mut Shadow| c.blur = x),
            Self::GradientStart => set(world, entity, |c: &mut Gradient| c.start = value),
            Self::GradientEnd => set(world, entity, |c: &mut Gradient| c.end = value),
            Self::GradientRotation => set(world, entity, |c: &mut Gradient| c.rotation = x),
            Self::ScrollPosition => {
                set(world, entity, |c: &mut ScrollFrame| c.scroll_position = two)
            }
        }
    }
}

/// A tween as it is asked for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tween {
    pub target: Entity,
    pub property: TweenProperty,
    pub to: [f32; 4],
    /// `None` starts from whatever the field holds when the delay ends.
    pub from: Option<[f32; 4]>,
    pub duration: f32,
    pub delay: f32,
    pub ease: Ease,
    /// Further cycles after the first; negative repeats forever.
    pub repeat: i32,
    /// Each cycle runs back to the start after reaching the end.
    pub reverses: bool,
}

impl Tween {
    pub fn new(target: Entity, property: TweenProperty, to: [f32; 4], duration: f32) -> Self {
        Self {
            target,
            property,
            to,
            from: None,
            duration,
            delay: 0.0,
            ease: Ease::default(),
            repeat: 0,
            reverses: false,
        }
    }
}

/// A started tween. Never zero as bits, and never reused: a slot's generation
/// moves on every time it frees.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TweenId {
    index: u32,
    generation: u32,
}

impl TweenId {
    pub fn to_bits(self) -> u64 {
        (u64::from(self.generation) << 32) | u64::from(self.index)
    }

    pub fn from_bits(bits: u64) -> Self {
        Self {
            index: bits as u32,
            generation: (bits >> 32) as u32,
        }
    }
}

/// Every live tween: a world resource, advanced by [`advance`].
///
/// A slot array with a free list, so a frame that starts and finishes tweens
/// allocates nothing once the array has grown to the most that were ever live
/// at once.
#[derive(Default)]
pub struct UiTweens {
    slots: Vec<Slot>,
    free: Vec<u32>,
    playing: usize,
    completed: Vec<TweenId>,
}

struct Slot {
    generation: u32,
    running: Option<Running>,
}

struct Running {
    tween: Tween,
    /// The start value, once the delay has ended.
    from: Option<[f32; 4]>,
    elapsed: f32,
}

impl UiTweens {
    /// Start `tween`, replacing any live one on the same field of the same
    /// element.
    pub fn play(&mut self, tween: Tween) -> TweenId {
        if let Some(index) = self.slots.iter().position(|slot| {
            slot.running.as_ref().is_some_and(|running| {
                running.tween.target == tween.target && running.tween.property == tween.property
            })
        }) {
            self.release(index);
        }
        let index = self.free.pop().unwrap_or_else(|| {
            self.slots.push(Slot {
                generation: 1,
                running: None,
            });
            self.slots.len() as u32 - 1
        });
        let slot = &mut self.slots[index as usize];
        slot.running = Some(Running {
            tween,
            from: None,
            elapsed: 0.0,
        });
        self.playing += 1;
        TweenId {
            index,
            generation: slot.generation,
        }
    }

    /// Stop `id` where it is. False when it was not playing.
    pub fn cancel(&mut self, id: TweenId) -> bool {
        if !self.is_playing(id) {
            return false;
        }
        self.release(id.index as usize);
        true
    }

    pub fn is_playing(&self, id: TweenId) -> bool {
        self.slots
            .get(id.index as usize)
            .is_some_and(|slot| slot.generation == id.generation && slot.running.is_some())
    }

    /// Whether nothing is playing.
    pub fn is_empty(&self) -> bool {
        self.playing == 0
    }

    /// Stop everything, reporting nothing.
    pub fn clear(&mut self) {
        for index in 0..self.slots.len() {
            if self.slots[index].running.is_some() {
                self.release(index);
            }
        }
        self.completed.clear();
    }

    /// The tweens that finished since the last call, in the order they did.
    pub fn take_completed(&mut self, into: &mut Vec<TweenId>) {
        into.append(&mut self.completed);
    }

    fn release(&mut self, index: usize) -> TweenId {
        let slot = &mut self.slots[index];
        let id = TweenId {
            index: index as u32,
            generation: slot.generation,
        };
        slot.running = None;
        slot.generation = slot.generation.checked_add(1).unwrap_or(1);
        self.free.push(index as u32);
        self.playing -= 1;
        id
    }
}

/// Where a running tween stands after `delta` more seconds.
enum Step {
    Wait,
    Write([f32; 4]),
    Finish([f32; 4]),
    Stop,
}

/// Move every live tween on by `delta` seconds and write it into its element.
pub fn advance(world: &World, delta: f32) {
    let Some(mut tweens) = world.get_resource_mut::<UiTweens>() else {
        return;
    };
    if tweens.is_empty() {
        return;
    }
    for index in 0..tweens.slots.len() {
        let Some(running) = &mut tweens.slots[index].running else {
            continue;
        };
        let step = running.step(world, delta);
        let target = (running.tween.target, running.tween.property);
        let written = match step {
            Step::Wait => true,
            Step::Write(value) | Step::Finish(value) => target.1.write(world, target.0, value),
            Step::Stop => false,
        };
        if !written {
            tweens.release(index);
        } else if let Step::Finish(_) = step {
            let id = tweens.release(index);
            tweens.completed.push(id);
        }
    }
}

impl Running {
    fn step(&mut self, world: &World, delta: f32) -> Step {
        let Tween {
            target,
            property,
            to,
            duration,
            delay,
            ease,
            repeat,
            reverses,
            ..
        } = self.tween;
        let first = self.elapsed == 0.0;
        self.elapsed += delta.max(0.0);
        let from = match self.from {
            Some(from) => from,
            None if self.elapsed < delay => {
                return match self.tween.from {
                    Some(from) if first => Step::Write(from),
                    _ => Step::Wait,
                };
            }
            None => match self.tween.from.or_else(|| property.read(world, target)) {
                Some(from) => *self.from.insert(from),
                None => return Step::Stop,
            },
        };
        let end = if reverses { from } else { to };
        if duration <= 0.0 {
            return Step::Finish(end);
        }
        let legs_per_cycle = if reverses { 2.0 } else { 1.0 };
        let mut active = self.elapsed - delay;
        if repeat < 0 {
            // Wrapped, so a tween left looping for an hour keeps its
            // precision.
            active %= duration * legs_per_cycle;
            self.elapsed = delay + active;
        } else if active >= duration * legs_per_cycle * (repeat as f32 + 1.0) {
            return Step::Finish(end);
        }
        let legs = active / duration;
        let leg = legs.floor();
        let progress = legs - leg;
        let backwards = reverses && (leg as u64) % 2 == 1;
        let eased = ease.apply(if backwards { 1.0 - progress } else { progress });
        let mut value = [0.0; 4];
        for lane in 0..4 {
            value[lane] = from[lane] + (to[lane] - from[lane]) * eased;
        }
        Step::Write(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn element(world: &mut World) -> Entity {
        let entity = world.spawn();
        world.insert(
            entity,
            RectTransform {
                anchored_position: Vec2::ZERO,
                ..RectTransform::default()
            },
        );
        world.insert(entity, Image::default());
        entity
    }

    fn world() -> World {
        let mut world = World::new();
        world.insert_resource(UiTweens::default());
        world
    }

    fn position(world: &World, entity: Entity) -> Vec2 {
        world
            .get::<RectTransform>(entity)
            .unwrap()
            .anchored_position
    }

    fn play(world: &World, tween: Tween) -> TweenId {
        world.resource_mut::<UiTweens>().play(tween)
    }

    fn playing(world: &World, id: TweenId) -> bool {
        world.resource::<UiTweens>().is_playing(id)
    }

    fn completed(world: &World) -> Vec<TweenId> {
        let mut out = Vec::new();
        world.resource_mut::<UiTweens>().take_completed(&mut out);
        out
    }

    fn slide(target: Entity, to: f32, duration: f32) -> Tween {
        Tween {
            ease: Ease::LINEAR,
            ..Tween::new(
                target,
                TweenProperty::Position,
                [to, 0.0, 0.0, 0.0],
                duration,
            )
        }
    }

    #[test]
    fn every_ease_starts_at_zero_and_ends_at_one() {
        for style in EaseStyle::ALL {
            for direction in EaseDirection::ALL {
                let ease = Ease::new(style, direction);
                assert!(
                    ease.apply(0.0).abs() < EPS,
                    "{ease:?} at 0 is {}",
                    ease.apply(0.0)
                );
                assert!(
                    (ease.apply(1.0) - 1.0).abs() < EPS,
                    "{ease:?} at 1 is {}",
                    ease.apply(1.0)
                );
            }
        }
    }

    /// Out is In played backwards and upside down, and InOut is symmetric about
    /// its middle — the identities every Penner curve satisfies.
    #[test]
    fn out_mirrors_in_and_in_out_is_symmetric() {
        for style in EaseStyle::ALL {
            let ease_in = Ease::new(style, EaseDirection::In);
            let ease_out = Ease::new(style, EaseDirection::Out);
            let in_out = Ease::new(style, EaseDirection::InOut);
            assert!(
                (in_out.apply(0.5) - 0.5).abs() < EPS,
                "{style:?} InOut middle"
            );
            for i in 0..=20 {
                let t = i as f32 / 20.0;
                let mirrored = 1.0 - ease_in.apply(1.0 - t);
                assert!(
                    (ease_out.apply(t) - mirrored).abs() < 1e-4,
                    "{style:?} Out at {t}"
                );
                let reflected = 1.0 - in_out.apply(1.0 - t);
                assert!(
                    (in_out.apply(t) - reflected).abs() < 1e-4,
                    "{style:?} InOut at {t}"
                );
            }
        }
    }

    #[test]
    fn the_curves_have_their_textbook_values() {
        let at = |style, direction, t| Ease::new(style, direction).apply(t);
        use EaseDirection::*;
        assert!((at(EaseStyle::Quad, In, 0.5) - 0.25).abs() < EPS);
        assert!((at(EaseStyle::Cubic, Out, 0.5) - 0.875).abs() < EPS);
        assert!((at(EaseStyle::Quint, In, 0.5) - 0.03125).abs() < EPS);
        assert!((at(EaseStyle::Sine, Out, 0.5) - std::f32::consts::FRAC_1_SQRT_2).abs() < EPS);
        assert!((at(EaseStyle::Linear, InOut, 0.3) - 0.3).abs() < EPS);
        assert!(at(EaseStyle::Back, In, 0.3) < 0.0, "back pulls back first");
        assert!(
            (0..100).any(|i| at(EaseStyle::Elastic, Out, i as f32 / 100.0) > 1.0),
            "elastic overshoots"
        );
        let bounce = at(EaseStyle::Bounce, Out, 0.5);
        assert!(
            (bounce - 0.765625).abs() < EPS,
            "bounce out at 0.5 is {bounce}"
        );
    }

    #[test]
    fn a_tween_eases_from_the_current_value_to_its_target_over_its_duration() {
        let mut world = world();
        let button = element(&mut world);
        world
            .get_mut::<RectTransform>(button)
            .unwrap()
            .anchored_position = Vec2::new(10.0, 5.0);
        let id = play(&world, slide(button, 110.0, 1.0));

        advance(&world, 0.25);
        assert!((position(&world, button).x - 35.0).abs() < 1e-3);
        assert!(
            (position(&world, button).y - 3.75).abs() < 1e-3,
            "every lane of the field moves"
        );
        assert!(playing(&world, id));
        advance(&world, 0.25);
        assert!((position(&world, button).x - 60.0).abs() < 1e-3);
        assert!(completed(&world).is_empty());
    }

    #[test]
    fn a_finished_tween_lands_exactly_on_its_target_and_reports_once() {
        let mut world = world();
        let button = element(&mut world);
        let id = play(
            &world,
            Tween {
                ease: Ease::new(EaseStyle::Elastic, EaseDirection::Out),
                ..slide(button, 100.0, 0.3)
            },
        );
        for _ in 0..7 {
            advance(&world, 0.1);
        }
        assert_eq!(position(&world, button).x, 100.0);
        assert!(!playing(&world, id));
        assert_eq!(completed(&world), vec![id]);
        advance(&world, 0.1);
        assert!(completed(&world).is_empty());
    }

    #[test]
    fn a_zero_length_tween_completes_on_its_first_frame() {
        let mut world = world();
        let button = element(&mut world);
        let id = play(&world, slide(button, 42.0, 0.0));
        advance(&world, 0.016);
        assert_eq!(position(&world, button).x, 42.0);
        assert_eq!(completed(&world), vec![id]);
    }

    #[test]
    fn a_second_tween_on_the_same_field_takes_over_from_where_the_first_got() {
        let mut world = world();
        let button = element(&mut world);
        let first = play(&world, slide(button, 100.0, 1.0));
        advance(&world, 0.5);
        let second = play(&world, slide(button, 0.0, 1.0));
        assert!(!playing(&world, first));
        advance(&world, 0.5);
        assert!((position(&world, button).x - 25.0).abs() < 1e-3);
        advance(&world, 0.5);
        assert_eq!(
            completed(&world),
            vec![second],
            "a replaced tween reports nothing"
        );
    }

    #[test]
    fn tweens_on_different_fields_of_one_element_run_side_by_side() {
        let mut world = world();
        let button = element(&mut world);
        let slide = play(&world, slide(button, 100.0, 1.0));
        let fade = play(
            &world,
            Tween {
                ease: Ease::LINEAR,
                ..Tween::new(button, TweenProperty::ImageColor, [1.0, 0.0, 0.5, 0.0], 1.0)
            },
        );
        advance(&world, 0.5);
        assert!(playing(&world, slide) && playing(&world, fade));
        let color = world.get::<Image>(button).unwrap().color;
        let want = [1.0, 0.5, 0.75, 0.5];
        for (got, want) in color.iter().zip(want) {
            assert!((got - want).abs() < 1e-4, "{color:?}");
        }
        assert!((position(&world, button).x - 50.0).abs() < 1e-3);
    }

    #[test]
    fn a_delay_holds_the_tween_and_a_named_start_is_written_while_it_waits() {
        let mut world = world();
        let button = element(&mut world);
        play(
            &world,
            Tween {
                from: Some([-50.0, 0.0, 0.0, 0.0]),
                delay: 0.5,
                ..slide(button, 50.0, 1.0)
            },
        );
        advance(&world, 0.25);
        assert_eq!(position(&world, button).x, -50.0);
        advance(&world, 0.25);
        assert_eq!(position(&world, button).x, -50.0);
        advance(&world, 0.5);
        assert!((position(&world, button).x - 0.0).abs() < 1e-3);
    }

    #[test]
    fn a_delayed_tween_reads_its_start_when_the_delay_ends() {
        let mut world = world();
        let button = element(&mut world);
        play(
            &world,
            Tween {
                delay: 0.5,
                ..slide(button, 100.0, 1.0)
            },
        );
        advance(&world, 0.25);
        world
            .get_mut::<RectTransform>(button)
            .unwrap()
            .anchored_position
            .x = 50.0;
        advance(&world, 0.25);
        advance(&world, 0.5);
        assert!((position(&world, button).x - 75.0).abs() < 1e-3);
    }

    #[test]
    fn a_reversing_repeat_runs_there_and_back_and_carries_time_across_cycles() {
        let mut world = world();
        let button = element(&mut world);
        let id = play(
            &world,
            Tween {
                repeat: 1,
                reverses: true,
                ..slide(button, 100.0, 1.0)
            },
        );
        // One cycle is there and back, two seconds; 2.5 is a quarter of the
        // way into the second.
        advance(&world, 2.5);
        assert!((position(&world, button).x - 50.0).abs() < 1e-3);
        assert!(playing(&world, id));
        advance(&world, 2.0);
        assert_eq!(
            position(&world, button).x,
            0.0,
            "a reversing tween ends where it began"
        );
        assert_eq!(completed(&world), vec![id]);
    }

    #[test]
    fn a_negative_repeat_never_finishes() {
        let mut world = world();
        let button = element(&mut world);
        let id = play(
            &world,
            Tween {
                repeat: -1,
                ..slide(button, 100.0, 0.5)
            },
        );
        for _ in 0..1000 {
            advance(&world, 0.1);
        }
        assert!(playing(&world, id));
        assert!(completed(&world).is_empty());
        // Restarted from the start at each cycle: 100.05 s is 0.05 into one.
        advance(&world, 0.05);
        assert!((position(&world, button).x - 10.0).abs() < 0.5);
    }

    #[test]
    fn a_cancelled_tween_stops_where_it_is_and_reports_nothing() {
        let mut world = world();
        let button = element(&mut world);
        let id = play(&world, slide(button, 100.0, 1.0));
        advance(&world, 0.5);
        assert!(world.resource_mut::<UiTweens>().cancel(id));
        assert!(!world.resource_mut::<UiTweens>().cancel(id));
        advance(&world, 1.0);
        assert!((position(&world, button).x - 50.0).abs() < 1e-3);
        assert!(completed(&world).is_empty());
        assert!(world.resource::<UiTweens>().is_empty());
    }

    #[test]
    fn a_tween_whose_element_is_gone_stops_without_completing() {
        let mut world = world();
        let button = element(&mut world);
        let bare = world.spawn();
        let gone = play(&world, slide(button, 100.0, 1.0));
        let missing = play(
            &world,
            Tween::new(bare, TweenProperty::TextSize, [10.0; 4], 1.0),
        );
        world.despawn(button);
        advance(&world, 0.1);
        assert!(!playing(&world, gone) && !playing(&world, missing));
        assert!(completed(&world).is_empty());
        assert!(world.resource::<UiTweens>().is_empty());
    }

    #[test]
    fn an_id_is_never_reused_and_never_zero() {
        let mut world = world();
        let button = element(&mut world);
        let first = play(&world, slide(button, 1.0, 0.0));
        advance(&world, 0.1);
        let second = play(&world, slide(button, 2.0, 1.0));
        assert_ne!(first, second);
        assert!(!playing(&world, first));
        assert!(playing(&world, second));
        assert_ne!(first.to_bits(), 0);
        assert_eq!(TweenId::from_bits(second.to_bits()), second);
    }

    /// Every property reads back what it wrote, through the component it names.
    #[test]
    fn every_property_round_trips_through_its_component() {
        let mut world = world();
        let entity = world.spawn();
        world.insert(entity, RectTransform::default());
        world.insert(entity, Image::default());
        world.insert(entity, Text::default());
        world.insert(entity, Corner::default());
        world.insert(entity, Stroke::default());
        world.insert(entity, Shadow::default());
        world.insert(entity, Gradient::default());
        world.insert(
            entity,
            ScrollFrame {
                content_size: Vec2::splat(10_000.0),
                ..ScrollFrame::default()
            },
        );
        for property in TweenProperty::ALL {
            let value = [3.0, 4.0, 0.25, 0.5];
            assert!(property.write(&world, entity, value), "{property:?}");
            let back = property.read(&world, entity).unwrap();
            let lanes = match property {
                TweenProperty::ImageColor
                | TweenProperty::TextColor
                | TweenProperty::StrokeColor
                | TweenProperty::ShadowColor
                | TweenProperty::GradientStart
                | TweenProperty::GradientEnd => 4,
                TweenProperty::Position
                | TweenProperty::Size
                | TweenProperty::Scale
                | TweenProperty::ShadowOffset
                | TweenProperty::ScrollPosition => 2,
                _ => 1,
            };
            assert_eq!(back[..lanes], value[..lanes], "{property:?}");
        }
        assert_eq!(world.get::<Text>(entity).unwrap().size, 3.0);
        assert_eq!(
            world.get::<Shadow>(entity).unwrap().offset,
            Vec2::new(3.0, 4.0)
        );
        let bare = world.spawn();
        assert!(!TweenProperty::Position.write(&world, bare, [0.0; 4]));
        assert_eq!(TweenProperty::Position.read(&world, bare), None);
    }
}
