//! `game_input`: an agent playtests the running game through a virtual input
//! device, and `input_names`: what it may name.
//!
//! **Through the seam, not around it.** The device is a backend like the
//! window's and the pads': it presses keys, mouse buttons and pad sources on
//! [`InputState`]'s virtual layer, and nothing else. Bindings, action
//! resolution, scripts, physics and UI events read the union of both devices
//! and cannot tell an agent's press from the developer's; an action named in a
//! step is pressed by pressing a source the config binds it to, so the whole
//! pipeline runs and a remap does not break an agent's script. Nothing here
//! touches [`Actions`] but to read what a name is bound to.
//!
//! **Fed before the resolve, answered after the tick.** [`GameInput::feed`]
//! runs once a frame from `app.rs` before the actions resolve — where the
//! window's events have just landed — and applies this frame of the sequence.
//! The remote's serve, after the script tick, answers it once it has played or
//! ended. Releases are fed too, at the start of the next frame: let go after
//! the tick, an edge would be cleared by `end_frame` before any script saw it.
//!
//! **Nothing sticks.** A sequence's holds are released when it ends however it
//! ends: played out, the game stopped (paused-then-stopped included), its
//! connection closed, its deadline passed, the permission turned off under it,
//! or the remote stopped. Each of those drops the sequence, and the next feed
//! with none running lets go of everything the virtual device holds.
//!
//! **One at a time, and bounded.** One sequence per editor; another is refused
//! as busy. A sequence is at most [`MAX_STEPS`] steps and [`MAX_SECONDS`] of
//! declared length, frames counted at 60 a second, so it is answered well
//! inside the client's 30 s; a longer playtest is several calls. Past
//! `command::DEADLINE` of wall time it is answered with what played.
//!
//! **Only while the game runs.** In edit mode it is refused. A paused game is
//! driven a stepped frame at a time — each input frame asks for one, as the
//! Step button does — so a paused playtest is deterministic.
//!
//! **Focus.** The window's devices go quiet without focus and the virtual one
//! does not, so a playtest runs while the developer works in another window.
//! A cursor lock its click asks for is the game's to see and never the
//! window's to grab: see [`CursorLock`](crate::scene::CursorLock).
//!
//! # Steps
//!
//! `steps` is a list; each step is an object with one of:
//!
//! | key | value | does |
//! | --- | --- | --- |
//! | `press`, `release`, `tap` | a source name or a list | keys (`W`), `Mouse.Left`, `Gamepad.South`, `Gamepad.RightTrigger`, as the input config names them |
//! | `press_action`, `release_action`, `tap_action` | an action name or a list | a source the config binds the action to |
//! | `axis` | `{name: value}` | an axis held at a value: a composed one presses its positive or negative action past ±0.5, an analog one sets its source so it reads the value |
//! | `mouse_move` | `[dx, dy]` | mouse motion in window pixels, spread over the step's `for` |
//! | `pointer` | `[x, y]` | the pointer, 0..1 from the window's top left |
//! | `click` | `[x, y]` | the pointer there, and the left button tapped |
//! | `stick` | `{left: [x, y], right: [x, y]}` | pad sticks, down positive as the mouse is |
//! | `trigger` | `{left: v, right: v}` | pad triggers, 0..1 |
//! | `wait` | `{frames: n}` or `{seconds: s}` | nothing, for that long |
//!
//! and optionally `for` (`{frames: n}` or `{seconds: s}`): hold what the step
//! pressed that long, then let it go. Without `for` a press holds until a
//! `release` or the sequence ends, and a tap or click lasts one frame. `player`
//! (0–3) picks the pad; keyboard and mouse are player 0's. Seconds are game
//! time.

use std::collections::VecDeque;
use std::time::Instant;

use orrin_ecs::World;
use serde::Deserialize;
use serde_json::{Value, json};

use super::Peer;
use crate::scene::input::actions::{Axis, Definition};
use crate::scene::input::binding::{self, Binding};
use crate::scene::input::keys::{self, MouseAxis, PadAxis, PadButton};
use crate::scene::input::{Actions, InputState, MAX_PLAYERS};
use crate::scene::{LogBuffer, LogLevel, Play, PlayState, Time};

/// Steps in one call.
pub const MAX_STEPS: usize = 256;

/// The longest a call's steps may last in all, in game seconds with frames
/// counted at [`FRAME_RATE`]: a third of the client's 30 s answer timeout, so
/// a game running at half speed still answers in time.
pub const MAX_SECONDS: f32 = 10.0;

/// What a frame is worth when frames and seconds are added up for
/// [`MAX_SECONDS`].
const FRAME_RATE: f32 = 60.0;

/// The editor's one virtual device: the sequence it is playing, if any, and
/// whether it has pressed anything since it last let go.
#[derive(Default)]
pub struct GameInput {
    playing: Option<Playtest>,
    driven: bool,
}

impl GameInput {
    /// This frame of the sequence, before the actions resolve; or, with none
    /// playing and something still held, let go of all of it. Two checks of a
    /// field otherwise.
    pub fn feed(&mut self, world: &mut World, extent: [u32; 2]) {
        match &mut self.playing {
            Some(test) => {
                test.feed(world, extent);
                self.driven = true;
            }
            None if self.driven => {
                self.driven = false;
                if let Some(mut input) = world.get_resource_mut::<InputState>() {
                    input.virtual_release_all();
                }
            }
            None => {}
        }
    }

    /// Drop the sequence without answering: its remote has gone.
    pub fn cancel(&mut self) {
        self.playing = None;
    }

    pub(super) fn busy(&self) -> bool {
        self.playing.is_some()
    }

    pub(super) fn start(&mut self, mut test: Playtest, peer: Peer, id: u64) {
        test.peer = peer;
        test.id = id;
        self.playing = Some(test);
    }

    /// The sequence's answer once it has ended, with who is waiting for it; it
    /// is dropped then, and its holds let go at the next feed. One whose
    /// connection closed ends with no answer.
    pub(super) fn settle(
        &mut self,
        world: &World,
        open: impl Fn(Peer) -> bool,
        allowed: Result<(), String>,
    ) -> Option<(Peer, u64, Result<Value, String>)> {
        let test = self.playing.as_ref()?;
        let answer = if !open(test.peer) {
            None
        } else if let Err(refused) = allowed {
            Some(Err(format!(
                "`game_input`: stopped after {} frames: {refused}",
                test.frames
            )))
        } else {
            Some(Ok(test.ended(world)?))
        };
        let test = self.playing.take()?;
        answer.map(|answer| (test.peer, test.id, answer))
    }
}

/// One source on the virtual device.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Source {
    Key(u32),
    Mouse(u32),
    PadButton(usize, PadButton),
    /// A pad axis held at a value; released to zero.
    PadAxis(usize, PadAxis, f32),
    /// Mouse motion every frame while held: an analog axis bound to the mouse.
    Motion(MouseAxis, f32),
}

enum Effect {
    Press(Source),
    Release(Source),
    Move(f32, f32),
    Point([f32; 2]),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Length {
    frames: Option<u32>,
    seconds: Option<f32>,
}

impl Length {
    fn read(self) -> Result<Remaining, String> {
        match (self.frames, self.seconds) {
            (Some(frames), None) => Ok(Remaining::Frames(frames)),
            (None, Some(seconds)) if seconds.is_finite() && seconds >= 0.0 => {
                Ok(Remaining::Seconds(seconds))
            }
            (None, Some(_)) => Err("a length in seconds is a number of at least 0".to_owned()),
            _ => Err(
                "a length is `{\"frames\": n}` or `{\"seconds\": s}`, one of the two".to_owned(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Remaining {
    Frames(u32),
    Seconds(f32),
}

impl Remaining {
    fn seconds(self) -> f32 {
        match self {
            Self::Frames(frames) => frames as f32 / FRAME_RATE,
            Self::Seconds(seconds) => seconds,
        }
    }

    fn left(self) -> bool {
        match self {
            Self::Frames(frames) => frames > 0,
            Self::Seconds(seconds) => seconds > 0.0,
        }
    }

    /// The share of what is left that this frame takes.
    fn share(self, delta: f32) -> f32 {
        match self {
            Self::Frames(frames) => 1.0 / frames.max(1) as f32,
            Self::Seconds(seconds) if seconds > 0.0 => (delta / seconds).min(1.0),
            Self::Seconds(_) => 1.0,
        }
    }

    fn consume(&mut self, delta: f32) {
        match self {
            Self::Frames(frames) => *frames = frames.saturating_sub(1),
            Self::Seconds(seconds) => *seconds -= delta,
        }
    }
}

struct Step {
    effects: Vec<Effect>,
    length: Option<Remaining>,
    /// Let go when the step's length runs out: its `for`, or a tap's frame.
    release_after: Vec<Source>,
    /// Mouse motion spread over the step's length.
    glide: Option<(f32, f32)>,
}

struct Current {
    remaining: Remaining,
    release_after: Vec<Source>,
    glide: Option<(f32, f32)>,
}

/// A sequence being played, and who is waiting for it.
pub(super) struct Playtest {
    peer: Peer,
    id: u64,
    steps: VecDeque<Step>,
    total: usize,
    current: Option<Current>,
    /// Mouse motion a held analog mouse axis adds every frame.
    motion: (f32, f32),
    frames: u32,
    seconds: f64,
    said_from: u64,
    raised: Instant,
    done: bool,
}

impl Playtest {
    fn feed(&mut self, world: &mut World, [width, height]: [u32; 2]) {
        if self.done
            || !world
                .get_resource::<Play>()
                .is_some_and(|play| play.is_live())
        {
            return;
        }
        {
            let mut play = world.resource_mut::<Play>();
            if play.state() == PlayState::Paused {
                play.step();
            }
        }
        let delta = world
            .get_resource::<Time>()
            .map_or(0.0, |time| time.delta_time());
        self.frames += 1;
        self.seconds += f64::from(delta);
        let Some(mut input) = world.get_resource_mut::<InputState>() else {
            return;
        };
        if self.motion != (0.0, 0.0) {
            input.virtual_move_mouse(self.motion);
        }
        loop {
            if let Some(current) = &mut self.current {
                if current.remaining.left() {
                    if let Some((dx, dy)) = current.glide {
                        let share = current.remaining.share(delta);
                        input.virtual_move_mouse((dx * share, dy * share));
                        current.glide = Some((dx * (1.0 - share), dy * (1.0 - share)));
                    }
                    current.remaining.consume(delta);
                    break;
                }
                let current = self.current.take().expect("just matched");
                for source in current.release_after {
                    release(&mut input, &mut self.motion, source);
                }
                continue;
            }
            let Some(step) = self.steps.pop_front() else {
                self.done = true;
                break;
            };
            for effect in step.effects {
                match effect {
                    Effect::Press(source) => press(&mut input, &mut self.motion, source),
                    Effect::Release(source) => release(&mut input, &mut self.motion, source),
                    Effect::Move(dx, dy) => input.virtual_move_mouse((dx, dy)),
                    Effect::Point([x, y]) => {
                        input.virtual_point((x * width as f32, y * height as f32));
                    }
                }
            }
            match step.length {
                Some(remaining) => {
                    self.current = Some(Current {
                        remaining,
                        release_after: step.release_after,
                        glide: step.glide,
                    });
                }
                None => {
                    for source in step.release_after {
                        release(&mut input, &mut self.motion, source);
                    }
                }
            }
        }
    }

    /// The answer, once the sequence has played out or can play no further.
    fn ended(&self, world: &World) -> Option<Value> {
        let play = world.get_resource::<Play>().map(|play| play.state());
        let why = if play.is_none_or(|play| play == PlayState::Editing) {
            Some("the game stopped before the sequence finished")
        } else if self.raised.elapsed() >= super::command::DEADLINE {
            Some(
                "the deadline passed before the sequence finished; a slow or paused frame rate plays it slower than declared",
            )
        } else if self.done {
            None
        } else {
            return None;
        };
        let warnings = world.get_resource::<LogBuffer>().map_or(0, |log| {
            log.since(self.said_from)
                .filter(|(_, entry)| entry.level >= LogLevel::Warning)
                .count()
        });
        let left = self.steps.len() + usize::from(self.current.is_some());
        let mut answer = json!({
            "completed": why.is_none(),
            "steps": self.total - left,
            "of": self.total,
            "frames": self.frames,
            "game_seconds": (self.seconds * 1000.0).round() / 1000.0,
            "play": match play {
                Some(PlayState::Playing) => "playing",
                Some(PlayState::Paused) => "paused",
                _ => "editing",
            },
            "warnings": warnings,
        });
        if let Some(why) = why {
            answer["ended"] = Value::String(why.to_owned());
        }
        if warnings > 0 {
            answer["read"] = Value::String(format!(
                "console_read with cursor {} and level warn",
                self.said_from
            ));
        }
        Some(answer)
    }
}

fn press(input: &mut InputState, motion: &mut (f32, f32), source: Source) {
    match source {
        Source::Key(code) => input.virtual_press(code),
        Source::Mouse(button) => input.virtual_press_mouse_button(button),
        Source::PadButton(player, button) => input.virtual_set_pad_button(player, button, true),
        Source::PadAxis(player, axis, value) => input.virtual_set_pad_axis(player, axis, value),
        Source::Motion(MouseAxis::X, value) => motion.0 = value,
        Source::Motion(MouseAxis::Y, value) => motion.1 = value,
    }
}

fn release(input: &mut InputState, motion: &mut (f32, f32), source: Source) {
    match source {
        Source::Key(code) => input.virtual_release(code),
        Source::Mouse(button) => input.virtual_release_mouse_button(button),
        Source::PadButton(player, button) => input.virtual_set_pad_button(player, button, false),
        Source::PadAxis(player, axis, _) => input.virtual_set_pad_axis(player, axis, 0.0),
        Source::Motion(MouseAxis::X, _) => motion.0 = 0.0,
        Source::Motion(MouseAxis::Y, _) => motion.1 = 0.0,
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Names {
    One(String),
    Many(Vec<String>),
}

impl Names {
    fn list(&self) -> Vec<&str> {
        match self {
            Self::One(name) => vec![name.as_str()],
            Self::Many(names) => names.iter().map(String::as_str).collect(),
        }
    }
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sides<T> {
    left: Option<T>,
    right: Option<T>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStep {
    press: Option<Names>,
    release: Option<Names>,
    tap: Option<Names>,
    press_action: Option<Names>,
    release_action: Option<Names>,
    tap_action: Option<Names>,
    axis: Option<serde_json::Map<String, Value>>,
    mouse_move: Option<[f32; 2]>,
    pointer: Option<[f32; 2]>,
    click: Option<[f32; 2]>,
    stick: Option<Sides<[f32; 2]>>,
    trigger: Option<Sides<f32>>,
    wait: Option<Length>,
    #[serde(rename = "for")]
    hold: Option<Length>,
    player: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    steps: Vec<Value>,
}

/// A sequence from `params`, its names resolved against the game's input
/// config now, or why it cannot be played.
pub(super) fn playtest(world: &World, params: &Value) -> Result<Playtest, String> {
    let params: Params = super::answer::params_of_required(params)?;
    if params.steps.is_empty() {
        return Err("`steps` is empty; give at least one".to_owned());
    }
    if params.steps.len() > MAX_STEPS {
        return Err(format!(
            "{} steps is more than one call takes ({MAX_STEPS}); split the playtest into several calls",
            params.steps.len()
        ));
    }
    let actions = world.get_resource::<Actions>();
    let mut steps = VecDeque::with_capacity(params.steps.len());
    let mut length = 0.0;
    for (index, raw) in params.steps.iter().enumerate() {
        let step =
            parse_step(raw, actions.as_deref()).map_err(|why| format!("step {index}: {why}"))?;
        length += step.length.map_or(0.0, Remaining::seconds);
        steps.push_back(step);
    }
    if length > MAX_SECONDS {
        return Err(format!(
            "the steps last {length:.1} s in all (frames counted at {FRAME_RATE} a second), \
             more than one call plays ({MAX_SECONDS} s); split the playtest into several calls"
        ));
    }
    Ok(Playtest {
        peer: 0,
        id: 0,
        total: steps.len(),
        steps,
        current: None,
        motion: (0.0, 0.0),
        frames: 0,
        seconds: 0.0,
        said_from: world
            .get_resource::<LogBuffer>()
            .map_or(0, |log| log.next_number()),
        raised: Instant::now(),
        done: false,
    })
}

fn parse_step(raw: &Value, actions: Option<&Actions>) -> Result<Step, String> {
    let raw: RawStep = serde_json::from_value(raw.clone()).map_err(|error| error.to_string())?;
    let player = raw.player.unwrap_or(0);
    if player >= MAX_PLAYERS {
        return Err(format!("`player` is 0 to {}", MAX_PLAYERS - 1));
    }
    let what = [
        raw.press.is_some(),
        raw.release.is_some(),
        raw.tap.is_some(),
        raw.press_action.is_some(),
        raw.release_action.is_some(),
        raw.tap_action.is_some(),
        raw.axis.is_some(),
        raw.mouse_move.is_some(),
        raw.pointer.is_some(),
        raw.click.is_some(),
        raw.stick.is_some(),
        raw.trigger.is_some(),
        raw.wait.is_some(),
    ]
    .into_iter()
    .filter(|&given| given)
    .count();
    if what != 1 {
        return Err(
            "a step does one thing: press, release, tap, press_action, release_action, \
             tap_action, axis, mouse_move, pointer, click, stick, trigger or wait"
                .to_owned(),
        );
    }
    let hold = raw.hold.map(Length::read).transpose()?;
    if raw.wait.is_some() && hold.is_some() {
        return Err("a wait's length is its own; leave out `for`".to_owned());
    }
    let mut step = Step {
        effects: Vec::new(),
        length: hold,
        release_after: Vec::new(),
        glide: None,
    };
    let held = |step: &mut Step, sources: Vec<Source>| {
        for source in sources {
            step.effects.push(Effect::Press(source));
            if hold.is_some() {
                step.release_after.push(source);
            }
        }
    };
    let tapped = |step: &mut Step, sources: Vec<Source>| {
        step.length = Some(hold.unwrap_or(Remaining::Frames(1)));
        for source in sources {
            step.effects.push(Effect::Press(source));
            step.release_after.push(source);
        }
    };
    let by_source = |names: &Names| -> Result<Vec<Source>, String> {
        names
            .list()
            .into_iter()
            .map(|name| source(name, player))
            .collect()
    };
    let by_action = |names: &Names| -> Result<Vec<Source>, String> {
        names
            .list()
            .into_iter()
            .map(|name| action_source(actions, name, player))
            .collect()
    };
    if let Some(names) = &raw.press {
        held(&mut step, by_source(names)?);
    } else if let Some(names) = &raw.press_action {
        held(&mut step, by_action(names)?);
    } else if let Some(names) = &raw.tap {
        tapped(&mut step, by_source(names)?);
    } else if let Some(names) = &raw.tap_action {
        tapped(&mut step, by_action(names)?);
    } else if let Some(names) = raw.release.as_ref().or(raw.release_action.as_ref()) {
        let sources = match raw.release.is_some() {
            true => by_source(names)?,
            false => by_action(names)?,
        };
        step.effects
            .extend(sources.into_iter().map(Effect::Release));
    } else if let Some(axes) = &raw.axis {
        for (name, value) in axes {
            let value = value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("axis `{name}`: its value is a number"))?;
            let (pressed, released) = axis_sources(actions, name, value as f32, player)?;
            step.effects
                .extend(released.into_iter().map(Effect::Release));
            held(&mut step, pressed);
        }
    } else if let Some([dx, dy]) = raw.mouse_move {
        if !(dx.is_finite() && dy.is_finite()) {
            return Err("`mouse_move` is two numbers".to_owned());
        }
        match hold {
            Some(_) => step.glide = Some((dx, dy)),
            None => step.effects.push(Effect::Move(dx, dy)),
        }
    } else if let Some(at) = raw.pointer {
        step.effects.push(Effect::Point(normalised(at)?));
    } else if let Some(at) = raw.click {
        step.effects.push(Effect::Point(normalised(at)?));
        tapped(&mut step, vec![Source::Mouse(0)]);
    } else if let Some(sticks) = &raw.stick {
        let mut sources = Vec::new();
        for (side, (x_axis, y_axis)) in [
            (sticks.left, (PadAxis::LeftStickX, PadAxis::LeftStickY)),
            (sticks.right, (PadAxis::RightStickX, PadAxis::RightStickY)),
        ] {
            if let Some([x, y]) = side {
                sources.push(Source::PadAxis(player, x_axis, unit(x, -1.0)?));
                sources.push(Source::PadAxis(player, y_axis, unit(y, -1.0)?));
            }
        }
        held(&mut step, sources);
    } else if let Some(triggers) = &raw.trigger {
        let mut sources = Vec::new();
        for (side, axis) in [
            (triggers.left, PadAxis::LeftTrigger),
            (triggers.right, PadAxis::RightTrigger),
        ] {
            if let Some(value) = side {
                sources.push(Source::PadAxis(player, axis, unit(value, 0.0)?));
            }
        }
        held(&mut step, sources);
    } else if let Some(wait) = raw.wait {
        step.length = Some(wait.read()?);
    }
    Ok(step)
}

fn unit(value: f32, low: f32) -> Result<f32, String> {
    if value.is_finite() && (low..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(format!("{value} is outside {low}..1"))
    }
}

fn normalised([x, y]: [f32; 2]) -> Result<[f32; 2], String> {
    Ok([unit(x, 0.0)?, unit(y, 0.0)?])
}

/// A source by the name the input config spells it with.
fn source(name: &str, player: usize) -> Result<Source, String> {
    let binding = binding::parse(name).map_err(|unknown| unknown.to_string())?;
    pressing(binding, player).ok_or_else(|| match binding {
        Binding::MouseAxis(_) => {
            format!("`{name}` is motion, not a button; move it with `mouse_move`")
        }
        _ => format!("`{name}` is player 0's; keyboard and mouse are one device"),
    })
}

/// What pressing `binding` holds: a button down, an analog source at full
/// travel. `None` for mouse motion, and for keyboard and mouse past player 0.
fn pressing(binding: Binding, player: usize) -> Option<Source> {
    match binding {
        Binding::Key(code) if player == 0 => Some(Source::Key(code)),
        Binding::MouseButton(button) if player == 0 => Some(Source::Mouse(u32::from(button))),
        Binding::PadButton(button) => Some(Source::PadButton(player, button)),
        Binding::PadAxis(axis) => Some(Source::PadAxis(player, axis, 1.0)),
        _ => None,
    }
}

/// The first source `name`'s bindings list that the device can press for
/// `player`: the order the config gives them in.
fn action_source(actions: Option<&Actions>, name: &str, player: usize) -> Result<Source, String> {
    match actions.and_then(|actions| actions.definition(name)) {
        Some(Definition::Action(bindings)) => bindings
            .iter()
            .find_map(|&binding| match binding {
                Binding::MouseAxis(axis) if player == 0 => Some(Source::Motion(axis, 1.0)),
                _ => pressing(binding, player),
            })
            .ok_or_else(|| format!("action `{name}` binds nothing player {player} can press")),
        Some(Definition::Axis(_)) => Err(format!(
            "`{name}` is an axis; hold it at a value with `axis`: {{\"{name}\": 1}}"
        )),
        _ => Err(format!(
            "the input config defines no action `{name}`; input_names lists what it does"
        )),
    }
}

/// What holding axis `name` at `value` presses, and what it lets go.
fn axis_sources(
    actions: Option<&Actions>,
    name: &str,
    value: f32,
    player: usize,
) -> Result<(Vec<Source>, Vec<Source>), String> {
    let actions_ref = actions;
    let Some(Definition::Axis(axis)) = actions.and_then(|actions| actions.definition(name)) else {
        return Err(match actions.and_then(|actions| actions.definition(name)) {
            Some(Definition::Action(_)) => {
                format!("`{name}` is an action; press it with `press_action`")
            }
            _ => {
                format!("the input config defines no axis `{name}`; input_names lists what it does")
            }
        });
    };
    match axis {
        Axis::Composed { positive, negative } => {
            let name_of = |id| {
                actions_ref
                    .and_then(|actions| actions.name(id))
                    .unwrap_or_default()
                    .to_owned()
            };
            let plus = action_source(actions_ref, &name_of(*positive), player)?;
            let minus = action_source(actions_ref, &name_of(*negative), player)?;
            Ok(if value >= 0.5 {
                (vec![plus], vec![minus])
            } else if value <= -0.5 {
                (vec![minus], vec![plus])
            } else {
                (Vec::new(), vec![plus, minus])
            })
        }
        Axis::Analog {
            source,
            deadzone,
            scale,
        } => {
            if *scale == 0.0 {
                return Err(format!("axis `{name}` has a scale of 0; no input moves it"));
            }
            let raw = value / scale;
            let source = match *source {
                Binding::PadAxis(pad) => {
                    let magnitude = raw.abs().min(1.0);
                    let deadzone = deadzone.clamp(0.0, crate::scene::input::actions::MAX_DEADZONE);
                    let travel = match magnitude > 0.0 {
                        true => deadzone + magnitude * (1.0 - deadzone),
                        false => 0.0,
                    };
                    Source::PadAxis(player, pad, travel.copysign(raw))
                }
                Binding::MouseAxis(axis) if player == 0 => Source::Motion(axis, raw),
                other => pressing(other, player).ok_or_else(|| {
                    format!("axis `{name}` reads a source player {player} cannot drive")
                })?,
            };
            Ok((vec![source], Vec::new()))
        }
    }
}

/// `input_names`: the game's actions and axes as its input config defines
/// them, and every source name the config and `game_input` accept.
pub(super) fn names(world: &World) -> Value {
    let mut actions = Vec::new();
    let mut axes = Vec::new();
    if let Some(defined) = world.get_resource::<Actions>() {
        for (name, definition) in defined.defined() {
            match definition {
                Definition::Action(bindings) => actions.push(json!({
                    "name": name,
                    "bound": bindings.iter().map(|&b| source_name(b)).collect::<Vec<_>>(),
                })),
                Definition::Axis(Axis::Composed { positive, negative }) => axes.push(json!({
                    "name": name,
                    "positive": defined.name(*positive),
                    "negative": defined.name(*negative),
                })),
                Definition::Axis(Axis::Analog { source, scale, .. }) => axes.push(json!({
                    "name": name,
                    "source": source_name(*source),
                    "scale": scale,
                })),
                Definition::Unbound => {}
            }
        }
    }
    json!({
        "actions": actions,
        "axes": axes,
        "keys": keys::KEYS.iter().map(|key| key.name).collect::<Vec<_>>(),
        "mouse": keys::MOUSE_BUTTONS
            .iter()
            .map(|(name, _)| format!("Mouse.{name}"))
            .chain(keys::MOUSE_AXES.iter().map(|(name, _)| format!("Mouse.{name}")))
            .collect::<Vec<_>>(),
        "gamepad": keys::PAD_BUTTONS
            .iter()
            .map(|(name, _, _)| format!("Gamepad.{name}"))
            .chain(keys::PAD_AXES.iter().map(|(name, _)| format!("Gamepad.{name}")))
            .collect::<Vec<_>>(),
    })
}

/// A binding as the config spells it.
fn source_name(binding: Binding) -> String {
    match binding {
        Binding::Key(code) => keys::KEYS
            .iter()
            .find(|key| key.code == code)
            .map_or_else(|| format!("key {code}"), |key| key.name.to_owned()),
        Binding::MouseButton(button) => keys::MOUSE_BUTTONS
            .iter()
            .find(|(_, index)| *index == button)
            .map_or_else(String::new, |(name, _)| format!("Mouse.{name}")),
        Binding::MouseAxis(axis) => keys::MOUSE_AXES
            .iter()
            .find(|(_, which)| *which == axis)
            .map_or_else(String::new, |(name, _)| format!("Mouse.{name}")),
        Binding::PadButton(button) => keys::PAD_BUTTONS
            .iter()
            .find(|(_, which, _)| *which == button)
            .map_or_else(String::new, |(name, _, _)| format!("Gamepad.{name}")),
        Binding::PadAxis(axis) => keys::PAD_AXES
            .iter()
            .find(|(_, which)| *which == axis)
            .map_or_else(String::new, |(name, _)| format!("Gamepad.{name}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::input::config;

    const W: u32 = 23;

    fn game() -> World {
        let mut world = World::new();
        let mut input = InputState::new();
        input.end_frame();
        world.insert_resource(input);
        let mut actions = Actions::new();
        let specs = config::parse(
            "format_version = 1\n[actions]\nForward = [\"W\"]\n",
            std::path::Path::new("input.toml"),
        )
        .unwrap();
        actions.apply(specs);
        world.insert_resource(actions);
        world.insert_resource(Time::new());
        world.insert_resource(LogBuffer::default());
        let mut play = Play::default();
        play.set(PlayState::Playing);
        world.insert_resource(play);
        world
    }

    /// A sequence holding a key, a mouse button, a pad button and a stick,
    /// started and fed one frame.
    fn holding(world: &mut World) -> GameInput {
        let steps = json!({"steps": [
            {"press_action": "Forward"},
            {"press": ["Mouse.Left", "Gamepad.South"]},
            {"stick": {"left": [0.0, -1.0]}},
            {"wait": {"seconds": 5}},
        ]});
        let mut device = GameInput::default();
        device.start(playtest(world, &steps).unwrap(), 7, 1);
        device.feed(world, [100, 100]);
        let input = world.resource::<InputState>();
        assert!(input.key_down(W) && input.mouse_button_down(0));
        assert!(input.pad_button_down(0, PadButton::South));
        assert_eq!(input.pad_axis(0, PadAxis::LeftStickY), -1.0);
        device
    }

    fn let_go(world: &mut World, device: &mut GameInput) {
        world.resource_mut::<InputState>().end_frame();
        device.feed(world, [100, 100]);
        let input = world.resource::<InputState>();
        assert!(
            !input.key_down(W) && !input.mouse_button_down(0),
            "a key stuck"
        );
        assert!(input.key_released(W), "the release was never seen");
        assert!(
            !input.pad_button_down(0, PadButton::South),
            "a pad button stuck"
        );
        assert_eq!(input.pad_axis(0, PadAxis::LeftStickY), 0.0, "a stick stuck");
        assert!(!input.virtual_holds());
    }

    /// The silent failure this exists to prevent: a key an agent pressed that
    /// nothing lets go of drives the game until the session ends, and every
    /// way a sequence can end early is a way to forget it.
    #[test]
    fn everything_a_sequence_pressed_is_let_go_however_it_ends_early() {
        let mut world = game();
        let mut device = holding(&mut world);
        assert!(
            device.settle(&world, |_| false, Ok(())).is_none(),
            "nobody to answer"
        );
        assert!(!device.busy());
        let_go(&mut world, &mut device);

        for ended in [
            &[PlayState::Editing][..],
            &[PlayState::Paused, PlayState::Editing],
        ] {
            let mut world = game();
            let mut device = holding(&mut world);
            for &state in ended {
                world.resource_mut::<Play>().set(state);
            }
            let (peer, id, answer) = device.settle(&world, |_| true, Ok(())).unwrap();
            let answer = answer.unwrap();
            assert_eq!((peer, id, &answer["completed"]), (7, 1, &json!(false)));
            let_go(&mut world, &mut device);
        }

        let mut world = game();
        let mut device = holding(&mut world);
        let refused = device.settle(&world, |_| true, Err("turned off".to_owned()));
        assert!(refused.unwrap().2.is_err());
        let_go(&mut world, &mut device);

        let mut world = game();
        let mut device = holding(&mut world);
        device.cancel();
        let_go(&mut world, &mut device);
    }
}
