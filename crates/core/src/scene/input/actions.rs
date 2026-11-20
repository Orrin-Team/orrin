//! Named actions and axes, resolved from bindings once per frame.
//!
//! Game code names an action, never a key. The indirection exists so that
//! remapping is a config change rather than an API break, which is only true if
//! nothing above this module ever learns what a binding is.
//!
//! **Resolution runs once per frame**, after the event pump and before scripts
//! tick, and it runs whether or not events arrived and whether or not scripts
//! are ticking. A resolve gated on either leaves the previous frame's state
//! standing, and every edge derived from it fires a frame late or twice.
//!
//! **Edges are the action's own, not any binding's.** An action is held when
//! any of its bindings is, and pressed on the frame its aggregate goes from
//! unheld to held. Aggregating each binding's own edges instead would fire a
//! second press when a second bound key joins one already down.
//!
//! **Ids are interned per name and stable for the process.** They are handed
//! out on first lookup, not by a config, so reloading bindings rewrites what an
//! id points at and never what it is — a script may cache one across a reload,
//! and a name nobody bound still has an id that reads as unpressed. That is
//! what makes reload cheap enough to do on every save.
//!
//! **Axes compose, they do not re-bind.** An axis names two actions, so a key
//! appears in the file exactly once and rebinding `Left` moves `Horizontal`
//! with it. An axis may also read one analog source directly; it may never name
//! another axis, which is what keeps resolution two passes and acyclic.

use std::collections::HashMap;

use super::binding::{Binding, MOTION_THRESHOLD, TRIGGER_THRESHOLD};
use super::keys::MouseAxis;
use super::{InputState, MAX_PLAYERS};

/// A name's handle. Stable for the life of the process; see the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionId(pub u32);

/// The largest deadzone a config may ask for. Past this the rescale divides by
/// almost nothing and a stick becomes a switch. Enforced by
/// [`config`](super::config), so a file naming a bigger one is refused rather
/// than quietly clamped; the clamp below is the guard for [`Actions::apply`]
/// being called with a `Spec` that came from somewhere else.
pub const MAX_DEADZONE: f32 = 0.9;

/// What a name resolves to. A name may be interned before anything defines it,
/// which is the shape a typo takes: an id that answers no to everything.
#[derive(Clone, Debug, PartialEq)]
pub enum Definition {
    Unbound,
    Action(Vec<Binding>),
    Axis(Axis),
}

/// A definition as a config file states it, before names become ids. An axis
/// names its components; only [`Actions`] may turn a name into an id, because
/// only it knows which ids exist.
#[derive(Clone, Debug, PartialEq)]
pub enum Spec {
    Action(Vec<Binding>),
    Composed {
        positive: String,
        negative: String,
    },
    Analog {
        source: Binding,
        deadzone: f32,
        scale: f32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Axis {
    /// One action each way, read as −1, 0 or +1.
    Composed {
        positive: ActionId,
        negative: ActionId,
    },
    /// One analog source, deadzoned if it is bounded, then scaled.
    Analog {
        source: Binding,
        deadzone: f32,
        scale: f32,
    },
}

/// The resolved input state game code reads.
#[derive(Default)]
pub struct Actions {
    names: Vec<String>,
    ids: HashMap<String, ActionId>,
    defs: Vec<Definition>,
    held: Vec<[bool; MAX_PLAYERS]>,
    held_last: Vec<[bool; MAX_PLAYERS]>,
    values: Vec<[f32; MAX_PLAYERS]>,
    /// Whether anything has ever asked for this name's id — which is what makes
    /// it worth reporting as unbound. A name interned only to be an axis's
    /// component is nobody's mistake until a script names it too.
    requested: Vec<bool>,
    /// Set when bindings change, cleared by the next resolve, which seeds the
    /// previous frame from the current one so the swap itself fires no edges.
    /// A rebind while the old key is held would otherwise read as a release,
    /// and the new one as a press nobody made.
    reseed: bool,
    warnings: Vec<String>,
}

impl Actions {
    pub fn new() -> Self {
        Self::default()
    }

    /// The id for a name, interning it if this is the first time it is asked
    /// for. Names are matched exactly: they are identifiers the game author
    /// chose on both sides of the file, and a case-folded match would hide the
    /// mismatch rather than report it.
    pub fn id(&mut self, name: &str) -> ActionId {
        let id = self.intern(name);
        let index = id.0 as usize;
        let first = !std::mem::replace(&mut self.requested[index], true);
        if first && self.defs[index] == Definition::Unbound {
            self.warn_unbound(index);
        }
        id
    }

    /// Report a name a caller asked for that nothing binds. Raised once when the
    /// name is first asked for, and again after any reload that leaves it
    /// unbound — the second is the one that catches an action renamed in the
    /// file and not in the script, which is the mistake live reload invites.
    fn warn_unbound(&mut self, index: usize) {
        self.warnings.push(format!(
            "input action `{}` is not bound; nothing in the input config defines it",
            self.names[index]
        ));
    }

    fn intern(&mut self, name: &str) -> ActionId {
        if let Some(id) = self.ids.get(name) {
            return *id;
        }
        let id = ActionId(self.names.len() as u32);
        self.names.push(name.to_owned());
        self.ids.insert(name.to_owned(), id);
        self.defs.push(Definition::Unbound);
        self.held.push([false; MAX_PLAYERS]);
        self.held_last.push([false; MAX_PLAYERS]);
        self.values.push([0.0; MAX_PLAYERS]);
        self.requested.push(false);
        id
    }

    /// Replace every definition. Names absent from `defs` keep their ids and
    /// fall back to unbound, so a binding deleted from the file stops firing
    /// without stranding a script that still asks for it — and anything that had
    /// already asked for one is warned about, here rather than at interning,
    /// because that is where the name went quiet.
    ///
    /// Sets `reseed`, which suppresses the next resolve's edges. That matters
    /// for the reload this is usually called for, and equally for the
    /// edit-to-play transition, where the same stale comparison would fire a
    /// press for every key already held when play began.
    pub fn apply(&mut self, specs: Vec<(String, Spec)>) {
        self.defs.fill(Definition::Unbound);
        for (name, spec) in specs {
            let id = self.intern(&name);
            let def = match spec {
                Spec::Action(bindings) => Definition::Action(bindings),
                Spec::Composed { positive, negative } => Definition::Axis(Axis::Composed {
                    positive: self.intern(&positive),
                    negative: self.intern(&negative),
                }),
                Spec::Analog {
                    source,
                    deadzone,
                    scale,
                } => Definition::Axis(Axis::Analog {
                    source,
                    deadzone,
                    scale,
                }),
            };
            self.defs[id.0 as usize] = def;
        }
        for index in 0..self.defs.len() {
            if self.requested[index] && self.defs[index] == Definition::Unbound {
                self.warn_unbound(index);
            }
        }
        self.reseed = true;
    }

    /// What `name` is defined as, without interning it: a name an agent's
    /// playtest got wrong is the agent's to be told, not the game's to warn
    /// about.
    pub fn definition(&self, name: &str) -> Option<&Definition> {
        self.ids.get(name).map(|id| &self.defs[id.0 as usize])
    }

    /// The name an id was interned under.
    pub fn name(&self, id: ActionId) -> Option<&str> {
        self.names.get(id.0 as usize).map(String::as_str)
    }

    /// Every name the config defines, in the order they were first seen.
    pub fn defined(&self) -> impl Iterator<Item = (&str, &Definition)> {
        self.names
            .iter()
            .zip(&self.defs)
            .filter(|(_, def)| **def != Definition::Unbound)
            .map(|(name, def)| (name.as_str(), def))
    }

    /// Drain the warnings interning has accumulated, for the caller to log.
    /// Kept out of this module so resolution has no opinion about console.
    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    /// Recompute every action and axis. Actions first: an axis reads its
    /// components' aggregates, so the two passes cannot merge.
    pub fn resolve(&mut self, input: &InputState) {
        std::mem::swap(&mut self.held, &mut self.held_last);

        for (index, def) in self.defs.iter().enumerate() {
            let mut held = [false; MAX_PLAYERS];
            if let Definition::Action(bindings) = def {
                for (player, slot) in held.iter_mut().enumerate() {
                    *slot = bindings
                        .iter()
                        .any(|binding| digital(input, *binding, player));
                }
            }
            self.held[index] = held;
        }

        for (index, def) in self.defs.iter().enumerate() {
            let mut values = [0.0; MAX_PLAYERS];
            if let Definition::Axis(axis) = def {
                for (player, slot) in values.iter_mut().enumerate() {
                    *slot = match axis {
                        Axis::Composed { positive, negative } => {
                            let plus = self.held[positive.0 as usize][player];
                            let minus = self.held[negative.0 as usize][player];
                            f32::from(plus) - f32::from(minus)
                        }
                        Axis::Analog {
                            source,
                            deadzone,
                            scale,
                        } => {
                            let raw = analog(input, *source, player);
                            let shaped = if source.is_normalised() {
                                apply_deadzone(raw, *deadzone)
                            } else {
                                raw
                            };
                            shaped * scale
                        }
                    };
                }
            }
            self.values[index] = values;
        }

        if self.reseed {
            self.held_last.copy_from_slice(&self.held);
            self.reseed = false;
        }
    }

    pub fn held(&self, id: ActionId, player: usize) -> bool {
        self.slot(&self.held, id, player)
    }

    pub fn pressed(&self, id: ActionId, player: usize) -> bool {
        self.slot(&self.held, id, player) && !self.slot(&self.held_last, id, player)
    }

    pub fn released(&self, id: ActionId, player: usize) -> bool {
        !self.slot(&self.held, id, player) && self.slot(&self.held_last, id, player)
    }

    /// An axis's value. An id naming an action rather than an axis reads zero,
    /// as does a player past [`MAX_PLAYERS`]: both are the caller asking for
    /// something that exists but holds nothing, which is the same answer an
    /// unbound name gives.
    pub fn axis(&self, id: ActionId, player: usize) -> f32 {
        self.values
            .get(id.0 as usize)
            .and_then(|players| players.get(player))
            .copied()
            .unwrap_or(0.0)
    }

    fn slot(&self, table: &[[bool; MAX_PLAYERS]], id: ActionId, player: usize) -> bool {
        table
            .get(id.0 as usize)
            .and_then(|players| players.get(player))
            .copied()
            .unwrap_or(false)
    }
}

/// Whether a source reads as pressed for this player.
///
/// Keyboard and mouse answer for player 0 only. They are one device between
/// however many players are in the room, and letting them answer for every slot
/// would have player two jumping on player one's spacebar.
///
/// An evented source reads held if it is *either* down now or was pressed during
/// this frame. The second half is what keeps a tap whose press and release both
/// landed between two resolves: on level state alone it was never down when
/// anyone looked, so the action would miss what the raw `key_pressed` beside it
/// catches. It reads held for one frame and released on the next, which is what
/// a tap is. A pad has no edges to consult — it is sampled, not evented — so a
/// tap shorter than a frame is invisible there and always was.
fn digital(input: &InputState, binding: Binding, player: usize) -> bool {
    match binding {
        Binding::Key(code) => player == 0 && (input.key_down(code) || input.key_pressed(code)),
        Binding::MouseButton(button) => {
            let button = button as u32;
            player == 0 && (input.mouse_button_down(button) || input.mouse_button_pressed(button))
        }
        Binding::MouseAxis(_) => {
            player == 0 && analog(input, binding, player).abs() >= MOTION_THRESHOLD
        }
        Binding::PadButton(button) => input.pad_button_down(player, button),
        Binding::PadAxis(axis) => input.pad_axis(player, axis).abs() >= TRIGGER_THRESHOLD,
    }
}

/// A source's analog value. Digital sources read as 0 or 1, so a key may stand
/// in for half an axis.
fn analog(input: &InputState, binding: Binding, player: usize) -> f32 {
    match binding {
        Binding::MouseAxis(MouseAxis::X) if player == 0 => input.mouse_delta().0,
        Binding::MouseAxis(MouseAxis::Y) if player == 0 => input.mouse_delta().1,
        Binding::MouseAxis(_) => 0.0,
        Binding::PadAxis(axis) => input.pad_axis(player, axis),
        _ => f32::from(digital(input, binding, player)),
    }
}

/// Scaled radial deadzone, on a source that rests near zero and saturates at
/// one. The rescale is the half that gets left out: without it a stick's full
/// deflection reads `1 - deadzone` and no game ever reaches full speed.
///
/// Applied per component, so a stick pushed to a corner still travels further
/// than one pushed straight up. Deadzoning the vector needs to know that two
/// axes are one stick, which this schema does not yet say.
fn apply_deadzone(value: f32, deadzone: f32) -> f32 {
    let deadzone = deadzone.clamp(0.0, MAX_DEADZONE);
    let magnitude = value.abs();
    if magnitude <= deadzone {
        return 0.0;
    }
    (((magnitude - deadzone) / (1.0 - deadzone)).min(1.0)).copysign(value)
}
