use orrin_ecs::{Entity, World};

/// Links an entity to one of its C# Behaviours via a .NET `GCHandle` (stored as
/// `u64`). An entity's Behaviours are held together in [`Scripts`].
///
/// Deliberately **not** `Clone`/`Copy`: the handle is uniquely owned and [`Drop`]
/// tears it down, so a copy would destroy the same managed object twice.
///
/// Two booleans track activation, deliberately split:
/// - `enabled` is the *desired* state — whoever wants the script on or off
///   (editor, engine code, another script) writes this and nothing else.
/// - `active` is the state the managed side last *observed* — only the script
///   tick writes it, after actually dispatching OnEnable/OnDisable.
///
/// The tick diffs the two to decide which transitions to fire; collapsing them
/// into one flag would make "flag changed" and "script was told" the same
/// event, and they aren't (the change can happen mid-frame, after this tick's
/// dispatch window closed).
///
/// `faulted` is fault isolation: set when a C# hook throws (the managed side
/// contains and logs the exception, then reports it back through the lifecycle
/// ABI). A faulted script is skipped by every future tick — no dispatch, no
/// per-frame log spam — so one bad script can't take the engine down, nor the
/// other Behaviours on its entity. It stays set until something clears it (the
/// inspector's "Clear fault" button, or a hot reload, which re-creates the
/// behaviour from the new code), which re-arms the script from wherever its
/// lifecycle left off.
///
/// `type_name` is the assembly-qualified name the behaviour was created from,
/// in the form [`canonical_type_name`] gives. It is kept on the Rust side
/// because a hot reload has to re-instantiate the script *after* the managed
/// object is gone: by the time the new assembly is live, the only surviving
/// record of what used to be attached here is this string.
///
/// `hooks` is which optional hooks the type overrides, as the managed side
/// reported when it built the instance: the per-step ones are skipped for a
/// type without them. `type_index` is the type's row in the script host's
/// per-type table, resolved once here so that timing a call indexes rather
/// than hashes a name.
#[derive(Debug)]
pub struct ScriptComponent {
    pub handle: u64,
    pub type_name: String,
    pub hooks: orrin_script::Hooks,
    pub type_index: u32,
    pub started: bool,
    pub enabled: bool,
    pub active: bool,
    pub faulted: bool,
}

impl Drop for ScriptComponent {
    fn drop(&mut self) {
        // Managed side fires OnDisable (if still active) and OnDestroy, then
        // frees the GCHandle — ordering lives in C# `Behaviours.Destroy`.
        // Runs outside a dispatch window, so those callbacks get no world
        // access (see `destroy_handle` docs).
        orrin_script::destroy_handle(self.handle);
    }
}

/// Every Behaviour on an entity, in the order they were attached: the ECS
/// component, of which an entity has one or none.
///
/// A list in one component rather than a component per Behaviour, because the
/// ECS keys storage by Rust type and every Behaviour is the same Rust type.
/// Attach order is dispatch order within each phase of the tick.
///
/// At most one Behaviour of a given type: [`push`](Self::push) refuses a
/// second. That keeps `GetBehaviour<T>()` unambiguous and matches the scene
/// format, which stores a Behaviour's fields under its type's component id,
/// once per entity.
///
/// Dropping the whole component drops every instance in attach order; taking
/// one out with [`take`](Self::take) leaves the others untouched.
#[derive(Debug, Default)]
pub struct Scripts(Vec<ScriptComponent>);

impl Scripts {
    pub fn iter(&self) -> std::slice::Iter<'_, ScriptComponent> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, ScriptComponent> {
        self.0.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The Behaviour of `type_name`, which must be in canonical form.
    pub fn get(&self, type_name: &str) -> Option<&ScriptComponent> {
        self.0.iter().find(|script| script.type_name == type_name)
    }

    pub fn get_mut(&mut self, type_name: &str) -> Option<&mut ScriptComponent> {
        self.0
            .iter_mut()
            .find(|script| script.type_name == type_name)
    }

    /// Append `script`, or hand it back when one of its type is already here.
    pub fn push(&mut self, script: ScriptComponent) -> Result<(), ScriptComponent> {
        if self.get(&script.type_name).is_some() {
            return Err(script);
        }
        self.0.push(script);
        Ok(())
    }

    /// Take the Behaviour of `type_name` out. Dropping what comes back is what
    /// destroys it, so a caller holding a world borrow should release it first.
    pub fn take(&mut self, type_name: &str) -> Option<ScriptComponent> {
        let index = self
            .0
            .iter()
            .position(|script| script.type_name == type_name)?;
        Some(self.0.remove(index))
    }

    /// Destroy `entity`'s Behaviour of `type_name` and leave its others; the
    /// entity loses its `Scripts` with the last one.
    ///
    /// The Behaviour is dropped only once the storage borrow is released,
    /// because that drop runs its OnDisable and OnDestroy.
    pub fn remove(world: &mut World, entity: Entity, type_name: &str) {
        let Some(mut scripts) = world.get_mut::<Scripts>(entity) else {
            return;
        };
        let taken = scripts.take(type_name);
        let emptied = scripts.is_empty();
        drop(scripts);
        drop(taken);
        if emptied {
            let _ = world.remove::<Scripts>(entity);
        }
    }
}

/// `Namespace.Type, Assembly`, the one form a Behaviour's type is named by on
/// the Rust side.
///
/// `World.AddScript<T>` sends `typeof(T).AssemblyQualifiedName`, which goes on
/// to the assembly's version, culture and key token; the component listing
/// sends the short form. Both reach the same type, and a duplicate check or a
/// registry lookup comparing them as strings would not know it. The version is
/// dropped rather than kept because a game assembly has one version at a time:
/// the live one. A name with no assembly is left as it is.
///
/// Commas inside brackets belong to a generic argument, not to the name.
pub fn canonical_type_name(name: &str) -> String {
    let mut depth = 0usize;
    let mut parts = name
        .split(|c| {
            match c {
                '[' => depth += 1,
                ']' => depth = depth.saturating_sub(1),
                _ => {}
            }
            c == ',' && depth == 0
        })
        .map(str::trim);
    match (parts.next(), parts.next()) {
        (Some(type_name), Some(assembly)) if !assembly.is_empty() => {
            format!("{type_name}, {assembly}")
        }
        (Some(type_name), _) => type_name.to_owned(),
        (None, _) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::canonical_type_name;

    #[test]
    fn a_type_name_has_one_canonical_form() {
        let short = "DemoGame.Target, DemoGame";
        assert_eq!(canonical_type_name(short), short);
        assert_eq!(
            canonical_type_name(
                "DemoGame.Target, DemoGame, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null"
            ),
            short
        );
        assert_eq!(canonical_type_name("  DemoGame.Target ,DemoGame "), short);
        assert_eq!(canonical_type_name("DemoGame.Target"), "DemoGame.Target");
        assert_eq!(
            canonical_type_name(
                "Game.Pool`1[[System.Int32, System.Private.CoreLib, Version=10.0.0.0]], Game, \
                 Version=1.0.0.0"
            ),
            "Game.Pool`1[[System.Int32, System.Private.CoreLib, Version=10.0.0.0]], Game"
        );
    }
}
