/// Links an entity to its C# Behaviour via a .NET `GCHandle` (stored as `u64`).
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
/// per-frame log spam — so one bad script can't take the engine down. It stays
/// set until something clears it (the inspector's "Clear fault" button, or a
/// hot reload, which re-creates the behaviour from the new code), which re-arms
/// the script from wherever its lifecycle left off.
///
/// `type_name` is the assembly-qualified name the behaviour was created from.
/// It is kept on the Rust side because a hot reload has to re-instantiate the
/// script *after* the managed object is gone: by the time the new assembly is
/// live, the only surviving record of what used to be attached here is this
/// string.
#[derive(Debug)]
pub struct ScriptComponent {
    pub handle: u64,
    pub type_name: String,
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
