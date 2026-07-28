namespace Orrin.Tests;

// Test-only behaviours driven by the Rust integration test
// (crates/orrin-script/tests/lifecycle.rs), which swaps the engine's log
// callback for a capture buffer and asserts on hook ordering. They ship in
// Orrin.dll because the host can only instantiate types from the assembly it
// loads; they are never attached by the engine itself.

/// Logs every lifecycle hook with a stable `probe:` prefix.
class LifecycleProbe : Behaviour
{
    public LifecycleProbe() => Native.Log("probe:ctor");
    public override void OnEnable() => Native.Log("probe:OnEnable");
    public override void OnStart() => Native.Log("probe:OnStart");
    public override void OnUpdate(float deltaTime) => Native.Log("probe:OnUpdate");
    public override void OnDisable() => Native.Log("probe:OnDisable");
    public override void OnDestroy() => Native.Log("probe:OnDestroy");
}

/// Exercises the `Create` guard: a user constructor that throws must yield a
/// 0 handle, not a process abort.
class ThrowingConstructor : Behaviour
{
    public ThrowingConstructor() => throw new InvalidOperationException("ctor boom");
}

/// Exercises the `Destroy` guard: a throwing OnDestroy must be logged and the
/// GCHandle still freed.
class ThrowingDestroy : Behaviour
{
    public override void OnDestroy() => throw new InvalidOperationException("destroy boom");
}

/// Exercises the fault channel: a throwing OnUpdate must be caught, logged, and
/// reported back (Update returns a nonzero fault byte) so the engine disables
/// the script — never an abort, never silent spamming every frame.
class ThrowingUpdate : Behaviour
{
    public override void OnUpdate(float deltaTime) =>
        throw new InvalidOperationException("update boom");
}

/// Exercises `BehaviourState` capture/apply across a reload: one field of each
/// shape the closed type set is meant to cover, plus a `[Transient]` one that
/// must *not* survive and a readonly one that cannot be written back. The test
/// creates it, mutates it by ticking, captures, destroys, re-creates, applies,
/// and asserts the logged line.
///
/// It lives in the bindings assembly like the other probes — the field walk
/// stops at `Behaviour`, not at an assembly boundary, so engine-owned lifecycle
/// state is excluded without making in-assembly types untestable.
class StatefulProbe : Behaviour
{
    public int Counter;
    public string Label = "fresh";
    public Math.Vector3 Offset;

    [Transient]
    public int Scratch;

    public readonly int Fixed = 7;

    public override void OnUpdate(float deltaTime)
    {
        Counter++;
        Scratch++;
        Label = $"tick{Counter}";
        Offset = new Math.Vector3(Counter, 0f, 0f);
        Native.Log(
            $"probe:state counter={Counter} label={Label} offset={Offset.x} "
            + $"scratch={Scratch} fixed={Fixed}");
    }
}
