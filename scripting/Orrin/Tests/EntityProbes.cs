namespace Orrin.Tests;

// Test-only behaviours for the engine's own script tick, driven by
// crates/core/tests/several_behaviours.rs and crates/core/benches/scripts.rs.
// They ship in Orrin.dll for the reason the lifecycle probes do.

/// Logs each hook as `{type}:{hook}`, so two probes on one entity can be told
/// apart and their order read back from the log.
abstract class OrderProbe : Behaviour
{
    protected OrderProbe() => Log("ctor");
    protected void Log(string hook) => Native.Log($"{GetType().Name}:{hook}");
    public override void OnEnable() => Log("OnEnable");
    public override void OnStart() => Log("OnStart");
    public override void OnUpdate(float deltaTime) => Log("OnUpdate");
    public override void OnDisable() => Log("OnDisable");
    public override void OnDestroy() => Log("OnDestroy");
    public override void OnCollisionEnter(Collision collision) => Log("OnCollisionEnter");
    public override void OnTriggerEnter(Collision collision) => Log("OnTriggerEnter");
}

class FirstProbe : OrderProbe;

class SecondProbe : OrderProbe;

/// Throws from OnUpdate, so the probe beside it shows whether a fault stays
/// with the Behaviour that raised it.
class FaultingProbe : OrderProbe
{
    public override void OnUpdate(float deltaTime)
    {
        Log("OnUpdate");
        throw new InvalidOperationException("update boom");
    }
}

/// Does nothing at all: what the scripts bench ticks, so its number is the
/// engine's dispatch rather than a script's work.
class IdleProbe : Behaviour;

/// Overrides the per-step hooks with empty bodies: one of these in a world is
/// what makes the scripts bench's physics phase sweep for OnFixedUpdate and
/// report stays.
class SteppedIdleProbe : Behaviour
{
    public override void OnFixedUpdate(float fixedDeltaTime) { }
    public override void OnCollisionStay(Collision other) { }
}
