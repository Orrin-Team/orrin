using System.Diagnostics;

using Orrin.Math;

namespace Orrin.Tests;

// Test-only behaviours for what a Behaviour hears from the physics step,
// driven by crates/core/tests/fixed_steps.rs. They ship in Orrin.dll for the
// reason the lifecycle probes do.

/// Logs OnFixedUpdate with the step it was told and the x its body's transform
/// reads there, and each other hook it overrides as `{type}:{hook}`.
abstract class SteppingProbe : Behaviour
{
    protected void Log(string line) => Native.Log($"{GetType().Name}:{line}");
    public override void OnStart() => Log("OnStart");
    public override void OnUpdate(float deltaTime) => Log("OnUpdate");
    public override void OnFixedUpdate(float fixedDeltaTime) =>
        Log($"OnFixedUpdate {fixedDeltaTime:R} {LocalTransform.Position.x:R}");
}

class SteppedProbe : SteppingProbe;

class SecondSteppedProbe : SteppingProbe;

/// Throws from OnFixedUpdate, after saying so.
class ThrowingStepProbe : SteppingProbe
{
    public override void OnFixedUpdate(float fixedDeltaTime)
    {
        Log("OnFixedUpdate");
        throw new InvalidOperationException("fixed boom");
    }
}

/// Pushes its body along +x with 2 N before every step.
class PushingProbe : Behaviour
{
    public override void OnFixedUpdate(float fixedDeltaTime)
    {
        Native.Log("PushingProbe:OnFixedUpdate");
        Physics.AddForce(Entity, new Vector3(2f, 0f, 0f));
    }
}

/// Logs every contact callback as
/// `{type}:{hook} {self} {other} {relative velocity xyz} {normal xyz}`.
class ContactProbe : Behaviour
{
    void Log(string hook, Collision c) => Native.Log(
        $"{GetType().Name}:{hook} {Entity.Index} {c.Other.Index} "
        + $"{c.RelativeVelocity.x:R} {c.RelativeVelocity.y:R} {c.RelativeVelocity.z:R} "
        + $"{c.Normal.x:R} {c.Normal.y:R} {c.Normal.z:R}");

    public override void OnUpdate(float deltaTime) => Native.Log($"{GetType().Name}:OnUpdate");
    public override void OnCollisionEnter(Collision c) => Log("OnCollisionEnter", c);
    public override void OnCollisionStay(Collision c) => Log("OnCollisionStay", c);
    public override void OnCollisionExit(Collision c) => Log("OnCollisionExit", c);
    public override void OnTriggerEnter(Collision c) => Log("OnTriggerEnter", c);
    public override void OnTriggerStay(Collision c) => Log("OnTriggerStay", c);
    public override void OnTriggerExit(Collision c) => Log("OnTriggerExit", c);
}

/// Throws from OnCollisionStay, after saying so.
class ThrowingStayProbe : Behaviour
{
    public override void OnCollisionStay(Collision c)
    {
        Native.Log("ThrowingStayProbe:OnCollisionStay");
        throw new InvalidOperationException("stay boom");
    }
}

/// Spends two milliseconds in every OnUpdate, so per-type timing has something
/// to attribute.
class SlowProbe : Behaviour
{
    public override void OnUpdate(float deltaTime)
    {
        var clock = Stopwatch.StartNew();
        while (clock.Elapsed.TotalMilliseconds < 2.0) { }
    }
}
