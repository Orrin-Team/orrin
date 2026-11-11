using Orrin.Math;

namespace Orrin.Tests;

// Test-only behaviours for what a script reads of the frame's clock, the wheel
// and the keys, and what the shape queries answer, driven by
// crates/core/tests/time_and_queries.rs. They ship in Orrin.dll for the reason
// the lifecycle probes do.

/// Halves the time scale on start and zeroes it when F5 goes down. Logs what
/// the clock, the wheel and F5 read in OnUpdate and the clock in
/// OnFixedUpdate, and when a half-second timer started on start fires.
class ClockProbe : Behaviour
{
    public override void OnStart()
    {
        Time.timeScale = 0.5f;
        After(0.5f, () => Native.Log("ClockProbe:Timer"));
    }

    public override void OnUpdate(float deltaTime)
    {
        if (Input.GetKeyDown(KeyCode.F5))
            Time.timeScale = 0f;
        var wheel = Input.scrollDelta;
        Native.Log(
            $"ClockProbe:OnUpdate {deltaTime:R} {Time.DeltaTime:R} {Time.unscaledDeltaTime:R} "
            + $"{Time.fixedDeltaTime:R} {Time.timeScale:R} {wheel.x:R} {wheel.y:R} "
            + $"{(Input.GetKey(KeyCode.F5) ? 1 : 0)}");
    }

    public override void OnFixedUpdate(float fixedDeltaTime) =>
        Native.Log($"ClockProbe:OnFixedUpdate {fixedDeltaTime:R} {Time.DeltaTime:R} {Time.unscaledDeltaTime:R}");
}

/// Asks every shape query of the scene the test builds — a wall down -z, a
/// trigger down +x, a crate down -x — and logs the answers as entity indices
/// and hit numbers.
class QueryProbe : Behaviour
{
    readonly Entity[] _one = new Entity[1];

    static string Indices(Entity[] entities) => string.Join(" ", entities.Select(e => e.Index));

    static string Hit(bool found, RaycastHit hit) =>
        $"{(found ? 1 : 0)} {hit.Entity.Index} {hit.Distance:R} {hit.Normal.x:R} {hit.Normal.y:R} {hit.Normal.z:R}";

    public override void OnUpdate(float deltaTime)
    {
        int written = Physics.OverlapSphere(Vector3.zero, 6f, _one, hitTriggers: true);
        long before = GC.GetAllocatedBytesForCurrentThread();
        Physics.OverlapSphere(Vector3.zero, 6f, _one, hitTriggers: true);
        Physics.OverlapBox(Vector3.zero, Vector3.one, _one);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Native.Log($"QueryProbe:OverlapInto {written} {_one[0].Index} {allocated}");

        Native.Log($"QueryProbe:OverlapSphere {Indices(Physics.OverlapSphere(Vector3.zero, 6f, hitTriggers: true))}");
        Native.Log($"QueryProbe:OverlapSphereSolid {Indices(Physics.OverlapSphere(Vector3.zero, 6f))}");

        var thin = new Vector3(3f, 0.1f, 0.1f);
        var ahead = new Vector3(0f, 0f, -2f);
        Native.Log($"QueryProbe:OverlapBoxTurned {Indices(Physics.OverlapBox(ahead, thin, Quaternion.Euler(0f, 90f, 0f)))}");
        Native.Log($"QueryProbe:OverlapBoxUnturned {Indices(Physics.OverlapBox(ahead, thin))}");

        var forward = new Vector3(0f, 0f, -1f);
        Native.Log($"QueryProbe:SphereCast {Hit(Physics.SphereCast(Vector3.zero, 0.5f, forward, out var sphere), sphere)}");
        Native.Log($"QueryProbe:BoxCast {Hit(Physics.BoxCast(Vector3.zero, 0.5f * Vector3.one, forward, out var box), box)}");
        Native.Log($"QueryProbe:SphereCastShort {Hit(Physics.SphereCast(Vector3.zero, 0.5f, forward, out var shortHit, 3.9f), shortHit)}");

        var right = new Vector3(1f, 0f, 0f);
        Native.Log($"QueryProbe:CastPastTrigger {Hit(Physics.SphereCast(Vector3.zero, 0.5f, right, out var past), past)}");
        Native.Log($"QueryProbe:CastAtTrigger {Hit(Physics.SphereCast(Vector3.zero, 0.5f, right, out var at, hitTriggers: true), at)}");
        Native.Log($"QueryProbe:CastMasked {Hit(Physics.SphereCast(Vector3.zero, 0.5f, forward, out var masked, layers: ~(1u << 2)), masked)}");
    }
}
