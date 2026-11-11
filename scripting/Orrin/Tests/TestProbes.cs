using System.Collections;

using Orrin.Math;
using Orrin.Testing;

namespace Orrin.Tests;

// Tests for the test runner, driven by crates/core/tests/test_runner.rs, which
// lists and runs the bindings' own tests rather than a game's. They ship in
// Orrin.dll for the reason the lifecycle probes do. Several fail on purpose;
// the Rust test asserts how each one fails.

/// Counts its own lifecycle, so a test can see the frames it yielded run.
class CountingProbe : Behaviour
{
    public int Starts;
    public int Updates;

    public override void OnStart() => Starts++;
    public override void OnUpdate(float deltaTime) => Updates++;
}

static class TestProbes
{
    static Entity _spawnedByAnEarlierTest;

    [Test]
    public static void SyncPasses()
    {
        var entity = World.Spawn(Transform.Identity);
        Assert.IsTrue(entity.IsValid);
        Assert.AreEqual(0UL, Time.FrameCount, "a test starts on a world no frame has run in");
    }

    /// A Behaviour starts on the first frame, a body falls under gravity, and
    /// the frames waited for are exactly the frames that ran.
    [Test]
    public static IEnumerator CoroutineRunsFrames()
    {
        var start = Transform.Identity;
        start.Position = new Vector3(0f, 10f, 0f);
        var body = World.Spawn(start);
        World.AddCollider(body, new SphereCollider());
        World.AddComponent(body, new Rigidbody());
        body.AddScript<CountingProbe>();
        yield return null;

        var probe = body.GetBehaviour<CountingProbe>();
        Assert.That(probe is not null, "the probe was attached");
        Assert.AreEqual(1, probe!.Starts);
        Assert.AreEqual(1, probe.Updates);
        Assert.AreApproximatelyEqual(1f / 60f, Time.DeltaTime, 1e-7f, "the runner's fixed frame");

        yield return Wait.Frames(30);
        Assert.AreEqual(31, probe.Updates);
        Assert.AreEqual(1, probe.Starts);
        Assert.That(body.LocalTransform.Position.y < 9f, $"the body fell, to {body.LocalTransform.Position.y}");

        yield return Wait.Seconds(0.5f);
        Assert.AreEqual(61, probe.Updates);
        Assert.AreEqual(61UL, Time.FrameCount);
    }

    [Test]
    public static void FailsByAssert() => Assert.AreEqual(1, 2);

    [Test]
    public static void FailsByThrowing() => throw new InvalidOperationException("thrown on purpose");

    [Test]
    public static IEnumerator FailsByAFaultedBehaviour()
    {
        World.Spawn(Transform.Identity).AddScript<ThrowingUpdate>();
        yield return Wait.Frames(5);
        Assert.Fail("resumed past a faulted Behaviour");
    }

    [Test(MaxFrames = 20)]
    public static IEnumerator RunsOutOfFrames()
    {
        while (true)
            yield return null;
    }

    [Test]
    public static int NotATest() => 0;

    [Test]
    public static void IsolationFirstSpawns()
    {
        _spawnedByAnEarlierTest = World.Spawn(Transform.Identity);
        _spawnedByAnEarlierTest.SetTag("isolation");
        _spawnedByAnEarlierTest.AddScript<CountingProbe>();
    }

    [Test]
    public static IEnumerator IsolationSecondSeesNothing()
    {
        yield return null;
        Assert.IsTrue(World.FindByTag("isolation") is null, "the earlier test's entity is still tagged");
        Assert.AreEqual(0, World.FindAll<CountingProbe>().Length, "the earlier test's Behaviours are alive");
        Assert.IsFalse(_spawnedByAnEarlierTest.HasComponent<Transform>(), "the earlier test's entity is alive");
    }

    [Test(Scene = "probe-scenes/tagged.orrin")]
    public static void SceneIsLoaded() =>
        Assert.IsTrue(World.FindByTag("from-scene") is not null, "the scene's entity is missing");

    [Test(Scene = "probe-scenes/missing.orrin")]
    public static void SceneIsMissing() => Assert.Fail("ran without its scene");
}
