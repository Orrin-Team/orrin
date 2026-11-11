using System.Collections;

using Orrin;
using Orrin.Math;
using Orrin.Testing;

namespace DemoGame;

// The showcase's parts, held to numbers. Run with the demo's other tests:
//   cargo run -p orrin-core -- --test Showcase
public static class ShowcaseTests
{
    static Entity Ball(Vector3 at, float gravity = 1f)
    {
        var transform = Transform.Identity;
        transform.Position = at;
        var ball = World.Spawn(transform);
        World.AddCollider(ball, new SphereCollider());
        World.AddComponent(ball, new Rigidbody { Mass = 2f, GravityScale = gravity });
        return ball;
    }

    /// A thruster pushes from OnFixedUpdate, so its ball climbs against gravity
    /// while one that is not firing falls.
    [Test]
    public static IEnumerator AThrusterLiftsItsBallOnlyWhileItFires()
    {
        var firing = Ball(new Vector3(0f, 10f, 0f));
        var idle = Ball(new Vector3(5f, 10f, 0f));
        firing.AddScript<Thruster>();
        idle.AddScript<Thruster>();
        yield return null;

        // 40 N on 2 kg is 20 m/s² up against 9.81 down.
        firing.GetBehaviour<Thruster>()!.Lift = 40f;
        yield return Wait.Seconds(1f);

        Assert.That(firing.LocalTransform.Position.y > 13f, $"the firing ball is at {firing.LocalTransform.Position.y}");
        Assert.That(idle.LocalTransform.Position.y < 7f, $"the idle ball is at {idle.LocalTransform.Position.y}");
    }

    /// An impact meter reads the closing speed off the contact, and counts the
    /// steps its ball goes on touching.
    [Test]
    public static IEnumerator AnImpactMeterReadsHowHardItsBallLanded()
    {
        var wallAt = Transform.Identity;
        wallAt.Position = new Vector3(4f, 0f, 0f);
        World.AddCollider(World.Spawn(wallAt), new BoxCollider { HalfExtents = new Vector3(0.5f, 3f, 3f) });

        var ball = Ball(Vector3.zero, gravity: 0f);
        ball.AddScript<ImpactMeter>();
        Physics.SetVelocity(ball, new Vector3(6f, 0f, 0f));
        yield return Wait.Seconds(1f);

        var meter = ball.GetBehaviour<ImpactMeter>()!;
        Assert.AreEqual(1, meter.Impacts, "one wall, one impact");
        Assert.AreApproximatelyEqual(6f, meter.Hardest, 0.5f, "the closing speed");
        Assert.That(meter.StepsTouching > 0, "it stayed against the wall for some steps");
    }

    /// The beam finds balls with an overlap query and fires only those.
    [Test]
    public static IEnumerator TheBeamFiresTheThrustersInsideIt()
    {
        var host = World.Spawn(Transform.Identity);
        host.AddScript<Showcase>();
        yield return null;
        var showcase = host.GetBehaviour<Showcase>()!;

        var near = showcase.Throw(new Vector3(0f, 5f, 0f), Vector3.zero);
        var far = showcase.Throw(new Vector3(30f, 5f, 0f), Vector3.zero);
        Assert.AreEqual(2, showcase.Balls.Count);
        Assert.AreEqual(far, showcase.LastBall);
        // A query sees a body from the step after the frame that spawned it.
        yield return Wait.Frames(3);

        Assert.AreEqual(1, showcase.Beam(near.LocalTransform.Position), "one ball inside the beam");
        Assert.AreEqual(showcase.Lift, near.GetBehaviour<Thruster>()!.Lift, "the near ball fires");
        Assert.AreEqual(0f, far.GetBehaviour<Thruster>()!.Lift, "the far ball does not");

        showcase.Release();
        Assert.AreEqual(0f, near.GetBehaviour<Thruster>()!.Lift, "released");
    }

    /// The time controls scale the game's delta, and a pause is a scale of zero
    /// that OnUpdate still runs through.
    [Test]
    public static IEnumerator SlowMotionAndPauseScaleTheGamesTime()
    {
        var host = World.Spawn(Transform.Identity);
        host.AddScript<Showcase>();
        yield return null;
        var showcase = host.GetBehaviour<Showcase>()!;

        showcase.ToggleSlowMotion();
        yield return null;
        Assert.AreApproximatelyEqual(showcase.SlowMotion, Time.timeScale);
        Assert.AreApproximatelyEqual(Time.unscaledDeltaTime * showcase.SlowMotion, Time.DeltaTime, 1e-6f);

        showcase.ToggleSlowMotion();
        yield return null;
        Assert.AreApproximatelyEqual(1f, Time.timeScale);

        showcase.TogglePause();
        yield return null;
        Assert.AreApproximatelyEqual(0f, Time.DeltaTime, 0f, "paused");
        Assert.That(Time.unscaledDeltaTime > 0f, "real time still passes");

        showcase.TogglePause();
        yield return null;
        Assert.AreApproximatelyEqual(1f, Time.timeScale);
    }
}
