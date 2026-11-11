using System.Collections;

using Orrin;
using Orrin.Math;
using Orrin.Testing;

namespace DemoGame;

// The demo's own tests, run by `orrin test` or `cargo run -p orrin-core --
// --test` from the repository root. Each starts in an empty world, playing,
// with frames of 1/60 s.
public static class GameTests
{
    [Test]
    public static IEnumerator ACrateFallsAndComesToRestOnTheFloor()
    {
        var floor = Transform.Identity;
        floor.Position = new Vector3(0f, -0.5f, 0f);
        World.AddCollider(World.Spawn(floor), new BoxCollider { HalfExtents = new Vector3(20f, 0.5f, 20f) });

        var start = Transform.Identity;
        start.Position = new Vector3(0f, 5f, 0f);
        var crate = World.Spawn(start);
        World.AddCollider(crate, new BoxCollider());
        World.AddComponent(crate, new Rigidbody());

        yield return Wait.Seconds(0.5f);
        Assert.That(crate.LocalTransform.Position.y < 4f, $"still at {crate.LocalTransform.Position.y} after half a second");

        yield return Wait.Seconds(3f);
        // Within the couple of centimetres a resting contact is allowed to sink.
        Assert.AreApproximatelyEqual(0.5f, crate.LocalTransform.Position.y, 0.05f, "resting on the floor");
        Assert.AreApproximatelyEqual(Vector3.zero, Physics.GetVelocity(crate), 0.05f, "at rest");
    }

    /// A Target counts a ball that hits it, and nothing else.
    [Test]
    public static IEnumerator ATargetCountsTheBallThatHitsIt()
    {
        var target = World.Spawn(Transform.Identity);
        World.AddCollider(target, new BoxCollider());
        target.AddScript<Target>();

        var crateAt = Transform.Identity;
        crateAt.Position = new Vector3(3f, 0f, 0f);
        var crate = World.Spawn(crateAt);
        World.AddCollider(crate, new BoxCollider());
        World.AddComponent(crate, new Rigidbody { GravityScale = 0f });
        Physics.SetVelocity(crate, new Vector3(-6f, 0f, 0f));

        var ballAt = Transform.Identity;
        ballAt.Position = new Vector3(-3f, 0f, 0f);
        var ball = World.Spawn(ballAt);
        ball.SetTag("ball");
        World.AddCollider(ball, new SphereCollider());
        World.AddComponent(ball, new Rigidbody { GravityScale = 0f });
        Physics.SetVelocity(ball, new Vector3(6f, 0f, 0f));

        yield return null;
        var counter = target.GetBehaviour<Target>();
        Assert.That(counter is not null, "the Target was attached");
        Assert.AreEqual("target", target.Tag, "a Target tags itself when it starts");
        Assert.AreEqual(0, counter!.Hits);

        yield return Wait.Seconds(1f);
        Assert.AreEqual(1, counter.Hits, "one ball, one hit; the crate does not count");
    }
}
