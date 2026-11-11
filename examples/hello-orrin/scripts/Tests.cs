using System.Collections;

using Orrin;
using Orrin.Math;
using Orrin.Testing;

namespace HelloOrrin;

// Run with `orrin test` from this directory. Each [Test] starts in an empty
// world that is playing, and `yield return null` runs one frame of 1/60 s.
public static class Tests
{
    [Test]
    public static IEnumerator MainSpawnsTheCubeAndSpinsIt()
    {
        var entry = World.Spawn(Transform.Identity);
        entry.AddScript<Main>();

        // Its first frame: Main is created and its OnStart spawns the cube.
        yield return null;
        var main = entry.GetBehaviour<Main>();
        Assert.That(main is not null, "Main was attached");
        Assert.IsTrue(main!.Cube.IsValid, "OnStart spawned the cube");

        // A second in all, so the cube has turned SpinSpeed degrees.
        yield return Wait.Frames(59);
        var cube = main.Cube.LocalTransform;
        Assert.AreApproximatelyEqual(new Vector3(0f, main.Height, 0f), cube.Position);
        var expected = Quaternion.Euler(0f, main.SpinSpeed, 0f) * Vector3.forward;
        Assert.AreApproximatelyEqual(expected, cube.Rotation * Vector3.forward, 1e-3f, "facing after one second");
    }
}
