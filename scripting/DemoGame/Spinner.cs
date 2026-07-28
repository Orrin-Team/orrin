using Orrin;
using Orrin.Math;
using Quaternion = Orrin.Math.Quaternion;

namespace DemoGame;

public class Spinner : Behaviour
{
    public float DegreesPerSecond = 90f;

    public override void OnStart() => Native.Log($"Spinner attached to {Entity}");

    public override void OnUpdate(float deltaTime)
    {
        var t = Transform;
        var radians = MathF.PI / 180f * DegreesPerSecond * deltaTime;
        t.Rotation =
            Quaternion.Normalize(Quaternion.Euler(0, radians,0));
        Transform = t;
    }
}
