using Orrin;
using Orrin.Math;
using Quaternion = Orrin.Math.Quaternion;

namespace DemoGame;

// Registry-visible: `DegreesPerSecond` is saved with the scene and editable in
// the inspector, because the class carries a stable id.
[Component("demo.spinner")]
public class Spinner : Behaviour
{
    public float DegreesPerSecond = 90f;

    public override void OnStart() => Debug.Log($"Spinner attached to {Entity}");

    public override void OnUpdate(float deltaTime)
    {
        var t = LocalTransform;
        var radians = MathF.PI / 180f * DegreesPerSecond * deltaTime;
        t.Rotation =
            Quaternion.Normalize(Quaternion.Euler(0, radians,0));
        LocalTransform = t;
    }
}
