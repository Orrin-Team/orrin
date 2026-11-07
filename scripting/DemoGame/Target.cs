using Orrin;

namespace DemoGame;

// Counts the balls that hit it and flashes when one does. Anything else
// bumping into it — a crate, the player — does not count.
[Component("demo.target")]
public class Target : Behaviour
{
    public int Hits;
    public string Material = "gold";
    public string Flash = "neon";

    [Transient] float _flashing;

    public static int TotalHits { get; private set; }

    public override void OnStart() => World.SetTag(Entity, "target");

    public override void OnCollisionEnter(Collision collision)
    {
        if (Native.GetTag(collision.Other) != "ball")
            return;
        Hits++;
        TotalHits++;
        World.SetMaterial(Entity, Flash);
        _flashing = 0.2f;
    }

    public override void OnUpdate(float deltaTime)
    {
        if (_flashing <= 0f)
            return;
        _flashing -= deltaTime;
        if (_flashing <= 0f)
            World.SetMaterial(Entity, Material);
    }
}

// Slides a plain collider back and forth: a moving target, and a kinematic
// body by the physics module's rules, since something moves it.
[Component("demo.shuttle")]
public class Shuttle : Behaviour
{
    public float Distance = 6f;
    public float Period = 4f;

    [Transient] float _time;
    [Transient] float _home = float.NaN;

    public override void OnUpdate(float deltaTime)
    {
        var transform = LocalTransform;
        if (float.IsNaN(_home))
            _home = transform.Position.x;
        _time += deltaTime;
        var offset = MathF.Sin(_time * MathF.Tau / Period) * Distance * 0.5f;
        transform.Position = new Orrin.Math.Vector3(_home + offset, transform.Position.y, transform.Position.z);
        LocalTransform = transform;
    }
}
