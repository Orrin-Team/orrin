using Ferron;
using Ferron.Math;

namespace DemoGame;

// Collision demo entry — run with:
//   FERRON_ENTRY="DemoGame.CollisionDemo, DemoGame" \
//     cargo run -p renderer-prototype --features scripting
//
// Spawns a cube and a sphere gliding toward each other; each carries a Bumper
// that logs the collision, stops, and swaps to the neon material on contact.
public class CollisionDemo : Behaviour
{
    public override void OnStart()
    {
        Spawn("cube", new BoxCollider(), new Vector3(-4f, 2f, 0f));
        Spawn("sphere", new SphereCollider(), new Vector3(4f, 2f, 0f));
        Native.Log("collision demo: two bodies closing in");
    }

    private static void Spawn(string mesh, Collider collider, Vector3 position)
    {
        var transform = Transform.Identity;
        transform.Position = position;
        var entity = World.SpawnRenderable(mesh, "gold", transform);
        World.AddCollider(entity, collider);
        World.AddScript<Bumper>(entity);
    }
}

// Glides toward the world origin until something is hit, then stops and turns
// neon; separating (e.g. after the MTV push-out) logs the exit.
public class Bumper : Behaviour
{
    public float Speed = 1.5f;

    private bool _stopped;

    public override void OnUpdate(float deltaTime)
    {
        if (_stopped)
            return;

        var transform = Transform;
        var toOrigin = -transform.Position;
        if (toOrigin.magnitude < 0.05f)
            return;

        transform.Position += toOrigin.normalized * (Speed * deltaTime);
        Transform = transform;
    }

    public override void OnCollisionEnter(Collision collision)
    {
        _stopped = true;
        Native.Log($"{Entity} hit {collision.Other} at {collision.ContactPoint} (normal {collision.Normal})");
        World.SetMaterial(Entity, "neon");
    }

    public override void OnCollisionExit(Collision collision)
    {
        Native.Log($"{Entity} separated from {collision.Other}");
    }
}
