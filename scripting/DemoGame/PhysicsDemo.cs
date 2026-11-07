using Orrin;
using Orrin.Math;

namespace DemoGame;

// Physics demo entry, over the physics testbed — run with:
//   ORRIN_SCENE=physics ORRIN_ENTRY="DemoGame.PhysicsDemo, DemoGame" \
//     cargo run -p orrin-core
//
// Space throws a ball from the camera. E shoves whatever the camera is looking
// at. Every ball carries a Thud, which logs what it lands on.
public class PhysicsDemo : Behaviour
{
    public float ThrowSpeed = 18f;
    public float Shove = 40f;

    public override void OnStart() =>
        Native.Log("physics demo: Space throws a ball, E shoves what the camera looks at");

    public override void OnUpdate(float deltaTime)
    {
        if (Input.GetKeyDown(KeyCode.Space))
            Throw();
        if (Input.GetKeyDown(KeyCode.E))
            ShoveWhatIsAhead();
    }

    private void Throw()
    {
        var transform = Transform.Identity;
        transform.Position = Camera.position + Camera.forward;
        transform.Scale = new Vector3(0.5f, 0.5f, 0.5f);
        var ball = World.SpawnRenderable("sphere", "copper", transform);
        World.AddCollider(ball, new SphereCollider { Restitution = 0.4f });
        World.AddComponent(ball, new Rigidbody { Mass = 2f, Continuous = true });
        World.AddScript<Thud>(ball);
        // Spawned this tick, so the body does not exist yet; the push waits for it.
        Physics.SetVelocity(ball, Camera.forward * ThrowSpeed);
    }

    private void ShoveWhatIsAhead()
    {
        if (!Physics.Raycast(Camera.position, Camera.forward, out var hit, 50f))
        {
            Native.Log("nothing ahead");
            return;
        }
        var shoved = Physics.AddForce(hit.Entity, Camera.forward * Shove, ForceMode.Impulse);
        Native.Log(shoved ? $"shoved {hit.Entity} at {hit.Distance:F1} m" : $"{hit.Entity} has no rigid body to shove");
    }
}

public class Thud : Behaviour
{
    public override void OnCollisionEnter(Collision collision)
    {
        var speed = Physics.GetVelocity(Entity).magnitude;
        if (speed > 2f)
            Native.Log($"{Entity} hit {collision.Other} at {speed:F1} m/s");
    }
}
