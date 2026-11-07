using Orrin;
using Orrin.Math;

namespace DemoGame;

// Left click fires a ball along the camera it rides on, carrying the velocity of
// the player that camera belongs to — so a ball fired while sliding goes
// further. The oldest balls are cleared past MaxBalls.
[Component("demo.ball_launcher")]
public class BallLauncher : Behaviour
{
    public float Speed = 40f;
    public float Radius = 0.15f;
    public float Mass = 0.5f;
    public int MaxBalls = 40;

    // The click that locks the cursor is not a shot: firing waits for a frame
    // that began locked.
    [Transient] bool _lockedLastFrame;
    [Transient] readonly Queue<Entity> _balls = new();

    public static int Fired { get; private set; }

    public override void OnUpdate(float deltaTime)
    {
        if (_lockedLastFrame && Input.GetMouseButtonDown(MouseButton.Left))
            Fire();
        _lockedLastFrame = Input.cursorLocked;
    }

    void Fire()
    {
        var view = LocalTransform;
        var eye = view.Position;
        var forward = view.Rotation * Vector3.forward;
        var owner = FpsPlayer.OwnerOf(Entity);
        var carried = owner is { } player ? Physics.GetVelocity(player) : Vector3.zero;
        var transform = Transform.Identity;
        // Clear of the player's own capsule.
        transform.Position = eye + forward * (FpsPlayer.Radius + Radius + 0.3f);
        transform.Scale = Vector3.one * (Radius * 2f);
        var ball = World.SpawnRenderable("sphere", "copper", transform);
        World.SetTag(ball, "ball");
        World.AddCollider(ball, new SphereCollider { Restitution = 0.35f, Friction = 0.4f });
        World.AddComponent(ball, new Rigidbody { Mass = Mass, Continuous = true });
        // The body does not exist until the next physics step; the velocity
        // waits for it.
        Physics.SetVelocity(ball, forward * Speed + carried);
        Fired++;

        _balls.Enqueue(ball);
        while (_balls.Count > MaxBalls)
            World.Despawn(_balls.Dequeue());
    }
}
