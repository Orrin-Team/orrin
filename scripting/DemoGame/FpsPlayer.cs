using Orrin;
using Orrin.Math;
using Quaternion = Orrin.Math.Quaternion;

namespace DemoGame;

// A first-person rigid body controller with the snappy, momentum-keeping feel
// of the classic indie FPS movement tutorials: a frictionless capsule pushed by
// forces, with the friction written by hand so the game decides what slows you
// down.
//
//   WASD     move          Space  jump (hold to hop on landing)
//   Ctrl/C   crouch-slide  mouse  look (click to lock, Esc to let go)
//
// What makes it feel the way it does:
//  - The capsule has no friction of its own. On the ground, any velocity you
//    are not asking for is taken away quickly ("counter-movement"), so you stop
//    dead when you let go and strafe on a dime, but nothing caps speed you got
//    some other way.
//  - Walking only pushes up to MaxSpeed. Anything above it — from a slide, a
//    ramp, a fall — is kept, and in the air is barely touched.
//  - Crouching while moving is a slide: a burst of speed, a little friction,
//    and gravity left on along the slope, so a slide downhill speeds up.
//  - Standing on a slope, gravity along it is cancelled, so you do not creep.
//
// The body is never rotated: its Transform is the physics step's, and writing
// a rotation every frame would teleport it each time. The view turns a camera
// of its own instead, placed at the eye every frame — and the BallLauncher, when
// there is one, rides on that camera, because an entity holds one behaviour.
[Component("demo.fps_player")]
public class FpsPlayer : Behaviour
{
    public float MaxSpeed = 10f;
    public float Acceleration = 70f;
    public float AirControl = 0.35f;
    public float CounterMovement = 12f;
    public float JumpSpeed = 6.5f;
    public float GravityScale = 1.6f;
    public float SlideBoost = 6f;
    public float SlideFriction = 4f;
    public float CrouchSpeed = 4f;
    public float MaxSlopeDegrees = 40f;
    public float Sensitivity = 0.12f;
    public float FieldOfView = 80f;
    public bool Launcher = true;

    public const uint Layer = 1;
    public const float Radius = 0.4f;
    public const float HalfHeight = 0.5f;
    const float CrouchScale = 0.5f;
    const float EyeAboveCentre = 0.6f;
    const float JumpCooldown = 0.25f;
    const float GroundProbe = 0.15f;

    [Transient] float _yaw;
    [Transient] float _pitch;
    [Transient] float _sinceJump = JumpCooldown;
    [Transient] bool _crouching;
    [Transient] Entity _camera;

    // The player a camera belongs to: what the launcher on a camera reads the
    // velocity it inherits from.
    public static Entity? OwnerOf(Entity camera)
    {
        foreach (var player in World.FindAll<FpsPlayer>())
        {
            if (player._camera == camera)
                return player.Entity;
        }
        return null;
    }

    Vector3 Gravity => new(0f, -9.81f * GravityScale, 0f);

    public override void OnStart()
    {
        World.SetTag(Entity, "player");
        World.AddCollider(Entity, new CapsuleCollider
        {
            Radius = Radius,
            HalfHeight = HalfHeight,
            Friction = 0f,
            Layer = Layer,
        });
        World.AddComponent(Entity, new Rigidbody
        {
            Mass = 70f,
            GravityScale = GravityScale,
            LinearDamping = 0f,
            AngularDamping = 0f,
            LockRotation = true,
            Continuous = true,
        });
        _camera = World.Spawn(Transform.Identity);
        World.AddCamera(_camera, new CameraComponent { FieldOfView = FieldOfView, Priority = 100 });
        if (Launcher)
            World.AddScript<BallLauncher>(_camera);
        PlaceCamera();
    }

    public override void OnUpdate(float deltaTime)
    {
        Look();
        _sinceJump += deltaTime;

        var transform = LocalTransform;
        var scale = transform.Scale.y;
        var below = HalfHeight * scale + Radius + GroundProbe;
        var grounded = Physics.Raycast(transform.Position, Vector3.down, out var ground, below, ~(1u << (int)Layer))
            && Vector3.Angle(ground.Normal, Vector3.up) <= MaxSlopeDegrees;
        var normal = grounded ? ground.Normal : Vector3.up;

        var velocity = Physics.GetVelocity(Entity);
        var flat = new Vector3(velocity.x, 0f, velocity.z);
        var facing = Quaternion.Euler(0f, _yaw, 0f);
        var wish = facing * Vector3.right * Axis(KeyCode.D, KeyCode.A)
            + facing * Vector3.forward * Axis(KeyCode.W, KeyCode.S);
        if (wish.magnitude > 1f)
            wish = wish.normalized;

        Crouch(ref transform, grounded, flat);
        var sliding = _crouching && grounded && flat.magnitude > CrouchSpeed;
        var push = Vector3.zero;

        if (sliding)
        {
            // Friction along the motion, and gravity left alone, so a slope
            // downhill wins against it.
            push -= flat.normalized * SlideFriction;
        }
        else
        {
            var top = _crouching ? CrouchSpeed : MaxSpeed;
            var control = grounded ? 1f : AirControl;
            if (wish.magnitude > 0f)
            {
                // Only up to the top speed along where you are asking to go, so
                // walking never overshoots and never takes away speed you had.
                var along = Vector3.Dot(flat, wish.normalized);
                var room = top * wish.magnitude - along;
                if (room > 0f)
                    push += wish.normalized * MathF.Min(Acceleration * control, room / deltaTime);
            }
            if (grounded)
            {
                // Everything you are not asking for goes, fast.
                var kept = wish.magnitude > 0f ? Vector3.Project(flat, wish.normalized) : Vector3.zero;
                if (Vector3.Dot(kept, wish) < 0f)
                    kept = Vector3.zero;
                push -= (flat - kept) * CounterMovement;
                // Stand on a slope without creeping down it.
                push -= Vector3.ProjectOnPlane(Gravity, normal);
            }
        }

        if (grounded && _sinceJump >= JumpCooldown && Input.GetKey(KeyCode.Space))
        {
            _sinceJump = 0f;
            Physics.SetVelocity(Entity, new Vector3(velocity.x, MathF.Max(velocity.y, 0f) + JumpSpeed, velocity.z));
        }
        else if (push.magnitude > 0f)
        {
            Physics.AddForce(Entity, push, ForceMode.Acceleration);
        }

        PlaceCamera();
    }

    void Look()
    {
        // The editor lets go on Escape by itself; a shipped game leaves it to
        // the game, and this is the game.
        if (Input.GetKeyDown(KeyCode.Escape))
            Input.cursorLocked = false;
        if (!Input.cursorLocked)
        {
            if (Input.GetMouseButtonDown(MouseButton.Left) && !UI.IsPointerOverUI)
                Input.cursorLocked = true;
            return;
        }
        var delta = Input.mouseDelta;
        // Right turns from -Z toward +X, which is a negative turn about +Y.
        _yaw -= delta.x * Sensitivity;
        _pitch = Math.Clamp(_pitch - delta.y * Sensitivity, -89f, 89f);
    }

    void Crouch(ref Transform transform, bool grounded, Vector3 flat)
    {
        var wants = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.C);
        if (wants == _crouching)
            return;
        _crouching = wants;
        // The capsule's half height shrinks with the scale and its radius does
        // not, so the centre drops by the difference to keep the feet planted.
        var drop = HalfHeight * (1f - CrouchScale);
        transform.Scale = new Vector3(1f, wants ? CrouchScale : 1f, 1f);
        transform.Position += Vector3.up * (wants ? -drop : drop);
        LocalTransform = transform;
        if (wants && grounded && flat.magnitude > CrouchSpeed)
            Physics.AddForce(Entity, flat.normalized * SlideBoost, ForceMode.VelocityChange);
    }

    void PlaceCamera()
    {
        var body = LocalTransform;
        var camera = Transform.Identity;
        camera.Position = body.Position + Vector3.up * (EyeAboveCentre * body.Scale.y);
        camera.Rotation = Quaternion.Euler(_pitch, _yaw, 0f);
        _camera.LocalTransform = camera;
    }

    static float Axis(KeyCode positive, KeyCode negative) =>
        (Input.GetKey(positive) ? 1f : 0f) - (Input.GetKey(negative) ? 1f : 0f);
}
