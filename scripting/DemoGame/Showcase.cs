using Orrin;
using Orrin.Math;

namespace DemoGame;

// A tour of the scripting API in one entry Behaviour, over the physics
// testbed — run with:
//   ORRIN_SCENE=physics ORRIN_ENTRY="DemoGame.Showcase, DemoGame" \
//     cargo run -p orrin-core
// then press Play.
//
//   Space    throw a ball from the camera
//   F (hold) the beam: every ball near the ring ahead of the camera fires its
//            thruster and climbs
//   wheel    more or less lift
//   [ and ]  a smaller or larger beam
//   E        sweep a sphere ahead and shove the first thing it touches
//   F2       slow motion on and off
//   F3       pause on and off — the camera, the keys and this HUD keep running
//
// What each part is there to show:
//   - Every ball carries two Behaviours, a Thruster and an ImpactMeter, and the
//     showcase reaches them with GetBehaviour.
//   - Thruster pushes from OnFixedUpdate, so the lift is the same at any frame
//     rate and in slow motion.
//   - ImpactMeter reads Collision.RelativeVelocity on enter and counts
//     OnCollisionStay.
//   - The beam is Physics.OverlapSphere into a stack buffer; the shove is
//     Physics.SphereCast.
//   - Time.timeScale drives slow motion and pause; Time.unscaledDeltaTime is
//     what still moves while paused.
//   - After despawns each ball, Every refreshes the HUD's numbers.
//   - Select the entity carrying this in the editor while it plays: the
//     fields below are sliders, a colour, a screen position, an entity slot
//     and a list, under their headers.
public class Showcase : Behaviour
{
    [Header("Throwing")]
    [Range(4f, 40f), Tooltip("How fast a ball leaves the camera, in m/s")]
    public float ThrowSpeed = 14f;
    [Tooltip("Seconds before a thrown ball is despawned")]
    public float BallLife = 30f;
    [Range(5f, 120f), Tooltip("The impulse E gives what the sweep touches, in N·s")]
    public float Shove = 40f;

    [Header("Beam")]
    [Range(0f, MaxLift), Tooltip("Newtons up on each ball in the beam. A ball weighs about 20.")]
    public float Lift = 30f;
    [Range(MinRadius, MaxRadius), Tooltip("Metres")]
    public float Radius = 2.5f;
    [Tooltip("How far ahead of the camera the beam is centred, in metres")]
    public float Reach = 6f;
    public Color BeamColor = Color.cyan;

    [Header("Time")]
    [Range(0.05f, 1f), Tooltip("The time scale F2 switches to")]
    public float SlowMotion = 0.25f;

    [Header("Live")]
    [Tooltip("The ball thrown last")]
    public Entity LastBall;
    public List<Entity> Balls = [];
    [Tooltip("The HUD's top left corner, in pixels")]
    public Vector2 HudAt = new(24f, 24f);

    const float MaxLift = 60f;
    const float MinRadius = 0.5f;
    const float MaxRadius = 6f;
    const float BallMass = 2f;

    [Transient] bool _firing;
    [Transient] int _inBeam;
    [Transient] int _impacts;
    [Transient] float _hardest;

    public override void OnStart()
    {
        Every(0.25f, Survey);
        Debug.Log("showcase: Space throws, hold F for the beam, wheel and [ ] tune it, E shoves, F2 slow motion, F3 pause");
    }

    public override void OnUpdate(float deltaTime)
    {
        if (Input.GetKeyDown(KeyCode.Space))
            Throw(Camera.position + Camera.forward, Camera.forward * ThrowSpeed);
        if (Input.GetKeyDown(KeyCode.E))
            ShoveWhatIsAhead();
        if (Input.GetKeyDown(KeyCode.F2))
            ToggleSlowMotion();
        if (Input.GetKeyDown(KeyCode.F3))
            TogglePause();

        Lift = Mathf.Clamp(Lift + Input.scrollDelta.y * 2f, 0f, MaxLift);
        // On real time, so the beam can still be resized while the game is paused.
        var resize = (Input.GetKey(KeyCode.RightBracket) ? 1f : 0f) - (Input.GetKey(KeyCode.LeftBracket) ? 1f : 0f);
        Radius = Mathf.Clamp(Radius + resize * 3f * Time.unscaledDeltaTime, MinRadius, MaxRadius);

        var centre = Camera.position + Camera.forward * Reach;
        if (Input.GetKey(KeyCode.F))
        {
            _inBeam = Beam(centre);
            _firing = true;
        }
        else if (_firing)
        {
            Release();
            _firing = false;
            _inBeam = 0;
        }

        DrawRing(centre);
        DrawHud();
    }

    /// A ball at `from` moving at `velocity`, carrying a Thruster and an
    /// ImpactMeter, despawned after BallLife seconds of game time.
    public Entity Throw(Vector3 from, Vector3 velocity)
    {
        var transform = Transform.Identity;
        transform.Position = from;
        transform.Scale = new Vector3(0.5f, 0.5f, 0.5f);
        var ball = World.SpawnRenderable("sphere", "copper", transform);
        World.AddCollider(ball, new SphereCollider { Restitution = 0.4f });
        World.AddComponent(ball, new Rigidbody { Mass = BallMass, Continuous = true });
        ball.AddScript<Thruster>();
        ball.AddScript<ImpactMeter>();
        // Spawned this tick, so the body does not exist yet; the push waits for it.
        Physics.SetVelocity(ball, velocity);

        Balls.Add(ball);
        LastBall = ball;
        After(BallLife, () =>
        {
            Balls.Remove(ball);
            if (LastBall == ball)
                LastBall = Entity.Null;
            ball.Despawn();
        });
        return ball;
    }

    /// Fire the thruster of every ball within Radius of `centre`, and of no
    /// other. How many fired.
    public int Beam(Vector3 centre)
    {
        Release();
        // A stack buffer, so asking every frame allocates nothing. The query
        // also finds the floor and the crates, which carry no thruster.
        Span<Entity> found = stackalloc Entity[64];
        var count = Physics.OverlapSphere(centre, Radius, found);
        var fired = 0;
        for (var i = 0; i < count; i++)
        {
            if (!found[i].TryGetBehaviour<Thruster>(out var thruster))
                continue;
            thruster.Lift = Lift;
            fired++;
        }
        return fired;
    }

    public void Release()
    {
        foreach (var ball in Balls)
        {
            if (ball.TryGetBehaviour<Thruster>(out var thruster))
                thruster.Lift = 0f;
        }
    }

    public void ToggleSlowMotion() =>
        Time.timeScale = MathF.Abs(Time.timeScale - SlowMotion) < 1e-4f ? 1f : SlowMotion;

    public void TogglePause() =>
        Time.timeScale = Time.timeScale == 0f ? 1f : 0f;

    void ShoveWhatIsAhead()
    {
        if (!Physics.SphereCast(Camera.position, 0.25f, Camera.forward, out var hit, 50f))
        {
            Debug.Log("nothing ahead");
            return;
        }
        Debug.DrawLine(Camera.position + Camera.up * -0.2f, hit.Point, Color.yellow, 0.5f);
        var shoved = Physics.AddForce(hit.Entity, Camera.forward * Shove, ForceMode.Impulse);
        Debug.Log(shoved ? $"shoved {hit.Entity} at {hit.Distance:F1} m" : $"{hit.Entity} has no rigid body to shove");
    }

    /// The HUD's numbers, on a timer rather than every frame: a lookup per
    /// ball is cheap, but nobody reads a number that changes sixty times a
    /// second.
    void Survey()
    {
        _impacts = 0;
        _hardest = 0f;
        foreach (var ball in Balls)
        {
            if (!ball.TryGetBehaviour<ImpactMeter>(out var meter))
                continue;
            _impacts += meter.Impacts;
            _hardest = MathF.Max(_hardest, meter.Hardest);
        }
    }

    void DrawRing(Vector3 centre)
    {
        const int Segments = 32;
        var color = _firing ? BeamColor : Color.Lerp(Color.black, BeamColor, 0.35f);
        var previous = centre + new Vector3(Radius, 0f, 0f);
        for (var i = 1; i <= Segments; i++)
        {
            var angle = i * MathF.Tau / Segments;
            var next = centre + new Vector3(MathF.Cos(angle), 0f, MathF.Sin(angle)) * Radius;
            Debug.DrawLine(previous, next, color);
            previous = next;
        }
    }

    void DrawHud()
    {
        var scale = Time.timeScale;
        var time = scale == 0f ? "paused" : scale == 1f ? "full speed" : $"x{scale:F2}";
        string[] lines =
        [
            $"Time {time}   step {Time.fixedDeltaTime * 1000f:F1} ms   frame {Time.unscaledDeltaTime * 1000f:F1} ms",
            $"Beam {Lift:F0} N over {Radius:F1} m   {(_firing ? $"firing {_inBeam}" : "idle")}",
            $"Balls {Balls.Count}   impacts {_impacts}   hardest {_hardest:F1} m/s",
        ];
        for (var i = 0; i < lines.Length; i++)
            UI.Label(lines[i], new Rect(HudAt.x, HudAt.y + i * 26f, 720f, 26f));
    }
}

// Pushes its body up by Lift newtons. From OnFixedUpdate, once per physics
// step, so the push is held for exactly the step it is asked before — the same
// climb at 30 fps, at 144, and in slow motion.
[Component("demo.thruster")]
public class Thruster : Behaviour
{
    [Range(0f, 60f), Tooltip("Newtons, straight up")]
    public float Lift;

    public override void OnFixedUpdate(float fixedDeltaTime)
    {
        if (Lift > 0f)
            Physics.AddForce(Entity, Vector3.up * Lift);
    }
}

// Measures what its body runs into: how many impacts, the hardest of them,
// and how many physics steps it has spent touching something.
[Component("demo.impact_meter")]
public class ImpactMeter : Behaviour
{
    [Tooltip("Closing speed below which a contact is not an impact, in m/s. A pair resting under gravity closes at one step's fall, about 0.16.")]
    public float Threshold = 1f;
    public int Impacts;
    public float Hardest;
    public int StepsTouching;

    public override void OnCollisionEnter(Collision collision)
    {
        // The relative velocity is this body's less the other's and the normal
        // points at the other, so along the normal is how fast they closed.
        var closing = Vector3.Dot(collision.RelativeVelocity, collision.Normal);
        if (closing < Threshold)
            return;
        Impacts++;
        Hardest = MathF.Max(Hardest, closing);
        Debug.Log($"{Entity} hit {collision.Other} at {closing:F1} m/s");
    }

    public override void OnCollisionStay(Collision collision) => StepsTouching++;
}
