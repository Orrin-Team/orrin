namespace Orrin;

/// <summary>
/// Keep constructors pure — no logging, no registration, no engine calls. The
/// engine builds a throwaway instance of each Behaviour type during a hot
/// reload to read its field defaults, so a constructor with side effects fires
/// more often than the lifecycle below suggests. <see cref="Behaviour.Entity"/> is not assigned yet
/// at construction either; do setup in OnStart.
/// </summary>
public abstract class Behaviour
{
    /// <summary>
    /// The entity this behaviour is attached to. The engine assigns it once the
    /// constructor has run, so it reads as <see cref="Orrin.Entity.Null"/> there.
    /// </summary>
    public Entity Entity { get; internal set; }

    /// Whether the behaviour is currently active, as last dispatched by the
    /// engine. Owned here (not only in Rust) so the destroy path can decide
    /// whether OnDisable is still owed without a round-trip.
    internal bool Active;

    /// <summary>
    /// Called each time the behaviour becomes active: on the first tick it is, and
    /// again after every deactivation.
    /// </summary>
    public virtual void OnEnable() { }

    /// <summary>
    /// Called once, on the first tick the behaviour is active, straight after that
    /// tick's <see cref="OnEnable"/> and before its <see cref="OnUpdate"/>.
    /// </summary>
    public virtual void OnStart() { }

    /// <summary>
    /// Called every tick while the behaviour is active, after the tick's collision
    /// and pointer callbacks. <paramref name="deltaTime"/> is
    /// <see cref="Time.DeltaTime"/>: seconds of game time, scaled by
    /// <see cref="Time.timeScale"/>, so zero while the game is paused.
    /// </summary>
    public virtual void OnUpdate(float deltaTime) { }

    /// <summary>
    /// Called before each physics step, while the behaviour is active and
    /// started — so never on the frame it starts. A frame takes as many steps
    /// as the time it covers, which can be none or several, and they all run
    /// before that frame's OnUpdate. A <see cref="ForceMode.Force"/> asked for
    /// here is held for the one step that follows, and a body's transform reads
    /// where the last step left it. Spawns, despawns and other structural
    /// changes land after the frame's last step, before OnUpdate.
    /// </summary>
    /// <remarks>
    /// The engine calls it only on a type that overrides it, so a behaviour
    /// without one costs nothing per step.
    /// </remarks>
    public virtual void OnFixedUpdate(float fixedDeltaTime) { }

    /// <summary>
    /// Called each time the behaviour stops being active, and when it is destroyed
    /// while still active, before <see cref="OnDestroy"/>.
    /// </summary>
    public virtual void OnDisable() { }

    /// <summary>
    /// Called once, when the behaviour is torn down: its entity despawned, its scene
    /// unloaded, or the game's scripts reloaded. It can run outside the tick, where
    /// the world cannot be read.
    /// </summary>
    public virtual void OnDestroy() { }

    /// <summary>
    /// Fired the first frame this entity's collider overlaps another; the
    /// engine dispatches collision callbacks before OnUpdate each tick.
    /// </summary>
    public virtual void OnCollisionEnter(Collision other) { }

    /// <summary>
    /// Fired once for every physics step a pair is still touching after the
    /// step it entered on, in the collision phase of the frame that took the
    /// steps. A pair at rest whose bodies have fallen asleep is still touching
    /// — it never exits — but is not reported again until one of them wakes.
    /// Called only on a type that overrides it.
    /// </summary>
    public virtual void OnCollisionStay(Collision other) { }

    /// <summary>
    /// Fired the first frame a previously-overlapping pair separates. The
    /// payload carries the *last known* contact — there is no contact this
    /// frame by definition.
    /// </summary>
    public virtual void OnCollisionExit(Collision other) { }

    /// <summary>
    /// The collision callbacks' counterpart for a pair where either side is a
    /// trigger: something entered this entity's trigger, or this entity
    /// entered one. A trigger never calls OnCollisionEnter.
    /// </summary>
    public virtual void OnTriggerEnter(Collision other) { }

    /// <summary>
    /// <see cref="OnCollisionStay"/> for a trigger pair, which never calls
    /// OnCollisionStay.
    /// </summary>
    public virtual void OnTriggerStay(Collision other) { }

    /// <summary>Fired when a trigger pair separates; carries the last known contact.</summary>
    public virtual void OnTriggerExit(Collision other) { }

    /// <summary>
    /// This entity's transform relative to its parent. For an entity with no
    /// parent it is also its world transform.
    /// </summary>
    /// <remarks>
    /// Named for the space it is in, because once an entity can have a parent
    /// "the transform" is ambiguous and the two disagree silently.
    /// </remarks>
    protected Transform LocalTransform
    {
        get => Native.GetTransform(Entity);
        set => Native.SetTransform(Entity, value);
    }

    /// <summary>
    /// This entity's transform in world space, whatever it is parented to.
    /// </summary>
    /// <remarks>
    /// Setting it composes with the inverse of the parent's transform, so an
    /// object can be placed in the world without a script knowing whether it
    /// has a parent. The scale that comes back is a best fit — see
    /// <see cref="Native.GetWorldTransform"/>.
    /// </remarks>
    protected Transform WorldTransform
    {
        get => Native.GetWorldTransform(Entity);
        set => Native.SetWorldTransform(Entity, value);
    }

    /// <summary>This entity's parent, or <see cref="Orrin.Entity.Null"/> if it is a root.</summary>
    protected Entity Parent => Native.GetParent(Entity);

    /// <summary>
    /// Attach this entity to <paramref name="parent"/>, or pass
    /// <see cref="Orrin.Entity.Null"/> to detach it.
    /// </summary>
    /// <remarks>
    /// Returns false and changes nothing if the move would create a cycle. The
    /// change is structural, so it lands after this tick's dispatch — <see cref="Parent"/>
    /// still reads the old value until the next frame.
    /// </remarks>
    protected bool SetParent(Entity parent, bool keepWorld = true) =>
        Native.SetParent(Entity, parent, keepWorld);

    /// Owned by the instance, so timers die with it and a hot reload, which
    /// builds a fresh instance, starts with none.
    List<TimerHandle>? _timers;

    /// <summary>
    /// Run <paramref name="callback"/> once, <paramref name="seconds"/> from
    /// now. Timers run just before <see cref="OnUpdate"/>, on the same delta,
    /// so one started earlier in the tick — in OnStart or a collision callback
    /// — counts this tick's delta too. They pause while the behaviour is
    /// disabled and are dropped when it is destroyed or its scripts are
    /// reloaded. A callback that throws faults the script exactly as a
    /// throwing OnUpdate does.
    /// </summary>
    protected TimerHandle After(float seconds, Action callback)
    {
        ArgumentNullException.ThrowIfNull(callback);
        if (float.IsNaN(seconds))
            throw new ArgumentOutOfRangeException(nameof(seconds), seconds, "a delay must be a number");
        return Schedule(new TimerHandle(seconds, callback, 0f));
    }

    /// <summary>
    /// Run <paramref name="callback"/> every <paramref name="seconds"/> until
    /// it is cancelled, first <paramref name="seconds"/> from now. It runs at
    /// most once a tick: a tick longer than the interval runs it once rather
    /// than once per interval missed. Otherwise as <see cref="After"/>.
    /// </summary>
    protected TimerHandle Every(float seconds, Action callback)
    {
        ArgumentNullException.ThrowIfNull(callback);
        if (!(seconds > 0f))
            throw new ArgumentOutOfRangeException(nameof(seconds), seconds, "an interval must be positive");
        return Schedule(new TimerHandle(seconds, callback, seconds));
    }

    TimerHandle Schedule(TimerHandle timer)
    {
        (_timers ??= []).Add(timer);
        return timer;
    }

    /// Run the timers due this tick, stopping at the first that throws and
    /// handing back which kind it was and what it threw. A timer started by a
    /// callback waits for the next tick.
    internal (string Hook, Exception Error)? TickTimers(float deltaTime)
    {
        if (_timers is not { Count: > 0 } timers)
            return null;
        (string, Exception)? failure = null;
        var due = timers.Count;
        for (var i = 0; i < due && failure is null; i++)
        {
            var timer = timers[i];
            try
            {
                timer.Tick(deltaTime);
            }
            catch (Exception e)
            {
                failure = (timer.Repeats ? nameof(Every) : nameof(After), e);
            }
        }
        timers.RemoveAll(static timer => !timer.IsPending);
        return failure;
    }
}
