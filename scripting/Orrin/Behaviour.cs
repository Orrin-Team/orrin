namespace Orrin;

/// Keep constructors pure — no logging, no registration, no engine calls. The
/// engine builds a throwaway instance of each Behaviour type during a hot
/// reload to read its field defaults, so a constructor with side effects fires
/// more often than the lifecycle below suggests. `Entity` is not assigned yet
/// at construction either; do setup in OnStart.
public abstract class Behaviour
{
    public Entity Entity { get; internal set; }

    /// Whether the behaviour is currently active, as last dispatched by the
    /// engine. Owned here (not only in Rust) so the destroy path can decide
    /// whether OnDisable is still owed without a round-trip.
    internal bool Active;

    public virtual void OnEnable() { }

    public virtual void OnStart() { }

    public virtual void OnUpdate(float deltaTime) { }

    public virtual void OnDisable() { }

    public virtual void OnDestroy() { }

    /// Fired the first frame this entity's collider overlaps another; the
    /// engine dispatches collision callbacks before OnUpdate each tick.
    public virtual void OnCollisionEnter(Collision other) { }

    /// Fired the first frame a previously-overlapping pair separates. The
    /// payload carries the *last known* contact — there is no contact this
    /// frame by definition.
    public virtual void OnCollisionExit(Collision other) { }

    protected Transform Transform
    {
        get => Native.GetTransform(Entity);
        set => Native.SetTransform(Entity, value);
    }
}
