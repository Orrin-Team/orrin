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

    /// This entity's transform relative to its parent. For an entity with no
    /// parent it is also its world transform.
    ///
    /// Named for the space it is in, because once an entity can have a parent
    /// "the transform" is ambiguous and the two disagree silently.
    protected Transform LocalTransform
    {
        get => Native.GetTransform(Entity);
        set => Native.SetTransform(Entity, value);
    }

    /// This entity's transform in world space, whatever it is parented to.
    ///
    /// Setting it composes with the inverse of the parent's transform, so an
    /// object can be placed in the world without a script knowing whether it
    /// has a parent. The scale that comes back is a best fit — see
    /// [Native.GetWorldTransform].
    protected Transform WorldTransform
    {
        get => Native.GetWorldTransform(Entity);
        set => Native.SetWorldTransform(Entity, value);
    }

    /// This entity's parent, or `Entity.Null` if it is a root.
    protected Entity Parent => Native.GetParent(Entity);

    /// Attach this entity to `parent`, or pass `Entity.Null` to detach it.
    ///
    /// Returns false and changes nothing if the move would create a cycle. The
    /// change is structural, so it lands after this tick's dispatch — `Parent`
    /// still reads the old value until the next frame.
    protected bool SetParent(Entity parent, bool keepWorld = true) =>
        Native.SetParent(Entity, parent, keepWorld);
}
