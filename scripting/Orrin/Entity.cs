using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

namespace Orrin;

// Layout must match Rust CEntity.
/// <summary>
/// A handle to an entity: the slot it occupies and which spawn in that slot it
/// names. The all-zeroes value is <see cref="Null"/>.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Entity : IEquatable<Entity>
{
    /// <summary>
    /// The entity's slot. The engine never allocates slot 0, so an index of 0 is the
    /// null handle.
    /// </summary>
    public readonly uint Index;
    /// <summary>
    /// Which spawn in its slot this handle names. A despawn and a respawn in the same
    /// slot bumps it, so a stale handle never equals the new occupant's.
    /// </summary>
    public readonly uint Generation;

    /// <summary>
    /// A handle from its two parts. Nothing checks that it names a live entity.
    /// </summary>
    public Entity(uint index, uint generation)
    {
        Index = index;
        Generation = generation;
    }

    /// <summary>
    /// The null handle: what a failed SpawnRenderable or lookup returns, and
    /// what <c>default(Entity)</c> is. Must match Rust CEntity::NULL ({0, 0}).
    /// </summary>
    /// <remarks>
    /// All-zeroes deliberately. C# gives every struct an all-zeroes <c>default</c>
    /// that types cannot override, so an <see cref="Entity"/> field you never assigned —
    /// most often one a hot reload could not restore — reads as this value. The
    /// engine reserves entity slot 0 and never allocates it, so the handle
    /// names nothing and <see cref="IsValid"/> catches the uninitialized case for free.
    /// </remarks>
    public static Entity Null => new(0, 0);

    /// <summary>
    /// False for the null handle (e.g. a SpawnRenderable given an unknown asset,
    /// an ABI call made outside the script dispatch window, or a field that was
    /// never assigned). This only rules out the null sentinel — it does NOT
    /// detect a stale handle to an entity that has since been despawned; use
    /// HasComponent or a World lookup for that.
    /// </summary>
    public bool IsValid => Index != 0;

    /// <summary>True while this handle is live and the entity has a T component.</summary>
    public bool HasComponent<T>() where T : struct =>
        Native.HasComponent(this, ComponentKinds.Of<T>());

    /// <summary>
    /// Typed read of an engine component, or null if the entity lacks it (or
    /// the handle is stale). Components come back by value; write back through
    /// the corresponding setter (<see cref="LocalTransform"/>, <see cref="SetTag"/>,
    /// World.SetCamera).
    /// Supports Transform, Tag and CameraComponent — script-defined components
    /// arrive with the byte-blob milestone.
    /// </summary>
    public T? GetComponent<T>() where T : struct
    {
        // typeof dispatch per supported component; the (T)(object) round-trip
        // boxes, which is fine at scripting call rates.
        if (typeof(T) == typeof(Transform))
            return HasComponent<Transform>() ? (T)(object)Native.GetTransform(this) : null;
        if (typeof(T) == typeof(Tag))
            return Native.GetTag(this) is { } tag ? (T)(object)new Tag(tag) : null;
        if (typeof(T) == typeof(CameraComponent))
            return Native.GetCameraLens(this) is { } camera ? (T)(object)camera : null;
        throw new ArgumentException($"{typeof(T).Name} is not a script-visible engine component");
    }

    /// <summary>
    /// This entity's transform relative to its parent; for a root, also its
    /// world transform. Setting it moves the entity from this frame.
    /// </summary>
    public Transform LocalTransform
    {
        get => Native.GetTransform(this);
        set => Native.SetTransform(this, value);
    }

    /// <summary>
    /// This entity's transform in world space, whatever it is parented to.
    /// Setting it composes with the inverse of the parent's transform; the
    /// scale read back is a best fit once a non-uniformly scaled ancestor
    /// introduces shear.
    /// </summary>
    public Transform WorldTransform
    {
        get => Native.GetWorldTransform(this);
        set => Native.SetWorldTransform(this, value);
    }

    /// <summary>This entity's parent, or <see cref="Null"/> for a root.</summary>
    public Entity Parent => Native.GetParent(this);

    /// <summary>
    /// Attach this entity to <paramref name="parent"/>, or pass
    /// <see cref="Null"/> to detach it. With <paramref name="keepWorld"/> it
    /// stays where it is in the world; without it, it keeps its local
    /// transform and moves into the parent's frame. Returns false, changing
    /// nothing, for a cycle or a stale handle. Applied after this tick, so
    /// <see cref="Parent"/> reads the old value until the next one.
    /// </summary>
    public bool SetParent(Entity parent, bool keepWorld = true) =>
        Native.SetParent(this, parent, keepWorld);

    /// <summary>This entity's tag, or null if it has none.</summary>
    public string? Tag => Native.GetTag(this);

    /// <summary>
    /// Set or replace this entity's tag. Applied after this tick, so
    /// <see cref="World.FindByTag"/> and <see cref="Tag"/> see it from the next
    /// one. Returns false for a stale handle.
    /// </summary>
    public bool SetTag(string tag) => Native.SetTag(this, tag);

    /// <summary>
    /// Despawn this entity, and everything parented to it, at the end of this
    /// tick. Returns false if the handle was already stale.
    /// </summary>
    public bool Despawn() => Native.Despawn(this);

    /// <summary>
    /// Attach a Behaviour of type <typeparamref name="T"/> to this entity. It is
    /// created when this tick's commands apply, and its OnEnable and OnStart run
    /// on the next tick. Returns false for a stale handle.
    /// </summary>
    public bool AddScript<T>() where T : Behaviour, new() => World.AddScript<T>(this);

    /// <summary>
    /// This entity's Behaviour of type <typeparamref name="T"/> (or derived from
    /// it), or null if it has none. A Behaviour attached this tick is found from
    /// the next, once it has been created; one being destroyed is found until
    /// its OnDestroy has run.
    /// </summary>
    public T? GetBehaviour<T>() where T : Behaviour => Behaviours.Find<T>(this);

    /// <summary><see cref="GetBehaviour{T}"/>, as a test.</summary>
    public bool TryGetBehaviour<T>([MaybeNullWhen(false)] out T behaviour) where T : Behaviour
    {
        behaviour = Behaviours.Find<T>(this);
        return behaviour is not null;
    }

    /// <summary>
    /// Handle equality: same index and generation. Two handles being equal means
    /// they name the same spawn of the same entity — a despawn-and-respawn at the
    /// same index bumps the generation, so a stale handle won't compare equal to
    /// the new occupant.
    /// </summary>
    public static bool operator ==(Entity a, Entity b) =>
        a.Index == b.Index && a.Generation == b.Generation;

    /// <summary>True when the handles differ in index or generation.</summary>
    public static bool operator !=(Entity a, Entity b) => !(a == b);

    /// <inheritdoc/>
    public bool Equals(Entity other) => this == other;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Entity other && this == other;

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(Index, Generation);

    /// <summary>The handle as <c>Entity(7v1)</c>: index 7, generation 1.</summary>
    public override string ToString() => $"Entity({Index}v{Generation})";
}
