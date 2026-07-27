using System.Runtime.InteropServices;

namespace Ferron;

// Layout must match Rust CEntity.
[StructLayout(LayoutKind.Sequential)]
public readonly struct Entity : IEquatable<Entity>
{
    public readonly uint Index;
    public readonly uint Generation;

    public Entity(uint index, uint generation)
    {
        Index = index;
        Generation = generation;
    }

    /// The null handle: what a failed SpawnRenderable or lookup returns, and
    /// what `default(Entity)` is. Must match Rust CEntity::NULL ({0, 0}).
    ///
    /// All-zeroes deliberately. C# gives every struct an all-zeroes `default`
    /// that types cannot override, so an `Entity` field you never assigned —
    /// most often one a hot reload could not restore — reads as this value. The
    /// engine reserves entity slot 0 and never allocates it, so the handle
    /// names nothing and `IsValid` catches the uninitialized case for free.
    public static Entity Null => new(0, 0);

    /// False for the null handle (e.g. a SpawnRenderable given an unknown asset,
    /// an ABI call made outside the script dispatch window, or a field that was
    /// never assigned). This only rules out the null sentinel — it does NOT
    /// detect a stale handle to an entity that has since been despawned; use
    /// HasComponent or a World lookup for that.
    public bool IsValid => Index != 0;

    /// True while this handle is live and the entity has a T component.
    public bool HasComponent<T>() where T : struct =>
        Native.HasComponent(this, ComponentKinds.Of<T>());

    /// Typed read of an engine component, or null if the entity lacks it (or
    /// the handle is stale). Components come back by value; write back through
    /// the corresponding setter (Native.SetTransform, World.SetTag). v1
    /// supports Transform and Tag — script-defined components arrive with the
    /// byte-blob milestone.
    public T? GetComponent<T>() where T : struct
    {
        // typeof dispatch per supported component; the (T)(object) round-trip
        // boxes, which is fine at scripting call rates.
        if (typeof(T) == typeof(Transform))
            return HasComponent<Transform>() ? (T)(object)Native.GetTransform(this) : null;
        if (typeof(T) == typeof(Tag))
            return Native.GetTag(this) is { } tag ? (T)(object)new Tag(tag) : null;
        throw new ArgumentException($"{typeof(T).Name} is not a script-visible engine component");
    }

    /// Handle equality: same index and generation. Two handles being equal means
    /// they name the same spawn of the same entity — a despawn-and-respawn at the
    /// same index bumps the generation, so a stale handle won't compare equal to
    /// the new occupant.
    public static bool operator ==(Entity a, Entity b) =>
        a.Index == b.Index && a.Generation == b.Generation;

    public static bool operator !=(Entity a, Entity b) => !(a == b);

    public bool Equals(Entity other) => this == other;

    public override bool Equals(object? obj) => obj is Entity other && this == other;

    public override int GetHashCode() => HashCode.Combine(Index, Generation);

    public override string ToString() => $"Entity({Index}v{Generation})";
}
