using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

/// <summary>
/// Shape descriptor for World.AddCollider. Dimensions are local-space and get
/// scaled by the entity's transform scale engine-side.
/// </summary>
public abstract class Collider
{
    /// <summary>
    /// Reports entering and leaving through OnTriggerEnter/OnTriggerExit
    /// instead of the collision callbacks, and pushes nothing: nothing is pushed
    /// out of a trigger, and a trigger is never pushed.
    /// </summary>
    public bool IsTrigger { get; init; }

    /// <summary>Where two colliders meet, friction is their geometric mean.</summary>
    public float Friction { get; init; } = 0.5f;

    /// <summary>0 is dead, 1 bounces back as fast as it arrived. Where two meet, the larger wins.</summary>
    public float Restitution { get; init; }

    /// <summary>A row of the scene's layer matrix, 0 to 7.</summary>
    public uint Layer { get; init; }

    private protected abstract (ColliderShape Shape, Vector3 Size) Describe();

    internal ColliderData ToData()
    {
        var (shape, size) = Describe();
        return new ColliderData
        {
            Shape = shape,
            Size = size,
            IsTrigger = IsTrigger ? 1u : 0u,
            Friction = Friction,
            Restitution = Restitution,
            Layer = Layer,
        };
    }
}

public sealed class BoxCollider : Collider
{
    /// <summary>Half the box's size on each axis; the default matches the unit cube mesh.</summary>
    public Vector3 HalfExtents { get; init; } = new(0.5f, 0.5f, 0.5f);

    private protected override (ColliderShape, Vector3) Describe() => (ColliderShape.Box, HalfExtents);
}

public sealed class SphereCollider : Collider
{
    /// <summary>The default matches the unit sphere mesh.</summary>
    public float Radius { get; init; } = 0.5f;

    private protected override (ColliderShape, Vector3) Describe() =>
        (ColliderShape.Sphere, new Vector3(Radius, 0f, 0f));
}

/// <summary>Along the local Y axis: two hemispheres <see cref="HalfHeight"/> either side of the centre.</summary>
public sealed class CapsuleCollider : Collider
{
    public float Radius { get; init; } = 0.5f;

    /// <summary>Excludes the caps: the whole is 2 × (HalfHeight + Radius) tall.</summary>
    public float HalfHeight { get; init; } = 0.5f;

    private protected override (ColliderShape, Vector3) Describe() =>
        (ColliderShape.Capsule, new Vector3(Radius, HalfHeight, 0f));
}

/// <summary>Along the local Y axis.</summary>
public sealed class CylinderCollider : Collider
{
    public float Radius { get; init; } = 0.5f;

    public float HalfHeight { get; init; } = 0.5f;

    private protected override (ColliderShape, Vector3) Describe() =>
        (ColliderShape.Cylinder, new Vector3(Radius, HalfHeight, 0f));
}

/// <summary>
/// The triangles of the entity's own mesh: exact, and hollow, so for what does
/// not move. A rigid body given one collides as its convex hull and logs why.
/// </summary>
public sealed class MeshCollider : Collider
{
    private protected override (ColliderShape, Vector3) Describe() => (ColliderShape.Mesh, Vector3.zero);
}

/// <summary>The tightest convex shape around the entity's own mesh: what a moving body built from a model uses.</summary>
public sealed class ConvexHullCollider : Collider
{
    private protected override (ColliderShape, Vector3) Describe() => (ColliderShape.ConvexHull, Vector3.zero);
}

// Numbering is lock-step with the engine's `add_collider`: append, never renumber.
internal enum ColliderShape : uint
{
    Box = 0,
    Sphere = 1,
    Capsule = 2,
    Cylinder = 3,
    Mesh = 4,
    ConvexHull = 5,
}

// Layout must match Rust CCollider: eight 4-byte fields.
[StructLayout(LayoutKind.Sequential)]
internal struct ColliderData
{
    public ColliderShape Shape;
    public Vector3 Size;
    public uint IsTrigger;
    public float Friction;
    public float Restitution;
    public uint Layer;
}
