using Orrin.Math;

namespace Orrin;

/// Shape descriptor for World.AddCollider. Dimensions are local-space and get
/// scaled by the entity's transform scale engine-side.
public abstract class Collider
{
    /// Triggers fire the same OnCollisionEnter/Exit callbacks as solid
    /// colliders but never resolve overlap — nothing is pushed out of a
    /// trigger, and a trigger is never pushed.
    public bool IsTrigger { get; init; }
}

public sealed class BoxCollider : Collider
{
    /// Half the box's size on each axis; the default matches the unit cube
    /// mesh. Note: the engine collides boxes as world-space AABBs, so a
    /// rotated entity gets a conservatively enlarged volume.
    public Vector3 HalfExtents { get; init; } = new(0.5f, 0.5f, 0.5f);
}

public sealed class SphereCollider : Collider
{
    /// The default matches the unit sphere mesh.
    public float Radius { get; init; } = 0.5f;
}
