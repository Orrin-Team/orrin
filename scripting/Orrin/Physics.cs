using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

/// <summary>How a push is read. Numbering is lock-step with the engine's.</summary>
public enum ForceMode : uint
{
    /// <summary>
    /// Newtons (newton-metres for a torque), held for this frame — or, asked
    /// from OnFixedUpdate, for the step that follows.
    /// </summary>
    Force = 0,

    /// <summary>N·s, all at once.</summary>
    Impulse = 1,

    /// <summary>m/s (rad/s), regardless of mass.</summary>
    VelocityChange = 2,

    /// <summary>m/s² (rad/s²), regardless of mass, held like a Force.</summary>
    Acceleration = 3,
}

/// <summary>What <see cref="Physics.Raycast"/>, <see cref="Physics.SphereCast"/> or <see cref="Physics.BoxCast"/> hit.</summary>
// Layout must match Rust CRaycastHit: an Entity, two Vector3s and a float.
[StructLayout(LayoutKind.Sequential)]
public readonly struct RaycastHit
{
    /// <summary>The entity whose collider was hit.</summary>
    public readonly Entity Entity;

    /// <summary>World space.</summary>
    public readonly Vector3 Point;

    /// <summary>The surface's normal at <see cref="Point"/>.</summary>
    public readonly Vector3 Normal;

    /// <summary>Metres from the ray's origin.</summary>
    public readonly float Distance;

    /// <inheritdoc/>
    public override string ToString() => $"RaycastHit({Entity} at {Point}, normal {Normal}, {Distance} m)";
}

// A sphere or a box placed in the world, as the overlaps and shape casts send
// it. Layout must match Rust CShapeQuery: Shape is 0 for a sphere, whose radius
// is Size.x, and 1 for a box, whose half extents are Size.
[StructLayout(LayoutKind.Sequential)]
internal struct ShapeQuery
{
    public uint Shape;
    public Vector3 Center;
    public Vector3 Size;
    public Quaternion Rotation;
    public Vector3 Direction;
    public float MaxDistance;
    public uint Layers;
    public uint HitTriggers;

    public static ShapeQuery Sphere(Vector3 center, float radius, uint layers, bool hitTriggers) => new()
    {
        Shape = 0,
        Center = center,
        Size = new Vector3(radius, 0f, 0f),
        Layers = layers,
        HitTriggers = hitTriggers ? 1u : 0u,
    };

    public static ShapeQuery Box(
        Vector3 center, Vector3 halfExtents, Quaternion orientation, uint layers, bool hitTriggers) => new()
    {
        Shape = 1,
        Center = center,
        Size = halfExtents,
        Rotation = orientation,
        Layers = layers,
        HitTriggers = hitTriggers ? 1u : 0u,
    };

    public ShapeQuery Along(Vector3 direction, float maxDistance)
    {
        Direction = direction;
        MaxDistance = maxDistance;
        return this;
    }
}

/// <summary>
/// Pushing rigid bodies and asking the world what is where. Only a dynamic
/// body — a <see cref="Rigidbody"/> that is not kinematic and has no
/// transformed parent — can be pushed; anything else returns false.
/// </summary>
/// <remarks>
/// A <see cref="ForceMode.Force"/> or <see cref="ForceMode.Acceleration"/> is
/// held for as long as the hook it is asked from covers: the frame, from
/// OnUpdate, or the step that follows, from OnFixedUpdate. Pushing every frame
/// or every step for a second is the same push at any frame rate. A body
/// spawned this frame can be pushed at once; the push waits for it.
/// </remarks>
public static class Physics
{
    /// <summary>Every layer, for a ray that should see them all.</summary>
    public const uint AllLayers = uint.MaxValue;

    /// <summary>
    /// Push <paramref name="entity"/>'s body through its centre of mass along the
    /// world-space <paramref name="force"/>, read as <paramref name="mode"/> says.
    /// False, pushing nothing, for a force that is not finite.
    /// </summary>
    public static bool AddForce(Entity entity, Vector3 force, ForceMode mode = ForceMode.Force) =>
        Native.PhysicsAddForce(entity, force, mode);

    /// <summary>
    /// Turn <paramref name="entity"/>'s body about the world-space axis of
    /// <paramref name="torque"/>, its length the strength, read as
    /// <paramref name="mode"/> says. False for a torque that is not finite.
    /// </summary>
    public static bool AddTorque(Entity entity, Vector3 torque, ForceMode mode = ForceMode.Force) =>
        Native.PhysicsAddTorque(entity, torque, mode);

    /// <summary>m/s as of the last step; zero for an entity with no body.</summary>
    public static Vector3 GetVelocity(Entity entity) =>
        Native.PhysicsGetVelocity(entity, angular: false, out var velocity) ? velocity : Vector3.zero;

    /// <summary>
    /// Replace <paramref name="entity"/>'s velocity with <paramref name="velocity"/>,
    /// in m/s, world space. False for a velocity that is not finite.
    /// </summary>
    public static bool SetVelocity(Entity entity, Vector3 velocity) =>
        Native.PhysicsSetVelocity(entity, angular: false, velocity);

    /// <summary>rad/s about each axis, as of the last step.</summary>
    public static Vector3 GetAngularVelocity(Entity entity) =>
        Native.PhysicsGetVelocity(entity, angular: true, out var velocity) ? velocity : Vector3.zero;

    /// <summary>
    /// Replace <paramref name="entity"/>'s spin with <paramref name="velocity"/>, in
    /// rad/s about each world axis. False for a velocity that is not finite.
    /// </summary>
    public static bool SetAngularVelocity(Entity entity, Vector3 velocity) =>
        Native.PhysicsSetVelocity(entity, angular: true, velocity);

    /// <summary>
    /// The first collider along the ray, within <paramref name="maxDistance"/>
    /// metres, on a layer whose bit is set in <paramref name="layers"/>.
    /// Triggers are passed through unless <paramref name="hitTriggers"/>: a
    /// trigger is a volume to be in, not a surface to see. Bodies are where the
    /// last step left them.
    /// </summary>
    public static bool Raycast(
        Vector3 origin,
        Vector3 direction,
        out RaycastHit hit,
        float maxDistance = float.PositiveInfinity,
        uint layers = AllLayers,
        bool hitTriggers = false) =>
        Native.PhysicsRaycast(origin, direction, maxDistance, layers, hitTriggers, out hit);

    /// <summary>
    /// Every entity whose collider overlaps the sphere, written into
    /// <paramref name="results"/>, each once and in entity order. Returns how
    /// many were written; any that did not fit are left out, so a buffer kept
    /// between calls makes this allocate nothing. Filtered as
    /// <see cref="Raycast"/> is.
    /// </summary>
    /// <remarks>
    /// Like every query here it sees the world as the last physics step left
    /// it. A collider added this frame is not found until the next step has
    /// built its body, and a collider without a <see cref="Rigidbody"/> that
    /// was moved this frame is found where it was until a step carries it
    /// there — which, with <see cref="Time.timeScale"/> at 0, is never.
    /// </remarks>
    public static int OverlapSphere(
        Vector3 center,
        float radius,
        Span<Entity> results,
        uint layers = AllLayers,
        bool hitTriggers = false) =>
        System.Math.Min(Native.PhysicsOverlap(ShapeQuery.Sphere(center, radius, layers, hitTriggers), results), results.Length);

    /// <summary>
    /// <see cref="OverlapSphere(Vector3, float, Span{Entity}, uint, bool)"/>
    /// into a new array holding every overlap. Allocates; prefer the buffer
    /// overload for a query asked every frame.
    /// </summary>
    public static Entity[] OverlapSphere(
        Vector3 center, float radius, uint layers = AllLayers, bool hitTriggers = false) =>
        Overlapping(ShapeQuery.Sphere(center, radius, layers, hitTriggers));

    /// <summary>
    /// Every entity whose collider overlaps the box, which is
    /// <paramref name="halfExtents"/> either side of <paramref name="center"/>
    /// turned by <paramref name="orientation"/> — <c>default</c> turns nothing.
    /// Written and counted as <see cref="OverlapSphere(Vector3, float, Span{Entity}, uint, bool)"/>
    /// writes them, and seeing the world as it does.
    /// </summary>
    public static int OverlapBox(
        Vector3 center,
        Vector3 halfExtents,
        Span<Entity> results,
        Quaternion orientation = default,
        uint layers = AllLayers,
        bool hitTriggers = false) =>
        System.Math.Min(
            Native.PhysicsOverlap(ShapeQuery.Box(center, halfExtents, orientation, layers, hitTriggers), results),
            results.Length);

    /// <summary>
    /// <see cref="OverlapBox(Vector3, Vector3, Span{Entity}, Quaternion, uint, bool)"/>
    /// into a new array holding every overlap. Allocates.
    /// </summary>
    public static Entity[] OverlapBox(
        Vector3 center,
        Vector3 halfExtents,
        Quaternion orientation = default,
        uint layers = AllLayers,
        bool hitTriggers = false) =>
        Overlapping(ShapeQuery.Box(center, halfExtents, orientation, layers, hitTriggers));

    /// <summary>
    /// The first collider a sphere of <paramref name="radius"/> meets, swept
    /// from <paramref name="origin"/> along <paramref name="direction"/> for at
    /// most <paramref name="maxDistance"/> metres, filtered as
    /// <see cref="Raycast"/> is. The hit's distance is how far the sphere
    /// travelled, its point where it touched, and its normal the surface's
    /// there. A sphere that starts inside a collider hits it at distance 0.
    /// Sees the world as <see cref="OverlapSphere(Vector3, float, Span{Entity}, uint, bool)"/> does.
    /// </summary>
    public static bool SphereCast(
        Vector3 origin,
        float radius,
        Vector3 direction,
        out RaycastHit hit,
        float maxDistance = float.PositiveInfinity,
        uint layers = AllLayers,
        bool hitTriggers = false) =>
        Native.PhysicsShapeCast(
            ShapeQuery.Sphere(origin, radius, layers, hitTriggers).Along(direction, maxDistance), out hit);

    /// <summary>
    /// <see cref="SphereCast"/> with a box of <paramref name="halfExtents"/>
    /// turned by <paramref name="orientation"/> — <c>default</c> turns nothing —
    /// swept from <paramref name="center"/>.
    /// </summary>
    public static bool BoxCast(
        Vector3 center,
        Vector3 halfExtents,
        Vector3 direction,
        out RaycastHit hit,
        Quaternion orientation = default,
        float maxDistance = float.PositiveInfinity,
        uint layers = AllLayers,
        bool hitTriggers = false) =>
        Native.PhysicsShapeCast(
            ShapeQuery.Box(center, halfExtents, orientation, layers, hitTriggers).Along(direction, maxDistance),
            out hit);

    static Entity[] Overlapping(in ShapeQuery query)
    {
        Span<Entity> first = stackalloc Entity[32];
        int total = Native.PhysicsOverlap(query, first);
        if (total <= first.Length)
            return first[..total].ToArray();
        // The second call sees what the first did: a query reads the last
        // step, and no step can run while a script is running.
        var all = new Entity[total];
        Native.PhysicsOverlap(query, all);
        return all;
    }
}
