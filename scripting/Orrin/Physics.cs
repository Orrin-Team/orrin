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

/// <summary>What <see cref="Physics.Raycast"/> hit.</summary>
// Layout must match Rust CRaycastHit: an Entity, two Vector3s and a float.
[StructLayout(LayoutKind.Sequential)]
public readonly struct RaycastHit
{
    public readonly Entity Entity;

    /// <summary>World space.</summary>
    public readonly Vector3 Point;

    /// <summary>The surface's normal at <see cref="Point"/>.</summary>
    public readonly Vector3 Normal;

    /// <summary>Metres from the ray's origin.</summary>
    public readonly float Distance;

    public override string ToString() => $"RaycastHit({Entity} at {Point}, normal {Normal}, {Distance} m)";
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

    public static bool AddForce(Entity entity, Vector3 force, ForceMode mode = ForceMode.Force) =>
        Native.PhysicsAddForce(entity, force, mode);

    public static bool AddTorque(Entity entity, Vector3 torque, ForceMode mode = ForceMode.Force) =>
        Native.PhysicsAddTorque(entity, torque, mode);

    /// <summary>m/s as of the last step; zero for an entity with no body.</summary>
    public static Vector3 GetVelocity(Entity entity) =>
        Native.PhysicsGetVelocity(entity, angular: false, out var velocity) ? velocity : Vector3.zero;

    public static bool SetVelocity(Entity entity, Vector3 velocity) =>
        Native.PhysicsSetVelocity(entity, angular: false, velocity);

    /// <summary>rad/s about each axis, as of the last step.</summary>
    public static Vector3 GetAngularVelocity(Entity entity) =>
        Native.PhysicsGetVelocity(entity, angular: true, out var velocity) ? velocity : Vector3.zero;

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
}
