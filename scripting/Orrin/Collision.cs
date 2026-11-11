using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

// Layout must match Rust CCollision (Entity + three Vector3s — eleven
// sequential 4-byte fields; Orrin.Math types are layout-compatible by
// construction). It only ever grows at the end: the engine hands it over by
// pointer, unchecked.
/// <summary>
/// A contact between this entity's collider and another's, as the collision and
/// trigger callbacks receive it, seen from this entity's side.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Collision
{
    /// <summary>The other entity involved in the collision.</summary>
    public readonly Entity Other;

    /// <summary>World-space contact point.</summary>
    public readonly Vector3 ContactPoint;

    /// <summary>Unit contact normal, pointing from this entity toward Other.</summary>
    public readonly Vector3 Normal;

    /// <summary>
    /// This entity's velocity at the contact point less Other's, in m/s, world
    /// space, taken before the step's solver acts on the contact. Its component
    /// along <see cref="Normal"/> is the closing speed, positive while they
    /// approach — what an impact sound scales by. A pair resting under gravity
    /// reads one step's worth of fall rather than zero (about 0.16 m/s at
    /// 60 Hz), so a sound wants a threshold above that. On an exit it is the
    /// last value the pair had.
    /// </summary>
    public readonly Vector3 RelativeVelocity;

    /// <inheritdoc/>
    public override string ToString() =>
        $"Collision(other: {Other}, at: {ContactPoint}, normal: {Normal}, relative velocity: {RelativeVelocity})";
}
