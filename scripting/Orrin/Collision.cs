using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

// Layout must match Rust CCollision (Entity + two Vector3s — eight sequential
// 4-byte fields; Orrin.Math types are layout-compatible by construction).
[StructLayout(LayoutKind.Sequential)]
public readonly struct Collision
{
    /// The other entity involved in the collision.
    public readonly Entity Other;

    /// World-space contact point.
    public readonly Vector3 ContactPoint;

    /// Unit contact normal, pointing from this entity toward Other.
    public readonly Vector3 Normal;

    public override string ToString() => $"Collision(other: {Other}, at: {ContactPoint}, normal: {Normal})";
}
