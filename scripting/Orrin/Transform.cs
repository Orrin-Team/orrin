using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

// Layout must match Rust CTransform (position, rotation xyzw, scale — ten
// sequential floats; Orrin.Math types are layout-compatible by construction).
/// <summary>
/// A position, rotation and scale. Whether it is relative to a parent or to the
/// world is a matter of where it came from: <see cref="Entity.LocalTransform"/>
/// or <see cref="Entity.WorldTransform"/>.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct Transform
{
    /// <summary>In metres.</summary>
    public Vector3 Position;
    /// <summary>Which way it faces: the rotation of <see cref="Vector3.forward"/> (-Z).</summary>
    public Quaternion Rotation;
    /// <summary>Per axis; one is the mesh's own size.</summary>
    public Vector3 Scale;

    /// <summary>
    /// At the origin, unrotated, at a scale of one. Start from this rather than
    /// <c>default</c>, whose rotation and scale are zero.
    /// </summary>
    public static Transform Identity => new()
    {
        Position = Vector3.zero,
        Rotation = Quaternion.identity,
        Scale = Vector3.one,
    };
}
