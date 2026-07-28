using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

// Layout must match Rust CTransform (position, rotation xyzw, scale — ten
// sequential floats; Orrin.Math types are layout-compatible by construction).
[StructLayout(LayoutKind.Sequential)]
public struct Transform
{
    public Vector3 Position;
    public Quaternion Rotation;
    public Vector3 Scale;

    public static Transform Identity => new()
    {
        Position = Vector3.zero,
        Rotation = Quaternion.identity,
        Scale = Vector3.one,
    };
}
