using System.Globalization;
using System.Runtime.InteropServices;

namespace Orrin.Math;

/// <summary>
/// A 3D vector of floats. Immutable; operations return new values.
///
/// Orrin is right-handed with -Z as forward (matching the Rust side's glam
/// conventions), so unlike Unity, <see cref="forward"/> is (0, 0, -1) — "into
/// the screen" from the default camera.
///
/// Layout is three sequential floats; it crosses the native ABI inside
/// <c>Transform</c>, so the field order must not change.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Vector3 : IEquatable<Vector3>
{
    /// <summary>The x component; +x is right.</summary>
    public readonly float x;
    /// <summary>The y component; +y is up.</summary>
    public readonly float y;
    /// <summary>The z component; +z is back, toward the viewer, since forward is -z.</summary>
    public readonly float z;

    /// <summary>A vector from its three components.</summary>
    public Vector3(float x, float y, float z)
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /// <summary>(<paramref name="x"/>, <paramref name="y"/>, 0).</summary>
    public Vector3(float x, float y) : this(x, y, 0f) { }

    /// <summary>(0, 0, 0).</summary>
    public static Vector3 zero => new(0f, 0f, 0f);
    /// <summary>(1, 1, 1).</summary>
    public static Vector3 one => new(1f, 1f, 1f);
    /// <summary>(0, 1, 0).</summary>
    public static Vector3 up => new(0f, 1f, 0f);
    /// <summary>(0, -1, 0).</summary>
    public static Vector3 down => new(0f, -1f, 0f);
    /// <summary>(-1, 0, 0).</summary>
    public static Vector3 left => new(-1f, 0f, 0f);
    /// <summary>(1, 0, 0).</summary>
    public static Vector3 right => new(1f, 0f, 0f);
    /// <summary>(0, 0, -1): Orrin is right-handed, -Z forward.</summary>
    public static Vector3 forward => new(0f, 0f, -1f);
    /// <summary>(0, 0, 1): the opposite of <see cref="forward"/>, toward the viewer.</summary>
    public static Vector3 back => new(0f, 0f, 1f);

    /// <summary>The length: the square root of <see cref="sqrMagnitude"/>.</summary>
    public float magnitude => MathF.Sqrt(x * x + y * y + z * z);

    /// <summary>
    /// The length squared: cheaper than <see cref="magnitude"/>, and enough to
    /// compare two lengths.
    /// </summary>
    public float sqrMagnitude => x * x + y * y + z * z;

    /// <summary>This vector with length 1, or zero if it is too small to normalize.</summary>
    public Vector3 normalized
    {
        get
        {
            var m = magnitude;
            return m > Mathf.Epsilon ? this / m : zero;
        }
    }

    /// <summary>The dot product: the product of the lengths and the cosine of the angle between them.</summary>
    public static float Dot(Vector3 a, Vector3 b) => a.x * b.x + a.y * b.y + a.z * b.z;

    /// <summary>Right-handed cross product.</summary>
    public static Vector3 Cross(Vector3 a, Vector3 b) => new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);

    /// <summary>How far apart two points are: the length of <c>b - a</c>.</summary>
    public static float Distance(Vector3 a, Vector3 b) => (b - a).magnitude;

    /// <summary>Linear interpolation; <paramref name="t"/> is clamped to [0, 1].</summary>
    public static Vector3 Lerp(Vector3 a, Vector3 b, float t) =>
        LerpUnclamped(a, b, Mathf.Clamp01(t));

    /// <summary>
    /// <see cref="Lerp"/> without the clamp: a <paramref name="t"/> outside [0, 1]
    /// extrapolates.
    /// </summary>
    public static Vector3 LerpUnclamped(Vector3 a, Vector3 b, float t) =>
        new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);

    /// <summary>Step from current toward target by at most <paramref name="maxDistanceDelta"/>.</summary>
    public static Vector3 MoveTowards(Vector3 current, Vector3 target, float maxDistanceDelta)
    {
        var to = target - current;
        var distance = to.magnitude;
        if (distance <= maxDistanceDelta || distance < Mathf.Epsilon)
            return target;
        return current + to / distance * maxDistanceDelta;
    }

    /// <summary>Component-wise multiplication.</summary>
    public static Vector3 Scale(Vector3 a, Vector3 b) => new(a.x * b.x, a.y * b.y, a.z * b.z);

    /// <summary>The smaller of each component.</summary>
    public static Vector3 Min(Vector3 a, Vector3 b) =>
        new(MathF.Min(a.x, b.x), MathF.Min(a.y, b.y), MathF.Min(a.z, b.z));

    /// <summary>The larger of each component.</summary>
    public static Vector3 Max(Vector3 a, Vector3 b) =>
        new(MathF.Max(a.x, b.x), MathF.Max(a.y, b.y), MathF.Max(a.z, b.z));

    /// <summary>Project a vector onto another vector.</summary>
    public static Vector3 Project(Vector3 vector, Vector3 onNormal)
    {
        var sqr = onNormal.sqrMagnitude;
        if (sqr < Mathf.Epsilon)
            return zero;
        return onNormal * (Dot(vector, onNormal) / sqr);
    }

    /// <summary>Project a vector onto the plane defined by a normal.</summary>
    public static Vector3 ProjectOnPlane(Vector3 vector, Vector3 planeNormal) =>
        vector - Project(vector, planeNormal);

    /// <summary>Reflect a vector off a surface with the given normal.</summary>
    public static Vector3 Reflect(Vector3 inDirection, Vector3 inNormal) =>
        inDirection - 2f * Dot(inDirection, inNormal) * inNormal;

    /// <summary>Unsigned angle between two vectors, in degrees.</summary>
    public static float Angle(Vector3 from, Vector3 to)
    {
        var denominator = MathF.Sqrt(from.sqrMagnitude * to.sqrMagnitude);
        if (denominator < Mathf.Epsilon)
            return 0f;
        var dot = Mathf.Clamp(Dot(from, to) / denominator, -1f, 1f);
        return MathF.Acos(dot) * Mathf.Rad2Deg;
    }

    /// <summary>Signed angle from one vector to another around an axis, in degrees.</summary>
    public static float SignedAngle(Vector3 from, Vector3 to, Vector3 axis)
    {
        var unsigned = Angle(from, to);
        var sign = Mathf.Sign(Dot(axis, Cross(from, to)));
        return unsigned * sign;
    }

    /// <summary>Clamp the vector's length to <paramref name="maxLength"/>.</summary>
    public static Vector3 ClampMagnitude(Vector3 vector, float maxLength)
    {
        var sqr = vector.sqrMagnitude;
        if (sqr <= maxLength * maxLength)
            return vector;
        return vector / MathF.Sqrt(sqr) * maxLength;
    }

    /// <summary>Component-wise sum.</summary>
    public static Vector3 operator +(Vector3 a, Vector3 b) => new(a.x + b.x, a.y + b.y, a.z + b.z);
    /// <summary>Component-wise difference.</summary>
    public static Vector3 operator -(Vector3 a, Vector3 b) => new(a.x - b.x, a.y - b.y, a.z - b.z);
    /// <summary>Every component negated.</summary>
    public static Vector3 operator -(Vector3 v) => new(-v.x, -v.y, -v.z);
    /// <summary>Every component multiplied by <paramref name="s"/>.</summary>
    public static Vector3 operator *(Vector3 v, float s) => new(v.x * s, v.y * s, v.z * s);
    /// <summary>Every component multiplied by <paramref name="s"/>.</summary>
    public static Vector3 operator *(float s, Vector3 v) => new(v.x * s, v.y * s, v.z * s);
    /// <summary>Every component divided by <paramref name="s"/>.</summary>
    public static Vector3 operator /(Vector3 v, float s) => new(v.x / s, v.y / s, v.z / s);

    /// <summary>Exact equality of every component, with no tolerance.</summary>
    public static bool operator ==(Vector3 a, Vector3 b) => a.x == b.x && a.y == b.y && a.z == b.z;
    /// <summary>Whether any component differs.</summary>
    public static bool operator !=(Vector3 a, Vector3 b) => !(a == b);

    /// <inheritdoc/>
    public bool Equals(Vector3 other) => this == other;
    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Vector3 other && this == other;
    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(x, y, z);

    /// <summary>The components to two decimal places, in the invariant culture.</summary>
    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"({x:F2}, {y:F2}, {z:F2})");
}
