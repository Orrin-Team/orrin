using System.Globalization;
using System.Runtime.InteropServices;

namespace Orrin.Math;

/// <summary>A 4D vector of floats. Immutable; operations return new values.</summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Vector4 : IEquatable<Vector4>
{
    /// <summary>The first component.</summary>
    public readonly float x;
    /// <summary>The second component.</summary>
    public readonly float y;
    /// <summary>The third component.</summary>
    public readonly float z;
    /// <summary>The fourth component.</summary>
    public readonly float w;

    /// <summary>A vector from its four components.</summary>
    public Vector4(float x, float y, float z, float w)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    /// <summary><paramref name="xyz"/>, with <paramref name="w"/> as its fourth component.</summary>
    public Vector4(Vector3 xyz, float w) : this(xyz.x, xyz.y, xyz.z, w) { }

    /// <summary>(0, 0, 0, 0).</summary>
    public static Vector4 zero => new(0f, 0f, 0f, 0f);
    /// <summary>(1, 1, 1, 1).</summary>
    public static Vector4 one => new(1f, 1f, 1f, 1f);

    /// <summary>The length: the square root of <see cref="sqrMagnitude"/>.</summary>
    public float magnitude => MathF.Sqrt(x * x + y * y + z * z + w * w);

    /// <summary>
    /// The length squared: cheaper than <see cref="magnitude"/>, and enough to
    /// compare two lengths.
    /// </summary>
    public float sqrMagnitude => x * x + y * y + z * z + w * w;

    /// <summary>This vector with length 1, or zero if it is too small to normalize.</summary>
    public Vector4 normalized
    {
        get
        {
            var m = magnitude;
            return m > Mathf.Epsilon ? this / m : zero;
        }
    }

    /// <summary>The dot product: the product of the lengths and the cosine of the angle between them.</summary>
    public static float Dot(Vector4 a, Vector4 b) =>
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

    /// <summary>How far apart two points are: the length of <c>b - a</c>.</summary>
    public static float Distance(Vector4 a, Vector4 b) => (b - a).magnitude;

    /// <summary>Linear interpolation; <paramref name="t"/> is clamped to [0, 1].</summary>
    public static Vector4 Lerp(Vector4 a, Vector4 b, float t) =>
        LerpUnclamped(a, b, Mathf.Clamp01(t));

    /// <summary>
    /// <see cref="Lerp"/> without the clamp: a <paramref name="t"/> outside [0, 1]
    /// extrapolates.
    /// </summary>
    public static Vector4 LerpUnclamped(Vector4 a, Vector4 b, float t) => new(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t);

    /// <summary>Component-wise multiplication.</summary>
    public static Vector4 Scale(Vector4 a, Vector4 b) =>
        new(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);

    /// <summary>The smaller of each component.</summary>
    public static Vector4 Min(Vector4 a, Vector4 b) => new(
        MathF.Min(a.x, b.x), MathF.Min(a.y, b.y), MathF.Min(a.z, b.z), MathF.Min(a.w, b.w));

    /// <summary>The larger of each component.</summary>
    public static Vector4 Max(Vector4 a, Vector4 b) => new(
        MathF.Max(a.x, b.x), MathF.Max(a.y, b.y), MathF.Max(a.z, b.z), MathF.Max(a.w, b.w));

    /// <summary>Component-wise sum.</summary>
    public static Vector4 operator +(Vector4 a, Vector4 b) =>
        new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);

    /// <summary>Component-wise difference.</summary>
    public static Vector4 operator -(Vector4 a, Vector4 b) =>
        new(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);

    /// <summary>Every component negated.</summary>
    public static Vector4 operator -(Vector4 v) => new(-v.x, -v.y, -v.z, -v.w);
    /// <summary>Every component multiplied by <paramref name="s"/>.</summary>
    public static Vector4 operator *(Vector4 v, float s) => new(v.x * s, v.y * s, v.z * s, v.w * s);
    /// <summary>Every component multiplied by <paramref name="s"/>.</summary>
    public static Vector4 operator *(float s, Vector4 v) => new(v.x * s, v.y * s, v.z * s, v.w * s);
    /// <summary>Every component divided by <paramref name="s"/>.</summary>
    public static Vector4 operator /(Vector4 v, float s) => new(v.x / s, v.y / s, v.z / s, v.w / s);

    /// <summary>Exact equality of every component, with no tolerance.</summary>
    public static bool operator ==(Vector4 a, Vector4 b) =>
        a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;

    /// <summary>Whether any component differs.</summary>
    public static bool operator !=(Vector4 a, Vector4 b) => !(a == b);

    /// <summary>The first three components; <see cref="w"/> is dropped.</summary>
    public static explicit operator Vector3(Vector4 v) => new(v.x, v.y, v.z);

    /// <inheritdoc/>
    public bool Equals(Vector4 other) => this == other;
    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Vector4 other && this == other;
    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(x, y, z, w);

    /// <summary>The components to two decimal places, in the invariant culture.</summary>
    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"({x:F2}, {y:F2}, {z:F2}, {w:F2})");
}
