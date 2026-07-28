using System.Globalization;
using System.Runtime.InteropServices;

namespace Orrin.Math;

/// <summary>A 4D vector of floats. Immutable; operations return new values.</summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Vector4 : IEquatable<Vector4>
{
    public readonly float x;
    public readonly float y;
    public readonly float z;
    public readonly float w;

    public Vector4(float x, float y, float z, float w)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    public Vector4(Vector3 xyz, float w) : this(xyz.x, xyz.y, xyz.z, w) { }

    public static Vector4 zero => new(0f, 0f, 0f, 0f);
    public static Vector4 one => new(1f, 1f, 1f, 1f);

    public float magnitude => MathF.Sqrt(x * x + y * y + z * z + w * w);

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

    public static float Dot(Vector4 a, Vector4 b) =>
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

    public static float Distance(Vector4 a, Vector4 b) => (b - a).magnitude;

    /// <summary>Linear interpolation; <paramref name="t"/> is clamped to [0, 1].</summary>
    public static Vector4 Lerp(Vector4 a, Vector4 b, float t) =>
        LerpUnclamped(a, b, Mathf.Clamp01(t));

    public static Vector4 LerpUnclamped(Vector4 a, Vector4 b, float t) => new(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t);

    /// <summary>Component-wise multiplication.</summary>
    public static Vector4 Scale(Vector4 a, Vector4 b) =>
        new(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);

    public static Vector4 Min(Vector4 a, Vector4 b) => new(
        MathF.Min(a.x, b.x), MathF.Min(a.y, b.y), MathF.Min(a.z, b.z), MathF.Min(a.w, b.w));

    public static Vector4 Max(Vector4 a, Vector4 b) => new(
        MathF.Max(a.x, b.x), MathF.Max(a.y, b.y), MathF.Max(a.z, b.z), MathF.Max(a.w, b.w));

    public static Vector4 operator +(Vector4 a, Vector4 b) =>
        new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);

    public static Vector4 operator -(Vector4 a, Vector4 b) =>
        new(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);

    public static Vector4 operator -(Vector4 v) => new(-v.x, -v.y, -v.z, -v.w);
    public static Vector4 operator *(Vector4 v, float s) => new(v.x * s, v.y * s, v.z * s, v.w * s);
    public static Vector4 operator *(float s, Vector4 v) => new(v.x * s, v.y * s, v.z * s, v.w * s);
    public static Vector4 operator /(Vector4 v, float s) => new(v.x / s, v.y / s, v.z / s, v.w / s);

    public static bool operator ==(Vector4 a, Vector4 b) =>
        a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;

    public static bool operator !=(Vector4 a, Vector4 b) => !(a == b);

    public static explicit operator Vector3(Vector4 v) => new(v.x, v.y, v.z);

    public bool Equals(Vector4 other) => this == other;
    public override bool Equals(object? obj) => obj is Vector4 other && this == other;
    public override int GetHashCode() => HashCode.Combine(x, y, z, w);

    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"({x:F2}, {y:F2}, {z:F2}, {w:F2})");
}
