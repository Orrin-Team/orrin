using System.Globalization;
using System.Runtime.InteropServices;

namespace Orrin.Math;

/// <summary>A 2D vector of floats. Immutable; operations return new values.</summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Vector2 : IEquatable<Vector2>
{
    public readonly float x;
    public readonly float y;

    public Vector2(float x, float y)
    {
        this.x = x;
        this.y = y;
    }

    public static Vector2 zero => new(0f, 0f);
    public static Vector2 one => new(1f, 1f);
    public static Vector2 up => new(0f, 1f);
    public static Vector2 down => new(0f, -1f);
    public static Vector2 left => new(-1f, 0f);
    public static Vector2 right => new(1f, 0f);

    public float magnitude => MathF.Sqrt(x * x + y * y);

    public float sqrMagnitude => x * x + y * y;

    /// <summary>This vector with length 1, or zero if it is too small to normalize.</summary>
    public Vector2 normalized
    {
        get
        {
            var m = magnitude;
            return m > Mathf.Epsilon ? this / m : zero;
        }
    }

    public static float Dot(Vector2 a, Vector2 b) => a.x * b.x + a.y * b.y;

    public static float Distance(Vector2 a, Vector2 b) => (b - a).magnitude;

    /// <summary>Linear interpolation; <paramref name="t"/> is clamped to [0, 1].</summary>
    public static Vector2 Lerp(Vector2 a, Vector2 b, float t) =>
        LerpUnclamped(a, b, Mathf.Clamp01(t));

    public static Vector2 LerpUnclamped(Vector2 a, Vector2 b, float t) =>
        new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);

    /// <summary>Step from current toward target by at most <paramref name="maxDistanceDelta"/>.</summary>
    public static Vector2 MoveTowards(Vector2 current, Vector2 target, float maxDistanceDelta)
    {
        var to = target - current;
        var distance = to.magnitude;
        if (distance <= maxDistanceDelta || distance < Mathf.Epsilon)
            return target;
        return current + to / distance * maxDistanceDelta;
    }

    /// <summary>Component-wise multiplication.</summary>
    public static Vector2 Scale(Vector2 a, Vector2 b) => new(a.x * b.x, a.y * b.y);

    public static Vector2 Min(Vector2 a, Vector2 b) =>
        new(MathF.Min(a.x, b.x), MathF.Min(a.y, b.y));

    public static Vector2 Max(Vector2 a, Vector2 b) =>
        new(MathF.Max(a.x, b.x), MathF.Max(a.y, b.y));

    /// <summary>Reflect a vector off a surface with the given normal.</summary>
    public static Vector2 Reflect(Vector2 inDirection, Vector2 inNormal) =>
        inDirection - 2f * Dot(inDirection, inNormal) * inNormal;

    /// <summary>The vector rotated 90 degrees counter-clockwise.</summary>
    public static Vector2 Perpendicular(Vector2 direction) => new(-direction.y, direction.x);

    /// <summary>Unsigned angle between two vectors, in degrees.</summary>
    public static float Angle(Vector2 from, Vector2 to)
    {
        var denominator = MathF.Sqrt(from.sqrMagnitude * to.sqrMagnitude);
        if (denominator < Mathf.Epsilon)
            return 0f;
        var dot = Mathf.Clamp(Dot(from, to) / denominator, -1f, 1f);
        return MathF.Acos(dot) * Mathf.Rad2Deg;
    }

    /// <summary>Clamp the vector's length to <paramref name="maxLength"/>.</summary>
    public static Vector2 ClampMagnitude(Vector2 vector, float maxLength)
    {
        var sqr = vector.sqrMagnitude;
        if (sqr <= maxLength * maxLength)
            return vector;
        return vector / MathF.Sqrt(sqr) * maxLength;
    }

    public static Vector2 operator +(Vector2 a, Vector2 b) => new(a.x + b.x, a.y + b.y);
    public static Vector2 operator -(Vector2 a, Vector2 b) => new(a.x - b.x, a.y - b.y);
    public static Vector2 operator -(Vector2 v) => new(-v.x, -v.y);
    public static Vector2 operator *(Vector2 v, float s) => new(v.x * s, v.y * s);
    public static Vector2 operator *(float s, Vector2 v) => new(v.x * s, v.y * s);
    public static Vector2 operator /(Vector2 v, float s) => new(v.x / s, v.y / s);

    public static bool operator ==(Vector2 a, Vector2 b) => a.x == b.x && a.y == b.y;
    public static bool operator !=(Vector2 a, Vector2 b) => !(a == b);

    public bool Equals(Vector2 other) => this == other;
    public override bool Equals(object? obj) => obj is Vector2 other && this == other;
    public override int GetHashCode() => HashCode.Combine(x, y);

    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"({x:F2}, {y:F2})");
}
