using System.Globalization;
using System.Runtime.InteropServices;

namespace Orrin.Math;

/// <summary>A 2D vector of floats. Immutable; operations return new values.</summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Vector2 : IEquatable<Vector2>
{
    /// <summary>The horizontal component.</summary>
    public readonly float x;
    /// <summary>
    /// The vertical component. Which way is up depends on the space: window and UI
    /// coordinates run y down.
    /// </summary>
    public readonly float y;

    /// <summary>A vector from its two components.</summary>
    public Vector2(float x, float y)
    {
        this.x = x;
        this.y = y;
    }

    /// <summary>(0, 0).</summary>
    public static Vector2 zero => new(0f, 0f);
    /// <summary>(1, 1).</summary>
    public static Vector2 one => new(1f, 1f);
    /// <summary>(0, 1).</summary>
    public static Vector2 up => new(0f, 1f);
    /// <summary>(0, -1).</summary>
    public static Vector2 down => new(0f, -1f);
    /// <summary>(-1, 0).</summary>
    public static Vector2 left => new(-1f, 0f);
    /// <summary>(1, 0).</summary>
    public static Vector2 right => new(1f, 0f);

    /// <summary>The length: the square root of <see cref="sqrMagnitude"/>.</summary>
    public float magnitude => MathF.Sqrt(x * x + y * y);

    /// <summary>
    /// The length squared: cheaper than <see cref="magnitude"/>, and enough to
    /// compare two lengths.
    /// </summary>
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

    /// <summary>The dot product: the product of the lengths and the cosine of the angle between them.</summary>
    public static float Dot(Vector2 a, Vector2 b) => a.x * b.x + a.y * b.y;

    /// <summary>How far apart two points are: the length of <c>b - a</c>.</summary>
    public static float Distance(Vector2 a, Vector2 b) => (b - a).magnitude;

    /// <summary>Linear interpolation; <paramref name="t"/> is clamped to [0, 1].</summary>
    public static Vector2 Lerp(Vector2 a, Vector2 b, float t) =>
        LerpUnclamped(a, b, Mathf.Clamp01(t));

    /// <summary>
    /// <see cref="Lerp"/> without the clamp: a <paramref name="t"/> outside [0, 1]
    /// extrapolates.
    /// </summary>
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

    /// <summary>The smaller of each component.</summary>
    public static Vector2 Min(Vector2 a, Vector2 b) =>
        new(MathF.Min(a.x, b.x), MathF.Min(a.y, b.y));

    /// <summary>The larger of each component.</summary>
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

    /// <summary>Component-wise sum.</summary>
    public static Vector2 operator +(Vector2 a, Vector2 b) => new(a.x + b.x, a.y + b.y);
    /// <summary>Component-wise difference.</summary>
    public static Vector2 operator -(Vector2 a, Vector2 b) => new(a.x - b.x, a.y - b.y);
    /// <summary>Every component negated.</summary>
    public static Vector2 operator -(Vector2 v) => new(-v.x, -v.y);
    /// <summary>Every component multiplied by <paramref name="s"/>.</summary>
    public static Vector2 operator *(Vector2 v, float s) => new(v.x * s, v.y * s);
    /// <summary>Every component multiplied by <paramref name="s"/>.</summary>
    public static Vector2 operator *(float s, Vector2 v) => new(v.x * s, v.y * s);
    /// <summary>Every component divided by <paramref name="s"/>.</summary>
    public static Vector2 operator /(Vector2 v, float s) => new(v.x / s, v.y / s);

    /// <summary>Exact equality of every component, with no tolerance.</summary>
    public static bool operator ==(Vector2 a, Vector2 b) => a.x == b.x && a.y == b.y;
    /// <summary>Whether any component differs.</summary>
    public static bool operator !=(Vector2 a, Vector2 b) => !(a == b);

    /// <inheritdoc/>
    public bool Equals(Vector2 other) => this == other;
    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Vector2 other && this == other;
    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(x, y);

    /// <summary>The components to two decimal places, in the invariant culture.</summary>
    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"({x:F2}, {y:F2})");
}
