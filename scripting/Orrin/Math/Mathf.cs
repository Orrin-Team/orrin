namespace Orrin.Math;

/// <summary>
/// Float math helpers for gameplay code. Angles at this API surface are in
/// degrees unless a name says otherwise; use <see cref="Deg2Rad"/> /
/// <see cref="Rad2Deg"/> at the boundary with radian-based math.
/// </summary>
public static class Mathf
{
    public const float PI = MathF.PI;
    public const float Deg2Rad = PI / 180f;
    public const float Rad2Deg = 180f / PI;
    /// <summary>A small value for approximate comparisons (not float.Epsilon).</summary>
    public const float Epsilon = 1e-6f;
    public const float Infinity = float.PositiveInfinity;
    public const float NegativeInfinity = float.NegativeInfinity;

    public static float Abs(float value) => MathF.Abs(value);
    public static float Min(float a, float b) => MathF.Min(a, b);
    public static float Max(float a, float b) => MathF.Max(a, b);
    public static float Sqrt(float value) => MathF.Sqrt(value);
    public static float Pow(float value, float power) => MathF.Pow(value, power);
    public static float Exp(float power) => MathF.Exp(power);
    public static float Log(float value) => MathF.Log(value);

    public static float Sin(float radians) => MathF.Sin(radians);
    public static float Cos(float radians) => MathF.Cos(radians);
    public static float Tan(float radians) => MathF.Tan(radians);
    public static float Asin(float value) => MathF.Asin(value);
    public static float Acos(float value) => MathF.Acos(value);
    public static float Atan(float value) => MathF.Atan(value);
    public static float Atan2(float y, float x) => MathF.Atan2(y, x);

    public static float Floor(float value) => MathF.Floor(value);
    public static float Ceil(float value) => MathF.Ceiling(value);
    public static float Round(float value) => MathF.Round(value);
    public static int FloorToInt(float value) => (int)MathF.Floor(value);
    public static int CeilToInt(float value) => (int)MathF.Ceiling(value);
    public static int RoundToInt(float value) => (int)MathF.Round(value);

    /// <summary>1 for zero or positive values, -1 for negative (Unity convention).</summary>
    public static float Sign(float value) => value >= 0f ? 1f : -1f;

    public static float Clamp(float value, float min, float max) =>
        value < min ? min : value > max ? max : value;

    public static int Clamp(int value, int min, int max) =>
        value < min ? min : value > max ? max : value;

    public static float Clamp01(float value) => Clamp(value, 0f, 1f);

    /// <summary>Linear interpolation; <paramref name="t"/> is clamped to [0, 1].</summary>
    public static float Lerp(float a, float b, float t) => a + (b - a) * Clamp01(t);

    public static float LerpUnclamped(float a, float b, float t) => a + (b - a) * t;

    /// <summary>Where <paramref name="value"/> sits between a and b, clamped to [0, 1].</summary>
    public static float InverseLerp(float a, float b, float value) =>
        a != b ? Clamp01((value - a) / (b - a)) : 0f;

    /// <summary>Step from current toward target by at most <paramref name="maxDelta"/>.</summary>
    public static float MoveTowards(float current, float target, float maxDelta) =>
        Abs(target - current) <= maxDelta ? target : current + Sign(target - current) * maxDelta;

    /// <summary>Hermite-smoothed interpolation between from and to.</summary>
    public static float SmoothStep(float from, float to, float t)
    {
        t = Clamp01(t);
        t = t * t * (3f - 2f * t);
        return from + (to - from) * t;
    }

    /// <summary>Wrap <paramref name="t"/> into [0, length) (like mod, but never negative).</summary>
    public static float Repeat(float t, float length) =>
        Clamp(t - Floor(t / length) * length, 0f, length);

    /// <summary>Bounce <paramref name="t"/> back and forth in [0, length].</summary>
    public static float PingPong(float t, float length)
    {
        t = Repeat(t, length * 2f);
        return length - Abs(t - length);
    }

    /// <summary>Shortest signed difference between two angles, in degrees (-180, 180].</summary>
    public static float DeltaAngle(float current, float target)
    {
        var delta = Repeat(target - current, 360f);
        if (delta > 180f)
            delta -= 360f;
        return delta;
    }

    /// <summary>Lerp between angles in degrees, taking the shortest path around the circle.</summary>
    public static float LerpAngle(float a, float b, float t) => a + DeltaAngle(a, b) * Clamp01(t);

    /// <summary>Approximate equality that scales with the magnitude of the operands.</summary>
    public static bool Approximately(float a, float b) =>
        Abs(b - a) < Max(Epsilon * Max(Abs(a), Abs(b)), Epsilon);
}
