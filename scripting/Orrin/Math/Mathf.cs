namespace Orrin.Math;

/// <summary>
/// Float math helpers for gameplay code. Angles at this API surface are in
/// degrees unless a name says otherwise; use <see cref="Deg2Rad"/> /
/// <see cref="Rad2Deg"/> at the boundary with radian-based math.
/// </summary>
public static class Mathf
{
    /// <summary>π, to float precision.</summary>
    public const float PI = MathF.PI;
    /// <summary>Degrees times this are radians.</summary>
    public const float Deg2Rad = PI / 180f;
    /// <summary>Radians times this are degrees.</summary>
    public const float Rad2Deg = 180f / PI;
    /// <summary>A small value for approximate comparisons (not float.Epsilon).</summary>
    public const float Epsilon = 1e-6f;
    /// <summary><see cref="float.PositiveInfinity"/>.</summary>
    public const float Infinity = float.PositiveInfinity;
    /// <summary><see cref="float.NegativeInfinity"/>.</summary>
    public const float NegativeInfinity = float.NegativeInfinity;

    /// <summary>The absolute value.</summary>
    public static float Abs(float value) => MathF.Abs(value);
    /// <summary>The smaller of the two.</summary>
    public static float Min(float a, float b) => MathF.Min(a, b);
    /// <summary>The larger of the two.</summary>
    public static float Max(float a, float b) => MathF.Max(a, b);
    /// <summary>The square root; NaN for a negative <paramref name="value"/>.</summary>
    public static float Sqrt(float value) => MathF.Sqrt(value);
    /// <summary><paramref name="value"/> raised to <paramref name="power"/>.</summary>
    public static float Pow(float value, float power) => MathF.Pow(value, power);
    /// <summary>e raised to <paramref name="power"/>.</summary>
    public static float Exp(float power) => MathF.Exp(power);
    /// <summary>The natural logarithm, base e.</summary>
    public static float Log(float value) => MathF.Log(value);

    /// <summary>The sine of an angle in radians, not degrees.</summary>
    public static float Sin(float radians) => MathF.Sin(radians);
    /// <summary>The cosine of an angle in radians, not degrees.</summary>
    public static float Cos(float radians) => MathF.Cos(radians);
    /// <summary>The tangent of an angle in radians, not degrees.</summary>
    public static float Tan(float radians) => MathF.Tan(radians);
    /// <summary>The angle in radians whose sine is <paramref name="value"/>; NaN outside [-1, 1].</summary>
    public static float Asin(float value) => MathF.Asin(value);
    /// <summary>The angle in radians whose cosine is <paramref name="value"/>; NaN outside [-1, 1].</summary>
    public static float Acos(float value) => MathF.Acos(value);
    /// <summary>The angle in radians, between -π/2 and π/2, whose tangent is <paramref name="value"/>.</summary>
    public static float Atan(float value) => MathF.Atan(value);
    /// <summary>
    /// The angle in radians, between -π and π, from the +x axis to the point
    /// (<paramref name="x"/>, <paramref name="y"/>).
    /// </summary>
    public static float Atan2(float y, float x) => MathF.Atan2(y, x);

    /// <summary>The largest whole number at or below <paramref name="value"/>.</summary>
    public static float Floor(float value) => MathF.Floor(value);
    /// <summary>The smallest whole number at or above <paramref name="value"/>.</summary>
    public static float Ceil(float value) => MathF.Ceiling(value);
    /// <summary>The nearest whole number; a value exactly halfway rounds to the even one.</summary>
    public static float Round(float value) => MathF.Round(value);
    /// <summary><see cref="Floor"/>, as an int.</summary>
    public static int FloorToInt(float value) => (int)MathF.Floor(value);
    /// <summary><see cref="Ceil"/>, as an int.</summary>
    public static int CeilToInt(float value) => (int)MathF.Ceiling(value);
    /// <summary><see cref="Round"/>, halfway to even, as an int.</summary>
    public static int RoundToInt(float value) => (int)MathF.Round(value);

    /// <summary>1 for zero or positive values, -1 for negative (Unity convention).</summary>
    public static float Sign(float value) => value >= 0f ? 1f : -1f;

    /// <summary><paramref name="value"/>, held between <paramref name="min"/> and <paramref name="max"/>.</summary>
    public static float Clamp(float value, float min, float max) =>
        value < min ? min : value > max ? max : value;

    /// <summary><paramref name="value"/>, held between <paramref name="min"/> and <paramref name="max"/>.</summary>
    public static int Clamp(int value, int min, int max) =>
        value < min ? min : value > max ? max : value;

    /// <summary><paramref name="value"/>, held between 0 and 1.</summary>
    public static float Clamp01(float value) => Clamp(value, 0f, 1f);

    /// <summary>Linear interpolation; <paramref name="t"/> is clamped to [0, 1].</summary>
    public static float Lerp(float a, float b, float t) => a + (b - a) * Clamp01(t);

    /// <summary>
    /// <see cref="Lerp"/> without the clamp: a <paramref name="t"/> outside [0, 1]
    /// extrapolates.
    /// </summary>
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
