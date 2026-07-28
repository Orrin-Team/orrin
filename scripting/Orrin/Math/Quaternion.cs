using System.Globalization;
using System.Runtime.InteropServices;

namespace Orrin.Math;

/// <summary>
/// A rotation, stored as x/y/z/w (vector part, then scalar — the same layout as
/// the Rust side's <c>[f32; 4]</c> in <c>CTransform</c>, so the field order
/// must not change).
///
/// Conventions: angles at the API surface are in degrees; Euler angles apply in
/// Z-X-Y order (roll, then pitch, then yaw — matching Unity); the canonical
/// "look" direction is <see cref="Vector3.forward"/> = (0, 0, -1).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Quaternion : IEquatable<Quaternion>
{
    public readonly float x;
    public readonly float y;
    public readonly float z;
    public readonly float w;

    public Quaternion(float x, float y, float z, float w)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = w;
    }

    public static Quaternion identity => new(0f, 0f, 0f, 1f);

    public Quaternion normalized
    {
        get
        {
            var m = MathF.Sqrt(x * x + y * y + z * z + w * w);
            return m > Mathf.Epsilon
                ? new Quaternion(x / m, y / m, z / m, w / m)
                : identity;
        }
    }

    /// <summary>
    /// The rotation as Euler angles in degrees (Z-X-Y application order),
    /// each wrapped into [0, 360). Note that Euler representations are not
    /// unique; <c>Euler(q.eulerAngles)</c> is the same rotation as <c>q</c>,
    /// but not necessarily the same three numbers you built it from.
    /// </summary>
    public Vector3 eulerAngles
    {
        get
        {
            // Rotation-matrix elements of the normalized quaternion, YXZ-decomposed.
            var q = normalized;
            var m12 = 2f * (q.y * q.z - q.w * q.x);

            float pitch, yaw, roll;
            if (MathF.Abs(m12) < 0.9999f)
            {
                pitch = MathF.Asin(-m12);
                yaw = MathF.Atan2(2f * (q.x * q.z + q.w * q.y), 1f - 2f * (q.x * q.x + q.y * q.y));
                roll = MathF.Atan2(2f * (q.x * q.y + q.w * q.z), 1f - 2f * (q.x * q.x + q.z * q.z));
            }
            else
            {
                // Gimbal lock (looking straight up/down): yaw and roll combine,
                // so put the whole twist in yaw.
                pitch = m12 <= -0.9999f ? MathF.PI / 2f : -MathF.PI / 2f;
                yaw = MathF.Atan2(-2f * (q.x * q.z - q.w * q.y), 1f - 2f * (q.y * q.y + q.z * q.z));
                roll = 0f;
            }

            return new Vector3(
                Mathf.Repeat(pitch * Mathf.Rad2Deg, 360f),
                Mathf.Repeat(yaw * Mathf.Rad2Deg, 360f),
                Mathf.Repeat(roll * Mathf.Rad2Deg, 360f));
        }
    }

    /// <summary>Rotation of <paramref name="angleDegrees"/> around <paramref name="axis"/>.</summary>
    public static Quaternion AngleAxis(float angleDegrees, Vector3 axis)
    {
        var unit = axis.normalized;
        if (unit == Vector3.zero)
            return identity;
        var half = angleDegrees * Mathf.Deg2Rad * 0.5f;
        var s = MathF.Sin(half);
        return new Quaternion(unit.x * s, unit.y * s, unit.z * s, MathF.Cos(half));
    }

    /// <summary>
    /// Rotation from Euler angles in degrees, applied Z (roll), then X (pitch),
    /// then Y (yaw) — the same convention as Unity.
    /// </summary>
    public static Quaternion Euler(float xDegrees, float yDegrees, float zDegrees)
    {
        // Each angle rotates about the positive coordinate axis (same numeric
        // convention as Unity and System.Numerics.CreateFromYawPitchRoll).
        var qx = AngleAxis(xDegrees, Vector3.right);
        var qy = AngleAxis(yDegrees, Vector3.up);
        var qz = AngleAxis(zDegrees, new Vector3(0f, 0f, 1f));
        // Applied right-to-left: roll first, then pitch, then yaw.
        return qy * qx * qz;
    }

    public static Quaternion Euler(Vector3 eulerDegrees) =>
        Euler(eulerDegrees.x, eulerDegrees.y, eulerDegrees.z);

    /// <summary>
    /// Rotation that turns <see cref="Vector3.forward"/> (-Z) to point along
    /// <paramref name="forward"/>, keeping "up" as close to <paramref name="up"/>
    /// as possible. Returns identity if <paramref name="forward"/> is zero.
    /// </summary>
    public static Quaternion LookRotation(Vector3 forward, Vector3 up)
    {
        var f = forward.normalized;
        if (f == Vector3.zero)
            return identity;

        // Right-handed basis with -Z looking down `forward`.
        var zAxis = -f;
        var xAxis = Vector3.Cross(up, zAxis);
        if (xAxis.sqrMagnitude < Mathf.Epsilon)
        {
            // forward is parallel to up (looking straight up/down); any
            // horizontal axis works, so derive one from the world forward.
            xAxis = Vector3.Cross(Vector3.forward, zAxis);
        }
        xAxis = xAxis.normalized;
        var yAxis = Vector3.Cross(zAxis, xAxis);

        return FromBasis(xAxis, yAxis, zAxis);
    }

    public static Quaternion LookRotation(Vector3 forward) => LookRotation(forward, Vector3.up);

    /// <summary>The shortest-arc rotation taking one direction to another.</summary>
    public static Quaternion FromToRotation(Vector3 from, Vector3 to)
    {
        var f = from.normalized;
        var t = to.normalized;
        if (f == Vector3.zero || t == Vector3.zero)
            return identity;

        var dot = Vector3.Dot(f, t);
        if (dot > 1f - Mathf.Epsilon)
            return identity;
        if (dot < -1f + Mathf.Epsilon)
        {
            // Opposite directions: 180 degrees around any axis orthogonal to `f`.
            var axis = Vector3.Cross(Vector3.right, f);
            if (axis.sqrMagnitude < Mathf.Epsilon)
                axis = Vector3.Cross(Vector3.up, f);
            return AngleAxis(180f, axis);
        }

        var cross = Vector3.Cross(f, t);
        return new Quaternion(cross.x, cross.y, cross.z, 1f + dot).normalized;
    }

    public static float Dot(Quaternion a, Quaternion b) =>
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

    /// <summary>The reverse rotation (assumes a unit quaternion).</summary>
    public static Quaternion Inverse(Quaternion q) => new(-q.x, -q.y, -q.z, q.w);

    public static Quaternion Normalize(Quaternion q) => q.normalized;

    /// <summary>Unsigned angle between two rotations, in degrees.</summary>
    public static float Angle(Quaternion a, Quaternion b)
    {
        var dot = MathF.Min(MathF.Abs(Dot(a, b)), 1f);
        // acos is ill-conditioned near 1: float noise in the dot product would
        // read as a fraction of a degree, so snap near-identical rotations to 0.
        return dot > 1f - Mathf.Epsilon ? 0f : 2f * MathF.Acos(dot) * Mathf.Rad2Deg;
    }

    /// <summary>Spherical interpolation; <paramref name="t"/> is clamped to [0, 1].</summary>
    public static Quaternion Slerp(Quaternion a, Quaternion b, float t) =>
        SlerpUnclamped(a, b, Mathf.Clamp01(t));

    public static Quaternion SlerpUnclamped(Quaternion a, Quaternion b, float t)
    {
        var dot = Dot(a, b);
        // Take the short way around: q and -q are the same rotation.
        if (dot < 0f)
        {
            b = new Quaternion(-b.x, -b.y, -b.z, -b.w);
            dot = -dot;
        }

        float scaleA, scaleB;
        if (dot > 0.9995f)
        {
            // Nearly parallel: slerp degenerates numerically, lerp is exact enough.
            scaleA = 1f - t;
            scaleB = t;
        }
        else
        {
            var theta = MathF.Acos(dot);
            var invSin = 1f / MathF.Sin(theta);
            scaleA = MathF.Sin((1f - t) * theta) * invSin;
            scaleB = MathF.Sin(t * theta) * invSin;
        }

        return new Quaternion(
            a.x * scaleA + b.x * scaleB,
            a.y * scaleA + b.y * scaleB,
            a.z * scaleA + b.z * scaleB,
            a.w * scaleA + b.w * scaleB).normalized;
    }

    /// <summary>Normalized linear interpolation (faster than slerp, slightly uneven speed).</summary>
    public static Quaternion Lerp(Quaternion a, Quaternion b, float t) =>
        LerpUnclamped(a, b, Mathf.Clamp01(t));

    public static Quaternion LerpUnclamped(Quaternion a, Quaternion b, float t)
    {
        if (Dot(a, b) < 0f)
            b = new Quaternion(-b.x, -b.y, -b.z, -b.w);
        return new Quaternion(
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            a.z + (b.z - a.z) * t,
            a.w + (b.w - a.w) * t).normalized;
    }

    /// <summary>Step from one rotation toward another by at most <paramref name="maxDegreesDelta"/>.</summary>
    public static Quaternion RotateTowards(Quaternion from, Quaternion to, float maxDegreesDelta)
    {
        var angle = Angle(from, to);
        if (angle < Mathf.Epsilon)
            return to;
        return SlerpUnclamped(from, to, MathF.Min(1f, maxDegreesDelta / angle));
    }

    /// <summary>Compose rotations: the result applies <paramref name="rhs"/> first, then <paramref name="lhs"/>.</summary>
    public static Quaternion operator *(Quaternion lhs, Quaternion rhs) => new(
        lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.w * rhs.y + lhs.y * rhs.w + lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.w * rhs.z + lhs.z * rhs.w + lhs.x * rhs.y - lhs.y * rhs.x,
        lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z);

    /// <summary>Rotate a vector by this rotation.</summary>
    public static Vector3 operator *(Quaternion rotation, Vector3 point)
    {
        // v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + w * v)
        var qv = new Vector3(rotation.x, rotation.y, rotation.z);
        var t = 2f * Vector3.Cross(qv, point);
        return point + rotation.w * t + Vector3.Cross(qv, t);
    }

    public static bool operator ==(Quaternion a, Quaternion b) =>
        a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;

    public static bool operator !=(Quaternion a, Quaternion b) => !(a == b);

    /// <summary>Quaternion from an orthonormal right-handed basis (matrix columns).</summary>
    private static Quaternion FromBasis(Vector3 xAxis, Vector3 yAxis, Vector3 zAxis)
    {
        // Shepperd's method: pick the largest diagonal term for stability.
        float m00 = xAxis.x, m01 = yAxis.x, m02 = zAxis.x;
        float m10 = xAxis.y, m11 = yAxis.y, m12 = zAxis.y;
        float m20 = xAxis.z, m21 = yAxis.z, m22 = zAxis.z;

        var trace = m00 + m11 + m22;
        if (trace > 0f)
        {
            var s = MathF.Sqrt(trace + 1f) * 2f;
            return new Quaternion((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25f * s);
        }
        if (m00 > m11 && m00 > m22)
        {
            var s = MathF.Sqrt(1f + m00 - m11 - m22) * 2f;
            return new Quaternion(0.25f * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
        }
        if (m11 > m22)
        {
            var s = MathF.Sqrt(1f + m11 - m00 - m22) * 2f;
            return new Quaternion((m01 + m10) / s, 0.25f * s, (m12 + m21) / s, (m02 - m20) / s);
        }
        else
        {
            var s = MathF.Sqrt(1f + m22 - m00 - m11) * 2f;
            return new Quaternion((m02 + m20) / s, (m12 + m21) / s, 0.25f * s, (m10 - m01) / s);
        }
    }

    public bool Equals(Quaternion other) => this == other;
    public override bool Equals(object? obj) => obj is Quaternion other && this == other;
    public override int GetHashCode() => HashCode.Combine(x, y, z, w);

    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"({x:F4}, {y:F4}, {z:F4}, {w:F4})");
}
