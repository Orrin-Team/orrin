// Checks Orrin.Math against System.Numerics as an oracle (same Hamilton
// quaternion conventions; CreateFromYawPitchRoll matches Orrin's Y-X-Z Euler
// order). Run with `dotnet run --project scripting/Orrin.MathTests`; exits
// non-zero on any failure. Dependency-free on purpose — convert to xunit when
// the scripting side grows a real test suite.

using Orrin.Math;

using SN = System.Numerics;

int failures = 0, passed = 0;

void Check(bool condition, string name)
{
    if (condition)
    {
        passed++;
    }
    else
    {
        failures++;
        Console.WriteLine($"FAIL {name}");
    }
}

void CheckNear(float actual, float expected, string name, float eps = 1e-4f) =>
    Check(MathF.Abs(actual - expected) <= eps, $"{name} (got {actual}, want {expected})");

void CheckVector(Vector3 actual, SN.Vector3 expected, string name, float eps = 1e-4f) =>
    Check(
        MathF.Abs(actual.x - expected.X) <= eps
            && MathF.Abs(actual.y - expected.Y) <= eps
            && MathF.Abs(actual.z - expected.Z) <= eps,
        $"{name} (got {actual}, want ({expected.X}, {expected.Y}, {expected.Z}))");

// q and -q are the same rotation, so compare via |dot| ~ 1.
void CheckRotation(Quaternion actual, SN.Quaternion expected, string name, float eps = 1e-4f)
{
    var dot = actual.x * expected.X + actual.y * expected.Y
        + actual.z * expected.Z + actual.w * expected.W;
    Check(MathF.Abs(dot) >= 1f - eps, $"{name} (got {actual}, want ({expected.X:F4}, {expected.Y:F4}, {expected.Z:F4}, {expected.W:F4}))");
}

SN.Vector3 Sn(Vector3 v) => new(v.x, v.y, v.z);
SN.Quaternion SnQ(Quaternion q) => new(q.x, q.y, q.z, q.w);

// --- Mathf -------------------------------------------------------------------

CheckNear(Mathf.Clamp(5f, 0f, 3f), 3f, "Mathf.Clamp above");
CheckNear(Mathf.Clamp(-1f, 0f, 3f), 0f, "Mathf.Clamp below");
CheckNear(Mathf.Lerp(0f, 10f, 0.25f), 2.5f, "Mathf.Lerp");
CheckNear(Mathf.Lerp(0f, 10f, 2f), 10f, "Mathf.Lerp clamps");
CheckNear(Mathf.LerpUnclamped(0f, 10f, 2f), 20f, "Mathf.LerpUnclamped");
CheckNear(Mathf.InverseLerp(10f, 20f, 15f), 0.5f, "Mathf.InverseLerp");
CheckNear(Mathf.Repeat(2.3f, 2f), 0.3f, "Mathf.Repeat");
CheckNear(Mathf.Repeat(-0.5f, 2f), 1.5f, "Mathf.Repeat negative");
CheckNear(Mathf.PingPong(0.5f, 1f), 0.5f, "Mathf.PingPong ascending");
CheckNear(Mathf.PingPong(1.2f, 1f), 0.8f, "Mathf.PingPong descending");
CheckNear(Mathf.PingPong(2.3f, 1f), 0.3f, "Mathf.PingPong wraps");
CheckNear(Mathf.DeltaAngle(350f, 10f), 20f, "Mathf.DeltaAngle wraps");
CheckNear(Mathf.DeltaAngle(10f, 350f), -20f, "Mathf.DeltaAngle negative");
CheckNear(Mathf.MoveTowards(0f, 10f, 3f), 3f, "Mathf.MoveTowards steps");
CheckNear(Mathf.MoveTowards(0f, 2f, 3f), 2f, "Mathf.MoveTowards arrives");
CheckNear(90f * Mathf.Deg2Rad, MathF.PI / 2f, "Mathf.Deg2Rad");
Check(Mathf.Approximately(1f, 1f + 1e-7f), "Mathf.Approximately near");
Check(!Mathf.Approximately(1f, 1.1f), "Mathf.Approximately far");

// --- Vector3 -----------------------------------------------------------------

var a = new Vector3(1.5f, -2f, 0.75f);
var b = new Vector3(-3f, 0.5f, 2f);

CheckNear(Vector3.Dot(a, b), SN.Vector3.Dot(Sn(a), Sn(b)), "Vector3.Dot");
CheckVector(Vector3.Cross(a, b), SN.Vector3.Cross(Sn(a), Sn(b)), "Vector3.Cross");
CheckVector(a.normalized, SN.Vector3.Normalize(Sn(a)), "Vector3.normalized");
CheckNear(a.magnitude, Sn(a).Length(), "Vector3.magnitude");
CheckVector(Vector3.Lerp(a, b, 0.3f), SN.Vector3.Lerp(Sn(a), Sn(b), 0.3f), "Vector3.Lerp");
CheckVector(
    Vector3.Reflect(a, Vector3.up), SN.Vector3.Reflect(Sn(a), SN.Vector3.UnitY), "Vector3.Reflect");
CheckNear(Vector3.Distance(a, b), SN.Vector3.Distance(Sn(a), Sn(b)), "Vector3.Distance");
Check(Vector3.zero.normalized == Vector3.zero, "Vector3 zero normalizes to zero");
CheckNear(Vector3.Angle(Vector3.right, Vector3.up), 90f, "Vector3.Angle");
CheckNear(
    Vector3.SignedAngle(Vector3.right, Vector3.forward, Vector3.up),
    90f,
    "Vector3.SignedAngle (right-handed, -Z forward)");

// --- Quaternion vs System.Numerics --------------------------------------------

var axis = new Vector3(0.3f, 1f, 0.25f).normalized;
CheckRotation(
    Quaternion.AngleAxis(47f, axis),
    SN.Quaternion.CreateFromAxisAngle(Sn(axis), 47f * Mathf.Deg2Rad),
    "Quaternion.AngleAxis");

foreach (var (ex, ey, ez) in new[] { (30f, 45f, 60f), (0f, 90f, 0f), (-15f, 200f, 5f), (90f, 0f, 0f) })
{
    CheckRotation(
        Quaternion.Euler(ex, ey, ez),
        SN.Quaternion.CreateFromYawPitchRoll(
            ey * Mathf.Deg2Rad, ex * Mathf.Deg2Rad, ez * Mathf.Deg2Rad),
        $"Quaternion.Euler({ex}, {ey}, {ez})");
}

var q1 = Quaternion.Euler(30f, 45f, 60f);
var q2 = Quaternion.AngleAxis(80f, axis);
CheckRotation(q1 * q2, SnQ(q1) * SnQ(q2), "Quaternion composition");

var rotated = q1 * a;
CheckVector(rotated, SN.Vector3.Transform(Sn(a), SnQ(q1)), "Quaternion * Vector3");

foreach (var t in new[] { 0f, 0.25f, 0.5f, 0.9f, 1f })
{
    CheckRotation(
        Quaternion.Slerp(q1, q2, t), SN.Quaternion.Slerp(SnQ(q1), SnQ(q2), t),
        $"Quaternion.Slerp t={t}");
}

CheckRotation(Quaternion.Inverse(q1), SN.Quaternion.Inverse(SnQ(q1)), "Quaternion.Inverse");

// eulerAngles: the extracted angles must rebuild the same rotation.
foreach (var (ex, ey, ez) in new[] { (30f, 45f, 60f), (10f, 350f, 0f), (-80f, 20f, 45f) })
{
    var q = Quaternion.Euler(ex, ey, ez);
    var rebuilt = Quaternion.Euler(q.eulerAngles);
    CheckRotation(rebuilt, SnQ(q), $"eulerAngles roundtrip ({ex}, {ey}, {ez})");
}

// LookRotation: rotates canonical forward (-Z) onto the target direction,
// keeping up roughly +Y.
foreach (var dir in new[]
{
    new Vector3(1f, 0f, 0f),
    new Vector3(-2f, 0.5f, 3f),
    new Vector3(0f, 0f, 1f),
    new Vector3(0.1f, -1f, 0.2f),
})
{
    var look = Quaternion.LookRotation(dir);
    CheckVector(look * Vector3.forward, Sn(dir.normalized), $"LookRotation({dir}) aims forward");
    Check((look * Vector3.right).y is > -1e-4f and < 1e-4f, $"LookRotation({dir}) keeps right level");
}
Check(
    Quaternion.Angle(Quaternion.LookRotation(Vector3.forward), Quaternion.identity) < 1e-3f,
    "LookRotation(forward) is identity");
// Degenerate case: looking straight up must still produce a valid rotation.
var upLook = Quaternion.LookRotation(Vector3.up);
CheckVector(upLook * Vector3.forward, Sn(Vector3.up), "LookRotation(up) aims forward");

// FromToRotation, including the antiparallel edge case.
foreach (var (from, to) in new[]
{
    (new Vector3(1f, 0f, 0f), new Vector3(0f, 1f, 0f)),
    (new Vector3(1f, 2f, 3f), new Vector3(-2f, 0.3f, 1f)),
    (new Vector3(0f, 1f, 0f), new Vector3(0f, -1f, 0f)),
})
{
    var q = Quaternion.FromToRotation(from, to);
    CheckVector(q * from.normalized, Sn(to.normalized), $"FromToRotation {from} -> {to}");
}

CheckNear(Quaternion.Angle(q1, q1), 0f, "Quaternion.Angle self");
CheckNear(
    Quaternion.Angle(Quaternion.identity, Quaternion.AngleAxis(90f, Vector3.up)),
    90f,
    "Quaternion.Angle 90");

// --- Color ---------------------------------------------------------------------

var orange = Color.FromHex("#FF8800");
CheckNear(orange.r, 1f, "FromHex r");
CheckNear(orange.g, 136f / 255f, "FromHex g");
CheckNear(orange.b, 0f, "FromHex b");
CheckNear(orange.a, 1f, "FromHex default alpha");
Check(Color.FromHex("F80") == orange, "FromHex shorthand");
CheckNear(Color.FromHex("80FF00CC").a, 204f / 255f, "FromHex alpha");
Check(orange.ToHex() == "FF8800", "ToHex roundtrip");
Check(Color.FromHex(Color.red.ToHex(includeAlpha: true)) == Color.red, "hex full roundtrip");
Check(Color.Lerp(Color.black, Color.white, 0.5f) == new Color(0.5f, 0.5f, 0.5f), "Color.Lerp");
var threw = false;
try { Color.FromHex("nope"); } catch (FormatException) { threw = true; }
Check(threw, "FromHex rejects garbage");

// --- interop layout ------------------------------------------------------------

unsafe
{
    Check(sizeof(Vector2) == 8, "Vector2 is 8 bytes");
    Check(sizeof(Vector3) == 12, "Vector3 is 12 bytes");
    Check(sizeof(Vector4) == 16, "Vector4 is 16 bytes");
    Check(sizeof(Quaternion) == 16, "Quaternion is 16 bytes");
    Check(sizeof(Color) == 16, "Color is 16 bytes");
    Check(sizeof(Orrin.Transform) == 40, "Transform is 10 floats");
}

Console.WriteLine($"{passed} passed, {failures} failed");
return failures == 0 ? 0 : 1;
