// Checks Orrin.Math against System.Numerics as an oracle (same Hamilton
// quaternion conventions; CreateFromYawPitchRoll matches Orrin's Y-X-Z Euler
// order). Run with `dotnet run --project scripting/Orrin.MathTests`; exits
// non-zero on any failure. Dependency-free on purpose — convert to xunit when
// the scripting side grows a real test suite.

using Orrin;
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

// --- property bag ABI ----------------------------------------------------------
//
// The other frozen layout, and the one with no `sizeof` to lean on: this file
// and `orrin-registry`'s `wire` module are two implementations of one grammar,
// so the check is that they produce and accept the same bytes. Both constants
// below are asserted on the Rust side too (`wire::tests::the_golden_vector_*`);
// changing the format means changing four places, which is the point.

string Hex(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(bytes);

// What a Behaviour with one field of every representable type encodes to.
const string BagGolden =
    "0808000000"
    + "020000004f6e0001"
    + "05000000436f756e7401feffffff"
    + "05000000496e6465780207000000"
    + "050000005370656564030000c03f"
    + "050000004c6162656c0402000000" + "6869"
    + "08000000506f736974696f6e050000803f0000004000004040"
    + "08000000526f746174696f6e060000000000000000000000000000803f"
    + "040000004d6f6465090400000053706f7400000000";

// Every variant, including the two a Behaviour cannot express. Written by Rust;
// the decoder must step over what it cannot represent and still land the rest.
const string FullGolden =
    "080a000000"
    + "020000006f6e0001"
    + "05000000636f756e7401feffffff"
    + "05000000696e6465780207000000"
    + "050000007370656564030000c03f"
    + "050000006c6162656c04020000006869"
    + "08000000706f736974696f6e050000803f0000004000004040"
    + "08000000726f746174696f6e060000000000000000000000000000803f"
    + "060000007461726765740700000000000000000000000000000000"
    + "06000000706f696e74730a01000000030000003f"
    + "040000006d6f6465090400000053706f7401000000020000006f6e0000";

var bag = new BagProbe();
Check(Hex(PropertyBag.Encode(bag)) == BagGolden, "property bag encodes to the golden bytes");

// Round trip through the format rather than through the object: a field that
// encoded but did not decode would otherwise pass by having never moved.
var restored = new BagProbe
{
    On = false, Count = 0, Index = 0, Speed = 0f, Label = "",
    Position = Vector3.zero, Rotation = new Quaternion(1, 0, 0, 0), Mode = BagMode.Point,
};
PropertyBag.Decode(restored, Convert.FromHexString(BagGolden));
Check(restored.On && restored.Count == -2 && restored.Index == 7u, "bag round trip: integers");
Check(restored.Speed == 1.5f && restored.Label == "hi", "bag round trip: float and string");
Check(restored.Position == new Vector3(1, 2, 3), "bag round trip: Vector3");
Check(restored.Rotation == Quaternion.identity, "bag round trip: Quaternion");
Check(restored.Mode == BagMode.Spot, "bag round trip: enum by member name");

// `Transient` and `readonly` are the two ways out of the registry, and a field
// of an unrepresentable type is simply not in it.
Check(
    PropertyBag.VisibleFields(typeof(BagProbe)).Select(f => f.Name).SequenceEqual(
        ["On", "Count", "Index", "Speed", "Label", "Position", "Rotation", "Mode"]),
    "only representable, non-transient, writable fields are visible");

// The lenient half. A buffer whose fields this build does not have must be
// stepped over rather than throwing or desynchronizing — including the `Entity`
// and `List` values no Behaviour can express, which is the case `FullGolden`
// exists for. Field names are matched ordinally, so none of its lower-case names
// land on BagProbe's.
var untouched = new BagProbe { Speed = 99f };
PropertyBag.Decode(untouched, Convert.FromHexString(FullGolden));
Check(untouched.Speed == 99f, "a bag of entirely unknown fields changes nothing");

// Mixed: one field this build has, one it does not, in that order — so a failure
// to step over the unknown one would land the wrong bytes on `Speed`.
var mixed = new BagProbe { Speed = 99f };
PropertyBag.Decode(mixed, Convert.FromHexString(
    "0802000000"
    + "05000000" + Hex("Ghost"u8) + "030000803f"
    + "05000000" + Hex("Speed"u8) + "0300002040"));
Check(mixed.Speed == 2.5f, "an unknown field is stepped over, not misread");

var renamed = new BagProbe { Mode = BagMode.Point };
PropertyBag.Decode(renamed, Convert.FromHexString(
    "080100000004000000" + Hex("Mode"u8) + "09040000004172656100000000"));
Check(renamed.Mode == BagMode.Point, "an enum member this build lost leaves the field alone");

// --- log attribution -----------------------------------------------------------
//
// A log line names its script from the path the compiler recorded, which is in
// the build machine's spelling rather than this one's.

Check(Native.ScriptName("/home/dev/game/scripts/Spinner.cs") == "Spinner", "a Unix path names its script");
Check(Native.ScriptName(@"C:\game\scripts\Spinner.cs") == "Spinner", "a Windows path names its script");
Check(Native.ScriptName("Spinner.cs") == "Spinner", "a bare file names its script");
Check(Native.ScriptName("/scripts.v2/Spinner") == "Spinner", "a dot in a directory is not an extension");
Check(Native.ScriptName("") == "", "no path is no name");

Console.WriteLine($"{passed} passed, {failures} failed");
return failures == 0 ? 0 : 1;

enum BagMode { Point, Spot }

/// Field order is the order the bag is written in, so this declaration is part
/// of the golden above.
sealed class BagProbe : Behaviour
{
    public bool On = true;
    public int Count = -2;
    public uint Index = 7;
    public float Speed = 1.5f;
    public string Label = "hi";
    public Vector3 Position = new(1, 2, 3);
    public Quaternion Rotation = Quaternion.identity;
    public BagMode Mode = BagMode.Spot;

    [Transient]
    public float Cache = 3f;
    public readonly int Constant = 5;
    public Vector2 Unrepresentable = new(1, 2);
}
