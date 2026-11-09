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

// --- Rect ----------------------------------------------------------------------

var rect = new Rect(10f, 20f, 100f, 50f);
Check(rect.xMax == 110f && rect.yMax == 70f, "Rect far edges");
Check(rect.center == new Vector2(60f, 45f), "Rect center");
Check(rect.Contains(new Vector2(10f, 20f)), "Rect contains its near corner");
Check(!rect.Contains(new Vector2(110f, 45f)), "Rect excludes its far x edge");
Check(!rect.Contains(new Vector2(60f, 70f)), "Rect excludes its far y edge");
Check(!rect.Contains(new Vector2(9.99f, 45f)), "Rect excludes a point before it");
Check(new Rect(new Vector2(10f, 20f), new Vector2(100f, 50f)) == rect, "Rect from position and size");

// The engine reads the anchor as a row-major index from the upper left
// (`scene::ui::TextAnchor`): column = value % 3, row = value / 3.
Check((uint)Orrin.TextAnchor.MiddleCenter == 4, "TextAnchor.MiddleCenter is 4");
Check((uint)Orrin.TextAnchor.UpperRight == 2, "TextAnchor.UpperRight is 2");
Check((uint)Orrin.TextAnchor.LowerRight == 8, "TextAnchor.LowerRight is 8");

// --- interop layout ------------------------------------------------------------

unsafe
{
    Check(sizeof(Vector2) == 8, "Vector2 is 8 bytes");
    Check(sizeof(Vector3) == 12, "Vector3 is 12 bytes");
    Check(sizeof(Vector4) == 16, "Vector4 is 16 bytes");
    Check(sizeof(Quaternion) == 16, "Quaternion is 16 bytes");
    Check(sizeof(Color) == 16, "Color is 16 bytes");
    Check(sizeof(Rect) == 16, "Rect is 16 bytes");
    Check(sizeof(Orrin.Transform) == 40, "Transform is 10 floats");
    Check(sizeof(Orrin.CameraComponent) == 16, "CameraComponent is 3 floats and an int");
    Check(new Orrin.CameraComponent().FieldOfView == 60f, "a new lens is the engine's default");
    // The retained UI's components and pointer payload, lock-step with
    // orrin-script's CCanvas, CRectTransform, CImage, CText, CButton and
    // CPointerEvent. The engine also checks each size on every call.
    Check(sizeof(Orrin.Canvas) == 20, "Canvas is 20 bytes");
    Check(sizeof(Orrin.RectTransform) == 48, "RectTransform is six Vector2s");
    Check(sizeof(Orrin.Image) == 52, "Image is an id, two Vector4s and a flag");
    Check(sizeof(Orrin.Text) == 48, "Text is an id, a colour and four words");
    Check(sizeof(Orrin.Button) == 68, "Button is four colours and a flag");
    Check(sizeof(Orrin.PointerEvent) == 28, "PointerEvent is two Vector2s, an Entity and a button");
    // The modifiers, lock-step with CGradient, CCorner, CStroke, CShadow and
    // CScrollFrame.
    Check(sizeof(Orrin.Gradient) == 44, "Gradient is two colours, an angle and a Vector2");
    Check(sizeof(Orrin.Corner) == 16, "Corner is four words");
    Check(sizeof(Orrin.Stroke) == 24, "Stroke is a colour and two words");
    Check(sizeof(Orrin.Shadow) == 32, "Shadow is a colour, a Vector2 and two floats");
    Check(sizeof(Orrin.ScrollFrame) == 52, "ScrollFrame is two Vector2s, two floats, a colour and three words");
    Check(sizeof(Orrin.TweenRequest) == 64, "TweenRequest is eight words and two Vector4s");
    Check((uint)Orrin.CornerStyle.Squircle == 1, "CornerStyle.Squircle is 1");
    Check((uint)Orrin.StrokePlacement.Outside == 2, "StrokePlacement.Outside is 2");
    Check((uint)Orrin.ScrollbarVisibility.Never == 2, "ScrollbarVisibility.Never is 2");
    Check((uint)Orrin.TweenProperty.ScrollPosition == 15 && (uint)Orrin.TweenProperty.ImageColor == 3,
        "TweenProperty numbers as the engine does");
    Check((uint)Orrin.EaseStyle.Bounce == 10 && (uint)Orrin.EaseDirection.InOut == 2,
        "EaseStyle and EaseDirection number as the engine does");
    Check(Orrin.Tween.Lanes(Orrin.TweenProperty.ShadowOffset) == 2
        && Orrin.Tween.Lanes(Orrin.TweenProperty.GradientEnd) == 4
        && Orrin.Tween.Lanes(Orrin.TweenProperty.TextSize) == 1,
        "a tween property knows how many floats it is");
    // Physics, lock-step with CRigidBody, CCollider and CRaycastHit, and with
    // the numbering `ForceMode::from_u32` and `add_collider` read.
    Check(sizeof(Orrin.Rigidbody) == 28, "Rigidbody is seven words");
    Check(sizeof(Orrin.ColliderData) == 32, "ColliderData is a shape, a Vector3 and four words");
    Check(sizeof(Orrin.RaycastHit) == 36, "RaycastHit is an Entity, two Vector3s and a float");
    Check((uint)Orrin.ComponentKind.Rigidbody == 14, "ComponentKind.Rigidbody is 14");
    Check((uint)Orrin.ForceMode.Impulse == 1 && (uint)Orrin.ForceMode.Acceleration == 3,
        "ForceMode numbers as the engine does");
    Check((uint)Orrin.ColliderShape.Capsule == 2 && (uint)Orrin.ColliderShape.ConvexHull == 5,
        "ColliderShape numbers as the engine does");
    // Audio, lock-step with CAudioSource and CSound, and with the numbering
    // `rolloff_of` and `find_asset` read.
    Check(sizeof(Orrin.AudioSource) == 44, "AudioSource is a clip's 16 bytes and seven words");
    Check(sizeof(Orrin.SoundRequest) == 60, "SoundRequest is a clip, an Entity, a Vector3 and six words");
    Check((uint)Orrin.ComponentKind.AudioSource == 15, "ComponentKind.AudioSource is 15");
    Check((uint)Orrin.AudioRolloff.Linear == 1, "AudioRolloff numbers as the engine does");
    Check(Orrin.AssetKinds.Of<Orrin.AudioClip>() == 7, "an AudioClip is asset kind 7");
    var speaker = new Orrin.AudioSource();
    Check(speaker.Volume == 1f && speaker.PlayOnStart && speaker.Spatial && !speaker.Loop
        && speaker.MinDistance == 1f && speaker.MaxDistance == 50f && !speaker.Clip.IsValid,
        "a new audio source is the engine's default, its flags read back");
    var body = new Orrin.Rigidbody { LockRotation = true };
    Check(body.Mass == 1f && body.GravityScale == 1f && body.LinearDamping == 0.05f
        && !body.Kinematic && !body.Continuous && body.LockRotation,
        "a new rigid body is the engine's default, its flags read back");
    var capsule = new Orrin.CapsuleCollider { Radius = 0.3f, HalfHeight = 0.9f, IsTrigger = true, Layer = 3 }
        .ToData();
    Check(capsule.Shape == Orrin.ColliderShape.Capsule && capsule.Size.x == 0.3f && capsule.Size.y == 0.9f
        && capsule.IsTrigger == 1 && capsule.Layer == 3 && capsule.Friction == 0.5f,
        "a capsule collider crosses as its radius and half height, with the default surface");

    var list = new Orrin.ScrollFrame { Horizontal = true };
    Check(list.Vertical && list.Horizontal && list.ScrollSpeed == 40f,
        "a new scroll frame is the engine's default, its flags read back");

    // An asset id crosses inside a struct as its canonical bytes, the order
    // `AssetId::from_bytes` reads, and the null reference as zeroes.
    var canonical = Guid.Parse("00112233-4455-6677-8899-aabbccddeeff");
    var textured = new Orrin.Image { Texture = new AssetRef<Texture>(canonical) };
    var raw = (byte*)&textured;
    Check(raw[0] == 0x00 && raw[1] == 0x11 && raw[4] == 0x44 && raw[15] == 0xff,
        "an image's texture crosses in canonical UUID order");
    Check(textured.Texture.Id == canonical, "an image's texture reads back");
    var untextured = new Orrin.Image();
    var zero = (byte*)&untextured;
    var allZero = true;
    for (var i = 0; i < 16; i++)
        allZero &= zero[i] == 0;
    Check(allZero && !untextured.Texture.IsValid, "no texture is sixteen zero bytes");
    var labelled = new Orrin.Text { Font = new AssetRef<Font>(canonical), Wrap = false };
    Check(labelled.Font.Id == canonical && !labelled.Wrap, "a text's font and flags read back");
}

// The engine's defaults, so a component made in C# starts where one made in
// the editor does.
Check(new Orrin.RectTransform().SizeDelta == new Vector2(100f, 100f)
    && new Orrin.RectTransform().AnchorMin == new Vector2(0.5f, 0.5f)
    && new Orrin.RectTransform().Scale == Vector2.one, "a new RectTransform is centred and 100 square");
Check(new Orrin.Image().RaycastTarget && new Orrin.Image().Color == Color.white, "a new Image is white and stops the pointer");
Check(new Orrin.Text().Wrap && !new Orrin.Text().RaycastTarget && new Orrin.Text().Size == 24f, "a new Text wraps and lets the pointer through");
Check(new Orrin.Button().Interactable && new Orrin.Button().Normal == Color.white, "a new Button is interactable");
Check(new Orrin.Canvas().ReferenceResolution == new Vector2(1920f, 1080f)
    && new Orrin.Canvas().ScaleMode == CanvasScaleMode.ConstantPixel, "a new Canvas is one unit to a pixel");

// --- component kinds and pointer events ------------------------------------------
//
// Numbered with the engine's `has_component` match and `ui::events::PointerKind`.

Check((uint)ComponentKind.Canvas == 4 && (uint)ComponentKind.RectTransform == 5
    && (uint)ComponentKind.Image == 6 && (uint)ComponentKind.Text == 7
    && (uint)ComponentKind.Button == 8, "the UI component kinds follow Camera");
Check(ComponentKinds.Blittable<Orrin.Image>() == 6 && ComponentKinds.Blittable<Orrin.Button>() == 8,
    "a blittable component's kind is its ComponentKind");
var refused = false;
try { ComponentKinds.Blittable<Orrin.CameraComponent>(); } catch (ArgumentException) { refused = true; }
Check(refused, "a component with no blittable form is refused loudly");
Check((uint)PointerEventKind.Enter == 0 && (uint)PointerEventKind.Exit == 1
    && (uint)PointerEventKind.Down == 2 && (uint)PointerEventKind.Up == 3
    && (uint)PointerEventKind.Click == 4, "pointer event kinds match the engine's");

var clicker = new ClickProbe();
var press = new PointerEvent(new Vector2(3f, 4f), new Vector2(1f, 2f), new Entity(7, 1), MouseButton.Left);
Check(PointerEvents.Dispatch(clicker, PointerEventKind.Click, press) && clicker.Clicks == 1
    && clicker.Last.Target == new Entity(7, 1) && clicker.Last.Position == new Vector2(3f, 4f),
    "a click handler takes a click, payload and all");
Check(!PointerEvents.Dispatch(clicker, PointerEventKind.Down, press),
    "a behaviour with no press handler leaves the press to its parent");
var hover = new HoverProbe();
Check(PointerEvents.Dispatch(hover, PointerEventKind.Enter, press)
    && PointerEvents.Dispatch(hover, PointerEventKind.Exit, press) && hover.Inside == false
    && hover.Entered == 1, "enter and exit reach their handlers");
Check(!PointerEvents.Dispatch(hover, (PointerEventKind)99, press), "a kind this build lacks is not handled");

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

// An asset reference crosses as tag 11 and its id's sixteen bytes in canonical
// UUID order — the bytes `wire::tests::an_asset_reference_crosses_as_its_canonical_bytes`
// pins on the Rust side. Kept out of `BagGolden`, so that string keeps meaning
// what it meant.
const string AssetGolden =
    "0801000000"
    + "04000000446f6f72" + "0b00112233445566778899aabbccddeeff";

var door = new AssetProbe { Door = new AssetRef<Prefab>(Guid.Parse("00112233-4455-6677-8899-aabbccddeeff")) };
Check(Hex(PropertyBag.Encode(door)) == AssetGolden, "an AssetRef field encodes to the golden bytes");
Check(door.Door.ToString() == "00112233-4455-6677-8899-aabbccddeeff", "an AssetRef prints its id");

var noDoor = new AssetProbe();
Check(!noDoor.Door.IsValid, "a default AssetRef is the null reference");
PropertyBag.Decode(noDoor, Convert.FromHexString(AssetGolden));
Check(noDoor.Door == door.Door, "bag round trip: AssetRef");

// The bytes name no kind, so an `AssetRef<Material>` crosses exactly as the
// prefab reference above does.
const string SkinGolden =
    "0801000000"
    + "04000000536b696e" + "0b00112233445566778899aabbccddeeff";

var skin = new SkinProbe { Skin = new AssetRef<Material>(door.Door.Id) };
Check(Hex(PropertyBag.Encode(skin)) == SkinGolden, "an AssetRef<Material> field encodes to the golden bytes");
var noSkin = new SkinProbe();
PropertyBag.Decode(noSkin, Convert.FromHexString(SkinGolden));
Check(noSkin.Skin == skin.Skin, "bag round trip: AssetRef<Material>");

// The kinds `find_asset` answers with, pinned to `ASSET_*` in orrin-script.
Check(AssetKinds.Of<Prefab>() == 1, "a prefab is asset kind 1");
Check(AssetKinds.Of<Mesh>() == 2, "a mesh is asset kind 2");
Check(AssetKinds.Of<Texture>() == 3, "a texture is asset kind 3");
Check(AssetKinds.Of<Material>() == 4, "a material is asset kind 4");
Check(
    (uint)AssetState.Loading == 0 && (uint)AssetState.Ready == 1 && (uint)AssetState.Failed == 2,
    "AssetState matches LOAD_* in orrin-script");
Check(default(AssetRef<Mesh>).State == AssetState.Failed, "the null reference never loads");

var stepped = new BagProbe { Speed = 99f };
PropertyBag.Decode(stepped, Convert.FromHexString(AssetGolden));
Check(stepped.Speed == 99f, "an asset field this type lacks is stepped over");

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

sealed class ClickProbe : Behaviour, IPointerClickHandler
{
    [Transient] public int Clicks;
    [Transient] public PointerEvent Last;

    public void OnPointerClick(PointerEvent pointer)
    {
        Clicks++;
        Last = pointer;
    }
}

sealed class HoverProbe : Behaviour, IPointerEnterHandler, IPointerExitHandler
{
    [Transient] public bool Inside;
    [Transient] public int Entered;

    public void OnPointerEnter(PointerEvent pointer)
    {
        Inside = true;
        Entered++;
    }

    public void OnPointerExit(PointerEvent pointer) => Inside = false;
}

sealed class AssetProbe : Behaviour
{
    public AssetRef<Prefab> Door;
}

sealed class SkinProbe : Behaviour
{
    public AssetRef<Material> Skin;
}
