// Checks Orrin.Math against System.Numerics as an oracle (same Hamilton
// quaternion conventions; CreateFromYawPitchRoll matches Orrin's Y-X-Z Euler
// order). Run with `dotnet run --project scripting/Orrin.MathTests`; exits
// non-zero on any failure. Dependency-free on purpose — convert to xunit when
// the scripting side grows a real test suite.

using System.Reflection;
using System.Xml.Linq;

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
    Check(sizeof(Orrin.Collision) == 44, "Collision is an Entity and three Vector3s");
    // The shape queries, lock-step with CShapeQuery.
    Check(sizeof(Orrin.ShapeQuery) == 68, "ShapeQuery is a shape, three Vector3s, a Quaternion and three words");
    // Key codes appended after RightAlt, as keys.rs numbers them: each group's
    // first and last, so an implicit increment that skipped would show.
    Check((uint)Orrin.KeyCode.F1 == 55 && (uint)Orrin.KeyCode.F12 == 66, "F1..F12 are 55..66");
    Check((uint)Orrin.KeyCode.Minus == 67 && (uint)Orrin.KeyCode.BackQuote == 77, "Minus..BackQuote are 67..77");
    Check((uint)Orrin.KeyCode.Keypad0 == 78 && (uint)Orrin.KeyCode.Keypad9 == 87, "Keypad0..9 are 78..87");
    Check((uint)Orrin.KeyCode.KeypadPlus == 88 && (uint)Orrin.KeyCode.KeypadEnter == 93, "KeypadPlus..Enter are 88..93");
    Check((uint)Orrin.KeyCode.Insert == 94 && (uint)Orrin.KeyCode.CapsLock == 100, "Insert..CapsLock are 94..100");
    // The optional hooks `Behaviours.Create` reports, lock-step with
    // `orrin_script::Hooks`: what the engine reads to skip a per-step call.
    Check(Orrin.Behaviours.FixedUpdateHook == 1
        && Orrin.Behaviours.CollisionStayHook == 2
        && Orrin.Behaviours.TriggerStayHook == 4,
        "the hook bits number as the engine does");
    Check(Orrin.Behaviours.HooksOf(typeof(Orrin.Tests.IdleProbe)) == 0,
        "a behaviour overriding nothing reports no hooks");
    Check(Orrin.Behaviours.HooksOf(typeof(Orrin.Tests.SecondSteppedProbe)) == Orrin.Behaviours.FixedUpdateHook,
        "an OnFixedUpdate inherited from a base class counts");
    Check(Orrin.Behaviours.HooksOf(typeof(Orrin.Tests.ContactProbe))
        == (Orrin.Behaviours.CollisionStayHook | Orrin.Behaviours.TriggerStayHook),
        "the Stay callbacks are reported apart from the other contact callbacks");
    Check((uint)Orrin.ComponentKind.Rigidbody == 14, "ComponentKind.Rigidbody is 14");
    Check((uint)Orrin.ForceMode.Impulse == 1 && (uint)Orrin.ForceMode.Acceleration == 3,
        "ForceMode numbers as the engine does");
    Check((uint)Orrin.ColliderShape.Capsule == 2 && (uint)Orrin.ColliderShape.ConvexHull == 5,
        "ColliderShape numbers as the engine does");
    // A mesh renderer, lock-step with CMeshRenderer: two assets' 16 bytes.
    Check(sizeof(Orrin.MeshRenderer) == 32, "MeshRenderer is two asset ids");
    Check((uint)Orrin.ComponentKind.MeshRenderer == 16, "ComponentKind.MeshRenderer is 16");
    Check(Orrin.ComponentKinds.Blittable<Orrin.MeshRenderer>() == 16,
        "a MeshRenderer is read and written as a struct");
    var renderer = new Orrin.MeshRenderer();
    var gold = new Orrin.AssetRef<Orrin.Material>(System.Guid.NewGuid());
    renderer.Material = gold;
    Check(renderer.Material.Id == gold.Id && !renderer.Mesh.IsValid,
        "MeshRenderer.Material round-trips its reference and leaves the mesh alone");
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

// Every variant of the original eleven, including a tag 7 entity id, which a
// Behaviour never receives (its references cross as handles). Written by Rust;
// the decoder must step over what it cannot assign and still land the rest.
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
// stepped over rather than throwing or desynchronizing — including a tag 7
// entity id and a list, which is the case `FullGolden` exists for. Field names are matched ordinally, so none of its lower-case names
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

// The tags appended after the asset one — Vector2 (12), Color (13) and a live
// entity handle (14) — plus lists, which a Behaviour can now hold. These are the
// bytes `wire::tests::the_bag_of_the_appended_field_types_has_not_moved` pins.
// An `Entity` crosses as its handle rather than as tag 7's sixteen-byte id: the
// handle is all this side has, and the engine translates it where the world is.
const string ShapesGolden =
    "0805000000"
    + "060000004f66667365740c0000803f00000040"
    + "0400000054696e740d0000803f0000003f000000000000803f"
    + "06000000546172676574" + "0e0500000002000000"
    + "0500000053746570730a02000000" + "01010000000102000000"
    + "0700000057656967687473" + "0a01000000030000003f";

var shapes = new ShapeProbe();
Check(Hex(PropertyBag.Encode(shapes)) == ShapesGolden, "Vector2, Color, Entity and list fields encode to the golden bytes");
var blank = new ShapeProbe { Offset = Vector2.zero, Tint = Color.clear, Target = Entity.Null, Steps = [], Weights = [] };
PropertyBag.Decode(blank, Convert.FromHexString(ShapesGolden));
Check(blank.Offset == new Vector2(1f, 2f) && blank.Tint == new Color(1f, 0.5f, 0f, 1f), "bag round trip: Vector2 and Color");
Check(blank.Target == new Entity(5, 2), "bag round trip: an Entity is its live handle");
Check(blank.Steps.SequenceEqual([1, 2]) && blank.Weights.SequenceEqual([0.5f]), "bag round trip: List<int> and float[]");

// A list whose elements this build reads as another type leaves the field
// alone rather than half-filling it, and the reader still steps over it whole.
var retyped = new ShapeProbe { Weights = [9f] };
PropertyBag.Decode(retyped, Convert.FromHexString(
    "0802000000" + "0700000057656967687473" + "0a01000000" + "0400000000"
    + "060000004f66667365740c0000803f00000040"));
Check(retyped.Weights.SequenceEqual([9f]) && retyped.Offset == new Vector2(1f, 2f),
    "a list of the wrong element type is stepped over and left alone");

Check(
    PropertyBag.VisibleFields(typeof(ShapeProbe)).Select(f => f.Name).SequenceEqual(
        ["Offset", "Tint", "Target", "Steps", "Weights"]),
    "Vector2, Color, Entity, List<T> and T[] are visible");

// A component's listing entry carries its fields' presentation attributes,
// only for the fields that have some. The bytes
// `scripting::tests::a_listing_entry_carries_its_fields_presentation` decodes.
const string ListingGolden =
    "0a010000000804000000020000006964040a000000746573742e74756e65640400000074797065041b000000"
    + "54756e656450726f62652c204f7272696e2e4d6174685465737473040000006e616d65040a00000054756e65"
    + "6450726f6265060000006669656c64730a020000000805000000040000006e616d6504050000005370656564"
    + "0600000068656164657204060000004d6f74696f6e07000000746f6f6c746970040f0000004d657472657320"
    + "61207365636f6e640500000072616e67650a02000000030000000003000020410600000068696464656e0000"
    + "0805000000040000006e616d6504060000005365637265740600000068656164657204000000000700000074"
    + "6f6f6c74697004000000000500000072616e67650a000000000600000068696464656e0001";
Check(Hex(PropertyBag.Listing([typeof(TunedProbe)])) == ListingGolden,
    "a listing entry encodes its fields' attributes to the golden bytes");

// --- log attribution -----------------------------------------------------------
//
// A log line names its script from the path the compiler recorded, which is in
// the build machine's spelling rather than this one's.

Check(Native.ScriptName("/home/dev/game/scripts/Spinner.cs") == "Spinner", "a Unix path names its script");
Check(Native.ScriptName(@"C:\game\scripts\Spinner.cs") == "Spinner", "a Windows path names its script");
Check(Native.ScriptName("Spinner.cs") == "Spinner", "a bare file names its script");
Check(Native.ScriptName("/scripts.v2/Spinner") == "Spinner", "a dot in a directory is not an extension");
Check(Native.ScriptName("") == "", "no path is no name");

// --- API documentation -----------------------------------------------------------
//
// A script author's editor reads the bindings' documentation out of Orrin.xml
// beside the Orrin.dll a project references, and shows only what is inside its
// elements: a plain `///` line compiles into the file and is displayed by
// nothing. So the check reads the file the build wrote, not the source, and asks
// for a <summary> (or an <inheritdoc/>) on every member a script can reach.
// That is stricter than CS1591, which a plain `///` line satisfies, and is why
// Orrin.csproj turns CS1591 off rather than having two lists of exemptions.

// The bindings' own build output, which a project outside the checkout
// references by path: this program's output directory, moved from
// scripting/Orrin.MathTests to scripting/Orrin.
var output = new DirectoryInfo(Path.TrimEndingDirectorySeparator(AppContext.BaseDirectory));
var docPath = Path.Combine(
    output.Parent!.Parent!.Parent!.Parent!.FullName, "Orrin", "bin", output.Parent.Name, output.Name, "Orrin.xml");
XDocument? apiDoc = null;
if (!File.Exists(docPath))
{
    Check(false, $"Orrin.xml is written beside the bindings' Orrin.dll (looked for {docPath})");
}
else
{
    try
    {
        apiDoc = XDocument.Load(docPath);
    }
    catch (System.Xml.XmlException e)
    {
        Check(false, $"Orrin.xml is well-formed ({e.Message})");
    }
}

if (apiDoc is not null)
{
    var members = new Dictionary<string, XElement>();
    foreach (var member in apiDoc.Descendants("member"))
        members.TryAdd((string)member.Attribute("name")!, member);
    bool Documented(string id) =>
        members.TryGetValue(id, out var member)
        && (member.Element("inheritdoc") is not null
            || !string.IsNullOrWhiteSpace(member.Element("summary")?.Value));

    // A fixed sample, spelled out by hand: it is what the coverage sweep below
    // stands on, since an id spelled differently from the compiler's would
    // read as undocumented rather than as a bug in the spelling.
    foreach (var (member, id) in new (MemberInfo, string)[]
    {
        (typeof(Entity), "T:Orrin.Entity"),
        (typeof(Behaviour).GetMethod(nameof(Behaviour.OnUpdate))!, "M:Orrin.Behaviour.OnUpdate(System.Single)"),
        (typeof(Behaviour).GetMethod(nameof(Behaviour.OnFixedUpdate))!, "M:Orrin.Behaviour.OnFixedUpdate(System.Single)"),
        (typeof(Time).GetProperty(nameof(Time.timeScale))!, "P:Orrin.Time.timeScale"),
        (typeof(Physics).GetMethod(
                nameof(Physics.OverlapSphere), [typeof(Vector3), typeof(float), typeof(Span<Entity>), typeof(uint), typeof(bool)])!,
            "M:Orrin.Physics.OverlapSphere(Orrin.Math.Vector3,System.Single,System.Span{Orrin.Entity},System.UInt32,System.Boolean)"),
        (typeof(Physics).GetMethod(nameof(Physics.Raycast))!,
            "M:Orrin.Physics.Raycast(Orrin.Math.Vector3,Orrin.Math.Vector3,Orrin.RaycastHit@,System.Single,System.UInt32,System.Boolean)"),
        (typeof(World).GetMethod(nameof(World.AddScript))!, "M:Orrin.World.AddScript``1(Orrin.Entity)"),
        (typeof(Vector3).GetMethod(nameof(Vector3.Cross))!, "M:Orrin.Math.Vector3.Cross(Orrin.Math.Vector3,Orrin.Math.Vector3)"),
        (typeof(Orrin.Testing.TestAttribute), "T:Orrin.Testing.TestAttribute"),
    })
    {
        Check(DocIds.Of(member) == id, $"DocIds spells {id} as the compiler does (got {DocIds.Of(member)})");
        Check(Documented(id), $"{id} has a summary in Orrin.xml");
    }

    // Public only because the engine calls into them through function
    // pointers, or because the reload machinery lives in this assembly.
    var engineOnly = new HashSet<Type>
    {
        typeof(OrrinApi), typeof(Bootstrap), typeof(Behaviours), typeof(BehaviourState),
        typeof(GameAssembly), typeof(PropertyBag), typeof(Stories), typeof(Tweens),
        typeof(Orrin.Testing.TestRunner),
    };
    // Enums whose members say everything their names say; the type carries
    // what is not obvious, such as a key code being a physical position.
    var selfNaming = new HashSet<Type>
    {
        typeof(KeyCode), typeof(MouseButton), typeof(TextAnchor), typeof(ComponentKind),
    };

    var undocumented = new List<string>();
    foreach (var type in typeof(Entity).Assembly.GetExportedTypes())
    {
        if (engineOnly.Contains(type))
            continue;
        if (!Documented(DocIds.Of(type)))
            undocumented.Add(DocIds.Of(type));
        if (selfNaming.Contains(type))
            continue;
        foreach (var member in type.GetMembers(
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static
            | BindingFlags.DeclaredOnly))
        {
            if (DocIds.ScriptVisible(member) && !Documented(DocIds.Of(member)))
                undocumented.Add(DocIds.Of(member));
        }
    }
    Check(undocumented.Count == 0,
        $"every script-facing member has a summary in Orrin.xml; {undocumented.Count} do not:\n  "
        + string.Join("\n  ", undocumented.Order()));
}

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
    public Vector4 Unrepresentable = new(1, 2, 3, 4);
}

sealed class ShapeProbe : Behaviour
{
    public Vector2 Offset = new(1f, 2f);
    public Color Tint = new(1f, 0.5f, 0f, 1f);
    public Entity Target = new(5, 2);
    public List<int> Steps = [1, 2];
    public float[] Weights = [0.5f];
}

[Component("test.tuned")]
sealed class TunedProbe : Behaviour
{
    [Header("Motion"), Tooltip("Metres a second"), Range(0f, 10f)]
    public float Speed = 4f;
    [HideInInspector] public int Secret = 7;
    public bool Plain = true;
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

/// The compiler's documentation ids (ECMA-334 annex D) for what reflection
/// hands back, so a member can be looked up in Orrin.xml. Covers the shapes the
/// bindings' public surface has; anything else throws rather than guessing.
static class DocIds
{
    public static string Of(MemberInfo member) => member switch
    {
        Type type => "T:" + Name(type),
        FieldInfo field => $"F:{Name(field.DeclaringType!)}.{field.Name}",
        PropertyInfo property => $"P:{Name(property.DeclaringType!)}.{property.Name}"
            + Parameters(property.GetIndexParameters()),
        EventInfo @event => $"E:{Name(@event.DeclaringType!)}.{@event.Name}",
        MethodBase method => Method(method),
        _ => throw new NotSupportedException($"no documentation id for {member}"),
    };

    /// Public or protected, and written by hand: accessors, `value__` and the
    /// parameterless constructor a class gets for free are not. A struct has
    /// one in metadata only when it declares it.
    public static bool ScriptVisible(MemberInfo member) => member switch
    {
        Type => false,
        FieldInfo field => (field.IsPublic || field.IsFamily || field.IsFamilyOrAssembly) && !field.IsSpecialName,
        PropertyInfo property => property.GetAccessors(nonPublic: true).Any(Reachable),
        EventInfo @event => @event.AddMethod is { } add && Reachable(add),
        ConstructorInfo constructor => Reachable(constructor) && !constructor.IsStatic
            && (constructor.GetParameters().Length > 0 || constructor.DeclaringType!.IsValueType),
        MethodInfo method => Reachable(method) && !Generated(method)
            && (!method.IsSpecialName || method.Name.StartsWith("op_")),
        _ => false,
    };

    static bool Reachable(MethodBase method) => method.IsPublic || method.IsFamily || method.IsFamilyOrAssembly;

    // An auto-property's accessors carry the attribute too, which is why it is
    // asked only of a method that stands on its own.
    static bool Generated(MethodBase method) =>
        method.GetCustomAttribute<System.Runtime.CompilerServices.CompilerGeneratedAttribute>() is not null;

    static string Method(MethodBase method)
    {
        var name = method is ConstructorInfo ? (method.IsStatic ? "#cctor" : "#ctor") : method.Name;
        var id = $"M:{Name(method.DeclaringType!)}.{name}";
        if (method.IsGenericMethodDefinition)
            id += "``" + method.GetGenericArguments().Length;
        id += Parameters(method.GetParameters());
        if (method is MethodInfo { Name: "op_Implicit" or "op_Explicit" } conversion)
            id += "~" + Reference(conversion.ReturnType);
        return id;
    }

    static string Parameters(ParameterInfo[] parameters) => parameters.Length == 0
        ? ""
        : "(" + string.Join(",", parameters.Select(parameter => Reference(parameter.ParameterType))) + ")";

    static string Name(Type type) => (type.FullName ?? throw new NotSupportedException($"no full name for {type}"))
        .Replace('+', '.');

    static string Reference(Type type)
    {
        if (type.IsByRef)
            return Reference(type.GetElementType()!) + "@";
        if (type.IsPointer)
            return Reference(type.GetElementType()!) + "*";
        if (type.IsArray)
            return Reference(type.GetElementType()!)
                + (type.GetArrayRank() == 1 ? "[]" : "[" + string.Join(",", Enumerable.Repeat("0:", type.GetArrayRank())) + "]");
        if (type.IsGenericParameter)
            return (type.DeclaringMethod is null ? "`" : "``") + type.GenericParameterPosition;
        if (type.IsGenericType)
        {
            var definition = Name(type.GetGenericTypeDefinition());
            return definition[..definition.LastIndexOf('`')]
                + "{" + string.Join(",", type.GetGenericArguments().Select(Reference)) + "}";
        }
        return Name(type);
    }
}
