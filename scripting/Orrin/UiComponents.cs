using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

// The retained UI's components, as they cross the ABI. Each layout must match
// its Rust `C…` struct in orrin-script, field for field; Orrin.MathTests
// asserts the sizes, and the engine refuses a struct of the wrong size by name
// on every call. Read and write them through World.GetComponent,
// World.SetComponent and World.AddComponent.
//
// `new T()` is the engine's default for each; `default(T)` is all zeroes,
// which C# does not let a struct change, and is not a useful element.

/// <summary>An asset id as the engine stores one: its sixteen bytes in canonical UUID order.</summary>
[InlineArray(16)]
internal struct AssetBytes
{
    byte _first;

    internal static AssetBytes Of(Guid id)
    {
        var bytes = new AssetBytes();
        id.TryWriteBytes(bytes, bigEndian: true, out _);
        return bytes;
    }

    internal readonly Guid ToGuid() => new(this, bigEndian: true);
}

/// <summary>How a canvas's units map to the window's pixels.</summary>
public enum CanvasScaleMode : uint
{
    /// <summary>One unit is one physical pixel, whatever the window's size.</summary>
    ConstantPixel,

    /// <summary>
    /// The canvas is <see cref="Canvas.ReferenceResolution"/> units across,
    /// scaled to the window.
    /// </summary>
    ScaleWithScreen,
}

/// <summary>
/// The root of a UI tree: the whole window, which its children's anchors are
/// fractions of. A canvas under another canvas is an ordinary element.
/// </summary>
// Layout must match Rust CCanvas.
[StructLayout(LayoutKind.Sequential)]
public struct Canvas
{
    /// <summary>How the canvas's units map to the window's pixels.</summary>
    public CanvasScaleMode ScaleMode;

    /// <summary>The window size a <c>ScaleWithScreen</c> layout is authored at.</summary>
    public Vector2 ReferenceResolution;

    /// <summary>
    /// 0 scales with the window's width, 1 with its height; between blends the
    /// two logarithmically.
    /// </summary>
    public float MatchWidthOrHeight;

    /// <summary>Canvases draw lowest first.</summary>
    public int SortOrder;

    /// <summary>One unit to a physical pixel, with a 1920 x 1080 reference resolution matched on width.</summary>
    public Canvas()
    {
        ScaleMode = CanvasScaleMode.ConstantPixel;
        ReferenceResolution = new Vector2(1920f, 1080f);
        MatchWidthOrHeight = 0f;
        SortOrder = 0;
    }
}

/// <summary>
/// A UI element's rectangle relative to its parent's, top-left origin with y
/// down: anchor (0, 0) is the parent's top left and (1, 1) its bottom right.
/// </summary>
/// <remarks>
/// The element is <c>(AnchorMax - AnchorMin) * parent + SizeDelta</c> across,
/// and its pivot stands at <c>lerp(AnchorMin, AnchorMax, Pivot)</c> of the
/// parent plus <see cref="AnchoredPosition"/>. <see cref="Scale"/> scales the
/// element and everything under it about that pivot.
/// </remarks>
// Layout must match Rust CRectTransform.
[StructLayout(LayoutKind.Sequential)]
public struct RectTransform
{
    /// <summary>The anchor box's top-left corner, as a fraction of the parent.</summary>
    public Vector2 AnchorMin;
    /// <summary>The anchor box's bottom-right corner, as a fraction of the parent.</summary>
    public Vector2 AnchorMax;
    /// <summary>
    /// The point the element is placed by and scales about, as a fraction of its own
    /// rectangle from its top left.
    /// </summary>
    public Vector2 Pivot;
    /// <summary>Where the pivot stands from its place in the anchor box, in canvas units, y down.</summary>
    public Vector2 AnchoredPosition;
    /// <summary>
    /// Canvas units added to the anchor box's span of the parent: with the anchors
    /// together, the element's size.
    /// </summary>
    public Vector2 SizeDelta;
    /// <summary>Scales the element and everything under it about its pivot.</summary>
    public Vector2 Scale;

    /// <summary>Centred in its parent, 100 units square.</summary>
    public RectTransform()
    {
        AnchorMin = new Vector2(0.5f, 0.5f);
        AnchorMax = new Vector2(0.5f, 0.5f);
        Pivot = new Vector2(0.5f, 0.5f);
        AnchoredPosition = Vector2.zero;
        SizeDelta = new Vector2(100f, 100f);
        Scale = Vector2.one;
    }
}

/// <summary>A rectangle filled with <see cref="Color"/>, or with a texture multiplied by it.</summary>
// Layout must match Rust CImage.
[StructLayout(LayoutKind.Sequential)]
public struct Image
{
    AssetBytes _texture;

    /// <summary>sRGB, straight alpha.</summary>
    public Color Color;

    /// <summary>
    /// The nine-slice borders in pixels of the texture: x left, y top, z right,
    /// w bottom. Zero stretches the whole texture.
    /// </summary>
    public Vector4 Slice;

    uint _raycastTarget;

    /// <summary>Plain white, unsliced, and stopping the pointer.</summary>
    public Image()
    {
        Color = Color.white;
        Slice = Vector4.zero;
        _raycastTarget = 1;
    }

    /// <summary>The null reference fills the rectangle with plain colour.</summary>
    public AssetRef<Texture> Texture
    {
        readonly get => new(_texture.ToGuid());
        set => _texture = AssetBytes.Of(value.Id);
    }

    /// <summary>Whether the pointer stops here rather than reaching what is behind.</summary>
    public bool RaycastTarget
    {
        readonly get => _raycastTarget != 0;
        set => _raycastTarget = value ? 1u : 0u;
    }
}

/// <summary>
/// Text set in the element's rectangle. The string itself is read and written
/// with <see cref="World.GetText"/> and <see cref="World.SetText"/>.
/// </summary>
// Layout must match Rust CText.
[StructLayout(LayoutKind.Sequential)]
public struct Text
{
    AssetBytes _font;

    /// <summary>sRGB, straight alpha.</summary>
    public Color Color;

    /// <summary>The height of a line's ascent and descent, in canvas units.</summary>
    public float Size;

    /// <summary>Where the text sits in the rectangle.</summary>
    public TextAnchor Anchor;

    uint _wrap;
    uint _raycastTarget;

    /// <summary>Dark grey, 24 units, at the top left, wrapping, and letting the pointer through.</summary>
    public Text()
    {
        Color = new Color(0.196f, 0.196f, 0.196f);
        Size = 24f;
        Anchor = TextAnchor.UpperLeft;
        _wrap = 1;
        _raycastTarget = 0;
    }

    /// <summary>The null reference is the engine's own face.</summary>
    public AssetRef<Font> Font
    {
        readonly get => new(_font.ToGuid());
        set => _font = AssetBytes.Of(value.Id);
    }

    /// <summary>Off, a line runs past the rectangle's sides rather than breaking.</summary>
    public bool Wrap
    {
        readonly get => _wrap != 0;
        set => _wrap = value ? 1u : 0u;
    }

    /// <summary>
    /// Whether the pointer stops here rather than reaching what is behind. Off for a
    /// new <see cref="Text"/>.
    /// </summary>
    public bool RaycastTarget
    {
        readonly get => _raycastTarget != 0;
        set => _raycastTarget = value ? 1u : 0u;
    }
}

/// <summary>
/// Tints the <see cref="Image"/> on the same entity by the pointer's state over
/// it. What a click does is a Behaviour's: implement
/// <see cref="IPointerClickHandler"/> on the button's entity or above it.
/// </summary>
// Layout must match Rust CButton.
[StructLayout(LayoutKind.Sequential)]
public struct Button
{
    /// <summary>The image's tint while the pointer is elsewhere.</summary>
    public Color Normal;
    /// <summary>The image's tint while the pointer is over it.</summary>
    public Color Hover;
    /// <summary>The image's tint while it is pressed.</summary>
    public Color Pressed;
    /// <summary>The image's tint while it is not <see cref="Interactable"/>.</summary>
    public Color Disabled;

    uint _interactable;

    /// <summary>
    /// Interactable: white, a shade darker under the pointer and darker again
    /// pressed, and half-transparent grey when disabled.
    /// </summary>
    public Button()
    {
        Normal = Color.white;
        Hover = new Color(0.96f, 0.96f, 0.96f);
        Pressed = new Color(0.78f, 0.78f, 0.78f);
        Disabled = new Color(0.78f, 0.78f, 0.78f, 0.5f);
        _interactable = 1;
    }

    /// <summary>
    /// Off, the button draws <see cref="Disabled"/>, and neither it nor anything
    /// under it gets a pointer event — though it still stops the pointer.
    /// </summary>
    public bool Interactable
    {
        readonly get => _interactable != 0;
        set => _interactable = value ? 1u : 0u;
    }
}

/// <summary>
/// Colours what the element fills — its <see cref="Image"/> and its
/// <see cref="Text"/> — with a blend from <see cref="Start"/> to
/// <see cref="End"/>, multiplied into the element's own colour.
/// </summary>
/// <remarks>
/// At <see cref="Rotation"/> 0 the blend runs left to right across the whole
/// box; the angle is in degrees, clockwise on screen. <see cref="Offset"/>
/// moves it by a fraction of the element's size.
/// </remarks>
// Layout must match Rust CGradient.
[StructLayout(LayoutKind.Sequential)]
public struct Gradient
{
    /// <summary>sRGB, straight alpha.</summary>
    public Color Start;

    /// <summary>sRGB, straight alpha.</summary>
    public Color End;

    /// <summary>Degrees, clockwise on screen.</summary>
    public float Rotation;

    /// <summary>A fraction of the element's size.</summary>
    public Vector2 Offset;

    /// <summary>White to black, left to right.</summary>
    public Gradient()
    {
        Start = Color.white;
        End = Color.black;
        Rotation = 0f;
        Offset = Vector2.zero;
    }
}

/// <summary>The curve a <see cref="Corner"/> is cut along.</summary>
public enum CornerStyle : uint
{
    /// <summary>A quarter circle.</summary>
    Round,

    /// <summary>
    /// A quarter superellipse, whose curvature eases in from the straight edge:
    /// the continuous corner of Apple's icons.
    /// </summary>
    Squircle,
}

/// <summary>
/// Rounds the element's box: its image, and the <see cref="Stroke"/> and
/// <see cref="Shadow"/> on it.
/// </summary>
// Layout must match Rust CCorner.
[StructLayout(LayoutKind.Sequential)]
public struct Corner
{
    /// <summary>In canvas units.</summary>
    public float Radius;

    /// <summary>
    /// Added to <see cref="Radius"/>, as a fraction of the box's shorter side:
    /// 0.5 is a pill whatever the size.
    /// </summary>
    public float RadiusFraction;

    /// <summary>The curve the corner is cut along.</summary>
    public CornerStyle Style;

    /// <summary>
    /// The squircle's superellipse exponent, 2 to 12: 2 is a circle, and the
    /// corner squares off as it grows. Ignored by <c>Round</c>.
    /// </summary>
    public float Exponent;

    /// <summary>Round, with an 8-unit radius.</summary>
    public Corner()
    {
        Radius = 8f;
        RadiusFraction = 0f;
        Style = CornerStyle.Round;
        Exponent = 4f;
    }
}

/// <summary>Where a <see cref="Stroke"/> sits against the edge it follows.</summary>
public enum StrokePlacement : uint
{
    /// <summary>Wholly inside the box.</summary>
    Inside,
    /// <summary>Straddling the edge, half inside the box and half out.</summary>
    Center,
    /// <summary>Wholly outside the box.</summary>
    Outside,
}

/// <summary>
/// An outline. On an element with an <see cref="Image"/>, or with nothing else
/// to draw, it follows the box and its <see cref="Corner"/>; on one that is only
/// <see cref="Text"/>, it rings each glyph, always outside it.
/// </summary>
// Layout must match Rust CStroke.
[StructLayout(LayoutKind.Sequential)]
public struct Stroke
{
    /// <summary>sRGB, straight alpha.</summary>
    public Color Color;

    /// <summary>In canvas units.</summary>
    public float Thickness;

    /// <summary>Where the stroke sits against the edge it follows.</summary>
    public StrokePlacement Placement;

    /// <summary>Black, 2 units thick, inside the box.</summary>
    public Stroke()
    {
        Color = Color.black;
        Thickness = 2f;
        Placement = StrokePlacement.Inside;
    }
}

/// <summary>A drop shadow under the element's box, following its <see cref="Corner"/>.</summary>
// Layout must match Rust CShadow.
[StructLayout(LayoutKind.Sequential)]
public struct Shadow
{
    /// <summary>sRGB, straight alpha.</summary>
    public Color Color;

    /// <summary>In canvas units, y down.</summary>
    public Vector2 Offset;

    /// <summary>The blur radius in canvas units, as CSS means it: twice the Gaussian's deviation.</summary>
    public float Blur;

    /// <summary>Grows the shadow's box on every side before it is blurred; negative shrinks it.</summary>
    public float Spread;

    /// <summary>Half-transparent black, 4 units down, with an 8-unit blur.</summary>
    public Shadow()
    {
        Color = new Color(0f, 0f, 0f, 0.5f);
        Offset = new Vector2(0f, 4f);
        Blur = 8f;
        Spread = 0f;
    }
}

/// <summary>When a <see cref="ScrollFrame"/>'s bars are drawn.</summary>
public enum ScrollbarVisibility : uint
{
    /// <summary>On a scrolling axis whose content is larger than the frame.</summary>
    Auto,

    /// <summary>On every scrolling axis.</summary>
    Always,

    /// <summary>On no axis.</summary>
    Never,
}

/// <summary>
/// Lays the element's children out in a content area <see cref="ContentSize"/>
/// units across, cut to the element's rect and scrolled by
/// <see cref="ScrollPosition"/>. The wheel scrolls it while the game runs, as
/// does dragging a bar; what it hides gets no pointer events.
/// </summary>
/// <remarks>
/// The content area is never smaller than the frame, so a stretched child
/// fills at least what is in view. A scroll past either end is laid out at the
/// end.
/// </remarks>
// Layout must match Rust CScrollFrame.
[StructLayout(LayoutKind.Sequential)]
public struct ScrollFrame
{
    /// <summary>In the frame's units; an axis smaller than the frame is the frame's.</summary>
    public Vector2 ContentSize;

    /// <summary>How far the content's top left is scrolled past the frame's, in units.</summary>
    public Vector2 ScrollPosition;

    /// <summary>Units scrolled per notch of the wheel.</summary>
    public float ScrollSpeed;

    /// <summary>In canvas units.</summary>
    public float ScrollbarThickness;

    /// <summary>sRGB, straight alpha.</summary>
    public Color ScrollbarColor;

    /// <summary>When the bars are drawn.</summary>
    public ScrollbarVisibility Scrollbar;

    uint _horizontal;
    uint _vertical;

    /// <summary>
    /// Scrolling vertically only, 40 units a notch, with a 6-unit bar while the
    /// content is taller than the frame.
    /// </summary>
    public ScrollFrame()
    {
        ContentSize = Vector2.zero;
        ScrollPosition = Vector2.zero;
        ScrollSpeed = 40f;
        ScrollbarThickness = 6f;
        ScrollbarColor = new Color(0.6f, 0.6f, 0.6f, 0.8f);
        Scrollbar = ScrollbarVisibility.Auto;
        _horizontal = 0;
        _vertical = 1;
    }

    /// <summary>Whether the content scrolls across.</summary>
    public bool Horizontal
    {
        readonly get => _horizontal != 0;
        set => _horizontal = value ? 1u : 0u;
    }

    /// <summary>Whether the content scrolls down.</summary>
    public bool Vertical
    {
        readonly get => _vertical != 0;
        set => _vertical = value ? 1u : 0u;
    }
}
