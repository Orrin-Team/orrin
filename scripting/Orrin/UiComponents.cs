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
    public Vector2 AnchorMin;
    public Vector2 AnchorMax;
    public Vector2 Pivot;
    public Vector2 AnchoredPosition;
    public Vector2 SizeDelta;
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

    public TextAnchor Anchor;

    uint _wrap;
    uint _raycastTarget;

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
    public Color Normal;
    public Color Hover;
    public Color Pressed;
    public Color Disabled;

    uint _interactable;

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
