using System.Buffers;
using System.Text;

using Orrin.Math;

namespace Orrin;

/// <summary>
/// Where text sits inside its rectangle. Lock-step with the engine's
/// <c>scene::ui::TextAnchor</c>: append, never renumber.
/// </summary>
public enum TextAnchor : uint
{
    UpperLeft,
    UpperCenter,
    UpperRight,
    MiddleLeft,
    MiddleCenter,
    MiddleRight,
    LowerLeft,
    LowerCenter,
    LowerRight,
}

/// <summary>
/// The in-game UI: HUDs, menus and overlays, drawn over the finished frame in a
/// shipped game as in the editor.
/// </summary>
/// <remarks>
/// Immediate mode. Call these from <c>OnUpdate</c> every frame they should be
/// on screen; a call draws over every call before it. Rectangles are in
/// physical pixels from the window's top left, the space
/// <see cref="Input.MousePosition"/> reports, and <see cref="ScreenSize"/> is
/// the window's size. Colours are sRGB, the numbers a colour picker shows.
///
/// The style properties are read at each call, so set them before the calls
/// they should apply to. They are not reset between frames.
/// </remarks>
public static class UI
{
    /// The longest text encoded on the stack rather than in a pooled buffer.
    const int StackBytes = 512;

    static ulong? _polled;
    static Vector2 _pointer;
    static bool _held;
    static bool _released;
    /// Where the left button went down, while it is held or on the frame it
    /// comes up.
    static Vector2? _press;

    /// <summary>The window's size in physical pixels.</summary>
    public static Vector2 ScreenSize => Native.UiScreenSize();

    /// <summary>
    /// Whether the pointer is over a retained UI element that stops it — an
    /// <see cref="Orrin.Image"/> or <see cref="Orrin.Text"/> with <c>RaycastTarget</c> on —
    /// as of the start of this tick. Use it to keep a click on a menu from also
    /// firing a weapon. The immediate calls on this class are not counted.
    /// </summary>
    public static bool IsPointerOverUI => Native.UiPointerOver();

    /// <summary>The font text is set in. The null reference is the engine's own.</summary>
    public static AssetRef<Orrin.Font> Font { get; set; }

    /// <summary>The height of a line's ascent and descent, in physical pixels.</summary>
    public static float FontSize { get; set; } = 18f;

    public static Color TextColor { get; set; } = Color.white;

    /// <summary>Where <see cref="Label"/> sets its text. A button's is always centred.</summary>
    public static TextAnchor Alignment { get; set; } = TextAnchor.UpperLeft;

    public static Color ButtonColor { get; set; } = new(0.16f, 0.16f, 0.18f, 0.9f);
    public static Color ButtonHoverColor { get; set; } = new(0.26f, 0.26f, 0.3f, 0.95f);
    public static Color ButtonPressedColor { get; set; } = new(0.1f, 0.1f, 0.11f, 1f);

    /// <summary>Fill <paramref name="rect"/> with <paramref name="color"/>.</summary>
    public static void Panel(Rect rect, Color color) => Native.UiPanel(rect, color);

    /// <summary>
    /// Draw <paramref name="text"/> in <paramref name="rect"/>, wrapped to its
    /// width, clipped to it, and placed by <see cref="Alignment"/>.
    /// </summary>
    public static void Label(string text, Rect rect) => Text(text, rect, Alignment);

    /// <summary>
    /// Draw a button and report whether it was clicked: the left mouse button
    /// came up over it this frame, having gone down over it.
    /// </summary>
    public static bool Button(string text, Rect rect)
    {
        Poll();
        var over = rect.Contains(_pointer);
        var pressedHere = _press is { } press && rect.Contains(press);
        Panel(rect, pressedHere && _held && over ? ButtonPressedColor
            : over ? ButtonHoverColor
            : ButtonColor);
        Text(text, rect, TextAnchor.MiddleCenter);
        return _released && over && pressedHere;
    }

    /// <summary>Stretch <paramref name="texture"/> over <paramref name="rect"/>.</summary>
    public static void DrawTexture(AssetRef<Texture> texture, Rect rect) =>
        DrawTexture(texture, rect, Color.white);

    /// <summary>
    /// Stretch <paramref name="texture"/> over <paramref name="rect"/>,
    /// multiplied by <paramref name="tint"/>. Draws nothing until the texture
    /// has loaded, and nothing for the null reference.
    /// </summary>
    public static void DrawTexture(AssetRef<Texture> texture, Rect rect, Color tint)
    {
        if (texture.IsValid)
            Native.UiImage(texture.Id, rect, tint);
    }

    static void Text(string text, Rect rect, TextAnchor anchor)
    {
        if (string.IsNullOrEmpty(text))
            return;
        var most = Encoding.UTF8.GetMaxByteCount(text.Length);
        byte[]? pooled = null;
        var buffer = most <= StackBytes
            ? stackalloc byte[StackBytes]
            : (pooled = ArrayPool<byte>.Shared.Rent(most));
        var length = Encoding.UTF8.GetBytes(text, buffer);
        Guid? font = Font.IsValid ? Font.Id : null;
        Native.UiText(buffer[..length], font, FontSize, anchor, rect, TextColor);
        if (pooled is not null)
            ArrayPool<byte>.Shared.Return(pooled);
    }

    /// <summary>
    /// The pointer and the left button as of this frame, read once whichever
    /// call comes first. A press is remembered only while it lasts, so a
    /// release over a button counts only for the button it began on.
    /// </summary>
    static void Poll()
    {
        var frame = Time.FrameCount;
        if (_polled == frame)
            return;
        _polled = frame;
        _pointer = Input.MousePosition;
        _held = Input.GetMouseButton(MouseButton.Left);
        _released = Input.GetMouseButtonUp(MouseButton.Left);
        if (Input.GetMouseButtonDown(MouseButton.Left))
            _press = _pointer;
        else if (!_held && !_released)
            _press = null;
    }
}
