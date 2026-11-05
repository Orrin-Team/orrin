using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

/// <summary>
/// What happened to the pointer over a UI element. Lock-step with the engine's
/// <c>ui::events::PointerKind</c>: append, never renumber.
/// </summary>
public enum PointerEventKind : uint
{
    Enter,
    Exit,
    Down,
    Up,
    Click,
}

/// <summary>What a pointer handler is told about the event.</summary>
// Layout must match Rust CPointerEvent.
[StructLayout(LayoutKind.Sequential)]
public readonly struct PointerEvent
{
    /// <summary>The pointer, in physical pixels from the window's top left.</summary>
    public readonly Vector2 Position;

    /// <summary>Where the button went down, for a press, a release and a click.</summary>
    public readonly Vector2 PressPosition;

    /// <summary>
    /// The element the pointer hit. For a release or a click, the one it was
    /// pressed on. A handler further up the hierarchy sees the element below it
    /// here, not itself.
    /// </summary>
    public readonly Entity Target;

    public readonly MouseButton Button;

    public PointerEvent(Vector2 position, Vector2 pressPosition, Entity target, MouseButton button)
    {
        Position = position;
        PressPosition = pressPosition;
        Target = target;
        Button = button;
    }
}

/// <summary>
/// The pointer started being over this element or one under it. Sent to every
/// element it enters, deepest first; it does not bubble.
/// </summary>
public interface IPointerEnterHandler
{
    void OnPointerEnter(PointerEvent pointer);
}

/// <summary>The pointer is no longer over this element or anything under it.</summary>
public interface IPointerExitHandler
{
    void OnPointerExit(PointerEvent pointer);
}

/// <summary>
/// The left button went down over this element. Bubbles: an element with no
/// handler passes it to its parent, and the first handler up the hierarchy
/// takes it.
/// </summary>
public interface IPointerDownHandler
{
    void OnPointerDown(PointerEvent pointer);
}

/// <summary>
/// The left button came up after going down on this element, wherever the
/// pointer is now. Bubbles like a press.
/// </summary>
public interface IPointerUpHandler
{
    void OnPointerUp(PointerEvent pointer);
}

/// <summary>
/// The left button went down and came up over this same element. Bubbles like
/// a press.
/// </summary>
public interface IPointerClickHandler
{
    void OnPointerClick(PointerEvent pointer);
}

internal static class PointerEvents
{
    /// Hand `pointer` to `behaviour`'s handler for `kind`, and say whether it
    /// had one — what decides if the engine offers it to the parent next.
    internal static bool Dispatch(Behaviour behaviour, PointerEventKind kind, in PointerEvent pointer)
    {
        switch (kind)
        {
            case PointerEventKind.Enter when behaviour is IPointerEnterHandler handler:
                handler.OnPointerEnter(pointer);
                return true;
            case PointerEventKind.Exit when behaviour is IPointerExitHandler handler:
                handler.OnPointerExit(pointer);
                return true;
            case PointerEventKind.Down when behaviour is IPointerDownHandler handler:
                handler.OnPointerDown(pointer);
                return true;
            case PointerEventKind.Up when behaviour is IPointerUpHandler handler:
                handler.OnPointerUp(pointer);
                return true;
            case PointerEventKind.Click when behaviour is IPointerClickHandler handler:
                handler.OnPointerClick(pointer);
                return true;
            default:
                return false;
        }
    }

    /// The handler `kind` calls, for a fault message.
    internal static string MethodName(PointerEventKind kind) => kind switch
    {
        PointerEventKind.Enter => nameof(IPointerEnterHandler.OnPointerEnter),
        PointerEventKind.Exit => nameof(IPointerExitHandler.OnPointerExit),
        PointerEventKind.Down => nameof(IPointerDownHandler.OnPointerDown),
        PointerEventKind.Up => nameof(IPointerUpHandler.OnPointerUp),
        PointerEventKind.Click => nameof(IPointerClickHandler.OnPointerClick),
        _ => $"pointer event {(uint)kind}",
    };
}
