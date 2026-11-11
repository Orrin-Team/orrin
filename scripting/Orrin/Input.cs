using System.Collections.Generic;

using Orrin.Math;

namespace Orrin;

// Values must match the table in crates/core/src/scene/input/keys.rs; extend
// both together.
public enum KeyCode : uint
{
    A = 1, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,

    Alpha0 = 30, Alpha1, Alpha2, Alpha3, Alpha4,
    Alpha5, Alpha6, Alpha7, Alpha8, Alpha9,

    LeftArrow = 40,
    RightArrow = 41,
    UpArrow = 42,
    DownArrow = 43,
    Space = 44,
    Return = 45,
    Escape = 46,
    Tab = 47,
    Backspace = 48,
    LeftShift = 49,
    RightShift = 50,
    LeftControl = 51,
    RightControl = 52,
    LeftAlt = 53,
    RightAlt = 54,

    F1 = 55, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,

    Minus = 67,
    Equals = 68,
    LeftBracket = 69,
    RightBracket = 70,
    Backslash = 71,
    Semicolon = 72,
    Quote = 73,
    Comma = 74,
    Period = 75,
    Slash = 76,
    BackQuote = 77,

    Keypad0 = 78, Keypad1, Keypad2, Keypad3, Keypad4,
    Keypad5, Keypad6, Keypad7, Keypad8, Keypad9,
    KeypadPlus = 88,
    KeypadMinus = 89,
    KeypadMultiply = 90,
    KeypadDivide = 91,
    KeypadPeriod = 92,
    KeypadEnter = 93,

    Insert = 94,
    Delete = 95,
    Home = 96,
    End = 97,
    PageUp = 98,
    PageDown = 99,
    CapsLock = 100,
}

public enum MouseButton : uint
{
    Left = 0,
    Right = 1,
    Middle = 2,
}

// Polled input, valid during OnStart/OnUpdate. The Down/Up queries are
// edge-triggered: true only on the frame the key or button changed state. Input
// the editor UI claims (e.g. typing in a panel) is not visible here.
//
// Prefer the named actions below to the raw key queries: a name is bound in
// the project's input.toml, so remapping a control is a text edit rather than
// a code change. The raw queries remain for editor tooling and for the debug
// keys every project grows, where a binding would be ceremony.
//
// Every named query takes an optional player. Slot 0 is the keyboard, the
// mouse, and the first pad; higher slots are pads only, so a second player
// cannot act on the first player's keyboard.
public static class Input
{
    /// True while the key is held.
    public static bool GetKey(KeyCode key) => Native.KeyDown((uint)key);

    /// True on the frame the key went down.
    public static bool GetKeyDown(KeyCode key) => Native.KeyPressed((uint)key);

    /// True on the frame the key was released.
    public static bool GetKeyUp(KeyCode key) => Native.KeyReleased((uint)key);

    /// True while the mouse button is held.
    public static bool GetMouseButton(MouseButton button) =>
        Native.MouseButtonDown((uint)button);

    /// True on the frame the mouse button went down.
    public static bool GetMouseButtonDown(MouseButton button) =>
        Native.MouseButtonPressed((uint)button);

    /// True on the frame the mouse button was released.
    public static bool GetMouseButtonUp(MouseButton button) =>
        Native.MouseButtonReleased((uint)button);

    /// True while any binding for the action is held.
    public static bool IsHeld(string action, int player = 0) =>
        Native.ActionHeld(Handle(action), Slot(player));

    /// True on the frame the action went down.
    ///
    /// The edge belongs to the action, not to a binding: an action already held
    /// on one binding does not fire again when a second one joins it.
    public static bool IsPressed(string action, int player = 0) =>
        Native.ActionPressed(Handle(action), Slot(player));

    /// True on the frame the last binding holding the action let go.
    public static bool IsReleased(string action, int player = 0) =>
        Native.ActionReleased(Handle(action), Slot(player));

    /// An axis's value. A composed axis reads −1, 0 or 1; an analog one reads
    /// its source, deadzoned and scaled as the config asked. An action name, or
    /// a name nothing binds, reads zero.
    public static float GetAxis(string axis, int player = 0) =>
        Native.AxisValue(Handle(axis), Slot(player));

    // Handles are stable for the life of the process — a binding reload changes
    // what one points at, never what it is — so this cache never needs
    // invalidating and a name crosses the boundary once rather than per query.
    private static readonly Dictionary<string, uint> Handles = new();

    // What the engine answers when there was no table to intern into. Mirrors
    // `orrin_script::NO_ACTION`.
    private const uint NoAction = uint.MaxValue;

    private static uint Handle(string name)
    {
        if (Handles.TryGetValue(name, out var id))
            return id;
        id = Native.ActionId(name);
        // A non-answer is not cached. The cache never invalidates, so storing
        // one would make a single unlucky early query — asked before the engine
        // had a world to intern into — permanent for the process, and the name
        // would read as unbound forever afterwards.
        if (id != NoAction)
            Handles[name] = id;
        return id;
    }

    private static uint Slot(int player) => (uint)System.Math.Max(player, 0);

    /// This frame's raw mouse motion, x right and y down, in the device's own
    /// units rather than pixels — what a first-person camera turns by. Arrives
    /// whether or not the cursor moves, so it keeps coming at a screen edge
    /// and while the cursor is locked.
    public static Vector2 mouseDelta => Native.MouseDelta();

    /// <summary>
    /// How far the wheel turned this frame, in lines: x right, y away from the
    /// user. A trackpad's pixels arrive at the rate the editor camera reads
    /// them. Zero while the editor has the pointer, so scrolling a panel does
    /// not also zoom the game.
    /// </summary>
    public static Vector2 scrollDelta => Native.ScrollDelta();

    /// Hide the cursor and hold it in the window, for mouse look. A request:
    /// the window holds it only while the game runs and has focus, and in the
    /// editor Escape lets go, so a locked game can always be left.
    public static bool cursorLocked
    {
        get => Native.CursorLocked();
        set => Native.SetCursorLocked(value);
    }

    /// Cursor position in window coordinates (physical pixels).
    public static Vector2 MousePosition
    {
        get
        {
            var (x, y) = Native.CursorPos();
            return new Vector2(x, y);
        }
    }
}
