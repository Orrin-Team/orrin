using Orrin.Math;

namespace Orrin;

// Values must match `map_key` in crates/core/src/scene/input.rs; extend
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
}

public enum MouseButton : uint
{
    Left = 0,
    Right = 1,
    Middle = 2,
}

// Polled input, valid during OnStart/OnUpdate. GetKeyDown/GetKeyUp are
// edge-triggered: true only on the frame the key changed state. Input the
// editor UI claims (e.g. typing in a panel) is not visible here.
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
