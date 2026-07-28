using System.Diagnostics;

using Orrin.Math;

namespace Orrin;

/// <summary>
/// Developer-facing debug output: structured logging that surfaces in the editor
/// console, and immediate-mode line drawing rendered as a scene overlay.
/// </summary>
/// <remarks>
/// The <c>Draw*</c> methods are marked <see cref="ConditionalAttribute"/> on the
/// <c>ORRIN_DEBUG</c> symbol, so in export builds (which do not define it) the
/// C# compiler removes the call sites entirely — including argument evaluation —
/// at zero cost. As a second line of defence the engine also ignores line data
/// when it is not running the editor overlay. The logging methods are always
/// live, like Unity's <c>Debug.Log</c>.
/// </remarks>
public static class Debug
{
    /// <summary>Log an informational message to the editor console.</summary>
    public static void Log(string message) => Native.Log(message);

    /// <summary>Log a warning to the editor console.</summary>
    public static void LogWarning(string message) => Native.LogWarn(message);

    /// <summary>Log an error to the editor console.</summary>
    public static void LogError(string message) => Native.LogError(message);

    /// <summary>
    /// Draw a line from <paramref name="from"/> to <paramref name="to"/> in world
    /// space, visible for <paramref name="duration"/> seconds (0 = one frame).
    /// Editor-only; stripped from export builds.
    /// </summary>
    [Conditional("ORRIN_DEBUG")]
    public static void DrawLine(Vector3 from, Vector3 to, Color color, float duration = 0f) =>
        Native.DebugDrawLine(from, to, color, duration);

    /// <summary>
    /// Draw a ray from <paramref name="origin"/> extending along
    /// <paramref name="direction"/> (its full length), visible for
    /// <paramref name="duration"/> seconds (0 = one frame). Editor-only.
    /// </summary>
    [Conditional("ORRIN_DEBUG")]
    public static void DrawRay(Vector3 origin, Vector3 direction, Color color, float duration = 0f) =>
        DrawLine(origin, origin + direction, color, duration);
}
