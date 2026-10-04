using System.Diagnostics;
using System.Runtime.CompilerServices;

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
/// when it is not running the editor overlay. <c>LogTrace</c> and
/// <c>LogDebug</c> are stripped the same way, and an export build's engine
/// would drop them regardless. <c>Log</c>, <c>LogWarning</c> and
/// <c>LogError</c> are always live, like Unity's <c>Debug.Log</c>.
///
/// Every log line names the script file and line it was written on. The two
/// trailing parameters are how: the compiler fills them in at the call site, so
/// they are never passed by hand.
/// </remarks>
public static class Debug
{
    /// <summary>Log at trace level. Editor-only; stripped from export builds.</summary>
    [Conditional("ORRIN_DEBUG")]
    public static void LogTrace(
        string message,
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) =>
        Native.LogEvent(LogLevel.Trace, message, Native.ScriptName(file), line);

    /// <summary>Log at debug level. Editor-only; stripped from export builds.</summary>
    [Conditional("ORRIN_DEBUG")]
    public static void LogDebug(
        string message,
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) =>
        Native.LogEvent(LogLevel.Debug, message, Native.ScriptName(file), line);

    /// <summary>Log an informational message to the editor console.</summary>
    public static void Log(
        string message,
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) =>
        Native.LogEvent(LogLevel.Info, message, Native.ScriptName(file), line);

    /// <summary>Log a warning to the editor console.</summary>
    public static void LogWarning(
        string message,
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) =>
        Native.LogEvent(LogLevel.Warning, message, Native.ScriptName(file), line);

    /// <summary>Log an error to the editor console.</summary>
    public static void LogError(
        string message,
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) =>
        Native.LogEvent(LogLevel.Error, message, Native.ScriptName(file), line);

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
