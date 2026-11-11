namespace Orrin;

/// <summary>Severity of a log line, quietest first.</summary>
/// <remarks>
/// The numbering crosses the ABI as a <c>uint</c> and matches the engine's own
/// <c>LogLevel</c>: append, never renumber — the same rule as the function
/// table and the key codes.
/// </remarks>
public enum LogLevel : uint
{
    /// <summary><see cref="Orrin.Debug.LogTrace"/>; never in an export build.</summary>
    Trace = 0,
    /// <summary><see cref="Orrin.Debug.LogDebug"/>; never in an export build.</summary>
    Debug = 1,
    /// <summary><see cref="Orrin.Debug.Log"/>.</summary>
    Info = 2,
    /// <summary><see cref="Orrin.Debug.LogWarning"/>.</summary>
    Warning = 3,
    /// <summary><see cref="Orrin.Debug.LogError"/>, and a script that throws.</summary>
    Error = 4,
}
