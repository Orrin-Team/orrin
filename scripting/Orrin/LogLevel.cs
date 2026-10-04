namespace Orrin;

/// <summary>Severity of a log line, quietest first.</summary>
/// <remarks>
/// The numbering crosses the ABI as a <c>uint</c> and matches the engine's own
/// <c>LogLevel</c>: append, never renumber — the same rule as the function
/// table and the key codes.
/// </remarks>
public enum LogLevel : uint
{
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
}
