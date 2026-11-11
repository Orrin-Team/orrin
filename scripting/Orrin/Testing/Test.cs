using System.Diagnostics.CodeAnalysis;

using Orrin.Math;

namespace Orrin.Testing;

/// <summary>
/// Marks a public static method as a test, run by <c>orrin test</c> in a fresh,
/// empty world with the physics step and the script tick of a playing game, and
/// no window. The method takes nothing and returns <c>void</c>, or
/// <c>IEnumerator</c>/<c>IEnumerable</c> to run across frames.
/// </summary>
/// <remarks>
/// A <c>void</c> test runs once and no frame passes during it. An iterator test
/// runs to its first <c>yield return</c>, then the engine runs one frame of
/// 1/60 s — physics, then every Behaviour's OnStart and OnUpdate, then the
/// spawns and despawns they asked for — and resumes it; <c>yield return
/// null</c> waits one frame, <see cref="Wait.Frames"/> and
/// <see cref="Wait.Seconds"/> longer. A test fails when it throws, when a
/// Behaviour throws while it runs, or when it is still running after
/// <see cref="MaxFrames"/> frames. Everything it spawned is gone before the
/// next test starts.
/// </remarks>
[AttributeUsage(AttributeTargets.Method, Inherited = false)]
public sealed class TestAttribute : Attribute
{
    /// <summary>The frame budget a test has unless it sets <see cref="MaxFrames"/>.</summary>
    public const int DefaultMaxFrames = 10_000;

    /// <summary>
    /// Frames the test may run before it fails as hung. A <c>void</c> test runs
    /// none, so this only limits an iterator test.
    /// </summary>
    public int MaxFrames { get; set; } = DefaultMaxFrames;

    /// <summary>
    /// A scene to load into the world before the test starts, relative to the
    /// project's assets directory, its Behaviours created and not yet started.
    /// </summary>
    public string? Scene { get; set; }
}

/// <summary>What a failed <see cref="Assert"/> throws.</summary>
public sealed class AssertionException(string message) : Exception(message);

/// <summary>Checks that fail a test, by throwing <see cref="AssertionException"/>.</summary>
public static class Assert
{
    /// <summary>Fail with <paramref name="message"/> unless <paramref name="condition"/> holds.</summary>
    public static void That(bool condition, string message)
    {
        if (!condition)
            throw new AssertionException(message);
    }

    /// <summary>Fail unless <paramref name="condition"/> is true.</summary>
    public static void IsTrue(bool condition, string? message = null) =>
        That(condition, message ?? "expected true, was false");

    /// <summary>Fail unless <paramref name="condition"/> is false.</summary>
    public static void IsFalse(bool condition, string? message = null) =>
        That(!condition, message ?? "expected false, was true");

    /// <summary>Fail unless <paramref name="actual"/> equals <paramref name="expected"/>.</summary>
    public static void AreEqual<T>(T expected, T actual, string? message = null)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            throw new AssertionException(Explain(message, $"expected {expected}, was {actual}"));
    }

    /// <summary>
    /// Fail unless <paramref name="actual"/> is within <paramref name="tolerance"/>
    /// of <paramref name="expected"/>.
    /// </summary>
    public static void AreApproximatelyEqual(float expected, float actual, float tolerance = 1e-5f, string? message = null)
    {
        if (!(MathF.Abs(expected - actual) <= tolerance))
            throw new AssertionException(Explain(message, $"expected {expected} ± {tolerance}, was {actual}"));
    }

    /// <summary>
    /// Fail unless <paramref name="actual"/> is within <paramref name="tolerance"/>
    /// of <paramref name="expected"/>, measured as a distance.
    /// </summary>
    public static void AreApproximatelyEqual(Vector3 expected, Vector3 actual, float tolerance = 1e-5f, string? message = null)
    {
        if (!(Vector3.Distance(expected, actual) <= tolerance))
            throw new AssertionException(Explain(message, $"expected {expected} ± {tolerance}, was {actual}"));
    }

    /// <summary>Fail now, with <paramref name="message"/>.</summary>
    [DoesNotReturn]
    public static void Fail(string message) => throw new AssertionException(message);

    static string Explain(string? message, string detail) =>
        message is null ? detail : $"{message}: {detail}";
}

/// <summary>
/// Something a test can <c>yield return</c> to wait longer than the one frame
/// <c>yield return null</c> waits.
/// </summary>
public sealed class Wait
{
    internal int FrameCount { get; }
    internal float SecondCount { get; }

    Wait(int frames, float seconds)
    {
        FrameCount = frames;
        SecondCount = seconds;
    }

    /// <summary>Resume after <paramref name="count"/> frames, and always after at least one.</summary>
    public static Wait Frames(int count) => new(System.Math.Max(count, 1), 0f);

    /// <summary>
    /// Resume once <paramref name="seconds"/> of game time have passed: the
    /// frames' <see cref="Time.DeltaTime"/>, so a <see cref="Time.timeScale"/>
    /// of 0 waits until the test's frame budget runs out.
    /// </summary>
    public static Wait Seconds(float seconds) => new(0, seconds);
}
