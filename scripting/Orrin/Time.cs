namespace Orrin;

/// <summary>Frame timing, read live from the engine's Time resource.</summary>
public static class Time
{
    /// <summary>
    /// Seconds of game time since the last frame: the real frame scaled by
    /// <see cref="timeScale"/>. Inside OnFixedUpdate it is the physics step,
    /// <see cref="fixedDeltaTime"/>, so code that multiplies by it reads the
    /// same from either hook.
    /// </summary>
    public static float DeltaTime => Native.TimeDelta();

    /// <summary>Total elapsed seconds since engine start.</summary>
    public static float Total => Native.TimeTotal();

    /// <summary>Frames ticked since engine start; handy for debugging frame-dependent bugs.</summary>
    public static ulong FrameCount => Native.TimeFrameCount();

    /// <summary>
    /// Seconds since the last frame, whatever the time scale: for a pause
    /// menu's own animation, or anything that must keep real time.
    /// </summary>
    public static float unscaledDeltaTime => Native.TimeUnscaledDelta();

    /// <summary>
    /// The physics step in seconds, from the scene's physics settings. The time
    /// scale never changes it: a slower game takes fewer steps of this length.
    /// </summary>
    public static float fixedDeltaTime => Native.TimeFixedDelta();

    /// <summary>
    /// How fast game time runs against real time; 1 by default. It scales
    /// OnUpdate's delta and <see cref="DeltaTime"/>, timers, tweens and how many
    /// physics steps a frame takes; it does not touch the editor. 0 pauses the
    /// game while OnUpdate keeps running with a zero delta. Below 0 reads as 0,
    /// and a value that is not finite is ignored. It takes effect from the next
    /// frame, lasts until changed or until the play session ends, and survives
    /// a scene load.
    /// </summary>
    public static float timeScale
    {
        get => Native.TimeScale();
        set => Native.SetTimeScale(value);
    }
}
