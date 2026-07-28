namespace Orrin;

/// <summary>Frame timing, read live from the engine's Time resource.</summary>
public static class Time
{
    /// <summary>Seconds since the last frame.</summary>
    public static float DeltaTime => Native.TimeDelta();

    /// <summary>Total elapsed seconds since engine start.</summary>
    public static float Total => Native.TimeTotal();

    /// <summary>Frames ticked since engine start; handy for debugging frame-dependent bugs.</summary>
    public static ulong FrameCount => Native.TimeFrameCount();
}
