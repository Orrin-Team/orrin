using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Orrin.Math;

namespace Orrin;

/// <summary>
/// A UI field a tween can drive. Lock-step with the engine's
/// <c>ui::tween::TweenProperty</c>: append, never renumber.
/// </summary>
public enum TweenProperty : uint
{
    /// <summary><see cref="RectTransform.AnchoredPosition"/>, a <see cref="Vector2"/>.</summary>
    Position,
    /// <summary><see cref="RectTransform.SizeDelta"/>, a <see cref="Vector2"/>.</summary>
    Size,
    /// <summary><see cref="RectTransform.Scale"/>, a <see cref="Vector2"/>.</summary>
    Scale,
    /// <summary><see cref="Image.Color"/>, a <see cref="Color"/>.</summary>
    ImageColor,
    /// <summary><see cref="Text.Color"/>, a <see cref="Color"/>.</summary>
    TextColor,
    /// <summary><see cref="Text.Size"/>, a float.</summary>
    TextSize,
    /// <summary><see cref="Corner.Radius"/>, a float.</summary>
    CornerRadius,
    /// <summary><see cref="Stroke.Color"/>, a <see cref="Color"/>.</summary>
    StrokeColor,
    /// <summary><see cref="Stroke.Thickness"/>, a float.</summary>
    StrokeThickness,
    /// <summary><see cref="Shadow.Color"/>, a <see cref="Color"/>.</summary>
    ShadowColor,
    /// <summary><see cref="Shadow.Offset"/>, a <see cref="Vector2"/>.</summary>
    ShadowOffset,
    /// <summary><see cref="Shadow.Blur"/>, a float.</summary>
    ShadowBlur,
    /// <summary><see cref="Gradient.Start"/>, a <see cref="Color"/>.</summary>
    GradientStart,
    /// <summary><see cref="Gradient.End"/>, a <see cref="Color"/>.</summary>
    GradientEnd,
    /// <summary><see cref="Gradient.Rotation"/>, a float in degrees.</summary>
    GradientRotation,
    /// <summary><see cref="ScrollFrame.ScrollPosition"/>, a <see cref="Vector2"/>.</summary>
    ScrollPosition,
}

/// <summary>
/// The shape of a tween's curve. Lock-step with the engine's
/// <c>ui::tween::EaseStyle</c>: append, never renumber.
/// </summary>
public enum EaseStyle : uint
{
    /// <summary>Constant speed, whatever the direction.</summary>
    Linear,
    /// <summary>A quarter of a cosine wave: the gentlest of the curves.</summary>
    Sine,
    /// <summary>As <see cref="EaseDirection.In"/>, t².</summary>
    Quad,
    /// <summary>As <see cref="EaseDirection.In"/>, t³.</summary>
    Cubic,
    /// <summary>As <see cref="EaseDirection.In"/>, t⁴.</summary>
    Quart,
    /// <summary>As <see cref="EaseDirection.In"/>, t⁵.</summary>
    Quint,
    /// <summary>As <see cref="EaseDirection.In"/>, 2^(10t − 10): all but still, then sudden.</summary>
    Expo,
    /// <summary>As <see cref="EaseDirection.In"/>, a quarter circle: 1 − √(1 − t²).</summary>
    Circ,
    /// <summary>Pulls back past the start before it moves.</summary>
    Back,
    /// <summary>Overshoots and rings like a spring.</summary>
    Elastic,
    /// <summary>Bounces to rest like a dropped ball, at the end the direction shapes.</summary>
    Bounce,
}

/// <summary>Which end of the curve the style shapes. Lock-step with <c>ui::tween::EaseDirection</c>.</summary>
public enum EaseDirection : uint
{
    /// <summary>The style's own curve: slow to leave the start.</summary>
    In,
    /// <summary>The curve mirrored: slow to reach the end.</summary>
    Out,
    /// <summary>In for the first half, Out for the second: slow at both ends.</summary>
    InOut,
}

/// <summary>How a tween moves: its length, curve, delay and repetition.</summary>
public readonly struct TweenInfo
{
    /// <summary>Seconds from start to end.</summary>
    public readonly float Duration;
    /// <summary>The shape of the curve.</summary>
    public readonly EaseStyle Style;
    /// <summary>Which end of the curve the style shapes.</summary>
    public readonly EaseDirection Direction;
    /// <summary>Seconds to wait, once, before the first cycle.</summary>
    public readonly float Delay;
    /// <summary>Further cycles after the first; negative repeats forever.</summary>
    public readonly int Repeat;
    /// <summary>Each cycle runs back to the start after reaching the end.</summary>
    public readonly bool Reverses;

    /// <summary>
    /// Each argument is the field of the same name. The defaults are a one-second
    /// <see cref="EaseStyle.Quad"/> <see cref="EaseDirection.Out"/> that runs once.
    /// </summary>
    public TweenInfo(
        float duration = 1f,
        EaseStyle style = EaseStyle.Quad,
        EaseDirection direction = EaseDirection.Out,
        float delay = 0f,
        int repeat = 0,
        bool reverses = false)
    {
        Duration = duration;
        Style = style;
        Direction = direction;
        Delay = delay;
        Repeat = repeat;
        Reverses = reverses;
    }
}

// Layout must match Rust CTween.
[StructLayout(LayoutKind.Sequential)]
internal struct TweenRequest
{
    public TweenProperty Property;
    public EaseStyle Style;
    public EaseDirection Direction;
    public int Repeat;
    public float Duration;
    public float Delay;
    public uint Reverses;
    public uint HasFrom;
    public Vector4 From;
    public Vector4 To;
}

/// <summary>
/// A UI field easing from one value to another, run by the engine. Start one
/// with <see cref="To(Entity, TweenProperty, float, TweenInfo)"/> or its
/// overloads; a second tween on the same field of the same element replaces
/// the first from wherever it got to.
/// </summary>
/// <remarks>
/// A tween first moves after the tick that started it, and its completion
/// callback runs at the start of a later tick. A tween that is cancelled,
/// replaced, or whose element is despawned never completes. Callbacks are
/// forgotten when the game's scripts are reloaded; the tweens themselves keep
/// running.
/// </remarks>
public readonly struct Tween : IEquatable<Tween>
{
    /// <summary>The engine's id for the tween; zero for one that was refused.</summary>
    public readonly ulong Id;

    internal Tween(ulong id) => Id = id;

    /// <summary>Whether the tween is still running.</summary>
    public bool IsPlaying => Id != 0 && Native.TweenPlaying(Id);

    /// <summary>Stop the tween where it is. False when it was not playing.</summary>
    public bool Cancel()
    {
        Tweens.Forget(Id);
        return Id != 0 && Native.TweenCancel(Id);
    }

    /// <summary>Run <paramref name="callback"/> when the tween finishes.</summary>
    public Tween OnComplete(Action callback)
    {
        if (Id != 0)
            Tweens.Add(Id, callback);
        return this;
    }

    /// <summary>
    /// Ease <paramref name="entity"/>'s <paramref name="property"/> from where it is
    /// now to <paramref name="to"/>, as <paramref name="info"/> describes. For a float
    /// property: one of another shape throws <see cref="ArgumentException"/>. A tween
    /// the engine refuses has an <see cref="Id"/> of zero.
    /// </summary>
    public static Tween To(Entity entity, TweenProperty property, float to, TweenInfo info) =>
        Play(entity, property, 1, null, new Vector4(to, 0f, 0f, 0f), info);

    /// <summary>
    /// <see cref="To(Entity, TweenProperty, float, TweenInfo)"/> for a
    /// <see cref="Vector2"/> property.
    /// </summary>
    public static Tween To(Entity entity, TweenProperty property, Vector2 to, TweenInfo info) =>
        Play(entity, property, 2, null, new Vector4(to.x, to.y, 0f, 0f), info);

    /// <summary>
    /// <see cref="To(Entity, TweenProperty, float, TweenInfo)"/> for a
    /// <see cref="Color"/> property.
    /// </summary>
    public static Tween To(Entity entity, TweenProperty property, Color to, TweenInfo info) =>
        Play(entity, property, 4, null, Lanes(to), info);

    /// <summary>
    /// <see cref="To(Entity, TweenProperty, float, TweenInfo)"/>, starting from
    /// <paramref name="from"/> rather than from where the property is now.
    /// </summary>
    public static Tween FromTo(Entity entity, TweenProperty property, float from, float to, TweenInfo info) =>
        Play(entity, property, 1, new Vector4(from, 0f, 0f, 0f), new Vector4(to, 0f, 0f, 0f), info);

    /// <summary>
    /// <see cref="FromTo(Entity, TweenProperty, float, float, TweenInfo)"/> for a
    /// <see cref="Vector2"/> property.
    /// </summary>
    public static Tween FromTo(Entity entity, TweenProperty property, Vector2 from, Vector2 to, TweenInfo info) =>
        Play(entity, property, 2, new Vector4(from.x, from.y, 0f, 0f), new Vector4(to.x, to.y, 0f, 0f), info);

    /// <summary>
    /// <see cref="FromTo(Entity, TweenProperty, float, float, TweenInfo)"/> for a
    /// <see cref="Color"/> property.
    /// </summary>
    public static Tween FromTo(Entity entity, TweenProperty property, Color from, Color to, TweenInfo info) =>
        Play(entity, property, 4, Lanes(from), Lanes(to), info);

    /// <summary>How many floats <paramref name="property"/>'s value is.</summary>
    internal static int Lanes(TweenProperty property) => property switch
    {
        TweenProperty.Position or TweenProperty.Size or TweenProperty.Scale
            or TweenProperty.ShadowOffset or TweenProperty.ScrollPosition => 2,
        TweenProperty.ImageColor or TweenProperty.TextColor or TweenProperty.StrokeColor
            or TweenProperty.ShadowColor or TweenProperty.GradientStart or TweenProperty.GradientEnd => 4,
        _ => 1,
    };

    static Vector4 Lanes(Color color) => new(color.r, color.g, color.b, color.a);

    // A value of the wrong shape would write zeroes into the lanes it does not
    // carry — a Vector2 into a colour is a transparent black — so it is
    // refused here, where the call is still on the stack.
    static Tween Play(Entity entity, TweenProperty property, int lanes, Vector4? from, Vector4 to, TweenInfo info)
    {
        if (Lanes(property) != lanes)
            throw new ArgumentException(
                $"{property} is {Lanes(property)} float(s), and the value given is {lanes}",
                nameof(property));
        var request = new TweenRequest
        {
            Property = property,
            Style = info.Style,
            Direction = info.Direction,
            Repeat = info.Repeat,
            Duration = info.Duration,
            Delay = info.Delay,
            Reverses = info.Reverses ? 1u : 0u,
            HasFrom = from.HasValue ? 1u : 0u,
            From = from ?? default,
            To = to,
        };
        return new Tween(Native.TweenPlay(entity, request));
    }

    /// <inheritdoc/>
    public bool Equals(Tween other) => Id == other.Id;
    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Tween other && Equals(other);
    /// <inheritdoc/>
    public override int GetHashCode() => Id.GetHashCode();
    /// <summary>Whether both are the same tween.</summary>
    public static bool operator ==(Tween a, Tween b) => a.Equals(b);
    /// <summary>Whether they are different tweens.</summary>
    public static bool operator !=(Tween a, Tween b) => !a.Equals(b);
}

/// The completion callbacks, and the entry point the engine reports a finished
/// tween through.
public static class Tweens
{
    static readonly Dictionary<ulong, Action> Callbacks = new();

    /// The size at which `Add` next sweeps out callbacks whose tween was
    /// replaced engine-side, which nothing reports. Doubling keeps the sweep
    /// amortised to a constant per callback.
    static int _sweepAt = 64;

    internal static void Add(ulong id, Action callback)
    {
        Callbacks[id] = Callbacks.TryGetValue(id, out var existing) ? existing + callback : callback;
        if (Callbacks.Count < _sweepAt)
            return;
        foreach (var stale in Callbacks.Keys.Where(key => !Native.TweenPlaying(key)).ToList())
            Callbacks.Remove(stale);
        _sweepAt = System.Math.Max(64, Callbacks.Count * 2);
    }

    internal static void Forget(ulong id) => Callbacks.Remove(id);

    /// Drop every callback. Each one is a delegate into the game assembly, and
    /// one surviving a reload pins the retired load context.
    internal static void ForgetGame()
    {
        Callbacks.Clear();
        _sweepAt = 64;
    }

    [UnmanagedCallersOnly]
    public static void Complete(ulong id)
    {
        if (!Callbacks.Remove(id, out var callback))
            return;
        try
        {
            callback();
        }
        catch (Exception e)
        {
            // Never across the boundary; see `Behaviours`.
            Native.LogEvent(
                LogLevel.Error,
                $"[script] a tween's OnComplete threw {e.GetType().Name}: {e.Message}\n{e.StackTrace}",
                "",
                0);
        }
    }
}
