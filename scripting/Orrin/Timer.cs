namespace Orrin;

/// <summary>
/// A callback a Behaviour scheduled with <c>After</c> or <c>Every</c>. Keep it
/// to <see cref="Cancel"/> the call; dropping it changes nothing.
/// </summary>
public sealed class TimerHandle
{
    Action? _callback;
    float _remaining;
    readonly float _interval;

    internal TimerHandle(float seconds, Action callback, float interval)
    {
        _remaining = seconds;
        _callback = callback;
        _interval = interval;
    }

    /// <summary>
    /// Whether the callback is still to run: false once a one-shot timer has
    /// fired, or once the timer is cancelled.
    /// </summary>
    public bool IsPending => _callback is not null;

    internal bool Repeats => _interval > 0f;

    /// <summary>
    /// Stop the callback from running again. False when it was no longer
    /// pending.
    /// </summary>
    public bool Cancel()
    {
        if (_callback is null)
            return false;
        _callback = null;
        return true;
    }

    /// Runs the callback at most once a tick, so a tick several intervals long
    /// drops the calls it skipped rather than bursting them. The remainder
    /// carries over, which keeps a repeating timer on its phase.
    internal void Tick(float deltaTime)
    {
        if (_callback is not { } callback)
            return;
        _remaining -= deltaTime;
        if (_remaining > 0f)
            return;
        if (Repeats)
            _remaining = _interval + _remaining % _interval;
        else
            _callback = null;
        callback();
    }
}
