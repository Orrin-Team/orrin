using Ferron;
using Ferron.Math;

namespace DemoGame;

// Transform-only "alive" animation: the cube drifts in a gentle figure-8, bobs,
// breathes (squash & stretch synced to the bob), and tumbles on a tilted axis.
public class Hover : Behaviour
{
    public float BobHeight = 0.5f;
    public float BobSpeed = 1.8f;
    public float DriftRadius = 1.2f;
    public float TumbleSpeed = 0.6f;
    public float Breathe = 0.15f;

    private Vector3 _basePosition;
    private Vector3 _baseScale;
    private float _time;

    public override void OnStart()
    {
        var t = Transform;
        _basePosition = t.Position;
        _baseScale = t.Scale;
        Native.Log($"Hover attached to {Entity}");
    }

    public override void OnUpdate(float deltaTime)
    {
        _time += deltaTime;

        float bob = Mathf.Sin(_time * BobSpeed);

        // Figure-8 on the XZ plane (Lissajous, 1:2 frequency ratio) plus the bob.
        var drift = new Vector3(
            Mathf.Sin(_time * 0.9f) * DriftRadius,
            bob * BobHeight,
            Mathf.Sin(_time * 1.8f) * DriftRadius * 0.5f);

        // Tall + thin at the top of the arc, short + wide at the bottom.
        float stretch = 1f + bob * Breathe;
        float squash = 1f - bob * Breathe * 0.5f;

        var axis = new Vector3(0.3f, 1f, 0.25f).normalized;

        var t = Transform;
        t.Position = _basePosition + drift;
        t.Rotation = Quaternion.AngleAxis(_time * TumbleSpeed * Mathf.Rad2Deg, axis);
        t.Scale = new Vector3(_baseScale.x * squash, _baseScale.y * stretch, _baseScale.z * squash);
        Transform = t;
    }
}
