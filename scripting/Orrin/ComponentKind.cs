namespace Orrin;

// Numbering is lock-step with the `kind` match in the renderer's scripting.rs
// (same rule as KeyCode: append, never renumber).
public enum ComponentKind : uint
{
    Transform = 0,
    Tag = 1,
    Collider = 2,
    Camera = 3,
    Canvas = 4,
    RectTransform = 5,
    Image = 6,
    Text = 7,
    Button = 8,
    Gradient = 9,
    Corner = 10,
    Stroke = 11,
    Shadow = 12,
    ScrollFrame = 13,
    Rigidbody = 14,
    AudioSource = 15,
    MeshRenderer = 16,
}

internal static class ComponentKinds
{
    /// The ABI kind id for a script-visible engine component type; throws for
    /// types the engine doesn't expose (better a loud error than a silent
    /// "false" from a typo'd HasComponent check).
    internal static uint Of<T>() where T : struct
    {
        if (typeof(T) == typeof(Transform)) return (uint)ComponentKind.Transform;
        if (typeof(T) == typeof(Tag)) return (uint)ComponentKind.Tag;
        if (typeof(T) == typeof(CameraComponent)) return (uint)ComponentKind.Camera;
        if (typeof(T) == typeof(Canvas)) return (uint)ComponentKind.Canvas;
        if (typeof(T) == typeof(RectTransform)) return (uint)ComponentKind.RectTransform;
        if (typeof(T) == typeof(Image)) return (uint)ComponentKind.Image;
        if (typeof(T) == typeof(Text)) return (uint)ComponentKind.Text;
        if (typeof(T) == typeof(Button)) return (uint)ComponentKind.Button;
        if (typeof(T) == typeof(Gradient)) return (uint)ComponentKind.Gradient;
        if (typeof(T) == typeof(Corner)) return (uint)ComponentKind.Corner;
        if (typeof(T) == typeof(Stroke)) return (uint)ComponentKind.Stroke;
        if (typeof(T) == typeof(Shadow)) return (uint)ComponentKind.Shadow;
        if (typeof(T) == typeof(ScrollFrame)) return (uint)ComponentKind.ScrollFrame;
        if (typeof(T) == typeof(Rigidbody)) return (uint)ComponentKind.Rigidbody;
        if (typeof(T) == typeof(AudioSource)) return (uint)ComponentKind.AudioSource;
        if (typeof(T) == typeof(MeshRenderer)) return (uint)ComponentKind.MeshRenderer;
        throw new ArgumentException($"{typeof(T).Name} is not a script-visible engine component");
    }

    /// The kind of a component World.GetComponent, SetComponent and
    /// AddComponent carry as a struct; throws for any other, which has its own
    /// calls (World.AddCamera, World.SetTag, …).
    internal static uint Blittable<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(Canvas)) return (uint)ComponentKind.Canvas;
        if (typeof(T) == typeof(RectTransform)) return (uint)ComponentKind.RectTransform;
        if (typeof(T) == typeof(Image)) return (uint)ComponentKind.Image;
        if (typeof(T) == typeof(Text)) return (uint)ComponentKind.Text;
        if (typeof(T) == typeof(Button)) return (uint)ComponentKind.Button;
        if (typeof(T) == typeof(Gradient)) return (uint)ComponentKind.Gradient;
        if (typeof(T) == typeof(Corner)) return (uint)ComponentKind.Corner;
        if (typeof(T) == typeof(Stroke)) return (uint)ComponentKind.Stroke;
        if (typeof(T) == typeof(Shadow)) return (uint)ComponentKind.Shadow;
        if (typeof(T) == typeof(ScrollFrame)) return (uint)ComponentKind.ScrollFrame;
        if (typeof(T) == typeof(Rigidbody)) return (uint)ComponentKind.Rigidbody;
        if (typeof(T) == typeof(AudioSource)) return (uint)ComponentKind.AudioSource;
        if (typeof(T) == typeof(MeshRenderer)) return (uint)ComponentKind.MeshRenderer;
        throw new ArgumentException(
            $"{typeof(T).Name} is not read or written as a struct; it has calls of its own on World");
    }
}
