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
        throw new ArgumentException(
            $"{typeof(T).Name} is not read or written as a struct; it has calls of its own on World");
    }
}
