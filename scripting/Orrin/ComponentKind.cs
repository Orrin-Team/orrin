namespace Orrin;

// Numbering is lock-step with the `kind` match in the renderer's scripting.rs
// (same rule as KeyCode: append, never renumber).
public enum ComponentKind : uint
{
    Transform = 0,
    Tag = 1,
    Collider = 2,
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
        throw new ArgumentException($"{typeof(T).Name} is not a script-visible engine component");
    }
}
