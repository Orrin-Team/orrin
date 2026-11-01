namespace Orrin;

/// A `.orrin` scene file under the project's assets directory.
public sealed class Scene : Asset
{
    Scene() { }
}

/// A loaded scene, for as long as it stays loaded. Handles are never reused,
/// so one kept after its scene has gone names nothing rather than whatever
/// loaded next. `default` is the null handle.
///
/// A Behaviour field holding one is `[Transient]`: a handle is this session's,
/// and means nothing in a saved scene.
public readonly struct SceneHandle : IEquatable<SceneHandle>
{
    internal readonly uint Value;

    internal SceneHandle(uint value) => Value = value;

    /// False for the null handle: what a failed `LoadAdditive` returns. Says
    /// nothing about whether the scene is still loaded.
    public bool IsValid => Value != 0;

    public static bool operator ==(SceneHandle a, SceneHandle b) => a.Value == b.Value;

    public static bool operator !=(SceneHandle a, SceneHandle b) => a.Value != b.Value;

    public bool Equals(SceneHandle other) => Value == other.Value;

    public override bool Equals(object? obj) => obj is SceneHandle other && Equals(other);

    public override int GetHashCode() => (int)Value;

    public override string ToString() => $"scene {Value}";
}

/// Loading and unloading scenes while the game runs.
///
/// Every change lands when this tick's scripts have all run, like a spawn or
/// a despawn: the scripts of a scene going away get OnDisable and OnDestroy
/// then, and those of a scene arriving get OnEnable and OnStart on the next
/// tick. OnDestroy runs outside the tick, so it cannot read the world.
///
/// The project's persistent scene (`[scenes] persistent` in orrin.toml) is
/// loaded beside the first scene and kept through every `Load` — the place
/// for game managers, music and anything else that outlives a level.
public static class SceneManager
{
    /// Replace the active scene with `scene`, and unload every other scene
    /// but the persistent one. Logs and returns false if the reference names
    /// no scene.
    public static bool Load(AssetRef<Scene> scene) => Native.LoadScene(scene.Id);

    /// Load `scene` beside what is loaded, and return its handle at once.
    /// Logs and returns the null handle if the reference names no scene.
    public static SceneHandle LoadAdditive(AssetRef<Scene> scene) =>
        new(Native.LoadSceneAdditive(scene.Id));

    /// Unload a scene `LoadAdditive` loaded, and everything parented into it.
    /// Logs and returns false for the active scene — `Load` replaces that —
    /// or a scene that is no longer loaded.
    public static bool Unload(SceneHandle scene) => Native.UnloadScene(scene.Value);

    /// The scene a `Load` would replace.
    public static SceneHandle Active => new(Native.ActiveScene());
}
