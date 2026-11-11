namespace Orrin;

/// <summary>A <c>.orrin</c> scene file under the project's assets directory.</summary>
public sealed class Scene : Asset
{
    Scene() { }
}

/// <summary>
/// A loaded scene, for as long as it stays loaded. Handles are never reused,
/// so one kept after its scene has gone names nothing rather than whatever
/// loaded next. <c>default</c> is the null handle.
/// </summary>
/// <remarks>
/// A Behaviour field holding one is <c>[Transient]</c>: a handle is this session's,
/// and means nothing in a saved scene.
/// </remarks>
public readonly struct SceneHandle : IEquatable<SceneHandle>
{
    internal readonly uint Value;

    internal SceneHandle(uint value) => Value = value;

    /// <summary>
    /// False for the null handle: what a failed
    /// <see cref="SceneManager.LoadAdditive"/> returns. Says nothing about
    /// whether the scene is still loaded.
    /// </summary>
    public bool IsValid => Value != 0;

    /// <summary>Whether both name the same load of a scene.</summary>
    public static bool operator ==(SceneHandle a, SceneHandle b) => a.Value == b.Value;

    /// <summary>Whether they name different loads.</summary>
    public static bool operator !=(SceneHandle a, SceneHandle b) => a.Value != b.Value;

    /// <inheritdoc/>
    public bool Equals(SceneHandle other) => Value == other.Value;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is SceneHandle other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => (int)Value;

    /// <inheritdoc/>
    public override string ToString() => $"scene {Value}";
}

/// <summary>Loading and unloading scenes while the game runs.</summary>
/// <remarks>
/// <para>
/// Every change lands when this tick's scripts have all run, like a spawn or
/// a despawn: the scripts of a scene going away get OnDisable and OnDestroy
/// then, and those of a scene arriving get OnEnable and OnStart on the next
/// tick. OnDestroy runs outside the tick, so it cannot read the world.
/// </para>
///
/// <para>
/// The project's persistent scene (<c>[scenes] persistent</c> in orrin.toml) is
/// loaded beside the first scene and kept through every <see cref="Load"/> — the place
/// for game managers, music and anything else that outlives a level.
/// </para>
/// </remarks>
public static class SceneManager
{
    /// <summary>
    /// Replace the active scene with <paramref name="scene"/>, and unload every other scene
    /// but the persistent one. Logs and returns false if the reference names
    /// no scene.
    /// </summary>
    public static bool Load(AssetRef<Scene> scene) => Native.LoadScene(scene.Id);

    /// <summary>
    /// Load <paramref name="scene"/> beside what is loaded, and return its handle at once.
    /// Logs and returns the null handle if the reference names no scene.
    /// </summary>
    public static SceneHandle LoadAdditive(AssetRef<Scene> scene) =>
        new(Native.LoadSceneAdditive(scene.Id));

    /// <summary>
    /// Unload a scene <see cref="LoadAdditive"/> loaded, and everything parented into it.
    /// Logs and returns false for the active scene — <see cref="Load"/> replaces that —
    /// or a scene that is no longer loaded.
    /// </summary>
    public static bool Unload(SceneHandle scene) => Native.UnloadScene(scene.Value);

    /// <summary>The scene a <see cref="Load"/> would replace.</summary>
    public static SceneHandle Active => new(Native.ActiveScene());
}
