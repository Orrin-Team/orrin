namespace Orrin;

/// What the engine's `find_asset` reports. Lock-step with `ASSET_*` in
/// orrin-script: append, never renumber.
static class AssetKinds
{
    public const uint None = 0;
    public const uint Prefab = 1;

    public static uint Of<T>() where T : Asset =>
        typeof(T) == typeof(Orrin.Prefab)
            ? Prefab
            : throw new ArgumentException($"{typeof(T).Name} is not an asset kind the engine knows");
}

public static class Assets
{
    /// The asset at `path`, relative to the project's assets directory with `/`
    /// separators, e.g. `"props/door.fprefab"`.
    ///
    /// The one place a path becomes a reference, for tools. Game code should
    /// hold an `AssetRef<T>` field instead: a path in code breaks the day the
    /// file moves, and a reference does not. Logs and returns the null
    /// reference if there is no `T` at `path`.
    public static AssetRef<T> Find<T>(string path) where T : Asset
    {
        var (id, kind) = Native.FindAsset(path);
        if (kind == AssetKinds.Of<T>())
            return new AssetRef<T>(id);
        Native.LogWarn($"Assets.Find: there is no {typeof(T).Name} at \"{path}\"");
        return default;
    }
}
