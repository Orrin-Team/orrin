namespace Orrin;

/// What the engine's `find_asset` reports. Lock-step with `ASSET_*` in
/// orrin-script: append, never renumber.
static class AssetKinds
{
    public const uint None = 0;
    public const uint Prefab = 1;
    public const uint Mesh = 2;
    public const uint Texture = 3;

    public static uint Of<T>() where T : Asset
    {
        if (typeof(T) == typeof(Orrin.Prefab))
            return Prefab;
        if (typeof(T) == typeof(Orrin.Mesh))
            return Mesh;
        if (typeof(T) == typeof(Orrin.Texture))
            return Texture;
        throw new ArgumentException($"{typeof(T).Name} is not an asset kind the engine knows");
    }
}

/// Whether what an asset reference draws is the asset yet. Lock-step with
/// `LOAD_*` in orrin-script: append, never renumber.
public enum AssetState : uint
{
    /// Importing; a placeholder draws in its place until it is ready.
    Loading = 0,
    Ready = 1,
    /// Missing or broken. A placeholder draws in its place, and the engine has
    /// logged an error naming the asset.
    Failed = 2,
}

/// A glTF 2.0 file under the project's assets directory, imported as one mesh.
public sealed class Mesh : Asset
{
    Mesh() { }
}

/// A PNG or JPEG under the project's assets directory.
public sealed class Texture : Asset
{
    Texture() { }
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
