namespace Orrin;

/// The kinds of asset an [AssetRef] can name. Closed: an asset type is added
/// here and on the engine side together.
public abstract class Asset
{
    private protected Asset() { }
}

/// What every `AssetRef<T>` is, for code that handles one without knowing `T`
/// — the property bag, which writes one as its id.
interface IAssetRef
{
    Guid Id { get; }
}

/// A reference to an asset by its id: what a Behaviour field, a scene and a
/// prefab store, and never a path. The id lives in the sidecar beside the
/// asset's file, so moving or renaming the file keeps every reference to it.
///
/// `default` is the null reference, the value of a field nothing assigned.
/// Assign one in the inspector by dropping an asset onto the field, or in tools
/// with [Assets.Find].
public readonly struct AssetRef<T> : IAssetRef, IEquatable<AssetRef<T>> where T : Asset
{
    readonly Guid _id;

    public AssetRef(Guid id) => _id = id;

    public Guid Id => _id;

    /// False for the null reference. Says nothing about whether the asset still
    /// exists.
    public bool IsValid => _id != Guid.Empty;

    public static bool operator ==(AssetRef<T> a, AssetRef<T> b) => a._id == b._id;

    public static bool operator !=(AssetRef<T> a, AssetRef<T> b) => a._id != b._id;

    public bool Equals(AssetRef<T> other) => _id == other._id;

    public override bool Equals(object? obj) => obj is AssetRef<T> other && Equals(other);

    public override int GetHashCode() => _id.GetHashCode();

    /// The id in canonical form, as the scene file spells it.
    public override string ToString() => _id.ToString();
}
