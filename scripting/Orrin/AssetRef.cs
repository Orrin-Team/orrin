namespace Orrin;

/// <summary>
/// The kinds of asset an <see cref="AssetRef{T}"/> can name. Closed: an asset type is added
/// here and on the engine side together.
/// </summary>
public abstract class Asset
{
    private protected Asset() { }
}

/// What every <see cref="AssetRef{T}"/> is, for code that handles one without knowing `T`
/// — the property bag, which writes one as its id.
interface IAssetRef
{
    Guid Id { get; }
}

/// <summary>
/// A reference to an asset by its id: what a Behaviour field, a scene and a
/// prefab store, and never a path. The id lives in the sidecar beside the
/// asset's file, so moving or renaming the file keeps every reference to it.
/// </summary>
/// <remarks>
/// <c>default</c> is the null reference, the value of a field nothing assigned.
/// Assign one in the inspector by dropping an asset onto the field, or in tools
/// with <see cref="Assets.Find{T}"/>.
/// </remarks>
public readonly struct AssetRef<T> : IAssetRef, IEquatable<AssetRef<T>> where T : Asset
{
    readonly Guid _id;

    /// <summary>
    /// The asset whose id is <paramref name="id"/>; <see cref="Guid.Empty"/> is the
    /// null reference.
    /// </summary>
    public AssetRef(Guid id) => _id = id;

    /// <summary>The asset's id; <see cref="Guid.Empty"/> for the null reference.</summary>
    public Guid Id => _id;

    /// <summary>
    /// False for the null reference. Says nothing about whether the asset still
    /// exists; <see cref="State"/> does.
    /// </summary>
    public bool IsValid => _id != Guid.Empty;

    /// <summary>
    /// Whether the asset is drawable yet. A missing or broken asset is
    /// <see cref="AssetState.Failed"/> rather than an exception: it draws as a placeholder, and the
    /// engine has already logged its name.
    /// </summary>
    public AssetState State => IsValid ? Native.GetAssetState(_id) : AssetState.Failed;

    /// <summary>Whether both name the same asset.</summary>
    public static bool operator ==(AssetRef<T> a, AssetRef<T> b) => a._id == b._id;

    /// <summary>Whether they name different assets.</summary>
    public static bool operator !=(AssetRef<T> a, AssetRef<T> b) => a._id != b._id;

    /// <inheritdoc/>
    public bool Equals(AssetRef<T> other) => _id == other._id;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is AssetRef<T> other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => _id.GetHashCode();

    /// <summary>The id in canonical form, as the scene file spells it.</summary>
    public override string ToString() => _id.ToString();
}
