namespace Orrin;

/// <summary>
/// An entity tree saved as a <c>.oprefab</c> file under the project's assets
/// directory, instantiated whole.
/// </summary>
public sealed class Prefab : Asset
{
    Prefab() { }

    /// <summary>Spawn the prefab's tree and return its root.</summary>
    /// <remarks>
    /// The root is a real handle at once, like <c>World.SpawnRenderable</c>'s, but
    /// the tree is built when this tick's structural changes apply: the root's
    /// components and its children are there from the next tick. Logs and
    /// returns <see cref="Entity.Null"/> if the reference names no prefab (check
    /// <see cref="Entity.IsValid"/>).
    /// </remarks>
    public static Entity Instantiate(AssetRef<Prefab> prefab) => Native.InstantiatePrefab(prefab.Id);
}
