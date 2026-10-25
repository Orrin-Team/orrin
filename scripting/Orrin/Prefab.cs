namespace Orrin;

/// An entity tree saved as a `.fprefab` file under the project's assets
/// directory, instantiated whole.
public sealed class Prefab : Asset
{
    Prefab() { }

    /// Spawn the prefab's tree and return its root.
    ///
    /// The root is a real handle at once, like `World.SpawnRenderable`'s, but
    /// the tree is built when this tick's structural changes apply: the root's
    /// components and its children are there from the next tick. Logs and
    /// returns `Entity.Null` if the reference names no prefab (check
    /// `IsValid`).
    public static Entity Instantiate(AssetRef<Prefab> prefab) => Native.InstantiatePrefab(prefab.Id);
}
