namespace Orrin;

// Entity-level world operations. Structural changes (spawns, despawns) are
// queued engine-side and applied after the current script tick, but the
// returned Entity handle is real and usable immediately. One consequence: a
// renderable spawned this tick has no readable Transform until next tick —
// keep your own copy if you need it in the same frame.
public static class World
{
    /// Spawn an entity rendered with a mesh and material registered in the
    /// engine's asset registry (meshes: "cube", "sphere", "plane"; materials:
    /// "gold", "copper", "glossy", "clay", "neon", "textured", "rock",
    /// "ground"). Logs and returns Entity.Null if a name is unknown (check
    /// IsValid).
    public static Entity SpawnRenderable(string mesh, string material, Transform transform) =>
        Native.SpawnRenderable(mesh, material, transform);

    /// Spawn an empty entity.
    public static Entity Spawn() => Native.Spawn();

    /// Queue the entity for despawn at the end of this tick. Returns false if
    /// the handle was already stale.
    public static bool Despawn(Entity entity) => Native.Despawn(entity);

    /// First entity whose Tag equals `tag`, or null if none. "First" is the
    /// engine's storage order — stable between structural changes but
    /// otherwise arbitrary, so don't rely on it when several entities share a
    /// tag; use FindAllByTag instead.
    public static Entity? FindByTag(string tag) => Native.FindByTag(tag);

    /// Every entity whose Tag equals `tag`; empty array if none.
    public static Entity[] FindAllByTag(string tag) => Native.FindAllByTag(tag);

    /// Queue a tag assignment (set or replace) for `entity`. Applied after
    /// this tick like other structural changes, so a FindByTag in the same
    /// tick won't see it yet. Returns false if the handle was stale.
    public static bool SetTag(Entity entity, string tag) => Native.SetTag(entity, tag);

    /// Queue attaching a collision shape to `entity`; it starts participating
    /// in collision detection next frame. Returns false if the handle was
    /// stale.
    public static bool AddCollider(Entity entity, Collider collider) => collider switch
    {
        BoxCollider box => Native.AddBoxCollider(entity, box.HalfExtents, box.IsTrigger),
        SphereCollider sphere => Native.AddSphereCollider(entity, sphere.Radius, sphere.IsTrigger),
        _ => throw new ArgumentException($"unsupported collider type {collider.GetType().Name}"),
    };

    /// Queue a material swap (names come from the same registry as
    /// SpawnRenderable). Logs and returns false if the name is unknown or the
    /// handle was stale.
    public static bool SetMaterial(Entity entity, string material) =>
        Native.SetMaterial(entity, material);

    /// Queue attaching a Behaviour of type T to `entity`. The instance is
    /// created when this tick's commands apply; OnEnable/OnStart fire on the
    /// next tick. Returns false if the handle was stale.
    public static bool AddScript<T>(Entity entity) where T : Behaviour, new() =>
        Native.AddScript(entity, typeof(T).AssemblyQualifiedName!);
}
