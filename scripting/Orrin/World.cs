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

    /// An empty entity standing at `transform`, at once: a camera, a pivot,
    /// anything placed with no mesh. Its Transform can be read and written this
    /// tick, unlike the components queued by the Add calls.
    public static Entity Spawn(Transform transform) => Native.SpawnWithTransform(transform);

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

    /// Queue attaching a collision shape to `entity`; it becomes a physics
    /// body next frame — static, unless something moves it or it has a
    /// <see cref="Rigidbody"/>. Returns false if the handle was stale.
    public static bool AddCollider(Entity entity, Collider collider) =>
        Native.AddCollider(entity, collider.ToData());

    /// Queue making `entity` a camera, looking along its Transform's -Z. It
    /// can be made <see cref="Camera.Active"/> this tick and draws from next
    /// frame. Logs and returns false for a lens that cannot project (see
    /// <see cref="CameraComponent"/>) or a stale handle.
    public static bool AddCamera(Entity entity, CameraComponent camera) =>
        Native.AddCamera(entity, camera);

    /// Change a camera's lens in place, from this frame. Logs and returns
    /// false for a lens that cannot project; returns false if `entity` is not
    /// a camera.
    public static bool SetCamera(Entity entity, CameraComponent camera) =>
        Native.SetCameraLens(entity, camera);

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

    /// <summary>
    /// <paramref name="entity"/>'s <see cref="Canvas"/>, <see cref="RectTransform"/>,
    /// <see cref="Image"/>, <see cref="Text"/> or <see cref="Button"/> as it is
    /// now, or null if it has none.
    /// </summary>
    public static T? GetComponent<T>(Entity entity) where T : unmanaged =>
        Native.GetComponent<T>(entity);

    /// <summary>
    /// Change <paramref name="entity"/>'s component in place, from this frame — or
    /// the one queued for it this tick by <see cref="AddComponent{T}"/>. Returns
    /// false if it has neither.
    /// </summary>
    public static bool SetComponent<T>(Entity entity, T value) where T : unmanaged =>
        Native.SetComponent(entity, value);

    /// <summary>
    /// Queue giving <paramref name="entity"/> a UI component; it lands when this
    /// tick's commands apply, replacing one already there. A
    /// <see cref="SetComponent{T}"/> or <see cref="SetText"/> later in the same
    /// tick changes the queued one. Returns false for a stale handle.
    /// </summary>
    public static bool AddComponent<T>(Entity entity, T value) where T : unmanaged =>
        Native.AddComponent(entity, value);

    /// <summary>The string <paramref name="entity"/>'s <see cref="Text"/> shows, or null if it has none.</summary>
    public static string? GetText(Entity entity) => Native.GetText(entity);

    /// <summary>
    /// Change what <paramref name="entity"/>'s <see cref="Text"/> shows. Returns
    /// false if it has no Text, nor one queued this tick.
    /// </summary>
    public static bool SetText(Entity entity, string text) => Native.SetText(entity, text);
}
