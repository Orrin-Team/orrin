using System.Runtime.InteropServices;
using System.Text;

using Orrin.Math;

namespace Orrin;

// Field order and types must match the Rust OrrinApi struct.
[StructLayout(LayoutKind.Sequential)]
public unsafe struct OrrinApi
{
    public delegate* unmanaged<byte*, void> Log;
    public delegate* unmanaged<Entity> Spawn;
    public delegate* unmanaged<Entity, Transform*, byte> GetTransform;
    public delegate* unmanaged<Entity, Transform*, byte> SetTransform;
    public delegate* unmanaged<uint, byte> KeyDown;
    public delegate* unmanaged<uint, byte> KeyPressed;
    public delegate* unmanaged<uint, byte> KeyReleased;
    public delegate* unmanaged<uint, byte> MouseButtonDown;
    public delegate* unmanaged<float*, float*, void> CursorPos;
    public delegate* unmanaged<byte*, byte*, Transform*, Entity> SpawnRenderable;
    public delegate* unmanaged<Entity, byte> Despawn;
    public delegate* unmanaged<float> TimeDelta;
    public delegate* unmanaged<float> TimeTotal;
    public delegate* unmanaged<ulong> TimeFrameCount;
    public delegate* unmanaged<byte*, Entity*, byte> FindByTag;
    public delegate* unmanaged<byte*, Entity*, int, int> FindAllByTag;
    public delegate* unmanaged<Entity, uint, byte> HasComponent;
    public delegate* unmanaged<Entity, byte*, int, int> GetTag;
    public delegate* unmanaged<Entity, byte*, byte> SetTag;
    public delegate* unmanaged<Entity, float, float, float, byte, byte> AddBoxCollider;
    public delegate* unmanaged<Entity, float, byte, byte> AddSphereCollider;
    public delegate* unmanaged<Entity, byte*, byte> SetMaterial;
    public delegate* unmanaged<Entity, byte*, byte> AddScript;
    public delegate* unmanaged<byte*, void> LogWarn;
    public delegate* unmanaged<byte*, void> LogError;
    // from(x,y,z), to(x,y,z), color(r,g,b,a), duration — loose floats to match
    // the Rust `debug_draw_line` signature.
    public delegate* unmanaged<
        float, float, float, float, float, float, float, float, float, float, float, void>
        DebugDrawLine;
    // Hierarchy. Appended after DebugDrawLine, matching the Rust struct; never
    // reordered above it.
    public delegate* unmanaged<Entity, Transform*, byte> GetWorldTransform;
    public delegate* unmanaged<Entity, Transform*, byte> SetWorldTransform;
    public delegate* unmanaged<Entity, Entity> GetParent;
    public delegate* unmanaged<Entity, Entity, byte, byte> SetParent;
}

public static unsafe class Native
{
    private static OrrinApi _api;

    internal static void Initialize(OrrinApi* api) => _api = *api;

    public static void Log(string message) => Emit(_api.Log, message);

    public static void LogWarn(string message) => Emit(_api.LogWarn, message);

    public static void LogError(string message) => Emit(_api.LogError, message);

    public static void DebugDrawLine(Vector3 from, Vector3 to, Color color, float duration) =>
        _api.DebugDrawLine(
            from.x, from.y, from.z, to.x, to.y, to.z, color.r, color.g, color.b, color.a, duration);

    // Marshal `message` as a nul-terminated UTF-8 buffer and hand it to a native
    // string sink (Log/LogWarn/LogError share this).
    private static void Emit(delegate* unmanaged<byte*, void> sink, string message)
    {
        var bytes = Encoding.UTF8.GetBytes(message);
        Span<byte> buffer = stackalloc byte[bytes.Length + 1];
        bytes.CopyTo(buffer);
        buffer[bytes.Length] = 0;
        fixed (byte* p = buffer)
            sink(p);
    }

    public static Entity Spawn() => _api.Spawn();

    public static Transform GetTransform(Entity entity)
    {
        Transform transform = default;
        _api.GetTransform(entity, &transform);
        return transform;
    }

    public static void SetTransform(Entity entity, Transform value) =>
        _api.SetTransform(entity, &value);

    /// The entity's transform in world space.
    ///
    /// The engine holds this as a matrix, so what comes back is the closest
    /// position/rotation/scale fit to it. That is exact for any chain of
    /// rotations and uniform scales, and approximate once a non-uniformly
    /// scaled ancestor introduces shear — the scale is the lossy part, the
    /// position never is.
    public static Transform GetWorldTransform(Entity entity)
    {
        Transform transform = default;
        _api.GetWorldTransform(entity, &transform);
        return transform;
    }

    /// Place the entity in world space, whatever it is parented to. The engine
    /// composes with the inverse of the parent's transform, so a script never
    /// has to know it has a parent.
    public static void SetWorldTransform(Entity entity, Transform value) =>
        _api.SetWorldTransform(entity, &value);

    /// The entity's parent, or `Entity.Null` if it is a root.
    public static Entity GetParent(Entity entity) => _api.GetParent(entity);

    /// Attach `entity` to `parent`, or pass `Entity.Null` to detach it.
    ///
    /// With `keepWorld` the entity stays where it is and only its local
    /// transform is rewritten; without it, the local transform is kept and the
    /// entity moves into its new parent's frame — which is what attaching a
    /// pickup to a hand wants.
    ///
    /// Returns false, and changes nothing, if the move would create a cycle,
    /// parent an entity to itself, or name a despawned entity. The reparent
    /// itself is structural, so it applies after the current tick's dispatch —
    /// `GetParent` will not reflect it until the next frame.
    public static bool SetParent(Entity entity, Entity parent, bool keepWorld = true) =>
        _api.SetParent(entity, parent, keepWorld ? (byte)1 : (byte)0) != 0;

    public static bool KeyDown(uint code) => _api.KeyDown(code) != 0;

    public static bool KeyPressed(uint code) => _api.KeyPressed(code) != 0;

    public static bool KeyReleased(uint code) => _api.KeyReleased(code) != 0;

    public static bool MouseButtonDown(uint button) => _api.MouseButtonDown(button) != 0;

    public static (float X, float Y) CursorPos()
    {
        float x = 0, y = 0;
        _api.CursorPos(&x, &y);
        return (x, y);
    }

    public static Entity SpawnRenderable(string mesh, string material, Transform transform)
    {
        var meshBytes = NulTerminated(mesh);
        var materialBytes = NulTerminated(material);
        fixed (byte* meshPtr = meshBytes)
        fixed (byte* materialPtr = materialBytes)
            return _api.SpawnRenderable(meshPtr, materialPtr, &transform);
    }

    public static bool Despawn(Entity entity) => _api.Despawn(entity) != 0;

    public static float TimeDelta() => _api.TimeDelta();

    public static float TimeTotal() => _api.TimeTotal();

    public static ulong TimeFrameCount() => _api.TimeFrameCount();

    public static Entity? FindByTag(string tag)
    {
        var tagBytes = NulTerminated(tag);
        Entity entity = default;
        fixed (byte* tagPtr = tagBytes)
            return _api.FindByTag(tagPtr, &entity) != 0 ? entity : null;
    }

    public static Entity[] FindAllByTag(string tag)
    {
        var tagBytes = NulTerminated(tag);
        var buffer = new Entity[16];
        while (true)
        {
            int total;
            fixed (byte* tagPtr = tagBytes)
            fixed (Entity* outPtr = buffer)
                total = _api.FindAllByTag(tagPtr, outPtr, buffer.Length);
            if (total <= buffer.Length)
                return buffer[..total];
            // The world can't change between the two calls (scripts are the
            // only mutator inside a tick), so one retry always suffices.
            buffer = new Entity[total];
        }
    }

    public static bool HasComponent(Entity entity, uint kind) =>
        _api.HasComponent(entity, kind) != 0;

    public static string? GetTag(Entity entity)
    {
        Span<byte> buffer = stackalloc byte[64];
        int length;
        fixed (byte* p = buffer)
            length = _api.GetTag(entity, p, buffer.Length);
        if (length < 0)
            return null;
        if (length <= buffer.Length)
            return Encoding.UTF8.GetString(buffer[..length]);
        var bytes = new byte[length];
        fixed (byte* p = bytes)
            _api.GetTag(entity, p, bytes.Length);
        return Encoding.UTF8.GetString(bytes);
    }

    public static bool SetTag(Entity entity, string tag)
    {
        var tagBytes = NulTerminated(tag);
        fixed (byte* tagPtr = tagBytes)
            return _api.SetTag(entity, tagPtr) != 0;
    }

    public static bool AddBoxCollider(Entity entity, Vector3 halfExtents, bool isTrigger) =>
        _api.AddBoxCollider(entity, halfExtents.x, halfExtents.y, halfExtents.z,
            (byte)(isTrigger ? 1 : 0)) != 0;

    public static bool AddSphereCollider(Entity entity, float radius, bool isTrigger) =>
        _api.AddSphereCollider(entity, radius, (byte)(isTrigger ? 1 : 0)) != 0;

    public static bool SetMaterial(Entity entity, string material)
    {
        var materialBytes = NulTerminated(material);
        fixed (byte* materialPtr = materialBytes)
            return _api.SetMaterial(entity, materialPtr) != 0;
    }

    public static bool AddScript(Entity entity, string typeName)
    {
        var nameBytes = NulTerminated(typeName);
        fixed (byte* namePtr = nameBytes)
            return _api.AddScript(entity, namePtr) != 0;
    }

    private static byte[] NulTerminated(string value)
    {
        var bytes = new byte[Encoding.UTF8.GetByteCount(value) + 1];
        Encoding.UTF8.GetBytes(value, 0, value.Length, bytes, 0);
        return bytes;
    }
}
