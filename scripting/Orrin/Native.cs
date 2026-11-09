using System.Runtime.CompilerServices;
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
    // Named input. Appended after SetParent, matching the Rust struct; never
    // reordered above it. The name crosses once, at ActionId; every query after
    // that passes the handle it returned.
    public delegate* unmanaged<byte*, uint> ActionId;
    public delegate* unmanaged<uint, uint, byte> ActionHeld;
    public delegate* unmanaged<uint, uint, byte> ActionPressed;
    public delegate* unmanaged<uint, uint, byte> ActionReleased;
    public delegate* unmanaged<uint, uint, float> AxisValue;
    // The mouse's edges, appended after the named input, matching the Rust
    // struct; never reordered above it.
    public delegate* unmanaged<uint, byte> MouseButtonPressed;
    public delegate* unmanaged<uint, byte> MouseButtonReleased;
    // Structured logging, appended after the mouse edges, matching the Rust
    // struct: level, the script's name (or null), its line (or 0), the message.
    // Every log goes through this one; Log/LogWarn/LogError above are what the
    // table carried before it and are no longer called.
    public delegate* unmanaged<uint, byte*, uint, byte*, void> LogEvent;
    // Assets, appended after LogEvent, matching the Rust struct. An id crosses
    // as its sixteen bytes in canonical UUID order, through a buffer this side
    // owns.
    public delegate* unmanaged<byte*, byte*, uint> FindAsset;
    public delegate* unmanaged<byte*, Entity> InstantiatePrefab;
    // Load state, appended after InstantiatePrefab, matching the Rust struct:
    // an id's sixteen bytes in, an `AssetState` out.
    public delegate* unmanaged<byte*, uint> AssetState;
    // Scenes, appended after AssetState, matching the Rust struct: a scene by
    // its id's sixteen bytes, a loaded one by a nonzero handle.
    public delegate* unmanaged<byte*, byte> LoadScene;
    public delegate* unmanaged<byte*, uint> LoadSceneAdditive;
    public delegate* unmanaged<uint, byte> UnloadScene;
    public delegate* unmanaged<uint> ActiveScene;
    public delegate* unmanaged<Transform*, byte> GetCamera;
    public delegate* unmanaged<Transform*, byte> SetCamera;
    // Camera entities, appended after SetCamera, matching the Rust struct;
    // never reordered above it.
    public delegate* unmanaged<Entity, CameraComponent*, byte> AddCamera;
    public delegate* unmanaged<Entity, CameraComponent*, byte> GetCameraLens;
    public delegate* unmanaged<Entity, CameraComponent*, byte> SetCameraLens;
    public delegate* unmanaged<Entity> GetActiveCamera;
    public delegate* unmanaged<Entity, byte> SetActiveCamera;
    // In-game UI, appended after SetActiveCamera, matching the Rust struct;
    // never reordered above it. A rect is x, y, width, height and a colour
    // r, g, b, a, as loose floats; text is UTF-8 bytes and their length, with
    // no terminator; a font or texture is an id's sixteen bytes, and a null
    // font is the engine's own.
    public delegate* unmanaged<float, float, float, float, float, float, float, float, void> UiPanel;
    public delegate* unmanaged<
        byte*, uint, byte*, float, uint, float, float, float, float, float, float, float, float, void>
        UiText;
    public delegate* unmanaged<byte*, float, float, float, float, float, float, float, float, void>
        UiImage;
    public delegate* unmanaged<float*, float*, void> UiScreenSize;
    // Engine components as blittable structs, appended after UiScreenSize,
    // matching the Rust struct; never reordered above it. A component kind
    // (ComponentKind), the struct and its byte size, which the engine checks;
    // text as UTF-8 through the GetTag protocol.
    public delegate* unmanaged<Entity, uint, byte*, uint, byte> GetComponent;
    public delegate* unmanaged<Entity, uint, byte*, uint, byte> SetComponent;
    public delegate* unmanaged<Entity, uint, byte*, uint, byte> AddComponent;
    public delegate* unmanaged<Entity, uint, byte*, int, int> GetComponentText;
    public delegate* unmanaged<Entity, uint, byte*, uint, byte> SetComponentText;
    public delegate* unmanaged<byte> UiPointerOver;
    // Tweens, appended after UiPointerOver; never reordered above it. A
    // request and its byte size, which the engine checks; an id as bits, zero
    // for refused.
    public delegate* unmanaged<Entity, byte*, uint, ulong> TweenPlay;
    public delegate* unmanaged<ulong, byte> TweenCancel;
    public delegate* unmanaged<ulong, byte> TweenPlaying;
    // Story controls: a name as UTF-8 and its byte length, then the default.
    public delegate* unmanaged<byte*, uint, float, float, float, float> StoryFloat;
    public delegate* unmanaged<byte*, uint, byte, byte> StoryBool;
    // Physics, appended after the story controls. A collider as ColliderData
    // and its byte size; a push as a vector and a ForceMode; velocities with a
    // flag choosing angular; a ray as origin, direction, length, layer mask
    // and whether triggers count, written into a RaycastHit of the size given.
    public delegate* unmanaged<Entity, byte*, uint, byte> AddCollider;
    public delegate* unmanaged<Entity, float, float, float, uint, byte> PhysicsAddForce;
    public delegate* unmanaged<Entity, float, float, float, uint, byte> PhysicsAddTorque;
    public delegate* unmanaged<Entity, byte, float*, byte> PhysicsGetVelocity;
    public delegate* unmanaged<Entity, byte, float, float, float, byte> PhysicsSetVelocity;
    public delegate* unmanaged<
        float, float, float, float, float, float, float, uint, byte, byte*, uint, byte> PhysicsRaycast;
    // Mouse look, appended after physics: this frame's raw motion as two
    // floats, and the cursor lock request.
    public delegate* unmanaged<float*, void> MouseDelta;
    public delegate* unmanaged<byte, void> SetCursorLocked;
    public delegate* unmanaged<byte> CursorLocked;
    // An empty entity standing at a transform, appended after mouse look.
    public delegate* unmanaged<Transform*, Entity> SpawnWithTransform;
    // Audio, appended after SpawnWithTransform; never reordered above it. A
    // SoundRequest and its byte size, which the engine checks; a sound's id
    // as bits, zero for refused; a clip as its 16 bytes.
    public delegate* unmanaged<byte*, uint, ulong> AudioPlay;
    public delegate* unmanaged<ulong, byte> AudioStop;
    public delegate* unmanaged<ulong, float, byte> AudioSetVolume;
    public delegate* unmanaged<ulong, byte> AudioPlaying;
    public delegate* unmanaged<Entity, ulong> AudioPlaySource;
    public delegate* unmanaged<Entity, byte> AudioStopSource;
    public delegate* unmanaged<byte*, byte, ulong> AudioPlayMusic;
    public delegate* unmanaged<void> AudioStopMusic;
    public delegate* unmanaged<float, void> AudioSetMasterVolume;
    public delegate* unmanaged<float, void> AudioSetMusicVolume;
}

public static unsafe class Native
{
    private static OrrinApi _api;

    internal static void Initialize(OrrinApi* api) => _api = *api;

    public static void Log(
        string message,
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) =>
        LogEvent(LogLevel.Info, message, ScriptName(file), line);

    public static void LogWarn(
        string message,
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) =>
        LogEvent(LogLevel.Warning, message, ScriptName(file), line);

    public static void LogError(
        string message,
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) =>
        LogEvent(LogLevel.Error, message, ScriptName(file), line);

    /// One line on the engine's log stream, attributed to `source` — a script's
    /// name, or empty for none — at `line` in it (0 for unknown).
    public static void LogEvent(LogLevel level, string message, string source, int line)
    {
        fixed (byte* sourceUtf8 = NulTerminated(source), messageUtf8 = NulTerminated(message))
            _api.LogEvent((uint)level, sourceUtf8, (uint)line, messageUtf8);
    }

    /// What a `[CallerFilePath]` is called in a log line: the file's name with
    /// no directory and no extension, which for a script is its Behaviour.
    ///
    /// Split by hand rather than with `Path`, which only knows the separator of
    /// the machine it runs on — and the path is the one the *compiler* saw, so a
    /// game built on Windows and run on Linux would keep its whole `C:\…` prefix.
    internal static string ScriptName(string path)
    {
        var start = path.LastIndexOfAny(['/', '\\']) + 1;
        var dot = path.LastIndexOf('.');
        return dot > start ? path[start..dot] : path[start..];
    }

    public static void DebugDrawLine(Vector3 from, Vector3 to, Color color, float duration) =>
        _api.DebugDrawLine(
            from.x, from.y, from.z, to.x, to.y, to.z, color.r, color.g, color.b, color.a, duration);

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

    public static bool MouseButtonPressed(uint button) => _api.MouseButtonPressed(button) != 0;

    public static bool MouseButtonReleased(uint button) => _api.MouseButtonReleased(button) != 0;

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

    /// The id and kind (`AssetKinds`) of the asset at `path`; kind `None` if
    /// there is none.
    internal static (Guid Id, uint Kind) FindAsset(string path)
    {
        Span<byte> id = stackalloc byte[16];
        var pathBytes = NulTerminated(path);
        uint kind;
        fixed (byte* pathPtr = pathBytes)
        fixed (byte* idPtr = id)
            kind = _api.FindAsset(pathPtr, idPtr);
        return kind == AssetKinds.None ? (Guid.Empty, kind) : (new Guid(id, bigEndian: true), kind);
    }

    internal static Entity InstantiatePrefab(Guid prefab)
    {
        Span<byte> id = stackalloc byte[16];
        prefab.TryWriteBytes(id, bigEndian: true, out _);
        fixed (byte* idPtr = id)
            return _api.InstantiatePrefab(idPtr);
    }

    internal static bool LoadScene(Guid scene)
    {
        Span<byte> id = stackalloc byte[16];
        scene.TryWriteBytes(id, bigEndian: true, out _);
        fixed (byte* idPtr = id)
            return _api.LoadScene(idPtr) != 0;
    }

    internal static uint LoadSceneAdditive(Guid scene)
    {
        Span<byte> id = stackalloc byte[16];
        scene.TryWriteBytes(id, bigEndian: true, out _);
        fixed (byte* idPtr = id)
            return _api.LoadSceneAdditive(idPtr);
    }

    internal static bool UnloadScene(uint handle) => _api.UnloadScene(handle) != 0;

    internal static uint ActiveScene() => _api.ActiveScene();

    internal static Transform GetCamera()
    {
        Transform transform = Transform.Identity;
        _api.GetCamera(&transform);
        return transform;
    }

    internal static void SetCamera(Transform transform) => _api.SetCamera(&transform);

    internal static bool AddCamera(Entity entity, CameraComponent camera) =>
        _api.AddCamera(entity, &camera) != 0;

    internal static CameraComponent? GetCameraLens(Entity entity)
    {
        CameraComponent camera = default;
        return _api.GetCameraLens(entity, &camera) != 0 ? camera : null;
    }

    internal static bool SetCameraLens(Entity entity, CameraComponent camera) =>
        _api.SetCameraLens(entity, &camera) != 0;

    internal static Entity GetActiveCamera() => _api.GetActiveCamera();

    internal static bool SetActiveCamera(Entity entity) => _api.SetActiveCamera(entity) != 0;

    internal static void UiPanel(Rect rect, Color color) =>
        _api.UiPanel(rect.x, rect.y, rect.width, rect.height, color.r, color.g, color.b, color.a);

    internal static void UiText(
        ReadOnlySpan<byte> utf8, Guid? font, float size, TextAnchor anchor, Rect rect, Color color)
    {
        Span<byte> id = stackalloc byte[16];
        fixed (byte* text = utf8)
        fixed (byte* idPtr = id)
        {
            byte* fontPtr = null;
            if (font is { } face)
            {
                face.TryWriteBytes(id, bigEndian: true, out _);
                fontPtr = idPtr;
            }
            _api.UiText(text, (uint)utf8.Length, fontPtr, size, (uint)anchor,
                rect.x, rect.y, rect.width, rect.height, color.r, color.g, color.b, color.a);
        }
    }

    internal static void UiImage(Guid texture, Rect rect, Color tint)
    {
        Span<byte> id = stackalloc byte[16];
        texture.TryWriteBytes(id, bigEndian: true, out _);
        fixed (byte* idPtr = id)
            _api.UiImage(idPtr, rect.x, rect.y, rect.width, rect.height,
                tint.r, tint.g, tint.b, tint.a);
    }

    internal static Vector2 UiScreenSize()
    {
        float width = 0, height = 0;
        _api.UiScreenSize(&width, &height);
        return new Vector2(width, height);
    }

    internal static T? GetComponent<T>(Entity entity) where T : unmanaged
    {
        T value = default;
        return _api.GetComponent(entity, ComponentKinds.Blittable<T>(), (byte*)&value, (uint)sizeof(T)) != 0
            ? value
            : null;
    }

    internal static bool SetComponent<T>(Entity entity, T value) where T : unmanaged =>
        _api.SetComponent(entity, ComponentKinds.Blittable<T>(), (byte*)&value, (uint)sizeof(T)) != 0;

    internal static bool AddComponent<T>(Entity entity, T value) where T : unmanaged =>
        _api.AddComponent(entity, ComponentKinds.Blittable<T>(), (byte*)&value, (uint)sizeof(T)) != 0;

    internal static string? GetText(Entity entity)
    {
        const uint kind = (uint)ComponentKind.Text;
        Span<byte> buffer = stackalloc byte[256];
        int length;
        fixed (byte* p = buffer)
            length = _api.GetComponentText(entity, kind, p, buffer.Length);
        if (length < 0)
            return null;
        if (length <= buffer.Length)
            return Encoding.UTF8.GetString(buffer[..length]);
        // Nothing but a script changes a text inside a tick, so one retry at
        // the length the first call reported always fits.
        var bytes = new byte[length];
        fixed (byte* p = bytes)
            _api.GetComponentText(entity, kind, p, bytes.Length);
        return Encoding.UTF8.GetString(bytes);
    }

    internal static bool SetText(Entity entity, string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        fixed (byte* p = bytes)
            return _api.SetComponentText(entity, (uint)ComponentKind.Text, p, (uint)bytes.Length) != 0;
    }

    internal static bool UiPointerOver() => _api.UiPointerOver() != 0;

    internal static ulong TweenPlay(Entity entity, in TweenRequest request)
    {
        fixed (TweenRequest* p = &request)
            return _api.TweenPlay(entity, (byte*)p, (uint)sizeof(TweenRequest));
    }

    internal static bool TweenCancel(ulong id) => _api.TweenCancel(id) != 0;

    internal static Vector2 MouseDelta()
    {
        var xy = stackalloc float[2];
        _api.MouseDelta(xy);
        return new Vector2(xy[0], xy[1]);
    }

    internal static void SetCursorLocked(bool locked) => _api.SetCursorLocked(locked ? (byte)1 : (byte)0);

    internal static bool CursorLocked() => _api.CursorLocked() != 0;

    internal static Entity SpawnWithTransform(Transform transform) => _api.SpawnWithTransform(&transform);

    internal static ulong AudioPlay(in SoundRequest request)
    {
        fixed (SoundRequest* p = &request)
            return _api.AudioPlay((byte*)p, (uint)sizeof(SoundRequest));
    }

    internal static bool AudioStop(ulong id) => _api.AudioStop(id) != 0;

    internal static bool AudioSetVolume(ulong id, float volume) => _api.AudioSetVolume(id, volume) != 0;

    internal static bool AudioPlaying(ulong id) => _api.AudioPlaying(id) != 0;

    internal static ulong AudioPlaySource(Entity entity) => _api.AudioPlaySource(entity);

    internal static bool AudioStopSource(Entity entity) => _api.AudioStopSource(entity) != 0;

    internal static ulong AudioPlayMusic(Guid clip, bool loop)
    {
        var bytes = AssetBytes.Of(clip);
        return _api.AudioPlayMusic((byte*)&bytes, loop ? (byte)1 : (byte)0);
    }

    internal static void AudioStopMusic() => _api.AudioStopMusic();

    internal static void AudioSetMasterVolume(float volume) => _api.AudioSetMasterVolume(volume);

    internal static void AudioSetMusicVolume(float volume) => _api.AudioSetMusicVolume(volume);

    internal static bool AddCollider(Entity entity, in ColliderData collider)
    {
        fixed (ColliderData* p = &collider)
            return _api.AddCollider(entity, (byte*)p, (uint)sizeof(ColliderData)) != 0;
    }

    internal static bool PhysicsAddForce(Entity entity, Vector3 force, ForceMode mode) =>
        _api.PhysicsAddForce(entity, force.x, force.y, force.z, (uint)mode) != 0;

    internal static bool PhysicsAddTorque(Entity entity, Vector3 torque, ForceMode mode) =>
        _api.PhysicsAddTorque(entity, torque.x, torque.y, torque.z, (uint)mode) != 0;

    internal static bool PhysicsGetVelocity(Entity entity, bool angular, out Vector3 velocity)
    {
        var xyz = stackalloc float[3];
        var found = _api.PhysicsGetVelocity(entity, angular ? (byte)1 : (byte)0, xyz) != 0;
        velocity = found ? new Vector3(xyz[0], xyz[1], xyz[2]) : Vector3.zero;
        return found;
    }

    internal static bool PhysicsSetVelocity(Entity entity, bool angular, Vector3 velocity) =>
        _api.PhysicsSetVelocity(entity, angular ? (byte)1 : (byte)0, velocity.x, velocity.y, velocity.z) != 0;

    internal static bool PhysicsRaycast(
        Vector3 origin, Vector3 direction, float maxDistance, uint layers, bool hitTriggers, out RaycastHit hit)
    {
        hit = default;
        fixed (RaycastHit* p = &hit)
            return _api.PhysicsRaycast(
                origin.x, origin.y, origin.z, direction.x, direction.y, direction.z,
                maxDistance, layers, hitTriggers ? (byte)1 : (byte)0, (byte*)p, (uint)sizeof(RaycastHit)) != 0;
    }

    internal static bool TweenPlaying(ulong id) => _api.TweenPlaying(id) != 0;

    internal static float StoryFloat(string name, float @default, float min, float max)
    {
        var bytes = Encoding.UTF8.GetBytes(name);
        fixed (byte* p = bytes)
            return _api.StoryFloat(p, (uint)bytes.Length, @default, min, max);
    }

    internal static bool StoryBool(string name, bool @default)
    {
        var bytes = Encoding.UTF8.GetBytes(name);
        fixed (byte* p = bytes)
            return _api.StoryBool(p, (uint)bytes.Length, @default ? (byte)1 : (byte)0) != 0;
    }

    internal static AssetState GetAssetState(Guid asset)
    {
        Span<byte> id = stackalloc byte[16];
        asset.TryWriteBytes(id, bigEndian: true, out _);
        fixed (byte* idPtr = id)
            return (AssetState)_api.AssetState(idPtr);
    }

    public static float TimeDelta() => _api.TimeDelta();

    public static float TimeTotal() => _api.TimeTotal();

    public static ulong TimeFrameCount() => _api.TimeFrameCount();

    /// The handle for an action name, interning it on first use. Stable for the
    /// life of the process: a binding reload changes what it points at, never
    /// what it is.
    public static uint ActionId(string name)
    {
        var nameBytes = NulTerminated(name);
        fixed (byte* namePtr = nameBytes)
            return _api.ActionId(namePtr);
    }

    public static bool ActionHeld(uint id, uint player) => _api.ActionHeld(id, player) != 0;

    public static bool ActionPressed(uint id, uint player) => _api.ActionPressed(id, player) != 0;

    public static bool ActionReleased(uint id, uint player) => _api.ActionReleased(id, player) != 0;

    public static float AxisValue(uint id, uint player) => _api.AxisValue(id, player);

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
