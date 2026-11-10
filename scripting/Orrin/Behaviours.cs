using System.Runtime.InteropServices;

namespace Orrin;

/// Lifecycle of a Behaviour, in dispatch order:
///
///   1. ctor + Entity assignment   — `Create`, when the engine attaches the script
///   2. OnEnable                   — first activation, and every re-activation
///   3. OnStart                    — once, on the first tick the behaviour is active
///   4. OnUpdate                   — every tick while active
///   5. OnDisable                  — every deactivation; also fired by `Destroy`
///                                   if the behaviour is still active
///   6. OnDestroy                  — once, just before the GCHandle is freed
///   7. GCHandle.Free              — last step of `Destroy`; the object is
///                                   unreachable from Rust after this
///
/// Steps 2–5 can cycle (enable/disable). `Destroy` is the only path that frees
/// the handle, so OnDestroy always precedes the free by construction.
public static unsafe class Behaviours
{
    // The lifecycle entry points return a fault byte so the engine can disable a
    // misbehaving script: 0 = the hook ran cleanly, 1 = it threw (already logged;
    // Rust flips the ScriptComponent's `faulted` flag and stops dispatching to
    // it). Keeping the exception on this side of the boundary is mandatory — a
    // managed exception unwinding into native CoreCLR frames is undefined
    // behaviour, i.e. exactly the engine crash this is preventing.
    const byte Ok = 0;
    const byte Faulted = 1;
    // Only from `Pointer`: the behaviour has no handler for the event, so the
    // engine offers it to the parent next. Rust's `orrin_script::Delivery`.
    const byte Unhandled = 2;

    // The handle is a GCHandle passed across the ABI as a fixed-width ulong, to
    // match Rust's `u64` fn-pointer signatures exactly. Using `nint` here would
    // silently disagree with Rust by 4 bytes on any 32-bit target; ulong is the
    // same 8 bytes on every platform. It's converted back to `nint` only at the
    // GCHandle calls, which take a pointer-sized value.
    internal static Behaviour? Resolve(ulong handle) =>
        handle != 0 && GCHandle.FromIntPtr((nint)handle).Target is Behaviour behaviour ? behaviour : null;

    /// Every live Behaviour's handle, by the entity it is on.
    ///
    /// Handles rather than objects: this assembly is never unloaded, so a
    /// `Behaviour` held here would pin the game's load context through every
    /// reload, while a handle is only a number that `Destroy` takes out of here
    /// before it frees it. A list per entity rather than one handle, so that an
    /// entity carrying several Behaviours needs no new shape.
    static readonly Dictionary<Entity, List<ulong>> ByEntity = [];

    static void Index(Entity entity, ulong handle)
    {
        if (!ByEntity.TryGetValue(entity, out var handles))
            ByEntity[entity] = handles = [];
        handles.Add(handle);
    }

    static void Unindex(Entity entity, ulong handle)
    {
        if (ByEntity.TryGetValue(entity, out var handles) && handles.Remove(handle) && handles.Count == 0)
            ByEntity.Remove(entity);
    }

    internal static T? Find<T>(Entity entity) where T : Behaviour
    {
        if (!ByEntity.TryGetValue(entity, out var handles))
            return null;
        foreach (var handle in handles)
        {
            if (Resolve(handle) is T found)
                return found;
        }
        return null;
    }

    internal static T? FindFirst<T>() where T : Behaviour
    {
        foreach (var handles in ByEntity.Values)
        {
            foreach (var handle in handles)
            {
                if (Resolve(handle) is T found)
                    return found;
            }
        }
        return null;
    }

    internal static T[] FindAll<T>() where T : Behaviour
    {
        var found = new List<T>();
        foreach (var handles in ByEntity.Values)
        {
            foreach (var handle in handles)
            {
                if (Resolve(handle) is T behaviour)
                    found.Add(behaviour);
            }
        }
        return [.. found];
    }

    /// Log a contained script exception with the script's type, the hook that
    /// threw, and the exception (type + message + stack trace), then report the
    /// fault so the caller stops dispatching to this behaviour.
    ///
    /// An error, attributed to the Behaviour rather than to this file: it is
    /// the script's line in the console, and the one an export's log file is
    /// there to keep.
    static byte Fault(Behaviour behaviour, string method, Exception e)
    {
        Native.LogEvent(
            LogLevel.Error,
            $"[script] {behaviour.GetType().Name}.{method} threw {e.GetType().Name}: {e.Message} "
            + $"— disabling this script until reload.\n{e.StackTrace}",
            behaviour.GetType().Name,
            0);
        return Faulted;
    }

    [UnmanagedCallersOnly]
    public static ulong Create(Entity entity, byte* typeName)
    {
        var name = Marshal.PtrToStringUTF8((nint)typeName);
        if (name is null)
            return 0;

        try
        {
            var type = ResolveType(name);
            if (type is null || Activator.CreateInstance(type) is not Behaviour behaviour)
                return 0;

            behaviour.Entity = entity;
            var handle = (ulong)GCHandle.ToIntPtr(GCHandle.Alloc(behaviour));
            Index(entity, handle);
            return handle;
        }
        catch (Exception e)
        {
            // A throwing user constructor must not escape to native code; 0 is
            // the existing "creation failed, don't attach" contract.
            Native.LogError($"[script] exception during create of {name}: {e}");
            return 0;
        }
    }

    [UnmanagedCallersOnly]
    public static byte Start(ulong handle)
    {
        if (Resolve(handle) is not { } behaviour)
            return Ok;
        try
        {
            behaviour.OnStart();
            return Ok;
        }
        catch (Exception e)
        {
            return Fault(behaviour, nameof(Behaviour.OnStart), e);
        }
    }

    [UnmanagedCallersOnly]
    public static byte Update(ulong handle, float deltaTime)
    {
        if (Resolve(handle) is not { } behaviour)
            return Ok;
        if (behaviour.TickTimers(deltaTime) is { } timer)
            return Fault(behaviour, timer.Hook, timer.Error);
        try
        {
            behaviour.OnUpdate(deltaTime);
            return Ok;
        }
        catch (Exception e)
        {
            return Fault(behaviour, nameof(Behaviour.OnUpdate), e);
        }
    }

    [UnmanagedCallersOnly]
    public static byte Enable(ulong handle)
    {
        if (Resolve(handle) is not { } behaviour)
            return Ok;
        try
        {
            if (!behaviour.Active)
            {
                behaviour.Active = true;
                behaviour.OnEnable();
            }
            return Ok;
        }
        catch (Exception e)
        {
            return Fault(behaviour, nameof(Behaviour.OnEnable), e);
        }
    }

    [UnmanagedCallersOnly]
    public static byte Disable(ulong handle)
    {
        if (Resolve(handle) is not { } behaviour)
            return Ok;
        try
        {
            if (behaviour.Active)
            {
                behaviour.Active = false;
                behaviour.OnDisable();
            }
            return Ok;
        }
        catch (Exception e)
        {
            return Fault(behaviour, nameof(Behaviour.OnDisable), e);
        }
    }

    [UnmanagedCallersOnly]
    public static byte CollisionEnter(ulong handle, Collision* collision)
    {
        if (Resolve(handle) is not { } behaviour)
            return Ok;
        try
        {
            behaviour.OnCollisionEnter(*collision);
            return Ok;
        }
        catch (Exception e)
        {
            return Fault(behaviour, nameof(Behaviour.OnCollisionEnter), e);
        }
    }

    [UnmanagedCallersOnly]
    public static byte CollisionExit(ulong handle, Collision* collision)
    {
        if (Resolve(handle) is not { } behaviour)
            return Ok;
        try
        {
            behaviour.OnCollisionExit(*collision);
            return Ok;
        }
        catch (Exception e)
        {
            return Fault(behaviour, nameof(Behaviour.OnCollisionExit), e);
        }
    }

    [UnmanagedCallersOnly]
    public static byte TriggerEnter(ulong handle, Collision* collision)
    {
        if (Resolve(handle) is not { } behaviour)
            return Ok;
        try
        {
            behaviour.OnTriggerEnter(*collision);
            return Ok;
        }
        catch (Exception e)
        {
            return Fault(behaviour, nameof(Behaviour.OnTriggerEnter), e);
        }
    }

    [UnmanagedCallersOnly]
    public static byte TriggerExit(ulong handle, Collision* collision)
    {
        if (Resolve(handle) is not { } behaviour)
            return Ok;
        try
        {
            behaviour.OnTriggerExit(*collision);
            return Ok;
        }
        catch (Exception e)
        {
            return Fault(behaviour, nameof(Behaviour.OnTriggerExit), e);
        }
    }

    /// Offer a pointer event, a <see cref="PointerEventKind"/>, to the
    /// behaviour's handler for it. One with no handler answers Unhandled, and a
    /// handler that throws is contained and reported as a fault, which also
    /// stops the event going further up.
    [UnmanagedCallersOnly]
    public static byte Pointer(ulong handle, uint kind, PointerEvent* pointer)
    {
        if (Resolve(handle) is not { } behaviour)
            return Unhandled;
        var which = (PointerEventKind)kind;
        try
        {
            return PointerEvents.Dispatch(behaviour, which, *pointer) ? Ok : Unhandled;
        }
        catch (Exception e)
        {
            return Fault(behaviour, PointerEvents.MethodName(which), e);
        }
    }

    /// Tears the behaviour down and frees its GCHandle. This is the single
    /// managed release point: Rust's `ScriptComponent::drop` lands here.
    [UnmanagedCallersOnly]
    public static void Destroy(ulong handle)
    {
        if (handle != 0 && GCHandle.FromIntPtr((nint)handle).Target is Behaviour behaviour)
        {
            try
            {
                if (behaviour.Active)
                {
                    behaviour.Active = false;
                    behaviour.OnDisable();
                }
                behaviour.OnDestroy();
            }
            catch (Exception e)
            {
                Native.LogEvent(
                    LogLevel.Error,
                    $"[script] exception during destroy: {e}",
                    behaviour.GetType().Name,
                    0);
            }
            finally
            {
                Unindex(behaviour.Entity, handle);
                GCHandle.FromIntPtr((nint)handle).Free();
            }
        }
    }

    /// Resolve an assembly-qualified Behaviour name (`MyGame.Player, MyGame`).
    ///
    /// The game assembly is searched first and by hand, because `Type.GetType`
    /// resolves through the *default* ALC: handed `MyGame.Player, MyGame` it
    /// would load a second copy of the game assembly into the uncollectible
    /// context, which pins those types for the life of the process and quietly
    /// defeats hot reload. Only the fallback, for engine-owned types such as
    /// the lifecycle probes, may go through the default ALC.
    internal static Type? ResolveType(string name)
    {
        // Behaviours are instantiated with a parameterless constructor, so they
        // are never generic and the first comma always separates the type from
        // its assembly.
        var comma = name.IndexOf(',');
        var typeName = (comma < 0 ? name : name[..comma]).Trim();
        var assemblyName = comma < 0 ? null : name[(comma + 1)..].Trim();

        foreach (var assembly in GameAssembly.CurrentAssemblies)
        {
            if (assemblyName is { Length: > 0 } && assembly.GetName().Name != assemblyName)
                continue;
            if (assembly.GetType(typeName) is { } fromGame)
                return fromGame;
        }

        if (Type.GetType(name) is { } qualified)
            return qualified;

        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.GetType(typeName) is { } loaded)
                return loaded;
        }
        return null;
    }
}
