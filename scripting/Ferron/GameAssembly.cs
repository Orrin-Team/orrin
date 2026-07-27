using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Loader;

namespace Ferron;

/// The collectible load context a project's game assembly lives in.
///
/// `Ferron.dll` itself is never loaded here. It is the ABI surface — Rust holds
/// raw `[UnmanagedCallersOnly]` function pointers into it — and a second copy
/// would be a second, incompatible set of types: the game's `Behaviour` would
/// not be the engine's `Behaviour`, and every call across the boundary would
/// fail on a cast. So `Load` hands back the *running* bindings assembly by
/// reference, which is what guarantees one shared type identity.
///
/// Returning it explicitly rather than returning null and letting resolution
/// fall through to the default context is load-bearing: hostfxr's
/// `load_assembly_and_get_function_pointer` puts `Ferron.dll` in an isolated
/// context of its own, so the default context cannot resolve it by name and the
/// fallback raises `FileNotFoundException` instead.
internal sealed class GameLoadContext : AssemblyLoadContext
{
    static readonly Assembly Bindings = typeof(GameLoadContext).Assembly;

    readonly string _directory;

    internal GameLoadContext(string directory)
        : base(name: "FerronGame", isCollectible: true) => _directory = directory;

    protected override Assembly? Load(AssemblyName name)
    {
        if (name.Name is null)
            return null;
        if (name.Name == FerronAssemblyName)
            return Bindings;

        // Anything else the game ships is loaded here, by copy, so it unloads
        // with the game. A name we can't find next to the DLL falls through to
        // the default context, which is where the framework assemblies live.
        var candidate = Path.Combine(_directory, name.Name + ".dll");
        return File.Exists(candidate) ? LoadFromCopy(this, candidate) : null;
    }

    internal static readonly string FerronAssemblyName = Bindings.GetName().Name!;

    /// Load an assembly from a byte copy rather than by path. Loading by path
    /// memory-maps the file and holds it open for the process lifetime, so the
    /// next `dotnet build` fails to overwrite it on Windows — which would make
    /// the reload loop unusable on the platform. Debug symbols are passed
    /// alongside when present so contained script exceptions keep their file
    /// and line numbers.
    internal static Assembly LoadFromCopy(AssemblyLoadContext context, string path)
    {
        var image = File.ReadAllBytes(path);
        var symbols = Path.ChangeExtension(path, ".pdb");
        using var assembly = new MemoryStream(image);
        if (File.Exists(symbols))
        {
            using var pdb = new MemoryStream(File.ReadAllBytes(symbols));
            return context.LoadFromStream(assembly, pdb);
        }
        return context.LoadFromStream(assembly);
    }
}

/// Engine-driven load/swap of the game assembly.
///
/// A reload is staged rather than in-place: `Load` builds a *pending* context
/// and only `Commit` retires the live one. That ordering is what lets the
/// engine abort on a bad build (a half-written DLL, a renamed dependency)
/// without having already torn down the running behaviours — `Rollback` drops
/// the pending context and the session carries on with the code it had.
///
/// Every entry point here is called from Rust (`ScriptHost`) outside a script
/// dispatch window and returns a status code rather than throwing: a managed
/// exception unwinding into native frames is undefined behaviour.
public static unsafe class GameAssembly
{
    const int Ok = 0;
    const int BadArgument = 1;
    const int NotFound = 2;
    const int LoadFailed = 3;
    const int NothingStaged = 4;

    /// Reported by `Commit` when the retired context was still alive after the
    /// collection attempts below. The new code is live either way; the old
    /// assembly just never unmaps, which costs memory and leaves stale
    /// finalizers running. It is a diagnostic, not a failure.
    const int Leaked = 5;

    /// How many collect/finalize rounds to give a retired context. Unloading is
    /// asynchronous: `Unload()` only marks the context, and the assembly is
    /// released once the GC proves nothing references it. Two rounds is the
    /// documented idiom — the first collection runs finalizers, which can drop
    /// the last references that the second then collects.
    const int UnloadAttempts = 10;

    static GameLoadContext? _current;
    static GameLoadContext? _pending;

    /// The assemblies `Behaviours.ResolveType` searches before falling back to
    /// the default ALC. Empty until the first successful `Commit`.
    internal static IEnumerable<Assembly> CurrentAssemblies =>
        _current?.Assemblies ?? [];

    /// Stage the game assembly at `dllPath` in a fresh collectible context.
    /// Does not disturb the live context; call `Commit` or `Rollback` next.
    [UnmanagedCallersOnly]
    public static int Load(byte* dllPath)
    {
        var path = Marshal.PtrToStringUTF8((nint)dllPath);
        if (string.IsNullOrEmpty(path))
            return BadArgument;
        if (!File.Exists(path))
            return NotFound;

        // A previous stage that was never committed would otherwise leak its
        // context; retiring it here keeps Load idempotent.
        RollbackCore();

        try
        {
            var directory = Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".";
            var context = new GameLoadContext(directory);
            // Touch the assembly's types now rather than at first Create: a
            // corrupt or mistargeted image should fail here, while rolling back
            // is still free.
            GameLoadContext.LoadFromCopy(context, path).GetExportedTypes();
            _pending = context;
            return Ok;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"[Ferron] failed to load game assembly {path}: {e}");
            _pending = null;
            return LoadFailed;
        }
    }

    /// Retire the live context and promote the staged one. The caller must have
    /// already destroyed every behaviour handle from the old assembly — any
    /// surviving `GCHandle` pins its type, and the context never unloads.
    [UnmanagedCallersOnly]
    public static int Commit()
    {
        if (_pending is null)
            return NothingStaged;

        BeginRetire(ref _current);
        _current = _pending;
        _pending = null;

        return FinishRetire() ? Ok : Leaked;
    }

    /// Drop a staged context without touching the live one.
    [UnmanagedCallersOnly]
    public static int Rollback() => RollbackCore();

    // `[UnmanagedCallersOnly]` methods are uncallable from managed code, so the
    // body lives here for `Load` to reuse.
    static int RollbackCore()
    {
        if (_pending is null)
            return NothingStaged;
        BeginRetire(ref _pending);
        return FinishRetire() ? Ok : Leaked;
    }

    /// Unload the live context outright (engine shutdown, or a reload the
    /// caller has decided to abandon).
    [UnmanagedCallersOnly]
    public static int Unload()
    {
        if (_current is null)
            return NothingStaged;
        BeginRetire(ref _current);
        return FinishRetire() ? Ok : Leaked;
    }

    /// Weak handle on the context currently being retired. A field rather than
    /// a local for the reason spelled out on `BeginRetire`.
    static WeakReference? _retiring;

    /// Clear `slot`, unload the context that was in it, and remember it weakly.
    ///
    /// The split from `FinishRetire` is the whole trick. Unloading only *marks*
    /// a context; it is released once the GC can prove nothing references it,
    /// and an ordinary local counts. This assembly ships as a Debug build, where
    /// locals stay live to the end of their scope rather than to last use, so a
    /// `GameLoadContext` held anywhere on the active call chain pins it for as
    /// long as the collection loop runs and every reload reports a leak.
    ///
    /// The only strong reference therefore lives in this frame, which has
    /// already returned by the time anything collects. `NoInlining` is what
    /// keeps that true — inlined, the local would be hoisted into a caller that
    /// is still alive during `FinishRetire`.
    [MethodImpl(MethodImplOptions.NoInlining)]
    static void BeginRetire(ref GameLoadContext? slot)
    {
        var context = slot;
        slot = null;
        if (context is null)
        {
            _retiring = null;
            return;
        }
        context.Unload();
        _retiring = new WeakReference(context);
    }

    /// Collect until the retired context is gone. False means something outside
    /// this class still holds one of its objects — an unfreed `GCHandle`, a
    /// static on a type from another context, a live event subscription or
    /// timer.
    static bool FinishRetire()
    {
        var alive = _retiring;
        _retiring = null;
        if (alive is null)
            return true;

        for (var attempt = 0; attempt < UnloadAttempts && alive.IsAlive; attempt++)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
        return !alive.IsAlive;
    }
}
