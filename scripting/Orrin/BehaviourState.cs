using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Orrin;

/// Behaviour field state carried across a hot reload.
///
/// The engine drives this around the assembly swap: `Capture` once per live
/// Behaviour before its GCHandle is freed, the swap, `Apply` once per re-created
/// Behaviour before OnEnable, then `Discard` for whatever went unclaimed because
/// its type was renamed or deleted between builds.
///
/// **A snapshot outlives the assembly it came from, so nothing reachable from
/// this class may reference a type declared in the game assembly** — not a
/// captured value, not a dictionary key. One boxed game enum or cached `Type` is
/// a live reference into the collectible context, and `GameAssembly.Commit` then
/// reports a leak that never resolves for the rest of the session. That is why
/// the capturable set is closed rather than "whatever reflection hands back",
/// and it is the constraint the component registry (issue #39) inherits when it
/// replaces `TryCapture`/`TryApply` with vtables.
///
/// No entry point may throw — a managed exception unwinding into native frames
/// is undefined behaviour. Diagnostics go to `Console.Error`: with no active
/// world the engine's log sink has nowhere to put them, and Rust logs the reload
/// summary to the editor console itself.
public static unsafe class BehaviourState
{
    /// Snapshots by id. Lives in the default ALC, and by the rule above holds
    /// only default-ALC values, so it can outlive the game assembly.
    static readonly Dictionary<ulong, Dictionary<string, object?>> Snapshots = [];

    /// Ids are opaque to Rust and never reused within a session; 0 is reserved
    /// for "no snapshot".
    static ulong _nextId = 1;

    /// The assembly this file is compiled into. A type from here is safe to
    /// box into a snapshot; a type from anywhere else is not.
    static readonly Assembly Bindings = typeof(Behaviour).Assembly;

    /// Constructor values per type, for the deviation check in `Capture`.
    ///
    /// Keyed by type *name*, never by `Type`. A `Type` object is a live
    /// reference into the load context that declared it, exactly like a boxed
    /// game enum is — caching one here pins the outgoing assembly through
    /// `GameAssembly.Commit`, which runs before `Discard` gets a chance to
    /// clear this.
    ///
    /// Cleared per reload anyway, for an unrelated reason: after a swap the same
    /// name denotes a freshly compiled type whose defaults are the ones the
    /// author just edited.
    ///
    /// A null value caches "defaults unavailable" so a throwing constructor is
    /// only hit once per reload.
    static readonly Dictionary<string, Dictionary<string, object?>?> Defaults = [];

    /// Snapshot `handle`'s reloadable fields. Returns the snapshot id, or 0 if
    /// the handle is dead or the behaviour has nothing capturable.
    [UnmanagedCallersOnly]
    public static ulong Capture(ulong handle)
    {
        try
        {
            if (Behaviours.Resolve(handle) is not { } behaviour)
                return 0;

            var type = behaviour.GetType();
            var defaults = DefaultsOf(type);
            Dictionary<string, object?>? bag = null;

            foreach (var field in CapturableFields(type))
            {
                if (TryCapture(field, behaviour, out var value))
                {
                    // A field still at its constructor value is the source's
                    // default, not state the user established. Restoring it
                    // would paste the old build's default over the newly edited
                    // one — why changing `public float Speed = 6f;` would
                    // otherwise appear to do nothing.
                    if (defaults is not null
                        && defaults.TryGetValue(field.Name, out var original)
                        && Equals(original, value))
                    {
                        continue;
                    }

                    bag ??= [];
                    // TryAdd, not Add: a derived class may shadow a base field
                    // with `new`, and CapturableFields yields derived-first, so
                    // the field the author's code actually reads wins the key.
                    bag.TryAdd(field.Name, value);
                }
                else
                {
                    Console.Error.WriteLine(
                        $"[Orrin] {type.Name}.{field.Name}: {field.FieldType.Name} is not "
                        + "preserved across a reload — mark it [Transient], or use a supported type.");
                }
            }

            if (bag is null)
                return 0;

            var id = _nextId++;
            Snapshots[id] = bag;
            return id;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"[Orrin] capture failed: {e}");
            return 0;
        }
    }

    /// Restore snapshot `id` onto the freshly created behaviour behind
    /// `handle`. The snapshot is consumed either way. Returns 1 if anything was
    /// restored, 0 otherwise.
    [UnmanagedCallersOnly]
    public static byte Apply(ulong handle, ulong id)
    {
        try
        {
            // Consumed before anything can fail, so an id is valid exactly once
            // by construction rather than by which error paths remembered to
            // remove it.
            if (!Snapshots.Remove(id, out var bag))
                return 0;
            if (Behaviours.Resolve(handle) is not { } behaviour)
                return 0;

            var type = behaviour.GetType();
            var restored = 0;

            // Driven by the reloaded type's fields, not the bag's keys: a
            // deleted field is then never looked up, and one newly marked
            // `[Transient]` or `readonly` is absent from CapturableFields, so
            // the reload carrying that edit already honours it.
            foreach (var field in CapturableFields(type))
            {
                if (!bag.TryGetValue(field.Name, out var value))
                    continue;
                if (TryApply(field, behaviour, value))
                {
                    restored++;
                }
                else
                {
                    Console.Error.WriteLine(
                        $"[Orrin] {type.Name}.{field.Name} changed shape across the reload; "
                        + "left at its constructor value.");
                }
            }

            return restored > 0 ? (byte)1 : (byte)0;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"[Orrin] apply failed: {e}");
            return 0;
        }
    }

    /// Drop every snapshot the engine did not claim.
    [UnmanagedCallersOnly]
    public static void Discard()
    {
        Snapshots.Clear();
        // Defaults are per-build: the next reload's types are freshly compiled
        // and their constructor values are whatever the author just wrote.
        Defaults.Clear();
    }

    /// The values `type`'s constructor produces, for the deviation check. Null
    /// when no throwaway instance could be made, in which case `Capture` keeps
    /// every field. Building that instance runs one constructor per type per
    /// reload, which is why `Behaviour` documents constructors as pure.
    static Dictionary<string, object?>? DefaultsOf(Type type)
    {
        var key = type.AssemblyQualifiedName ?? type.FullName ?? type.Name;
        if (Defaults.TryGetValue(key, out var cached))
            return cached;

        Dictionary<string, object?>? values = null;
        try
        {
            if (Activator.CreateInstance(type) is Behaviour fresh)
            {
                values = [];
                foreach (var field in CapturableFields(type))
                {
                    if (TryCapture(field, fresh, out var value))
                        values.TryAdd(field.Name, value);
                }
            }
        }
        catch (Exception e)
        {
            // A constructor that throws only costs the deviation check; the
            // reload still preserves state, just less selectively.
            Console.Error.WriteLine(
                $"[Orrin] could not read {type.Name}'s defaults ({e.GetType().Name}); "
                + "every field will be preserved, so edits to field initializers "
                + "will not take effect until the next full restart.");
        }

        Defaults[key] = values;
        return values;
    }

    /// Instance fields of `type` that a reload should carry, most-derived first
    /// so that a base field shadowed with `new` loses to the derived one when
    /// keyed by name downstream.
    static IEnumerable<FieldInfo> CapturableFields(Type type)
    {
        // DeclaredOnly is required, not an optimization: asked for a leaf type,
        // reflection returns inherited public and protected fields but *not*
        // inherited private ones, so a base class's private state would silently
        // stop surviving reloads.
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public
            | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;

        for (var t = type; t is not null && t != typeof(Behaviour); t = t.BaseType)
            foreach (var field in t.GetFields(flags))
                // `readonly` is excluded deliberately — reflection *could*
                // write it, but the value belongs to the constructor.
                if (!field.IsInitOnly && !IsTransient(field))
                    yield return field;
    }

    /// Whether `field` is opted out of preservation, directly or through the
    /// property it backs.
    static bool IsTransient(FieldInfo field)
    {
        if (field.IsDefined(typeof(TransientAttribute), inherit: true))
            return true;

        // An auto-property's storage is a compiler-generated field carrying none
        // of the author's attributes — `[Transient]` sits on the property, so
        // without this mapping it is silently ignored and the field is preserved
        // against the author's stated intent. `<Name>k__BackingField` is a Roslyn
        // implementation detail with no supported API to replace it; if the
        // mangling changes this stops working and nothing fails loudly.
        var name = field.Name;
        if (!name.StartsWith('<') || !name.EndsWith(">k__BackingField", StringComparison.Ordinal))
            return false;

        // NonPublic is required: a private auto-property still yields a field
        // here, and its property would otherwise not be found.
        var property = field.DeclaringType?.GetProperty(
            name[1..name.IndexOf('>')],
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        return property?.IsDefined(typeof(TransientAttribute), inherit: true) is true;
    }

    /// Convert a live field value into something safe to outlive the assembly.
    /// Returns false when the type is outside the closed set.
    /// Kept in lock-step with [`TryApply`]: it must accept exactly what this
    /// emits. Widen one without the other and fields silently stop restoring.
    static bool TryCapture(FieldInfo field, object instance, out object? value)
    {
        var type = field.FieldType;

        if (type.IsEnum)
        {
            // Never store the boxed enum: declared in the game assembly, it is
            // a live reference into the collectible context. The transient box
            // `GetValue` produces is fine — it is a local.
            var raw = Convert.ChangeType(
                field.GetValue(instance),
                type.GetEnumUnderlyingType(),
                CultureInfo.InvariantCulture);
            value = new EnumValue(type.FullName ?? type.Name, raw!);
            return true;
        }

        // `IsValueType` rather than assembly alone, because `Behaviour` is
        // itself a type from this assembly and a `Behaviour`-typed field holds a
        // game subclass whose reference would pin the context. Sound only while
        // Orrin's value types stay plain data: one that grew a reference field
        // would reopen the hole. `decimal` is not `IsPrimitive`; `string` is a
        // reference type but safe (CoreLib, immutable).
        if (type.IsPrimitive
            || type == typeof(string)
            || type == typeof(decimal)
            || (type.IsValueType && type.Assembly == Bindings))
        {
            value = field.GetValue(instance);
            return true;
        }

        value = null;
        return false;
    }

    /// Write a captured value back onto a field of the reloaded type. Returns
    /// false when the field no longer matches what was captured.
    /// The mirror of [`TryCapture`]; every guard here exists because the field's
    /// type may have changed since capture, which is the premise of a reload
    /// rather than an edge case.
    static bool TryApply(FieldInfo field, object instance, object? value)
    {
        var type = field.FieldType;

        if (value is EnumValue captured)
        {
            // The name check, not just IsEnum: two unrelated enums both store as
            // int, so a field retyped from one to the other would otherwise
            // restore a plausible-looking wrong member. Survives renaming enum
            // members but not reordering them; capturing the member name instead
            // would flip that trade-off and lose flag combinations, so a scheme
            // handling both waits for the component registry (issue #39).
            if (!type.IsEnum || type.FullName != captured.TypeName)
                return false;
            try
            {
                field.SetValue(instance, Enum.ToObject(type, captured.Underlying));
                return true;
            }
            catch (Exception)
            {
                // Per field (the underlying type may have narrowed) so one bad
                // field skips itself instead of aborting the rest.
                return false;
            }
        }

        if (value is null)
        {
            // Not a no-op: `IsInstanceOfType(null)` is always false, so without
            // this branch a captured null falls through to "not restored" and
            // the constructor's value resurrects state the program discarded.
            if (type.IsValueType)
                return false;
            field.SetValue(instance, null);
            return true;
        }

        // No coercion by design: `int Health` -> `float Health` resets to the
        // constructor's value instead of silently reinterpreting.
        if (!type.IsInstanceOfType(value))
            return false;

        field.SetValue(instance, value);
        return true;
    }

    /// An enum captured structurally: the boxed enum itself would be a
    /// game-assembly instance and would pin the load context.
    readonly record struct EnumValue(string TypeName, object Underlying);
}
