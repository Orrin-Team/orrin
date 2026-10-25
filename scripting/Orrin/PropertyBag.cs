using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

using Orrin.Math;

[assembly: InternalsVisibleTo("Orrin.MathTests")]

namespace Orrin;

/// A Behaviour's fields, as one flattened buffer the engine's component
/// registry can read, write, diff and inspect without knowing the type.
///
/// The grammar is `orrin-registry`'s `wire` module, byte for byte, and
/// `Orrin.MathTests` asserts the two agree on a fixed vector. **Tag numbers are
/// permanent** — append, never renumber, the same rule [ComponentKind] follows.
///
/// One buffer carries one whole component, never a field at a time. A registry
/// that read five fields in five calls would pay five managed transitions and,
/// worse, could observe a Behaviour halfway through its own `OnUpdate`.
///
/// The visible set is closed to the types that have a `Value` variant on the
/// Rust side: bool, int, uint, float, string, Vector3, Quaternion, enums, and
/// `AssetRef<T>`.
/// Deliberately absent for now:
/// - `Entity`, because a slot handle means nothing in a saved file. Persisting
///   one needs the engine's slot-to-EntityId translation, which lives where the
///   world is in scope — the same reason `orrin.parent` is resolved a level
///   above the registry rather than in a conversion.
/// - `Vector2`, `Vector4`, `Color`, arrays and lists, which need a `Value`
///   variant or a resize rule that no component has yet asked for. A variant
///   there lands in the scene format forever, so it waits for a demand.
///
/// A field outside that set is left alone: not saved, not inspectable, and
/// reported by the analyzer so it is a decision rather than a surprise. It is
/// still carried across a hot reload if [BehaviourState] can carry it.
///
/// No entry point may throw — a managed exception unwinding into native frames
/// is undefined behaviour.
public static unsafe class PropertyBag
{
    /// Lock-step with `wire::tag` in orrin-registry.
    static class Tag
    {
        public const byte Bool = 0;
        public const byte I32 = 1;
        public const byte U32 = 2;
        public const byte F32 = 3;
        public const byte String = 4;
        public const byte Vec3 = 5;
        public const byte Quat = 6;
        public const byte Entity = 7;
        public const byte Struct = 8;
        public const byte Enum = 9;
        public const byte List = 10;
        public const byte Asset = 11;
    }

    /// Negative returns from every entry point below. A non-negative result is
    /// the number of bytes the value needs; it was written only if that fits in
    /// the capacity the caller offered, so a caller that guessed too small
    /// retries with the number it got back rather than losing the value.
    const int NoSuchHandle = -1;
    const int Failed = -2;

    /// Encode the behaviour behind `handle` into `buffer`.
    [UnmanagedCallersOnly]
    public static int Read(ulong handle, byte* buffer, int capacity)
    {
        try
        {
            if (Behaviours.Resolve(handle) is not { } behaviour)
                return NoSuchHandle;
            var writer = new Writer();
            WriteBag(ref writer, behaviour);
            return writer.CopyTo(buffer, capacity);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"[Orrin] reading a property bag threw {e.GetType().Name}: {e.Message}");
            return Failed;
        }
    }

    /// Overwrite the behaviour's visible fields from `data`.
    ///
    /// Fields the buffer does not mention keep their current values, which is
    /// what makes a scene saved before a field existed load without erasing it.
    [UnmanagedCallersOnly]
    public static int Write(ulong handle, byte* data, int length)
    {
        try
        {
            if (Behaviours.Resolve(handle) is not { } behaviour)
                return NoSuchHandle;
            var reader = new Reader(new ReadOnlySpan<byte>(data, length));
            ReadBag(ref reader, behaviour);
            return 0;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"[Orrin] writing a property bag threw {e.GetType().Name}: {e.Message}");
            return Failed;
        }
    }

    /// The bag a freshly constructed instance of `typeName` produces — the
    /// component's default value, and the layout every other buffer for this
    /// type agrees with.
    [UnmanagedCallersOnly]
    public static int Default(byte* typeName, byte* buffer, int capacity)
    {
        try
        {
            if (Marshal.PtrToStringUTF8((nint)typeName) is not { } name)
                return Failed;
            if (Behaviours.ResolveType(name) is not { } type)
                return NoSuchHandle;
            if (Activator.CreateInstance(type) is not Behaviour fresh)
                return Failed;

            var writer = new Writer();
            WriteBag(ref writer, fresh);
            return writer.CopyTo(buffer, capacity);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"[Orrin] reading defaults threw {e.GetType().Name}: {e.Message}");
            return Failed;
        }
    }

    /// Every registry-visible Behaviour in the currently loaded game assemblies,
    /// as a `List` of `{ id, type, name }` structs.
    ///
    /// This is the game assembly's half of `register_components`: the engine
    /// calls it after each load and registers what comes back, then drops those
    /// entries before the next swap. Enumerating types here rather than having
    /// each assembly export a native symbol is the same explicit-registration
    /// rule from the other side — nothing is discovered by a linker section,
    /// which would not survive the collectible load context anyway.
    [UnmanagedCallersOnly]
    public static int Components(byte* buffer, int capacity)
    {
        try
        {
            var found = new List<(string Id, string Type, string Name)>();
            foreach (var assembly in GameAssembly.CurrentAssemblies)
            {
                foreach (var type in assembly.GetTypes())
                {
                    if (type.IsAbstract || !type.IsSubclassOf(typeof(Behaviour)))
                        continue;
                    if (type.GetCustomAttribute<ComponentAttribute>() is not { } component)
                        continue;
                    var qualified = $"{type.FullName}, {assembly.GetName().Name}";
                    found.Add((component.Id, qualified, component.Name ?? type.Name));
                }
            }

            // Sorted by id so that the engine's registration order — and so the
            // inspector's section order — does not depend on the order
            // reflection happens to hand back types in.
            found.Sort((a, b) => string.CompareOrdinal(a.Id, b.Id));

            var writer = new Writer();
            writer.Byte(Tag.List);
            writer.U32((uint)found.Count);
            foreach (var (id, type, name) in found)
            {
                writer.Byte(Tag.Struct);
                writer.U32(3);
                writer.Field("id", Tag.String);
                writer.Str(id);
                writer.Field("type", Tag.String);
                writer.Str(type);
                writer.Field("name", Tag.String);
                writer.Str(name);
            }
            return writer.CopyTo(buffer, capacity);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"[Orrin] listing components threw {e.GetType().Name}: {e.Message}");
            return Failed;
        }
    }

    /// The encode and decode halves without the FFI around them, for
    /// `Orrin.MathTests` — which is where the claim that this file and
    /// `orrin-registry`'s `wire` module implement one grammar is actually
    /// checked, the way `sizeof(Transform) == 40` is.
    internal static byte[] Encode(Behaviour behaviour)
    {
        var writer = new Writer();
        WriteBag(ref writer, behaviour);
        return writer.ToArray();
    }

    internal static void Decode(Behaviour behaviour, ReadOnlySpan<byte> bytes)
    {
        var reader = new Reader(bytes);
        ReadBag(ref reader, behaviour);
    }

    static void WriteBag(ref Writer writer, Behaviour behaviour)
    {
        var type = behaviour.GetType();
        var fields = VisibleFields(type);
        writer.Byte(Tag.Struct);
        writer.U32((uint)fields.Count);
        foreach (var field in fields)
        {
            writer.Str(field.Name);
            WriteValue(ref writer, field.FieldType, field.GetValue(behaviour));
        }
    }

    static void WriteValue(ref Writer writer, Type type, object? value)
    {
        if (type.IsEnum)
        {
            // Written as its member name, not its underlying integer. Reordering
            // an enum's members is a routine edit and would otherwise silently
            // reinterpret every saved value; renaming one is rarer and fails
            // loudly instead, which is the right way round. `ToString` renders a
            // flags combination as "A, B", which `Enum.Parse` reads back.
            writer.Byte(Tag.Enum);
            writer.Str(value?.ToString() ?? "");
            writer.U32(0);
            return;
        }

        switch (value)
        {
            case bool v:
                writer.Byte(Tag.Bool);
                writer.Byte(v ? (byte)1 : (byte)0);
                break;
            case int v:
                writer.Byte(Tag.I32);
                writer.U32(unchecked((uint)v));
                break;
            case uint v:
                writer.Byte(Tag.U32);
                writer.U32(v);
                break;
            case float v:
                writer.Byte(Tag.F32);
                writer.F32(v);
                break;
            case string v:
                writer.Byte(Tag.String);
                writer.Str(v);
                break;
            case Vector3 v:
                writer.Byte(Tag.Vec3);
                writer.F32(v.x);
                writer.F32(v.y);
                writer.F32(v.z);
                break;
            case Quaternion v:
                writer.Byte(Tag.Quat);
                writer.F32(v.x);
                writer.F32(v.y);
                writer.F32(v.z);
                writer.F32(v.w);
                break;
            case IAssetRef v:
                writer.Byte(Tag.Asset);
                writer.Guid(v.Id);
                break;
            // A null string is the one visible-typed value that can be null.
            // Written as empty rather than skipped: the field count is already
            // in the buffer, so a hole would desynchronize the reader.
            case null when type == typeof(string):
                writer.Byte(Tag.String);
                writer.Str("");
                break;
            default:
                throw new InvalidOperationException(
                    $"{type.Name} is not a registry-visible field type");
        }
    }

    static void ReadBag(ref Reader reader, Behaviour behaviour)
    {
        var type = behaviour.GetType();
        if (reader.Byte() != Tag.Struct)
            throw new InvalidOperationException("a property bag must be a struct");

        var count = reader.U32();
        // Indexed by name, because the buffer's order is the order the *writer*
        // saw and a field may have been added, removed or moved since.
        var byName = new Dictionary<string, FieldInfo>(StringComparer.Ordinal);
        foreach (var field in VisibleFields(type))
            byName.TryAdd(field.Name, field);

        for (var i = 0; i < count; i++)
        {
            var name = reader.Str();
            // Every value is read whether or not it is wanted: skipping by
            // seeking would need a length prefix the format does not carry, and
            // a field the type no longer has must still be stepped over.
            var value = ReadValue(ref reader, byName.GetValueOrDefault(name)?.FieldType);
            if (value is not null && byName.TryGetValue(name, out var field))
                field.SetValue(behaviour, value);
        }
    }

    /// Returns null when the value was read but cannot be assigned — an unknown
    /// field, or one whose type changed since the buffer was written. Leaving
    /// the field at its constructor value is the same choice [BehaviourState]
    /// makes, and for the same reason: no silent coercion.
    static object? ReadValue(ref Reader reader, Type? target)
    {
        var tag = reader.Byte();
        switch (tag)
        {
            case Tag.Bool:
            {
                var v = reader.Byte() != 0;
                return target == typeof(bool) ? v : null;
            }
            case Tag.I32:
            {
                var v = unchecked((int)reader.U32());
                return target == typeof(int) ? v : null;
            }
            case Tag.U32:
            {
                var v = reader.U32();
                return target == typeof(uint) ? v : null;
            }
            case Tag.F32:
            {
                var v = reader.F32();
                return target == typeof(float) ? v : null;
            }
            case Tag.String:
            {
                var v = reader.Str();
                return target == typeof(string) ? v : null;
            }
            case Tag.Vec3:
            {
                var v = new Vector3(reader.F32(), reader.F32(), reader.F32());
                return target == typeof(Vector3) ? v : null;
            }
            case Tag.Quat:
            {
                var v = new Quaternion(reader.F32(), reader.F32(), reader.F32(), reader.F32());
                return target == typeof(Quaternion) ? v : null;
            }
            case Tag.Entity:
            {
                reader.Skip(16);
                return null;
            }
            case Tag.Asset:
            {
                var id = reader.Guid();
                // Any `AssetRef<T>` takes any id: the kind is the field's
                // declaration, and the bytes carry none.
                return target is { IsGenericType: true } && target.GetGenericTypeDefinition() == typeof(AssetRef<>)
                    ? Activator.CreateInstance(target, id)
                    : null;
            }
            case Tag.Enum:
            {
                var member = reader.Str();
                var fields = reader.U32();
                for (var i = 0; i < fields; i++)
                {
                    reader.Str();
                    ReadValue(ref reader, null);
                }
                if (target is null || !target.IsEnum || member.Length == 0)
                    return null;
                // A member this build no longer has leaves the field alone
                // rather than landing on whatever happens to share its number.
                return Enum.TryParse(target, member, ignoreCase: false, out var parsed)
                    ? parsed
                    : null;
            }
            case Tag.Struct:
            {
                var fields = reader.U32();
                for (var i = 0; i < fields; i++)
                {
                    reader.Str();
                    ReadValue(ref reader, null);
                }
                return null;
            }
            case Tag.List:
            {
                var items = reader.U32();
                for (var i = 0; i < items; i++)
                    ReadValue(ref reader, null);
                return null;
            }
            default:
                throw new InvalidOperationException($"value tag {tag} is not one this build knows");
        }
    }

    /// The fields the registry sees: what a hot reload would carry, narrowed to
    /// the types that have a `Value` variant.
    ///
    /// Sharing [BehaviourState]'s enumeration is the point — `readonly`,
    /// `[Transient]` and the auto-property backing-field mapping are one rule
    /// with two consumers, and a field that reloads but does not save (or the
    /// reverse) would be a genuinely baffling thing to debug.
    internal static List<FieldInfo> VisibleFields(Type type)
    {
        var fields = new List<FieldInfo>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var field in BehaviourState.CapturableFields(type))
        {
            // Derived-first, so a base field shadowed with `new` loses to the
            // one the author's code actually reads — matching the `TryAdd` in
            // BehaviourState.Capture.
            if (IsVisible(field.FieldType) && seen.Add(field.Name))
                fields.Add(field);
        }
        // Left in the order `CapturableFields` yields, which is declaration
        // order within each level of the hierarchy — the order an inspector
        // should draw. The Rust side re-sorts when it writes the scene file,
        // where determinism matters and drawing order does not.
        return fields;
    }

    static bool IsVisible(Type type) =>
        type.IsEnum
        || type == typeof(bool)
        || type == typeof(int)
        || type == typeof(uint)
        || type == typeof(float)
        || type == typeof(string)
        || type == typeof(Vector3)
        || type == typeof(Quaternion)
        || (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(AssetRef<>));

    /// Grows by doubling and copies out once, so an entry point never has to
    /// guess a size and the caller never sees a partial buffer.
    struct Writer
    {
        byte[] _bytes = new byte[256];
        int _len = 0;

        public Writer() { }

        void Need(int extra)
        {
            if (_len + extra <= _bytes.Length)
                return;
            var size = _bytes.Length;
            while (size < _len + extra)
                size *= 2;
            Array.Resize(ref _bytes, size);
        }

        public void Byte(byte value)
        {
            Need(1);
            _bytes[_len++] = value;
        }

        /// Little-endian by hand rather than through `BitConverter`, which
        /// writes native order — the wire format is fixed, so a big-endian host
        /// must not be able to change it.
        public void U32(uint value)
        {
            Need(4);
            _bytes[_len++] = (byte)value;
            _bytes[_len++] = (byte)(value >> 8);
            _bytes[_len++] = (byte)(value >> 16);
            _bytes[_len++] = (byte)(value >> 24);
        }

        /// Bits, not a decimal rendering: the boundary must carry a NaN and a
        /// negative zero unchanged, because the Rust side's diff distinguishes
        /// them and would otherwise report an edit nobody made.
        public void F32(float value) => U32(unchecked((uint)BitConverter.SingleToInt32Bits(value)));

        public void Str(string value)
        {
            var bytes = Encoding.UTF8.GetBytes(value);
            U32((uint)bytes.Length);
            Need(bytes.Length);
            bytes.CopyTo(_bytes.AsSpan(_len));
            _len += bytes.Length;
        }

        /// The sixteen bytes in canonical UUID order, which `Guid`'s default
        /// byte layout is not.
        public void Guid(Guid value)
        {
            Need(16);
            value.TryWriteBytes(_bytes.AsSpan(_len, 16), bigEndian: true, out _);
            _len += 16;
        }

        /// A field name followed by its value's tag, for the hand-built buffers
        /// in [Components].
        public void Field(string name, byte tag)
        {
            Str(name);
            Byte(tag);
        }

        public readonly byte[] ToArray() => _bytes[.._len];

        /// Copies into `buffer` only if the whole value fits; returns the size
        /// either way, so the caller can size a second attempt exactly.
        public readonly int CopyTo(byte* buffer, int capacity)
        {
            if (_len <= capacity && buffer is not null)
                _bytes.AsSpan(0, _len).CopyTo(new Span<byte>(buffer, capacity));
            return _len;
        }
    }

    ref struct Reader(ReadOnlySpan<byte> bytes)
    {
        readonly ReadOnlySpan<byte> _bytes = bytes;
        int _at = 0;

        ReadOnlySpan<byte> Take(int n)
        {
            if (n < 0 || _at + n > _bytes.Length)
                throw new InvalidOperationException("the property bag ended mid-value");
            var slice = _bytes.Slice(_at, n);
            _at += n;
            return slice;
        }

        public void Skip(int n) => Take(n);

        public byte Byte() => Take(1)[0];

        public uint U32()
        {
            var s = Take(4);
            return (uint)(s[0] | (s[1] << 8) | (s[2] << 16) | (s[3] << 24));
        }

        public float F32() => BitConverter.Int32BitsToSingle(unchecked((int)U32()));

        public string Str() => Encoding.UTF8.GetString(Take(checked((int)U32())));

        public Guid Guid() => new(Take(16), bigEndian: true);
    }
}
