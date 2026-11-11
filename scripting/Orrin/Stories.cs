using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace Orrin;

/// <summary>
/// Marks a static method as a storybook story: a piece of UI built on its own,
/// in a world of its own, to be looked at and played with in the editor's
/// Storybook. The method takes a <see cref="StoryContext"/>, or nothing.
/// </summary>
/// <remarks>
/// A story builds its elements under <see cref="StoryContext.Canvas"/> the way
/// game code builds them anywhere, so it can start tweens and attach
/// behaviours that then run, take the pointer and tick while the story is
/// open. Slashes in the name group stories in the Storybook's list; with no
/// name, a story is listed as <c>Type/Method</c>. The engine builds it again
/// whenever a control changes or the scripts reload.
/// </remarks>
[AttributeUsage(AttributeTargets.Method, Inherited = false)]
public sealed class StoryAttribute : Attribute
{
    /// <summary>
    /// What the Storybook lists the story as, slashes grouping it; null lists it as
    /// <c>Type/Method</c>.
    /// </summary>
    public string? Name { get; }

    /// <summary>
    /// Mark the method as a story listed as <paramref name="name"/>, or as
    /// <c>Type/Method</c> without one.
    /// </summary>
    public StoryAttribute(string? name = null) => Name = name;
}

/// <summary>What a story is built onto, and the controls it is built with.</summary>
public sealed class StoryContext
{
    /// <summary>
    /// The story's root canvas, already in the world. Parent the story's
    /// elements to it, and change its <see cref="Orrin.Canvas"/> to change how
    /// the story scales.
    /// </summary>
    public Entity Canvas { get; }

    internal StoryContext(Entity canvas) => Canvas = canvas;

    /// <summary>
    /// A slider in the Storybook, from <paramref name="min"/> to
    /// <paramref name="max"/>: its value, which is <paramref name="default"/>
    /// until someone moves it.
    /// </summary>
    public float Float(string name, float @default, float min = 0f, float max = 1f) =>
        Native.StoryFloat(name, @default, min, max);

    /// <summary>A checkbox in the Storybook: its value, <paramref name="default"/> until ticked.</summary>
    public bool Bool(string name, bool @default = false) => Native.StoryBool(name, @default);
}

/// The engine's way in: which stories the game declares, and building one.
/// Nothing here caches a type or a method, so nothing holds the game's load
/// context past a reload.
public static unsafe class Stories
{
    const byte Built = 0;
    const byte Faulted = 1;
    const byte Missing = 2;

    /// Every story, by name, in name order. A name declared twice is listed
    /// once, for the first method that claimed it.
    internal static List<(string Name, MethodInfo Method)> Discover()
    {
        var stories = new List<(string, MethodInfo)>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var assembly in GameAssembly.CurrentAssemblies)
        {
            Type[] types;
            try
            {
                types = assembly.GetTypes();
            }
            catch (ReflectionTypeLoadException e)
            {
                types = e.Types.Where(type => type is not null).ToArray()!;
            }
            foreach (var type in types)
            foreach (var method in type.GetMethods(BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic))
            {
                if (method.GetCustomAttribute<StoryAttribute>() is not { } story)
                    continue;
                var parameters = method.GetParameters();
                var shaped = method.ReturnType == typeof(void)
                    && !method.ContainsGenericParameters
                    && (parameters.Length == 0
                        || (parameters.Length == 1 && parameters[0].ParameterType == typeof(StoryContext)));
                var name = story.Name is { Length: > 0 } given ? given : $"{type.Name}/{method.Name}";
                if (!shaped)
                {
                    Native.LogWarn($"[story] {type.FullName}.{method.Name} is not a static void method taking a StoryContext or nothing, so it is not a story");
                    continue;
                }
                if (!seen.Add(name))
                {
                    Native.LogWarn($"[story] \"{name}\" is declared twice; {type.FullName}.{method.Name} is ignored");
                    continue;
                }
                stories.Add((name, method));
            }
        }
        stories.Sort((a, b) => string.CompareOrdinal(a.Item1, b.Item1));
        return stories;
    }

    /// Every story's name, one per line, through the `get_tag` protocol.
    [UnmanagedCallersOnly]
    public static int Names(byte* buffer, int capacity)
    {
        try
        {
            var bytes = Encoding.UTF8.GetBytes(string.Join('\n', Discover().Select(story => story.Name)));
            if (bytes.Length <= capacity)
                bytes.CopyTo(new Span<byte>(buffer, capacity));
            return bytes.Length;
        }
        catch (Exception e)
        {
            Native.LogError($"[story] listing the stories threw: {e}");
            return -1;
        }
    }

    [UnmanagedCallersOnly]
    public static byte Build(byte* name, uint length, Entity canvas)
    {
        var wanted = Encoding.UTF8.GetString(name, (int)length);
        MethodInfo? method;
        try
        {
            method = Discover().FirstOrDefault(story => story.Name == wanted).Method;
        }
        catch (Exception e)
        {
            Native.LogError($"[story] finding \"{wanted}\" threw: {e}");
            return Faulted;
        }
        if (method is null)
            return Missing;
        try
        {
            var arguments = method.GetParameters().Length == 0 ? null : new object[] { new StoryContext(canvas) };
            method.Invoke(null, arguments);
            return Built;
        }
        catch (Exception e)
        {
            var thrown = e is TargetInvocationException { InnerException: { } inner } ? inner : e;
            Native.LogEvent(
                LogLevel.Error,
                $"[story] \"{wanted}\" threw {thrown.GetType().Name}: {thrown.Message}\n{thrown.StackTrace}",
                method.DeclaringType?.Name ?? "",
                0);
            return Faulted;
        }
    }
}
