using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace Orrin.Testing;

/// The engine's way in to a project's tests: which there are, and running one
/// a turn at a time. The engine owns the frame loop; an iterator test is
/// advanced to its next `yield` by `Begin` and then by one `Resume` per frame,
/// each inside a dispatch window, so test code never runs a frame itself.
///
/// The running iterator is a game-assembly object held in a static of this
/// never-unloaded assembly, so it is dropped the moment a test ends and again
/// by `ForgetGame` at every commit. Nothing else here outlives a call: a test
/// is found by name each time, as a story is, rather than through a cached
/// `MethodInfo` that would pin the game's load context through a reload.
public static unsafe class TestRunner
{
    // Lock-step with Rust's `orrin_script::TestStep`.
    const byte Passed = 0;
    const byte Running = 1;
    const byte Missing = 2;
    const byte AssertionFailed = 3;
    const byte Threw = 4;
    const byte Faulted = 5;
    const byte OverBudget = 6;
    const byte Malformed = 7;

    static IEnumerator? _routine;
    static int _maxFrames;
    static int _framesLeft;
    static float _secondsLeft;
    static bool _running;
    static string? _fault;
    static (string File, int Line) _faultAt;
    static string _failure = "";

    /// The assembly named `name`: the bindings, so the engine's own probes are
    /// reachable from its integration tests, or a live game assembly.
    static Assembly? Find(string name)
    {
        if (name == GameLoadContext.OrrinAssemblyName)
            return typeof(TestRunner).Assembly;
        return GameAssembly.CurrentAssemblies.FirstOrDefault(assembly => assembly.GetName().Name == name);
    }

    /// Every `[Test]` method in `assembly`, by name, in name order. One with the
    /// wrong shape is listed all the same, and fails when run, so a test cannot
    /// be skipped by a typo in its signature.
    internal static List<(string Name, MethodInfo Method, TestAttribute Test)> Discover(string assembly)
    {
        var tests = new List<(string, MethodInfo, TestAttribute)>();
        if (Find(assembly) is not { } found)
            return tests;
        Type[] types;
        try
        {
            types = found.GetTypes();
        }
        catch (ReflectionTypeLoadException e)
        {
            types = e.Types.Where(type => type is not null).ToArray()!;
        }
        const BindingFlags all = BindingFlags.Static | BindingFlags.Instance | BindingFlags.Public
            | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
        foreach (var type in types)
        foreach (var method in type.GetMethods(all))
        {
            if (method.GetCustomAttribute<TestAttribute>() is { } test)
                tests.Add(($"{type.FullName?.Replace('+', '.')}.{method.Name}", method, test));
        }
        tests.Sort((a, b) => string.CompareOrdinal(a.Item1, b.Item1));
        return tests;
    }

    static bool Shaped(MethodInfo method) =>
        method.IsStatic
        && method.IsPublic
        && !method.ContainsGenericParameters
        && method.GetParameters().Length == 0
        && (method.ReturnType == typeof(void)
            || method.ReturnType == typeof(IEnumerator)
            || method.ReturnType == typeof(IEnumerable));

    /// Every test in `assembly` as `name\tmax frames\tscene` lines, through the
    /// `get_tag` protocol.
    [UnmanagedCallersOnly]
    public static int List(byte* assembly, uint assemblyLength, byte* buffer, int capacity)
    {
        try
        {
            var lines = Discover(Encoding.UTF8.GetString(assembly, (int)assemblyLength))
                .Select(test => $"{test.Name}\t{test.Test.MaxFrames}\t{test.Test.Scene ?? ""}");
            var bytes = Encoding.UTF8.GetBytes(string.Join('\n', lines));
            if (bytes.Length <= capacity)
                bytes.CopyTo(new Span<byte>(buffer, capacity));
            return bytes.Length;
        }
        catch (Exception e)
        {
            Native.LogError($"[test] listing the tests threw: {e}");
            return -1;
        }
    }

    /// Start test `name` in `assembly`, and run it to its end or its first yield.
    [UnmanagedCallersOnly]
    public static byte Begin(byte* assembly, uint assemblyLength, byte* name, uint nameLength)
    {
        Reset();
        var wanted = Encoding.UTF8.GetString(name, (int)nameLength);
        try
        {
            var method = Discover(Encoding.UTF8.GetString(assembly, (int)assemblyLength))
                .FirstOrDefault(test => test.Name == wanted);
            if (method.Method is null)
                return Missing;
            if (!Shaped(method.Method))
            {
                return End(Malformed, null,
                    $"{wanted} is not a test: a [Test] must be public and static, take no parameters, "
                    + "and return void, IEnumerator or IEnumerable");
            }
            _maxFrames = method.Test.MaxFrames;
            _running = true;
            object? result;
            try
            {
                result = method.Method.Invoke(null, null);
            }
            catch (TargetInvocationException e) when (e.InnerException is { } inner)
            {
                return Fail(inner);
            }
            _routine = result switch
            {
                IEnumerator routine => routine,
                IEnumerable sequence => sequence.GetEnumerator(),
                _ => null,
            };
            return _routine is null ? End(Passed, null, "") : Step();
        }
        catch (Exception e)
        {
            return Fail(e);
        }
    }

    /// Carry the running test on after a frame: to its next yield, or not at
    /// all while what it yielded is still waiting.
    [UnmanagedCallersOnly]
    public static byte Resume()
    {
        try
        {
            if (_routine is null)
                return Missing;
            if (_fault is { } fault)
                return End(Faulted, _faultAt, fault);
            if (_framesLeft > 0 && --_framesLeft > 0)
                return Running;
            if (_secondsLeft > 0f && (_secondsLeft -= Time.DeltaTime) > 1e-5f)
                return Running;
            return Step();
        }
        catch (Exception e)
        {
            return Fail(e);
        }
    }

    /// Stop the running test because it has used up its frames. Its `finally`
    /// blocks run, inside the window the engine opened for this call.
    [UnmanagedCallersOnly]
    public static byte Abandon()
    {
        try
        {
            return End(OverBudget, null, $"still running after {_maxFrames} frames (raise it with [Test(MaxFrames = …)])");
        }
        catch (Exception e)
        {
            return Fail(e);
        }
    }

    /// Why the last test failed, as `file\tline\tmessage`, through the
    /// `get_tag` protocol. The file is empty, and the line 0, when the failure
    /// has no line of a script to point at.
    [UnmanagedCallersOnly]
    public static int Failure(byte* buffer, int capacity)
    {
        var bytes = Encoding.UTF8.GetBytes(_failure);
        if (bytes.Length <= capacity)
            bytes.CopyTo(new Span<byte>(buffer, capacity));
        return bytes.Length;
    }

    /// A Behaviour threw from a hook while a test ran, which fails the test on
    /// its next turn. The first fault is the one reported; the rest follow from
    /// it as often as not.
    internal static void NoteFault(string behaviour, string hook, Exception e)
    {
        if (!_running || _fault is not null)
            return;
        _fault = $"{behaviour}.{hook} threw {e.GetType().Name}: {e.Message}";
        _faultAt = Locate(e);
    }

    /// Drop a test left running, before the game's load context is retired.
    internal static void ForgetGame() => Reset();

    static byte Step()
    {
        try
        {
            if (!_routine!.MoveNext())
                return End(Passed, null, "");
            _framesLeft = 0;
            _secondsLeft = 0f;
            switch (_routine.Current)
            {
                case null:
                    _framesLeft = 1;
                    break;
                case Wait { FrameCount: > 0 } wait:
                    _framesLeft = wait.FrameCount;
                    break;
                case Wait { SecondCount: > 0f } wait:
                    _secondsLeft = wait.SecondCount;
                    break;
                case Wait:
                    _framesLeft = 1;
                    break;
                case var other:
                    return End(Malformed, null,
                        $"yielded a {other.GetType().Name}, which a test cannot wait on; "
                        + "yield null, Wait.Frames or Wait.Seconds");
            }
            return Running;
        }
        catch (Exception e)
        {
            return Fail(e);
        }
    }

    static byte Fail(Exception e) =>
        End(e is AssertionException ? AssertionFailed : Threw, Locate(e),
            e is AssertionException ? e.Message : $"{e.GetType().Name}: {e.Message}");

    static byte End(byte status, (string File, int Line)? at, string message)
    {
        var routine = _routine;
        Reset();
        if (routine is IDisposable disposable)
        {
            try
            {
                disposable.Dispose();
            }
            catch (Exception e)
            {
                Native.LogWarn($"[test] cleaning up after the test threw {e.GetType().Name}: {e.Message}");
            }
        }
        // A tween's completion callback is game code bound to a world the
        // engine is about to drop.
        Tweens.ForgetGame();
        var (file, line) = at ?? ("", 0);
        _failure = status == Passed ? "" : $"{file}\t{line}\t{message}";
        return status;
    }

    static void Reset()
    {
        _routine = null;
        _running = false;
        _fault = null;
        _faultAt = ("", 0);
        _framesLeft = 0;
        _secondsLeft = 0f;
    }

    /// The first line of a script the exception passed through, skipping the
    /// assertion that threw it.
    static (string File, int Line) Locate(Exception e)
    {
        foreach (var frame in new StackTrace(e, fNeedFileInfo: true).GetFrames())
        {
            if (frame.GetMethod()?.DeclaringType?.Namespace == typeof(Assert).Namespace)
                continue;
            if (frame.GetFileName() is { Length: > 0 } file)
                return (file, frame.GetFileLineNumber());
        }
        return ("", 0);
    }
}
