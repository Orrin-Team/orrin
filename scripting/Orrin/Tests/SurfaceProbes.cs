namespace Orrin.Tests;

// Probes for the script-facing surface — reaching another Behaviour, an
// entity's own members, asset-reference spawns and timers — driven by
// crates/orrin-script/tests/lifecycle.rs. The test supplies recording stubs for
// the ABI calls these make, so each probe logs what it saw and the test
// asserts on both the log and the calls.

/// What `IndexProbe` looks for, on entity 2.
class IndexMarker : Behaviour { }

/// Asks the behaviour index about entity 2, and about itself, and logs it.
class IndexProbe : Behaviour
{
    public override void OnUpdate(float deltaTime)
    {
        var marked = new Entity(2, 0);
        var got = marked.GetBehaviour<IndexMarker>();
        var tried = marked.TryGetBehaviour<IndexMarker>(out var viaTry);
        var respawned = new Entity(2, 1).GetBehaviour<IndexMarker>();
        var all = World.FindAll<IndexMarker>();
        Native.Log(
            $"probe:index get={Describe(got)} try={tried}:{Describe(viaTry)} respawned={Describe(respawned)} "
            + $"all={string.Join(",", all.Select(Describe))} first={Describe(World.FindFirst<IndexMarker>())} "
            + $"self={ReferenceEquals(Entity.GetBehaviour<IndexProbe>(), this)} "
            + $"wrongType={Describe(marked.GetBehaviour<IndexProbe>())}");
    }

    static string Describe(Behaviour? behaviour) =>
        behaviour is null ? "none" : behaviour.Entity.Index.ToString();
}

/// Drives every entity-level member once, on itself and on entity 6v2.
class EntityProbe : Behaviour
{
    public override void OnUpdate(float deltaTime)
    {
        var other = new Entity(6, 2);
        var placed = Transform.Identity;
        placed.Position = new Math.Vector3(1f, 2f, 3f);
        other.LocalTransform = placed;
        other.WorldTransform = placed;
        Native.Log(
            $"probe:entity local={Entity.LocalTransform.Position.x} world={Entity.WorldTransform.Position.x} "
            + $"parent={Entity.Parent} tag={Entity.Tag ?? "none"}");
        Native.Log(
            $"probe:entity reparent={Entity.SetParent(other, keepWorld: false)} tag={Entity.SetTag("hero")} "
            + $"script={Entity.AddScript<LifecycleProbe>()} despawn={Entity.Despawn()}");
    }
}

/// Spawns by asset reference, is refused a null one, and swaps a material.
class AssetProbe : Behaviour
{
    static readonly Guid MeshId = Guid.Parse("00112233-4455-6677-8899-aabbccddeeff");
    static readonly Guid FirstMaterial = Guid.Parse("10000000-0000-0000-0000-000000000001");
    static readonly Guid SecondMaterial = Guid.Parse("20000000-0000-0000-0000-000000000002");

    public override void OnUpdate(float deltaTime)
    {
        var mesh = new AssetRef<Mesh>(MeshId);
        var spawned = World.SpawnRenderable(mesh, new AssetRef<Material>(FirstMaterial), Transform.Identity);
        var refused = World.SpawnRenderable(mesh, default(AssetRef<Material>), Transform.Identity);
        var swapped = World.SetMaterial(spawned, new AssetRef<Material>(SecondMaterial));
        var bare = World.SetMaterial(new Entity(4, 0), new AssetRef<Material>(SecondMaterial));
        Native.Log($"probe:asset spawned={spawned} refused={refused} swapped={swapped} bare={bare}");
    }
}

/// One timer that fires once, one that repeats, and one cancelled before it
/// is due. Ticked by the test at a quarter of a second, then once at 1.25 s.
class TimerProbe : Behaviour
{
    [Transient] TimerHandle? _once;
    [Transient] TimerHandle? _cancelled;
    [Transient] int _tick;

    public override void OnStart()
    {
        _once = After(0.5f, () => Native.Log("probe:timer once"));
        Every(0.5f, () => Native.Log("probe:timer every"));
        _cancelled = After(0.75f, () => Native.Log("probe:timer cancelled"));
    }

    public override void OnUpdate(float deltaTime)
    {
        _tick++;
        Native.Log($"probe:update {_tick}");
        if (_tick == 2)
            Native.Log($"probe:cancel {_cancelled!.Cancel()} {_once!.Cancel()} pending={_once.IsPending}");
    }
}

/// A timer callback that throws must fault the script as a throwing OnUpdate
/// does, and keep that tick's OnUpdate from running.
class ThrowingTimer : Behaviour
{
    public override void OnStart() => After(0f, () => throw new InvalidOperationException("timer boom"));

    public override void OnUpdate(float deltaTime) => Native.Log("probe:throwing-timer update");
}
