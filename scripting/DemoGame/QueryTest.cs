using Ferron;
using Ferron.Math;

namespace DemoGame;

// Self-checking exercise of the entity-querying ABI (issue #12). Run with
//   FERRON_ENTRY="DemoGame.QueryTest, DemoGame"
// and read the [PASS]/[FAIL] lines; the summary prints once phase 2 finishes.
//
// Phases are separated by update ticks because structural changes (SetTag,
// Despawn, spawned components) apply when the dispatch window closes — each
// phase observes what the previous one queued.
public class QueryTest : Behaviour
{
    // 20 > the 16-slot initial buffer in Native.FindAllByTag, so the
    // resize-and-retry path runs; the long tag overflows GetTag's 64-byte
    // stackalloc the same way.
    private const int Crowd = 20;
    private static readonly string LongTag = new('x', 100);

    private Entity _untagged;
    private Entity _longTagged;
    private Entity _victim;
    private int _tick;
    private int _failed;
    private int _checked;

    public override void OnStart()
    {
        // Nothing is tagged yet anywhere in the demo scene.
        Check(World.FindByTag("qt-entry") is null, "world starts with no qt-entry tag");

        Check(World.SetTag(Entity, "qt-entry"), "SetTag on live entity returns true");
        Check(World.FindByTag("qt-entry") is null, "SetTag is not visible in the same tick");

        // Spawned handles are allocated immediately (only their components are
        // deferred), so tagging them in the same tick must succeed.
        for (var i = 0; i < Crowd; i++)
        {
            var t = Transform.Identity;
            t.Position = new Vector3(i - Crowd / 2f, 0.5f, -4f);
            t.Scale = Vector3.one * 0.4f;
            var cube = World.SpawnRenderable("cube", "copper", t);
            Check(World.SetTag(cube, "qt-crowd"), $"SetTag on spawned cube {i} returns true");
        }

        _untagged = World.SpawnRenderable("cube", "clay", Transform.Identity);
        _longTagged = World.SpawnRenderable("cube", "neon", Transform.Identity);
        Check(World.SetTag(_longTagged, LongTag), "SetTag with a 100-byte tag returns true");
    }

    public override void OnUpdate(float deltaTime)
    {
        // Tick 1 can share a dispatch window with OnStart; by tick 2 the
        // deferred commands from OnStart have definitely been applied.
        _tick++;
        if (_tick == 2) VerifyQueries();
        if (_tick == 4) VerifyStaleHandles();
    }

    private void VerifyQueries()
    {
        var found = World.FindByTag("qt-entry");
        Check(found is { } f && f.Equals(Entity), "FindByTag resolves to the entry entity");
        Check(World.FindByTag("qt-nonexistent") is null, "FindByTag misses an unknown tag");

        Check(Entity.HasComponent<Tag>(), "entry entity has a Tag component");
        Check(!Entity.HasComponent<Transform>(), "entry entity has no Transform component");
        Check(_untagged.HasComponent<Transform>(), "spawned cube has a Transform component");
        Check(!_untagged.HasComponent<Tag>(), "untagged cube has no Tag component");

        Check(Entity.GetComponent<Tag>() is { Value: "qt-entry" }, "GetComponent reads the tag back");
        Check(_untagged.GetComponent<Tag>() is null, "GetComponent is null without a Tag");
        Check(_longTagged.GetComponent<Tag>() is { } lt && lt.Value == LongTag,
            "GetComponent survives the 64-byte buffer retry");

        var crowd = World.FindAllByTag("qt-crowd");
        Check(crowd.Length == Crowd, $"FindAllByTag finds all {Crowd} (got {crowd.Length})");
        Check(World.FindAllByTag("qt-nonexistent").Length == 0, "FindAllByTag misses cleanly");

        // Queue work for the stale-handle phase.
        _victim = crowd[0];
        Check(World.Despawn(_victim), "Despawn on live entity returns true");
    }

    private void VerifyStaleHandles()
    {
        Check(World.FindAllByTag("qt-crowd").Length == Crowd - 1,
            "despawned entity left the tag query");
        Check(!_victim.HasComponent<Tag>(), "stale handle has no components");
        Check(_victim.GetComponent<Tag>() is null, "stale handle reads no tag");
        Check(!World.SetTag(_victim, "qt-zombie"), "SetTag on stale handle returns false");

        Native.Log(_failed == 0
            ? $"[QueryTest] ALL PASS ({_checked} checks)"
            : $"[QueryTest] {_failed}/{_checked} CHECKS FAILED");
    }

    private void Check(bool condition, string name)
    {
        _checked++;
        if (!condition) _failed++;
        Native.Log($"[QueryTest] {(condition ? "PASS" : "FAIL")}: {name}");
    }
}
