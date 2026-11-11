using Orrin;
using Orrin.Math;
using Quaternion = Orrin.Math.Quaternion;

namespace DemoGame;

// Movement and shooting demo entry — run with:
//   ORRIN_ENTRY="DemoGame.MovementDemo, DemoGame" cargo run -p orrin-core
// then press Play, click the viewport to lock the mouse, and see FpsPlayer for
// the controls. Esc lets go of the mouse.
//
// Builds a course of its own a hundred metres down -Z, clear of whatever scene
// it is run in: ramps to slide down and launch off, a raised deck, pillars to
// weave through, and at the far end a range of targets, one of them moving,
// and a stack of crates to knock down.
public class MovementDemo : Behaviour
{
    static readonly Vector3 Origin = new(0f, 0f, -100f);

    public override void OnStart()
    {
        Floor();
        Ramps();
        Pillars();
        Range();

        var spawn = Transform.Identity;
        spawn.Position = Origin + new Vector3(0f, 1.2f, 25f);
        var player = World.Spawn(spawn);
        World.AddScript<FpsPlayer>(player);
        Debug.Log("movement demo: Play, click to lock the mouse; WASD, Space, Ctrl to slide, click to fire");
    }

    public override void OnUpdate(float deltaTime)
    {
        var screen = UI.ScreenSize;
        var centre = screen * 0.5f;
        UI.Panel(new Rect(centre.x - 1f, centre.y - 8f, 2f, 16f), new Color(1f, 1f, 1f, 0.8f));
        UI.Panel(new Rect(centre.x - 8f, centre.y - 1f, 16f, 2f), new Color(1f, 1f, 1f, 0.8f));
        UI.Label($"Hits {Target.TotalHits} / {BallLauncher.Fired} balls", new Rect(24f, 24f, 400f, 30f));
        if (!Input.cursorLocked)
            UI.Label("Click to play · Esc to let go of the mouse", new Rect(24f, 56f, 600f, 30f));
    }

    /// A box that collides, placed by its centre and full size.
    static Entity Block(string material, Vector3 centre, Vector3 size, Quaternion rotation)
    {
        var transform = Transform.Identity;
        transform.Position = Origin + centre;
        transform.Rotation = rotation;
        transform.Scale = size;
        var block = World.SpawnRenderable("cube", material, transform);
        World.AddCollider(block, new BoxCollider());
        return block;
    }

    static Entity Block(string material, Vector3 centre, Vector3 size) =>
        Block(material, centre, size, Quaternion.identity);

    static void Floor()
    {
        Block("ground", new Vector3(0f, -0.5f, 0f), new Vector3(80f, 1f, 80f));
        foreach (var (centre, size) in new[]
        {
            (new Vector3(0f, 1f, -40f), new Vector3(80f, 2f, 1f)),
            (new Vector3(0f, 1f, 40f), new Vector3(80f, 2f, 1f)),
            (new Vector3(-40f, 1f, 0f), new Vector3(1f, 2f, 80f)),
            (new Vector3(40f, 1f, 0f), new Vector3(1f, 2f, 80f)),
        })
            Block("rock", centre, size);
    }

    static void Ramps()
    {
        // A long climb to a deck and a steep drop off its far side: run up,
        // crouch at the top, and the slide down comes out fast.
        Block("clay", new Vector3(-22f, 2.7f, 10f), new Vector3(6f, 0.5f, 16f), Quaternion.Euler(20f, 0f, 0f));
        Block("glossy", new Vector3(-22f, 5.3f, -4f), new Vector3(10f, 0.5f, 12f));
        Block("clay", new Vector3(-22f, 2.6f, -15.5f), new Vector3(6f, 0.5f, 12f), Quaternion.Euler(-28f, 0f, 0f));
        // Kickers: slide into one and leave the ground.
        Block("clay", new Vector3(14f, 0.6f, 12f), new Vector3(5f, 0.4f, 4f), Quaternion.Euler(15f, 0f, 0f));
        Block("clay", new Vector3(22f, 1.0f, 4f), new Vector3(5f, 0.4f, 6f), Quaternion.Euler(22f, 0f, 0f));
    }

    static void Pillars()
    {
        for (var row = 0; row < 3; row++)
        for (var column = 0; column < 4; column++)
        {
            var x = 8f + column * 4f + (row % 2) * 2f;
            var z = -6f - row * 5f;
            Block("rock", new Vector3(x, 2f, z), new Vector3(1f, 4f, 1f));
        }
    }

    static void Range()
    {
        foreach (var (x, y, z) in new[] { (-10f, 2f, -30f), (-3f, 3.5f, -34f), (4f, 1.5f, -28f), (11f, 4.5f, -36f) })
        {
            var board = Block("gold", new Vector3(x, y, z), new Vector3(2.5f, 2.5f, 0.3f));
            World.AddScript<Target>(board);
        }
        var shuttle = Block("gold", new Vector3(0f, 6.5f, -37f), new Vector3(2f, 2f, 0.3f));
        World.AddScript<Target>(shuttle);
        World.AddScript<Shuttle>(shuttle);

        // A stack of crates to knock over, on the range's left.
        for (var level = 0; level < 4; level++)
        for (var i = 0; i < 4 - level; i++)
        {
            var transform = Transform.Identity;
            transform.Position = Origin + new Vector3(-24f + i * 1.05f + level * 0.525f, 0.5f + level, -30f);
            var crate = World.SpawnRenderable("cube", "copper", transform);
            World.AddCollider(crate, new BoxCollider());
            World.AddComponent(crate, new Rigidbody { Mass = 8f });
        }
    }
}
