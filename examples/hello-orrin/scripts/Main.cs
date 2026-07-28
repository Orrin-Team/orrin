using Orrin;
using Orrin.Math;

namespace HelloOrrin;

// The project's entry Behaviour, named by `scripts.entry` in orrin.toml. The
// engine attaches exactly this one at startup; everything else in the scene is
// spawned from here through the script API.
//
// This file is the hot-reload demo. With the engine running, change a value
// below, `dotnet build`, and hit "Reload scripts" in the editor: the cube keeps
// its position because _spin is preserved across the swap, while the new
// SpinSpeed takes effect immediately.
public class Main : Behaviour
{
    public float SpinSpeed = 45f;
    public float Height = 1.5f;

    private Entity _cube;
    private float _spin;

    public override void OnStart()
    {
        var spawn = Transform.Identity;
        spawn.Position = new Vector3(0f, Height, 0f);
        _cube = World.SpawnRenderable("cube", "gold", spawn);
        Debug.Log($"hello-orrin started; cube = {_cube}");
    }

    public override void OnUpdate(float deltaTime)
    {
        _spin += SpinSpeed * deltaTime;

        var t = Transform.Identity;
        t.Position = new Vector3(0f, Height, 0f);
        t.Rotation = Quaternion.Euler(0f, _spin, 0f);
        Native.SetTransform(_cube, t);
    }
}
