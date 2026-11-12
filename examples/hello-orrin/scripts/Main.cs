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

    // A sphere beside the cube, drawn with assets/materials/rising_bands.omat,
    // whose surface shader is assets/shaders/rising_bands.oshader. Edit and
    // save the shader while the engine runs and the sphere changes with it.
    private Entity _sphere;
    private AssetRef<Material> _bands;
    private bool _banded;

    /// <summary>The cube OnStart spawned, which <c>Tests.cs</c> looks at.</summary>
    public Entity Cube => _cube;

    public override void OnStart()
    {
        var spawn = Transform.Identity;
        spawn.Position = new Vector3(0f, Height, 0f);
        _cube = World.SpawnRenderable("cube", "gold", spawn);
        Debug.Log($"hello-orrin started; cube = {_cube}");

        spawn.Position = new Vector3(2f, Height, 0f);
        _sphere = World.SpawnRenderable("sphere", "clay", spawn);
        // By path, because this Behaviour is attached by the engine rather than
        // saved in a scene; one in a scene would hold an AssetRef<Material>
        // field set in the inspector instead.
        _bands = Assets.Find<Material>("materials/rising_bands.omat");
    }

    public override void OnUpdate(float deltaTime)
    {
        // The sphere's renderer lands when OnStart's spawn applies, so the
        // material goes on the first frame that finds it.
        if (!_banded && _bands.IsValid && World.SetMaterial(_sphere, _bands))
        {
            _banded = true;
            Debug.Log("the sphere is drawn with materials/rising_bands.omat");
        }

        _spin += SpinSpeed * deltaTime;

        var t = Transform.Identity;
        t.Position = new Vector3(0f, Height, 0f);
        t.Rotation = Quaternion.Euler(0f, _spin, 0f);
        _cube.LocalTransform = t;
    }
}
