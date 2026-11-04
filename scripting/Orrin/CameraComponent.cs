using System.Runtime.InteropServices;

namespace Orrin;

/// <summary>
/// The lens of a camera entity. Where the camera stands and which way it looks
/// is the entity's Transform (-Z forward); add one with
/// <see cref="World.AddCamera"/> and draw from it with
/// <see cref="Camera.Active"/>.
/// </summary>
// Layout must match Rust CCamera (three floats and an int).
[StructLayout(LayoutKind.Sequential)]
public struct CameraComponent
{
    /// <summary>The vertical field of view, in degrees, between 0 and 180.</summary>
    public float FieldOfView;

    /// <summary>The near plane, in metres. Greater than zero.</summary>
    public float Near;

    /// <summary>The far plane, in metres. Beyond <see cref="Near"/>.</summary>
    public float Far;

    /// <summary>
    /// Which camera draws when <see cref="Camera.Active"/> has not been set:
    /// the highest wins.
    /// </summary>
    public int Priority;

    /// <summary>The engine's default lens: 60°, 0.1 m to 1000 m, priority 0.</summary>
    public CameraComponent()
    {
        FieldOfView = 60f;
        Near = 0.1f;
        Far = 1000f;
        Priority = 0;
    }
}
