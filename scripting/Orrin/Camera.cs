using Orrin.Math;

namespace Orrin;

/// <summary>
/// The camera the frame is drawn from. A script moves it by setting
/// <see cref="position"/> and <see cref="rotation"/>, which take effect in the
/// same frame. While a camera entity is <see cref="Active"/> these move that
/// entity, and the view follows it; with none, they move the engine's own view.
/// </summary>
public static class Camera
{
    /// <summary>
    /// The camera entity the game draws from, or null when there is none. Set
    /// it to switch cameras; set null to go back to the highest
    /// <see cref="CameraComponent.Priority"/>, which is also what draws when
    /// the selected camera is despawned. Setting an entity that is not a
    /// camera logs and changes nothing.
    /// </summary>
    public static Entity? Active
    {
        get => Native.GetActiveCamera() is { IsValid: true } camera ? camera : null;
        set => Native.SetActiveCamera(value ?? Entity.Null);
    }

    /// <summary>Where the camera stands, in metres.</summary>
    public static Vector3 position
    {
        get => Native.GetCamera().Position;
        set
        {
            var t = Native.GetCamera();
            t.Position = value;
            Native.SetCamera(t);
        }
    }

    /// <summary>
    /// Which way the camera faces: the rotation of <see cref="Vector3.forward"/>
    /// (-Z) and <see cref="Vector3.up"/>.
    /// </summary>
    public static Quaternion rotation
    {
        get => Native.GetCamera().Rotation;
        set
        {
            var t = Native.GetCamera();
            t.Rotation = value;
            Native.SetCamera(t);
        }
    }

    /// <summary>The direction the camera looks.</summary>
    public static Vector3 forward => rotation * Vector3.forward;

    /// <summary>The camera's right, level with its view.</summary>
    public static Vector3 right => rotation * Vector3.right;

    /// <summary>The camera's up.</summary>
    public static Vector3 up => rotation * Vector3.up;

    /// <summary>Stand at <paramref name="at"/> facing <paramref name="facing"/>, in one call.</summary>
    public static void Place(Vector3 at, Quaternion facing)
    {
        var t = Transform.Identity;
        t.Position = at;
        t.Rotation = facing;
        Native.SetCamera(t);
    }
}
