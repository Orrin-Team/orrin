using System.Runtime.InteropServices;

namespace Orrin;

/// <summary>
/// Makes an entity's collider a simulated body: physics owns its pose and
/// writes it into the Transform. Without one a collider moves only when
/// something moves its Transform, and nothing pushes it. Read and write it
/// with <see cref="World.GetComponent{T}"/>, <see cref="World.SetComponent{T}"/>
/// and <see cref="World.AddComponent{T}"/>; push it with <see cref="Physics"/>.
/// SI units: kilograms, and a gravity scale of 1 is the scene's gravity. The
/// surface it slides and bounces on is its collider's.
/// </summary>
// Layout must match Rust CRigidBody: seven 4-byte fields, flags as words.
[StructLayout(LayoutKind.Sequential)]
public struct Rigidbody
{
    uint _kinematic;

    /// <summary>Kilograms. Must be positive.</summary>
    public float Mass;

    public float GravityScale;

    /// <summary>The fraction of velocity lost per second.</summary>
    public float LinearDamping;

    public float AngularDamping;

    uint _continuous;
    uint _lockRotation;

    public Rigidbody()
    {
        Mass = 1f;
        GravityScale = 1f;
        LinearDamping = 0.05f;
        AngularDamping = 0.05f;
    }

    /// <summary>
    /// Moved by its Transform like a plain collider, but still pushes what it
    /// moves into: a lift, a door.
    /// </summary>
    public bool Kinematic
    {
        readonly get => _kinematic != 0;
        set => _kinematic = value ? 1u : 0u;
    }

    /// <summary>
    /// Sweeps the body along its motion each step, so a fast one cannot pass
    /// through something thinner than a step's travel. Costs a cast per step.
    /// </summary>
    public bool Continuous
    {
        readonly get => _continuous != 0;
        set => _continuous = value ? 1u : 0u;
    }

    /// <summary>Never turns: a character's capsule, a crate that should only slide.</summary>
    public bool LockRotation
    {
        readonly get => _lockRotation != 0;
        set => _lockRotation = value ? 1u : 0u;
    }
}
