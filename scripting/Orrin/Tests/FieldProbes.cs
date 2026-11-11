using Orrin.Math;

namespace Orrin.Tests;

// Test-only behaviour for crates/core/tests/script_fields.rs, shipped in
// Orrin.dll for the reason the lifecycle probes are. No game assembly lists it,
// so that test registers it by hand.

/// One field of each type the registry can save beyond the originals: an
/// entity reference, a Vector2, a Color, and lists of both shapes.
class FieldsProbe : Behaviour
{
    public Entity Target;
    public Vector2 Offset = new(1f, 2f);
    public Color Tint = Color.white;
    public List<Entity> Others = [];
    public int[] Steps = [1, 2];
}
