namespace Orrin;

/// A queryable gameplay label; mirrors the Rust `Tag` component. Fetched by
/// value — mutating a copy does not write back; assign via World.SetTag.
public readonly struct Tag
{
    public readonly string Value;

    public Tag(string value) => Value = value;

    public override string ToString() => Value;
}
