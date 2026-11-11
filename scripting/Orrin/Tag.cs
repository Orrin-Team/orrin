namespace Orrin;

/// <summary>
/// A queryable gameplay label; mirrors the Rust <c>Tag</c> component. Fetched by
/// value — mutating a copy does not write back; assign via World.SetTag.
/// </summary>
public readonly struct Tag
{
    /// <summary>The label's text.</summary>
    public readonly string Value;

    /// <summary>A tag reading <paramref name="value"/>.</summary>
    public Tag(string value) => Value = value;

    /// <summary>The tag's <see cref="Value"/>.</summary>
    public override string ToString() => Value;
}
