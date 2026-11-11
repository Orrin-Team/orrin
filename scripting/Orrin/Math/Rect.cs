using System.Globalization;
using System.Runtime.InteropServices;

namespace Orrin.Math;

/// <summary>
/// An axis-aligned rectangle: a corner and a size. In UI space the corner is the
/// top left and y grows down, in physical pixels — the space
/// <see cref="Input.MousePosition"/> reports.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Rect : IEquatable<Rect>
{
    /// <summary>The corner's x.</summary>
    public readonly float x;
    /// <summary>The corner's y: the top edge in UI space.</summary>
    public readonly float y;
    /// <summary>The size along x.</summary>
    public readonly float width;
    /// <summary>The size along y.</summary>
    public readonly float height;

    /// <summary>A rectangle from its corner and its size.</summary>
    public Rect(float x, float y, float width, float height)
    {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    /// <summary>A rectangle with its corner at <paramref name="position"/>, <paramref name="size"/> across.</summary>
    public Rect(Vector2 position, Vector2 size) : this(position.x, position.y, size.x, size.y) { }

    /// <summary><see cref="x"/>, the near edge.</summary>
    public float xMin => x;
    /// <summary><see cref="y"/>, the near edge.</summary>
    public float yMin => y;
    /// <summary>The far edge, <see cref="x"/> plus <see cref="width"/>, which <see cref="Contains"/> leaves out.</summary>
    public float xMax => x + width;
    /// <summary>The far edge, <see cref="y"/> plus <see cref="height"/>, which <see cref="Contains"/> leaves out.</summary>
    public float yMax => y + height;

    /// <summary>The corner: (<see cref="x"/>, <see cref="y"/>).</summary>
    public Vector2 position => new(x, y);
    /// <summary>(<see cref="width"/>, <see cref="height"/>).</summary>
    public Vector2 size => new(width, height);
    /// <summary>The middle point.</summary>
    public Vector2 center => new(x + width * 0.5f, y + height * 0.5f);

    /// <summary>
    /// Whether <paramref name="point"/> is inside. The near edges are in and the
    /// far edges out, so two rectangles that share an edge never both contain a
    /// point on it.
    /// </summary>
    public bool Contains(Vector2 point) =>
        point.x >= x && point.x < xMax && point.y >= y && point.y < yMax;

    /// <summary>Exact equality of corner and size, with no tolerance.</summary>
    public static bool operator ==(Rect a, Rect b) =>
        a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;

    /// <summary>Whether the corner or the size differs.</summary>
    public static bool operator !=(Rect a, Rect b) => !(a == b);

    /// <inheritdoc/>
    public bool Equals(Rect other) => this == other;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is Rect other && this == other;

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode.Combine(x, y, width, height);

    /// <summary><c>(x:…, y:…, width:…, height:…)</c>, in the invariant culture.</summary>
    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture, "(x:{0}, y:{1}, width:{2}, height:{3})",
            x, y, width, height);
}
