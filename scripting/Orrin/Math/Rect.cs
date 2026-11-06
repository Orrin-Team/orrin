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
    public readonly float x;
    public readonly float y;
    public readonly float width;
    public readonly float height;

    public Rect(float x, float y, float width, float height)
    {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public Rect(Vector2 position, Vector2 size) : this(position.x, position.y, size.x, size.y) { }

    public float xMin => x;
    public float yMin => y;
    public float xMax => x + width;
    public float yMax => y + height;

    public Vector2 position => new(x, y);
    public Vector2 size => new(width, height);
    public Vector2 center => new(x + width * 0.5f, y + height * 0.5f);

    /// <summary>
    /// Whether <paramref name="point"/> is inside. The near edges are in and the
    /// far edges out, so two rectangles that share an edge never both contain a
    /// point on it.
    /// </summary>
    public bool Contains(Vector2 point) =>
        point.x >= x && point.x < xMax && point.y >= y && point.y < yMax;

    public static bool operator ==(Rect a, Rect b) =>
        a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;

    public static bool operator !=(Rect a, Rect b) => !(a == b);

    public bool Equals(Rect other) => this == other;

    public override bool Equals(object? obj) => obj is Rect other && this == other;

    public override int GetHashCode() => HashCode.Combine(x, y, width, height);

    public override string ToString() =>
        string.Format(CultureInfo.InvariantCulture, "(x:{0}, y:{1}, width:{2}, height:{3})",
            x, y, width, height);
}
