using System.Globalization;
using System.Runtime.InteropServices;

namespace Orrin.Math;

/// <summary>
/// An RGBA color with float components, nominally in [0, 1] (values above 1
/// are meaningful for HDR/emissive use and are not clamped).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Color : IEquatable<Color>
{
    public readonly float r;
    public readonly float g;
    public readonly float b;
    public readonly float a;

    public Color(float r, float g, float b, float a = 1f)
    {
        this.r = r;
        this.g = g;
        this.b = b;
        this.a = a;
    }

    public static Color white => new(1f, 1f, 1f);
    public static Color black => new(0f, 0f, 0f);
    public static Color red => new(1f, 0f, 0f);
    public static Color green => new(0f, 1f, 0f);
    public static Color blue => new(0f, 0f, 1f);
    public static Color yellow => new(1f, 1f, 0f);
    public static Color cyan => new(0f, 1f, 1f);
    public static Color magenta => new(1f, 0f, 1f);
    public static Color gray => new(0.5f, 0.5f, 0.5f);
    /// <summary>Fully transparent black.</summary>
    public static Color clear => new(0f, 0f, 0f, 0f);

    /// <summary>
    /// Parse a hex color: "RGB", "RRGGBB", or "RRGGBBAA", with or without a
    /// leading '#'. Throws <see cref="FormatException"/> on anything else.
    /// </summary>
    public static Color FromHex(string hex)
    {
        var span = hex.AsSpan().Trim();
        if (!span.IsEmpty && span[0] == '#')
            span = span[1..];

        static float Channel(ReadOnlySpan<char> two) =>
            byte.Parse(two, NumberStyles.HexNumber, CultureInfo.InvariantCulture) / 255f;

        return span.Length switch
        {
            // Shorthand "F80" == "FF8800".
            3 => new Color(
                Channel([span[0], span[0]]),
                Channel([span[1], span[1]]),
                Channel([span[2], span[2]])),
            6 => new Color(Channel(span[..2]), Channel(span[2..4]), Channel(span[4..6])),
            8 => new Color(
                Channel(span[..2]), Channel(span[2..4]), Channel(span[4..6]), Channel(span[6..8])),
            _ => throw new FormatException(
                $"'{hex}' is not a hex color (expected RGB, RRGGBB, or RRGGBBAA)"),
        };
    }

    /// <summary>This color as "RRGGBB" (or "RRGGBBAA"), clamped to [0, 1] per channel.</summary>
    public string ToHex(bool includeAlpha = false)
    {
        static int Byte(float channel) => (int)MathF.Round(Mathf.Clamp01(channel) * 255f);
        var rgb = string.Create(
            CultureInfo.InvariantCulture, $"{Byte(r):X2}{Byte(g):X2}{Byte(b):X2}");
        return includeAlpha
            ? rgb + string.Create(CultureInfo.InvariantCulture, $"{Byte(a):X2}")
            : rgb;
    }

    /// <summary>Linear interpolation; <paramref name="t"/> is clamped to [0, 1].</summary>
    public static Color Lerp(Color a, Color b, float t) => LerpUnclamped(a, b, Mathf.Clamp01(t));

    public static Color LerpUnclamped(Color a, Color b, float t) => new(
        a.r + (b.r - a.r) * t,
        a.g + (b.g - a.g) * t,
        a.b + (b.b - a.b) * t,
        a.a + (b.a - a.a) * t);

    /// <summary>This color with a different alpha.</summary>
    public Color WithAlpha(float alpha) => new(r, g, b, alpha);

    public static Color operator +(Color a, Color b) => new(a.r + b.r, a.g + b.g, a.b + b.b, a.a + b.a);
    public static Color operator -(Color a, Color b) => new(a.r - b.r, a.g - b.g, a.b - b.b, a.a - b.a);
    /// <summary>Component-wise (modulate) multiplication.</summary>
    public static Color operator *(Color a, Color b) => new(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a);
    public static Color operator *(Color c, float s) => new(c.r * s, c.g * s, c.b * s, c.a * s);
    public static Color operator *(float s, Color c) => new(c.r * s, c.g * s, c.b * s, c.a * s);

    public static bool operator ==(Color a, Color b) =>
        a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;

    public static bool operator !=(Color a, Color b) => !(a == b);

    public static explicit operator Vector4(Color c) => new(c.r, c.g, c.b, c.a);
    public static explicit operator Color(Vector4 v) => new(v.x, v.y, v.z, v.w);

    public bool Equals(Color other) => this == other;
    public override bool Equals(object? obj) => obj is Color other && this == other;
    public override int GetHashCode() => HashCode.Combine(r, g, b, a);

    public override string ToString() =>
        string.Create(CultureInfo.InvariantCulture, $"RGBA({r:F3}, {g:F3}, {b:F3}, {a:F3})");
}
