namespace Orrin;

/// <summary>
/// Draws a float, int or uint field as a slider between <paramref name="min"/>
/// and <paramref name="max"/>. The bounds are what the slider can produce, not a
/// rule about the value: one loaded from a scene or set by a script outside
/// them is shown as it is and never clamped.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public sealed class RangeAttribute(float min, float max) : Attribute
{
    public float Min { get; } = min;
    public float Max { get; } = max;
}

/// <summary>Text shown when the pointer rests on the field's label in the inspector.</summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public sealed class TooltipAttribute(string text) : Attribute
{
    public string Text { get; } = text;
}

/// <summary>
/// A label drawn in the inspector above this field, to group it with the
/// fields that follow.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public sealed class HeaderAttribute(string text) : Attribute
{
    public string Text { get; } = text;
}

/// <summary>
/// Keeps the field out of the inspector. It is still saved with the scene and
/// still carried across a hot reload; use <see cref="TransientAttribute"/> for
/// a field that should be neither.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public sealed class HideInInspectorAttribute : Attribute;
