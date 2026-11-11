namespace Orrin;

/// <summary>
/// Gives a Behaviour a stable identity in the component registry, so its fields
/// are saved with the scene, shown in the inspector, and carried across a hot
/// reload.
/// </summary>
/// <remarks>
/// <para>
/// The id is written into every scene file that references the component, and
/// is the *only* thing that ties a saved entity back to this class. It is
/// therefore deliberately not derived from the type's name: rename
/// <c>Spinner</c> to <c>Rotator</c> and every scene keeps loading, because the id did not
/// move. Change the id and they all stop — which is a rename of the component
/// type as far as the project is concerned, and should be as loud as one.
/// </para>
///
/// <para>
/// Convention is <c>game.thing</c>, lower case, dotted. Anything unique will do; the
/// engine reserves the <c>orrin.</c> prefix for its own components.
/// </para>
///
/// <para>
/// A Behaviour without this attribute still runs — it just has no registry
/// entry, so its fields are not saved or inspectable. Hot reload still carries
/// them (see <see cref="BehaviourState"/>).
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, Inherited = false)]
public sealed class ComponentAttribute(string id) : Attribute
{
    /// <summary>
    /// The component's id in the registry, as every scene that uses it spells it.
    /// </summary>
    public string Id { get; } = id;

    /// <summary>
    /// What the inspector calls it. Defaults to the type name, and unlike <see cref="Id"/>
    /// may change freely.
    /// </summary>
    public string? Name { get; init; }
}
