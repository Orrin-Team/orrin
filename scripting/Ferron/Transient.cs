namespace Ferron;

/// Marks a Behaviour field or property as *not* worth preserving across a hot
/// reload: a cache, a handle re-acquired in OnStart, anything derivable.
///
/// Also silences the "not preserved" warning for a field whose type the reload
/// could not carry anyway.
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
public sealed class TransientAttribute : Attribute;
