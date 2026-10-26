using System.Collections.Immutable;

using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace Orrin.Analyzers;

/// Makes the registry-visible / `[Transient]` split a compile-time decision
/// rather than something you discover the first time a scene reloads.
///
/// Every mutable instance field on a Behaviour is one of three things:
///
/// 1. **Registry-visible** — a type the component registry can save, inspect and
///    carry across a hot reload. Nothing to do.
/// 2. **Explicitly `[Transient]`** — a cache, a handle re-acquired in OnStart,
///    anything derivable. Nothing to do.
/// 3. **Neither**, which is the case this reports. The field silently does not
///    survive a save, and may or may not survive a reload depending on whether
///    `BehaviourState` happens to be able to carry its type. That "may or may
///    not" is the whole problem: the failure shows up as a value quietly
///    reverting, minutes later, with nothing pointing at the field.
///
/// `readonly` fields and auto-property backing storage are excluded for the same
/// reasons `BehaviourState.CapturableFields` excludes them — a `readonly` value
/// belongs to the constructor, and a property carries its own attributes.
///
/// This is a warning, never an error: a field that is genuinely neither is a
/// legitimate thing to have while a Behaviour is being written, and an analyzer
/// that stops the build during that is an analyzer people turn off.
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class BehaviourFieldAnalyzer : DiagnosticAnalyzer
{
    public const string UnmarkedFieldId = "ORRIN001";

    static readonly DiagnosticDescriptor UnmarkedField = new(
        UnmarkedFieldId,
        title: "Behaviour field is neither registry-visible nor [Transient]",
        messageFormat:
            "'{0}' is of type '{1}', which the component registry cannot save, and it is not "
            + "marked [Transient] — it will not survive a save, and may not survive a hot reload. "
            + "Mark it [Transient], or use bool, int, uint, float, string, Vector3, Quaternion, "
            + "AssetRef<T>, or an enum.",
        category: "Orrin.Registry",
        defaultSeverity: DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description:
            "The component registry saves, inspects and reloads a closed set of field types. "
            + "A field outside it must say so, so that losing its value is a decision rather "
            + "than a surprise.");

    /// The engine's two visible value types, matched by full name rather than by
    /// symbol identity: the analyzer has no reference to the Orrin assembly's own
    /// compilation, and a name is what survives that.
    ///
    /// The primitives are *not* here. `ToDisplayString` renders them as the C#
    /// keyword (`float`, not `System.Single`), so a name table would silently
    /// match none of them and report every ordinary field — the loudest possible
    /// way for an analyzer to be useless. `SpecialType` is the identity the
    /// compiler actually guarantees.
    static readonly ImmutableHashSet<string> VisibleEngineTypes = ImmutableHashSet.Create(
        "Orrin.Math.Vector3",
        "Orrin.Math.Quaternion");

    static readonly ImmutableHashSet<SpecialType> VisiblePrimitives = ImmutableHashSet.Create(
        SpecialType.System_Boolean,
        SpecialType.System_Int32,
        SpecialType.System_UInt32,
        SpecialType.System_Single,
        SpecialType.System_String);

    /// Generic, so matched on its definition: `AssetRef<Prefab>` and every other
    /// instantiation share it.
    const string AssetRefType = "Orrin.AssetRef<T>";

    const string BehaviourType = "Orrin.Behaviour";
    const string TransientAttribute = "Orrin.TransientAttribute";

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics =>
        ImmutableArray.Create(UnmarkedField);

    public override void Initialize(AnalysisContext context)
    {
        // Generated code is nobody's to fix, and running concurrently is safe
        // because nothing here holds state between symbols.
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSymbolAction(Analyze, SymbolKind.Field);
    }

    static void Analyze(SymbolAnalysisContext context)
    {
        var field = (IFieldSymbol)context.Symbol;

        // A `readonly` value belongs to the constructor and is re-run on every
        // reload; `const` and `static` are not per-instance state at all.
        if (field.IsReadOnly || field.IsConst || field.IsStatic || field.IsImplicitlyDeclared)
            return;

        if (!InheritsBehaviour(field.ContainingType))
            return;
        if (IsVisible(field.Type) || HasTransient(field))
            return;

        context.ReportDiagnostic(Diagnostic.Create(
            UnmarkedField,
            field.Locations.FirstOrDefault(),
            $"{field.ContainingType.Name}.{field.Name}",
            field.Type.ToDisplayString()));
    }

    static bool InheritsBehaviour(INamedTypeSymbol? type)
    {
        for (var t = type?.BaseType; t is not null; t = t.BaseType)
        {
            if (t.ToDisplayString() == BehaviourType)
                return true;
        }
        return false;
    }

    /// An enum is visible whatever its underlying type: the bag writes the
    /// member's name, so the width never reaches the wire.
    static bool IsVisible(ITypeSymbol type) =>
        type.TypeKind == TypeKind.Enum
        || VisiblePrimitives.Contains(type.SpecialType)
        || VisibleEngineTypes.Contains(type.ToDisplayString())
        || (type is INamedTypeSymbol { IsGenericType: true } named
            && named.OriginalDefinition.ToDisplayString() == AssetRefType);

    /// Directly, or through the property this field backs — `[Transient]` sits
    /// on the property, and its compiler-generated storage carries none of the
    /// author's attributes.
    static bool HasTransient(IFieldSymbol field) =>
        field.GetAttributes().Any(Marks)
        || (field.AssociatedSymbol?.GetAttributes().Any(Marks) ?? false);

    static bool Marks(AttributeData attribute) =>
        attribute.AttributeClass?.ToDisplayString() == TransientAttribute;
}
