using System.Runtime.InteropServices;

namespace Orrin;

/// <summary>
/// What an entity draws: a mesh, and the material it is shaded with. Read and
/// write it with <see cref="World.GetComponent{T}"/>,
/// <see cref="World.SetComponent{T}"/> and <see cref="World.AddComponent{T}"/>.
/// Both are asset references, so a field holding one survives the file being
/// moved; assign them from such a field, from <see cref="Assets.Find{T}"/>, or
/// by dropping an asset on the inspector's slot.
/// A mesh or material loaded with no asset behind it — a model's own, or one
/// made at run time — reads as the null reference, and setting that null back
/// keeps it, so such a renderer can still be read, changed and written. Any
/// other null is refused, as is adding a renderer with one, and the engine
/// logs why.
/// </summary>
// Layout must match Rust CMeshRenderer: two asset ids' 16 bytes.
[StructLayout(LayoutKind.Sequential)]
public struct MeshRenderer
{
    AssetBytes _mesh;
    AssetBytes _material;

    /// <summary>The geometry drawn.</summary>
    public AssetRef<Mesh> Mesh
    {
        readonly get => new(_mesh.ToGuid());
        set => _mesh = AssetBytes.Of(value.Id);
    }

    /// <summary>
    /// The material the mesh is shaded with: a <c>.omat</c> asset, or one the
    /// engine builds. A material is shared, so editing the asset changes every
    /// renderer that draws with it; assigning another changes only this one.
    /// </summary>
    public AssetRef<Material> Material
    {
        readonly get => new(_material.ToGuid());
        set => _material = AssetBytes.Of(value.Id);
    }
}
