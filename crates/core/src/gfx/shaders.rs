//! Every shader, translated for each backend, with the binding map a driver
//! needs to wire it.
//!
//! `build.rs` produces the table: one GLSL compile per manifest row to SPIR-V,
//! then MSL translated from that same SPIR-V. A Vulkan driver consumes the
//! SPIR-V and a Metal one the MSL, so the two cannot disagree about what a shader
//! says. Nothing reads this yet; it is the input the RHI drivers are built
//! against, and `tests/shader_translation.rs` is what holds it honest without a
//! GPU.
//!
//! The buffer-index assignment (a descriptor set is the argument buffer at its
//! own index, push constants sit at one fixed slot, vertex streams start at a
//! fixed base) is emitted from the build script alongside the table, because
//! the translator is what has to agree with it.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResourceKind {
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    Sampler,
    StorageImage,
    CombinedImageSampler,
}

/// One binding in one descriptor set. `count` is the array length, one for a
/// plain resource.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Resource {
    pub set: u32,
    pub binding: u32,
    pub kind: ResourceKind,
    pub count: u32,
    /// The resource's `[[id]]` in its set's argument buffer. A combined image
    /// sampler's samplers follow its `count` textures.
    pub index: u32,
}

#[derive(Debug)]
pub struct BindingMap {
    pub resources: &'static [Resource],
    pub push_constant_bytes: u32,
    /// Only a compute shader has one.
    pub workgroup_size: Option<[u32; 3]>,
}

#[derive(Debug)]
pub struct ShaderBlob {
    pub name: &'static str,
    /// Relative to `shaders/`.
    pub path: &'static str,
    pub stage: Stage,
    pub defines: &'static [(&'static str, &'static str)],
    /// Little-endian SPIR-V words.
    pub spirv: &'static [u8],
    pub msl: &'static str,
    pub bindings: BindingMap,
}

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

impl ShaderId {
    pub fn blob(self) -> &'static ShaderBlob {
        &SHADERS[self as usize]
    }
}
