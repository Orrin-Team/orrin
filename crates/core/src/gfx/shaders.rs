//! Every shader, translated for each backend, with the binding map a driver
//! needs to wire it.
//!
//! `build.rs` produces the table: one GLSL compile per manifest row to SPIR-V,
//! then MSL translated from that same SPIR-V. A Vulkan driver consumes the
//! SPIR-V and a Metal one the MSL, so the two cannot disagree about what a shader
//! says. `tests/shader_translation.rs` is what holds it honest without a GPU.
//!
//! The buffer-index assignment (a descriptor set is the argument buffer at its
//! own index, push constants sit at one fixed slot, vertex streams start at a
//! fixed base) is emitted from the build script alongside the table, because
//! the translator is what has to agree with it.
//!
//! With the `shader-compiler` feature, [`compile`] compiles a shader at runtime
//! through the same `orrin-shaderc` the build uses, against the embedded
//! [`SOURCES`] and the layout rebuilt from [`LAYOUT_WIDTHS`], so a built-in
//! shader compiled here comes out byte for byte as the table has it —
//! `tests/shader_compiler.rs` asserts exactly that. [`CompiledShader`] exists
//! without the feature, because an exported game loads compiled shaders and
//! compiles none.
//!
//! A driver reads a table entry and a shader handed to `Device::load_shader`
//! through one [`ShaderView`], so there is no second path for a loaded shader to
//! drift along.

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

/// A shader compiled at runtime: a [`ShaderBlob`] that owns its data. It has
/// no `path`, because its source was never a file under `shaders/`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CompiledShader {
    pub name: String,
    pub stage: Stage,
    pub defines: Vec<(String, String)>,
    /// Little-endian SPIR-V words.
    pub spirv: Vec<u8>,
    pub msl: String,
    pub bindings: CompiledBindings,
}

/// A [`BindingMap`] that owns its resources.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CompiledBindings {
    pub resources: Vec<Resource>,
    pub push_constant_bytes: u32,
    /// Only a compute shader has one.
    pub workgroup_size: Option<[u32; 3]>,
}

/// What a driver reads of a shader, borrowed from either kind.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShaderView<'a> {
    pub name: &'a str,
    pub spirv: &'a [u8],
    pub msl: &'a str,
    pub resources: &'a [Resource],
    pub push_constant_bytes: u32,
    pub workgroup_size: Option<[u32; 3]>,
}

impl ShaderBlob {
    pub(crate) fn view(&self) -> ShaderView<'_> {
        ShaderView {
            name: self.name,
            spirv: self.spirv,
            msl: self.msl,
            resources: self.bindings.resources,
            push_constant_bytes: self.bindings.push_constant_bytes,
            workgroup_size: self.bindings.workgroup_size,
        }
    }
}

impl CompiledShader {
    pub(crate) fn view(&self) -> ShaderView<'_> {
        ShaderView {
            name: &self.name,
            spirv: &self.spirv,
            msl: &self.msl,
            resources: &self.bindings.resources,
            push_constant_bytes: self.bindings.push_constant_bytes,
            workgroup_size: self.bindings.workgroup_size,
        }
    }
}

impl From<&ShaderBlob> for CompiledShader {
    fn from(blob: &ShaderBlob) -> Self {
        Self {
            name: blob.name.to_string(),
            stage: blob.stage,
            defines: blob
                .defines
                .iter()
                .map(|&(key, value)| (key.to_string(), value.to_string()))
                .collect(),
            spirv: blob.spirv.to_vec(),
            msl: blob.msl.to_string(),
            bindings: CompiledBindings {
                resources: blob.bindings.resources.to_vec(),
                push_constant_bytes: blob.bindings.push_constant_bytes,
                workgroup_size: blob.bindings.workgroup_size,
            },
        }
    }
}

#[cfg(feature = "shader-compiler")]
mod runtime;
#[cfg(feature = "shader-compiler")]
pub use runtime::compile;

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));
include!(concat!(env!("OUT_DIR"), "/shader_sources.rs"));

impl ShaderId {
    pub fn blob(self) -> &'static ShaderBlob {
        &SHADERS[self as usize]
    }
}
