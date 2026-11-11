//! Surface shaders in the renderer: the pipelines a loaded `.oshader` draws
//! with, the parameter buffer they bind, and how both change.
//!
//! **Twins.** A material read through a surface shader is drawn by a *twin* of
//! the built-in pipeline that would otherwise draw it: the same vertex shader,
//! formats, depth test, blend, cull mode and sample count, built by the pass
//! that owns the built-in from the very description it builds that from, with
//! the fragment shader swapped for the artifact's variant of it. Its layout is
//! the built-in's sets taken verbatim (`layout_prefix_from`) plus the set that
//! holds the `SurfaceParams` buffer (`gfx::surface::PARAMS_BINDING`), so the
//! descriptor sets the pass already bound stay bound and only that set is
//! added — the forward family's five, which the transparency pass and the
//! preview share, are still built once a frame. [`Twin`] lists every built-in
//! a twin exists for; the shadow passes have none, because a masked custom
//! material casts the shadow of its standard alpha (`gfx::surface`).
//!
//! **Built on first use.** Compiling the forward shader for a new attachment
//! shape is tens of milliseconds a pipeline, and a surface shader has thirteen
//! possible twins of which a scene uses two or three. So loading builds the
//! two every opaque material needs in the frame's current shape, which is also
//! what proves an artifact builds at all; the rest are built before the first
//! frame that draws with them records. A twin that fails there marks its
//! shader failed and every material using it draws as standard — a frame cannot
//! return an error, so it is logged naming the shader. Nothing panics on a
//! user's shader.
//!
//! **Swapped staged.** Replacing a shader builds every twin it has with the new
//! artifact before anything changes; on any failure those are destroyed and
//! the old twins keep drawing. On success the old pipelines and shader modules
//! are retired through the device's frame-delayed destroy, so a frame still in
//! flight finishes with what it recorded — the contract C# hot reload has.
//!
//! **Classes.** Each loaded shader is a surface of its own in the opaque
//! queue's pipeline classes (`cull::Surface::Custom`), by its slot here. A
//! material moves class when it is attached, detached, or its shader fails or
//! recovers, and every such change is followed by `CullPass::forget_classes`
//! — see `cull::Batching` for why that is the invariant the per-row cache
//! rests on.
//!
//! **The parameter buffer** is packed on the CPU only when a value or a table
//! changes ([`ParamBuffer`]); a frame writes the time into word 0 and uploads
//! the words like any other per-frame block — a few hundred bytes, against a
//! persistent buffer that would need a copy and a barrier the graph does not
//! own, or a host write under a frame still reading it. A session with no
//! surface shader loaded does none of this.

use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, BlockKind, BufferId, Device, PipelineId, ShaderModuleId,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::surface::{
    DRAWN_VARIANTS, DefaultTexture, ParamBuffer, SurfaceArtifact, SurfaceClock, SurfaceShaderError,
    SurfaceShaderHandle, check_artifact,
};

use super::cull::{Surface, Variant};

/// A built-in pipeline that a surface shader has a twin of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Twin {
    /// The forward pass in one of its attachment shapes.
    Forward {
        msaa: bool,
        subsurface: bool,
        masked: bool,
    },
    Prepass {
        masked: bool,
    },
    Transparency,
    Refraction,
    MaterialPreview {
        masked: bool,
    },
}

impl Twin {
    /// The built-in fragment shader the twin's stands in for.
    pub(super) fn variant(self) -> ShaderId {
        match self {
            // Multisampled, a cutout spends its samples through alpha to coverage
            // rather than testing, so it runs the plain shader.
            Twin::Forward {
                msaa: true,
                subsurface,
                ..
            } => {
                if subsurface {
                    ShaderId::ForwardSssFrag
                } else {
                    ShaderId::ForwardFrag
                }
            }
            Twin::Forward {
                msaa: false,
                subsurface,
                masked,
            } => match (subsurface, masked) {
                (false, false) => ShaderId::ForwardFrag,
                (false, true) => ShaderId::ForwardFragAlphaTest,
                (true, false) => ShaderId::ForwardSssFrag,
                (true, true) => ShaderId::ForwardSssFragAlphaTest,
            },
            Twin::Prepass { masked: false } => ShaderId::PrepassFrag,
            Twin::Prepass { masked: true } => ShaderId::PrepassFragMasked,
            Twin::Transparency => ShaderId::OitFrag,
            Twin::Refraction => ShaderId::RefractionFrag,
            Twin::MaterialPreview { masked: false } => ShaderId::ForwardFrag,
            Twin::MaterialPreview { masked: true } => ShaderId::ForwardFragAlphaTest,
        }
    }

    /// Whether the twin binds the forward family's set of the parameter
    /// buffer, rather than the prepass's or one of its own.
    fn forward_family(self) -> bool {
        matches!(
            self,
            Twin::Forward { .. } | Twin::Transparency | Twin::MaterialPreview { .. }
        )
    }
}

/// One loaded surface shader.
struct Entry {
    /// By position in [`DRAWN_VARIANTS`].
    modules: [ShaderModuleId; DRAWN_VARIANTS.len()],
    twins: Vec<(Twin, PipelineId)>,
    /// A twin failed to build on first use, so every material using this
    /// shader draws as standard until it is replaced.
    failed: bool,
}

impl Entry {
    fn twin(&self, twin: Twin) -> Option<PipelineId> {
        self.twins
            .iter()
            .find(|(built, _)| *built == twin)
            .map(|(_, pipeline)| *pipeline)
    }

    fn module(&self, variant: ShaderId) -> ShaderModuleId {
        let index = DRAWN_VARIANTS
            .iter()
            .position(|id| *id == variant)
            .expect("every twin's variant is a drawn one");
        self.modules[index]
    }
}

/// What a frame binds for its surface shaders, resolved before it records.
#[derive(Clone, Copy)]
pub(super) struct SurfaceFrame {
    /// This frame's `SurfaceParams`.
    pub params: BufferId,
    /// Set 5 of every forward, transparency and preview twin.
    forward: Option<BindGroupId>,
    /// Set 5 of every prepass twin.
    prepass: Option<BindGroupId>,
}

/// Every loaded surface shader, by slot, and the CPU half of their buffer.
pub(super) struct SurfaceShaders {
    entries: Vec<Entry>,
    pub book: ParamBuffer,
    clock: SurfaceClock,
    /// Whether the renderer has made the black default texture yet.
    black: bool,
}

/// The renderer's means of building one twin: which pass builds `Twin` with
/// the given fragment shader.
pub(super) type Builder<'a> = dyn Fn(Twin, ShaderModuleId) -> Result<PipelineId, String> + 'a;

impl SurfaceShaders {
    /// `white` and `normal` are the renderer's default texture slots; black is
    /// created when the first shader loads, by [`set_black`](Self::set_black).
    pub(super) fn new(white: u32, normal: u32) -> Self {
        Self {
            entries: Vec::new(),
            book: ParamBuffer::new(|texture| match texture {
                DefaultTexture::White | DefaultTexture::Black => white,
                DefaultTexture::Normal => normal,
            }),
            clock: SurfaceClock::default(),
            black: false,
        }
    }

    pub(super) fn has_black(&self) -> bool {
        self.black
    }

    pub(super) fn set_black(&mut self, index: u32) {
        self.black = true;
        self.book.set_default_texture(DefaultTexture::Black, index);
    }

    pub(super) fn loaded(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn pin_time(&mut self, seconds: Option<f32>) {
        self.clock.pin(seconds);
    }

    /// The slot `material` draws through, if a working shader reads it.
    pub(super) fn slot_of(&self, material: u32) -> Option<u32> {
        let shader = self
            .book
            .shader_of(crate::scene::MaterialHandle(material))?;
        (!self.entries[shader.0 as usize].failed).then_some(shader.0)
    }

    /// `built_in` with its surface swapped for the shader that reads
    /// `material`, if one does.
    pub(super) fn variant(&self, material: u32, built_in: Variant) -> Variant {
        match self.slot_of(material) {
            Some(slot) => Variant {
                masked: built_in.masked,
                surface: Surface::Custom(slot),
            },
            None => built_in,
        }
    }

    pub(super) fn twin(&self, slot: u32, twin: Twin) -> Option<PipelineId> {
        let entry = self.entries.get(slot as usize)?;
        if entry.failed {
            return None;
        }
        entry.twin(twin)
    }

    /// Load `artifact` and build `first` with it, or refuse it whole.
    pub(super) fn load(
        &mut self,
        device: &Device,
        artifact: &SurfaceArtifact,
        first: &[Twin],
        build: &Builder<'_>,
    ) -> Result<SurfaceShaderHandle, SurfaceShaderError> {
        check_artifact(artifact)?;
        if self.entries.len() >= super::cull::MAX_SURFACE_SHADERS {
            return Err(SurfaceShaderError::TooMany);
        }
        let modules = load_modules(device, artifact);
        let twins = build_all(device, &modules, first, build)?;
        self.entries.push(Entry {
            modules,
            twins,
            failed: false,
        });
        let handle = self.book.load(artifact.params.clone());
        debug_assert_eq!(handle.0 as usize + 1, self.entries.len());
        Ok(handle)
    }

    /// Build every twin `shader` has, plus `first`, from `artifact`; swap them
    /// in only if all of them built. Returns whether the shader had failed
    /// before, since a recovered one moves its materials' class.
    pub(super) fn replace(
        &mut self,
        device: &Device,
        shader: SurfaceShaderHandle,
        artifact: &SurfaceArtifact,
        first: &[Twin],
        build: &Builder<'_>,
    ) -> Result<bool, SurfaceShaderError> {
        let entry = self
            .entries
            .get(shader.0 as usize)
            .ok_or(SurfaceShaderError::UnknownShader(shader))?;
        check_artifact(artifact)?;
        let mut wanted: Vec<Twin> = entry.twins.iter().map(|(twin, _)| *twin).collect();
        for twin in first {
            if !wanted.contains(twin) {
                wanted.push(*twin);
            }
        }
        let modules = load_modules(device, artifact);
        let twins = build_all(device, &modules, &wanted, build)?;

        let entry = &mut self.entries[shader.0 as usize];
        let old_modules = std::mem::replace(&mut entry.modules, modules);
        let old_twins = std::mem::replace(&mut entry.twins, twins);
        let was_failed = std::mem::replace(&mut entry.failed, false);
        for (_, pipeline) in old_twins {
            device.destroy_pipeline(pipeline);
        }
        for module in old_modules {
            device.destroy_shader(module);
        }
        Ok(was_failed)
    }

    /// Build every twin a material needs that its shader lacks: `wants` says
    /// which twins a material is drawn by in this frame's shape. Returns
    /// whether a shader failed, which moves its materials back to the standard
    /// classes.
    pub(super) fn ensure(
        &mut self,
        wants: impl Fn(u32) -> [Option<Twin>; 3],
        build: &Builder<'_>,
    ) -> bool {
        let mut failed = false;
        let Self { entries, book, .. } = self;
        for (material, shader) in book.attached() {
            for twin in wants(material.0).into_iter().flatten() {
                let slot = shader.0;
                let entry = &mut entries[slot as usize];
                if entry.failed || entry.twin(twin).is_some() {
                    continue;
                }
                match build(twin, entry.module(twin.variant())) {
                    Ok(pipeline) => entry.twins.push((twin, pipeline)),
                    Err(error) => {
                        tracing::warn!(
                            "surface shader {slot} did not build its {twin:?} pipeline, so its \
                             materials draw as standard until it is replaced: {error}"
                        );
                        entry.failed = true;
                        failed = true;
                    }
                }
            }
        }
        failed
    }

    /// Advance the clock, upload the buffer and build this frame's two shared
    /// sets — or nothing at all while no shader is loaded.
    pub(super) fn begin_frame(&mut self, device: &Device, dt: f32) -> Option<SurfaceFrame> {
        if self.entries.is_empty() {
            return None;
        }
        let time = self.clock.tick(dt);
        self.book.set_time(time);
        let params =
            device.upload_block(BlockKind::Storage, bytemuck::cast_slice(self.book.words()));
        // Any twin of each family will do: the set holds the one binding in
        // every one of them, so the layouts are identical and the set binds to
        // all of them.
        let find = |family: fn(Twin) -> bool| {
            self.entries
                .iter()
                .filter(|entry| !entry.failed)
                .flat_map(|entry| &entry.twins)
                .find(|(twin, _)| family(*twin))
                .map(|(_, pipeline)| *pipeline)
        };
        let group = |pipeline| params_group(device, pipeline, params, &[]);
        Some(SurfaceFrame {
            params,
            forward: find(Twin::forward_family).map(group),
            prepass: find(|twin| matches!(twin, Twin::Prepass { .. })).map(group),
        })
    }
}

/// Set 5 of a twin: the parameter buffer at its binding, after whatever the
/// built-in's own set 5 already holds.
pub(super) fn params_group(
    device: &Device,
    pipeline: PipelineId,
    params: BufferId,
    before: &[Binding],
) -> BindGroupId {
    let (set, binding) = crate::gfx::surface::PARAMS_BINDING;
    let mut bindings = before.to_vec();
    bindings.push(Binding {
        binding,
        resource: BindingResource::StorageBuffer(params),
    });
    device.create_transient_bind_group(pipeline, set, &bindings)
}

/// The pipeline and set 5 a run of `material` draws with as `twin`, if a
/// working shader reads it and the frame has its set.
#[derive(Clone, Copy)]
pub(super) struct Surfaces<'a> {
    pub shaders: &'a SurfaceShaders,
    pub frame: Option<SurfaceFrame>,
}

impl Surfaces<'_> {
    /// For a class the cull named: its twin and the set it binds.
    pub(super) fn for_slot(&self, slot: u32, twin: Twin) -> Option<(PipelineId, BindGroupId)> {
        Some((self.shaders.twin(slot, twin)?, self.group(twin)?))
    }

    /// For a run of the blended or refractive queue, which is drawn item by
    /// item rather than by class. `None` draws it as standard.
    pub(super) fn for_material(&self, material: u32, twin: Twin) -> Option<PipelineId> {
        self.frame?;
        self.shaders.twin(self.shaders.slot_of(material)?, twin)
    }

    /// Set 5 of a forward-family or prepass twin. A refraction twin builds its
    /// own, since its set 5 holds the scene's pyramid too.
    pub(super) fn group(&self, twin: Twin) -> Option<BindGroupId> {
        let frame = self.frame?;
        if twin.forward_family() {
            frame.forward
        } else {
            frame.prepass
        }
    }

    pub(super) fn params(&self) -> Option<BufferId> {
        Some(self.frame?.params)
    }
}

fn load_modules(device: &Device, artifact: &SurfaceArtifact) -> [ShaderModuleId; 8] {
    DRAWN_VARIANTS.map(|id| {
        let variant = artifact
            .variants
            .iter()
            .find(|variant| variant.builtin() == Some(id))
            .expect("check_artifact found every drawn variant");
        device.load_shader(&variant.shader)
    })
}

/// Every twin in `twins`, or none of them and the error: what was built before
/// a failure is destroyed with the modules, so a refused artifact leaves
/// nothing behind.
fn build_all(
    device: &Device,
    modules: &[ShaderModuleId; 8],
    twins: &[Twin],
    build: &Builder<'_>,
) -> Result<Vec<(Twin, PipelineId)>, SurfaceShaderError> {
    let mut built = Vec::with_capacity(twins.len());
    for &twin in twins {
        let index = DRAWN_VARIANTS
            .iter()
            .position(|id| *id == twin.variant())
            .expect("every twin's variant is a drawn one");
        match build(twin, modules[index]) {
            Ok(pipeline) => built.push((twin, pipeline)),
            Err(error) => {
                for (_, pipeline) in built {
                    device.destroy_pipeline(pipeline);
                }
                for module in modules {
                    device.destroy_shader(*module);
                }
                return Err(SurfaceShaderError::Pipeline(error));
            }
        }
    }
    Ok(built)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A twin's fragment shader must be one the artifact carries, and the
    /// shapes that differ only in how they cut must not share one where the
    /// built-in pipelines do not.
    #[test]
    fn every_twin_stands_in_for_a_drawn_variant() {
        let mut twins = vec![
            Twin::Prepass { masked: false },
            Twin::Prepass { masked: true },
            Twin::Transparency,
            Twin::Refraction,
            Twin::MaterialPreview { masked: false },
            Twin::MaterialPreview { masked: true },
        ];
        for msaa in [false, true] {
            for subsurface in [false, true] {
                for masked in [false, true] {
                    twins.push(Twin::Forward {
                        msaa,
                        subsurface,
                        masked,
                    });
                }
            }
        }
        let mut used: Vec<ShaderId> = twins.iter().map(|twin| twin.variant()).collect();
        for variant in &used {
            assert!(DRAWN_VARIANTS.contains(variant), "{variant:?}");
        }
        used.sort_by_key(|id| *id as usize);
        used.dedup();
        assert_eq!(
            used.len(),
            DRAWN_VARIANTS.len(),
            "a drawn variant is unused"
        );
        assert_eq!(
            Twin::Forward {
                msaa: true,
                subsurface: false,
                masked: true
            }
            .variant(),
            ShaderId::ForwardFrag,
            "multisampled, a cutout is cut by coverage, not by a test"
        );
    }
}
