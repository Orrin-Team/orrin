use std::fmt;

use super::params::{DefaultTexture, Ignored, ParamTable, ParamValue};
use super::{FRAME_WORDS, SurfaceArtifact};
use crate::gfx::shaders::ShaderId;
use crate::scene::MaterialHandle;

/// A surface shader a backend has loaded. It keeps its meaning across
/// [`replace_surface_shader`](crate::gfx::RenderBackend::replace_surface_shader).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SurfaceShaderHandle(pub u32);

/// The built-in fragment shaders whose pipelines a renderer draws a surface
/// shader's materials with, and so the variants an artifact must carry.
pub const DRAWN_VARIANTS: [ShaderId; 8] = [
    ShaderId::ForwardFrag,
    ShaderId::ForwardFragAlphaTest,
    ShaderId::ForwardSssFrag,
    ShaderId::ForwardSssFragAlphaTest,
    ShaderId::OitFrag,
    ShaderId::PrepassFrag,
    ShaderId::PrepassFragMasked,
    ShaderId::RefractionFrag,
];

/// Why a backend refused a surface shader or an attachment.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SurfaceShaderError {
    /// Compiled against other shader sources than this engine's.
    ForeignBuild,
    /// The artifact has no compiled variant for this built-in shader.
    MissingVariant(&'static str),
    /// [`MAX_SURFACE_SHADERS`](crate::gfx::vulkan::MAX_SURFACE_SHADERS) are
    /// already loaded.
    TooMany,
    UnknownShader(SurfaceShaderHandle),
    UnknownMaterial(MaterialHandle),
    /// The driver refused a pipeline built from it, in its own words.
    Pipeline(String),
}

impl fmt::Display for SurfaceShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignBuild => f.write_str(
                "the surface shader was compiled by another build of the engine; recompile it",
            ),
            Self::MissingVariant(name) => write!(
                f,
                "the surface shader has no `{name}` variant; recompile it with this engine"
            ),
            Self::TooMany => write!(
                f,
                "{} surface shaders are already loaded, which is as many as the renderer can \
                 draw; unload one first",
                crate::gfx::vulkan::MAX_SURFACE_SHADERS
            ),
            Self::UnknownShader(handle) => {
                write!(f, "surface shader {} was never loaded", handle.0)
            }
            Self::UnknownMaterial(handle) => write!(f, "material {} was never loaded", handle.0),
            Self::Pipeline(message) => {
                write!(f, "the surface shader did not build a pipeline: {message}")
            }
        }
    }
}

impl std::error::Error for SurfaceShaderError {}

/// What every backend checks of an artifact before it builds anything.
pub fn check_artifact(artifact: &SurfaceArtifact) -> Result<(), SurfaceShaderError> {
    if !artifact.matches_this_build() {
        return Err(SurfaceShaderError::ForeignBuild);
    }
    match DRAWN_VARIANTS
        .iter()
        .find(|id| !artifact.variants.iter().any(|v| v.builtin() == Some(**id)))
    {
        Some(missing) => Err(SurfaceShaderError::MissingVariant(missing.blob().name)),
        None => Ok(()),
    }
}

/// One material's surface shader, the values it was given by name, and where
/// its block starts.
#[derive(Clone, Debug)]
struct Attached {
    shader: SurfaceShaderHandle,
    values: Vec<(String, ParamValue)>,
    base: u32,
}

/// The CPU half of the `SurfaceParams` buffer: each loaded shader's parameter
/// table, each material's shader and *named* values, and the words packed from
/// them.
///
/// Values are kept by name rather than as packed words because a replaced
/// shader may lay its parameters out differently, and the only way to put a
/// material's colour back where the new table wants it is to pack it again from
/// what it was called. Every change repacks every block, in material order:
/// the blocks are a few words each and a change is an edit, so compacting on
/// every change costs nothing worth a free list and leaves no holes behind a
/// detached material.
pub struct ParamBuffer {
    tables: Vec<ParamTable>,
    attached: Vec<Option<Attached>>,
    words: Vec<[u32; 4]>,
    /// The bindless index of each [`DefaultTexture`], in declaration order.
    defaults: [u32; 3],
}

impl ParamBuffer {
    /// `defaults` says where the white, black and flat-normal textures live,
    /// which only the backend knows.
    pub fn new(defaults: impl Fn(DefaultTexture) -> u32) -> Self {
        Self {
            tables: Vec::new(),
            attached: Vec::new(),
            words: vec![[0; 4]; FRAME_WORDS as usize],
            defaults: [
                DefaultTexture::White,
                DefaultTexture::Black,
                DefaultTexture::Normal,
            ]
            .map(defaults),
        }
    }

    /// Point `texture`'s default at another index, for a backend that creates
    /// one only once a shader needs it. Repacks, since blocks hold the index.
    pub fn set_default_texture(&mut self, texture: DefaultTexture, index: u32) {
        self.defaults[texture as usize] = index;
        self.repack();
    }

    /// How many shaders are loaded. While none is, a frame has nothing to do
    /// for this feature at all.
    pub fn loaded(&self) -> usize {
        self.tables.len()
    }

    pub fn load(&mut self, table: ParamTable) -> SurfaceShaderHandle {
        self.tables.push(table);
        SurfaceShaderHandle(self.tables.len() as u32 - 1)
    }

    pub fn table(&self, shader: SurfaceShaderHandle) -> Option<&ParamTable> {
        self.tables.get(shader.0 as usize)
    }

    /// Give `shader` a new table and repack every material, returning what each
    /// material using it had that the new table ignores.
    pub fn replace(
        &mut self,
        shader: SurfaceShaderHandle,
        table: ParamTable,
    ) -> Result<Vec<(MaterialHandle, Ignored)>, SurfaceShaderError> {
        let slot = self
            .tables
            .get_mut(shader.0 as usize)
            .ok_or(SurfaceShaderError::UnknownShader(shader))?;
        *slot = table;
        Ok(self
            .repack()
            .into_iter()
            .filter(|(material, _)| self.shader_of(*material) == Some(shader))
            .collect())
    }

    /// Read `material` through `surface`'s shader with its values, or through
    /// none. Returns what the shader ignores of the values.
    pub fn attach(
        &mut self,
        material: MaterialHandle,
        surface: Option<(SurfaceShaderHandle, &[(String, ParamValue)])>,
    ) -> Result<Vec<Ignored>, SurfaceShaderError> {
        let index = material.0 as usize;
        match surface {
            Some((shader, values)) => {
                if self.table(shader).is_none() {
                    return Err(SurfaceShaderError::UnknownShader(shader));
                }
                if self.attached.len() <= index {
                    self.attached.resize(index + 1, None);
                }
                self.attached[index] = Some(Attached {
                    shader,
                    values: values.to_vec(),
                    base: 0,
                });
            }
            None => {
                if let Some(slot) = self.attached.get_mut(index) {
                    *slot = None;
                }
            }
        }
        Ok(self
            .repack()
            .into_iter()
            .filter(|(m, _)| *m == material)
            .map(|(_, ignored)| ignored)
            .collect())
    }

    pub fn shader_of(&self, material: MaterialHandle) -> Option<SurfaceShaderHandle> {
        Some(self.attached.get(material.0 as usize)?.as_ref()?.shader)
    }

    /// The word `material`'s block starts at, which is its
    /// `GpuMaterial::surface.x`.
    pub fn base_of(&self, material: MaterialHandle) -> Option<u32> {
        Some(self.attached.get(material.0 as usize)?.as_ref()?.base)
    }

    /// Every material read through a shader, and which.
    pub fn attached(&self) -> impl Iterator<Item = (MaterialHandle, SurfaceShaderHandle)> + '_ {
        self.attached
            .iter()
            .enumerate()
            .filter_map(|(index, attached)| {
                Some((MaterialHandle(index as u32), attached.as_ref()?.shader))
            })
    }

    /// Every material read through `shader`.
    pub fn materials_using(
        &self,
        shader: SurfaceShaderHandle,
    ) -> impl Iterator<Item = MaterialHandle> + '_ {
        self.attached
            .iter()
            .enumerate()
            .filter(move |(_, attached)| attached.as_ref().is_some_and(|a| a.shader == shader))
            .map(|(index, _)| MaterialHandle(index as u32))
    }

    /// The whole buffer: the frame's words, then every block.
    pub fn words(&self) -> &[[u32; 4]] {
        &self.words
    }

    pub fn set_time(&mut self, seconds: f32) {
        self.words[0][0] = seconds.to_bits();
    }

    fn repack(&mut self) -> Vec<(MaterialHandle, Ignored)> {
        self.words.truncate(FRAME_WORDS as usize);
        let defaults = self.defaults;
        let mut ignored = Vec::new();
        for (index, attached) in self.attached.iter_mut().enumerate() {
            let Some(attached) = attached else { continue };
            let table = &self.tables[attached.shader.0 as usize];
            let (block, skipped) = table.words(
                attached
                    .values
                    .iter()
                    .map(|(name, value)| (name.as_str(), *value)),
                |texture| defaults[texture as usize],
            );
            attached.base = self.words.len() as u32;
            self.words.extend(block);
            ignored.extend(
                skipped
                    .into_iter()
                    .map(|i| (MaterialHandle(index as u32), i)),
            );
        }
        ignored
    }
}

/// What `i.time` reads: seconds the renderer has drawn while a surface shader
/// was loaded, from the `dt` each frame is handed — which advances in the
/// editor as well as in play, so a shader animates while it is authored — and
/// wrapped at [`PERIOD`](Self::PERIOD), as Godot wraps its `TIME`, so a float
/// still resolves a sixtieth of a second after a long session. A capture pins
/// it, so its pixels do not depend on how many frames it warmed up for.
#[derive(Clone, Copy, Debug, Default)]
pub struct SurfaceClock {
    seconds: f64,
    pinned: Option<f32>,
}

impl SurfaceClock {
    pub const PERIOD: f64 = 3600.0;

    /// Advance by `dt` and return the time this frame reads.
    pub fn tick(&mut self, dt: f32) -> f32 {
        self.seconds = (self.seconds + f64::from(dt.max(0.0))) % Self::PERIOD;
        self.pinned.unwrap_or(self.seconds as f32)
    }

    pub fn pin(&mut self, seconds: Option<f32>) {
        self.pinned = seconds;
    }
}
