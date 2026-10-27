//! The `<file>.meta` beside every source asset: its identity and its import
//! settings, in TOML a person can read in a diff.
//!
//! ```toml
//! id = "5f0c…"
//!
//! [import]
//! usage = "color"
//! compress = true
//! max_size = 0
//! ```
//!
//! **Written once, then only read.** A sidecar is created with a fresh id and
//! the kind's default settings the first time a source is seen, and never
//! rewritten after that: it is committed, so a rewrite would be a diff nobody
//! made, and one that cannot be read is refused rather than replaced, because
//! replacing it would give the asset a new identity and orphan every scene that
//! names the old one. The one exception is a folder's tint, which the editor
//! writes because the user asked it to.
//!
//! **Settings are checked, not tolerated.** An unknown key under `[import]` is
//! an error naming the key and the ones that exist; a typo that silently fell
//! back to a default would be a texture imported wrong with nothing to say why.
//! A missing key takes its default, which is what makes a prefab's sidecar —
//! `id` and nothing else — valid here too.
//!
//! The settings, per kind:
//!
//! - texture: `usage` (`"color"`, decoded from sRGB to linear on sample;
//!   `"normal"`, a tangent-space normal map; or `"data"`, read as stored),
//!   `compress` (block-compressed where the GPU samples it, or stored as
//!   RGBA8 — for pixel art and UI, where BC's 4x4 blocks show) and `max_size`
//!   (the longest side in texels the source is scaled down to fit; `0` keeps
//!   it as it is).
//! - mesh: `scale`, multiplying every position, for a file not authored in
//!   metres.
//! - folder: `tint`, the colour its icon is drawn in, as `"#rrggbb"`.

use std::path::{Path, PathBuf};

use orrin_registry::AssetId;

/// What a source file is, decided by its extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AssetKind {
    Texture,
    Mesh,
    Prefab,
    Material,
    Folder,
}

impl AssetKind {
    /// The kind a file is imported as, or `None` for a file that is not an
    /// asset — a glTF's `.bin`, a sidecar, a readme. Never [`Folder`](Self::Folder).
    pub fn of_path(path: &str) -> Option<Self> {
        let file = path.rsplit('/').next().unwrap_or(path);
        let (stem, extension) = file.rsplit_once('.')?;
        if stem.is_empty() {
            return None;
        }
        match extension.to_ascii_lowercase().as_str() {
            "png" | "jpg" | "jpeg" => Some(Self::Texture),
            "gltf" | "glb" => Some(Self::Mesh),
            "fprefab" => Some(Self::Prefab),
            super::material::EXTENSION => Some(Self::Material),
            _ => None,
        }
    }

    /// The word an error or the editor uses for it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Texture => "texture",
            Self::Mesh => "mesh",
            Self::Prefab => "prefab",
            Self::Material => "material",
            Self::Folder => "folder",
        }
    }
}

/// What a texture's texels mean, which decides how its mips are filtered and
/// which block format it compresses to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Usage {
    /// sRGB-encoded colour, filtered in linear light.
    Color,
    /// A tangent-space normal map: X and Y in red and green, Z rebuilt from
    /// them on sample, so only the two are kept once compressed.
    Normal,
    /// Anything else stored linearly: roughness, metalness, height, masks.
    Data,
}

impl Usage {
    /// The word a sidecar spells it with.
    pub fn name(self) -> &'static str {
        match self {
            Self::Color => "color",
            Self::Normal => "normal",
            Self::Data => "data",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        [Self::Color, Self::Normal, Self::Data]
            .into_iter()
            .find(|usage| usage.name() == name)
    }

    /// Whether texels are sRGB-encoded and decoded to linear on sample.
    pub fn is_srgb(self) -> bool {
        self == Self::Color
    }
}

/// How a texture is imported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureSettings {
    pub usage: Usage,
    pub compress: bool,
    pub max_size: u32,
}

/// How a mesh is imported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshSettings {
    pub scale: f32,
}

/// A source's import settings, as its sidecar states them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Settings {
    Texture(TextureSettings),
    Mesh(MeshSettings),
    /// A kind with nothing to set: a prefab or a material is read as it is.
    None,
}

impl Settings {
    /// What a new sidecar is written with.
    ///
    /// A texture whose name says it holds data — a normal, roughness or height
    /// map — starts with that usage. The guess is made once, here, and the
    /// sidecar is what is read from then on, so a wrong guess is corrected by
    /// editing one line rather than by renaming the file.
    pub fn defaults_for(kind: AssetKind, rel: &str) -> Self {
        match kind {
            AssetKind::Texture => Self::Texture(TextureSettings {
                usage: usage_of(rel),
                compress: true,
                max_size: 0,
            }),
            AssetKind::Mesh => Self::Mesh(MeshSettings { scale: 1.0 }),
            AssetKind::Prefab | AssetKind::Material | AssetKind::Folder => Self::None,
        }
    }

    /// The settings as one deterministic string: what the cache key hashes, so
    /// two spellings of the same settings in two sidecars share one artifact.
    pub fn canonical(&self) -> String {
        match self {
            Self::Texture(texture) => format!(
                "usage={};compress={};max_size={}",
                texture.usage.name(),
                texture.compress,
                texture.max_size
            ),
            Self::Mesh(mesh) => format!("scale={:?}", mesh.scale),
            Self::None => String::new(),
        }
    }

    /// The keys `[import]` may hold, in the order a new sidecar writes them.
    fn keys(&self) -> &'static [&'static str] {
        match self {
            Self::Texture(_) => &["usage", "compress", "max_size"],
            Self::Mesh(_) => &["scale"],
            Self::None => &[],
        }
    }

    /// Set `key` from a sidecar's value, or say why it cannot be.
    fn set(&mut self, key: &str, value: &toml::Value) -> Result<(), String> {
        let wrong = |expected: &str| format!("`{key}` must be {expected}, not {value}");
        match (self, key) {
            (Self::Texture(texture), "usage") => {
                texture.usage = value
                    .as_str()
                    .and_then(Usage::parse)
                    .ok_or_else(|| wrong("\"color\", \"normal\" or \"data\""))?;
            }
            (Self::Texture(texture), "compress") => {
                texture.compress = value.as_bool().ok_or_else(|| wrong("true or false"))?;
            }
            (Self::Texture(texture), "max_size") => {
                texture.max_size = value
                    .as_integer()
                    .and_then(|size| u32::try_from(size).ok())
                    .ok_or_else(|| wrong("a whole number of texels, 0 for no limit"))?;
            }
            (Self::Mesh(mesh), "scale") => {
                let scale = match value {
                    toml::Value::Float(scale) => *scale as f32,
                    toml::Value::Integer(scale) => *scale as f32,
                    _ => return Err(wrong("a number")),
                };
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(wrong("a positive number"));
                }
                mesh.scale = scale;
            }
            _ => unreachable!("checked against `keys`"),
        }
        Ok(())
    }

    /// `[import]` as a new sidecar writes it, or nothing for a kind without
    /// settings.
    fn import_table(&self) -> String {
        match self {
            Self::Texture(texture) => format!(
                "\n[import]\nusage = \"{}\"\ncompress = {}\nmax_size = {}\n",
                texture.usage.name(),
                texture.compress,
                texture.max_size
            ),
            Self::Mesh(mesh) => format!("\n[import]\nscale = {:?}\n", mesh.scale),
            Self::None => String::new(),
        }
    }
}

/// What a texture's name says it holds. Matched on the lower-cased file name,
/// by the words texture sets use.
fn usage_of(rel: &str) -> Usage {
    let file = rel.rsplit('/').next().unwrap_or(rel).to_ascii_lowercase();
    const NORMAL: [&str; 2] = ["normal", "nrm"];
    const DATA: [&str; 6] = [
        "rough",
        "height",
        "displacement",
        "metal",
        "occlusion",
        "bump",
    ];
    let named = |words: &[&str]| words.iter().any(|word| file.contains(word));
    if named(&NORMAL) {
        Usage::Normal
    } else if named(&DATA) {
        Usage::Data
    } else {
        Usage::Color
    }
}

/// An sRGB colour a folder's icon is drawn in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tint(pub [u8; 3]);

impl Tint {
    /// `"#rrggbb"`, the form a sidecar stores.
    pub fn parse(text: &str) -> Option<Self> {
        let hex = text.strip_prefix('#')?;
        if hex.len() != 6 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let channel = |at: usize| u8::from_str_radix(&hex[at..at + 2], 16).ok();
        Some(Self([channel(0)?, channel(2)?, channel(4)?]))
    }

    pub fn to_hex(self) -> String {
        let [r, g, b] = self.0;
        format!("#{r:02x}{g:02x}{b:02x}")
    }
}

/// One sidecar, read.
#[derive(Clone, Debug, PartialEq)]
pub struct Sidecar {
    pub id: AssetId,
    pub settings: Settings,
    pub tint: Option<Tint>,
}

/// Where `source`'s sidecar lives: beside it, with `.meta` appended to the
/// whole name, so `rock.png` and `rock.jpg` cannot share one.
pub fn path_of(source: &Path) -> PathBuf {
    let mut meta = source.as_os_str().to_owned();
    meta.push(".meta");
    PathBuf::from(meta)
}

/// Read the sidecar of the source at `source` (`rel` from the assets
/// directory, for messages), writing one first if there is none.
pub fn read_or_create(source: &Path, rel: &str, kind: AssetKind) -> Result<Sidecar, String> {
    let path = path_of(source);
    if let Some(text) = read(&path, rel)? {
        return parse(&text, rel, kind);
    }
    let settings = Settings::defaults_for(kind, rel);
    let id = AssetId::new();
    let text = format!("id = \"{id}\"\n{}", settings.import_table());
    // `create_new`, so a sidecar another process wrote since the read above —
    // a second editor on the same project — is read rather than replaced.
    let created = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .and_then(|mut file| std::io::Write::write_all(&mut file, text.as_bytes()));
    match created {
        Ok(()) => Ok(Sidecar {
            id,
            settings,
            tint: None,
        }),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let text = read(&path, rel)?.unwrap_or_default();
            parse(&text, rel, kind)
        }
        Err(error) => Err(format!(
            "{rel} has no sidecar, and writing {rel}.meta failed: {error}"
        )),
    }
}

/// A folder's sidecar, which exists only once a tint has been set.
pub fn read_folder(dir: &Path, rel: &str) -> Result<Option<Sidecar>, String> {
    match read(&path_of(dir), rel)? {
        Some(text) => parse(&text, rel, AssetKind::Folder).map(Some),
        None => Ok(None),
    }
}

/// Set or clear a folder's tint, keeping the id it already has. Returns the
/// folder's id.
pub fn write_folder_tint(dir: &Path, rel: &str, tint: Option<Tint>) -> Result<AssetId, String> {
    let id = read_folder(dir, rel)?.map_or_else(AssetId::new, |sidecar| sidecar.id);
    let mut text = format!("id = \"{id}\"\n");
    if let Some(tint) = tint {
        text.push_str(&format!("tint = \"{}\"\n", tint.to_hex()));
    }
    std::fs::write(path_of(dir), text).map_err(|error| format!("{rel}.meta: {error}"))?;
    Ok(id)
}

/// The sidecar's text, or `None` when there is none.
fn read(path: &Path, rel: &str) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("{rel}.meta: {error}")),
    }
}

fn parse(text: &str, rel: &str, kind: AssetKind) -> Result<Sidecar, String> {
    let table: toml::Table =
        toml::from_str(text).map_err(|error| format!("{rel}.meta: {}", error.message()))?;
    let id = table
        .get("id")
        .and_then(toml::Value::as_str)
        .and_then(|id| id.parse::<AssetId>().ok())
        .filter(|id| !id.is_nil())
        .ok_or_else(|| {
            format!(
                "{rel}.meta has no valid `id`. Restore it from version control: deleting it \
                 gives the {} a new id, and everything that references the old one loses it",
                kind.label()
            )
        })?;

    let mut sidecar = Sidecar {
        id,
        settings: Settings::defaults_for(kind, rel),
        tint: None,
    };
    for (key, value) in &table {
        match (key.as_str(), kind) {
            ("id", _) => {}
            ("tint", AssetKind::Folder) => {
                let tint = value.as_str().and_then(Tint::parse).ok_or_else(|| {
                    format!("{rel}.meta: `tint` must be a colour written \"#rrggbb\", not {value}")
                })?;
                sidecar.tint = Some(tint);
            }
            ("import", kind) if kind != AssetKind::Folder => {
                let import = value
                    .as_table()
                    .ok_or_else(|| format!("{rel}.meta: `import` must be a table"))?;
                for (key, value) in import {
                    if !sidecar.settings.keys().contains(&key.as_str()) {
                        return Err(unknown_setting(rel, kind, key, sidecar.settings.keys()));
                    }
                    sidecar
                        .settings
                        .set(key, value)
                        .map_err(|error| format!("{rel}.meta: {error}"))?;
                }
            }
            (key, _) => {
                return Err(format!(
                    "{rel}.meta: `{key}` is not something a {} sidecar holds",
                    kind.label()
                ));
            }
        }
    }
    Ok(sidecar)
}

fn unknown_setting(rel: &str, kind: AssetKind, key: &str, keys: &[&str]) -> String {
    if keys.is_empty() {
        return format!(
            "{rel}.meta: `{key}` is not a {} import setting; a {} has none",
            kind.label(),
            kind.label()
        );
    }
    format!(
        "{rel}.meta: `{key}` is not a {} import setting; the settings are {}",
        kind.label(),
        keys.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_db::tests::Dir;

    #[test]
    fn a_file_is_an_asset_by_its_extension() {
        assert_eq!(AssetKind::of_path("rocks/a.PNG"), Some(AssetKind::Texture));
        assert_eq!(AssetKind::of_path("a.jpeg"), Some(AssetKind::Texture));
        assert_eq!(AssetKind::of_path("a.jpg"), Some(AssetKind::Texture));
        assert_eq!(AssetKind::of_path("m/tree.gltf"), Some(AssetKind::Mesh));
        assert_eq!(AssetKind::of_path("tree.glb"), Some(AssetKind::Mesh));
        assert_eq!(AssetKind::of_path("door.fprefab"), Some(AssetKind::Prefab));
        assert_eq!(AssetKind::of_path("brick.FMAT"), Some(AssetKind::Material));
        assert_eq!(AssetKind::of_path("tree.bin"), None);
        assert_eq!(AssetKind::of_path("a.png.meta"), None);
        assert_eq!(AssetKind::of_path("png"), None);
    }

    /// The id is the asset's identity for as long as the file exists, so the
    /// first read has to leave it where the next one finds it — and the second
    /// read must not touch a file that is committed.
    #[test]
    fn a_source_without_a_sidecar_gets_one_with_its_defaults_and_keeps_it() {
        let dir = Dir::new("sidecar-create");
        let source = dir.write("rock.png", b"not read here");

        let first = read_or_create(&source, "rock.png", AssetKind::Texture).unwrap();
        let text = std::fs::read_to_string(dir.path().join("rock.png.meta")).unwrap();
        assert_eq!(
            text,
            format!(
                "id = \"{}\"\n\n[import]\nusage = \"color\"\ncompress = true\nmax_size = 0\n",
                first.id
            )
        );
        assert_eq!(
            first.settings,
            Settings::Texture(TextureSettings {
                usage: Usage::Color,
                compress: true,
                max_size: 0
            })
        );

        let second = read_or_create(&source, "rock.png", AssetKind::Texture).unwrap();
        assert_eq!(second, first);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("rock.png.meta")).unwrap(),
            text
        );
    }

    /// The guess a new sidecar starts from, by the words texture sets use: a
    /// normal map is told apart from other data, because it filters and
    /// compresses differently.
    #[test]
    fn a_texture_starts_with_the_usage_its_name_suggests() {
        let usage = |name: &str| {
            let Settings::Texture(settings) = Settings::defaults_for(AssetKind::Texture, name)
            else {
                panic!("a texture's settings");
            };
            settings.usage
        };
        for name in ["Rock_NormalGL.png", "bark_nrm.jpg", "wall_Normal.png"] {
            assert_eq!(usage(name), Usage::Normal, "{name}");
        }
        for name in ["bark_roughness.jpg", "wall_height.png", "Metal_mask.png"] {
            assert_eq!(usage(name), Usage::Data, "{name}");
        }
        assert_eq!(usage("props/Rock_Color.png"), Usage::Color);
    }

    #[test]
    fn a_texture_sidecar_reads_every_setting() {
        let dir = Dir::new("sidecar-texture");
        let source = dir.write("ui/icon.png", b"");
        dir.write(
            "ui/icon.png.meta",
            b"id = \"5f0c3a4e-0000-4000-8000-000000000001\"\n\n[import]\nusage = \"data\"\ncompress = false\nmax_size = 64\n",
        );
        let sidecar = read_or_create(&source, "ui/icon.png", AssetKind::Texture).unwrap();
        assert_eq!(
            sidecar.settings,
            Settings::Texture(TextureSettings {
                usage: Usage::Data,
                compress: false,
                max_size: 64
            })
        );
        assert_eq!(
            sidecar.settings.canonical(),
            "usage=data;compress=false;max_size=64"
        );
    }

    #[test]
    fn a_usage_that_is_not_one_of_the_three_is_refused_by_name() {
        let dir = Dir::new("sidecar-usage");
        let source = dir.write("rock.png", b"");
        dir.write(
            "rock.png.meta",
            b"id = \"5f0c3a4e-0000-4000-8000-000000000001\"\n\n[import]\nusage = \"albedo\"\n",
        );
        let err = read_or_create(&source, "rock.png", AssetKind::Texture).unwrap_err();
        assert_eq!(
            err,
            "rock.png.meta: `usage` must be \"color\", \"normal\" or \"data\", not \"albedo\""
        );
    }

    /// A prefab's sidecar is `id` alone, and it is the same file format: it has
    /// to read, and nothing may be added to it.
    #[test]
    fn a_sidecar_with_only_an_id_takes_the_defaults_and_is_left_alone() {
        let dir = Dir::new("sidecar-id-only");
        let source = dir.write("tree.gltf", b"{}");
        let id = "5f0c3a4e-0000-4000-8000-000000000001";
        dir.write("tree.gltf.meta", format!("id = \"{id}\"\n").as_bytes());

        let sidecar = read_or_create(&source, "tree.gltf", AssetKind::Mesh).unwrap();
        assert_eq!(sidecar.id, id.parse().unwrap());
        assert_eq!(
            sidecar.settings,
            Settings::Mesh(MeshSettings { scale: 1.0 })
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("tree.gltf.meta")).unwrap(),
            format!("id = \"{id}\"\n")
        );
    }

    #[test]
    fn an_unknown_setting_is_refused_by_name() {
        let dir = Dir::new("sidecar-unknown");
        let source = dir.write("rock.png", b"");
        dir.write(
            "rock.png.meta",
            b"id = \"5f0c3a4e-0000-4000-8000-000000000001\"\n\n[import]\nsrbg = false\n",
        );
        let err = read_or_create(&source, "rock.png", AssetKind::Texture).unwrap_err();
        assert_eq!(
            err,
            "rock.png.meta: `srbg` is not a texture import setting; the settings are \
             usage, compress, max_size"
        );
    }

    #[test]
    fn a_setting_of_the_wrong_type_is_refused_by_name() {
        let dir = Dir::new("sidecar-type");
        let source = dir.write("tree.glb", b"");
        dir.write(
            "tree.glb.meta",
            b"id = \"5f0c3a4e-0000-4000-8000-000000000001\"\n\n[import]\nscale = \"big\"\n",
        );
        let err = read_or_create(&source, "tree.glb", AssetKind::Mesh).unwrap_err();
        assert_eq!(err, "tree.glb.meta: `scale` must be a number, not \"big\"");
    }

    #[test]
    fn a_sidecar_without_a_valid_id_is_refused_rather_than_replaced() {
        let dir = Dir::new("sidecar-bad-id");
        let source = dir.write("rock.png", b"");
        dir.write("rock.png.meta", b"id = \"nope\"\n");
        let err = read_or_create(&source, "rock.png", AssetKind::Texture).unwrap_err();
        assert!(
            err.starts_with("rock.png.meta has no valid `id`. Restore it from version control"),
            "{err}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("rock.png.meta")).unwrap(),
            "id = \"nope\"\n"
        );
    }

    #[test]
    fn a_tint_round_trips_as_hex() {
        assert_eq!(Tint::parse("#e5534B"), Some(Tint([0xe5, 0x53, 0x4b])));
        assert_eq!(Tint([0x0a, 0xb0, 0x0c]).to_hex(), "#0ab00c");
        assert_eq!(Tint::parse("e5534b"), None);
        assert_eq!(Tint::parse("#e5534"), None);
        assert_eq!(Tint::parse("#zz534b"), None);
    }

    /// A folder has no sidecar until it is tinted, and clearing the tint keeps
    /// the id the first tint gave it.
    #[test]
    fn a_folder_tint_is_written_kept_and_cleared() {
        let dir = Dir::new("sidecar-folder");
        let props = dir.path().join("props");
        std::fs::create_dir_all(&props).unwrap();
        assert_eq!(read_folder(&props, "props").unwrap(), None);

        let id = write_folder_tint(&props, "props", Some(Tint([0xe5, 0x53, 0x4b]))).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("props.meta")).unwrap(),
            format!("id = \"{id}\"\ntint = \"#e5534b\"\n")
        );
        let read = read_folder(&props, "props").unwrap().unwrap();
        assert_eq!(read.id, id);
        assert_eq!(read.tint, Some(Tint([0xe5, 0x53, 0x4b])));

        assert_eq!(write_folder_tint(&props, "props", None).unwrap(), id);
        let cleared = read_folder(&props, "props").unwrap().unwrap();
        assert_eq!((cleared.id, cleared.tint), (id, None));
    }

    #[test]
    fn a_folder_tint_that_is_not_a_colour_is_refused() {
        let dir = Dir::new("sidecar-folder-bad");
        let props = dir.path().join("props");
        std::fs::create_dir_all(&props).unwrap();
        dir.write(
            "props.meta",
            b"id = \"5f0c3a4e-0000-4000-8000-000000000001\"\ntint = \"red\"\n",
        );
        let err = read_folder(&props, "props").unwrap_err();
        assert_eq!(
            err,
            "props.meta: `tint` must be a colour written \"#rrggbb\", not \"red\""
        );
    }
}
