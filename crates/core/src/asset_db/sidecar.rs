//! The `<file>.meta` beside every source asset: its identity and its import
//! settings, in TOML a person can read in a diff.
//!
//! ```toml
//! id = "5f0c…"
//!
//! [import]
//! srgb = true
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
//! - texture: `srgb` (colour, decoded to linear on sample, or data read as
//!   stored) and `max_size` (the longest side in texels the source is scaled
//!   down to fit; `0` keeps it as it is).
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
    Folder,
}

impl AssetKind {
    /// The kind a file is imported as, or `None` for a file that is not an
    /// asset — a glTF's `.bin`, a sidecar, a readme. Never [`Folder`](Self::Folder).
    pub fn of_path(path: &str) -> Option<Self> {
        todo!("{path}")
    }

    /// The word an error or the editor uses for it.
    pub fn label(self) -> &'static str {
        todo!("{self:?}")
    }
}

/// How a texture is imported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureSettings {
    pub srgb: bool,
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
    /// A kind with nothing to set: a prefab is read as it is.
    None,
}

impl Settings {
    /// What a new sidecar is written with.
    ///
    /// A texture whose name says it holds data — a normal, roughness or height
    /// map — starts linear. The guess is made once, here, and the sidecar is
    /// what is read from then on, so a wrong guess is corrected by editing one
    /// line rather than by renaming the file.
    pub fn defaults_for(kind: AssetKind, rel: &str) -> Self {
        todo!("{kind:?} {rel}")
    }

    /// The settings as one deterministic string: what the cache key hashes, so
    /// two spellings of the same settings in two sidecars share one artifact.
    pub fn canonical(&self) -> String {
        todo!()
    }
}

/// An sRGB colour a folder's icon is drawn in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tint(pub [u8; 3]);

impl Tint {
    /// `"#rrggbb"`, the form a sidecar stores.
    pub fn parse(text: &str) -> Option<Self> {
        todo!("{text}")
    }

    pub fn to_hex(self) -> String {
        todo!()
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
    todo!("{} {rel} {kind:?}", source.display())
}

/// A folder's sidecar, which exists only once a tint has been set.
pub fn read_folder(dir: &Path, rel: &str) -> Result<Option<Sidecar>, String> {
    todo!("{} {rel}", dir.display())
}

/// Set or clear a folder's tint, keeping the id it already has. Returns the
/// folder's id.
pub fn write_folder_tint(dir: &Path, rel: &str, tint: Option<Tint>) -> Result<AssetId, String> {
    todo!("{} {rel} {tint:?}", dir.display())
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
                "id = \"{}\"\n\n[import]\nsrgb = true\nmax_size = 0\n",
                first.id
            )
        );
        assert_eq!(
            first.settings,
            Settings::Texture(TextureSettings {
                srgb: true,
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

    #[test]
    fn a_data_texture_starts_linear() {
        for name in ["Rock_NormalGL.png", "bark_roughness.jpg", "wall_height.png"] {
            let Settings::Texture(settings) = Settings::defaults_for(AssetKind::Texture, name)
            else {
                panic!("a texture's settings");
            };
            assert!(!settings.srgb, "{name}");
        }
        let Settings::Texture(colour) =
            Settings::defaults_for(AssetKind::Texture, "props/Rock_Color.png")
        else {
            panic!("a texture's settings");
        };
        assert!(colour.srgb);
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
        assert_eq!(sidecar.settings, Settings::Mesh(MeshSettings { scale: 1.0 }));
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
             srgb, max_size"
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
