use std::fmt;

use serde::{Deserialize, Serialize};

/// One declared parameter: its name, type with default and hints, and where it
/// sits in its material's block of words.
#[derive(Clone, PartialEq, Debug)]
pub struct Param {
    pub name: String,
    pub kind: ParamKind,
    pub slot: Slot,
}

/// A parameter's type, carrying its default and the hints that belong to it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ParamKind {
    Float {
        default: f32,
        /// `(min, max)`: what the editor's slider spans. Values are not clamped.
        range: Option<[f32; 2]>,
    },
    Vec2 {
        default: [f32; 2],
    },
    Vec3 {
        default: [f32; 3],
    },
    Vec4 {
        default: [f32; 4],
    },
    /// A `vec4` the editor shows as a colour picker. Linear.
    Color {
        default: [f32; 4],
    },
    /// A bindless texture index.
    Texture {
        default: DefaultTexture,
        /// How an image dropped on it is imported.
        color_space: ColorSpace,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DefaultTexture {
    White,
    Black,
    /// The flat tangent-space normal.
    Normal,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

/// Where a parameter starts: `word` from its material's first word, and the
/// component within it. It runs on for [`ParamKind::components`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Slot {
    pub word: u32,
    pub component: u32,
}

/// Every parameter a surface shader declares, in declaration order, and the
/// number of words one material's block of them takes.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct ParamTable {
    pub params: Vec<Param>,
    pub word_count: u32,
}

/// A material's value for one parameter. A `color` takes a `Vec4`; a `texture`
/// takes the bindless index the caller resolved its image to.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum ParamValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Texture(u32),
}

/// A value [`ParamTable::words`] did not use, and why. Reported rather than
/// fatal, so a renamed or retyped parameter warns instead of failing a load.
#[derive(Clone, PartialEq, Debug)]
pub struct Ignored {
    pub name: String,
    pub reason: IgnoredReason,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IgnoredReason {
    /// The shader declares no parameter of this name.
    Undeclared,
    WrongType {
        declared: &'static str,
        given: &'static str,
    },
}

impl fmt::Display for Ignored {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.reason {
            IgnoredReason::Undeclared => write!(
                f,
                "`{}` is not a parameter of this shader; the value is ignored",
                self.name
            ),
            IgnoredReason::WrongType { declared, given } => write!(
                f,
                "`{}` is a {declared} but was given a {given}; the shader's default is used",
                self.name
            ),
        }
    }
}

impl ParamKind {
    /// How many `f32`-sized components it takes.
    pub fn components(&self) -> u32 {
        match self {
            ParamKind::Float { .. } | ParamKind::Texture { .. } => 1,
            ParamKind::Vec2 { .. } => 2,
            ParamKind::Vec3 { .. } => 3,
            ParamKind::Vec4 { .. } | ParamKind::Color { .. } => 4,
        }
    }

    /// The type as the source spells it.
    pub fn type_name(&self) -> &'static str {
        match self {
            ParamKind::Float { .. } => "float",
            ParamKind::Vec2 { .. } => "vec2",
            ParamKind::Vec3 { .. } => "vec3",
            ParamKind::Vec4 { .. } => "vec4",
            ParamKind::Color { .. } => "color",
            ParamKind::Texture { .. } => "texture",
        }
    }
}

impl ParamValue {
    fn type_name(&self) -> &'static str {
        match self {
            ParamValue::Float(_) => "float",
            ParamValue::Vec2(_) => "vec2",
            ParamValue::Vec3(_) => "vec3",
            ParamValue::Vec4(_) => "vec4",
            ParamValue::Texture(_) => "texture",
        }
    }
}

/// Give each parameter its slot in declaration order, by the rule in the module
/// docs, and return how many words the block takes.
pub(super) fn pack(params: &mut [Param]) -> u32 {
    let (mut word, mut component) = (0, 0);
    for param in params {
        let n = param.kind.components();
        if component + n > 4 {
            word += 1;
            component = 0;
        }
        param.slot = Slot { word, component };
        component += n;
    }
    if component > 0 { word + 1 } else { word }
}

impl ParamTable {
    pub fn get(&self, name: &str) -> Option<&Param> {
        self.params.iter().find(|param| param.name == name)
    }

    /// One material's block: the shader's defaults, overwritten by each value
    /// whose name and type match a declared parameter. `default_texture` says
    /// which index each default texture lives at, which only the renderer knows.
    pub fn words<'a>(
        &self,
        values: impl IntoIterator<Item = (&'a str, ParamValue)>,
        default_texture: impl Fn(DefaultTexture) -> u32,
    ) -> (Vec<[u32; 4]>, Vec<Ignored>) {
        let mut words = vec![[0u32; 4]; self.word_count as usize];
        for param in &self.params {
            let bits = match param.kind {
                ParamKind::Float { default, .. } => vec![default.to_bits()],
                ParamKind::Vec2 { default } => default.map(f32::to_bits).to_vec(),
                ParamKind::Vec3 { default } => default.map(f32::to_bits).to_vec(),
                ParamKind::Vec4 { default } | ParamKind::Color { default } => {
                    default.map(f32::to_bits).to_vec()
                }
                ParamKind::Texture { default, .. } => vec![default_texture(default)],
            };
            write(&mut words, param.slot, &bits);
        }

        let mut ignored = Vec::new();
        for (name, value) in values {
            let Some(param) = self.get(name) else {
                ignored.push(Ignored {
                    name: name.to_string(),
                    reason: IgnoredReason::Undeclared,
                });
                continue;
            };
            let bits = match (param.kind, value) {
                (ParamKind::Float { .. }, ParamValue::Float(v)) => vec![v.to_bits()],
                (ParamKind::Vec2 { .. }, ParamValue::Vec2(v)) => v.map(f32::to_bits).to_vec(),
                (ParamKind::Vec3 { .. }, ParamValue::Vec3(v)) => v.map(f32::to_bits).to_vec(),
                (ParamKind::Vec4 { .. } | ParamKind::Color { .. }, ParamValue::Vec4(v)) => {
                    v.map(f32::to_bits).to_vec()
                }
                (ParamKind::Texture { .. }, ParamValue::Texture(index)) => vec![index],
                (kind, value) => {
                    ignored.push(Ignored {
                        name: name.to_string(),
                        reason: IgnoredReason::WrongType {
                            declared: kind.type_name(),
                            given: value.type_name(),
                        },
                    });
                    continue;
                }
            };
            write(&mut words, param.slot, &bits);
        }
        (words, ignored)
    }
}

fn write(words: &mut [[u32; 4]], slot: Slot, bits: &[u32]) {
    let word = &mut words[slot.word as usize];
    let at = slot.component as usize;
    word[at..at + bits.len()].copy_from_slice(bits);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(kinds: &[ParamKind]) -> ParamTable {
        let mut params: Vec<Param> = kinds
            .iter()
            .enumerate()
            .map(|(index, &kind)| Param {
                name: format!("p{index}"),
                kind,
                slot: Slot {
                    word: 0,
                    component: 0,
                },
            })
            .collect();
        let word_count = pack(&mut params);
        ParamTable { params, word_count }
    }

    fn slots(table: &ParamTable) -> Vec<(u32, u32)> {
        table
            .params
            .iter()
            .map(|p| (p.slot.word, p.slot.component))
            .collect()
    }

    const FLOAT: ParamKind = ParamKind::Float {
        default: 0.0,
        range: None,
    };
    const VEC2: ParamKind = ParamKind::Vec2 { default: [0.0; 2] };
    const VEC3: ParamKind = ParamKind::Vec3 { default: [0.0; 3] };
    const VEC4: ParamKind = ParamKind::Vec4 { default: [0.0; 4] };
    const TEXTURE: ParamKind = ParamKind::Texture {
        default: DefaultTexture::White,
        color_space: ColorSpace::Srgb,
    };

    fn lookup(texture: DefaultTexture) -> u32 {
        match texture {
            DefaultTexture::White => 10,
            DefaultTexture::Black => 11,
            DefaultTexture::Normal => 12,
        }
    }

    #[test]
    fn parameters_pack_in_declaration_order_without_crossing_a_word() {
        let packed = table(&[FLOAT, VEC3]);
        assert_eq!(slots(&packed), [(0, 0), (0, 1)]);
        assert_eq!(packed.word_count, 1);

        let packed = table(&[FLOAT, FLOAT, VEC4]);
        assert_eq!(slots(&packed), [(0, 0), (0, 1), (1, 0)]);
        assert_eq!(packed.word_count, 2);

        let packed = table(&[VEC3, VEC2, TEXTURE]);
        assert_eq!(slots(&packed), [(0, 0), (1, 0), (1, 2)], "no back-filling");
        assert_eq!(packed.word_count, 2);

        let packed = table(&[TEXTURE, FLOAT, VEC2, FLOAT]);
        assert_eq!(slots(&packed), [(0, 0), (0, 1), (0, 2), (1, 0)]);
        assert_eq!(packed.word_count, 2);

        assert_eq!(table(&[]).word_count, 0);
    }

    #[test]
    fn packing_is_deterministic_and_appending_moves_nothing() {
        let kinds = [VEC2, FLOAT, VEC3, TEXTURE, VEC4, FLOAT];
        let first = table(&kinds);
        assert_eq!(first, table(&kinds));
        let mut longer = kinds.to_vec();
        longer.push(VEC3);
        assert_eq!(slots(&table(&longer))[..kinds.len()], slots(&first)[..]);
    }

    #[test]
    fn a_material_with_no_values_gets_the_shaders_defaults() {
        let packed = table(&[
            ParamKind::Float {
                default: 0.5,
                range: Some([0.0, 1.0]),
            },
            ParamKind::Texture {
                default: DefaultTexture::Normal,
                color_space: ColorSpace::Linear,
            },
            ParamKind::Vec2 {
                default: [1.0, 2.0],
            },
            ParamKind::Color {
                default: [0.25, 0.5, 0.75, 1.0],
            },
        ]);
        let (words, ignored) = packed.words([], lookup);
        assert!(ignored.is_empty());
        assert_eq!(
            words,
            [
                [0.5f32.to_bits(), 12, 1.0f32.to_bits(), 2.0f32.to_bits()],
                [0.25f32, 0.5, 0.75, 1.0].map(f32::to_bits),
            ]
        );
    }

    #[test]
    fn a_value_overrides_its_default_and_leaves_the_rest() {
        let packed = table(&[FLOAT, VEC3, TEXTURE, ParamKind::Color { default: [1.0; 4] }]);
        let (words, ignored) = packed.words(
            [
                ("p1", ParamValue::Vec3([1.0, 2.0, 3.0])),
                ("p2", ParamValue::Texture(77)),
                ("p3", ParamValue::Vec4([0.0, 0.1, 0.2, 0.3])),
            ],
            lookup,
        );
        assert!(ignored.is_empty());
        assert_eq!(
            words,
            [
                [0.0f32, 1.0, 2.0, 3.0].map(f32::to_bits),
                [77, 0, 0, 0],
                [0.0f32, 0.1, 0.2, 0.3].map(f32::to_bits),
            ]
        );
    }

    #[test]
    fn unknown_and_mistyped_values_are_reported_and_do_not_stop_the_rest() {
        let packed = table(&[FLOAT, TEXTURE]);
        let (words, ignored) = packed.words(
            [
                ("renamed", ParamValue::Float(9.0)),
                ("p0", ParamValue::Vec3([1.0; 3])),
                ("p1", ParamValue::Float(3.0)),
                ("p0", ParamValue::Float(0.25)),
            ],
            lookup,
        );
        assert_eq!(words, [[0.25f32.to_bits(), 10, 0, 0]]);
        assert_eq!(
            ignored,
            [
                Ignored {
                    name: "renamed".into(),
                    reason: IgnoredReason::Undeclared,
                },
                Ignored {
                    name: "p0".into(),
                    reason: IgnoredReason::WrongType {
                        declared: "float",
                        given: "vec3",
                    },
                },
                Ignored {
                    name: "p1".into(),
                    reason: IgnoredReason::WrongType {
                        declared: "texture",
                        given: "float",
                    },
                },
            ]
        );
        assert!(ignored[0].to_string().contains("renamed"));
        assert!(ignored[1].to_string().contains("float"));
    }

    /// A texture index rides in a float lane, and the GPU reads it back with
    /// `floatBitsToUint`. Indices whose bit patterns are a quiet and a signalling
    /// NaN must arrive exactly, through a value and through a default.
    #[test]
    fn a_texture_index_survives_as_bits_even_when_it_is_a_nan() {
        for index in [0x7fc0_0001u32, 0x7f80_0001, 0xffff_ffff] {
            assert!(f32::from_bits(index).is_nan());
            let packed = table(&[FLOAT, TEXTURE]);
            let (words, _) = packed.words([("p1", ParamValue::Texture(index))], lookup);
            assert_eq!(words[0][1], index);
            let (words, _) = packed.words([], |_| index);
            assert_eq!(words[0][1], index);
        }
    }
}
