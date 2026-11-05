use glam::Vec2;
use orrin_registry::{AssetId, Reflect, Value, ValueError, take};

use crate::scene::TextAnchor;

/// How a canvas's units map to the window's pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum ScaleMode {
    /// One unit is one physical pixel, whatever the window's size.
    #[default]
    ConstantPixel,
    /// The canvas is [`Canvas::reference_resolution`] units across, scaled to
    /// the window by a blend of its width and height ratios.
    ScaleWithScreen,
}

/// The root of a UI tree: the window, as the rectangle every child's anchors
/// are fractions of. A `Canvas` under another canvas is laid out as an ordinary
/// element and its own settings are ignored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Canvas {
    pub scale_mode: ScaleMode,
    /// The window size a `ScaleWithScreen` layout is authored at.
    pub reference_resolution: Vec2,
    /// 0 scales with the window's width, 1 with its height, and between blends
    /// the two logarithmically, as uGUI's scaler does.
    pub match_width_or_height: f32,
    /// Canvases draw lowest first; a tie keeps hierarchy order.
    pub sort_order: i32,
}

impl Default for Canvas {
    fn default() -> Self {
        Self {
            scale_mode: ScaleMode::ConstantPixel,
            reference_resolution: Vec2::new(1920.0, 1080.0),
            match_width_or_height: 0.0,
            sort_order: 0,
        }
    }
}

/// A UI element's rectangle relative to its parent's, with uGUI's semantics
/// in a top-left, y-down frame: anchor `(0, 0)` is the parent's top left.
///
/// The element is `(anchor_max - anchor_min) * parent + size_delta` across, and
/// its pivot stands at `lerp(anchor_min, anchor_max, pivot)` of the parent plus
/// `anchored_position`. `scale` scales the element and everything under it
/// about that pivot. UI entities carry this instead of a `LocalTransform`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RectTransform {
    pub anchor_min: Vec2,
    pub anchor_max: Vec2,
    pub pivot: Vec2,
    pub anchored_position: Vec2,
    pub size_delta: Vec2,
    pub scale: Vec2,
}

impl Default for RectTransform {
    /// Centred in its parent, 100 units square.
    fn default() -> Self {
        Self {
            anchor_min: Vec2::splat(0.5),
            anchor_max: Vec2::splat(0.5),
            pivot: Vec2::splat(0.5),
            anchored_position: Vec2::ZERO,
            size_delta: Vec2::splat(100.0),
            scale: Vec2::ONE,
        }
    }
}

impl RectTransform {
    /// Filling its parent edge to edge.
    pub fn stretched() -> Self {
        Self {
            anchor_min: Vec2::ZERO,
            anchor_max: Vec2::ONE,
            size_delta: Vec2::ZERO,
            ..Self::default()
        }
    }

    /// This element's rectangle in its parent's frame, as `(min, size)`,
    /// before its own scale.
    pub fn rect_in(&self, parent_min: Vec2, parent_size: Vec2) -> (Vec2, Vec2) {
        let size = (self.anchor_max - self.anchor_min) * parent_size + self.size_delta;
        let reference = parent_min
            + (self.anchor_min + (self.anchor_max - self.anchor_min) * self.pivot) * parent_size;
        let pivot = reference + self.anchored_position;
        (pivot - self.pivot * size, size)
    }
}

/// A rectangle filled with `color`, or with `texture` multiplied by it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Image {
    /// `None` fills the rectangle with plain colour.
    pub texture: Option<AssetId>,
    /// sRGB, straight alpha.
    pub color: [f32; 4],
    /// Left, top, right, bottom, in pixels of the texture as imported. Zero
    /// stretches the whole texture; anything else draws nine pieces, the
    /// corners unstretched and the edges along one axis only.
    pub slice: [f32; 4],
    /// Whether the pointer stops here, rather than reaching what is behind.
    pub raycast_target: bool,
}

impl Default for Image {
    fn default() -> Self {
        Self {
            texture: None,
            color: [1.0; 4],
            slice: [0.0; 4],
            raycast_target: true,
        }
    }
}

/// A run of text set in the element's rectangle.
#[derive(Clone, Debug, PartialEq)]
pub struct Text {
    pub text: String,
    /// `None` is the engine's built-in face.
    pub font: Option<AssetId>,
    /// Ascent to descent, in canvas units.
    pub size: f32,
    /// sRGB, straight alpha.
    pub color: [f32; 4],
    pub anchor: TextAnchor,
    /// Off, a line runs past the rectangle's sides rather than breaking.
    pub wrap: bool,
    pub raycast_target: bool,
}

impl Default for Text {
    fn default() -> Self {
        Self {
            text: String::new(),
            font: None,
            size: 24.0,
            color: [0.196, 0.196, 0.196, 1.0],
            anchor: TextAnchor::UpperLeft,
            wrap: true,
            raycast_target: false,
        }
    }
}

/// Tints the `Image` on the same entity by the pointer's state over it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Button {
    pub normal: [f32; 4],
    pub hover: [f32; 4],
    pub pressed: [f32; 4],
    pub disabled: [f32; 4],
    /// Off, the button draws `disabled` and neither it nor anything under it
    /// receives a pointer event — while still stopping the pointer.
    pub interactable: bool,
}

impl Default for Button {
    /// uGUI's colour-tint defaults.
    fn default() -> Self {
        Self {
            normal: [1.0; 4],
            hover: [0.96, 0.96, 0.96, 1.0],
            pressed: [0.78, 0.78, 0.78, 1.0],
            disabled: [0.78, 0.78, 0.78, 0.5],
            interactable: true,
        }
    }
}

/// A `Vec2` as a scene stores it: a struct of `x` and `y`. The registry has no
/// two-component leaf, and adding one would change the file and wire formats
/// for a type only this module needs.
struct V2(Vec2);

impl Reflect for V2 {
    fn to_value(&self) -> Value {
        Value::strukt([("x", self.0.x.to_value()), ("y", self.0.y.to_value())])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self(Vec2::new(take(value, "x")?, take(value, "y")?)))
    }
}

const RGBA: [&str; 4] = ["r", "g", "b", "a"];
const SIDES: [&str; 4] = ["left", "top", "right", "bottom"];

/// A colour, or a slice's four borders, as named channels.
fn four(values: [f32; 4], names: [&str; 4]) -> Value {
    Value::Struct(
        names
            .iter()
            .zip(values)
            .map(|(name, v)| ((*name).to_owned(), v.to_value()))
            .collect(),
    )
}

fn take_four(value: &Value, field: &str, names: [&str; 4]) -> Result<[f32; 4], ValueError> {
    let inner = value
        .field(field)
        .ok_or_else(|| ValueError::missing(field))?;
    let mut out = [0.0; 4];
    for (slot, name) in out.iter_mut().zip(names) {
        *slot = take(inner, name).map_err(|e| e.at_field(field))?;
    }
    Ok(out)
}

fn vec2(value: &Value, field: &str) -> Result<Vec2, ValueError> {
    take::<V2>(value, field).map(|v| v.0)
}

/// An optional asset is a field that is present or absent, so a scene names
/// only the assets an element actually uses.
fn optional_asset(value: &Value, field: &str) -> Result<Option<AssetId>, ValueError> {
    value
        .field(field)
        .map(|inner| AssetId::from_value(inner).map_err(|e| e.at_field(field)))
        .transpose()
}

fn with_asset(mut value: Value, field: &str, asset: Option<AssetId>) -> Value {
    if let (Value::Struct(fields), Some(id)) = (&mut value, asset) {
        fields.push((field.to_owned(), id.to_value()));
    }
    value
}

/// Hand-written for its `Vec2` fields; see [`V2`].
impl Reflect for Canvas {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("scale_mode", self.scale_mode.to_value()),
            (
                "reference_resolution",
                V2(self.reference_resolution).to_value(),
            ),
            (
                "match_width_or_height",
                self.match_width_or_height.to_value(),
            ),
            ("sort_order", self.sort_order.to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            scale_mode: take(value, "scale_mode")?,
            reference_resolution: vec2(value, "reference_resolution")?,
            match_width_or_height: take(value, "match_width_or_height")?,
            sort_order: take(value, "sort_order")?,
        })
    }
}

impl Reflect for RectTransform {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("anchor_min", V2(self.anchor_min).to_value()),
            ("anchor_max", V2(self.anchor_max).to_value()),
            ("pivot", V2(self.pivot).to_value()),
            ("anchored_position", V2(self.anchored_position).to_value()),
            ("size_delta", V2(self.size_delta).to_value()),
            ("scale", V2(self.scale).to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            anchor_min: vec2(value, "anchor_min")?,
            anchor_max: vec2(value, "anchor_max")?,
            pivot: vec2(value, "pivot")?,
            anchored_position: vec2(value, "anchored_position")?,
            size_delta: vec2(value, "size_delta")?,
            scale: vec2(value, "scale")?,
        })
    }
}

impl Reflect for Image {
    fn to_value(&self) -> Value {
        let value = Value::strukt([
            ("color", four(self.color, RGBA)),
            ("slice", four(self.slice, SIDES)),
            ("raycast_target", self.raycast_target.to_value()),
        ]);
        with_asset(value, "texture", self.texture)
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            texture: optional_asset(value, "texture")?,
            color: take_four(value, "color", RGBA)?,
            slice: take_four(value, "slice", SIDES)?,
            raycast_target: take(value, "raycast_target")?,
        })
    }
}

impl Reflect for Text {
    fn to_value(&self) -> Value {
        let value = Value::strukt([
            ("text", self.text.to_value()),
            ("size", self.size.to_value()),
            ("color", four(self.color, RGBA)),
            ("anchor", self.anchor.to_value()),
            ("wrap", self.wrap.to_value()),
            ("raycast_target", self.raycast_target.to_value()),
        ]);
        with_asset(value, "font", self.font)
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            text: take(value, "text")?,
            font: optional_asset(value, "font")?,
            size: take(value, "size")?,
            color: take_four(value, "color", RGBA)?,
            anchor: take(value, "anchor")?,
            wrap: take(value, "wrap")?,
            raycast_target: take(value, "raycast_target")?,
        })
    }
}

impl Reflect for Button {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("normal", four(self.normal, RGBA)),
            ("hover", four(self.hover, RGBA)),
            ("pressed", four(self.pressed, RGBA)),
            ("disabled", four(self.disabled, RGBA)),
            ("interactable", self.interactable.to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            normal: take_four(value, "normal", RGBA)?,
            hover: take_four(value, "hover", RGBA)?,
            pressed: take_four(value, "pressed", RGBA)?,
            disabled: take_four(value, "disabled", RGBA)?,
            interactable: take(value, "interactable")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<T: Reflect + PartialEq + std::fmt::Debug>(value: T) {
        assert_eq!(T::from_value(&value.to_value()).unwrap(), value);
    }

    #[test]
    fn every_ui_component_reads_back_what_it_wrote() {
        round_trip(Canvas {
            scale_mode: ScaleMode::ScaleWithScreen,
            reference_resolution: Vec2::new(1280.0, 720.0),
            match_width_or_height: 0.25,
            sort_order: -3,
        });
        round_trip(RectTransform {
            anchor_min: Vec2::new(0.1, 0.2),
            anchor_max: Vec2::new(0.9, 0.8),
            pivot: Vec2::new(0.0, 1.0),
            anchored_position: Vec2::new(-4.0, 12.5),
            size_delta: Vec2::new(30.0, -8.0),
            scale: Vec2::new(2.0, 0.5),
        });
        round_trip(Image {
            texture: Some(AssetId::new()),
            color: [0.1, 0.2, 0.3, 0.4],
            slice: [4.0, 5.0, 6.0, 7.0],
            raycast_target: false,
        });
        round_trip(Image::default());
        round_trip(Text {
            text: "Play".into(),
            font: Some(AssetId::new()),
            size: 31.0,
            color: [1.0, 0.5, 0.25, 1.0],
            anchor: TextAnchor::LowerRight,
            wrap: false,
            raycast_target: true,
        });
        round_trip(Text::default());
        round_trip(Button {
            interactable: false,
            ..Button::default()
        });
    }

    /// A scene names only the assets an element uses.
    #[test]
    fn an_image_with_no_texture_writes_no_texture_field() {
        let value = Image::default().to_value();
        assert!(value.field("texture").is_none(), "{value:?}");
        let textured = Image {
            texture: Some(AssetId::new()),
            ..Image::default()
        };
        assert!(textured.to_value().field("texture").is_some());
    }
}
