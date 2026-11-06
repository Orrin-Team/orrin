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

/// One axis of an anchor preset: where on the parent the anchors sit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorPreset {
    /// The left or top edge.
    Min,
    Middle,
    /// The right or bottom edge.
    Max,
    /// Both edges, so the element stretches with its parent.
    Stretch,
}

impl AnchorPreset {
    pub const ALL: [Self; 4] = [Self::Min, Self::Middle, Self::Max, Self::Stretch];

    /// The anchors' `(min, max)` along the axis.
    pub fn anchors(self) -> (f32, f32) {
        match self {
            Self::Min => (0.0, 0.0),
            Self::Middle => (0.5, 0.5),
            Self::Max => (1.0, 1.0),
            Self::Stretch => (0.0, 1.0),
        }
    }

    /// The pivot that goes with the preset: on the anchored edge, or centred.
    pub fn pivot(self) -> f32 {
        match self {
            Self::Min => 0.0,
            Self::Max => 1.0,
            Self::Middle | Self::Stretch => 0.5,
        }
    }

    /// The preset `(min, max)` anchors are, if they are one.
    pub fn of(min: f32, max: f32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|preset| preset.anchors() == (min, max))
    }
}

impl RectTransform {
    /// Move the anchors to `min` and `max` without moving the element in a
    /// parent `parent` units across: the size delta and the position absorb
    /// the change.
    pub fn set_anchors_keeping_rect(&mut self, parent: Vec2, min: Vec2, max: Vec2) {
        let (corner, size) = self.rect_in(Vec2::ZERO, parent);
        self.anchor_min = min;
        self.anchor_max = max;
        self.size_delta = size - (max - min) * parent;
        self.place_pivot(parent, corner + self.pivot * size);
    }

    /// Move the pivot to `pivot` without moving the element's unscaled
    /// rectangle.
    pub fn set_pivot_keeping_rect(&mut self, parent: Vec2, pivot: Vec2) {
        let (corner, size) = self.rect_in(Vec2::ZERO, parent);
        self.pivot = pivot;
        self.place_pivot(parent, corner + pivot * size);
    }

    /// Set the position that puts the pivot at `point` of the parent.
    fn place_pivot(&mut self, parent: Vec2, point: Vec2) {
        let reference =
            (self.anchor_min + (self.anchor_max - self.anchor_min) * self.pivot) * parent;
        self.anchored_position = point - reference;
    }

    /// uGUI's anchor presets: anchor each axis by its preset, keeping the
    /// element where it is. With `pivot` the pivot moves to the matching edge
    /// or centre too; with `position` the element moves to its anchors —
    /// stretched axes filling the parent.
    pub fn apply_preset(
        &mut self,
        parent: Vec2,
        presets: [AnchorPreset; 2],
        pivot: bool,
        position: bool,
    ) {
        let [across, down] = presets;
        let ((x_min, x_max), (y_min, y_max)) = (across.anchors(), down.anchors());
        self.set_anchors_keeping_rect(parent, Vec2::new(x_min, y_min), Vec2::new(x_max, y_max));
        if pivot {
            self.set_pivot_keeping_rect(parent, Vec2::new(across.pivot(), down.pivot()));
        }
        if position {
            self.anchored_position = Vec2::ZERO;
            for (axis, preset) in presets.into_iter().enumerate() {
                if preset == AnchorPreset::Stretch {
                    self.size_delta[axis] = 0.0;
                }
            }
        }
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

/// Colours what the element fills — its `Image` and its `Text` — with a linear
/// blend from `start` to `end`, multiplied into the element's own colour.
///
/// At `rotation` 0 the blend runs left to right; the angle is in degrees,
/// clockwise on screen, and the blend always spans the element's whole box
/// along that axis. `offset` moves it by a fraction of the element's size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gradient {
    /// sRGB, straight alpha.
    pub start: [f32; 4],
    /// sRGB, straight alpha.
    pub end: [f32; 4],
    pub rotation: f32,
    pub offset: Vec2,
}

impl Default for Gradient {
    fn default() -> Self {
        Self {
            start: [1.0; 4],
            end: [0.0, 0.0, 0.0, 1.0],
            rotation: 0.0,
            offset: Vec2::ZERO,
        }
    }
}

/// The curve a [`Corner`] is cut along.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum CornerStyle {
    /// A quarter circle.
    #[default]
    Round,
    /// A quarter superellipse, whose curvature eases in from the straight edge
    /// rather than starting at once — the continuous corner of Apple's icons.
    Squircle,
}

/// Rounds the element's box: its image, and the [`Stroke`] and [`Shadow`] on
/// it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corner {
    /// In canvas units.
    pub radius: f32,
    /// Added to `radius`, as a fraction of the box's shorter side: 0.5 is a
    /// pill whatever the size.
    pub radius_fraction: f32,
    pub style: CornerStyle,
    /// The squircle's superellipse exponent: 2 is a circle, and the corner
    /// squares off as it grows. Ignored by `Round`.
    pub exponent: f32,
}

impl Default for Corner {
    fn default() -> Self {
        Self {
            radius: 8.0,
            radius_fraction: 0.0,
            style: CornerStyle::Round,
            exponent: 4.0,
        }
    }
}

impl Corner {
    /// The exponents a corner may be cut with: below 2 the corner is no longer
    /// convex, and past 12 it is indistinguishable from a square one.
    pub const EXPONENTS: std::ops::RangeInclusive<f32> = 2.0..=12.0;

    /// The radius in pixels on a box `size` pixels across, `pixels` to a unit.
    pub fn radius_px(&self, size: Vec2, pixels: f32) -> f32 {
        let short = size.x.min(size.y).max(0.0);
        (self.radius * pixels + self.radius_fraction * short).clamp(0.0, short / 2.0)
    }

    /// The superellipse exponent the shape is cut with.
    pub fn exponent(&self) -> f32 {
        match self.style {
            CornerStyle::Round => 2.0,
            CornerStyle::Squircle => self
                .exponent
                .clamp(*Self::EXPONENTS.start(), *Self::EXPONENTS.end()),
        }
    }
}

/// Where a [`Stroke`] sits against the edge it follows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum StrokePlacement {
    /// Wholly inside the box, so the element covers no more than its rect.
    #[default]
    Inside,
    Center,
    Outside,
}

/// An outline. On an element with an `Image`, or with nothing else to draw, it
/// follows the box — and its [`Corner`]; on one that is only `Text`, it rings
/// each glyph, always outside it, since a stroke over the ink would thin it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    /// sRGB, straight alpha.
    pub color: [f32; 4],
    /// In canvas units.
    pub thickness: f32,
    pub placement: StrokePlacement,
}

impl Default for Stroke {
    fn default() -> Self {
        Self {
            color: [0.0, 0.0, 0.0, 1.0],
            thickness: 2.0,
            placement: StrokePlacement::Inside,
        }
    }
}

/// A drop shadow under the element's box, following its [`Corner`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shadow {
    /// sRGB, straight alpha.
    pub color: [f32; 4],
    /// In canvas units, y down.
    pub offset: Vec2,
    /// The blur radius in canvas units, as CSS means it: twice the Gaussian's
    /// standard deviation.
    pub blur: f32,
    /// Grows the shadow's box on every side before it is blurred; negative
    /// shrinks it.
    pub spread: f32,
}

impl Default for Shadow {
    fn default() -> Self {
        Self {
            color: [0.0, 0.0, 0.0, 0.5],
            offset: Vec2::new(0.0, 4.0),
            blur: 8.0,
            spread: 0.0,
        }
    }
}

/// When a [`ScrollFrame`]'s bars are drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum ScrollbarVisibility {
    /// On a scrolling axis whose content is larger than the frame.
    #[default]
    Auto,
    /// On every scrolling axis.
    Always,
    Never,
}

/// Lays the element's children out in a content area `content_size` units
/// across, cut to the element's rect and scrolled by `scroll_position`.
///
/// The children's anchors are fractions of the content area, which is never
/// smaller than the frame itself, so a stretched child fills at least what is
/// in view. The wheel scrolls it while the game runs, as does dragging a bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollFrame {
    /// In the frame's units; an axis smaller than the frame is the frame's.
    pub content_size: Vec2,
    /// How far the content's top left is scrolled past the frame's, in units.
    /// Laid out clamped to what the content allows.
    pub scroll_position: Vec2,
    pub horizontal: bool,
    pub vertical: bool,
    /// Units scrolled per notch of the wheel.
    pub scroll_speed: f32,
    pub scrollbar: ScrollbarVisibility,
    /// In canvas units.
    pub scrollbar_thickness: f32,
    /// sRGB, straight alpha.
    pub scrollbar_color: [f32; 4],
}

impl Default for ScrollFrame {
    fn default() -> Self {
        Self {
            content_size: Vec2::ZERO,
            scroll_position: Vec2::ZERO,
            horizontal: false,
            vertical: true,
            scroll_speed: 40.0,
            scrollbar: ScrollbarVisibility::Auto,
            scrollbar_thickness: 6.0,
            scrollbar_color: [0.6, 0.6, 0.6, 0.8],
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

impl Reflect for Gradient {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("start", four(self.start, RGBA)),
            ("end", four(self.end, RGBA)),
            ("rotation", self.rotation.to_value()),
            ("offset", V2(self.offset).to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            start: take_four(value, "start", RGBA)?,
            end: take_four(value, "end", RGBA)?,
            rotation: take(value, "rotation")?,
            offset: vec2(value, "offset")?,
        })
    }
}

impl Reflect for Corner {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("radius", self.radius.to_value()),
            ("radius_fraction", self.radius_fraction.to_value()),
            ("style", self.style.to_value()),
            ("exponent", self.exponent.to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            radius: take(value, "radius")?,
            radius_fraction: take(value, "radius_fraction")?,
            style: take(value, "style")?,
            exponent: take(value, "exponent")?,
        })
    }
}

impl Reflect for Stroke {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("color", four(self.color, RGBA)),
            ("thickness", self.thickness.to_value()),
            ("placement", self.placement.to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            color: take_four(value, "color", RGBA)?,
            thickness: take(value, "thickness")?,
            placement: take(value, "placement")?,
        })
    }
}

impl Reflect for Shadow {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("color", four(self.color, RGBA)),
            ("offset", V2(self.offset).to_value()),
            ("blur", self.blur.to_value()),
            ("spread", self.spread.to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            color: take_four(value, "color", RGBA)?,
            offset: vec2(value, "offset")?,
            blur: take(value, "blur")?,
            spread: take(value, "spread")?,
        })
    }
}

impl Reflect for ScrollFrame {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("content_size", V2(self.content_size).to_value()),
            ("scroll_position", V2(self.scroll_position).to_value()),
            ("horizontal", self.horizontal.to_value()),
            ("vertical", self.vertical.to_value()),
            ("scroll_speed", self.scroll_speed.to_value()),
            ("scrollbar", self.scrollbar.to_value()),
            ("scrollbar_thickness", self.scrollbar_thickness.to_value()),
            ("scrollbar_color", four(self.scrollbar_color, RGBA)),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            content_size: vec2(value, "content_size")?,
            scroll_position: vec2(value, "scroll_position")?,
            horizontal: take(value, "horizontal")?,
            vertical: take(value, "vertical")?,
            scroll_speed: take(value, "scroll_speed")?,
            scrollbar: take(value, "scrollbar")?,
            scrollbar_thickness: take(value, "scrollbar_thickness")?,
            scrollbar_color: take_four(value, "scrollbar_color", RGBA)?,
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
    fn every_ui_modifier_reads_back_what_it_wrote() {
        round_trip(Gradient {
            start: [0.1, 0.2, 0.3, 0.4],
            end: [0.5, 0.6, 0.7, 0.8],
            rotation: 45.0,
            offset: Vec2::new(0.25, -0.5),
        });
        round_trip(Gradient::default());
        round_trip(Corner {
            radius: 12.0,
            radius_fraction: 0.25,
            style: CornerStyle::Squircle,
            exponent: 5.0,
        });
        round_trip(Corner::default());
        round_trip(Stroke {
            color: [1.0, 0.0, 0.5, 1.0],
            thickness: 3.5,
            placement: StrokePlacement::Outside,
        });
        round_trip(Stroke::default());
        round_trip(Shadow {
            color: [0.0, 0.0, 0.0, 0.25],
            offset: Vec2::new(2.0, 6.0),
            blur: 16.0,
            spread: -2.0,
        });
        round_trip(Shadow::default());
        round_trip(ScrollFrame {
            content_size: Vec2::new(0.0, 900.0),
            scroll_position: Vec2::new(0.0, 120.0),
            horizontal: true,
            vertical: false,
            scroll_speed: 25.0,
            scrollbar: ScrollbarVisibility::Always,
            scrollbar_thickness: 4.0,
            scrollbar_color: [1.0, 1.0, 1.0, 0.5],
        });
        round_trip(ScrollFrame::default());
    }

    /// A radius is units plus a share of the shorter side, and never more than
    /// half of it, which is where the two corners on a side meet.
    #[test]
    fn a_corner_radius_is_units_plus_a_fraction_clamped_to_half_the_short_side() {
        let corner = Corner {
            radius: 10.0,
            radius_fraction: 0.1,
            ..Corner::default()
        };
        assert_eq!(corner.radius_px(Vec2::new(200.0, 100.0), 2.0), 30.0);
        let pill = Corner {
            radius: 0.0,
            radius_fraction: 0.5,
            ..Corner::default()
        };
        assert_eq!(pill.radius_px(Vec2::new(200.0, 40.0), 1.0), 20.0);
        let huge = Corner {
            radius: 1000.0,
            ..Corner::default()
        };
        assert_eq!(huge.radius_px(Vec2::new(60.0, 80.0), 1.0), 30.0);
        assert_eq!(Corner::default().exponent(), 2.0, "round is a circle");
        let squircle = Corner {
            style: CornerStyle::Squircle,
            exponent: 40.0,
            ..Corner::default()
        };
        assert_eq!(squircle.exponent(), 12.0);
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

    fn drawn(rect: &RectTransform, parent: Vec2) -> (Vec2, Vec2) {
        rect.rect_in(Vec2::ZERO, parent)
    }

    fn near(a: (Vec2, Vec2), b: (Vec2, Vec2)) -> bool {
        a.0.abs_diff_eq(b.0, 1e-3) && a.1.abs_diff_eq(b.1, 1e-3)
    }

    #[test]
    fn moving_the_anchors_or_the_pivot_leaves_the_element_where_it_was() {
        let parent = Vec2::new(800.0, 600.0);
        let mut rect = RectTransform {
            anchored_position: Vec2::new(30.0, -20.0),
            size_delta: Vec2::new(120.0, 40.0),
            ..RectTransform::default()
        };
        let before = drawn(&rect, parent);
        rect.set_anchors_keeping_rect(parent, Vec2::new(0.0, 0.25), Vec2::new(1.0, 0.25));
        assert!(near(drawn(&rect, parent), before), "{rect:?}");
        assert_eq!(rect.anchor_max, Vec2::new(1.0, 0.25));
        rect.set_pivot_keeping_rect(parent, Vec2::new(1.0, 0.0));
        assert!(near(drawn(&rect, parent), before), "{rect:?}");
        assert_eq!(rect.pivot, Vec2::new(1.0, 0.0));
    }

    #[test]
    fn a_preset_anchors_each_axis_and_shift_and_alt_take_the_pivot_and_position() {
        use AnchorPreset::*;
        let parent = Vec2::new(800.0, 600.0);
        let start = RectTransform {
            anchored_position: Vec2::new(30.0, -20.0),
            size_delta: Vec2::new(120.0, 40.0),
            ..RectTransform::default()
        };

        let mut plain = start;
        plain.apply_preset(parent, [Max, Min], false, false);
        assert_eq!(
            (plain.anchor_min, plain.anchor_max),
            (Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0))
        );
        assert_eq!(plain.pivot, start.pivot);
        assert!(
            near(drawn(&plain, parent), drawn(&start, parent)),
            "kept in place"
        );

        let mut pivoted = start;
        pivoted.apply_preset(parent, [Max, Min], true, false);
        assert_eq!(pivoted.pivot, Vec2::new(1.0, 0.0));
        assert!(near(drawn(&pivoted, parent), drawn(&start, parent)));

        let mut placed = start;
        placed.apply_preset(parent, [Max, Min], true, true);
        assert_eq!(
            placed.anchored_position,
            Vec2::ZERO,
            "in the top right corner"
        );
        assert_eq!(
            drawn(&placed, parent),
            (Vec2::new(680.0, 0.0), Vec2::new(120.0, 40.0))
        );

        let mut filled = start;
        filled.apply_preset(parent, [Stretch, Stretch], true, true);
        assert_eq!(
            filled,
            RectTransform::stretched(),
            "Alt on stretch fills the parent"
        );

        assert_eq!(AnchorPreset::of(1.0, 1.0), Some(Max));
        assert_eq!(AnchorPreset::of(0.0, 1.0), Some(Stretch));
        assert_eq!(AnchorPreset::of(0.2, 0.7), None);
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
