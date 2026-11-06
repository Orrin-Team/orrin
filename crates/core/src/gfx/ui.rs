//! The in-game UI, laid out: [`UiCanvas`] in, a list of screen-space quads out.
//!
//! Device-free, like `gfx::graph`, so the layout is asserted in CI rather than
//! eyeballed. `gfx::vulkan::ui` draws what this produces.
//!
//! # One quad per instance
//!
//! Everything the UI draws is an axis-aligned rectangle with a texture
//! rectangle and a colour — a panel, a glyph, an image — so a [`UiQuad`] is 36
//! bytes and the vertex shader builds its six corners from `gl_VertexIndex`.
//! That is a third of what a triangle list of `OverlayVertex`es would move, and
//! nothing here ever needs a shape that is not a rectangle.
//!
//! # Styled rectangles are a second stream
//!
//! A rounded, ringed, blurred or graded rectangle — what the retained UI's
//! `Corner`, `Stroke`, `Shadow` and `Gradient` ask for — is a [`UiShape`]: the
//! quad's fields plus the box it is cut from, the cut, and a two-colour blend,
//! 88 bytes, drawn by a pipeline of its own. The alternatives each charge every
//! glyph for a few panels: widening `UiQuad` to carry the style (uGUI's route,
//! with its spare texture coordinates) more than doubles a glyph, and an index
//! into a style buffer (closer to Bevy's per-node data) is four more bytes a
//! glyph and a storage binding the pass does not otherwise need. A separate
//! stream costs a plain quad nothing, and a styled element one pipeline switch
//! at each seam between the two — which is why the painter's call order, not
//! its type, decides where the seams are.
//!
//! Everything about a shape is evaluated per fragment, from its box: a signed
//! distance to a rounded box whose corners are quarter superellipses (exponent
//! 2 is a circle, a squircle is 4 or 5), an antialiased edge one pixel wide, a
//! ring as the distance's absolute value, and a drop shadow as the box
//! convolved with a Gaussian — exact across each row and summed down four, as
//! Evan Wallace's rounded-rectangle shadows do. None of it needs a texture, a
//! mask or a pass, so the render graph is untouched by the whole feature, and a
//! shadow costs only the pixels within three sigmas of its box.
//!
//! Gradient text is its glyphs drawn as shapes over the atlas, cut by nothing.
//! A glyph's outline is the glyph's coverage dilated by the stroke's width at
//! the moment it is rasterised, and kept in the atlas as a glyph of its own: the
//! atlas holds coverage, not distances, so growing the ink is the one way to
//! ring it, and a stroke that is drawn every frame is then paid for once.
//!
//! # Clipping
//!
//! A clip is cut on the CPU, like the screen's bounds always were: a quad loses
//! what lies outside and its texture rectangle shrinks in proportion. A shape's
//! box does not move when what it covers is cut, so a rounded panel half
//! scrolled out of a frame keeps the corners still in view.
//!
//! # Batches follow the call order
//!
//! Immediate mode means a later call draws over an earlier one, so the quads
//! stay in call order and a [`UiBatch`] is a maximal run sharing one texture.
//! A [`UiShapeBatch`] is the same for shapes, and records which quad batch it
//! goes before.
//! Panels and text share the glyph atlas — a panel samples a block of white
//! texels reserved at its corner — so a HUD of text on panels is one draw, and
//! only an image starts another. Images are sampled from the material texture
//! array by their handle, which a draw's push constant names; that keeps the
//! index dynamically uniform, which indexing a descriptor array needs without
//! `nonuniformEXT`.
//!
//! # Glyphs
//!
//! Rasterised by `ab_glyph` on first use at a whole-pixel size and kept in one
//! single-channel [`Atlas`] for the session. A glyph's quad is placed at a
//! whole-pixel origin, so its texels land one-to-one on the screen's and stay
//! sharp; a pen that rounds each glyph gives up sub-pixel positioning, which is
//! invisible at UI sizes. The atlas grows by doubling and, at its limit, is
//! emptied and refilled from the next frame — a frame never draws a glyph
//! whose texels have moved underneath it. Texture coordinates for the atlas are
//! in *texels*, not normalised, so growing it moves nothing that is already
//! laid out.
//!
//! # Fonts
//!
//! Slot 0 is JetBrains Mono (OFL), the face the editor already embeds, so the
//! default costs nothing the binary did not already carry. A project's `.ttf`
//! or `.otf` is an asset like any other, read through [`UiAssets::font`] on
//! first use. A font that will not load is a named error, once, and its text
//! is set in the default face rather than dropped.

use std::ops::Range;

use ab_glyph::{Font as _, FontArc, GlyphId, PxScale, ScaleFont as _};
use bytemuck::{Pod, Zeroable};
use orrin_ecs::FxHashMap;
use orrin_registry::AssetId;

use orrin_ecs::World;

use crate::asset_db::AssetDatabase;
use crate::gfx::{RenderBackend, TextureHandle};
use crate::scene::{Rect, ShapePaint, ShapeStyle, TextStyle, UiCanvas, UiCommand, UiStyle};

/// The engine's own face, shared with the editor's monospace family.
pub const DEFAULT_FONT: &[u8] = include_bytes!("../../assets/fonts/JetBrainsMono-Regular.ttf");

/// One rectangle on screen, as the UI pass's instance stream reads it.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct UiQuad {
    /// Left, top, right, bottom, in physical pixels.
    pub rect: [f32; 4],
    /// The same corners in the source: texels for the atlas, normalised for an
    /// image.
    pub uv: [f32; 4],
    /// sRGB, straight alpha.
    pub color: [u8; 4],
}

/// One rectangle of a styled element — cut to a rounded box, ringed, blurred
/// or coloured along a gradient — as the shape pipeline reads it.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct UiShape {
    /// What it covers: left, top, right, bottom, in physical pixels.
    pub rect: [f32; 4],
    /// The source at those corners, as [`UiQuad::uv`].
    pub uv: [f32; 4],
    /// The box it is cut from, which clipping never moves: centre and half
    /// extents.
    pub shape: [f32; 4],
    /// Corner radius, superellipse exponent, half the ring's width (0 fills),
    /// the blur's standard deviation (0 is a sharp edge).
    pub params: [f32; 4],
    /// The blend from `colors[0]` to `colors[1]`: a point is
    /// `dot(p - gradient.xy, gradient.zw)` of the way.
    pub gradient: [f32; 4],
    /// sRGB, straight alpha.
    pub colors: [[u8; 4]; 2],
}

/// A run of shapes drawn with one source, and where among the quad batches it
/// falls: before `batches[before]`, or after the last.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UiShapeBatch {
    pub source: UiSource,
    pub shapes: Range<u32>,
    pub before: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiSource {
    /// Glyph coverage, and the white block panels sample.
    Atlas,
    Texture(TextureHandle),
}

/// A run of quads drawn with one source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UiBatch {
    pub source: UiSource,
    pub quads: Range<u32>,
}

/// What the UI pass draws this frame.
#[derive(Clone, Copy)]
pub struct UiFrame<'a> {
    pub quads: &'a [UiQuad],
    pub batches: &'a [UiBatch],
    pub shapes: &'a [UiShape],
    pub shape_batches: &'a [UiShapeBatch],
    pub atlas: &'a Atlas,
}

impl UiFrame<'_> {
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty() && self.shapes.is_empty()
    }
}

/// Where a layer's commands land in the window, and what of the window they may
/// cover.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub origin: [f32; 2],
    /// Window pixels per command pixel.
    pub scale: f32,
    pub clip: Rect,
}

impl Placement {
    /// Commands in window pixels, cut to nothing.
    pub const UNBOUNDED: Self = Self {
        origin: [0.0; 2],
        scale: 1.0,
        clip: Rect::new(-1.0e9, -1.0e9, 2.0e9, 2.0e9),
    };

    /// Commands already in window pixels, cut to `clip`.
    pub fn clipped(clip: Rect) -> Self {
        Self {
            clip,
            ..Self::UNBOUNDED
        }
    }

    /// Commands in the game's pixels, drawn where `screen` is.
    pub fn screen(screen: &crate::scene::GameScreen) -> Self {
        Self {
            origin: screen.origin,
            scale: screen.scale,
            clip: screen.clip,
        }
    }

    fn place(&self, rect: &Rect) -> Rect {
        Rect::new(
            self.origin[0] + rect.x * self.scale,
            self.origin[1] + rect.y * self.scale,
            rect.width * self.scale,
            rect.height * self.scale,
        )
    }

    /// `style`'s lengths and points, in the window's pixels.
    fn style(&self, style: &UiStyle) -> UiStyle {
        let scale = self.scale;
        match *style {
            UiStyle::Outline { width } => UiStyle::Outline {
                width: width * scale,
            },
            UiStyle::Shape(shape) => UiStyle::Shape(ShapeStyle {
                bounds: shape.bounds.map(|bounds| self.place(&bounds)),
                radius: shape.radius * scale,
                exponent: shape.exponent,
                paint: match shape.paint {
                    ShapePaint::Fill => ShapePaint::Fill,
                    ShapePaint::Ring { width } => ShapePaint::Ring {
                        width: width * scale,
                    },
                    ShapePaint::Blur { sigma } => ShapePaint::Blur {
                        sigma: sigma * scale,
                    },
                },
                gradient: shape.gradient.map(|gradient| crate::scene::GradientPaint {
                    origin: [
                        self.origin[0] + gradient.origin[0] * scale,
                        self.origin[1] + gradient.origin[1] * scale,
                    ],
                    step: gradient.step.map(|step| step / scale),
                    ..gradient
                }),
            }),
        }
    }
}

/// What the painter asks of the world: the assets a canvas names.
pub trait UiAssets {
    /// The texture slot for image `id`, or `None` for one that cannot be drawn.
    fn texture(&mut self, id: AssetId) -> Option<TextureHandle>;
    /// The bytes of font `id`'s file, or why there are none.
    fn font(&mut self, id: AssetId) -> Result<Vec<u8>, String>;
}

/// Lay out this frame's retained UI and, over it, its [`UiCanvas`] into
/// `painter`. Runs after the asset
/// service and before the frame records, with the backend in hand, so an image
/// seen for the first time gets a texture slot here rather than in the script
/// tick.
pub fn paint(world: &World, backend: &mut impl RenderBackend, painter: &mut Painter) {
    let tree = world.get_resource::<crate::ui::UiTree>();
    let canvas = world.get_resource::<UiCanvas>();
    let mut db = world.get_resource_mut::<AssetDatabase>();
    if let Some(db) = &mut db {
        for id in db.take_font_changes() {
            painter.forget_font(id);
        }
    }
    let mut assets = WorldAssets {
        db: db.as_deref_mut(),
        backend,
    };
    // The retained layer is laid out in window pixels already; the immediate
    // one is in the game's, and both are cut to the game's screen.
    let (retained, immediate) = match world.get_resource::<crate::scene::GameScreen>() {
        Some(screen) => (Placement::clipped(screen.clip), Placement::screen(&screen)),
        None => (Placement::UNBOUNDED, Placement::UNBOUNDED),
    };
    let under = tree.as_ref().map(|tree| (&tree.canvas, retained));
    let over = canvas.as_deref().map(|canvas| (canvas, immediate));
    match (under, over) {
        (Some(under), Some(over)) => painter.paint_layers(&[under, over], &mut assets),
        (Some(only), None) | (None, Some(only)) => painter.paint_layers(&[only], &mut assets),
        (None, None) => painter.paint_layers(&[], &mut assets),
    }
}

/// The project's assets, as the painter asks for them.
struct WorldAssets<'a, B> {
    db: Option<&'a mut AssetDatabase>,
    backend: &'a mut B,
}

impl<B: RenderBackend> UiAssets for WorldAssets<'_, B> {
    fn texture(&mut self, id: AssetId) -> Option<TextureHandle> {
        self.db.as_mut()?.ui_texture(id, self.backend)
    }

    fn font(&mut self, id: AssetId) -> Result<Vec<u8>, String> {
        match &self.db {
            Some(db) => db.font(id),
            None => Err(format!("font asset {id} has no project to be read from")),
        }
    }
}

/// The glyph atlas: one byte of coverage per texel, packed in shelves.
pub struct Atlas {
    pixels: Vec<u8>,
    extent: [u32; 2],
    shelves: Vec<Shelf>,
    /// The first row no shelf has claimed.
    bottom: u32,
    /// Bumped whenever a texel changes, so the pass knows to upload.
    generation: u64,
}

#[derive(Clone, Copy)]
struct Shelf {
    y: u32,
    height: u32,
    /// The first column nothing on this shelf has claimed.
    x: u32,
}

/// Where a glyph's texels are, and where they go relative to the pen.
#[derive(Clone, Copy)]
struct Glyph {
    texels: [u32; 4],
    /// From the pen on the baseline to the bitmap's top left, in pixels.
    offset: [f32; 2],
}

/// A glyph of a font slot at a pixel size, grown by an outline of so many
/// pixels: packed in one word, since it is hashed and compared once for every
/// glyph drawn every frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphKey(u64);

impl GlyphKey {
    fn new(font: u32, glyph: u16, size: u16, grow: u8) -> Self {
        Self(
            u64::from(font as u16)
                | u64::from(grow) << 16
                | u64::from(glyph) << 32
                | u64::from(size) << 48,
        )
    }

    fn font(self) -> u32 {
        (self.0 & 0xffff) as u32
    }
}

/// How a text command's glyphs are drawn.
#[derive(Clone, Copy)]
enum Glyphs<'a> {
    Plain,
    /// Grown by this many pixels, as the ring round them.
    Outline(u8),
    /// As shapes, for the gradient they carry.
    Shaped(&'a ShapeStyle),
}

/// A shape's half extent when it is cut by nothing: further than any screen.
const UNCUT: f32 = 1.0e6;

/// The outline the painter grows a glyph by for a stroke `width` pixels wide:
/// whole pixels, since every grown glyph is a glyph of its own in the atlas,
/// plus one for the edge to fade over. `None` for a stroke too thin to see.
fn outline_px(width: f32) -> Option<u8> {
    let whole = width.round().min(f32::from(MAX_OUTLINE));
    (width > 0.0).then(|| whole.max(1.0) as u8 + 1)
}

/// The widest outline a glyph is grown by, in pixels: each pixel of it costs
/// the dilation a ring of taps round every texel, once per glyph and size.
const MAX_OUTLINE: u8 = 16;

/// `coverage`, `size` texels across, spread `grow` texels further on every
/// side: each texel takes the most of any ink within `grow - 1` of it, fading
/// out over the last texel so the ring's edge is antialiased.
fn dilate(coverage: &[u8], [width, height]: [u32; 2], grow: u32, out: &mut Vec<u8>) {
    let (wide, tall) = (width + 2 * grow, height + 2 * grow);
    out.clear();
    out.resize((wide * tall) as usize, 0);
    let reach = grow as i32;
    let edge = grow as f32;
    for oy in 0..tall as i32 {
        for ox in 0..wide as i32 {
            let (sx, sy) = (ox - reach, oy - reach);
            let mut best = 0.0f32;
            for dy in -reach..=reach {
                let y = sy + dy;
                if y < 0 || y >= height as i32 {
                    continue;
                }
                let row = &coverage[(y as u32 * width) as usize..][..width as usize];
                for dx in -reach..=reach {
                    let x = sx + dx;
                    if x < 0 || x >= width as i32 {
                        continue;
                    }
                    let ink = row[x as usize];
                    if ink == 0 {
                        continue;
                    }
                    let distance = ((dx * dx + dy * dy) as f32).sqrt();
                    let weight = (edge - distance).clamp(0.0, 1.0);
                    best = best.max(f32::from(ink) * weight);
                }
            }
            out[(oy as u32 * wide + ox as u32) as usize] = best.round() as u8;
        }
    }
}

/// Two sRGB colours multiplied channel by channel, as a tint multiplies.
fn multiply(a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
    std::array::from_fn(|i| ((u16::from(a[i]) * u16::from(b[i]) + 127) / 255) as u8)
}

/// Two `[left, top, right, bottom]` rectangles' overlap, inverted when there is
/// none — which [`clipped`] then refuses everything against.
fn overlap(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

/// One line of laid-out text: a byte range, how wide it sets, and where it
/// starts on its paragraph's running x.
#[derive(Clone)]
struct Line {
    bytes: Range<usize>,
    width: f32,
    origin: f32,
}

/// One inked character, where the wrap left its pen on the paragraph's
/// running x.
#[derive(Clone, Copy)]
struct Pen {
    byte: u32,
    glyph: GlyphId,
    x: f32,
}

/// The UI's layout state, kept across frames: fonts, the glyph cache and the
/// quads it last produced.
pub struct Painter {
    fonts: Vec<FontArc>,
    /// Each asset font's slot; a font that failed to load maps to slot 0.
    by_asset: FxHashMap<AssetId, u32>,
    /// Fonts whose files changed, and the slot each is read back into.
    reload: FxHashMap<AssetId, u32>,
    glyphs: FxHashMap<GlyphKey, Option<Glyph>>,
    /// Each character's glyph and unscaled advance, by font slot.
    metrics: FxHashMap<(u32, char), (GlyphId, f32)>,
    atlas: Atlas,
    /// The atlas filled up last frame and is emptied before this one.
    full: bool,
    quads: Vec<UiQuad>,
    batches: Vec<UiBatch>,
    shapes: Vec<UiShape>,
    shape_batches: Vec<UiShapeBatch>,
    /// Whether the last thing pushed was a shape, so the next quad starts a
    /// batch of its own after it.
    shaping: bool,
    lines: Vec<Line>,
    pens: Vec<Pen>,
    /// One glyph's coverage on its way into the atlas.
    scratch: Vec<u8>,
    /// The same, grown into an outline.
    grown: Vec<u8>,
}

impl Default for Painter {
    fn default() -> Self {
        Self::new()
    }
}

impl Painter {
    pub fn new() -> Self {
        let default = FontArc::try_from_slice(DEFAULT_FONT).expect("the built-in font parses");
        Self {
            fonts: vec![default],
            by_asset: FxHashMap::default(),
            reload: FxHashMap::default(),
            glyphs: FxHashMap::default(),
            metrics: FxHashMap::default(),
            atlas: Atlas::new(),
            full: false,
            quads: Vec::new(),
            batches: Vec::new(),
            shapes: Vec::new(),
            shape_batches: Vec::new(),
            shaping: false,
            lines: Vec::new(),
            pens: Vec::new(),
            scratch: Vec::new(),
            grown: Vec::new(),
        }
    }

    pub fn frame(&self) -> UiFrame<'_> {
        UiFrame {
            quads: &self.quads,
            batches: &self.batches,
            shapes: &self.shapes,
            shape_batches: &self.shape_batches,
            atlas: &self.atlas,
        }
    }

    /// Read font `id` again the next time it is drawn with: its file changed.
    /// It keeps its slot, and its glyphs are rasterised afresh.
    pub fn forget_font(&mut self, id: AssetId) {
        if let Some(slot) = self.by_asset.remove(&id)
            && slot != 0
        {
            self.glyphs.retain(|key, _| key.font() != slot);
            self.metrics.retain(|key, _| key.0 != slot);
            self.reload.insert(id, slot);
        }
    }

    /// Lay `canvas` out into this frame's quads.
    pub fn paint(&mut self, canvas: &UiCanvas, assets: &mut impl UiAssets) {
        self.paint_layers(&[(canvas, Placement::UNBOUNDED)], assets);
    }

    /// Lay `layers` out into this frame's quads, each over the one before.
    pub fn paint_layers(&mut self, layers: &[(&UiCanvas, Placement)], assets: &mut impl UiAssets) {
        if std::mem::take(&mut self.full) {
            self.glyphs.clear();
            self.atlas.reset();
        }
        self.quads.clear();
        self.batches.clear();
        self.shapes.clear();
        self.shape_batches.clear();
        self.shaping = false;
        for (canvas, placement) in layers {
            self.paint_canvas(canvas, placement, assets);
        }
    }

    fn paint_canvas(
        &mut self,
        canvas: &UiCanvas,
        placement: &Placement,
        assets: &mut impl UiAssets,
    ) {
        let layer = corners(&placement.clip);
        let mut bounds = layer;
        let place = |rect: &Rect| placement.place(rect);
        let mut styles = canvas.styles().iter().peekable();
        let mut clips = canvas.clips().iter().peekable();
        for (index, command) in canvas.commands().iter().enumerate() {
            let index = index as u32;
            while let Some((_, clip)) = clips.next_if(|(at, _)| *at <= index) {
                bounds = match clip {
                    Some(clip) => overlap(corners(&place(clip)), layer),
                    None => layer,
                };
            }
            let style = styles
                .next_if(|(at, _)| *at == index)
                .map(|(_, style)| placement.style(style));
            match command {
                UiCommand::Panel { rect, color } => {
                    let uv = self.atlas.white();
                    let rect = corners(&place(rect));
                    match &style {
                        Some(UiStyle::Shape(shape)) => {
                            self.push_shape(UiSource::Atlas, rect, uv, *color, shape, bounds)
                        }
                        _ => {
                            if let Some((rect, uv)) = clipped(rect, uv, bounds) {
                                let color = *color;
                                self.push(UiSource::Atlas, UiQuad { rect, uv, color });
                            }
                        }
                    }
                }
                UiCommand::Image {
                    rect,
                    texture,
                    uv,
                    tint,
                } => {
                    let rect = corners(&place(rect));
                    if clipped(rect, *uv, bounds).is_none() {
                        continue;
                    }
                    let Some(handle) = assets.texture(*texture) else {
                        continue;
                    };
                    let source = UiSource::Texture(handle);
                    match &style {
                        Some(UiStyle::Shape(shape)) => {
                            self.push_shape(source, rect, *uv, *tint, shape, bounds)
                        }
                        _ => {
                            if let Some((rect, uv)) = clipped(rect, *uv, bounds) {
                                let color = *tint;
                                self.push(source, UiQuad { rect, uv, color });
                            }
                        }
                    }
                }
                UiCommand::Text {
                    rect,
                    style: text_style,
                    text,
                } => {
                    let slot = self.font_slot(text_style.font, assets);
                    let text_style = TextStyle {
                        size: text_style.size * placement.scale,
                        ..*text_style
                    };
                    let glyphs = match &style {
                        None => Glyphs::Plain,
                        Some(UiStyle::Outline { width }) => match outline_px(*width) {
                            Some(grow) => Glyphs::Outline(grow),
                            None => continue,
                        },
                        Some(UiStyle::Shape(shape)) => Glyphs::Shaped(shape),
                    };
                    let text = canvas.str(text);
                    self.set_text(slot, &place(rect), &text_style, bounds, text, glyphs);
                }
            }
        }
    }

    fn push(&mut self, source: UiSource, quad: UiQuad) {
        let index = self.quads.len() as u32;
        self.quads.push(quad);
        match self.batches.last_mut() {
            Some(batch) if !self.shaping && batch.source == source => batch.quads.end = index + 1,
            _ => self.batches.push(UiBatch {
                source,
                quads: index..index + 1,
            }),
        }
        self.shaping = false;
    }

    /// `rect` with `uv` over it, cut to `bounds` and then by `style`'s shape.
    fn push_shape(
        &mut self,
        source: UiSource,
        rect: [f32; 4],
        uv: [f32; 4],
        color: [u8; 4],
        style: &ShapeStyle,
        bounds: [f32; 4],
    ) {
        let Some((rect, uv)) = clipped(rect, uv, bounds) else {
            return;
        };
        let shape = match style.bounds {
            Some(cut) => [
                cut.x + cut.width / 2.0,
                cut.y + cut.height / 2.0,
                cut.width / 2.0,
                cut.height / 2.0,
            ],
            None => [
                (rect[0] + rect[2]) / 2.0,
                (rect[1] + rect[3]) / 2.0,
                UNCUT,
                UNCUT,
            ],
        };
        let (ring, sigma) = match style.paint {
            ShapePaint::Fill => (0.0, 0.0),
            ShapePaint::Ring { width } => (width / 2.0, 0.0),
            ShapePaint::Blur { sigma } => (0.0, sigma),
        };
        let (gradient, colors) = match style.gradient {
            Some(gradient) => (
                [
                    gradient.origin[0],
                    gradient.origin[1],
                    gradient.step[0],
                    gradient.step[1],
                ],
                [
                    multiply(color, gradient.start),
                    multiply(color, gradient.end),
                ],
            ),
            None => ([0.0; 4], [color; 2]),
        };
        let index = self.shapes.len() as u32;
        self.shapes.push(UiShape {
            rect,
            uv,
            shape,
            params: [style.radius, style.exponent, ring, sigma],
            gradient,
            colors,
        });
        match self.shape_batches.last_mut() {
            Some(batch) if self.shaping && batch.source == source => {
                batch.shapes.end = index + 1;
            }
            _ => self.shape_batches.push(UiShapeBatch {
                source,
                shapes: index..index + 1,
                before: self.batches.len() as u32,
            }),
        }
        self.shaping = true;
    }

    fn font_slot(&mut self, font: Option<AssetId>, assets: &mut impl UiAssets) -> u32 {
        let Some(id) = font else {
            return 0;
        };
        if let Some(&slot) = self.by_asset.get(&id) {
            return slot;
        }
        let loaded = assets.font(id).and_then(|bytes| {
            FontArc::try_from_vec(bytes)
                .map_err(|_| format!("font asset {id} is not a TrueType or OpenType font"))
        });
        let slot = match loaded {
            Ok(font) => match self.reload.remove(&id) {
                Some(slot) => {
                    self.fonts[slot as usize] = font;
                    slot
                }
                None => {
                    self.fonts.push(font);
                    self.fonts.len() as u32 - 1
                }
            },
            Err(message) => {
                tracing::error!("{message}, so its text is set in the built-in font");
                0
            }
        };
        self.by_asset.insert(id, slot);
        slot
    }

    /// Wrap `text` to `rect`'s width, place the block by `style`'s anchor, and
    /// push a quad per inked glyph, clipped to `rect`.
    /// Nothing is drawn outside `bounds`, whatever the style allows.
    fn set_text(
        &mut self,
        slot: u32,
        rect: &Rect,
        style: &TextStyle,
        bounds: [f32; 4],
        text: &str,
        glyphs: Glyphs<'_>,
    ) {
        if bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
            return;
        }
        let grow = match glyphs {
            Glyphs::Outline(grow) => grow,
            _ => 0,
        };
        // A cheap handle, cloned so the caches can be borrowed mutably.
        let font = self.fonts[slot as usize].clone();
        let size = style.size.round().clamp(1.0, f32::from(u16::MAX));
        let scaled = font.as_scaled(PxScale::from(size));
        let line_height = scaled.height() + scaled.line_gap();

        let mut lines = std::mem::take(&mut self.lines);
        let mut pens = std::mem::take(&mut self.pens);
        let scale = scaled.h_scale_factor();
        let width = if style.wrap {
            rect.width
        } else {
            f32::INFINITY
        };
        self.wrap(slot, &font, scale, text, width, &mut lines, &mut pens);
        let block = line_height * (lines.len() - 1) as f32 + scaled.height();
        let margin = f32::from(grow);
        let [left, top, right, bottom] = corners(rect);
        let mut clip = [left - margin, top - margin, right + margin, bottom + margin];
        if !style.wrap {
            clip[0] = f32::NEG_INFINITY;
            clip[2] = f32::INFINITY;
        }
        let clip = [
            clip[0].max(bounds[0]),
            clip[1].max(bounds[1]),
            clip[2].min(bounds[2]),
            clip[3].min(bounds[3]),
        ];
        let mut baseline =
            rect.y + (rect.height - block) * style.anchor.vertical() + scaled.ascent();

        let mut next = 0;
        for line in &lines {
            if baseline - scaled.ascent() >= clip[3] {
                break;
            }
            let first = next;
            while next < pens.len() && (pens[next].byte as usize) < line.bytes.end {
                next += 1;
            }
            if baseline - scaled.descent() > clip[1] {
                let y = baseline.round();
                let left = rect.x + (rect.width - line.width) * style.anchor.horizontal();
                for pen in &pens[first..next] {
                    let Some(glyph) = self.glyph(slot, &font, pen.glyph, size, grow) else {
                        continue;
                    };
                    let x = (left + pen.x - line.origin).round() + glyph.offset[0];
                    let top = y + glyph.offset[1];
                    let [u0, v0, u1, v1] = glyph.texels.map(|t| t as f32);
                    let rect = [x, top, x + (u1 - u0), top + (v1 - v0)];
                    let uv = [u0, v0, u1, v1];
                    let color = style.color;
                    match glyphs {
                        Glyphs::Shaped(shape) => {
                            self.push_shape(UiSource::Atlas, rect, uv, color, shape, clip);
                        }
                        _ => {
                            if let Some((rect, uv)) = clipped(rect, uv, clip) {
                                self.push(UiSource::Atlas, UiQuad { rect, uv, color });
                            }
                        }
                    }
                }
            }
            baseline += line_height;
        }
        self.lines = lines;
        self.pens = pens;
    }

    /// Break `text` into the lines it sets in a box `max` pixels wide: at each
    /// newline, at the last whitespace before a line overflows, and between
    /// characters for a word wider than the box on its own. Whitespace a line
    /// breaks at belongs to neither line, so it moves no anchor.
    ///
    /// Measuring a line already finds where each glyph stands on it, so that is
    /// kept in `pens` rather than found a second time to place them: a
    /// character costs one cached lookup and one kerning pair.
    #[allow(clippy::too_many_arguments)]
    fn wrap(
        &mut self,
        slot: u32,
        font: &FontArc,
        scale: f32,
        text: &str,
        max: f32,
        lines: &mut Vec<Line>,
        pens: &mut Vec<Pen>,
    ) {
        /// The whitespace run a line can break at, by byte and running x.
        #[derive(Clone, Copy)]
        struct Break {
            end: usize,
            end_x: f32,
            next: usize,
            next_x: f32,
        }

        lines.clear();
        pens.clear();
        let mut offset = 0;
        for paragraph in text.split('\n') {
            let mut start = offset;
            // The paragraph's running x, and where on it the line began.
            let mut x = 0.0;
            let mut origin = 0.0;
            let mut previous = None;
            let mut gap: Option<Break> = None;
            for (index, c) in paragraph.char_indices() {
                let index = offset + index;
                let (id, advance) = self.metrics(slot, font, c);
                let advance = advance * scale;
                let mut kern =
                    previous.map_or(0.0, |previous| font.kern_unscaled(previous, id) * scale);
                if c.is_whitespace() {
                    match &mut gap {
                        Some(run) if run.next == index => run.next = index + c.len_utf8(),
                        _ => {
                            gap = Some(Break {
                                end: index,
                                end_x: x,
                                next: index + c.len_utf8(),
                                next_x: 0.0,
                            });
                        }
                    }
                    x += kern + advance;
                    if let Some(run) = &mut gap {
                        run.next_x = x;
                    }
                } else {
                    if x + kern + advance - origin > max && index > start {
                        if let Some(run) = gap.take().filter(|run| run.end > start) {
                            lines.push(Line {
                                bytes: start..run.end,
                                width: run.end_x - origin,
                                origin,
                            });
                            start = run.next;
                            origin = run.next_x;
                        }
                        if x + kern + advance - origin > max && index > start {
                            lines.push(Line {
                                bytes: start..index,
                                width: x - origin,
                                origin,
                            });
                            start = index;
                            origin = x;
                            kern = 0.0;
                        }
                    }
                    pens.push(Pen {
                        byte: index as u32,
                        glyph: id,
                        x: x + kern,
                    });
                    x += kern + advance;
                }
                previous = Some(id);
            }
            let end = offset + paragraph.len();
            let line = match gap {
                Some(run) if run.next == end && run.end >= start => Line {
                    bytes: start..run.end,
                    width: run.end_x - origin,
                    origin,
                },
                _ => Line {
                    bytes: start..end,
                    width: x - origin,
                    origin,
                },
            };
            lines.push(line);
            offset = end + 1;
        }
    }

    /// Character `c`'s glyph in font `slot` and its unscaled advance, looked up
    /// in the font once.
    fn metrics(&mut self, slot: u32, font: &FontArc, c: char) -> (GlyphId, f32) {
        *self.metrics.entry((slot, c)).or_insert_with(|| {
            let id = font.glyph_id(c);
            (id, font.h_advance_unscaled(id))
        })
    }

    /// Glyph `id` of font `slot` at `size`, rasterised into the atlas the
    /// first time it is asked for. `None` for a glyph with no ink, and for one
    /// the atlas has no room for, which is not remembered.
    ///
    /// With `grow`, the coverage is spread that many pixels further on every
    /// side — the glyph's outline — and kept as a glyph of its own.
    fn glyph(
        &mut self,
        slot: u32,
        font: &FontArc,
        id: GlyphId,
        size: f32,
        grow: u8,
    ) -> Option<Glyph> {
        let key = GlyphKey::new(slot, id.0, size as u16, grow);
        match self.glyphs.get(&key) {
            Some(&glyph) => glyph,
            None => self.rasterise(key, font, id, size, grow),
        }
    }

    /// [`glyph`](Self::glyph)'s first sight of a glyph. Out of line, so the
    /// lookup every other frame makes stays small in the loop that makes it.
    #[cold]
    #[inline(never)]
    fn rasterise(
        &mut self,
        key: GlyphKey,
        font: &FontArc,
        id: GlyphId,
        size: f32,
        grow: u8,
    ) -> Option<Glyph> {
        let glyph = match font.outline_glyph(id.with_scale(PxScale::from(size))) {
            None => None,
            Some(outlined) => {
                let bounds = outlined.px_bounds();
                let ink = [bounds.width() as u32, bounds.height() as u32];
                let margin = u32::from(grow);
                let size = ink.map(|side| side + 2 * margin);
                let Some(at) = self.atlas.allocate(size) else {
                    self.full = true;
                    return None;
                };
                let [width, height] = ink;
                self.scratch.clear();
                self.scratch.resize((width * height) as usize, 0);
                outlined.draw(|x, y, coverage| {
                    self.scratch[(y * width + x) as usize] =
                        (coverage.clamp(0.0, 1.0) * 255.0).round() as u8;
                });
                if grow == 0 {
                    self.atlas.write_rect(at, size, &self.scratch);
                } else {
                    let mut grown = std::mem::take(&mut self.grown);
                    dilate(&self.scratch, ink, margin, &mut grown);
                    self.atlas.write_rect(at, size, &grown);
                    self.grown = grown;
                }
                let margin = margin as f32;
                Some(Glyph {
                    texels: [at[0], at[1], at[0] + size[0], at[1] + size[1]],
                    offset: [bounds.min.x - margin, bounds.min.y - margin],
                })
            }
        };
        self.glyphs.insert(key, glyph);
        glyph
    }
}

fn corners(rect: &Rect) -> [f32; 4] {
    let [right, bottom] = rect.max();
    [rect.x, rect.y, right, bottom]
}

/// `rect` cut to `clip`, with `uv` cut in proportion; `None` if nothing is
/// left.
fn clipped(rect: [f32; 4], uv: [f32; 4], clip: [f32; 4]) -> Option<([f32; 4], [f32; 4])> {
    let cut = [
        rect[0].max(clip[0]),
        rect[1].max(clip[1]),
        rect[2].min(clip[2]),
        rect[3].min(clip[3]),
    ];
    if cut[0] >= cut[2] || cut[1] >= cut[3] {
        return None;
    }
    if cut == rect {
        return Some((rect, uv));
    }
    let scale = [
        (uv[2] - uv[0]) / (rect[2] - rect[0]),
        (uv[3] - uv[1]) / (rect[3] - rect[1]),
    ];
    let uv = [
        uv[0] + (cut[0] - rect[0]) * scale[0],
        uv[1] + (cut[1] - rect[1]) * scale[1],
        uv[2] - (rect[2] - cut[2]) * scale[0],
        uv[3] - (rect[3] - cut[3]) * scale[1],
    ];
    Some((cut, uv))
}

impl Atlas {
    const INITIAL: u32 = 256;
    const MAX: u32 = 2048;
    /// The block a panel samples; filtering inside it reads white.
    const WHITE: u32 = 4;

    fn new() -> Self {
        let mut atlas = Self {
            pixels: Vec::new(),
            extent: [0, 0],
            shelves: Vec::new(),
            bottom: 0,
            generation: 0,
        };
        atlas.reset();
        atlas
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Rows `0..` this have ever been written; the rest has never been sampled.
    pub fn used_rows(&self) -> u32 {
        self.bottom
    }

    /// The panel's texture rectangle: the middle of the white block.
    fn white(&self) -> [f32; 4] {
        [1.5, 1.5, 2.5, 2.5]
    }

    fn reset(&mut self) {
        self.extent = [Self::INITIAL; 2];
        self.pixels.clear();
        self.pixels
            .resize((Self::INITIAL * Self::INITIAL) as usize, 0);
        self.shelves.clear();
        self.bottom = 0;
        self.generation += 1;
        let white = self
            .allocate([Self::WHITE; 2])
            .expect("an empty atlas fits");
        self.write_rect(
            white,
            [Self::WHITE; 2],
            &[255; (Self::WHITE * Self::WHITE) as usize],
        );
    }

    /// A free `size` rectangle's top left, growing the atlas if it must, or
    /// `None` once it is as large as it may get. Each rectangle keeps a texel
    /// of gutter to its right and below, so a filtered tap at its edge never
    /// reads a neighbour.
    fn allocate(&mut self, [width, height]: [u32; 2]) -> Option<[u32; 2]> {
        let size = [width + 1, height + 1];
        loop {
            if let Some(at) = self.place(size) {
                return Some(at);
            }
            if self.extent[0] >= Self::MAX {
                return None;
            }
            self.grow();
        }
    }

    /// The best-fitting shelf with room, or a new one under the last.
    fn place(&mut self, [width, height]: [u32; 2]) -> Option<[u32; 2]> {
        let fits = |shelf: &Shelf| {
            // A shelf much taller than the glyph wastes the difference for
            // every glyph after it; a new shelf is cheaper.
            shelf.height >= height
                && shelf.height <= height + height / 2
                && shelf.x + width <= self.extent[0]
        };
        if let Some(shelf) = self
            .shelves
            .iter_mut()
            .filter(|shelf| fits(shelf))
            .min_by_key(|shelf| shelf.height)
        {
            let at = [shelf.x, shelf.y];
            shelf.x += width;
            return Some(at);
        }
        if width > self.extent[0] || self.bottom + height > self.extent[1] {
            return None;
        }
        let at = [0, self.bottom];
        self.shelves.push(Shelf {
            y: self.bottom,
            height,
            x: width,
        });
        self.bottom += height;
        Some(at)
    }

    /// Double both sides. Every texel keeps its coordinates, which is what lets
    /// a glyph already laid out survive it.
    fn grow(&mut self) {
        let [width, height] = self.extent;
        let wider = (width * 2) as usize;
        let mut pixels = vec![0; wider * (height * 2) as usize];
        for (row, old) in self
            .pixels
            .chunks_exact(width as usize)
            .take(self.bottom as usize)
            .enumerate()
        {
            pixels[row * wider..row * wider + width as usize].copy_from_slice(old);
        }
        self.pixels = pixels;
        self.extent = [width * 2, height * 2];
        self.generation += 1;
    }

    fn write_rect(&mut self, at: [u32; 2], [width, height]: [u32; 2], coverage: &[u8]) {
        let stride = self.extent[0] as usize;
        for (row, texels) in coverage
            .chunks_exact(width as usize)
            .take(height as usize)
            .enumerate()
        {
            let start = (at[1] as usize + row) * stride + at[0] as usize;
            self.pixels[start..start + width as usize].copy_from_slice(texels);
        }
        self.generation += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{GameScreen, TextAnchor};

    /// Answers every texture with its slot and every font with the built-in
    /// face's bytes, counting what it was asked.
    #[derive(Default)]
    struct Fake {
        textures: FxHashMap<AssetId, TextureHandle>,
        fonts_asked: usize,
        broken_fonts: bool,
    }

    impl UiAssets for Fake {
        fn texture(&mut self, id: AssetId) -> Option<TextureHandle> {
            self.textures.get(&id).copied()
        }

        fn font(&mut self, _: AssetId) -> Result<Vec<u8>, String> {
            self.fonts_asked += 1;
            if self.broken_fonts {
                Err("fonts/broken.ttf: not a font".to_owned())
            } else {
                Ok(DEFAULT_FONT.to_vec())
            }
        }
    }

    fn style(anchor: TextAnchor) -> TextStyle {
        TextStyle {
            font: None,
            size: 20.0,
            anchor,
            color: [255, 255, 255, 255],
            wrap: true,
        }
    }

    fn paint(canvas: &UiCanvas) -> Painter {
        let mut painter = Painter::new();
        painter.paint(canvas, &mut Fake::default());
        painter
    }

    fn text(rect: Rect, anchor: TextAnchor, text: &str) -> UiCanvas {
        let mut canvas = UiCanvas::default();
        canvas.begin([1280, 720]);
        canvas.text(rect, style(anchor), text);
        canvas
    }

    /// The smallest rectangle round every quad.
    fn bounds(quads: &[UiQuad]) -> [f32; 4] {
        quads.iter().fold(
            [f32::MAX, f32::MAX, f32::MIN, f32::MIN],
            |[l, t, r, b], quad| {
                [
                    l.min(quad.rect[0]),
                    t.min(quad.rect[1]),
                    r.max(quad.rect[2]),
                    b.max(quad.rect[3]),
                ]
            },
        )
    }

    /// How many lines the quads sit on: their bottoms, clustered. A line's
    /// glyphs disagree by an overshoot; two lines, by a line height.
    fn rows(quads: &[UiQuad]) -> usize {
        let mut bottoms: Vec<f32> = quads.iter().map(|q| q.rect[3]).collect();
        bottoms.sort_by(f32::total_cmp);
        1 + bottoms
            .windows(2)
            .filter(|pair| pair[1] - pair[0] > 5.0)
            .count()
    }

    fn inside(quad: &UiQuad, rect: Rect) -> bool {
        let [right, bottom] = rect.max();
        quad.rect[0] >= rect.x
            && quad.rect[1] >= rect.y
            && quad.rect[2] <= right
            && quad.rect[3] <= bottom
    }

    #[test]
    fn a_panel_is_one_quad_over_white_texels() {
        let mut canvas = UiCanvas::default();
        canvas.panel(Rect::new(10.0, 20.0, 300.0, 40.0), [20, 40, 60, 200]);
        let painter = paint(&canvas);
        let frame = painter.frame();

        assert_eq!(frame.quads.len(), 1);
        let quad = frame.quads[0];
        assert_eq!(quad.rect, [10.0, 20.0, 310.0, 60.0]);
        assert_eq!(quad.color, [20, 40, 60, 200]);
        assert_eq!(
            frame.batches,
            [UiBatch {
                source: UiSource::Atlas,
                quads: 0..1
            }]
        );
        let width = frame.atlas.extent()[0] as usize;
        for (x, y) in [(quad.uv[0], quad.uv[1]), (quad.uv[2], quad.uv[3])] {
            // Every texel a bilinear tap at the corner can reach.
            for (tx, ty) in [(x - 0.5, y - 0.5), (x + 0.5, y + 0.5)] {
                let texel = frame.atlas.pixels()[ty as usize * width + tx as usize];
                assert_eq!(texel, 255, "({tx}, {ty}) is not white");
            }
        }
    }

    #[test]
    fn a_label_sets_one_glyph_per_inked_character_on_one_baseline() {
        let rect = Rect::new(100.0, 50.0, 400.0, 40.0);
        let painter = paint(&text(rect, TextAnchor::UpperLeft, "HH H"));
        let quads = painter.frame().quads;

        assert_eq!(quads.len(), 3, "the space has no ink and no quad");
        assert!(quads.iter().all(|q| q.rect[1] == quads[0].rect[1]));
        assert!(quads.iter().all(|q| q.rect[3] == quads[0].rect[3]));
        assert!(quads.iter().all(|q| inside(q, rect)), "{quads:?}");
        // Upper left: the ink starts within a bearing of the corner.
        let [left, top, ..] = bounds(quads);
        assert!(left - rect.x < 4.0, "{left}");
        assert!(top - rect.y < 8.0, "{top}");
        // Whole pixels, so the texels land one-to-one on the screen's.
        for quad in quads {
            assert!(quad.rect.iter().all(|v| v.fract() == 0.0), "{quad:?}");
            assert_eq!(quad.rect[2] - quad.rect[0], quad.uv[2] - quad.uv[0]);
            assert_eq!(quad.rect[3] - quad.rect[1], quad.uv[3] - quad.uv[1]);
        }
    }

    #[test]
    fn an_anchor_moves_the_line_by_its_share_of_the_slack() {
        let rect = Rect::new(0.0, 0.0, 600.0, 200.0);
        let left = bounds(
            paint(&text(rect, TextAnchor::UpperLeft, "Score"))
                .frame()
                .quads,
        );
        let right = bounds(
            paint(&text(rect, TextAnchor::UpperRight, "Score"))
                .frame()
                .quads,
        );
        let centre = bounds(
            paint(&text(rect, TextAnchor::MiddleCenter, "Score"))
                .frame()
                .quads,
        );
        let low = bounds(
            paint(&text(rect, TextAnchor::LowerLeft, "Score"))
                .frame()
                .quads,
        );

        let shift = right[0] - left[0];
        assert!(
            (shift - (right[2] - left[2])).abs() < 1.0,
            "the line moves whole"
        );
        assert!(
            rect.width - right[2] < 4.0,
            "right-aligned ink ends at the edge: {right:?}"
        );
        let gaps = (centre[0] - rect.x, rect.width - centre[2]);
        assert!((gaps.0 - gaps.1).abs() < 4.0, "centred: {gaps:?}");
        assert!(((centre[1] + centre[3]) / 2.0 - rect.height / 2.0).abs() < 6.0);
        assert!(
            rect.height - low[3] < 10.0,
            "bottom-aligned ink is a descent off the edge"
        );
    }

    #[test]
    fn text_wraps_at_whitespace_to_its_width() {
        let rect = Rect::new(0.0, 0.0, 50.0, 300.0);
        let painter = paint(&text(rect, TextAnchor::UpperLeft, "aaa bbb ccc"));
        let quads = painter.frame().quads;

        assert_eq!(quads.len(), 9);
        assert_eq!(rows(quads), 3, "{quads:?}");
        assert!(quads.iter().all(|q| inside(q, rect)), "{quads:?}");
    }

    #[test]
    fn a_word_wider_than_the_rect_breaks_between_characters() {
        let rect = Rect::new(0.0, 0.0, 40.0, 300.0);
        let painter = paint(&text(rect, TextAnchor::UpperLeft, "abcdefghij"));
        let quads = painter.frame().quads;
        assert_eq!(quads.len(), 10, "nothing is dropped");
        assert!(rows(quads) > 1);
        assert!(quads.iter().all(|q| inside(q, rect)), "{quads:?}");
    }

    #[test]
    fn a_newline_starts_a_line() {
        let rect = Rect::new(0.0, 0.0, 600.0, 300.0);
        let painter = paint(&text(rect, TextAnchor::UpperLeft, "x\nx\n\nx"));
        let quads = painter.frame().quads;
        assert_eq!(quads.len(), 3);
        let gap = quads[1].rect[3] - quads[0].rect[3];
        assert!(gap > 0.0);
        let skipped = quads[2].rect[3] - quads[1].rect[3];
        assert!(
            (skipped - 2.0 * gap).abs() <= 1.0,
            "an empty line keeps its height"
        );
    }

    #[test]
    fn text_is_clipped_to_its_rect_texels_and_all() {
        let rect = Rect::new(0.0, 0.0, 600.0, 12.0);
        let painter = paint(&text(rect, TextAnchor::UpperLeft, "Clipped\nhidden"));
        let quads = painter.frame().quads;

        assert!(!quads.is_empty());
        assert!(quads.iter().all(|q| inside(q, rect)), "{quads:?}");
        assert_eq!(rows(quads), 1, "the second line is wholly outside");
        // A clipped glyph samples the matching part of its texels.
        for quad in quads {
            let scale = (quad.uv[3] - quad.uv[1]) / (quad.rect[3] - quad.rect[1]);
            assert!((scale - 1.0).abs() < 1e-4, "{quad:?}");
        }
    }

    #[test]
    fn a_glyph_is_rasterised_once() {
        let canvas = text(
            Rect::new(0.0, 0.0, 600.0, 60.0),
            TextAnchor::UpperLeft,
            "again",
        );
        let mut painter = Painter::new();
        let mut assets = Fake::default();
        painter.paint(&canvas, &mut assets);
        let generation = painter.frame().atlas.generation();
        let first = painter.frame().quads.to_vec();
        painter.paint(&canvas, &mut assets);
        assert_eq!(painter.frame().atlas.generation(), generation);
        assert_eq!(painter.frame().quads, first);
    }

    #[test]
    fn a_batch_breaks_only_where_the_source_changes() {
        let icon = AssetId::new();
        let mut canvas = UiCanvas::default();
        let rect = Rect::new(0.0, 0.0, 200.0, 40.0);
        canvas.panel(rect, [0, 0, 0, 255]);
        canvas.text(rect, style(TextAnchor::UpperLeft), "hp");
        canvas.image(rect, icon, [255; 4]);
        canvas.image(rect, icon, [255; 4]);
        canvas.panel(rect, [0, 0, 0, 255]);
        let mut assets = Fake::default();
        assets.textures.insert(icon, TextureHandle(7));
        let mut painter = Painter::new();
        painter.paint(&canvas, &mut assets);

        assert_eq!(
            painter.frame().batches,
            [
                UiBatch {
                    source: UiSource::Atlas,
                    quads: 0..3
                },
                UiBatch {
                    source: UiSource::Texture(TextureHandle(7)),
                    quads: 3..5
                },
                UiBatch {
                    source: UiSource::Atlas,
                    quads: 5..6
                },
            ]
        );
        assert_eq!(painter.frame().quads[3].uv, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn an_image_region_samples_only_its_part_of_the_texture() {
        let icon = AssetId::new();
        let mut canvas = UiCanvas::default();
        let rect = Rect::new(10.0, 10.0, 32.0, 32.0);
        canvas.image_region(rect, icon, [0.25, 0.0, 0.75, 0.5], [255; 4]);
        let mut assets = Fake::default();
        assets.textures.insert(icon, TextureHandle(3));
        let mut painter = Painter::new();
        painter.paint(&canvas, &mut assets);
        assert_eq!(painter.frame().quads[0].uv, [0.25, 0.0, 0.75, 0.5]);
        assert_eq!(painter.frame().quads[0].rect, [10.0, 10.0, 42.0, 42.0]);
    }

    #[test]
    fn unwrapped_text_runs_on_one_line_past_its_rect() {
        let rect = Rect::new(0.0, 0.0, 50.0, 300.0);
        let mut canvas = UiCanvas::default();
        let unwrapped = TextStyle {
            wrap: false,
            ..style(TextAnchor::UpperLeft)
        };
        canvas.text(rect, unwrapped, "aaa bbb ccc");
        let painter = paint(&canvas);
        let quads = painter.frame().quads;
        assert_eq!(quads.len(), 9, "nothing is cut at the sides");
        assert_eq!(rows(quads), 1, "{quads:?}");
        assert!(bounds(quads)[2] > rect.width, "{quads:?}");
    }

    /// The immediate canvas is in the game's pixels: a previewed screen draws
    /// it scaled into its place in the window, and nothing of it outside.
    #[test]
    fn a_layer_is_scaled_into_its_screen_and_cut_to_it() {
        let mut canvas = UiCanvas::default();
        canvas.panel(Rect::new(0.0, 0.0, 100.0, 100.0), [255; 4]);
        canvas.panel(Rect::new(-50.0, 0.0, 100.0, 100.0), [255; 4]);
        let label = Rect::new(0.0, 200.0, 600.0, 100.0);
        canvas.text(label, style(TextAnchor::UpperLeft), "H");
        let screen = GameScreen {
            size: [1920.0, 1080.0],
            origin: [10.0, 20.0],
            scale: 0.5,
            clip: Rect::new(10.0, 20.0, 960.0, 540.0),
        };
        let mut painter = Painter::new();
        painter.paint_layers(
            &[(&canvas, Placement::screen(&screen))],
            &mut Fake::default(),
        );
        let quads = painter.frame().quads;
        assert_eq!(quads.len(), 3, "{quads:?}");
        assert_eq!(quads[0].rect, [10.0, 20.0, 60.0, 70.0]);
        assert_eq!(
            quads[1].rect,
            [10.0, 20.0, 35.0, 70.0],
            "cut at the screen's edge"
        );

        let full = paint(&text(label, TextAnchor::UpperLeft, "H"));
        let height = |quad: &UiQuad| quad.rect[3] - quad.rect[1];
        let ratio = height(&quads[2]) / height(&full.frame().quads[0]);
        assert!((ratio - 0.5).abs() < 0.1, "set at half the size: {ratio}");
        assert!(quads[2].rect[1] >= 120.0, "below its place in the screen");
    }

    /// The retained layer goes under the immediate one, and a panel on each is
    /// still one draw.
    #[test]
    fn layers_paint_in_order_and_share_a_batch_across_the_seam() {
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        let mut under = UiCanvas::default();
        under.panel(rect, [1, 0, 0, 255]);
        let mut over = UiCanvas::default();
        over.panel(rect, [2, 0, 0, 255]);
        let mut painter = Painter::new();
        painter.paint_layers(
            &[
                (&under, Placement::UNBOUNDED),
                (&over, Placement::UNBOUNDED),
            ],
            &mut Fake::default(),
        );
        let frame = painter.frame();
        let colours: Vec<u8> = frame.quads.iter().map(|q| q.color[0]).collect();
        assert_eq!(colours, [1, 2]);
        assert_eq!(frame.batches.len(), 1);
    }

    #[test]
    fn the_retained_tree_draws_under_the_immediate_calls() {
        let mut world = World::new();
        let mut tree = crate::ui::UiTree::default();
        tree.canvas
            .panel(Rect::new(0.0, 0.0, 10.0, 10.0), [1, 0, 0, 255]);
        world.insert_resource(tree);
        let mut immediate = UiCanvas::default();
        immediate.panel(Rect::new(0.0, 0.0, 10.0, 10.0), [2, 0, 0, 255]);
        world.insert_resource(immediate);

        let mut painter = Painter::new();
        super::paint(
            &world,
            &mut crate::gfx::HeadlessBackend::new(),
            &mut painter,
        );
        let colours: Vec<u8> = painter.frame().quads.iter().map(|q| q.color[0]).collect();
        assert_eq!(colours, [1, 2]);
    }

    #[test]
    fn an_image_that_does_not_resolve_draws_nothing() {
        let mut canvas = UiCanvas::default();
        canvas.image(Rect::new(0.0, 0.0, 64.0, 64.0), AssetId::new(), [255; 4]);
        assert!(paint(&canvas).frame().is_empty());
    }

    #[test]
    fn a_font_is_read_once_and_again_after_it_changes() {
        let font = AssetId::new();
        let mut canvas = UiCanvas::default();
        let styled = TextStyle {
            font: Some(font),
            ..style(TextAnchor::UpperLeft)
        };
        canvas.text(Rect::new(0.0, 0.0, 300.0, 40.0), styled, "own face");
        let mut painter = Painter::new();
        let mut assets = Fake::default();
        painter.paint(&canvas, &mut assets);
        painter.paint(&canvas, &mut assets);
        assert_eq!(assets.fonts_asked, 1);
        assert_eq!(painter.frame().quads.len(), 7);

        painter.forget_font(font);
        painter.paint(&canvas, &mut assets);
        assert_eq!(assets.fonts_asked, 2);
    }

    #[test]
    fn a_font_that_does_not_load_sets_its_text_in_the_default_face() {
        let font = AssetId::new();
        let mut canvas = UiCanvas::default();
        let styled = TextStyle {
            font: Some(font),
            ..style(TextAnchor::UpperLeft)
        };
        canvas.text(Rect::new(0.0, 0.0, 300.0, 40.0), styled, "fallback");
        let mut painter = Painter::new();
        let mut assets = Fake {
            broken_fonts: true,
            ..Fake::default()
        };
        painter.paint(&canvas, &mut assets);
        painter.paint(&canvas, &mut assets);
        assert_eq!(
            assets.fonts_asked, 1,
            "a broken font is not reread every frame"
        );

        let plain = paint(&text(
            Rect::new(0.0, 0.0, 300.0, 40.0),
            TextAnchor::UpperLeft,
            "fallback",
        ));
        assert_eq!(painter.frame().quads, plain.frame().quads);
    }

    #[test]
    fn the_atlas_grows_without_moving_a_texel() {
        let mut atlas = Atlas::new();
        let first = atlas.allocate([10, 10]).unwrap();
        atlas.write_rect(first, [10, 10], &[200; 100]);
        let before = atlas.extent();
        while atlas.extent() == before {
            atlas.allocate([100, 100]).expect("room to grow");
        }
        let width = atlas.extent()[0] as usize;
        let at = (first[1] as usize + 5) * width + first[0] as usize + 5;
        assert_eq!(atlas.pixels()[at], 200);
        assert!(atlas.used_rows() <= atlas.extent()[1]);
    }

    #[test]
    fn a_full_atlas_refuses_and_the_painter_starts_it_again() {
        let mut atlas = Atlas::new();
        // The big rectangles leave rows under the last shelf; the small ones
        // take them.
        for side in [500, 1] {
            while atlas.allocate([side; 2]).is_some() {}
        }
        assert_eq!(atlas.extent(), [Atlas::MAX; 2]);

        let mut painter = Painter::new();
        painter.atlas = atlas;
        let canvas = text(Rect::new(0.0, 0.0, 600.0, 60.0), TextAnchor::UpperLeft, "x");
        let mut assets = Fake::default();
        painter.paint(&canvas, &mut assets);
        assert!(painter.frame().is_empty(), "no room this frame");
        painter.paint(&canvas, &mut assets);
        assert_eq!(painter.frame().quads.len(), 1, "room again the next");
        assert_eq!(painter.frame().atlas.extent(), [Atlas::INITIAL; 2]);
    }

    use crate::scene::{GradientPaint, ShapePaint, ShapeStyle, UiStyle};

    fn cut(bounds: Rect, radius: f32, exponent: f32, paint: ShapePaint) -> UiStyle {
        UiStyle::Shape(ShapeStyle {
            bounds: Some(bounds),
            radius,
            exponent,
            paint,
            gradient: None,
        })
    }

    /// The modifiers leave the plain stream exactly as it was: 36 bytes a
    /// quad, and no shape for a canvas that asks for none.
    #[test]
    fn a_plain_canvas_draws_no_shapes() {
        assert_eq!(std::mem::size_of::<UiQuad>(), 36);
        assert_eq!(std::mem::size_of::<UiShape>(), 88);
        let canvas = text(
            Rect::new(0.0, 0.0, 300.0, 40.0),
            TextAnchor::UpperLeft,
            "plain",
        );
        let painter = paint(&canvas);
        assert!(!painter.frame().quads.is_empty());
        assert!(painter.frame().shapes.is_empty());
        assert!(painter.frame().shape_batches.is_empty());
    }

    /// A rounded panel is one shape: what it covers, the box it is cut from,
    /// and how.
    #[test]
    fn a_styled_panel_is_one_shape_cut_to_its_box() {
        let mut canvas = UiCanvas::default();
        let rect = Rect::new(10.0, 20.0, 200.0, 100.0);
        let style = cut(rect, 12.0, 4.0, ShapePaint::Fill);
        canvas.styled(style, |canvas| canvas.panel(rect, [20, 40, 60, 200]));
        let painter = paint(&canvas);
        let frame = painter.frame();
        assert!(frame.quads.is_empty());
        let [shape] = frame.shapes else {
            panic!("{:?}", frame.shapes);
        };
        assert_eq!(shape.rect, [10.0, 20.0, 210.0, 120.0]);
        assert_eq!(shape.uv, painter.atlas.white());
        assert_eq!(shape.shape, [110.0, 70.0, 100.0, 50.0]);
        assert_eq!(shape.params, [12.0, 4.0, 0.0, 0.0]);
        assert_eq!(shape.colors, [[20, 40, 60, 200]; 2]);
        assert_eq!(shape.gradient, [0.0; 4], "flat");
        assert_eq!(
            frame.shape_batches,
            [UiShapeBatch {
                source: UiSource::Atlas,
                shapes: 0..1,
                before: 0
            }]
        );
    }

    #[test]
    fn a_ring_carries_half_its_width_and_a_blur_its_sigma() {
        let mut canvas = UiCanvas::default();
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        let ring = cut(rect, 0.0, 2.0, ShapePaint::Ring { width: 6.0 });
        let blur = cut(rect, 8.0, 2.0, ShapePaint::Blur { sigma: 3.0 });
        canvas.styled(ring, |canvas| canvas.panel(rect, [255; 4]));
        canvas.styled(blur, |canvas| canvas.panel(rect, [255; 4]));
        let painter = paint(&canvas);
        let params: Vec<[f32; 4]> = painter.frame().shapes.iter().map(|s| s.params).collect();
        assert_eq!(params, [[0.0, 2.0, 3.0, 0.0], [8.0, 2.0, 0.0, 3.0]]);
    }

    /// A gradient's stops are multiplied into the command's colour, and its
    /// axis is handed on as the shader reads it.
    #[test]
    fn a_gradient_multiplies_into_the_commands_colour() {
        let mut canvas = UiCanvas::default();
        let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
        let style = UiStyle::Shape(ShapeStyle {
            bounds: Some(rect),
            radius: 0.0,
            exponent: 2.0,
            paint: ShapePaint::Fill,
            gradient: Some(GradientPaint {
                start: [255, 255, 255, 255],
                end: [0, 128, 255, 0],
                origin: [0.0, 25.0],
                step: [0.01, 0.0],
            }),
        });
        canvas.styled(style, |canvas| canvas.panel(rect, [255, 255, 128, 255]));
        let painter = paint(&canvas);
        let shape = painter.frame().shapes[0];
        assert_eq!(shape.colors, [[255, 255, 128, 255], [0, 128, 128, 0]]);
        assert_eq!(shape.gradient, [0.0, 25.0, 0.01, 0.0]);
    }

    /// Clipping cuts what a shape covers and the source in proportion, and
    /// leaves the box it is cut from where it was — so a rounded panel half
    /// scrolled out keeps the corners it has in view.
    #[test]
    fn a_clip_cuts_what_a_shape_covers_but_not_its_box() {
        let icon = AssetId::new();
        let mut canvas = UiCanvas::default();
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        canvas.clip(Some(Rect::new(0.0, 50.0, 300.0, 300.0)));
        canvas.styled(cut(rect, 10.0, 2.0, ShapePaint::Fill), |canvas| {
            canvas.image(rect, icon, [255; 4])
        });
        canvas.panel(rect, [255; 4]);
        canvas.clip(None);
        canvas.panel(rect, [255; 4]);
        let mut assets = Fake::default();
        assets.textures.insert(icon, TextureHandle(2));
        let mut painter = Painter::new();
        painter.paint(&canvas, &mut assets);
        let frame = painter.frame();
        let shape = frame.shapes[0];
        assert_eq!(shape.rect, [0.0, 50.0, 100.0, 100.0]);
        assert_eq!(shape.uv, [0.0, 0.5, 1.0, 1.0]);
        assert_eq!(shape.shape, [50.0, 50.0, 50.0, 50.0]);
        let rects: Vec<[f32; 4]> = frame.quads.iter().map(|q| q.rect).collect();
        assert_eq!(
            rects,
            [[0.0, 50.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]],
            "a plain panel is cut too, until the clip is lifted"
        );

        let mut hidden = UiCanvas::default();
        hidden.clip(Some(Rect::new(500.0, 500.0, 10.0, 10.0)));
        hidden.panel(rect, [255; 4]);
        hidden.text(rect, style(TextAnchor::UpperLeft), "gone");
        assert!(paint(&hidden).frame().is_empty(), "wholly outside its clip");
    }

    /// Shapes and quads keep the order they were called in: a shape between
    /// two panels splits what would otherwise be one batch.
    #[test]
    fn shapes_and_quads_draw_in_call_order() {
        let mut canvas = UiCanvas::default();
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        canvas.panel(rect, [1, 0, 0, 255]);
        canvas.styled(cut(rect, 2.0, 2.0, ShapePaint::Fill), |canvas| {
            canvas.panel(rect, [2, 0, 0, 255])
        });
        canvas.styled(cut(rect, 2.0, 2.0, ShapePaint::Fill), |canvas| {
            canvas.panel(rect, [3, 0, 0, 255])
        });
        canvas.panel(rect, [4, 0, 0, 255]);
        let painter = paint(&canvas);
        let frame = painter.frame();
        assert_eq!(
            frame.batches,
            [
                UiBatch {
                    source: UiSource::Atlas,
                    quads: 0..1
                },
                UiBatch {
                    source: UiSource::Atlas,
                    quads: 1..2
                },
            ]
        );
        assert_eq!(
            frame.shape_batches,
            [UiShapeBatch {
                source: UiSource::Atlas,
                shapes: 0..2,
                before: 1
            }],
            "two shapes in a row are one draw, between the two panels"
        );
    }

    /// An outline is the same glyphs grown on every side, set under the fill
    /// and in the stroke's colour, and no more cut by the label's rect than
    /// the ink it rings.
    #[test]
    fn an_outline_sets_each_glyph_grown_by_its_width() {
        let rect = Rect::new(100.0, 50.0, 400.0, 60.0);
        let mut canvas = UiCanvas::default();
        let ink = TextStyle {
            color: [0, 0, 0, 255],
            ..style(TextAnchor::UpperLeft)
        };
        canvas.styled(UiStyle::Outline { width: 2.0 }, |canvas| {
            canvas.text(rect, ink, "HI")
        });
        canvas.text(rect, style(TextAnchor::UpperLeft), "HI");
        let painter = paint(&canvas);
        let quads = painter.frame().quads;
        assert_eq!(quads.len(), 4);
        let (outline, fill) = quads.split_at(2);
        for (ring, glyph) in outline.iter().zip(fill) {
            assert_eq!(ring.color, [0, 0, 0, 255]);
            assert_eq!(glyph.color, [255; 4]);
            // Two pixels of ring and one more to fade over, uncut by the
            // label's rect: the stroke is the text's, not the box's.
            assert_eq!(
                ring.rect,
                [
                    glyph.rect[0] - 3.0,
                    glyph.rect[1] - 3.0,
                    glyph.rect[2] + 3.0,
                    glyph.rect[3] + 3.0
                ]
            );
        }
        // The grown coverage reaches past where the glyph's own ends.
        let atlas = painter.frame().atlas;
        let width = atlas.extent()[0] as usize;
        let texel = |quad: &UiQuad, dx: f32, dy: f32| {
            let (x, y) = ((quad.uv[0] + dx) as usize, (quad.uv[1] + dy) as usize);
            atlas.pixels()[y * width + x]
        };
        let ring = outline[0];
        let middle = (ring.uv[3] - ring.uv[1]) / 2.0;
        assert_eq!(texel(&ring, 0.0, middle), 0, "nothing at its very edge");
        assert!(
            texel(&ring, 1.0, middle) > 0,
            "ink a pixel out from the H's stem"
        );
        assert_eq!(painter.frame().batches.len(), 1, "still one atlas draw");
    }

    /// Gradient text keeps its glyphs' coverage and adds the blend: every
    /// glyph is a shape over the atlas, cut by nothing.
    #[test]
    fn gradient_text_sets_its_glyphs_as_shapes() {
        let rect = Rect::new(0.0, 0.0, 400.0, 60.0);
        let gradient = GradientPaint {
            start: [255; 4],
            end: [0, 0, 0, 255],
            origin: [0.0, 0.0],
            step: [0.0025, 0.0],
        };
        let mut canvas = UiCanvas::default();
        let glyphs = UiStyle::Shape(ShapeStyle {
            bounds: None,
            radius: 0.0,
            exponent: 2.0,
            paint: ShapePaint::Fill,
            gradient: Some(gradient),
        });
        canvas.styled(glyphs, |canvas| {
            canvas.text(rect, style(TextAnchor::UpperLeft), "abc")
        });
        let painter = paint(&canvas);
        let plain = paint(&text(rect, TextAnchor::UpperLeft, "abc"));
        let frame = painter.frame();
        assert!(frame.quads.is_empty());
        assert_eq!(frame.shapes.len(), 3);
        for (shape, glyph) in frame.shapes.iter().zip(plain.frame().quads) {
            assert_eq!((shape.rect, shape.uv), (glyph.rect, glyph.uv));
            assert!(
                shape.shape[2] > 1.0e5 && shape.shape[3] > 1.0e5,
                "{shape:?}"
            );
            assert_eq!(shape.gradient, [0.0, 0.0, 0.0025, 0.0]);
        }
    }

    /// A layer drawn at half size halves its shapes' lengths with it.
    #[test]
    fn a_scaled_layer_scales_its_shapes() {
        let mut canvas = UiCanvas::default();
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        canvas.styled(
            cut(rect, 10.0, 2.0, ShapePaint::Blur { sigma: 4.0 }),
            |canvas| canvas.panel(rect, [255; 4]),
        );
        let screen = GameScreen {
            size: [1920.0, 1080.0],
            origin: [10.0, 20.0],
            scale: 0.5,
            clip: Rect::new(10.0, 20.0, 960.0, 540.0),
        };
        let mut painter = Painter::new();
        painter.paint_layers(
            &[(&canvas, Placement::screen(&screen))],
            &mut Fake::default(),
        );
        let shape = painter.frame().shapes[0];
        assert_eq!(shape.rect, [10.0, 20.0, 60.0, 70.0]);
        assert_eq!(shape.shape, [35.0, 45.0, 25.0, 25.0]);
        assert_eq!(shape.params, [5.0, 2.0, 0.0, 2.0]);
    }
}
