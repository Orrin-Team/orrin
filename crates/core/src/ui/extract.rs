use glam::Vec2;
use orrin_ecs::World;
use orrin_registry::AssetId;

use super::events::UiPointer;
use super::layout::{Presence, ScrollView};
use super::{
    Button, Corner, Gradient, Image, Layout, ScrollFrame, Shadow, Stroke, StrokePlacement, Text,
};
use crate::scene::{
    GradientPaint, Rect, ShapePaint, ShapeStyle, TextStyle, UiCanvas, UiStyle, color_bytes,
};

/// Record what `layout`'s elements draw into `out`, back to front: each
/// element's shadow, image, stroke and text, then its children, and a scroll
/// frame's bars over all of those. `extent` is a texture's uploaded size, which
/// a sliced image is measured in.
pub fn extract(
    world: &World,
    layout: &Layout,
    extent: impl Fn(AssetId) -> Option<[u32; 2]>,
    out: &mut UiCanvas,
) {
    let present = layout.presence();
    if present.modifiers() || !layout.scrolls().is_empty() {
        record(world, layout, present, &extent, out);
    } else {
        record_plain(world, layout, &extent, out);
    }
}

/// [`record`] for a frame with no modifier and no scroll frame anywhere, which
/// needs none of its bookkeeping: kept apart so the loop over plain elements
/// compiles to what it was before there was any, less the lookups the layout
/// already knows will miss.
fn record_plain(
    world: &World,
    layout: &Layout,
    extent: &impl Fn(AssetId) -> Option<[u32; 2]>,
    out: &mut UiCanvas,
) {
    let pointer = world.get_resource::<UiPointer>();
    for (index, element) in layout.elements().iter().enumerate() {
        let entity = element.entity;
        // Screen pixels per unit of the element's own frame.
        let pixels = element.units * element.scale;
        let (has_image, has_text) = layout.draws(index);
        if has_image && let Some(image) = world.get::<Image>(entity).map(|image| *image) {
            let tint = match world.get::<Button>(entity) {
                Some(button) => button_tint(&button, pointer.as_deref(), layout, index),
                None => [1.0; 4],
            };
            let color = color_bytes(std::array::from_fn(|i| image.color[i] * tint[i]));
            match image.texture {
                None => out.panel(element.rect, color),
                Some(texture) => match extent(texture).filter(|_| image.slice != [0.0; 4]) {
                    Some(size) => sliced(
                        out,
                        element.rect,
                        texture,
                        &image,
                        size,
                        pixels,
                        color,
                        None,
                    ),
                    None => out.image(element.rect, texture, color),
                },
            }
        }
        if has_text && let Some(text) = world.get::<Text>(entity) {
            out.text(
                element.rect,
                text_style(&text, pixels, color_bytes(text.color)),
                &text.text,
            );
        }
    }
}

/// [`extract`] for a frame with a modifier or a scroll frame somewhere: the
/// clips and bars, and each element's modifiers looked up.
#[inline(never)]
fn record(
    world: &World,
    layout: &Layout,
    present: Presence,
    extent: &impl Fn(AssetId) -> Option<[u32; 2]>,
    out: &mut UiCanvas,
) {
    let pointer = world.get_resource::<UiPointer>();
    let scrolls = layout.scrolls();
    // Frames whose subtree is still being drawn, innermost last.
    let mut open: Vec<&ScrollView> = Vec::new();
    let mut next_scroll = 0;
    let mut clip = None;
    for (index, element) in layout.elements().iter().enumerate() {
        while let Some(view) = open.last().filter(|view| view.end as usize <= index) {
            bars(world, layout, view, &mut clip, out);
            open.pop();
        }
        let cut = layout.clip(index);
        if cut != clip {
            out.clip(cut);
            clip = cut;
        }
        let entity = element.entity;
        let pixels = element.units * element.scale;
        let (has_image, has_text) = layout.draws(index);
        let image = has_image
            .then(|| world.get::<Image>(entity).map(|image| *image))
            .flatten();
        let modifiers = Modifiers::of(
            world,
            present,
            entity,
            element.rect,
            pixels,
            image.is_some(),
        )
        .unwrap_or_default();
        let draw = Draw {
            world,
            layout,
            pointer: pointer.as_deref(),
            index,
            entity,
            rect: element.rect,
            pixels,
        };
        draw.element(image, has_text, &modifiers, extent, out);
        if scrolls
            .get(next_scroll)
            .is_some_and(|view| view.element as usize == index)
        {
            open.push(&scrolls[next_scroll]);
            next_scroll += 1;
        }
    }
    while let Some(view) = open.pop() {
        bars(world, layout, view, &mut clip, out);
    }
}

/// A run of text as `text` asks for it, `pixels` to a unit.
fn text_style(text: &Text, pixels: Vec2, color: [u8; 4]) -> TextStyle {
    TextStyle {
        font: text.font,
        size: text.size * pixels.y,
        anchor: text.anchor,
        color,
        wrap: text.wrap,
    }
}

/// What drawing element `index` needs to know about it.
struct Draw<'a> {
    world: &'a World,
    layout: &'a Layout,
    pointer: Option<&'a UiPointer>,
    index: usize,
    entity: orrin_ecs::Entity,
    rect: Rect,
    /// Screen pixels per unit of the element's own frame.
    pixels: Vec2,
}

impl Draw<'_> {
    fn image(
        &self,
        image: &Image,
        extent: &impl Fn(AssetId) -> Option<[u32; 2]>,
        fill: Option<UiStyle>,
        out: &mut UiCanvas,
    ) {
        let tint = match self.world.get::<Button>(self.entity) {
            Some(button) => button_tint(&button, self.pointer, self.layout, self.index),
            None => [1.0; 4],
        };
        let color = color_bytes(std::array::from_fn(|i| image.color[i] * tint[i]));
        let rect = self.rect;
        let sized = image
            .texture
            .and_then(|texture| Some((texture, extent(texture)?)))
            .filter(|_| image.slice != [0.0; 4]);
        match (image.texture, sized) {
            (_, Some((texture, size))) => {
                sliced(out, rect, texture, image, size, self.pixels, color, fill);
            }
            (None, _) => styled(out, fill, |out| out.panel(rect, color)),
            (Some(texture), None) => styled(out, fill, |out| out.image(rect, texture, color)),
        }
    }

    /// The element with its modifiers: shadow, image, stroke, text.
    fn element(
        &self,
        image: Option<Image>,
        has_text: bool,
        modifiers: &Modifiers,
        extent: &impl Fn(AssetId) -> Option<[u32; 2]>,
        out: &mut UiCanvas,
    ) {
        if let Some((shape, rect, color)) = modifiers.shadow {
            out.styled(UiStyle::Shape(shape), |out| out.panel(rect, color));
        }
        if let Some(image) = image {
            self.image(&image, extent, modifiers.fill.map(UiStyle::Shape), out);
        }
        if let Some((shape, rect, color)) = modifiers.ring {
            out.styled(UiStyle::Shape(shape), |out| out.panel(rect, color));
        }
        if !has_text {
            return;
        }
        let Some(text) = self.world.get::<Text>(self.entity) else {
            return;
        };
        if let Some((width, color)) = modifiers.outline {
            let style = text_style(&text, self.pixels, color);
            out.styled(UiStyle::Outline { width }, |out| {
                out.text(self.rect, style, &text.text)
            });
        }
        let fill = text_style(&text, self.pixels, color_bytes(text.color));
        match modifiers.gradient {
            Some(gradient) => {
                let glyphs = ShapeStyle {
                    bounds: None,
                    radius: 0.0,
                    exponent: 2.0,
                    paint: ShapePaint::Fill,
                    gradient: Some(gradient),
                };
                out.styled(UiStyle::Shape(glyphs), |out| {
                    out.text(self.rect, fill, &text.text)
                });
            }
            None => out.text(self.rect, fill, &text.text),
        }
    }
}

/// Record what `draw` records, with `style` if there is one.
fn styled(out: &mut UiCanvas, style: Option<UiStyle>, draw: impl FnOnce(&mut UiCanvas)) {
    match style {
        Some(style) => out.styled(style, draw),
        None => draw(out),
    }
}

/// A frame's bars, as pills over its content, cut to what cuts the frame.
#[inline(never)]
fn bars(
    world: &World,
    layout: &Layout,
    view: &ScrollView,
    clip: &mut Option<Rect>,
    out: &mut UiCanvas,
) {
    if view.bars == [None, None] {
        return;
    }
    let entity = layout.elements()[view.element as usize].entity;
    let Some(color) = world
        .get::<ScrollFrame>(entity)
        .map(|frame| color_bytes(frame.scrollbar_color))
    else {
        return;
    };
    let cut = layout.clip(view.element as usize);
    if cut != *clip {
        out.clip(cut);
        *clip = cut;
    }
    for bar in view.bars.iter().flatten() {
        let thumb = bar.thumb;
        let pill = ShapeStyle {
            bounds: Some(thumb),
            radius: thumb.width.min(thumb.height) / 2.0,
            exponent: 2.0,
            paint: ShapePaint::Fill,
            gradient: None,
        };
        out.styled(UiStyle::Shape(pill), |out| out.panel(thumb, color));
    }
}

/// What an element's modifiers ask of each thing it draws, in pixels: each
/// shape with the rectangle it is drawn over and its colour.
#[derive(Default)]
struct Modifiers {
    shadow: Option<(ShapeStyle, Rect, [u8; 4])>,
    /// How the image is cut and coloured, if it is at all.
    fill: Option<ShapeStyle>,
    ring: Option<(ShapeStyle, Rect, [u8; 4])>,
    /// A label's ring: its width and colour.
    outline: Option<(f32, [u8; 4])>,
    gradient: Option<GradientPaint>,
}

impl Modifiers {
    /// `None` for an element with none. `filled` is whether it has an image;
    /// one that is not filled but has text is a label, whose stroke rings its
    /// glyphs and which casts no shadow, and anything else is a box.
    #[inline(never)]
    fn of(
        world: &World,
        present: Presence,
        entity: orrin_ecs::Entity,
        rect: Rect,
        pixels: Vec2,
        filled: bool,
    ) -> Option<Self> {
        fn get<T: Copy + 'static>(
            world: &World,
            present: bool,
            entity: orrin_ecs::Entity,
        ) -> Option<T> {
            present
                .then(|| world.get::<T>(entity).map(|c| *c))
                .flatten()
        }
        let corner = get::<Corner>(world, present.corner, entity);
        let gradient = get::<Gradient>(world, present.gradient, entity).map(|g| gradient(&g, rect));
        let stroke = get::<Stroke>(world, present.stroke, entity);
        let shadow = get::<Shadow>(world, present.shadow, entity);
        if corner.is_none() && gradient.is_none() && stroke.is_none() && shadow.is_none() {
            return None;
        }
        let boxed = filled || !world.has::<Text>(entity);
        let size = Vec2::new(rect.width, rect.height);
        let unit = pixels.x.min(pixels.y);
        let (radius, exponent) = corner.map_or((0.0, 2.0), |corner| {
            (corner.radius_px(size, unit), corner.exponent())
        });
        let cut = |bounds: Rect, radius: f32, paint| ShapeStyle {
            bounds: Some(bounds),
            radius: radius.max(0.0),
            exponent,
            paint,
            gradient: None,
        };
        let mut out = Self {
            gradient,
            ..Self::default()
        };
        if corner.is_some() || gradient.is_some() {
            out.fill = Some(ShapeStyle {
                gradient,
                ..cut(rect, radius, ShapePaint::Fill)
            });
        }
        if let Some(shadow) = shadow.filter(|_| boxed) {
            let offset = shadow.offset * pixels;
            let spread = shadow.spread * unit;
            let sigma = shadow.blur.max(0.0) * unit / 2.0;
            let bounds = grow(
                Rect::new(
                    rect.x + offset.x,
                    rect.y + offset.y,
                    rect.width,
                    rect.height,
                ),
                spread,
            );
            let paint = if sigma > 0.0 {
                ShapePaint::Blur { sigma }
            } else {
                ShapePaint::Fill
            };
            out.shadow = Some((
                cut(bounds, radius + spread, paint),
                grow(bounds, 3.0 * sigma),
                color_bytes(shadow.color),
            ));
        }
        if let Some(stroke) = stroke {
            let color = color_bytes(stroke.color);
            if boxed {
                let width = stroke.thickness.max(0.0) * unit;
                let (centre, band) = match stroke.placement {
                    StrokePlacement::Inside => (-width / 2.0, width),
                    StrokePlacement::Center => (0.0, width),
                    // Reaching a pixel in under the image, so the ring's inner
                    // edge fades over the image rather than over the image's
                    // own fading edge, where the two would let the background
                    // show through as a hairline between them.
                    StrokePlacement::Outside if filled => ((width - 1.0) / 2.0, width + 1.0),
                    StrokePlacement::Outside => (width / 2.0, width),
                };
                out.ring = Some((
                    cut(
                        grow(rect, centre),
                        radius + centre,
                        ShapePaint::Ring { width: band },
                    ),
                    grow(rect, centre + band / 2.0),
                    color,
                ));
            } else {
                out.outline = Some((stroke.thickness.max(0.0) * pixels.y, color));
            }
        }
        Some(out)
    }
}

/// `rect` grown by `by` on every side, or shrunk by a negative one to no less
/// than its centre.
fn grow(rect: Rect, by: f32) -> Rect {
    let x = (by * 2.0).max(-rect.width);
    let y = (by * 2.0).max(-rect.height);
    Rect::new(
        rect.x - x / 2.0,
        rect.y - y / 2.0,
        rect.width + x,
        rect.height + y,
    )
}

/// `gradient` laid over `rect`: along its rotated axis, from the box's first
/// corner on that axis to its last, moved by its offset.
fn gradient(gradient: &Gradient, rect: Rect) -> GradientPaint {
    let (sin, cos) = gradient.rotation.to_radians().sin_cos();
    let axis = Vec2::new(cos, sin);
    let half = Vec2::new(rect.width, rect.height) / 2.0;
    let reach = (half.x * cos).abs() + (half.y * sin).abs();
    let centre = Vec2::new(rect.x, rect.y) + half + gradient.offset * half * 2.0;
    let step = if reach > 0.0 {
        axis / (2.0 * reach)
    } else {
        Vec2::ZERO
    };
    GradientPaint {
        start: color_bytes(gradient.start),
        end: color_bytes(gradient.end),
        origin: (centre - axis * reach).to_array(),
        step: step.to_array(),
    }
}

/// What button element `index` multiplies its image by: pressed while a press
/// that began on it or under it is held with the pointer still inside, hovered
/// while the pointer is inside.
fn button_tint(
    button: &Button,
    pointer: Option<&UiPointer>,
    layout: &Layout,
    index: usize,
) -> [f32; 4] {
    if !button.interactable {
        return button.disabled;
    }
    let Some(pointer) = pointer else {
        return button.normal;
    };
    let entity = layout.elements()[index].entity;
    if !pointer.is_inside(entity) {
        return button.normal;
    }
    let pressed_here = pointer
        .pressed()
        .and_then(|pressed| layout.index_of(pressed))
        .is_some_and(|pressed| layout.is_at_or_above(entity, pressed));
    if pressed_here {
        button.pressed
    } else {
        button.hover
    }
}

/// Nine pieces: corners at the borders' own size, edges stretched along one
/// axis, the middle along both. Borders that do not fit shrink together.
/// With `style`, every piece is cut to the whole box, so the corners stay the
/// box's rather than becoming each piece's.
#[allow(clippy::too_many_arguments)]
fn sliced(
    out: &mut UiCanvas,
    rect: Rect,
    texture: AssetId,
    image: &Image,
    [width, height]: [u32; 2],
    pixels: Vec2,
    color: [u8; 4],
    style: Option<UiStyle>,
) {
    let [left, top, right, bottom] = image.slice.map(|border| border.max(0.0));
    let texels = Vec2::new(width.max(1) as f32, height.max(1) as f32);
    let fit = |near: f32, far: f32, span: f32, scale: f32| {
        let wanted = (near + far) * scale;
        let shrink = if wanted > span { span / wanted } else { 1.0 };
        (near * scale * shrink, far * scale * shrink)
    };
    let (l, r) = fit(left, right, rect.width.max(0.0), pixels.x);
    let (t, b) = fit(top, bottom, rect.height.max(0.0), pixels.y);
    let [x_end, y_end] = rect.max();
    let xs = [rect.x, rect.x + l, x_end - r, x_end];
    let ys = [rect.y, rect.y + t, y_end - b, y_end];
    let us = [0.0, left / texels.x, 1.0 - right / texels.x, 1.0];
    let vs = [0.0, top / texels.y, 1.0 - bottom / texels.y, 1.0];
    for row in 0..3 {
        for column in 0..3 {
            let piece = Rect::new(
                xs[column],
                ys[row],
                xs[column + 1] - xs[column],
                ys[row + 1] - ys[row],
            );
            let uv = [us[column], vs[row], us[column + 1], vs[row + 1]];
            styled(out, style, |out| {
                out.image_region(piece, texture, uv, color)
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use orrin_ecs::Entity;

    use super::*;
    use crate::scene::{Hierarchy, TextAnchor, UiCommand, ensure_current, reparent};
    use crate::ui::{Canvas, RectTransform, ScaleMode, layout};

    struct Ui {
        world: World,
        canvas: Entity,
    }

    impl Ui {
        fn new(canvas: Canvas) -> Self {
            let mut world = World::new();
            let root = world.spawn();
            world.insert(root, canvas);
            Self {
                world,
                canvas: root,
            }
        }

        fn element(&mut self, parent: Option<Entity>, rect: RectTransform) -> Entity {
            let entity = self.world.spawn();
            self.world.insert(entity, rect);
            let parent = parent.unwrap_or(self.canvas);
            reparent(&mut self.world, entity, Some(parent), false).unwrap();
            entity
        }

        fn commands_with(&mut self, extent: Option<[u32; 2]>) -> UiCanvas {
            ensure_current(&mut self.world);
            let mut resolved = Layout::default();
            layout::resolve(
                &self.world,
                &self.world.resource::<Hierarchy>(),
                &crate::scene::GameScreen::window([1920, 1080]),
                &mut resolved,
            );
            let mut out = UiCanvas::default();
            extract(&self.world, &resolved, |_| extent, &mut out);
            out
        }

        fn commands(&mut self) -> UiCanvas {
            self.commands_with(None)
        }
    }

    fn box_at(x: f32, y: f32, width: f32, height: f32) -> RectTransform {
        RectTransform {
            anchor_min: Vec2::ZERO,
            anchor_max: Vec2::ZERO,
            pivot: Vec2::ZERO,
            anchored_position: Vec2::new(x, y),
            size_delta: Vec2::new(width, height),
            scale: Vec2::ONE,
        }
    }

    fn filled(red: f32) -> Image {
        Image {
            color: [red, 0.0, 0.0, 1.0],
            ..Image::default()
        }
    }

    /// The red channel of each panel, and `T` for each text, in draw order.
    fn summary(canvas: &UiCanvas) -> Vec<String> {
        canvas
            .commands()
            .iter()
            .map(|command| match command {
                UiCommand::Panel { color, .. } => color[0].to_string(),
                UiCommand::Text { text, .. } => format!("T:{}", canvas.str(text)),
                UiCommand::Image { .. } => "I".to_owned(),
            })
            .collect()
    }

    #[test]
    fn an_element_draws_its_image_then_its_text_then_its_children() {
        let mut ui = Ui::new(Canvas::default());
        let a = ui.element(None, RectTransform::stretched());
        ui.world.insert(a, filled(0.1));
        ui.world.insert(
            a,
            Text {
                text: "a".into(),
                ..Text::default()
            },
        );
        let a_child = ui.element(Some(a), RectTransform::stretched());
        ui.world.insert(a_child, filled(0.2));
        let b = ui.element(None, RectTransform::stretched());
        ui.world.insert(b, filled(0.3));

        let canvas = ui.commands();
        assert_eq!(summary(&canvas), ["26", "T:a", "51", "77"]);
    }

    #[test]
    fn a_higher_sort_order_draws_over_a_lower_one() {
        let mut ui = Ui::new(Canvas {
            sort_order: 1,
            ..Canvas::default()
        });
        let over = ui.element(None, RectTransform::stretched());
        ui.world.insert(over, filled(1.0));
        let under_canvas = ui.world.spawn();
        ui.world.insert(under_canvas, Canvas::default());
        let under = ui.world.spawn();
        ui.world.insert(under, RectTransform::stretched());
        ui.world.insert(under, filled(0.0));
        reparent(&mut ui.world, under, Some(under_canvas), false).unwrap();

        assert_eq!(summary(&ui.commands()), ["0", "255"]);
    }

    #[test]
    fn a_button_tints_its_image_and_a_disabled_one_draws_its_disabled_colour() {
        let mut ui = Ui::new(Canvas::default());
        let button = ui.element(None, box_at(0.0, 0.0, 100.0, 40.0));
        ui.world.insert(
            button,
            Image {
                color: [1.0, 0.5, 1.0, 1.0],
                ..Image::default()
            },
        );
        ui.world.insert(
            button,
            Button {
                normal: [0.5, 1.0, 1.0, 1.0],
                disabled: [1.0, 1.0, 1.0, 0.5],
                ..Button::default()
            },
        );
        let colour = |canvas: &UiCanvas| match canvas.commands()[0] {
            UiCommand::Panel { color, .. } => color,
            ref other => panic!("{other:?}"),
        };
        assert_eq!(colour(&ui.commands()), [128, 128, 255, 255]);

        ui.world.get_mut::<Button>(button).unwrap().interactable = false;
        assert_eq!(colour(&ui.commands()), [255, 128, 255, 128]);
    }

    #[test]
    fn a_buttons_tint_follows_the_pointer() {
        use crate::ui::events::{PointerInput, UiEvents, UiPointer, step};

        let mut ui = Ui::new(Canvas::default());
        let button = ui.element(None, box_at(0.0, 0.0, 100.0, 40.0));
        ui.world.insert(button, Image::default());
        ui.world.insert(
            button,
            Button {
                normal: [1.0, 0.0, 0.0, 1.0],
                hover: [0.0, 1.0, 0.0, 1.0],
                pressed: [0.0, 0.0, 1.0, 1.0],
                ..Button::default()
            },
        );
        let label = ui.element(Some(button), box_at(0.0, 0.0, 20.0, 20.0));
        ui.world.insert(label, Image::default());
        ui.world.insert_resource(UiPointer::default());

        let frame = |ui: &mut Ui, x: f32, pressed: bool, released: bool| {
            ensure_current(&mut ui.world);
            let mut resolved = Layout::default();
            layout::resolve(
                &ui.world,
                &ui.world.resource::<Hierarchy>(),
                &crate::scene::GameScreen::window([1920, 1080]),
                &mut resolved,
            );
            let input = PointerInput {
                position: [x, 10.0],
                pressed,
                released,
            };
            step(
                &ui.world,
                &resolved,
                &mut ui.world.resource_mut::<UiPointer>(),
                input,
                &mut UiEvents::default(),
            );
            let canvas = ui.commands();
            match canvas.commands()[0] {
                UiCommand::Panel { color, .. } => [color[0], color[1], color[2]],
                ref other => panic!("{other:?}"),
            }
        };
        assert_eq!(frame(&mut ui, 500.0, false, false), [255, 0, 0], "normal");
        assert_eq!(frame(&mut ui, 50.0, false, false), [0, 255, 0], "hovered");
        assert_eq!(
            frame(&mut ui, 10.0, true, false),
            [0, 0, 255],
            "pressed through its child"
        );
        assert_eq!(
            frame(&mut ui, 500.0, false, false),
            [255, 0, 0],
            "held, but the pointer left"
        );
        assert_eq!(frame(&mut ui, 50.0, false, true), [0, 255, 0]);
    }

    #[test]
    fn a_sliced_image_is_nine_pieces_with_corners_at_their_own_size() {
        let mut ui = Ui::new(Canvas::default());
        let texture = AssetId::new();
        let panel = ui.element(None, box_at(10.0, 20.0, 200.0, 100.0));
        ui.world.insert(
            panel,
            Image {
                texture: Some(texture),
                slice: [8.0, 4.0, 16.0, 12.0],
                ..Image::default()
            },
        );
        let canvas = ui.commands_with(Some([64, 32]));
        let pieces: Vec<(Rect, [f32; 4])> = canvas
            .commands()
            .iter()
            .map(|command| match command {
                UiCommand::Image { rect, uv, .. } => (*rect, *uv),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(pieces.len(), 9);
        // Top left: the border's own size, the texture's own corner.
        assert_eq!(pieces[0].0, Rect::new(10.0, 20.0, 8.0, 4.0));
        assert_eq!(pieces[0].1, [0.0, 0.0, 8.0 / 64.0, 4.0 / 32.0]);
        // The middle stretches over everything between the borders.
        assert_eq!(pieces[4].0, Rect::new(18.0, 24.0, 176.0, 84.0));
        assert_eq!(
            pieces[4].1,
            [8.0 / 64.0, 4.0 / 32.0, 1.0 - 16.0 / 64.0, 1.0 - 12.0 / 32.0]
        );
        // Bottom right.
        assert_eq!(pieces[8].0, Rect::new(194.0, 108.0, 16.0, 12.0));
        assert_eq!(
            pieces[8].1,
            [1.0 - 16.0 / 64.0, 1.0 - 12.0 / 32.0, 1.0, 1.0]
        );
    }

    /// Too small for its borders, the borders shrink together rather than
    /// overlapping, as uGUI's do.
    #[test]
    fn a_sliced_image_smaller_than_its_borders_shrinks_them() {
        let mut ui = Ui::new(Canvas::default());
        let panel = ui.element(None, box_at(0.0, 0.0, 12.0, 100.0));
        ui.world.insert(
            panel,
            Image {
                texture: Some(AssetId::new()),
                slice: [8.0, 0.0, 16.0, 0.0],
                ..Image::default()
            },
        );
        let canvas = ui.commands_with(Some([64, 64]));
        let widths: Vec<f32> = canvas
            .commands()
            .iter()
            .map(|command| match command {
                UiCommand::Image { rect, .. } => rect.width,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(widths, [4.0, 8.0], "the middle has no width left");
    }

    #[test]
    fn an_image_whose_size_is_not_known_yet_draws_stretched() {
        let mut ui = Ui::new(Canvas::default());
        let panel = ui.element(None, box_at(0.0, 0.0, 50.0, 50.0));
        ui.world.insert(
            panel,
            Image {
                texture: Some(AssetId::new()),
                slice: [8.0; 4],
                ..Image::default()
            },
        );
        let canvas = ui.commands();
        assert!(matches!(
            canvas.commands(),
            [UiCommand::Image {
                uv: [0.0, 0.0, 1.0, 1.0],
                ..
            }]
        ));
    }

    #[test]
    fn text_is_sized_in_canvas_units() {
        let mut ui = Ui::new(Canvas {
            scale_mode: ScaleMode::ScaleWithScreen,
            reference_resolution: Vec2::new(960.0, 540.0),
            ..Canvas::default()
        });
        let label = ui.element(None, box_at(10.0, 10.0, 100.0, 20.0));
        ui.world.insert(
            label,
            Text {
                text: "Score".into(),
                size: 16.0,
                anchor: TextAnchor::MiddleCenter,
                wrap: false,
                ..Text::default()
            },
        );
        let canvas = ui.commands();
        let [UiCommand::Text { rect, style, .. }] = canvas.commands() else {
            panic!("{:?}", canvas.commands());
        };
        assert_eq!(*rect, Rect::new(20.0, 20.0, 200.0, 40.0));
        assert_eq!(style.size, 32.0);
        assert_eq!(style.anchor, TextAnchor::MiddleCenter);
        assert!(!style.wrap);
    }

    use crate::scene::{GradientPaint, ShapePaint, ShapeStyle, UiStyle};
    use crate::ui::{
        Corner, CornerStyle, Gradient, ScrollFrame, ScrollbarVisibility, Shadow, Stroke,
        StrokePlacement,
    };

    fn shape(canvas: &UiCanvas, index: u32) -> ShapeStyle {
        match canvas.styles().iter().find(|(at, _)| *at == index) {
            Some((_, UiStyle::Shape(shape))) => *shape,
            other => panic!("command {index} has {other:?}"),
        }
    }

    fn rect_of(command: &UiCommand) -> Rect {
        match command {
            UiCommand::Panel { rect, .. }
            | UiCommand::Image { rect, .. }
            | UiCommand::Text { rect, .. } => *rect,
        }
    }

    /// The modifiers cost an unstyled tree nothing: no style and no clip is
    /// recorded for it.
    #[test]
    fn a_plain_tree_records_no_styles_and_no_clips() {
        let mut ui = Ui::new(Canvas::default());
        let panel = ui.element(None, box_at(0.0, 0.0, 100.0, 40.0));
        ui.world.insert(panel, filled(0.5));
        let label = ui.element(Some(panel), RectTransform::stretched());
        ui.world.insert(
            label,
            Text {
                text: "hp".into(),
                ..Text::default()
            },
        );
        let canvas = ui.commands();
        assert_eq!(canvas.commands().len(), 2);
        assert!(canvas.styles().is_empty(), "{:?}", canvas.styles());
        assert!(canvas.clips().is_empty(), "{:?}", canvas.clips());
    }

    /// A corner cuts the image to its element's box, and a gradient across it
    /// starts at one side and finishes at the other.
    #[test]
    fn a_corner_and_a_gradient_cut_and_colour_the_image() {
        let mut ui = Ui::new(Canvas::default());
        let card = ui.element(None, box_at(10.0, 20.0, 200.0, 100.0));
        ui.world.insert(card, filled(1.0));
        ui.world.insert(
            card,
            Corner {
                radius: 10.0,
                style: CornerStyle::Squircle,
                exponent: 5.0,
                ..Corner::default()
            },
        );
        ui.world.insert(
            card,
            Gradient {
                start: [1.0, 1.0, 1.0, 1.0],
                end: [0.0, 0.0, 0.0, 0.0],
                rotation: 90.0,
                offset: Vec2::new(0.0, 0.25),
            },
        );
        let canvas = ui.commands();
        assert_eq!(canvas.commands().len(), 1);
        let style = shape(&canvas, 0);
        let rect = Rect::new(10.0, 20.0, 200.0, 100.0);
        assert_eq!(style.bounds, Some(rect));
        assert_eq!((style.radius, style.exponent), (10.0, 5.0));
        assert_eq!(style.paint, ShapePaint::Fill);
        let GradientPaint {
            start,
            end,
            origin,
            step,
        } = style.gradient.expect("a gradient");
        assert_eq!((start, end), ([255; 4], [0; 4]));
        // Down the box, moved a quarter of its height: all start at y 45, all
        // end at y 145.
        assert!(
            (origin[0] - 110.0).abs() < 1e-3 && (origin[1] - 45.0).abs() < 1e-3,
            "{origin:?}"
        );
        assert!(
            step[0].abs() < 1e-6 && (step[1] - 0.01).abs() < 1e-6,
            "{step:?}"
        );
    }

    /// Under the element its shadow, then its image, then the stroke over it,
    /// each following the corner.
    #[test]
    fn a_shadow_goes_under_the_image_and_a_stroke_over_it() {
        let mut ui = Ui::new(Canvas::default());
        let card = ui.element(None, box_at(100.0, 100.0, 200.0, 100.0));
        ui.world.insert(card, filled(1.0));
        ui.world.insert(
            card,
            Corner {
                radius: 12.0,
                ..Corner::default()
            },
        );
        ui.world.insert(
            card,
            Shadow {
                color: [0.0, 0.0, 0.0, 0.5],
                offset: Vec2::new(0.0, 8.0),
                blur: 10.0,
                spread: 2.0,
            },
        );
        ui.world.insert(
            card,
            Stroke {
                color: [0.0, 0.0, 1.0, 1.0],
                thickness: 4.0,
                placement: StrokePlacement::Inside,
            },
        );
        let canvas = ui.commands();
        let colours: Vec<[u8; 4]> = canvas
            .commands()
            .iter()
            .map(|command| match command {
                UiCommand::Panel { color, .. } => *color,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(
            colours,
            [[0, 0, 0, 128], [255, 0, 0, 255], [0, 0, 255, 255]]
        );

        let shadow = shape(&canvas, 0);
        // Offset down by 8 and grown by 2 on every side; blurred by a sigma of
        // half the blur radius, and drawn out to three of them.
        assert_eq!(shadow.bounds, Some(Rect::new(98.0, 106.0, 204.0, 104.0)));
        assert_eq!(shadow.radius, 14.0);
        assert_eq!(shadow.paint, ShapePaint::Blur { sigma: 5.0 });
        assert_eq!(
            rect_of(&canvas.commands()[0]),
            Rect::new(83.0, 91.0, 234.0, 134.0)
        );

        assert_eq!(
            shape(&canvas, 1).bounds,
            Some(Rect::new(100.0, 100.0, 200.0, 100.0))
        );
        assert_eq!(shape(&canvas, 1).radius, 12.0);

        // Inside: the band's centre is half its width in from the edge, and so
        // is the curve it follows.
        let stroke = shape(&canvas, 2);
        assert_eq!(stroke.bounds, Some(Rect::new(102.0, 102.0, 196.0, 96.0)));
        assert_eq!(stroke.radius, 10.0);
        assert_eq!(stroke.paint, ShapePaint::Ring { width: 4.0 });
        assert_eq!(
            rect_of(&canvas.commands()[2]),
            Rect::new(100.0, 100.0, 200.0, 100.0)
        );

        // Outside, over an image: the band reaches a pixel in under the edge,
        // so no background shows between the two.
        ui.world.get_mut::<Stroke>(card).unwrap().placement = StrokePlacement::Outside;
        let canvas = ui.commands();
        let stroke = shape(&canvas, 2);
        assert_eq!(stroke.bounds, Some(Rect::new(98.5, 98.5, 203.0, 103.0)));
        assert_eq!(stroke.radius, 13.5);
        assert_eq!(stroke.paint, ShapePaint::Ring { width: 5.0 });
        assert_eq!(
            rect_of(&canvas.commands()[2]),
            Rect::new(96.0, 96.0, 208.0, 108.0)
        );
    }

    /// A stroke on a label rings its glyphs, under them; on an element that
    /// also has an image it is the box's.
    #[test]
    fn a_stroke_on_text_alone_outlines_its_glyphs() {
        let mut ui = Ui::new(Canvas::default());
        let label = ui.element(None, box_at(0.0, 0.0, 200.0, 40.0));
        ui.world.insert(
            label,
            Text {
                text: "Play".into(),
                color: [1.0; 4],
                ..Text::default()
            },
        );
        ui.world.insert(
            label,
            Stroke {
                color: [0.0, 0.0, 0.0, 1.0],
                thickness: 3.0,
                placement: StrokePlacement::Center,
            },
        );
        ui.world.insert(label, Gradient::default());
        let canvas = ui.commands();
        let colours: Vec<[u8; 4]> = canvas
            .commands()
            .iter()
            .map(|command| match command {
                UiCommand::Text { style, .. } => style.color,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(colours, [[0, 0, 0, 255], [255; 4]], "the outline first");
        assert_eq!(canvas.styles()[0], (0, UiStyle::Outline { width: 3.0 }));
        let fill = shape(&canvas, 1);
        assert_eq!(fill.bounds, None, "a gradient colours glyphs and cuts none");
        assert!(fill.gradient.is_some());

        ui.world.insert(label, filled(1.0));
        assert_eq!(
            summary(&ui.commands()),
            ["255", "0", "T:Play"],
            "the image, the box's stroke, then the text"
        );
    }

    /// A frame's content is cut to it, and its bars are drawn over the content
    /// once the frame's subtree is done.
    #[test]
    fn a_scroll_frames_content_is_cut_to_it_and_its_bars_drawn_over_it() {
        let mut ui = Ui::new(Canvas::default());
        let frame = ui.element(None, box_at(0.0, 0.0, 200.0, 100.0));
        ui.world.insert(frame, filled(0.1));
        ui.world.insert(
            frame,
            ScrollFrame {
                content_size: Vec2::new(0.0, 400.0),
                scrollbar: ScrollbarVisibility::Auto,
                scrollbar_thickness: 8.0,
                scrollbar_color: [0.0, 1.0, 0.0, 1.0],
                ..ScrollFrame::default()
            },
        );
        let row = ui.element(Some(frame), box_at(0.0, 0.0, 200.0, 30.0));
        ui.world.insert(row, filled(0.2));
        let after = ui.element(None, box_at(0.0, 200.0, 50.0, 50.0));
        ui.world.insert(after, filled(0.3));

        let canvas = ui.commands();
        assert_eq!(summary(&canvas), ["26", "51", "0", "77"]);
        assert_eq!(
            canvas.clips(),
            [(1, Some(Rect::new(0.0, 0.0, 200.0, 100.0))), (2, None)],
            "the row is cut, the bar and what follows are not"
        );
        let UiCommand::Panel { rect, color } = canvas.commands()[2] else {
            panic!("{:?}", canvas.commands()[2]);
        };
        assert_eq!(color, [0, 255, 0, 255]);
        assert_eq!(rect, Rect::new(192.0, 0.0, 8.0, 25.0));
        let thumb = shape(&canvas, 2);
        assert_eq!(thumb.bounds, Some(rect));
        assert_eq!(thumb.radius, 4.0, "a pill");
    }

    /// Nine pieces, one box: each is cut to the element's corners rather than
    /// its own, and each is styled once.
    #[test]
    fn a_sliced_image_with_a_corner_cuts_every_piece_to_the_whole_box() {
        let mut ui = Ui::new(Canvas::default());
        let panel = ui.element(None, box_at(10.0, 20.0, 200.0, 100.0));
        ui.world.insert(
            panel,
            Image {
                texture: Some(AssetId::new()),
                slice: [8.0; 4],
                ..Image::default()
            },
        );
        ui.world.insert(panel, Corner::default());
        let canvas = ui.commands_with(Some([64, 64]));
        assert_eq!(canvas.commands().len(), 9);
        let indices: Vec<u32> = canvas.styles().iter().map(|(at, _)| *at).collect();
        assert_eq!(indices, (0..9).collect::<Vec<_>>());
        for at in 0..9 {
            assert_eq!(
                shape(&canvas, at).bounds,
                Some(Rect::new(10.0, 20.0, 200.0, 100.0))
            );
        }
    }
}
