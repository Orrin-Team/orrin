use glam::Vec2;
use orrin_ecs::World;
use orrin_registry::AssetId;

use super::events::UiPointer;
use super::{Button, Image, Layout, Text};
use crate::scene::{Rect, TextStyle, UiCanvas, color_bytes};

/// Record what `layout`'s elements draw into `out`, back to front: each
/// element's image, then its text, then its children. `extent` is a texture's
/// uploaded size, which a sliced image is measured in.
pub fn extract(
    world: &World,
    layout: &Layout,
    extent: impl Fn(AssetId) -> Option<[u32; 2]>,
    out: &mut UiCanvas,
) {
    let pointer = world.get_resource::<UiPointer>();
    for (index, element) in layout.elements().iter().enumerate() {
        let entity = element.entity;
        // Screen pixels per unit of the element's own frame.
        let pixels = element.units * element.scale;
        if let Some(image) = world.get::<Image>(entity).map(|image| *image) {
            let tint = match world.get::<Button>(entity) {
                Some(button) => button_tint(&button, pointer.as_deref(), layout, index),
                None => [1.0; 4],
            };
            let color = color_bytes(std::array::from_fn(|i| image.color[i] * tint[i]));
            match image.texture {
                None => out.panel(element.rect, color),
                Some(texture) => match extent(texture).filter(|_| image.slice != [0.0; 4]) {
                    Some(size) => sliced(out, element.rect, texture, &image, size, pixels, color),
                    None => out.image(element.rect, texture, color),
                },
            }
        }
        if let Some(text) = world.get::<Text>(entity) {
            let style = TextStyle {
                font: text.font,
                size: text.size * pixels.y,
                anchor: text.anchor,
                color: color_bytes(text.color),
                wrap: text.wrap,
            };
            out.text(element.rect, style, &text.text);
        }
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
fn sliced(
    out: &mut UiCanvas,
    rect: Rect,
    texture: AssetId,
    image: &Image,
    [width, height]: [u32; 2],
    pixels: Vec2,
    color: [u8; 4],
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
            out.image_region(piece, texture, uv, color);
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
}
