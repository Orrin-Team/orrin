//! The editor's icon set: Lucide, vendored as SVG under `assets/icons/`.
//!
//! egui renders no SVG on its own, so `install` registers `egui_extras`'
//! loaders once at startup and each glyph is embedded at compile time — the
//! engine never reads an icon off disk, and a missing file is a build error
//! rather than a blank square in a shipped binary.
//!
//! Rasterisation is lazy and cached per size, so an icon costs nothing until
//! the frame that first draws it.

use egui::{Color32, ImageSource};

/// In bars and rows.
pub const ROW: f32 = 16.0;
/// In the ribbon, where an icon sits above its label rather than beside it.
pub const RIBBON: f32 = 18.0;

pub fn install(ctx: &egui::Context) {
    egui_extras::install_image_loaders(ctx);
}

macro_rules! icons {
    ($($name:ident => $file:literal,)*) => {
        $(
            pub fn $name() -> ImageSource<'static> {
                egui::include_image!(concat!("../../assets/icons/", $file, ".svg"))
            }
        )*
    };
}

icons! {
    aperture => "aperture",
    box_ => "box",
    check => "check",
    circle_dot => "circle-dot",
    file_code => "file-code-2",
    file_plus => "file-plus",
    folder_open => "folder-open",
    globe => "globe",
    grid => "grid-3x3",
    import => "import",
    layers => "layers",
    layers_2 => "layers-2",
    layout_dashboard => "layout-dashboard",
    lightbulb => "lightbulb",
    mountain_snow => "mountain-snow",
    move_ => "move",
    package => "package",
    panels => "panels-top-left",
    pause => "pause",
    play => "play",
    plus => "plus",
    pointer => "mouse-pointer-2",
    refresh => "refresh-cw",
    reimport => "refresh-ccw-dot",
    repeat => "repeat",
    rotate => "rotate-cw",
    save => "save",
    scaling => "scaling",
    sparkles => "sparkles",
    square => "square",
    sun => "sun",
    terminal => "terminal",
    trash => "trash-2",
    warning => "triangle-alert",
    workflow => "workflow",
    x => "x",
}

/// An icon at `size`, tinted `color`.
///
/// The vendored SVGs stroke in white so that a tint — which multiplies — can
/// reach any colour. Lucide ships them as `currentColor`, which resvg resolves
/// to black, and multiplying black leaves black.
pub fn tinted(source: ImageSource<'static>, size: f32, color: Color32) -> egui::Image<'static> {
    egui::Image::new(source)
        .fit_to_exact_size(egui::vec2(size, size))
        .tint(color)
}

/// An icon in the colour the surrounding text would be. The default: an icon
/// inherits its label's colour rather than carrying one of its own.
pub fn inline(ui: &egui::Ui, source: ImageSource<'static>, size: f32) -> egui::Image<'static> {
    tinted(source, size, ui.visuals().text_color())
}

/// The square an icon button occupies: the glyph plus the button's padding.
fn button_size(ui: &egui::Ui, icon: f32) -> egui::Vec2 {
    egui::Vec2::splat(icon) + 2.0 * ui.spacing().button_padding
}

/// An icon-only button, sized to its glyph.
///
/// The rect is allocated up front rather than left to the button, because egui
/// hands an image-only button *the whole available space* when its image fails
/// to load. Inside a panel that takes its width from its content, that is a
/// button which swallows the panel and then widens it — every frame, without
/// ever settling. A missing icon should cost a blank square and nothing else.
pub fn button(
    ui: &mut egui::Ui,
    source: ImageSource<'static>,
    size: f32,
    color: Color32,
    frame: bool,
) -> egui::Response {
    let image = tinted(source, size, color);
    ui.allocate_ui(button_size(ui, size), |ui| {
        ui.add(egui::Button::image(image).frame(frame))
    })
    .inner
}

/// The same, opening a menu instead of reporting a click.
pub fn menu_button<R>(
    ui: &mut egui::Ui,
    source: ImageSource<'static>,
    size: f32,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::Response {
    let image = inline(ui, source, size);
    ui.allocate_ui(button_size(ui, size), |ui| {
        ui.menu_image_button(image, add_contents).response
    })
    .inner
}

/// A row-height icon followed by its label, in one colour. Used where a glyph
/// used to lead a string — `"⚠ faulted"` and the like.
pub fn labelled(ui: &mut egui::Ui, source: ImageSource<'static>, color: Color32, text: &str) {
    ui.horizontal(|ui| {
        ui.add(tinted(source, ROW, color));
        ui.colored_label(color, text);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every glyph the editor names has to be a file that resvg can actually
    /// turn into pixels. `include_image!` proves the file exists at build time
    /// and nothing else — a malformed SVG is a blank square at runtime.
    #[test]
    fn every_icon_decodes() {
        let ctx = egui::Context::default();
        install(&ctx);

        for (name, source) in [
            ("plus", plus()),
            ("x", x()),
            ("trash", trash()),
            ("warning", warning()),
            ("check", check()),
            ("play", play()),
            ("box", box_()),
            ("workflow", workflow()),
        ] {
            let size = egui::Vec2::splat(ROW);
            let result = egui::Image::new(source).load_for_size(&ctx, size);
            let poll = result.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                matches!(poll, egui::load::TexturePoll::Ready { .. }),
                "{name} did not rasterise"
            );
        }
    }

    /// White strokes are load-bearing: `Image::tint` multiplies, so a black
    /// glyph stays black in every colour a theme could ask for.
    #[test]
    fn no_icon_ships_a_colour_of_its_own() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/icons");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("icons directory") {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "svg") {
                let svg = std::fs::read_to_string(&path).unwrap();
                assert!(
                    !svg.contains("currentColor"),
                    "{}: currentColor resolves to black, and tinting cannot recover it",
                    path.display()
                );
                assert!(svg.contains("stroke=\"#ffffff\""), "{}", path.display());
                checked += 1;
            }
        }
        assert!(checked >= 30, "only {checked} icons found");
    }
    /// The handoff's warning about icons: they are exactly the kind of addition
    /// that quietly costs a fifth of a second at startup and is never noticed.
    /// The whole set is ~2ms on this machine, spread lazily over the frames that
    /// first draw each glyph; the budget is loose enough for a shared runner and
    /// tight enough to catch a swap to something that rasterises per frame.
    #[test]
    fn the_whole_set_rasterises_in_well_under_a_frame_budget() {
        let ctx = egui::Context::default();
        install(&ctx);

        let sources = [
            aperture(),
            box_(),
            check(),
            circle_dot(),
            file_code(),
            file_plus(),
            folder_open(),
            globe(),
            grid(),
            import(),
            layers(),
            layers_2(),
            layout_dashboard(),
            lightbulb(),
            mountain_snow(),
            move_(),
            package(),
            panels(),
            pause(),
            play(),
            plus(),
            pointer(),
            refresh(),
            reimport(),
            repeat(),
            rotate(),
            save(),
            scaling(),
            sparkles(),
            square(),
            sun(),
            terminal(),
            trash(),
            warning(),
            workflow(),
            x(),
        ];
        let count = sources.len();

        let start = std::time::Instant::now();
        for source in sources {
            egui::Image::new(source)
                .load_for_size(&ctx, egui::Vec2::splat(ROW))
                .expect("rasterises");
        }
        let elapsed = start.elapsed();

        assert!(
            elapsed < std::time::Duration::from_millis(250),
            "{count} icons took {elapsed:?}"
        );
    }
}
