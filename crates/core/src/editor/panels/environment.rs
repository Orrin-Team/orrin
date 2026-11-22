//! The Environment tab: the scene's render settings, and below them the
//! editor's own camera.
//!
//! About seventy controls, so they are found rather than scanned for: each
//! group is a section bar of the inspector's kind, which folds and remembers
//! being folded, and the inspector's filter runs over the lot — a section
//! whose title matches shows whole, any other only its matching rows, and a
//! match inside a folded section shows all the same. The sections are laid
//! out in as many columns as the tab can give each a label column that holds
//! the tab's longest label, so a narrow tab is one column of whole labels
//! rather than four of cut ones; a label longer than any label column is
//! whole on hover.
//!
//! **The camera is the editor's, not the scene's.** Position, target and
//! field of view are the `Camera` resource, which the fly camera drives and
//! no scene file holds: a running game overwrites it from its own camera each
//! frame, and Stop puts the editor's back. So that section is named for
//! what it is, comes after everything the scene keeps, starts folded, and
//! takes the tab's whole width, which three axis fields need.
//!
//! **The sky is picked, not typed.** The HDRI is a file under the assets
//! folder and the asset database does not list `.hdr` images, so it is
//! chosen in the system's file dialog — on a thread of its own, polled, as
//! the File menu's is — and shown read only. A file outside the assets folder
//! is refused in the console, since a scene can only name one inside it.

use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use orrin_ecs::World;
use orrin_registry::Reflect;
use orrin_registry::inspect::{LABEL_INSET, label_column};

use super::inspector::{Filter, Rows};
use super::{color_row, vec3_row};
use crate::editor::icons::{self, Icon};
use crate::editor::state::EditorState;
use crate::editor::widgets::{self, Opening};
use crate::gfx::shadows::MAX_CASCADES;
use crate::scene::registry::RENDER_SETTINGS;
use crate::scene::render_settings;
use crate::scene::{
    AmbientLight, BloomSettings, Camera, ContactShadowSettings, DecalSettings, DofSettings,
    EnvironmentSettings, FogSettings, HdrSettings, MotionBlurSettings, RefractionSettings,
    ShadowSettings, SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings,
    TransparencySettings,
};

/// What the stand-in for the HDRI file dialog picks, in tests: `None` inside
/// is a dialog closed without a pick.
#[cfg(test)]
pub(super) static HDRI_PICKED: Mutex<Option<Option<PathBuf>>> = Mutex::new(None);

/// A group of settings under one bar: its rows' labels, which the filter
/// and the column count read, and what draws them.
struct Section {
    title: &'static str,
    hint: Option<&'static str>,
    icon: Icon,
    open: bool,
    rows: &'static [&'static str],
    draw: fn(&mut egui::Ui, &World, Rows),
}

const fn section(
    title: &'static str,
    hint: Option<&'static str>,
    icon: Icon,
    rows: &'static [&'static str],
    draw: fn(&mut egui::Ui, &World, Rows),
) -> Section {
    Section {
        title,
        hint,
        icon,
        open: true,
        rows,
        draw,
    }
}

const SCREEN_SPACE: &[Section] = &[
    section(
        "Ambient occlusion",
        None,
        icons::cube_transparent(),
        &["Enabled", "Radius", "Bias", "Power", "Half resolution"],
        ambient_occlusion,
    ),
    section(
        "Temporal AA",
        Some("Jitter the camera each frame and accumulate the results"),
        icons::frame_corners(),
        &["Enabled", "Feedback", "Jitter", "4x MSAA"],
        temporal_aa,
    ),
    section(
        "Reflections",
        Some("Reflect the frame off itself, where the frame contains what a surface faces"),
        icons::selection_background(),
        &[
            "Screen-space reflections",
            "Intensity",
            "Max roughness",
            "Thickness",
            "Range",
            "Steps",
        ],
        reflections,
    ),
    section(
        "Subsurface scattering",
        Some("Light that enters a surface, scatters inside it and leaves somewhere else"),
        icons::circle(),
        &[
            "Screen-space diffusion",
            "Max radius",
            "Red reach",
            "Green reach",
            "Blue reach",
        ],
        subsurface,
    ),
    section(
        "Transparency",
        Some("Weighted-blended order-independent transparency (McGuire & Bavoil)"),
        icons::eye_slash(),
        &["Enabled"],
        transparency,
    ),
    section(
        "Refraction",
        Some("Screen-space refraction: a transmissive material samples the frame behind it"),
        icons::eye(),
        &["Enabled"],
        refraction,
    ),
    section(
        "Decals",
        Some("Boxes that stamp maps onto the surfaces inside them, before shading"),
        icons::image(),
        &["Enabled"],
        decals,
    ),
];

const SHADOWS: &[Section] = &[
    section(
        "Shadows",
        None,
        icons::square(),
        &[
            "Enabled",
            "Cascades",
            "Resolution",
            "Distance",
            "Split blend",
            "Pullback",
            "Bias",
            "Slope bias",
            "Softness",
            "Sharpness",
            "Strength",
            "Tint cascades",
        ],
        shadows,
    ),
    section(
        "Point & spot shadows",
        Some("One atlas, a tile per cube face, drawn in a single pass"),
        icons::lightbulb(),
        &[
            "Enabled",
            "Atlas",
            "Near",
            "Bias",
            "Slope bias",
            "Softness",
            "Sharpness",
        ],
        punctual_shadows,
    ),
    section(
        "Contact shadows",
        Some("March the depth buffer for the band a cascade texel is too coarse to hold"),
        icons::bounding_box(),
        &[
            "Enabled",
            "Length",
            "Steps",
            "Bias",
            "Thickness",
            "Strength",
            "Range",
        ],
        contact_shadows,
    ),
];

const LIGHTING: &[Section] = &[
    section(
        "Tonemap",
        None,
        icons::sliders(),
        &[
            "Auto exposure",
            "Compensation",
            "Brighten",
            "Darken",
            "Meter floor",
            "Meter ceiling",
            "EV100",
        ],
        tonemap,
    ),
    section(
        "Bloom",
        Some("Filtered on the exposed image, so the strength holds across lighting changes"),
        icons::sun(),
        &["Enabled", "Strength", "Radius", "Scatter"],
        bloom,
    ),
    section(
        "Ambient",
        Some("Fallback only: an environment's irradiance replaces this when one is loaded"),
        icons::globe(),
        &["Color", "Luminance"],
        ambient,
    ),
    section(
        "Sky",
        None,
        icons::mountain_snow(),
        &[
            "HDRI",
            "Show skybox",
            "Background",
            "Sky",
            "Offset",
            "Rotation",
        ],
        sky,
    ),
    section(
        "Fog",
        None,
        icons::layers(),
        &[
            "Albedo",
            "Density",
            "Falloff",
            "Height",
            "Volumetric",
            "Distance",
            "Anisotropy",
            "Feedback",
        ],
        fog,
    ),
];

const LENS: &[Section] = &[
    section(
        "Lens",
        Some("Defocus everything the lens is not focused on"),
        icons::circle_dot(),
        &["Depth of field", "Focus", "Aperture", "Sensor"],
        lens,
    ),
    section(
        "Shutter",
        Some("Reconstruct the exposure from the frame's motion vectors"),
        icons::hourglass(),
        &["Motion blur", "Angle"],
        shutter,
    ),
];

/// The groups the columns are dealt, in reading order.
const COLUMNS: [&[Section]; 4] = [SCREEN_SPACE, SHADOWS, LIGHTING, LENS];

const EDITOR_CAMERA: Section = Section {
    title: "Editor camera",
    hint: Some(
        "The editor's view of the scene: not saved with it, and replaced by the \
         scene's own camera while the game plays",
    ),
    icon: icons::camera(),
    open: false,
    rows: &["Position", "Target", "FOV"],
    draw: editor_camera,
};

/// A slider plus the label that sits after it. Narrower than this and a column's
/// contents run into its neighbour instead of wrapping — `Ui::columns` divides
/// the width but does nothing to make the contents fit it.
///
/// This was once a full-window strip along the bottom, where four cramped
/// columns were the intended look and folding was only a guard against a
/// negative width. Docked, the same panel is a 300px tab, and four columns there
/// is four overlapping ones. The number is now what it says it is.
const MIN_COLUMN_WIDTH: f32 = 190.0;

/// How many columns `available` can carry, `gap` apart: each at least
/// [`MIN_COLUMN_WIDTH`], and wide enough that its label column holds a label
/// `label` points long — or, for a label longer than any label column, is as
/// wide as a label column gets. Never zero: `Ui::columns` divides by the
/// count, and it asserts on the negative width a shortfall would produce.
fn column_count(available: f32, label: f32, gap: f32) -> usize {
    let widest = label_column(f32::INFINITY);
    (1..=COLUMNS.len())
        .rev()
        .find(|&count| {
            let column = (available - gap * (count - 1) as f32) / count as f32;
            // A section's rows stop short of its right edge by its margin.
            let labels = label_column(column - widgets::PAD);
            column >= MIN_COLUMN_WIDTH && (labels - LABEL_INSET >= label || labels >= widest)
        })
        .unwrap_or(1)
}

/// The widest of the columns' row labels in the body font, measured once per
/// font rather than every frame.
fn longest_label(ui: &egui::Ui) -> f32 {
    let font = egui::TextStyle::Body.resolve(ui.style());
    let memory = egui::Id::new("environment longest label");
    if let Some((measured, width)) = ui.data(|data| data.get_temp::<(egui::FontId, f32)>(memory))
        && measured == font
    {
        return width;
    }
    let width = ui.fonts(|fonts| {
        COLUMNS
            .iter()
            .flat_map(|group| group.iter())
            .flat_map(|section| section.rows.iter())
            .map(|label| {
                fonts
                    .layout_no_wrap(
                        (*label).to_owned(),
                        font.clone(),
                        egui::Color32::PLACEHOLDER,
                    )
                    .size()
                    .x
            })
            .fold(0.0, f32::max)
    });
    ui.data_mut(|data| data.insert_temp(memory, (font, width)));
    width
}

fn slider(ui: &mut egui::Ui, label: &str, slider: egui::Slider<'_>) -> egui::Response {
    widgets::property(ui, label, |ui| ui.add(slider))
}

fn toggle(ui: &mut egui::Ui, label: &str, value: &mut bool) -> egui::Response {
    widgets::property(ui, label, |ui| ui.checkbox(value, ""))
}

pub fn body(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    // Stacked whatever the parent's layout, so the note sits above the columns
    // rather than taking their width.
    ui.vertical(|ui| {
        if render_settings::source(world).is_none() {
            unsaved_note(ui, state);
        }
        widgets::search(ui, &mut state.environment_query, "Filter settings");
        ui.add_space(4.0);
        let filter = Filter::new(&state.environment_query);
        // A docked tab is as wide as its split, which can be nothing at all.
        let available = ui.available_width();
        if available > 0.0 {
            let gap = ui.spacing().item_spacing.x;
            let columns = column_count(available, longest_label(ui), gap);
            edit_settings(world, state, |world| {
                // In here, so a pick reaches the scene's settings as an edit.
                poll_dialog(ui.ctx(), world);
                ui.columns(columns, |cols| {
                    for (index, group) in COLUMNS.iter().enumerate() {
                        let target = index * columns / COLUMNS.len();
                        for section in *group {
                            show(&mut cols[target], world, &filter, section);
                        }
                    }
                });
            });
            ui.add_space(6.0);
            show(ui, world, &filter, &EDITOR_CAMERA);
        }
        ui.add_space(4.0);
    });
}

fn show(ui: &mut egui::Ui, world: &World, filter: &Filter, section: &Section) {
    let Some(rows) = filter.section(section.title, section.rows) else {
        return;
    };
    let opening = Opening {
        open: section.open,
        held: !filter.is_empty(),
    };
    let tint = ui.visuals().weak_text_color();
    widgets::section_with(
        ui,
        egui::Id::new(("environment section", section.title)),
        opening,
        section.icon,
        tint,
        section.title,
        section.hint,
        |ui| (section.draw)(ui, world, rows),
    );
}

/// Said where the settings are, because without it nothing on screen tells the
/// user that the evening's tuning will be gone on the next load.
pub fn unsaved_note(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.add(
        egui::Label::new(
            egui::RichText::new("These settings are not part of the scene, and are lost on load.")
                .weak(),
        )
        .wrap(),
    );
    if ui
        .button("Save with scene")
        .on_hover_text("Add a Render Settings entity holding them, saved with the scene")
        .clicked()
    {
        state.request_add_render_settings();
    }
    ui.add_space(4.0);
}

/// Run `draw`, which edits the renderer's resources directly, and carry what it
/// changed into the scene's [`RenderSettings`](render_settings::RenderSettings)
/// when the scene has one.
///
/// The widgets keep writing resources rather than a copy of the component, so
/// the same controls serve a scene with settings and one without. The
/// resources are brought in line with the scene first: the editor draws before
/// the frame syncs, and without it an edit on the frame after a load would
/// write the previous scene's values into this one.
pub fn edit_settings(world: &mut World, state: &mut EditorState, draw: impl FnOnce(&mut World)) {
    let Some(entity) = render_settings::source(world) else {
        draw(world);
        return;
    };
    render_settings::sync(world);
    let before = render_settings::capture(world);
    draw(world);
    let after = render_settings::capture(world);
    if after != before {
        state.edit(entity, RENDER_SETTINGS, after.to_value());
    }
}

fn ambient_occlusion(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut s = world.resource_mut::<SsaoSettings>();
    if rows.shows("Enabled") {
        toggle(ui, "Enabled", &mut s.enabled);
    }
    if rows.shows("Radius") {
        slider(ui, "Radius", egui::Slider::new(&mut s.radius, 0.0..=4.0));
    }
    if rows.shows("Bias") {
        slider(ui, "Bias", egui::Slider::new(&mut s.bias, 0.0..=0.1));
    }
    if rows.shows("Power") {
        slider(ui, "Power", egui::Slider::new(&mut s.power, 0.1..=4.0));
    }
    if rows.shows("Half resolution") {
        toggle(ui, "Half resolution", &mut s.half_resolution).on_hover_text(
            "Resolve the occlusion at half the frame's extent and let the \
                 bilateral blur upsample it: a quarter of the pixels through the \
                 frame's most expensive loop, for a term that is low-frequency \
                 and blurred either way",
        );
    }
}

fn temporal_aa(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut taa = world.resource_mut::<TaaSettings>();
    // Frame structure, like SSAO and bloom: it registers the resolve node
    // and it is what keeps the geometry prepass in a frame with SSAO off.
    if rows.shows("Enabled") {
        toggle(ui, "Enabled", &mut taa.enabled);
    }
    if rows.shows("Feedback") {
        slider(
            ui,
            "Feedback",
            egui::Slider::new(&mut taa.feedback, 0.5..=0.98),
        )
        .on_hover_text("How much of the reprojected history each frame keeps");
    }
    if rows.shows("Jitter") {
        slider(
            ui,
            "Jitter",
            egui::Slider::new(&mut taa.jitter_scale, 0.0..=1.5),
        )
        .on_hover_text("Fraction of a pixel the camera samples across");
    }
    if rows.shows("4x MSAA") {
        toggle(ui, "4x MSAA", &mut taa.msaa).on_hover_text(
            "Rasterise the forward pass at four samples instead of one. The \
             alternative to the temporal resolve above, not a companion to it: \
             it costs roughly a quarter of the frame, and at one sample the \
             forward pass borrows the prepass depth and shades each pixel once",
        );
    }
}

fn reflections(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut ssr = world.resource_mut::<SsrSettings>();
    // Frame structure like the rest: it registers the pyramid, the trace and
    // the composite, and it is another reason the prepass exists.
    if rows.shows("Screen-space reflections") {
        toggle(ui, "Screen-space reflections", &mut ssr.enabled);
    }
    if rows.shows("Intensity") {
        slider(
            ui,
            "Intensity",
            egui::Slider::new(&mut ssr.intensity, 0.0..=1.0),
        )
        .on_hover_text("1.0 is the physical answer; lower hides the screen-space failures");
    }
    if rows.shows("Max roughness") {
        slider(
            ui,
            "Max roughness",
            egui::Slider::new(&mut ssr.max_roughness, 0.05..=1.0),
        )
        .on_hover_text("Rougher than this is left to the environment cubemap");
    }
    if rows.shows("Thickness") {
        slider(
            ui,
            "Thickness",
            egui::Slider::new(&mut ssr.thickness, 0.01..=2.0)
                .logarithmic(true)
                .suffix(" m"),
        )
        .on_hover_text(
            "How deep a surface is assumed to be — the depth buffer records only its front",
        );
    }
    if rows.shows("Range") {
        slider(
            ui,
            "Range",
            egui::Slider::new(&mut ssr.max_distance, 1.0..=200.0)
                .logarithmic(true)
                .suffix(" m"),
        );
    }
    if rows.shows("Steps") {
        slider(ui, "Steps", egui::Slider::new(&mut ssr.max_steps, 8..=128))
            .on_hover_text("A step crosses a whole cell of the depth pyramid, not a texel");
    }
}

fn subsurface(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut subsurface = world.resource_mut::<SubsurfaceSettings>();
    // Frame structure, and more of it than the rest: this switch decides which
    // of the two forward render passes the frame opens, so it registers three
    // passes *and* a second colour attachment. It is also one more reason the
    // prepass exists — the blur weights its taps by depth.
    if rows.shows("Screen-space diffusion") {
        toggle(ui, "Screen-space diffusion", &mut subsurface.enabled).on_hover_text(
            "Off keeps scattering: the analytic wrap widens to stand in for these passes",
        );
    }
    if rows.shows("Max radius") {
        slider(
            ui,
            "Max radius",
            egui::Slider::new(&mut subsurface.max_radius, 4.0..=128.0)
                .logarithmic(true)
                .suffix(" px"),
        )
        .on_hover_text(
            "A performance clamp. The physical width comes from the material's mean free path",
        );
    }
    // Three sliders rather than a colour picker: these are distances, not a
    // colour, and the widest is pinned at one by construction — only the
    // ratios reach the shader.
    if rows.shows("Red reach") {
        slider(
            ui,
            "Red reach",
            egui::Slider::new(&mut subsurface.profile.x, 0.0..=1.0),
        );
    }
    if rows.shows("Green reach") {
        slider(
            ui,
            "Green reach",
            egui::Slider::new(&mut subsurface.profile.y, 0.0..=1.0),
        );
    }
    if rows.shows("Blue reach") {
        slider(
            ui,
            "Blue reach",
            egui::Slider::new(&mut subsurface.profile.z, 0.0..=1.0),
        )
        .on_hover_text("How far each channel travels, relative to the furthest");
    }
}

fn transparency(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut transparency = world.resource_mut::<TransparencySettings>();
    // Frame structure like the rest: it registers the accumulation and
    // composite nodes, and it is one more thing that keeps the prepass
    // alive — the accumulation depth-tests against what that pass wrote.
    if rows.shows("Enabled") {
        toggle(ui, "Enabled", &mut transparency.enabled)
            .on_hover_text("Off draws nothing at all where a blended material is used");
    }
}

fn refraction(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut refraction = world.resource_mut::<RefractionSettings>();
    // Beside transparency rather than under it: the two queues answer to
    // opposite rules about ordering, and each is its own A/B when a
    // non-opaque surface looks wrong.
    if rows.shows("Enabled") {
        toggle(ui, "Enabled", &mut refraction.enabled)
            .on_hover_text("Off draws nothing at all where a transmissive material is used");
    }
}

fn decals(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut decals = world.resource_mut::<DecalSettings>();
    // Unlike everything above it in this column, this is *not* frame
    // structure: decals add no pass and no target, so the graph is the same
    // either way and nothing is reallocated when it is clicked. It is here
    // as the A/B — decals land under the lighting, so "is this a decal or a
    // light?" is otherwise a hard question to ask of a still frame.
    if rows.shows("Enabled") {
        toggle(ui, "Enabled", &mut decals.enabled)
            .on_hover_text("Off leaves every surface as its own material describes it");
    }
}

fn shadows(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut s = world.resource_mut::<ShadowSettings>();
    if rows.shows("Enabled") {
        toggle(ui, "Enabled", &mut s.enabled);
    }
    // Cascade count and resolution are frame *structure*: changing either
    // recompiles the graph and reallocates the maps, so they are steppers
    // rather than sliders — a drag would do that once per frame.
    if rows.shows("Cascades") {
        widgets::property(ui, "Cascades", |ui| {
            ui.add(egui::DragValue::new(&mut s.cascade_count).range(1..=MAX_CASCADES as u32));
        });
    }
    if rows.shows("Resolution") {
        widgets::property(ui, "Resolution", |ui| {
            // The near cascades' tiles, then the tiles of the levels past them.
            // Both come out of the atlas the lamps share, so the first is held
            // to half its edge and the second to a quarter.
            let ShadowSettings {
                resolution,
                far_resolution,
                ..
            } = &mut *s;
            for (salt, value) in [
                ("shadow_resolution", resolution),
                ("shadow_far_resolution", far_resolution),
            ] {
                egui::ComboBox::from_id_salt(salt)
                    .selected_text(format!("{value}"))
                    .show_ui(ui, |ui| {
                        for size in [512u32, 1024, 2048, 4096] {
                            ui.selectable_value(value, size, format!("{size}"));
                        }
                    });
            }
        });
    }
    if rows.shows("Distance") {
        slider(
            ui,
            "Distance",
            egui::Slider::new(&mut s.max_distance, 10.0..=500.0).logarithmic(true),
        );
    }
    if rows.shows("Split blend") {
        slider(
            ui,
            "Split blend",
            egui::Slider::new(&mut s.lambda, 0.0..=1.0),
        );
    }
    if rows.shows("Pullback") {
        slider(
            ui,
            "Pullback",
            egui::Slider::new(&mut s.pullback, 0.0..=200.0),
        );
    }
    if rows.shows("Bias") {
        slider(
            ui,
            "Bias",
            egui::Slider::new(&mut s.constant_bias, 0.0..=8.0),
        );
    }
    if rows.shows("Slope bias") {
        slider(
            ui,
            "Slope bias",
            egui::Slider::new(&mut s.slope_bias, 0.0..=8.0),
        );
    }
    if rows.shows("Softness") {
        softness(ui, &mut s.softness);
    }
    if rows.shows("Sharpness") {
        sharpness(ui, &mut s.sharpness);
    }
    if rows.shows("Strength") {
        slider(
            ui,
            "Strength",
            egui::Slider::new(&mut s.strength, 0.0..=1.0),
        );
    }
    if rows.shows("Tint cascades") {
        toggle(ui, "Tint cascades", &mut s.debug_cascades);
    }
}

fn punctual_shadows(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut s = world.resource_mut::<ShadowSettings>();
    // Whether a *given* light casts is on the light, in the inspector; these are
    // the shape of the atlas it competes for.
    if rows.shows("Enabled") {
        toggle(ui, "Enabled", &mut s.punctual_enabled);
    }
    if rows.shows("Atlas") {
        widgets::property(ui, "Atlas", |ui| {
            // 4096 is the ceiling on purpose. Vulkan guarantees
            // `maxImageDimension2D` of only 4096, so offering 8192 is offering a
            // startup panic on a conformant device that Metal happens to let this
            // one get away with.
            egui::ComboBox::from_id_salt("shadow_atlas_resolution")
                .selected_text(format!("{}", s.atlas_resolution))
                .show_ui(ui, |ui| {
                    for size in [1024u32, 2048, 4096] {
                        ui.selectable_value(&mut s.atlas_resolution, size, format!("{size}"));
                    }
                });
            egui::ComboBox::from_id_salt("shadow_atlas_tile")
                .selected_text(format!("{}", s.atlas_tile_size))
                .show_ui(ui, |ui| {
                    for size in [128u32, 256, 512, 1024] {
                        ui.selectable_value(&mut s.atlas_tile_size, size, format!("{size}"));
                    }
                });
        });
        // Both are structure, so the reader deserves to know what they bought: a
        // point light spends six of these and a spot one.
        let tiles = s.punctual_tiles();
        ui.weak(format!(
            "{tiles} tiles beside the sun's — {} point lights",
            tiles / 6
        ));
    }
    if rows.shows("Near") {
        slider(
            ui,
            "Near",
            egui::Slider::new(&mut s.punctual_near, 0.005..=1.0)
                .logarithmic(true)
                .suffix(" m"),
        )
        .on_hover_text("Depth precision near the light, which is where its contacts are");
    }
    if rows.shows("Bias") {
        slider(
            ui,
            "Bias",
            egui::Slider::new(&mut s.punctual_constant_bias, 0.0..=8.0),
        );
    }
    if rows.shows("Slope bias") {
        slider(
            ui,
            "Slope bias",
            egui::Slider::new(&mut s.punctual_slope_bias, 0.0..=8.0),
        );
    }
    if rows.shows("Softness") {
        softness(ui, &mut s.punctual_softness);
    }
    if rows.shows("Sharpness") {
        sharpness(ui, &mut s.punctual_sharpness);
    }
}

fn softness(ui: &mut egui::Ui, value: &mut f32) {
    slider(
        ui,
        "Softness",
        egui::Slider::new(value, 0.0..=0.5)
            .logarithmic(true)
            .suffix(" m"),
    )
    .on_hover_text("Penumbra width, held wherever a map's texels are fine enough to hold it");
}

fn sharpness(ui: &mut egui::Ui, value: &mut f32) {
    slider(
        ui,
        "Sharpness",
        egui::Slider::new(value, 0.0..=crate::scene::MAX_SHADOW_SHARPNESS),
    )
    .on_hover_text("Narrows the visible penumbra and keeps the kernel that antialiases it");
}

fn contact_shadows(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut s = world.resource_mut::<ContactShadowSettings>();
    // Frame structure like the rest: it registers the march and it is one
    // more thing that keeps the geometry prepass in the frame.
    if rows.shows("Enabled") {
        toggle(ui, "Enabled", &mut s.enabled);
    }
    if rows.shows("Length") {
        slider(
            ui,
            "Length",
            egui::Slider::new(&mut s.ray_length, 0.01..=2.0)
                .logarithmic(true)
                .suffix(" m"),
        )
        .on_hover_text("Longer than a near cascade's texel spends steps on shadows the maps have");
    }
    if rows.shows("Steps") {
        slider(ui, "Steps", egui::Slider::new(&mut s.steps, 4..=64))
            .on_hover_text("The shortest occluder the march can find is one step of the ray");
    }
    if rows.shows("Bias") {
        slider(
            ui,
            "Bias",
            egui::Slider::new(&mut s.bias, 0.0..=0.2)
                .logarithmic(true)
                .suffix(" m"),
        )
        .on_hover_text("How far in front of the ray a surface must be to be something else");
    }
    if rows.shows("Thickness") {
        slider(
            ui,
            "Thickness",
            egui::Slider::new(&mut s.thickness, 0.01..=2.0)
                .logarithmic(true)
                .suffix(" m"),
        )
        .on_hover_text(
            "How deep a surface is assumed to be past the bias — the buffer records only its front",
        );
    }
    if rows.shows("Strength") {
        slider(
            ui,
            "Strength",
            egui::Slider::new(&mut s.intensity, 0.0..=1.0),
        );
    }
    if rows.shows("Range") {
        slider(
            ui,
            "Range",
            egui::Slider::new(&mut s.fade_distance, 1.0..=200.0)
                .logarithmic(true)
                .suffix(" m"),
        )
        .on_hover_text("Where the band is thinner than a pixel and the march is only noise");
    }
}

fn tonemap(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut hdr = world.resource_mut::<HdrSettings>();
    // Metering on or off is frame *structure*: it registers or drops the two
    // compute passes, so it recompiles the graph. A checkbox, not something
    // draggable.
    if rows.shows("Auto exposure") {
        toggle(ui, "Auto exposure", &mut hdr.auto_exposure)
            .on_hover_text("Meter the frame's luminance and adapt to it");
    }
    if rows.shows("Compensation") {
        slider(
            ui,
            "Compensation",
            egui::Slider::new(&mut hdr.exposure_compensation, -4.0..=4.0).suffix(" EV"),
        );
    }
    if hdr.auto_exposure {
        if rows.shows("Brighten") {
            slider(
                ui,
                "Brighten",
                egui::Slider::new(&mut hdr.adaptation_brighten, 0.0..=4.0).suffix(" s"),
            );
        }
        if rows.shows("Darken") {
            slider(
                ui,
                "Darken",
                egui::Slider::new(&mut hdr.adaptation_darken, 0.0..=4.0).suffix(" s"),
            );
        }
        if rows.shows("Meter floor") {
            slider(
                ui,
                "Meter floor",
                egui::Slider::new(&mut hdr.min_log_luminance, -16.0..=0.0),
            );
        }
        if rows.shows("Meter ceiling") {
            slider(
                ui,
                "Meter ceiling",
                egui::Slider::new(&mut hdr.max_log_luminance, 0.0..=20.0),
            );
        }
    } else if rows.shows("EV100") {
        slider(
            ui,
            "EV100",
            egui::Slider::new(&mut hdr.manual_ev100, -6.0..=16.0),
        );
    }
}

fn bloom(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut bloom = world.resource_mut::<BloomSettings>();
    // Like metering, on/off is frame structure: it registers or drops the
    // whole chain, so it recompiles the graph.
    if rows.shows("Enabled") {
        toggle(ui, "Enabled", &mut bloom.enabled);
    }
    if rows.shows("Strength") {
        slider(
            ui,
            "Strength",
            egui::Slider::new(&mut bloom.strength, 0.0..=0.3),
        );
    }
    if rows.shows("Radius") {
        slider(
            ui,
            "Radius",
            egui::Slider::new(&mut bloom.radius, 0.5..=3.0),
        );
    }
    if rows.shows("Scatter") {
        slider(
            ui,
            "Scatter",
            egui::Slider::new(&mut bloom.scatter, 0.0..=0.95),
        )
        .on_hover_text("Weight toward the blurrier levels — how far the glow reaches");
    }
}

fn ambient(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut ambient = world.resource_mut::<AmbientLight>();
    if rows.shows("Color") {
        color_row(ui, "Color", &mut ambient.color);
    }
    if rows.shows("Luminance") {
        slider(
            ui,
            "Luminance",
            egui::Slider::new(&mut ambient.nits, 0.0..=10_000.0)
                .logarithmic(true)
                .suffix(" cd/m²"),
        )
        .on_hover_text("Clear zenith a few thousand, overcast under a thousand, dusk tens");
    }
}

fn sky(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut env = world.resource_mut::<EnvironmentSettings>();
    if rows.shows("HDRI") {
        hdri_row(ui, world, &mut env);
    }
    if rows.shows("Show skybox") {
        toggle(ui, "Show skybox", &mut env.show_skybox);
    }
    // Only reachable with the skybox off or nothing baked, so it sits right
    // under the toggle that makes it visible. Drag values rather than a
    // colour picker because the number is a luminance in cd/m² and a picker
    // would quietly clamp it to the 0–1 the rest of the frame is not in.
    // Bound out of the resource guard once: three `&mut env.background.x`
    // would each go through `DerefMut` and count as three whole-resource
    // borrows, where three fields of one `Vec3` are disjoint.
    if rows.shows("Background") {
        let background = &mut env.background;
        widgets::property(ui, "Background", |ui| {
            ui.horizontal(|ui| {
                for (channel, label) in [
                    (&mut background.x, "R"),
                    (&mut background.y, "G"),
                    (&mut background.z, "B"),
                ] {
                    ui.add(
                        egui::DragValue::new(channel)
                            .speed(0.01)
                            .range(0.0..=f32::INFINITY)
                            .prefix(format!("{label} ")),
                    );
                }
            })
            .response
        })
        .on_hover_text(
            "What the frame clears to where nothing was drawn, in cd/m². Visible \
             only with no environment baked or the skybox off. Black is the one \
             value AMD documents on the compressed fast-clear path, though the \
             difference did not show up in a frame time on RADV — so treat it as \
             free rather than as a saving.",
        );
    }
    // Logarithmic over five orders of magnitude, because that is the range
    // real skies cover and because a downloaded `.hdr` needs a factor in the
    // thousands to reach any of it. The old control was a raw multiplier
    // capped at 4.0, which could not express the calibration a real capture
    // wants at all.
    if rows.shows("Sky") {
        slider(
            ui,
            "Sky",
            egui::Slider::new(&mut env.sky_luminance, 1.0..=50_000.0)
                .logarithmic(true)
                .suffix(" cd/m²"),
        )
        .on_hover_text(
            "How bright this sky is, whatever the file's own numbers are: clear \
             zenith ~8 000, overcast 1 000–2 000, dusk in the tens. Set the sun \
             beside it to match — a daylight capture wants tens of thousands of lux.",
        );
    }
    if rows.shows("Offset") {
        slider(
            ui,
            "Offset",
            egui::Slider::new(&mut env.exposure_offset, -4.0..=4.0).suffix(" EV"),
        )
        .on_hover_text("Stops on top of the calibration. Zero is what the sky above says it is.");
    }
    // Rotates the sampling direction, so it needs no rebake and can be
    // dragged.
    if rows.shows("Rotation") {
        slider(
            ui,
            "Rotation",
            egui::Slider::new(&mut env.yaw, -180.0..=180.0),
        );
    }
}

/// The sky's file: shown, picked in the system's dialog, and reloaded or
/// cleared from its menu. A pick or a reload asks for a bake rather than
/// baking: the bake blocks on the GPU, so the app does it at the one safe
/// point in its loop, and reports a file that fails there.
fn hdri_row(ui: &mut egui::Ui, world: &World, env: &mut EnvironmentSettings) {
    let picking = picking(ui.ctx());
    let named = !env.hdri.trim().is_empty();
    widgets::property(ui, "HDRI", |ui| {
        let shown = if named { env.hdri.as_str() } else { "None" };
        let field = ui
            .add_enabled_ui(!picking, |ui| {
                file_field(
                    ui,
                    shown,
                    "A Radiance .hdr in the assets folder: click to pick one, \
                     right-click to reload or clear it",
                )
            })
            .inner;
        if field.clicked() {
            start_dialog(ui.ctx(), crate::editor::state::assets_dir(world));
        }
        field.context_menu(|ui| {
            if ui
                .add_enabled(!picking, egui::Button::new("Choose…"))
                .clicked()
            {
                start_dialog(ui.ctx(), crate::editor::state::assets_dir(world));
                ui.close_menu();
            }
            if ui
                .add_enabled(named, egui::Button::new("Reload"))
                .on_hover_text("Bake it again from the file as it is now")
                .clicked()
            {
                env.reload_requested = true;
                ui.close_menu();
            }
            if ui.add_enabled(named, egui::Button::new("Clear")).clicked() {
                env.hdri.clear();
                ui.close_menu();
            }
        });
    });
}

/// A read-only well holding `text`, cut to fit with the whole of it on hover,
/// that answers a click.
fn file_field(ui: &mut egui::Ui, text: &str, hint: &str) -> egui::Response {
    let size = egui::vec2(ui.available_width(), ui.spacing().interact_size.y);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let stroke = if response.hovered() {
        ui.visuals().widgets.hovered.bg_stroke
    } else {
        egui::Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color)
    };
    ui.painter().rect(
        rect,
        ui.visuals().widgets.inactive.corner_radius,
        ui.visuals().extreme_bg_color,
        stroke,
        egui::StrokeKind::Inside,
    );
    let weak = ui.visuals().weak_text_color();
    let glyph = egui::Rect::from_center_size(
        egui::pos2(rect.left() + 11.0, rect.center().y),
        egui::Vec2::splat(icons::ROW - 3.0),
    );
    icons::paint(ui, icons::image(), glyph, weak);
    let left = glyph.right() + 5.0;
    let galley = egui::WidgetText::from(super::figures(text)).into_galley(
        ui,
        Some(egui::TextWrapMode::Truncate),
        (rect.right() - 4.0 - left).max(0.0),
        egui::TextStyle::Monospace,
    );
    let colour = if ui.is_enabled() {
        ui.visuals().text_color()
    } else {
        weak
    };
    let at = egui::pos2(left, rect.center().y - galley.size().y / 2.0);
    ui.painter().galley(at, galley.clone(), colour);
    if galley.elided {
        response.on_hover_text(format!("{text}\n{hint}"))
    } else {
        response.on_hover_text(hint)
    }
}

/// An HDRI dialog open on a thread of its own, so the editor goes on drawing
/// behind it, and the assets folder its pick is read against. Kept in egui's
/// memory, as the File menu keeps its scene dialog.
#[derive(Clone)]
struct HdriDialog {
    assets: PathBuf,
    /// Empty until the dialog closes; then the file picked, or nothing.
    picked: Arc<Mutex<Option<Option<PathBuf>>>>,
}

fn dialog_id() -> egui::Id {
    egui::Id::new("hdri_file_dialog")
}

/// Whether an HDRI dialog is open. Public to the panel tests.
pub(super) fn picking(ctx: &egui::Context) -> bool {
    ctx.data(|data| data.get_temp::<HdriDialog>(dialog_id()).is_some())
}

fn start_dialog(ctx: &egui::Context, assets: PathBuf) {
    let picked = Arc::new(Mutex::new(None));
    let (into, repaint, folder) = (picked.clone(), ctx.clone(), assets.clone());
    let dialog = std::thread::Builder::new()
        .name("orrin-hdri-dialog".to_owned())
        .spawn(move || {
            let file = pick_hdri(folder);
            if let Ok(mut slot) = into.lock() {
                *slot = Some(file);
            }
            repaint.request_repaint();
        });
    match dialog {
        Ok(_) => ctx.data_mut(|data| data.insert_temp(dialog_id(), HdriDialog { assets, picked })),
        Err(error) => tracing::error!("could not open a file dialog: {error}"),
    }
}

/// Take a closed dialog's pick into the sky, as a path under the assets
/// folder, and ask for the bake; a file anywhere else is refused by name.
fn poll_dialog(ctx: &egui::Context, world: &World) {
    let Some(dialog) = ctx.data(|data| data.get_temp::<HdriDialog>(dialog_id())) else {
        return;
    };
    let Some(file) = dialog.picked.lock().ok().and_then(|mut slot| slot.take()) else {
        return;
    };
    ctx.data_mut(|data| data.remove::<HdriDialog>(dialog_id()));
    let Some(file) = file else {
        return;
    };
    match under(&dialog.assets, &file) {
        Some(relative) => {
            let mut env = world.resource_mut::<EnvironmentSettings>();
            env.hdri = relative;
            env.reload_requested = true;
        }
        None => super::assets::log_error(
            world,
            format!(
                "`{}` is outside the assets folder `{}`; a scene can only name a sky \
                 inside it, so copy the file in and pick it there",
                file.display(),
                dialog.assets.display()
            ),
        ),
    }
}

/// `file` relative to `assets`, `/`-separated as a scene file spells every
/// path, or `None` when it is not inside it. Compared as given and then
/// canonicalised, since a dialog and the database need not agree on how a
/// folder is spelled.
fn under(assets: &Path, file: &Path) -> Option<String> {
    let inside = |assets: &Path, file: &Path| file.strip_prefix(assets).ok().map(Path::to_owned);
    let relative = inside(assets, file).or_else(|| {
        inside(
            &std::fs::canonicalize(assets).ok()?,
            &std::fs::canonicalize(file).ok()?,
        )
    })?;
    let mut parts = Vec::new();
    for part in relative.components() {
        match part {
            Component::Normal(part) => parts.push(part.to_string_lossy()),
            _ => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// The `.hdr` picked in the operating system's dialog; `None` if it was
/// cancelled. Blocks, so it runs on a thread of its own, through the async
/// dialog for the reason the asset browser's is (`assets::pick_files`).
#[cfg(not(test))]
fn pick_hdri(folder: PathBuf) -> Option<PathBuf> {
    let dialog = rfd::AsyncFileDialog::new()
        .set_title("Choose the sky")
        .add_filter("Radiance HDR", &["hdr"])
        .set_directory(folder);
    pollster::block_on(dialog.pick_file()).map(|file| file.path().to_owned())
}

#[cfg(test)]
fn pick_hdri(_: PathBuf) -> Option<PathBuf> {
    HDRI_PICKED.lock().unwrap().take().flatten()
}

fn fog(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut fog = world.resource_mut::<FogSettings>();
    if rows.shows("Albedo") {
        color_row(ui, "Albedo", &mut fog.albedo);
    }
    if rows.shows("Density") {
        slider(
            ui,
            "Density",
            egui::Slider::new(&mut fog.density, 0.0..=0.1).logarithmic(true),
        )
        .on_hover_text(
            "Extinction per metre at the reference height. 0.005 is a light haze, \
             0.05 is thick enough to lose a building at fifty metres.",
        );
    }
    if rows.shows("Falloff") {
        slider(
            ui,
            "Falloff",
            egui::Slider::new(&mut fog.height_falloff, 0.0..=1.0),
        );
    }
    if rows.shows("Height") {
        slider(
            ui,
            "Height",
            egui::Slider::new(&mut fog.height, -20.0..=20.0),
        );
    }
    if rows.shows("Volumetric") {
        toggle(ui, "Volumetric", &mut fog.volumetric).on_hover_text(
            "March the first `Distance` metres as a froxel volume, so the shadow \
                 maps reach the air and the sun casts shafts. Off, the same medium is \
                 integrated analytically along each view ray — smooth, and unshadowed.",
        );
    }
    ui.add_enabled_ui(fog.volumetric, |ui| {
        if rows.shows("Distance") {
            slider(
                ui,
                "Distance",
                egui::Slider::new(&mut fog.distance, 8.0..=256.0).suffix(" m"),
            )
            .on_hover_text(
                "How far the volume reaches. The froxel count is fixed, so this trades \
                 reach against resolution and not against cost.",
            );
        }
        if rows.shows("Anisotropy") {
            slider(
                ui,
                "Anisotropy",
                egui::Slider::new(&mut fog.anisotropy, -0.9..=0.9),
            )
            .on_hover_text(
                "Henyey-Greenstein g. Positive scatters forward, which is what makes \
                     looking toward the sun through haze so much brighter than looking away.",
            );
        }
        if rows.shows("Feedback") {
            slider(
                ui,
                "Feedback",
                egui::Slider::new(&mut fog.feedback, 0.0..=0.98),
            )
            .on_hover_text(
                "How much reprojected history each froxel keeps. Low values make the \
                     depth jitter visible as crawling slices.",
            );
        }
    });
}

/// Beside the shutter rather than the other post-process effects, because
/// its input is the camera's: the lens takes its focal length from the view's
/// field of view (the editor camera's FOV, or the game camera's while it
/// plays), so widening the shot deepens the image exactly as it would on a
/// real one.
fn lens(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let fov_y = world.resource::<Camera>().fov_y;
    let mut dof = world.resource_mut::<DofSettings>();
    // Frame structure, like bloom and TAA: it registers or drops all four
    // passes, so it recompiles the graph.
    if rows.shows("Depth of field") {
        toggle(ui, "Depth of field", &mut dof.enabled);
    }
    if rows.shows("Focus") {
        slider(
            ui,
            "Focus",
            egui::Slider::new(&mut dof.focus_distance, 0.1..=100.0)
                .logarithmic(true)
                .suffix(" m"),
        );
    }
    if rows.shows("Aperture") {
        slider(
            ui,
            "Aperture",
            egui::Slider::new(&mut dof.f_number, 1.0..=22.0)
                .logarithmic(true)
                .prefix("f/"),
        )
        .on_hover_text("Larger is a smaller opening, and a deeper image");
    }
    if rows.shows("Sensor") {
        slider(
            ui,
            "Sensor",
            egui::Slider::new(&mut dof.sensor_height, 5.0..=36.0).suffix(" mm"),
        )
        .on_hover_text("24mm is full frame, 14.2mm is Super 35");
        // Derived, never set — showing it is what makes the coupling to the FOV
        // slider legible rather than something to be discovered.
        ui.label(format!("≈{:.0}mm lens", dof.focal_length(fov_y) * 1e3))
            .on_hover_text("Derived from the field of view and the sensor height");
    }
}

fn shutter(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut motion_blur = world.resource_mut::<MotionBlurSettings>();
    if rows.shows("Motion blur") {
        toggle(ui, "Motion blur", &mut motion_blur.enabled);
    }
    if rows.shows("Angle") {
        slider(
            ui,
            "Angle",
            egui::Slider::new(&mut motion_blur.shutter_angle, 0.0..=360.0).suffix("°"),
        )
        .on_hover_text("180° is the film standard: half a frame of travel");
    }
}

fn editor_camera(ui: &mut egui::Ui, world: &World, rows: Rows) {
    let mut cam = world.resource_mut::<Camera>();
    if rows.shows("Position") {
        vec3_row(ui, "Position", &mut cam.position, 0.1);
    }
    if rows.shows("Target") {
        vec3_row(ui, "Target", &mut cam.target, 0.1);
    }
    if rows.shows("FOV") {
        let mut fov = cam.fov_y.to_degrees();
        if ui
            .add(egui::Slider::new(&mut fov, 20.0..=110.0).text("FOV"))
            .changed()
        {
            cam.fov_y = fov.to_radians();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::editor::state::EditorState;
    use crate::scene::render_settings::{self, RenderSettings};
    use crate::scene::{FogSettings, Name};
    use orrin_registry::Registry;

    fn session() -> (World, Registry, EditorState) {
        let mut world = World::new();
        crate::App::install_default_resources(&mut world);
        let mut registry = Registry::new();
        crate::scene::register_components(&mut registry);
        (world, registry, EditorState::default())
    }

    fn with_settings(world: &mut World, settings: RenderSettings) -> orrin_ecs::Entity {
        let entity = world.spawn();
        world.insert(entity, Name::new("Render Settings"));
        world.insert(entity, settings);
        render_settings::sync(world);
        entity
    }

    fn density(world: &World, entity: orrin_ecs::Entity) -> f32 {
        world.get::<RenderSettings>(entity).unwrap().fog.density
    }

    /// A scene with settings owns them: a slider moved here is a change to
    /// the scene, which is saved with it, marks it dirty and can be undone.
    #[test]
    fn a_change_to_a_scene_with_settings_is_an_undoable_edit() {
        let (mut world, registry, mut state) = session();
        let entity = with_settings(&mut world, RenderSettings::default());

        edit_settings(&mut world, &mut state, |world| {
            world.resource_mut::<FogSettings>().density = 0.4;
        });
        state.apply(&mut world, &registry);
        assert_eq!(density(&world, entity), 0.4);
        assert!(state.scene_dirty());

        state.request_undo();
        state.apply(&mut world, &registry);
        assert_eq!(density(&world, entity), 0.0);
    }

    /// Without settings in the scene there is nothing to write to: the change
    /// is this session's, as every change here was before scenes saved them.
    #[test]
    fn a_change_to_a_scene_without_settings_is_the_sessions_alone() {
        let (mut world, registry, mut state) = session();
        edit_settings(&mut world, &mut state, |world| {
            world.resource_mut::<FogSettings>().density = 0.4;
        });
        state.apply(&mut world, &registry);
        assert_eq!(world.resource::<FogSettings>().density, 0.4);
        assert!(!state.scene_dirty());
    }

    /// The editor draws before the frame syncs, so on the frame after a load
    /// the resources still hold the last scene. An edit then must not write
    /// that scene's values into this one.
    #[test]
    fn an_edit_starts_from_the_scene_not_from_the_last_frame() {
        let (mut world, registry, mut state) = session();
        let mut settings = RenderSettings::default();
        settings.fog.density = 0.05;
        let entity = world.spawn();
        world.insert(entity, settings);
        world.resource_mut::<FogSettings>().density = 0.3;

        edit_settings(&mut world, &mut state, |world| {
            world.resource_mut::<crate::scene::BloomSettings>().strength = 0.5;
        });
        state.apply(&mut world, &registry);
        assert_eq!(density(&world, entity), 0.05);
        let saved = world.get::<RenderSettings>(entity).unwrap().bloom.strength;
        assert_eq!(saved, 0.5);
    }

    /// Giving a scene settings keeps what it is being drawn with, so the
    /// tuning done before the click is what gets saved.
    #[test]
    fn adding_settings_keeps_what_the_session_was_drawn_with() {
        let (mut world, registry, mut state) = session();
        world.resource_mut::<FogSettings>().density = 0.3;
        state.request_add_render_settings();
        state.apply(&mut world, &registry);

        let entity = render_settings::source(&world).expect("the scene has settings");
        assert_eq!(density(&world, entity), 0.3);
        assert_eq!(
            world.get::<Name>(entity).map(|name| name.0.clone()),
            Some("Render Settings".to_owned())
        );

        state.request_undo();
        state.apply(&mut world, &registry);
        assert!(render_settings::source(&world).is_none());
    }
}
