use orrin_ecs::World;

use crate::gfx::shadows::MAX_CASCADES;

use super::{color_row, vec3_row};
use crate::editor::widgets;
use crate::scene::{
    AmbientLight, BloomSettings, Camera, ContactShadowSettings, DecalSettings, DofSettings,
    EnvironmentSettings, FogSettings, HdrSettings, MotionBlurSettings, RefractionSettings,
    ShadowSettings, SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings,
    TransparencySettings,
};

type Column = fn(&mut egui::Ui, &World);

const COLUMNS: [Column; 4] = [
    screen_space_column,
    shadow_column,
    lighting_column,
    camera_column,
];

/// A slider plus the label that sits after it. Narrower than this and a column's
/// contents run into its neighbour instead of wrapping — `Ui::columns` divides
/// the width but does nothing to make the contents fit it.
///
/// This was once a full-window strip along the bottom, where four cramped
/// columns were the intended look and folding was only a guard against a
/// negative width. Docked, the same panel is a 300px tab, and four columns there
/// is four overlapping ones. The number is now what it says it is.
const MIN_COLUMN_WIDTH: f32 = 190.0;

/// How many columns `available` can carry. Never zero: `Ui::columns` divides by
/// the count, and it asserts on the negative width a shortfall would produce.
fn column_count(available: f32) -> usize {
    ((available / MIN_COLUMN_WIDTH) as usize).clamp(1, COLUMNS.len())
}

/// A group's title: the inspector's section bar without the disclosure. These
/// groups are short and a settings panel is read top to bottom, so folding
/// would only add a click.
fn heading(ui: &mut egui::Ui, title: &str) -> egui::Response {
    ui.add_space(2.0);
    let size = egui::vec2(ui.available_width(), 24.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
    let visuals = ui.visuals();
    ui.painter().rect_filled(
        rect,
        visuals.widgets.inactive.corner_radius,
        visuals.window_fill.to_opaque(),
    );
    ui.painter().text(
        rect.left_center() + egui::vec2(8.0, 0.0),
        egui::Align2::LEFT_CENTER,
        title,
        egui::TextStyle::Button.resolve(ui.style()),
        visuals.strong_text_color(),
    );
    ui.add_space(2.0);
    response
}

fn slider(ui: &mut egui::Ui, label: &str, slider: egui::Slider<'_>) -> egui::Response {
    widgets::property(ui, label, |ui| ui.add(slider))
}

fn toggle(ui: &mut egui::Ui, label: &str, value: &mut bool) -> egui::Response {
    widgets::property(ui, label, |ui| ui.checkbox(value, ""))
}

pub fn body(ui: &mut egui::Ui, world: &mut World) {
    // A docked tab is as wide as its split, which can be nothing at all.
    let available = ui.available_width();
    if available > 0.0 {
        let columns = column_count(available);
        ui.columns(columns, |cols| {
            let mut previous = usize::MAX;
            for (index, column) in COLUMNS.iter().enumerate() {
                let target = index * columns / COLUMNS.len();
                if target == previous {
                    cols[target].add_space(10.0);
                    cols[target].separator();
                }
                column(&mut cols[target], world);
                previous = target;
            }
        });
    }
    ui.add_space(4.0);
}

fn screen_space_column(ui: &mut egui::Ui, world: &World) {
    heading(ui, "SSAO");
    {
        let mut s = world.resource_mut::<SsaoSettings>();
        toggle(ui, "Enabled", &mut s.enabled);
        slider(ui, "Radius", egui::Slider::new(&mut s.radius, 0.0..=4.0));
        slider(ui, "Bias", egui::Slider::new(&mut s.bias, 0.0..=0.1));
        slider(ui, "Power", egui::Slider::new(&mut s.power, 0.1..=4.0));
        toggle(ui, "Half resolution", &mut s.half_resolution).on_hover_text(
            "Resolve the occlusion at half the frame's extent and let the \
                 bilateral blur upsample it: a quarter of the pixels through the \
                 frame's most expensive loop, for a term that is low-frequency \
                 and blurred either way",
        );
    }
    ui.add_space(6.0);
    heading(ui, "Temporal AA")
        .on_hover_text("Jitter the camera each frame and accumulate the results");
    {
        let mut taa = world.resource_mut::<TaaSettings>();
        // Frame structure, like SSAO and bloom: it registers the resolve node
        // and it is what keeps the geometry prepass in a frame with SSAO off.
        toggle(ui, "Enabled", &mut taa.enabled);
        slider(
            ui,
            "Feedback",
            egui::Slider::new(&mut taa.feedback, 0.5..=0.98),
        )
        .on_hover_text("How much of the reprojected history each frame keeps");
        slider(
            ui,
            "Jitter",
            egui::Slider::new(&mut taa.jitter_scale, 0.0..=1.5),
        )
        .on_hover_text("Fraction of a pixel the camera samples across");
        toggle(ui, "4x MSAA", &mut taa.msaa).on_hover_text(
            "Rasterise the forward pass at four samples instead of one. The \
             alternative to the temporal resolve above, not a companion to it: \
             it costs roughly a quarter of the frame, and at one sample the \
             forward pass borrows the prepass depth and shades each pixel once",
        );
    }
    ui.add_space(6.0);
    heading(ui, "Reflections").on_hover_text(
        "Reflect the frame off itself, where the frame contains what a surface faces",
    );
    {
        let mut ssr = world.resource_mut::<SsrSettings>();
        // Frame structure like the rest: it registers the pyramid, the trace and
        // the composite, and it is another reason the prepass exists.
        toggle(ui, "Screen-space reflections", &mut ssr.enabled);
        slider(
            ui,
            "Intensity",
            egui::Slider::new(&mut ssr.intensity, 0.0..=1.0),
        )
        .on_hover_text("1.0 is the physical answer; lower hides the screen-space failures");
        slider(
            ui,
            "Max roughness",
            egui::Slider::new(&mut ssr.max_roughness, 0.05..=1.0),
        )
        .on_hover_text("Rougher than this is left to the environment cubemap");
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
        slider(
            ui,
            "Range",
            egui::Slider::new(&mut ssr.max_distance, 1.0..=200.0)
                .logarithmic(true)
                .suffix(" m"),
        );
        slider(ui, "Steps", egui::Slider::new(&mut ssr.max_steps, 8..=128))
            .on_hover_text("A step crosses a whole cell of the depth pyramid, not a texel");
    }
    ui.add_space(6.0);
    heading(ui, "Subsurface scattering")
        .on_hover_text("Light that enters a surface, scatters inside it and leaves somewhere else");
    {
        let mut subsurface = world.resource_mut::<SubsurfaceSettings>();
        // Frame structure, and more of it than the rest: this switch decides which
        // of the two forward render passes the frame opens, so it registers three
        // passes *and* a second colour attachment. It is also one more reason the
        // prepass exists — the blur weights its taps by depth.
        toggle(ui, "Screen-space diffusion", &mut subsurface.enabled).on_hover_text(
            "Off keeps scattering: the analytic wrap widens to stand in for these passes",
        );
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
        // Three sliders rather than a colour picker: these are distances, not a
        // colour, and the widest is pinned at one by construction — only the
        // ratios reach the shader.
        slider(
            ui,
            "Red reach",
            egui::Slider::new(&mut subsurface.profile.x, 0.0..=1.0),
        );
        slider(
            ui,
            "Green reach",
            egui::Slider::new(&mut subsurface.profile.y, 0.0..=1.0),
        );
        slider(
            ui,
            "Blue reach",
            egui::Slider::new(&mut subsurface.profile.z, 0.0..=1.0),
        )
        .on_hover_text("How far each channel travels, relative to the furthest");
    }
    ui.add_space(6.0);
    heading(ui, "Transparency")
        .on_hover_text("Weighted-blended order-independent transparency (McGuire & Bavoil)");
    {
        let mut transparency = world.resource_mut::<TransparencySettings>();
        // Frame structure like the rest: it registers the accumulation and
        // composite nodes, and it is one more thing that keeps the prepass
        // alive — the accumulation depth-tests against what that pass wrote.
        toggle(ui, "Enabled", &mut transparency.enabled)
            .on_hover_text("Off draws nothing at all where a blended material is used");
    }
    ui.add_space(6.0);
    heading(ui, "Refraction").on_hover_text(
        "Screen-space refraction: a transmissive material samples the frame behind it",
    );
    {
        let mut refraction = world.resource_mut::<RefractionSettings>();
        // Beside transparency rather than under it: the two queues answer to
        // opposite rules about ordering, and each is its own A/B when a
        // non-opaque surface looks wrong.
        toggle(ui, "Enabled", &mut refraction.enabled)
            .on_hover_text("Off draws nothing at all where a transmissive material is used");
    }
    ui.add_space(6.0);
    heading(ui, "Decals")
        .on_hover_text("Boxes that stamp maps onto the surfaces inside them, before shading");
    {
        let mut decals = world.resource_mut::<DecalSettings>();
        // Unlike everything above it in this column, this is *not* frame
        // structure: decals add no pass and no target, so the graph is the same
        // either way and nothing is reallocated when it is clicked. It is here
        // as the A/B — decals land under the lighting, so "is this a decal or a
        // light?" is otherwise a hard question to ask of a still frame.
        toggle(ui, "Enabled", &mut decals.enabled)
            .on_hover_text("Off leaves every surface as its own material describes it");
    }
}

fn shadow_column(ui: &mut egui::Ui, world: &World) {
    heading(ui, "Shadows");
    let mut s = world.resource_mut::<ShadowSettings>();
    toggle(ui, "Enabled", &mut s.enabled);
    // Cascade count and resolution are frame *structure*: changing either
    // recompiles the graph and reallocates the maps, so they are steppers
    // rather than sliders — a drag would do that once per frame.
    widgets::property(ui, "Cascades", |ui| {
        ui.add(egui::DragValue::new(&mut s.cascade_count).range(1..=MAX_CASCADES));
    });
    widgets::property(ui, "Resolution", |ui| {
        egui::ComboBox::from_id_salt("shadow_resolution")
            .selected_text(format!("{}", s.resolution))
            .show_ui(ui, |ui| {
                for size in [512u32, 1024, 2048, 4096] {
                    ui.selectable_value(&mut s.resolution, size, format!("{size}"));
                }
            });
    });
    slider(
        ui,
        "Distance",
        egui::Slider::new(&mut s.max_distance, 10.0..=500.0).logarithmic(true),
    );
    slider(
        ui,
        "Split blend",
        egui::Slider::new(&mut s.lambda, 0.0..=1.0),
    );
    slider(
        ui,
        "Pullback",
        egui::Slider::new(&mut s.pullback, 0.0..=200.0),
    );
    slider(
        ui,
        "Bias",
        egui::Slider::new(&mut s.constant_bias, 0.0..=8.0),
    );
    slider(
        ui,
        "Slope bias",
        egui::Slider::new(&mut s.slope_bias, 0.0..=8.0),
    );
    slider(
        ui,
        "Strength",
        egui::Slider::new(&mut s.strength, 0.0..=1.0),
    );
    toggle(ui, "Tint cascades", &mut s.debug_cascades);

    ui.add_space(6.0);
    heading(ui, "Point & spot shadows")
        .on_hover_text("One atlas, a tile per cube face, drawn in a single pass");
    // Whether a *given* light casts is on the light, in the inspector; these are
    // the shape of the atlas it competes for.
    toggle(ui, "Enabled", &mut s.punctual_enabled);
    widgets::property(ui, "Atlas", |ui| {
        // 4096 is the ceiling on purpose, twice over. Vulkan guarantees
        // `maxImageDimension2D` of only 4096, so offering 8192 is offering a
        // startup panic on a conformant device that Metal happens to let this
        // one get away with. And it would buy nothing if it worked: the budget
        // is `MAX_SHADOW_LIGHTS` casters, so the most faces anything can ask for
        // is 48, and 4096 at 512 already holds 64.
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
    {
        let tiles = s.atlas_config().map_or(0, |config| config.capacity());
        ui.weak(format!("{tiles} tiles — {} point lights", tiles / 6));
    }
    slider(
        ui,
        "Near",
        egui::Slider::new(&mut s.punctual_near, 0.005..=1.0)
            .logarithmic(true)
            .suffix(" m"),
    )
    .on_hover_text("Depth precision near the light, which is where its contacts are");
    slider(
        ui,
        "Bias",
        egui::Slider::new(&mut s.punctual_constant_bias, 0.0..=8.0),
    );
    slider(
        ui,
        "Slope bias",
        egui::Slider::new(&mut s.punctual_slope_bias, 0.0..=8.0),
    );
    drop(s);

    ui.add_space(6.0);
    heading(ui, "Contact shadows")
        .on_hover_text("March the depth buffer for the band a cascade texel is too coarse to hold");
    {
        let mut s = world.resource_mut::<ContactShadowSettings>();
        // Frame structure like the rest: it registers the march and it is one
        // more thing that keeps the geometry prepass in the frame.
        toggle(ui, "Enabled", &mut s.enabled);
        slider(
            ui,
            "Length",
            egui::Slider::new(&mut s.ray_length, 0.01..=2.0)
                .logarithmic(true)
                .suffix(" m"),
        )
        .on_hover_text("Longer than a near cascade's texel spends steps on shadows the maps have");
        slider(ui, "Steps", egui::Slider::new(&mut s.steps, 4..=64))
            .on_hover_text("The shortest occluder the march can find is one step of the ray");
        slider(
            ui,
            "Bias",
            egui::Slider::new(&mut s.bias, 0.0..=0.2)
                .logarithmic(true)
                .suffix(" m"),
        )
        .on_hover_text("How far in front of the ray a surface must be to be something else");
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
        slider(
            ui,
            "Strength",
            egui::Slider::new(&mut s.intensity, 0.0..=1.0),
        );
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

fn lighting_column(ui: &mut egui::Ui, world: &World) {
    heading(ui, "Tonemap");
    {
        let mut hdr = world.resource_mut::<HdrSettings>();
        // Metering on or off is frame *structure*: it registers or drops the two
        // compute passes, so it recompiles the graph. A checkbox, not something
        // draggable.
        toggle(ui, "Auto exposure", &mut hdr.auto_exposure)
            .on_hover_text("Meter the frame's luminance and adapt to it");
        slider(
            ui,
            "Compensation",
            egui::Slider::new(&mut hdr.exposure_compensation, -4.0..=4.0).suffix(" EV"),
        );
        if hdr.auto_exposure {
            slider(
                ui,
                "Brighten",
                egui::Slider::new(&mut hdr.adaptation_brighten, 0.0..=4.0).suffix(" s"),
            );
            slider(
                ui,
                "Darken",
                egui::Slider::new(&mut hdr.adaptation_darken, 0.0..=4.0).suffix(" s"),
            );
            slider(
                ui,
                "Meter floor",
                egui::Slider::new(&mut hdr.min_log_luminance, -16.0..=0.0),
            );
            slider(
                ui,
                "Meter ceiling",
                egui::Slider::new(&mut hdr.max_log_luminance, 0.0..=20.0),
            );
        } else {
            slider(
                ui,
                "EV100",
                egui::Slider::new(&mut hdr.manual_ev100, -6.0..=16.0),
            );
        }
    }
    ui.add_space(6.0);
    heading(ui, "Bloom").on_hover_text(
        "Filtered on the exposed image, so the strength holds across lighting changes",
    );
    {
        let mut bloom = world.resource_mut::<BloomSettings>();
        // Like metering, on/off is frame structure: it registers or drops the
        // whole chain, so it recompiles the graph.
        toggle(ui, "Enabled", &mut bloom.enabled);
        slider(
            ui,
            "Strength",
            egui::Slider::new(&mut bloom.strength, 0.0..=0.3),
        );
        slider(
            ui,
            "Radius",
            egui::Slider::new(&mut bloom.radius, 0.5..=3.0),
        );
        slider(
            ui,
            "Scatter",
            egui::Slider::new(&mut bloom.scatter, 0.0..=0.95),
        )
        .on_hover_text("Weight toward the blurrier levels — how far the glow reaches");
    }
    ui.add_space(6.0);
    heading(ui, "Ambient").on_hover_text(
        "Fallback only: an environment's irradiance replaces this when one is loaded",
    );
    {
        let mut ambient = world.resource_mut::<AmbientLight>();
        color_row(ui, "Color", &mut ambient.color);
        slider(
            ui,
            "Luminance",
            egui::Slider::new(&mut ambient.nits, 0.0..=10_000.0)
                .logarithmic(true)
                .suffix(" cd/m²"),
        )
        .on_hover_text("Clear zenith a few thousand, overcast under a thousand, dusk tens");
    }
    ui.add_space(6.0);
    heading(ui, "Sky");
    {
        let mut env = world.resource_mut::<EnvironmentSettings>();
        widgets::property(ui, "HDRI", |ui| {
            // A button rather than applying as you type: the bake blocks on the
            // GPU for every keystroke otherwise, and every prefix of a filename
            // is an error worth reporting exactly once.
            if ui.button("Load").clicked() {
                env.reload_requested = true;
            }
        });
        ui.add(
            egui::TextEdit::singleline(&mut env.hdri)
                .hint_text("relative to assets/")
                .desired_width(f32::INFINITY),
        );
        toggle(ui, "Show skybox", &mut env.show_skybox);
        // Only reachable with the skybox off or nothing baked, so it sits right
        // under the toggle that makes it visible. Drag values rather than a
        // colour picker because the number is a luminance in cd/m² and a picker
        // would quietly clamp it to the 0–1 the rest of the frame is not in.
        // Bound out of the resource guard once: three `&mut env.background.x`
        // would each go through `DerefMut` and count as three whole-resource
        // borrows, where three fields of one `Vec3` are disjoint.
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
        // Logarithmic over five orders of magnitude, because that is the range
        // real skies cover and because a downloaded `.hdr` needs a factor in the
        // thousands to reach any of it. The old control was a raw multiplier
        // capped at 4.0, which could not express the calibration a real capture
        // wants at all.
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
        slider(
            ui,
            "Offset",
            egui::Slider::new(&mut env.exposure_offset, -4.0..=4.0).suffix(" EV"),
        )
        .on_hover_text("Stops on top of the calibration. Zero is what the sky above says it is.");
        // Rotates the sampling direction, so it needs no rebake and can be
        // dragged.
        slider(
            ui,
            "Rotation",
            egui::Slider::new(&mut env.yaw, -180.0..=180.0),
        );
    }
    ui.add_space(6.0);
    heading(ui, "Fog");
    let mut fog = world.resource_mut::<FogSettings>();
    color_row(ui, "Albedo", &mut fog.albedo);
    slider(
        ui,
        "Density",
        egui::Slider::new(&mut fog.density, 0.0..=0.1).logarithmic(true),
    )
    .on_hover_text(
        "Extinction per metre at the reference height. 0.005 is a light haze, \
         0.05 is thick enough to lose a building at fifty metres.",
    );
    slider(
        ui,
        "Falloff",
        egui::Slider::new(&mut fog.height_falloff, 0.0..=1.0),
    );
    slider(
        ui,
        "Height",
        egui::Slider::new(&mut fog.height, -20.0..=20.0),
    );
    toggle(ui, "Volumetric", &mut fog.volumetric).on_hover_text(
        "March the first `Distance` metres as a froxel volume, so the shadow \
             maps reach the air and the sun casts shafts. Off, the same medium is \
             integrated analytically along each view ray — smooth, and unshadowed.",
    );
    ui.add_enabled_ui(fog.volumetric, |ui| {
        slider(
            ui,
            "Distance",
            egui::Slider::new(&mut fog.distance, 8.0..=256.0).suffix(" m"),
        )
        .on_hover_text(
            "How far the volume reaches. The froxel count is fixed, so this trades \
             reach against resolution and not against cost.",
        );
        slider(
            ui,
            "Anisotropy",
            egui::Slider::new(&mut fog.anisotropy, -0.9..=0.9),
        )
        .on_hover_text(
            "Henyey-Greenstein g. Positive scatters forward, which is what makes \
                 looking toward the sun through haze so much brighter than looking away.",
        );
        slider(
            ui,
            "Feedback",
            egui::Slider::new(&mut fog.feedback, 0.0..=0.98),
        )
        .on_hover_text(
            "How much reprojected history each froxel keeps. Low values make the \
                 depth jitter visible as crawling slices.",
        );
    });
}

fn camera_column(ui: &mut egui::Ui, world: &World) {
    heading(ui, "Camera");
    let fov_y = {
        let mut cam = world.resource_mut::<Camera>();
        vec3_row(ui, "Position", &mut cam.position, 0.1);
        vec3_row(ui, "Target", &mut cam.target, 0.1);

        let mut fov = cam.fov_y.to_degrees();
        if ui
            .add(egui::Slider::new(&mut fov, 20.0..=110.0).text("FOV"))
            .changed()
        {
            cam.fov_y = fov.to_radians();
        }
        cam.fov_y
    };

    // Under the camera rather than beside the other post-process effects,
    // because that is where their inputs are: the lens takes its focal length
    // from the field of view above, so widening the shot deepens the image
    // exactly as it would on a real one.
    ui.add_space(6.0);
    heading(ui, "Lens").on_hover_text("Defocus everything the lens is not focused on");
    {
        let mut dof = world.resource_mut::<DofSettings>();
        // Frame structure, like bloom and TAA: it registers or drops all four
        // passes, so it recompiles the graph.
        toggle(ui, "Depth of field", &mut dof.enabled);
        slider(
            ui,
            "Focus",
            egui::Slider::new(&mut dof.focus_distance, 0.1..=100.0)
                .logarithmic(true)
                .suffix(" m"),
        );
        slider(
            ui,
            "Aperture",
            egui::Slider::new(&mut dof.f_number, 1.0..=22.0)
                .logarithmic(true)
                .prefix("f/"),
        )
        .on_hover_text("Larger is a smaller opening, and a deeper image");
        slider(
            ui,
            "Sensor",
            egui::Slider::new(&mut dof.sensor_height, 5.0..=36.0).suffix(" mm"),
        )
        .on_hover_text("24mm is full frame, 14.2mm is Super 35");
        // Derived, never set — showing it is what makes the coupling to the FOV
        // slider above legible rather than something to be discovered.
        ui.label(format!("≈{:.0}mm lens", dof.focal_length(fov_y) * 1e3))
            .on_hover_text("Derived from the field of view and the sensor height");
    }

    ui.add_space(6.0);
    heading(ui, "Shutter")
        .on_hover_text("Reconstruct the exposure from the frame's motion vectors");
    {
        let mut motion_blur = world.resource_mut::<MotionBlurSettings>();
        toggle(ui, "Motion blur", &mut motion_blur.enabled);
        slider(
            ui,
            "Angle",
            egui::Slider::new(&mut motion_blur.shutter_angle, 0.0..=360.0).suffix("°"),
        )
        .on_hover_text("180° is the film standard: half a frame of travel");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The width this panel is docked at. One column, because four at 75px each
    /// overlap into an unreadable mess — which is exactly what shipped when this
    /// threshold was still tuned for a full-window strip.
    #[test]
    fn the_docked_width_folds_to_one_column() {
        assert_eq!(column_count(300.0), 1);
    }

    /// Widen it — float it, or drag the split out — and the columns come back.
    #[test]
    fn a_wide_tool_spreads_back_out() {
        assert_eq!(column_count(400.0), 2);
        assert_eq!(column_count(600.0), 3);
        assert_eq!(column_count(800.0), 4);
        assert_eq!(column_count(2000.0), 4);
    }

    #[test]
    fn a_sliver_still_yields_a_usable_count() {
        assert_eq!(column_count(20.0), 1);
        assert_eq!(column_count(0.0), 1);
    }
}
