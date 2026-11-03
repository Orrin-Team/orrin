//! The hub's one window: a table of projects, the dialog that makes a new
//! one, and the activity panel that shows whatever the CLI is doing for them.
//!
//! Every action is an `orrin` command run as a [`Task`]. One task runs at a
//! time — a second export while the first is compiling would fight over the
//! same cargo lock — so the controls that would start another are disabled
//! with the reason in their tooltip rather than queued.
//!
//! A row opens its project when clicked anywhere but on one of its controls.
//! That works only because the row's own click is sensed *before* the controls
//! are laid out on it: egui gives a click to the last thing registered under
//! the pointer, so a row sensed afterwards would take every click meant for
//! its star, its platform and its menu.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use egui::{Align, Color32, CornerRadius, Layout, Rect, RichText, Sense, Stroke, vec2};
use orrin_project::export::Platform;

use crate::recents::{self, Details, Recent, Recents, Sort};
use crate::task::{Kind, State, Task};
use crate::theme::{self, glyph, icon};

const DOCS: &str = "https://orrinengine.com/docs/getting-started/";
const ROADMAP: &str = "https://orrinengine.com/roadmap/";
const GITHUB: &str = "https://github.com/Orrin-Team/orrin";
const DISCORD: &str = "https://discord.gg/sY8YBGFmRy";

/// Every control across the top of the projects panel.
const CONTROL_HEIGHT: f32 = 32.0;

/// How stale a row's disk-read details may get before they are read again.
const REFRESH: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Projects,
    Learn,
}

/// The new-project dialog, while it is open.
struct Draft {
    name: String,
    location: PathBuf,
    open_after: bool,
    focused: bool,
}

/// A line at the top of the page: what just happened, or what went wrong.
struct Notice {
    text: String,
    error: bool,
}

pub struct Hub {
    page: Page,
    cli: Result<PathBuf, String>,
    recents: Recents,
    recents_file: Option<PathBuf>,
    /// Every remembered project with what its row reads off the disk.
    rows: Vec<(Recent, Details)>,
    rows_read: Option<Instant>,
    search: String,
    sort: Sort,
    ascending: bool,
    platform_filter: Option<Platform>,
    draft: Option<Draft>,
    task: Option<Task>,
    show_output: bool,
    notice: Option<Notice>,
}

impl Hub {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::install(&cc.egui_ctx);
        let recents_file = recents::file();
        let recents = recents_file
            .as_deref()
            .map(Recents::load)
            .unwrap_or_default();
        Self::with(recents, recents_file, find_cli())
    }

    fn with(recents: Recents, recents_file: Option<PathBuf>, cli: Result<PathBuf, String>) -> Self {
        Self {
            page: Page::Projects,
            cli,
            recents,
            recents_file,
            rows: Vec::new(),
            rows_read: None,
            search: String::new(),
            sort: Sort::Modified,
            ascending: false,
            platform_filter: None,
            draft: None,
            task: None,
            show_output: false,
            notice: None,
        }
    }

    /// Write the list, and read every row again on the next frame.
    fn save_recents(&mut self) {
        self.rows_read = None;
        if let Some(file) = &self.recents_file
            && let Err(error) = self.recents.save(file)
        {
            self.notice = Some(Notice {
                text: format!("Could not remember your projects: {error}"),
                error: true,
            });
        }
    }

    fn refresh_rows(&mut self) {
        if self.rows_read.is_some_and(|at| at.elapsed() < REFRESH) {
            return;
        }
        self.rows = self
            .recents
            .projects
            .iter()
            .map(|recent| (recent.clone(), Details::of(recent)))
            .collect();
        self.rows_read = Some(Instant::now());
    }

    fn busy(&self) -> bool {
        self.task
            .as_ref()
            .is_some_and(|task| task.state == State::Running)
    }

    /// Start `orrin args…` as the current task.
    fn run(&mut self, ctx: &egui::Context, args: Vec<OsString>, title: String, kind: Kind) {
        let cli = match &self.cli {
            Ok(cli) => cli.clone(),
            Err(error) => {
                self.notice = Some(Notice {
                    text: error.clone(),
                    error: true,
                });
                return;
            }
        };
        self.notice = None;
        self.show_output = matches!(kind, Kind::Build);
        self.task = Some(Task::spawn(&cli, &args, title, kind, ctx));
    }

    fn name_of(&self, dir: &Path) -> String {
        Recent {
            path: dir.to_path_buf(),
            ..Default::default()
        }
        .name()
    }

    fn open_in_editor(&mut self, ctx: &egui::Context, dir: &Path) {
        self.recents.touch(dir, recents::now());
        self.save_recents();
        let name = self.name_of(dir);
        self.run(
            ctx,
            vec!["run".into(), "--project".into(), dir.into()],
            format!("Opening {name} in the editor"),
            Kind::Edit,
        );
    }

    fn export(&mut self, ctx: &egui::Context, dir: &Path, platform: Platform) {
        let name = self.name_of(dir);
        let out = dir.join("exports").join(format!("{name}-{platform}"));
        self.run(
            ctx,
            vec![
                "export".into(),
                "--project".into(),
                dir.into(),
                "--platform".into(),
                platform.name().into(),
            ],
            format!("Exporting {name} for {}", platform_label(platform)),
            Kind::Export { platform, out },
        );
    }

    fn build(&mut self, ctx: &egui::Context, dir: &Path) {
        let name = self.name_of(dir);
        self.run(
            ctx,
            vec!["build".into(), "--project".into(), dir.into()],
            format!("Building {name}'s scripts"),
            Kind::Build,
        );
    }

    fn new_draft(&mut self) {
        self.draft = Some(Draft {
            name: String::new(),
            location: default_location(&self.recents),
            open_after: true,
            focused: false,
        });
    }

    /// A folder picked to add: the project it is in, remembered and listed, or
    /// a notice saying why it is not one.
    fn add_from_disk(&mut self) {
        let Some(dir) = rfd::FileDialog::new()
            .set_title("Add an Orrin project")
            .pick_folder()
        else {
            return;
        };
        match orrin_project::Project::locate(&dir) {
            Ok(Some(project)) => {
                let root = project.root().to_path_buf();
                self.recents.touch(&root, recents::now());
                self.save_recents();
                self.notice = Some(Notice {
                    text: format!("Added {}.", project.name()),
                    error: false,
                });
            }
            Ok(None) => {
                self.notice = Some(Notice {
                    text: format!(
                        "{} is not an Orrin project: there is no orrin.toml in it or above it. \
                         Pick a project folder, or create a new project.",
                        dir.display()
                    ),
                    error: true,
                })
            }
            Err(error) => {
                self.notice = Some(Notice {
                    text: format!("That project's orrin.toml does not load: {error}"),
                    error: true,
                })
            }
        }
    }

    /// React to the task ending: remember a new project, open it if asked.
    fn finished(&mut self, ctx: &egui::Context) {
        let Some(task) = &self.task else {
            return;
        };
        let (state, kind) = (task.state, task.kind.clone());
        match (state, kind) {
            (State::Done, Kind::Create { dir, open_after }) => {
                self.recents.touch(&dir, recents::now());
                self.save_recents();
                if open_after {
                    self.open_in_editor(ctx, &dir);
                } else {
                    self.notice = Some(Notice {
                        text: format!("Created {}.", dir.display()),
                        error: false,
                    });
                    self.task = None;
                }
            }
            (State::Failed, Kind::Create { .. }) => {
                let reason = task
                    .last_line()
                    .map(|line| line.trim_start_matches("orrin: ").to_owned())
                    .unwrap_or_else(|| "orrin new failed".to_owned());
                self.notice = Some(Notice {
                    text: format!("Could not create the project: {reason}"),
                    error: true,
                });
                self.task = None;
            }
            (State::Done, Kind::Export { .. }) => self.rows_read = None,
            (State::Failed, _) => self.show_output = true,
            _ => {}
        }
    }

    fn apply(&mut self, ctx: &egui::Context, dir: PathBuf, action: Action) {
        match action {
            Action::Open => self.open_in_editor(ctx, &dir),
            Action::Build => self.build(ctx, &dir),
            Action::Export(platform) => self.export(ctx, &dir, platform),
            Action::Reveal => reveal(&dir),
            Action::CopyPath => ctx.copy_text(dir.display().to_string()),
            Action::Favourite => {
                self.recents.toggle_favourite(&dir);
                self.save_recents();
            }
            Action::Target(platform) => {
                self.recents.set_platform(&dir, platform);
                self.save_recents();
            }
            Action::Forget => {
                self.recents.forget(&dir);
                self.save_recents();
            }
        }
    }
}

impl eframe::App for Hub {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.draw(ctx);
    }
}

impl Hub {
    /// One frame of the whole window. Apart from `update` so a test can run
    /// it on a bare `egui::Context`, with no window behind it.
    fn draw(&mut self, ctx: &egui::Context) {
        if self.task.as_mut().is_some_and(Task::poll) {
            self.finished(ctx);
        }
        if self.busy() {
            // Spinners and elapsed state animate even when no line arrives.
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        self.refresh_rows();

        self.title_bar(ctx);
        self.sidebar(ctx);
        self.activity(ctx);
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(theme::CHROME)
                    .inner_margin(egui::Margin {
                        left: 0,
                        right: 12,
                        top: 0,
                        bottom: 12,
                    }),
            )
            .show(ctx, |ui| {
                theme::panel().show(ui, |ui| {
                    ui.set_min_size(ui.available_size());
                    match self.page {
                        Page::Projects => self.projects(ui),
                        Page::Learn => self.learn(ui),
                    }
                });
            });
        self.new_project_dialog(ctx);
    }
}

impl Hub {
    fn title_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("title")
            .exact_height(44.0)
            .show_separator_line(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::CHROME)
                    .inner_margin(egui::Margin::symmetric(16, 0)),
            )
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.add(
                        egui::Image::new(egui::include_image!("../../core/assets/logo.svg"))
                            .fit_to_exact_size(vec2(20.0, 20.0)),
                    );
                    ui.label(RichText::new("Orrin Hub").size(17.0).strong());
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new(icon(glyph::GITHUB, 18.0)).frame(false))
                            .on_hover_text("Orrin on GitHub")
                            .clicked()
                        {
                            ctx.open_url(egui::OpenUrl::new_tab(GITHUB));
                        }
                        if ui
                            .add(egui::Button::new(icon(glyph::GRADUATION_CAP, 18.0)).frame(false))
                            .on_hover_text("Learn")
                            .clicked()
                        {
                            self.page = Page::Learn;
                        }
                    });
                });
            });
    }

    fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("nav")
            .exact_width(236.0)
            .resizable(false)
            .show_separator_line(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::CHROME)
                    .inner_margin(egui::Margin {
                        left: 12,
                        right: 12,
                        top: 0,
                        bottom: 12,
                    }),
            )
            .show(ctx, |ui| {
                theme::panel()
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.spacing_mut().item_spacing.y = 4.0;
                        for (page, glyph, label) in [
                            (Page::Projects, glyph::FOLDER, "Projects"),
                            (Page::Learn, glyph::LIGHTBULB, "Learn"),
                        ] {
                            if nav_item(ui, glyph, label, self.page == page).clicked() {
                                self.page = page;
                            }
                        }
                    });

                ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
                    match &self.cli {
                        Ok(_) => ui.label(
                            RichText::new(format!("Orrin {}", env!("CARGO_PKG_VERSION")))
                                .small()
                                .color(theme::INK_4),
                        ),
                        Err(error) => ui
                            .label(
                                RichText::new(format!("{} orrin CLI not found", glyph::WARNING))
                                    .small()
                                    .color(theme::ERROR),
                            )
                            .on_hover_text(error),
                    };
                    ui.horizontal(|ui| {
                        for (label, url) in [("Docs", DOCS), ("Discord", DISCORD)] {
                            let link = ui.add(
                                egui::Label::new(RichText::new(label).small().color(theme::INK_3))
                                    .sense(Sense::click()),
                            );
                            if link
                                .on_hover_cursor(egui::CursorIcon::PointingHand)
                                .clicked()
                            {
                                ctx.open_url(egui::OpenUrl::new_tab(url));
                            }
                        }
                    });
                });
            });
    }

    fn notice(&mut self, ui: &mut egui::Ui) {
        let Some(notice) = &self.notice else {
            return;
        };
        let colour = if notice.error {
            theme::ERROR
        } else {
            theme::OK
        };
        let mut dismissed = false;
        egui::Frame::new()
            .fill(colour.gamma_multiply(0.12))
            .stroke(Stroke::new(1.0_f32, colour.gamma_multiply(0.4)))
            .corner_radius(CornerRadius::same(theme::RADIUS_SM))
            .inner_margin(egui::Margin::symmetric(12, 8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let mark = if notice.error {
                        glyph::WARNING
                    } else {
                        glyph::CHECK_CIRCLE
                    };
                    ui.label(icon(mark, 16.0).color(colour));
                    ui.add(egui::Label::new(&notice.text).wrap());
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        dismissed = ui
                            .add(egui::Button::new(icon(glyph::X, 14.0)).frame(false))
                            .on_hover_text("Dismiss")
                            .clicked();
                    });
                });
            });
        ui.add_space(12.0);
        if dismissed {
            self.notice = None;
        }
    }

    fn projects(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let busy = self.busy();
        egui::Frame::new()
            .inner_margin(egui::Margin {
                left: 20,
                right: 20,
                top: 18,
                bottom: 14,
            })
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Projects");
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        // One height for the whole row: a framed button is
                        // never shorter than `interact_size`, and the search
                        // field is given the same.
                        ui.spacing_mut().interact_size.y = CONTROL_HEIGHT;
                        if ui
                            .add_enabled(
                                !busy,
                                theme::primary(format!("{}  New project", glyph::PLUS)),
                            )
                            .on_disabled_hover_text("Wait for the current task to finish")
                            .clicked()
                        {
                            self.new_draft();
                        }
                        ui.menu_button(format!("Add  {}", glyph::CARET_DOWN), |ui| {
                            ui.set_min_width(220.0);
                            if ui.button("Add project from disk…").clicked() {
                                ui.close_menu();
                                self.add_from_disk();
                            }
                        });
                        ui.add(
                            egui::TextEdit::singleline(&mut self.search)
                                .hint_text(format!("{}  Search", glyph::SEARCH))
                                .desired_width(240.0)
                                .min_size(vec2(240.0, CONTROL_HEIGHT))
                                .vertical_align(Align::Center)
                                .margin(vec2(10.0, 0.0)),
                        );
                        let chip = match self.platform_filter {
                            Some(platform) => {
                                format!("{}  {}", platform_label(platform), glyph::CARET_DOWN)
                            }
                            None => format!("Platform  {}", glyph::CARET_DOWN),
                        };
                        ui.menu_button(chip, |ui| {
                            if ui
                                .selectable_label(self.platform_filter.is_none(), "All platforms")
                                .clicked()
                            {
                                self.platform_filter = None;
                                ui.close_menu();
                            }
                            for platform in Platform::ALL {
                                let picked = self.platform_filter == Some(platform);
                                if ui
                                    .selectable_label(picked, platform_label(platform))
                                    .clicked()
                                {
                                    self.platform_filter = Some(platform);
                                    ui.close_menu();
                                }
                            }
                        });
                    });
                });
                ui.add_space(4.0);
                self.notice(ui);
            });

        if self.recents.projects.is_empty() {
            self.empty_state(ui);
            return;
        }

        let columns = Columns::across(ui.available_rect_before_wrap());
        if let Some(sort) = header(ui, &columns, self.sort, self.ascending) {
            if sort == self.sort {
                self.ascending = !self.ascending;
            } else {
                self.sort = sort;
                self.ascending = sort == Sort::Name;
            }
        }

        let now = recents::now();
        let mut chosen = None;
        let shown: Vec<(Recent, Details)> = recents::order(
            &self.rows,
            self.sort,
            self.ascending,
            &self.search,
            self.platform_filter,
        )
        .into_iter()
        .cloned()
        .collect();
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                if shown.is_empty() {
                    ui.add_space(24.0);
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("No project matches.").color(theme::INK_3));
                    });
                }
                for row in &shown {
                    if let Some(action) = project_row(ui, row, &columns, now, busy) {
                        chosen = Some((row.0.path.clone(), action));
                    }
                }
            });
        if let Some((dir, action)) = chosen {
            self.apply(&ctx, dir, action);
        }
    }

    fn empty_state(&mut self, ui: &mut egui::Ui) {
        ui.add_space(64.0);
        ui.vertical_centered(|ui| {
            ui.add(
                egui::Image::new(egui::include_image!("../../core/assets/logo.svg"))
                    .fit_to_exact_size(vec2(52.0, 52.0)),
            );
            ui.add_space(14.0);
            ui.label(RichText::new("Make your first game").size(20.0).strong());
            ui.label(
                RichText::new(
                    "A project holds your scenes, your assets and the C# that brings them to life.",
                )
                .color(theme::INK_3),
            );
            ui.add_space(16.0);
            if ui
                .add(theme::primary(format!("{}  New project", glyph::PLUS)))
                .clicked()
            {
                self.new_draft();
            }
            ui.add_space(4.0);
            if ui
                .add(
                    egui::Button::new(
                        RichText::new("or add an existing project").color(theme::ACCENT_HOVER),
                    )
                    .frame(false),
                )
                .clicked()
            {
                self.add_from_disk();
            }
        });
    }

    fn learn(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .inner_margin(egui::Margin::symmetric(20, 18))
            .show(ui, |ui| {
                ui.heading("Learn");
                ui.label(RichText::new("Where to go next.").color(theme::INK_3));
                ui.add_space(16.0);
                for (glyph, title, about, url) in [
                    (
                        glyph::LIGHTBULB,
                        "Getting started",
                        "Make a project, write a Behaviour, and run it.",
                        DOCS,
                    ),
                    (
                        glyph::GLOBE,
                        "Roadmap",
                        "What is built, what is next, and what is not yet.",
                        ROADMAP,
                    ),
                    (
                        glyph::TERMINAL,
                        "Source",
                        "Read the engine, file an issue, or send a fix.",
                        GITHUB,
                    ),
                    (
                        glyph::PLAY,
                        "Discord",
                        "Ask questions and show what you made.",
                        DISCORD,
                    ),
                ] {
                    let width = ui.available_width();
                    let (rect, response) =
                        ui.allocate_exact_size(vec2(width, 64.0), Sense::click());
                    let fill = if response.hovered() {
                        theme::HOVER
                    } else {
                        theme::PANEL
                    };
                    ui.painter().rect(
                        rect,
                        CornerRadius::same(theme::RADIUS_MD),
                        fill,
                        Stroke::new(1.0_f32, theme::OUTLINE),
                        egui::StrokeKind::Inside,
                    );
                    ui.scope_builder(
                        egui::UiBuilder::new().max_rect(rect.shrink2(vec2(16.0, 10.0))),
                        |ui| {
                            ui.horizontal_centered(|ui| {
                                ui.label(icon(glyph, 22.0).color(theme::ACCENT_HOVER));
                                ui.add_space(6.0);
                                ui.vertical(|ui| {
                                    ui.label(RichText::new(title).size(15.0).strong());
                                    ui.label(RichText::new(about).color(theme::INK_3));
                                });
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    ui.label(icon(glyph::ARROW_OUT, 16.0).color(theme::INK_4));
                                });
                            });
                        },
                    );
                    if response
                        .on_hover_cursor(egui::CursorIcon::PointingHand)
                        .clicked()
                    {
                        ui.ctx().open_url(egui::OpenUrl::new_tab(url));
                    }
                    ui.add_space(8.0);
                }
            });
    }
    fn new_project_dialog(&mut self, ctx: &egui::Context) {
        let Some(draft) = &mut self.draft else {
            return;
        };
        let mut create = false;
        let mut close = false;
        let folder = draft.location.join(draft.name.trim());
        let problem = name_problem(draft.name.trim(), &folder);
        let modal = egui::Modal::new(egui::Id::new("new-project"))
            .frame(
                egui::Frame::new()
                    .fill(theme::PANEL)
                    .stroke(Stroke::new(1.0_f32, theme::OUTLINE))
                    .corner_radius(CornerRadius::same(theme::RADIUS_MD))
                    .inner_margin(egui::Margin::same(24)),
            )
            .show(ctx, |ui| {
                ui.set_width(440.0);
                ui.label(RichText::new("New project").size(20.0).strong());
                ui.label(
                    RichText::new(
                        "A folder with a manifest, a C# project and a spinning cube to start from.",
                    )
                    .color(theme::INK_3),
                );
                ui.add_space(18.0);

                ui.label(RichText::new("Name").color(theme::INK_2));
                let field = ui.add(
                    egui::TextEdit::singleline(&mut draft.name)
                        .hint_text("space-shooter")
                        .desired_width(f32::INFINITY)
                        .margin(vec2(10.0, 8.0)),
                );
                if !draft.focused {
                    field.request_focus();
                    draft.focused = true;
                }
                if field.lost_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter))
                    && problem.is_none()
                {
                    create = true;
                }
                ui.add_space(12.0);

                ui.label(RichText::new("Location").color(theme::INK_2));
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Label::new(
                            RichText::new(tilde(&draft.location))
                                .monospace()
                                .color(theme::INK),
                        )
                        .truncate(),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.add(theme::secondary("Change…")).clicked()
                            && let Some(dir) = rfd::FileDialog::new()
                                .set_title("Where should the project go?")
                                .set_directory(&draft.location)
                                .pick_folder()
                        {
                            draft.location = dir;
                        }
                    });
                });
                ui.add_space(10.0);

                match &problem {
                    Some(problem) if !draft.name.trim().is_empty() => {
                        ui.label(RichText::new(problem).color(theme::ERROR));
                    }
                    Some(_) => {
                        ui.label(RichText::new(" ").small());
                    }
                    None => {
                        ui.label(
                            RichText::new(format!("Creates {}", tilde(&folder)))
                                .small()
                                .color(theme::INK_3),
                        );
                    }
                }
                ui.add_space(8.0);
                ui.checkbox(
                    &mut draft.open_after,
                    "Open it in the editor when it is ready",
                );
                ui.add_space(18.0);

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add_enabled(problem.is_none(), theme::primary("Create project"))
                        .clicked()
                    {
                        create = true;
                    }
                    if ui.add(theme::secondary("Cancel")).clicked() {
                        close = true;
                    }
                });
            });
        if modal.should_close() {
            close = true;
        }
        if create {
            let draft = self.draft.take().expect("the dialog is open");
            let name = draft.name.trim().to_owned();
            let dir = draft.location.join(&name);
            self.run(
                ctx,
                vec![
                    "new".into(),
                    name.clone().into(),
                    "--path".into(),
                    draft.location.clone().into(),
                ],
                format!("Creating {name}"),
                Kind::Create {
                    dir,
                    open_after: draft.open_after,
                },
            );
        } else if close {
            self.draft = None;
        }
    }

    fn activity(&mut self, ctx: &egui::Context) {
        let Some(task) = &self.task else {
            return;
        };
        let mut dismiss = false;
        let mut cancel = false;
        let mut reveal_out = None;
        // A panel of its own on the chrome, like the two above it, rather than
        // a strip with egui's separator: that line would stop at the sidebar.
        egui::TopBottomPanel::bottom("activity")
            .resizable(self.show_output)
            .show_separator_line(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::CHROME)
                    .inner_margin(egui::Margin {
                        left: 0,
                        right: 12,
                        top: 0,
                        bottom: 12,
                    }),
            )
            .show(ctx, |ui| {
                theme::panel()
                    .inner_margin(egui::Margin::symmetric(16, 12))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.spacing_mut().item_spacing.y = 2.0;
                                // The mark on the title's own line, so it lines up
                                // with the words it marks whether or not a detail
                                // line follows.
                                ui.horizontal(|ui| {
                                    match task.state {
                                        State::Running => {
                                            ui.add(
                                                egui::Spinner::new()
                                                    .size(16.0)
                                                    .color(theme::ACCENT),
                                            );
                                        }
                                        State::Done => {
                                            ui.label(
                                                icon(glyph::CHECK_CIRCLE, 18.0).color(theme::OK),
                                            );
                                        }
                                        State::Failed => {
                                            ui.label(
                                                icon(glyph::X_CIRCLE, 18.0).color(theme::ERROR),
                                            );
                                        }
                                        State::Cancelled => {
                                            ui.label(
                                                icon(glyph::X_CIRCLE, 18.0).color(theme::INK_4),
                                            );
                                        }
                                    }
                                    ui.label(RichText::new(status_title(task)).strong());
                                });
                                let detail = status_detail(task);
                                if !detail.is_empty() {
                                    ui.horizontal(|ui| {
                                        ui.add_space(26.0);
                                        ui.add(
                                            egui::Label::new(
                                                RichText::new(detail).small().color(theme::INK_3),
                                            )
                                            .truncate(),
                                        );
                                    });
                                }
                            });
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if task.state == State::Running {
                                    if task.kind != Kind::Edit
                                        && ui.add(theme::secondary("Cancel")).clicked()
                                    {
                                        cancel = true;
                                    }
                                } else if ui.add(theme::secondary("Dismiss")).clicked() {
                                    dismiss = true;
                                }
                                if let (State::Done, Kind::Export { out, .. }) =
                                    (task.state, &task.kind)
                                    && ui
                                        .add(theme::primary(format!(
                                            "{}  Show the game",
                                            glyph::ARROW_OUT
                                        )))
                                        .clicked()
                                {
                                    reveal_out = Some(out.clone());
                                }
                                let label = if self.show_output {
                                    "Hide output"
                                } else {
                                    "Show output"
                                };
                                if ui
                                    .add(
                                        egui::Button::new(RichText::new(label).color(theme::INK_3))
                                            .frame(false),
                                    )
                                    .clicked()
                                {
                                    self.show_output = !self.show_output;
                                }
                            });
                        });
                        if let (State::Running, Some(step)) = (task.state, task.step()) {
                            ui.add_space(4.0);
                            ui.add(
                                egui::ProgressBar::new((step.index as f32 - 0.5) / step.of as f32)
                                    .desired_height(4.0)
                                    .fill(theme::ACCENT),
                            );
                        }
                        if self.show_output {
                            ui.add_space(8.0);
                            egui::Frame::new()
                                .fill(theme::PANEL)
                                .corner_radius(CornerRadius::same(theme::RADIUS_SM))
                                .inner_margin(egui::Margin::same(10))
                                .show(ui, |ui| {
                                    egui::ScrollArea::vertical()
                                        .stick_to_bottom(true)
                                        .auto_shrink([false, true])
                                        .max_height(260.0)
                                        .show(ui, |ui| {
                                            for line in task.lines() {
                                                ui.label(
                                                    RichText::new(line)
                                                        .monospace()
                                                        .color(theme::INK_2),
                                                );
                                            }
                                        });
                                });
                        }
                    });
            });
        if let Some(out) = reveal_out {
            reveal(&out);
        }
        if cancel && let Some(task) = &mut self.task {
            task.cancel();
        }
        if dismiss {
            self.task = None;
            self.show_output = false;
        }
    }
}

#[derive(Debug, PartialEq)]
enum Action {
    Open,
    Build,
    Export(Platform),
    Reveal,
    CopyPath,
    Favourite,
    Target(Platform),
    Forget,
}

/// Where each column sits, shared by the header and every row so they line up.
struct Columns {
    star: Rect,
    git: Rect,
    name: Rect,
    platform: Rect,
    version: Rect,
    modified: Rect,
    more: Rect,
}

impl Columns {
    /// Fixed widths at both ends, and the name taking what is left.
    fn across(area: Rect) -> Self {
        const PAD: f32 = 12.0;
        let left = area.left() + PAD;
        let right = area.right() - PAD;
        let span = |from: f32, to: f32| Rect::from_x_y_ranges(from..=to, 0.0..=0.0);
        let more = span(right - 40.0, right);
        let modified = span(more.left() - 130.0, more.left());
        let version = span(modified.left() - 130.0, modified.left());
        let platform = span(version.left() - 150.0, version.left());
        let star = span(left, left + 36.0);
        let git = span(star.right(), star.right() + 36.0);
        let name = span(
            git.right() + 8.0,
            (platform.left() - 12.0).max(git.right() + 120.0),
        );
        Self {
            star,
            git,
            name,
            platform,
            version,
            modified,
            more,
        }
    }

    /// `column`'s span, given the vertical extent of the row it sits in.
    fn cell(column: Rect, row: Rect) -> Rect {
        Rect::from_x_y_ranges(column.x_range(), row.y_range())
    }
}

/// The table's header strip. Returns the column clicked to sort by.
fn header(ui: &mut egui::Ui, columns: &Columns, sort: Sort, ascending: bool) -> Option<Sort> {
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(width, 40.0), Sense::hover());
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, theme::HEADER);
    let dim = |glyph| icon(glyph, 16.0).color(theme::INK_4);
    ui.put(
        Columns::cell(columns.star, rect),
        egui::Label::new(dim(glyph::STAR)),
    )
    .on_hover_text("Favourites, listed first");
    ui.put(
        Columns::cell(columns.git, rect),
        egui::Label::new(dim(glyph::GIT_BRANCH)),
    )
    .on_hover_text("In a git repository");
    let mut clicked = None;
    for (column, label, by) in [
        (columns.name, "Name", Some(Sort::Name)),
        (columns.platform, "Platform", None),
        (columns.version, "Editor version", None),
        (columns.modified, "Modified", Some(Sort::Modified)),
    ] {
        let arrow = match by {
            Some(by) if by == sort => {
                format!(
                    "  {}",
                    if ascending {
                        glyph::ARROW_UP
                    } else {
                        glyph::ARROW_DOWN
                    }
                )
            }
            _ => String::new(),
        };
        let text = RichText::new(format!("{label}{arrow}"))
            .strong()
            .color(theme::INK_2);
        let cell = Columns::cell(column, rect);
        let response = ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(cell)
                .layout(Layout::left_to_right(Align::Center)),
            |ui| match by {
                Some(_) => ui.add(egui::Button::new(text).frame(false)),
                None => ui.label(text),
            },
        );
        if let Some(by) = by
            && response.inner.clicked()
        {
            clicked = Some(by);
        }
    }
    clicked
}

fn project_row(
    ui: &mut egui::Ui,
    (recent, details): &(Recent, Details),
    columns: &Columns,
    now: u64,
    busy: bool,
) -> Option<Action> {
    let mut action = None;
    let missing = details.missing;
    let width = ui.available_width();
    // Sensed here, before anything is put on it: see the module docs.
    let (rect, row) = ui.allocate_exact_size(vec2(width, 60.0), Sense::click());
    if ui.rect_contains_pointer(rect) {
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, theme::HOVER);
    }
    ui.painter().hline(
        rect.x_range(),
        rect.bottom() - 0.5,
        Stroke::new(1.0_f32, theme::RULE),
    );
    let cell = |column: Rect| Columns::cell(column, rect);

    let star = if recent.favourite {
        theme::STAR
    } else {
        theme::INK_4
    };
    if ui
        .put(
            cell(columns.star),
            egui::Button::new(icon(glyph::STAR, 18.0).color(star)).frame(false),
        )
        .on_hover_text(if recent.favourite {
            "Remove from favourites"
        } else {
            "Add to favourites"
        })
        .clicked()
    {
        action = Some(Action::Favourite);
    }
    if details.git {
        ui.put(
            cell(columns.git),
            egui::Label::new(icon(glyph::GIT_BRANCH, 18.0).color(theme::INK_2)),
        )
        .on_hover_text("In a git repository");
    }

    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(cell(columns.name))
            .layout(Layout::top_down(Align::Min)),
        |ui| {
            ui.add_space(10.0);
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.horizontal(|ui| {
                let title = RichText::new(&details.name).size(15.0);
                ui.add(
                    egui::Label::new(if missing {
                        title.color(theme::INK_4)
                    } else {
                        title
                    })
                    .truncate(),
                );
                if missing {
                    ui.label(
                        RichText::new("Folder not found")
                            .small()
                            .color(theme::ERROR),
                    );
                }
            });
            ui.add(
                egui::Label::new(
                    RichText::new(tilde(&recent.path))
                        .size(12.5)
                        .color(theme::INK_3),
                )
                .truncate(),
            );
        },
    );

    if !missing {
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(cell(columns.platform).shrink2(vec2(0.0, 14.0)))
                .layout(Layout::left_to_right(Align::Center)),
            |ui| {
                quiet(ui);
                let target = recent.target();
                egui::ComboBox::from_id_salt(("platform", &recent.path))
                    .selected_text(platform_label(target))
                    .width(126.0)
                    .show_ui(ui, |ui| {
                        for platform in Platform::ALL {
                            if ui
                                .selectable_label(platform == target, platform_label(platform))
                                .clicked()
                            {
                                action = Some(Action::Target(platform));
                            }
                        }
                    })
                    .response
                    .on_hover_text("What this project's Export builds for");
            },
        );
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(cell(columns.version))
                .layout(Layout::left_to_right(Align::Center)),
            |ui| ui.label(RichText::new(env!("CARGO_PKG_VERSION")).color(theme::INK_2)),
        );
        let modified = details
            .modified
            .map_or_else(|| "—".to_owned(), |at| recents::ago(at, now));
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(cell(columns.modified))
                .layout(Layout::left_to_right(Align::Center)),
            |ui| ui.label(RichText::new(modified).color(theme::INK_2)),
        );
    }

    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(cell(columns.more))
            .layout(Layout::right_to_left(Align::Center)),
        |ui| {
            quiet(ui);
            ui.menu_button(icon(glyph::ELLIPSIS, 18.0), |ui| {
                ui.set_min_width(230.0);
                if !missing {
                    ui.add_enabled_ui(!busy, |ui| {
                        if ui
                            .button(format!("{}  Open in the editor", glyph::PLAY))
                            .clicked()
                        {
                            action = Some(Action::Open);
                            ui.close_menu();
                        }
                        ui.separator();
                        ui.label(
                            RichText::new("Export a playable game")
                                .small()
                                .color(theme::INK_4),
                        );
                        for platform in Platform::ALL {
                            if ui
                                .button(format!("{}  {}", glyph::BOX, platform_label(platform)))
                                .clicked()
                            {
                                action = Some(Action::Export(platform));
                                ui.close_menu();
                            }
                        }
                        ui.separator();
                        if ui
                            .button(format!("{}  Build scripts", glyph::TERMINAL))
                            .clicked()
                        {
                            action = Some(Action::Build);
                            ui.close_menu();
                        }
                    });
                    if ui
                        .button(format!("{}  {}", glyph::FOLDER, reveal_label()))
                        .clicked()
                    {
                        action = Some(Action::Reveal);
                        ui.close_menu();
                    }
                    if ui.button("Copy path").clicked() {
                        action = Some(Action::CopyPath);
                        ui.close_menu();
                    }
                    ui.separator();
                }
                if ui
                    .button(RichText::new("Remove from list").color(theme::ERROR))
                    .clicked()
                {
                    action = Some(Action::Forget);
                    ui.close_menu();
                }
            })
            .response
            .on_hover_text("More");
        },
    );

    if action.is_none() && row.clicked() && !missing && !busy {
        action = Some(Action::Open);
    }
    if !missing {
        row.on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text("Open in the editor");
    }
    action
}

/// A control that reads as text on its row until the pointer reaches it.
fn quiet(ui: &mut egui::Ui) {
    let idle = &mut ui.visuals_mut().widgets.inactive;
    idle.weak_bg_fill = Color32::TRANSPARENT;
    idle.bg_fill = Color32::TRANSPARENT;
    idle.bg_stroke = Stroke::NONE;
}

fn nav_item(ui: &mut egui::Ui, glyph: char, label: &str, selected: bool) -> egui::Response {
    let (fill, colour) = if selected {
        (theme::SELECTED, theme::INK)
    } else {
        (Color32::TRANSPARENT, theme::INK_2)
    };
    let text = RichText::new(format!("{glyph}   {label}")).color(colour);
    // Justified and aligned to the start, so the label sits at the left of a
    // full-width row as a sidebar's does, rather than centred in it.
    ui.with_layout(Layout::top_down_justified(Align::Min), |ui| {
        ui.add(
            egui::Button::new(text)
                .fill(fill)
                .stroke(Stroke::NONE)
                .corner_radius(CornerRadius::same(theme::RADIUS_SM + 2))
                .min_size(vec2(0.0, 36.0)),
        )
    })
    .inner
}

fn status_title(task: &Task) -> String {
    match (&task.kind, task.state) {
        (_, State::Running) => format!("{}…", task.title),
        (Kind::Edit, State::Done) => "The editor closed".to_owned(),
        (Kind::Export { platform, .. }, State::Done) => {
            format!("The {} build is ready", platform_label(*platform))
        }
        (_, State::Done) => format!("{} — done", task.title),
        (_, State::Failed) => format!("{} — failed", task.title),
        (_, State::Cancelled) => format!("{} — cancelled", task.title),
    }
}

fn status_detail(task: &Task) -> String {
    match (&task.kind, task.state) {
        (_, State::Running) => match task.step() {
            Some(step) => format!("Step {} of {}: {}", step.index, step.of, step.what),
            None if task.kind == Kind::Edit => {
                "The first launch compiles the engine, which takes a few minutes.".to_owned()
            }
            None => task.last_line().unwrap_or_default(),
        },
        (Kind::Export { out, .. }, State::Done) => tilde(out),
        (_, State::Failed) => task
            .last_line()
            .map(|line| line.trim_start_matches("orrin: ").to_owned())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

fn platform_label(platform: Platform) -> &'static str {
    match platform {
        Platform::Mac => "macOS",
        Platform::Linux => "Linux",
        Platform::Windows => "Windows",
    }
}

/// Why `name` cannot be created at `folder`, or `None` when it can. The CLI
/// checks again and has the last word; this is so the button can say no
/// before it is pressed.
fn name_problem(name: &str, folder: &Path) -> Option<String> {
    if name.is_empty() {
        return Some("Give it a name.".to_owned());
    }
    if !name.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
        return Some("Start the name with a letter.".to_owned());
    }
    if let Some(bad) = name
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || *c == '-' || *c == '_' || *c == ' '))
    {
        return Some(format!(
            "`{bad}` cannot be in a project name; use letters, digits, - and _."
        ));
    }
    if folder.exists() && std::fs::read_dir(folder).is_ok_and(|mut dir| dir.next().is_some()) {
        return Some(format!("There is already a folder called {name} there."));
    }
    None
}

/// Where a new project goes: beside the most recent one, else `~/Orrin`.
fn default_location(recents: &Recents) -> PathBuf {
    recents
        .projects
        .iter()
        .find(|recent| !recent.is_missing())
        .and_then(|recent| recent.path.parent().map(Path::to_path_buf))
        .or_else(|| home().map(|home| home.join("Orrin")))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn home() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

/// `path` with the home directory written as `~`, which is how people read it.
fn tilde(path: &Path) -> String {
    match home().and_then(|home| path.strip_prefix(home).ok().map(Path::to_path_buf)) {
        Some(rest) if !cfg!(windows) => format!("~/{}", rest.display()),
        _ => path.display().to_string(),
    }
}

fn reveal_label() -> &'static str {
    if cfg!(target_os = "macos") {
        "Show in Finder"
    } else if cfg!(windows) {
        "Show in Explorer"
    } else {
        "Show in Files"
    }
}

fn reveal(path: &Path) {
    let mut command = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else if cfg!(windows) {
        std::process::Command::new("explorer")
    } else {
        std::process::Command::new("xdg-open")
    };
    let _ = command.arg(path).spawn();
}

/// The `orrin` CLI the hub drives: the one beside it, as an install lays them
/// out and as cargo builds them, else whatever `orrin` is on the PATH.
fn find_cli() -> Result<PathBuf, String> {
    let name = format!("orrin{}", std::env::consts::EXE_SUFFIX);
    let beside = std::env::current_exe()
        .ok()
        .map(|exe| exe.canonicalize().unwrap_or(exe))
        .and_then(|exe| exe.parent().map(|dir| dir.join(&name)))
        .filter(|cli| cli.is_file());
    if let Some(cli) = beside {
        return Ok(cli);
    }
    let on_path = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(&name))
            .find(|cli| cli.is_file())
    });
    on_path.ok_or_else(|| {
        "The hub runs everything through the `orrin` command, and it is not beside the hub or \
         on your PATH. Build it with `cargo build -p orrin-cli`."
            .to_owned()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_the_cli_would_refuse_is_refused_before_create_is_pressed() {
        let dir = tempfile::tempdir().unwrap();
        assert!(name_problem("", dir.path()).is_some());
        assert!(name_problem("1st", &dir.path().join("1st")).is_some());
        assert!(
            name_problem("a/b", &dir.path().join("a/b"))
                .unwrap()
                .contains('/')
        );
        assert_eq!(
            name_problem("space-shooter", &dir.path().join("space-shooter")),
            None
        );

        std::fs::create_dir_all(dir.path().join("taken")).unwrap();
        std::fs::write(dir.path().join("taken/file"), b"").unwrap();
        assert!(
            name_problem("taken", &dir.path().join("taken"))
                .unwrap()
                .contains("already")
        );
    }

    /// Where `text` was drawn last frame, found in the frame's own shapes, so
    /// a test can click a widget by what it says.
    fn drawn_at(output: &egui::FullOutput, text: &str) -> Option<egui::Pos2> {
        fn walk(shape: &egui::Shape, text: &str) -> Option<egui::Pos2> {
            match shape {
                egui::Shape::Text(drawn) if drawn.galley.text() == text => {
                    Some(drawn.pos + drawn.galley.rect.center().to_vec2())
                }
                egui::Shape::Vec(shapes) => shapes.iter().find_map(|shape| walk(shape, text)),
                _ => None,
            }
        }
        output
            .shapes
            .iter()
            .find_map(|clipped| walk(&clipped.shape, text))
    }

    /// Draw one project's row, then click each label in `steps` in turn — a
    /// pointer press and release where it was drawn — and return the action
    /// the row reported.
    fn clicked_through(steps: &[&str]) -> Option<Action> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("game");
        let row = (
            Recent {
                path: path.clone(),
                platform: Some(Platform::Mac),
                ..Default::default()
            },
            Details {
                name: "game".to_owned(),
                missing: false,
                git: true,
                modified: Some(0),
            },
        );
        let ctx = egui::Context::default();
        crate::theme::install(&ctx);
        let mut chosen = None;
        let frame = |events: Vec<egui::Event>, chosen: &mut Option<Action>| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    vec2(1100.0, 700.0),
                )),
                events,
                ..Default::default()
            };
            ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let columns = Columns::across(ui.available_rect_before_wrap());
                    if let Some(action) = project_row(ui, &row, &columns, 0, false) {
                        *chosen = Some(action);
                    }
                });
            })
        };
        // The first frame installs the fonts; the second is laid out in them.
        frame(Vec::new(), &mut chosen);
        let mut output = frame(Vec::new(), &mut chosen);
        for label in steps {
            let at = drawn_at(&output, label).unwrap_or_else(|| panic!("`{label}` is not drawn"));
            let button = |pressed| egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            };
            frame(vec![egui::Event::PointerMoved(at)], &mut chosen);
            frame(vec![button(true)], &mut chosen);
            frame(vec![button(false)], &mut chosen);
            output = frame(Vec::new(), &mut chosen);
        }
        chosen
    }

    /// Every control on a row must receive its own click, and a click anywhere
    /// else on the row opens the project.
    #[test]
    fn each_control_on_a_row_receives_its_click_and_the_rest_opens_it() {
        assert_eq!(clicked_through(&["game"]), Some(Action::Open));
        assert_eq!(clicked_through(&["\u{e46a}"]), Some(Action::Favourite));
        assert_eq!(
            clicked_through(&["macOS", "Windows"]),
            Some(Action::Target(Platform::Windows))
        );
        assert_eq!(
            clicked_through(&["\u{e1fe}", "Remove from list"]),
            Some(Action::Forget)
        );
        assert_eq!(
            clicked_through(&["\u{e1fe}", "\u{e1da}  Linux"]),
            Some(Action::Export(Platform::Linux))
        );
    }

    /// Run `hub` for `frames` frames at a typical window size and return what
    /// the last one painted: each string with its rect, and every rectangle.
    fn painted(
        hub: &mut Hub,
        ctx: &egui::Context,
        frames: usize,
    ) -> (Vec<(String, Rect)>, Vec<Rect>) {
        let mut output = None;
        for _ in 0..frames {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, vec2(1180.0, 720.0))),
                ..Default::default()
            };
            output = Some(ctx.run(input, |ctx| hub.draw(ctx)));
        }
        fn walk(shape: &egui::Shape, texts: &mut Vec<(String, Rect)>, rects: &mut Vec<Rect>) {
            match shape {
                egui::Shape::Text(text) => {
                    texts.push((text.galley.text().to_owned(), text.visual_bounding_rect()))
                }
                egui::Shape::Rect(rect) => rects.push(rect.rect),
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, texts, rects)),
                _ => {}
            }
        }
        let (mut texts, mut rects) = (Vec::new(), Vec::new());
        for clipped in &output.expect("a frame ran").shapes {
            walk(&clipped.shape, &mut texts, &mut rects);
        }
        (texts, rects)
    }

    fn text_rect(texts: &[(String, Rect)], wanted: &str) -> Rect {
        texts
            .iter()
            .find(|(text, _)| text == wanted)
            .map(|(_, rect)| *rect)
            .unwrap_or_else(|| panic!("`{wanted}` is not painted"))
    }

    fn hub_with_one_project() -> (tempfile::TempDir, Hub, egui::Context) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("game");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join("orrin.toml"),
            "format_version = 1\n[project]\nname = \"game\"\nversion = \"1\"\n\
             [scripts]\ndir = \"scripts\"\nentry = \"A.B, A\"\n",
        )
        .unwrap();
        let mut recents = Recents::default();
        recents.touch(&path, 0);
        let ctx = egui::Context::default();
        crate::theme::install(&ctx);
        let hub = Hub::with(recents, None, Err("no CLI in a test".to_owned()));
        (dir, hub, ctx)
    }

    /// The controls across the top of the projects panel are one row of one
    /// height, the platform filter included.
    #[test]
    fn the_header_controls_share_one_row_and_one_height() {
        let (_dir, mut hub, ctx) = hub_with_one_project();
        let (texts, rects) = painted(&mut hub, &ctx, 3);

        let labels = [
            format!("{}  Search", glyph::SEARCH),
            format!("Add  {}", glyph::CARET_DOWN),
            format!("{}  New project", glyph::PLUS),
            format!("Platform  {}", glyph::CARET_DOWN),
        ];
        let boxes: Vec<(String, Rect)> = labels
            .iter()
            .map(|label| {
                let text = text_rect(&texts, label);
                let frame = rects
                    .iter()
                    .filter(|rect| rect.contains(text.center()) && rect.height() < 60.0)
                    .min_by(|a, b| a.area().total_cmp(&b.area()))
                    .copied()
                    .unwrap_or_else(|| panic!("`{label}` has no box"));
                (label.clone(), frame)
            })
            .collect();
        let (_, first) = &boxes[0];
        for (label, frame) in &boxes {
            assert!(
                (frame.center().y - first.center().y).abs() <= 1.0,
                "`{label}` at {frame:?} is not on the row of {first:?}"
            );
            assert!(
                (frame.height() - first.height()).abs() <= 1.0,
                "`{label}` is {} tall and the search field {}",
                frame.height(),
                first.height()
            );
        }
    }

    /// A finished task's mark sits on the line of the words it marks.
    #[cfg(unix)]
    #[test]
    fn the_activity_mark_sits_on_its_title_line() {
        let (_dir, mut hub, ctx) = hub_with_one_project();
        hub.task = Some(Task::spawn(
            Path::new("/bin/echo"),
            &["done".into()],
            "Opening game in the editor".to_owned(),
            Kind::Edit,
            &ctx,
        ));
        let deadline = Instant::now() + Duration::from_secs(5);
        while hub
            .task
            .as_ref()
            .is_some_and(|task| task.state == State::Running)
        {
            painted(&mut hub, &ctx, 1);
            assert!(Instant::now() < deadline, "echo never finished");
            std::thread::sleep(Duration::from_millis(10));
        }
        let (texts, _) = painted(&mut hub, &ctx, 2);

        let mark = text_rect(&texts, &glyph::CHECK_CIRCLE.to_string());
        let title = text_rect(&texts, "The editor closed");
        assert!(
            (mark.center().y - title.center().y).abs() <= 1.5,
            "the tick at {mark:?} and its title at {title:?} are on different lines"
        );
    }
}
