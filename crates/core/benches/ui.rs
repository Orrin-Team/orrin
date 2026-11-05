//! The in-game UI's CPU cost per frame: laying a HUD's canvas out into quads,
//! once its glyphs are in the atlas — the steady state, which is every frame
//! after the first few a HUD is on screen.
//!
//! The canvas is a busy HUD rather than a minimal one: forty labels of mixed
//! sizes and anchors, wrapped and clipped, on twenty panels.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

use orrin_core::gfx::TextureHandle;
use orrin_core::gfx::ui::{Painter, UiAssets};
use orrin_core::scene::{Rect, TextAnchor, TextStyle, UiCanvas};
use orrin_registry::AssetId;

struct NoAssets;

impl UiAssets for NoAssets {
    fn texture(&mut self, _: AssetId) -> Option<TextureHandle> {
        None
    }

    fn font(&mut self, _: AssetId) -> Result<Vec<u8>, String> {
        Err(String::new())
    }
}

fn hud() -> UiCanvas {
    let mut canvas = UiCanvas::default();
    canvas.begin([1920, 1080]);
    for i in 0..20 {
        let row = (i / 4) as f32;
        let column = (i % 4) as f32;
        let panel = Rect::new(20.0 + column * 470.0, 20.0 + row * 210.0, 450.0, 190.0);
        canvas.panel(panel, [12, 14, 20, 200]);
        let heading = TextStyle {
            font: None,
            size: 28.0,
            anchor: TextAnchor::UpperLeft,
            color: [255; 4],
        };
        canvas.text(
            Rect::new(panel.x + 12.0, panel.y + 8.0, 426.0, 40.0),
            heading,
            &format!("Objective {i}: hold the line"),
        );
        let body = TextStyle {
            size: 18.0,
            anchor: TextAnchor::MiddleCenter,
            color: [230, 210, 160, 255],
            ..heading
        };
        canvas.text(
            Rect::new(panel.x + 12.0, panel.y + 52.0, 426.0, 126.0),
            body,
            "Defend the relay until the convoy arrives. Ammunition is limited, \
             so pick your shots; reinforcements reach the east gate in 3:20.",
        );
    }
    canvas
}

fn paint(c: &mut Criterion) {
    let canvas = hud();
    let mut painter = Painter::new();
    painter.paint(&canvas, &mut NoAssets);
    let quads = painter.frame().quads.len() as u64;

    let mut group = c.benchmark_group("ui");
    group.throughput(Throughput::Elements(quads));
    group.bench_function("paint a cached hud", |b| {
        b.iter(|| painter.paint(black_box(&canvas), &mut NoAssets));
    });
    group.finish();
}

criterion_group!(benches, paint);
criterion_main!(benches);
