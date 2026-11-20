//! `screenshot`: what the viewport renders, as a PNG scaled to fit
//! `max_size` — an image is paid for in the model's context, so it travels no
//! larger than asked. `view` is `game`, the scene with the game's own UI over
//! it as a player sees it; `scene`, the scene alone; or `editor`, the window as
//! the developer sees it — the wireframe, the outline, the handles and the
//! editor's UI drawn a second time, each by its window twin's pipeline, the UI
//! from the primitives the window's overlay draws that frame. The UI's texture
//! changes are recorded after the frame, so an `editor` shot is drawn on a
//! frame that has none, within the deadline.
//!
//! **On the frame path only when asked.** The request asks the renderer for
//! one, the next frame draws the picture a second time into an image of its
//! own (`FrameConfig::screenshot`), and the frame after that frame's readback
//! hands the pixels here. A frame nobody asked of compiles and records the
//! frame it always did. The readback waits for its frame, as a capture does;
//! the scaling and the encoding run on a thread of their own, which sends the
//! PNG as a blob frame and then the answer, so the frame's thread never waits
//! on either. One screenshot at a time a connection.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Peer, Shared};
use crate::gfx::vulkan::frame::ShotView;

/// What `max_size` is when not given: enough to read a scene by, a fraction
/// of a full window.
const DEFAULT_SIZE: u32 = 1024;

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Params {
    /// The longest side the PNG may have, in pixels.
    max_size: Option<u32>,
    view: Option<String>,
}

/// The longest side a screenshot may be asked for at, and what it shows,
/// from `params`.
pub(super) fn asked(params: &Value) -> Result<(u32, ShotView), String> {
    let params: Params = super::answer::params_of(params)?;
    let view = match params.view.as_deref() {
        None | Some("game") => ShotView::Game,
        Some("scene") => ShotView::Scene,
        Some("editor") => ShotView::Editor,
        Some(other) => {
            return Err(format!(
                "`{other}` is not a view; `game`, `scene` or `editor`"
            ));
        }
    };
    Ok((
        params.max_size.unwrap_or(DEFAULT_SIZE).clamp(16, 4096),
        view,
    ))
}

/// A screenshot asked for, and whether the renderer has been asked yet.
pub(super) struct Shot {
    pub(super) peer: Peer,
    pub(super) id: u64,
    pub(super) max_size: u32,
    pub(super) view: ShotView,
    pub(super) raised: Instant,
    pub(super) asked: bool,
}

/// Scale, encode and send `pixels` for `shot` on a thread of its own.
pub(super) fn deliver(
    shared: &Arc<Shared>,
    shot: Shot,
    pixels: Arc<Vec<u8>>,
    extent: [u32; 2],
    readback: Duration,
) {
    let shared = shared.clone();
    let spawned = std::thread::Builder::new()
        .name("orrin-remote-shot".to_owned())
        .spawn(move || {
            let answer = encode(&pixels, extent, shot.max_size, readback).map(|(png, answer)| {
                if let Ok(frame) = orrin_remote::wire::blob(shot.id, &png) {
                    shared.send(shot.peer, frame);
                }
                answer
            });
            shared.send(shot.peer, super::encoded(shot.id, "screenshot", answer));
        });
    if let Err(error) = spawned {
        tracing::warn!("could not start a thread to encode a screenshot: {error}");
    }
}

/// The PNG of `pixels` at most `max_size` a side, and what to say about it.
fn encode(
    pixels: &[u8],
    [width, height]: [u32; 2],
    max_size: u32,
    readback: Duration,
) -> Result<(Vec<u8>, Value), String> {
    use image::ImageEncoder;
    let started = Instant::now();
    // Opaque: the frame's alpha is whatever the tonemap left, and a picture of
    // a scene has none.
    let rgb: Vec<u8> = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[r, g, b, _]| [r, g, b])
        .collect();
    let full = image::RgbImage::from_raw(width, height, rgb)
        .ok_or_else(|| "the frame read back was not the size it was drawn at".to_owned())?;
    let scale = (max_size as f32 / width.max(height) as f32).min(1.0);
    let (out_width, out_height) = (
        ((width as f32 * scale).round() as u32).max(1),
        ((height as f32 * scale).round() as u32).max(1),
    );
    let scaled = match scale < 1.0 {
        true => image::imageops::resize(
            &full,
            out_width,
            out_height,
            image::imageops::FilterType::Triangle,
        ),
        false => full,
    };
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(
            scaled.as_raw(),
            out_width,
            out_height,
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|error| format!("the screenshot could not be encoded: {error}"))?;
    let answer = json!({
        "width": out_width,
        "height": out_height,
        "frame": [width, height],
        "bytes": png.len(),
        "readback_ms": readback.as_secs_f64() * 1000.0,
        "encode_ms": started.elapsed().as_secs_f64() * 1000.0,
    });
    Ok((png, answer))
}
