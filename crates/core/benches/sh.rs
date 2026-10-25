//! The CPU half of an environment load (#73): the diffuse SH projection and the
//! sky measurement, both of which walk every texel of the source before the GPU
//! bake starts.
//!
//! Measured on the repository's own 2K capture, and on that capture doubled to
//! 4K by repetition so the larger size has the same content rather than a
//! synthetic sky's easy branch pattern. Each runs once on the pool and once in a
//! one-thread pool, which takes the `ORRIN_THREADS=1` path, so the two numbers
//! in a report always come from the same session.

use std::path::Path;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

use orrin_core::gfx::sh;
use orrin_core::scene::load_hdri;

struct Source {
    label: &'static str,
    pixels: Vec<f32>,
    extent: [u32; 2],
}

fn sources() -> Vec<Source> {
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets");
    let hdri = load_hdri(&assets, "meadow_2k.hdr").expect("assets/meadow_2k.hdr");
    let (width, height) = (hdri.width as usize, hdri.height as usize);

    let mut doubled = Vec::with_capacity(hdri.pixels.len() * 4);
    for y in 0..height * 2 {
        let row = &hdri.pixels[(y / 2) * width * 4..(y / 2 + 1) * width * 4];
        for texel in row.as_chunks::<4>().0 {
            doubled.extend_from_slice(texel);
            doubled.extend_from_slice(texel);
        }
    }

    vec![
        Source {
            label: "2k",
            extent: [hdri.width, hdri.height],
            pixels: hdri.pixels,
        },
        Source {
            label: "4k",
            extent: [hdri.width * 2, hdri.height * 2],
            pixels: doubled,
        },
    ]
}

fn bake(c: &mut Criterion) {
    orrin_core::threads::init();
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a one-thread pool");

    let mut group = c.benchmark_group("sh_bake");
    group.sample_size(10);
    for source in sources() {
        let texels = source.extent[0] as u64 * source.extent[1] as u64;
        group.throughput(Throughput::Elements(texels));
        let pixels = &source.pixels;
        let extent = source.extent;

        group.bench_function(format!("project_equirect/{}/pool", source.label), |b| {
            b.iter(|| sh::project_equirect(black_box(pixels), extent))
        });
        group.bench_function(format!("project_equirect/{}/serial", source.label), |b| {
            serial.install(|| b.iter(|| sh::project_equirect(black_box(pixels), extent)))
        });
        group.bench_function(format!("sky_luminance/{}/pool", source.label), |b| {
            b.iter(|| sh::sky_luminance(black_box(pixels), extent))
        });
        group.bench_function(format!("sky_luminance/{}/serial", source.label), |b| {
            serial.install(|| b.iter(|| sh::sky_luminance(black_box(pixels), extent)))
        });
    }
    group.finish();
}

criterion_group!(benches, bake);
criterion_main!(benches);
