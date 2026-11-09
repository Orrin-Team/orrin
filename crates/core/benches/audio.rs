//! What sound costs, on each of its two threads.
//!
//! - `update`: the main-thread half as `app.rs` runs it once a frame, with
//!   every voice a looping spatial `AudioSource` on an entity that moved since
//!   the last frame, so each one is spatialised and sends a new gain. The
//!   worst frame for this phase: a still source sends nothing.
//! - `mix_block`: the device thread's half, one block of [`BLOCK`] frames with
//!   that many voices sounding. A block is 5.3 ms of audio, so the time over
//!   5.3 ms is the fraction of one core the mix takes.
//!
//! Both read a real project: the clip is a WAV in a temporary directory,
//! decoded on the loading thread before anything is timed.

use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

use glam::{Mat4, Vec3};

use orrin_core::asset_db::AssetDatabase;
use orrin_core::audio::mixer::{BLOCK, Mixer, RATE};
use orrin_core::audio::{self, Audio};
use orrin_core::scene::{AudioSource, Camera, Play, PlayState, WorldTransform};
use orrin_ecs::{Entity, World};

const COUNTS: [usize; 3] = [16, 64, 256];

/// A one-second 16-bit mono WAV of a 440 Hz tone.
fn wav() -> Vec<u8> {
    let samples: Vec<i16> = (0..RATE)
        .map(|i| ((i as f32 / RATE as f32 * 440.0 * std::f32::consts::TAU).sin() * 8000.0) as i16)
        .collect();
    let data = (samples.len() * 2) as u32;
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&RATE.to_le_bytes());
    out.extend_from_slice(&(RATE * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

struct Scene {
    world: World,
    mixer: Mixer,
    speakers: Vec<Entity>,
    _dir: tempdir::Dir,
}

mod tempdir {
    /// A directory removed when dropped.
    pub struct Dir(pub std::path::PathBuf);

    impl Dir {
        pub fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("orrin-bench-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// `count` looping spatial sources around a listener at the origin, every
/// one sounding.
fn scene(count: usize) -> Scene {
    let dir = tempdir::Dir::new(&format!("audio-{count}"));
    std::fs::write(dir.0.join("tone.wav"), wav()).unwrap();
    let (db, issues) = AssetDatabase::open(&dir.0, dir.0.join(".cache"));
    assert!(issues.is_empty(), "{issues:?}");
    let clip = db.find("tone.wav").unwrap().id;

    let (audio, mixer) = Audio::with_mixer();
    let mut world = World::new();
    world.insert_resource(db);
    world.insert_resource(audio);
    world.insert_resource(Camera::default());
    let mut play = Play::default();
    play.set(PlayState::Playing);
    world.insert_resource(play);

    let speakers = (0..count)
        .map(|i| {
            let entity = world.spawn();
            world.insert(
                entity,
                WorldTransform::new(Mat4::from_translation(place(i, 0))),
            );
            world.insert(
                entity,
                AudioSource {
                    clip,
                    looping: true,
                    ..AudioSource::default()
                },
            );
            entity
        })
        .collect();
    let mut scene = Scene {
        world,
        mixer,
        speakers,
        _dir: dir,
    };

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        audio::update(&scene.world);
        scene.pull();
        let ready = scene.world.resource::<Audio>().clip_state(clip)
            == Some(orrin_core::asset_db::LoadState::Ready);
        if ready {
            audio::update(&scene.world);
            scene.pull();
            break;
        }
        assert!(Instant::now() < deadline, "the tone never decoded");
        std::thread::sleep(Duration::from_millis(1));
    }
    scene
}

/// Speaker `i`'s position on frame `frame`: a ring that turns.
fn place(i: usize, frame: u64) -> Vec3 {
    let angle = i as f32 * 0.37 + frame as f32 * 0.01;
    Vec3::new(angle.cos(), 0.0, angle.sin()) * (2.0 + (i % 7) as f32)
}

impl Scene {
    fn pull(&mut self) {
        for _ in 0..BLOCK * 2 {
            black_box(self.mixer.next());
        }
    }

    fn step(&mut self, frame: u64) {
        for (i, &entity) in self.speakers.iter().enumerate() {
            *self.world.get_mut::<WorldTransform>(entity).unwrap() =
                WorldTransform::new(Mat4::from_translation(place(i, frame)));
        }
    }
}

fn update(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_update");
    for count in COUNTS {
        let mut scene = scene(count);
        let mut frame = 0;
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(count.to_string(), |b| {
            b.iter_custom(|iterations| {
                let mut spent = Duration::ZERO;
                for _ in 0..iterations {
                    frame += 1;
                    scene.step(frame);
                    let start = Instant::now();
                    audio::update(&scene.world);
                    spent += start.elapsed();
                    scene.pull();
                }
                spent
            });
        });
    }
    group.finish();
}

fn mix_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_mix_block");
    for count in COUNTS {
        let mut scene = scene(count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(count.to_string(), |b| b.iter(|| scene.pull()));
    }
    group.finish();
}

criterion_group!(benches, update, mix_block);
criterion_main!(benches);
