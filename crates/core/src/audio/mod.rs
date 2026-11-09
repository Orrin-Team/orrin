//! Sound: clips played flat or from a point in the world, a music channel,
//! and the [`AudioSource`] and [`AudioListener`] components a scene authors
//! them with.
//!
//! # Two threads, one direction
//!
//! [`Audio`] is a world resource on the main thread and owns every decision:
//! which voices exist, which clip each plays, and the gain each ear gets this
//! frame. The [`mixer`] runs on the device's thread and owns only samples. It
//! is told what to do through a command exchange and answers with what has
//! finished, and nothing the main thread does can make it wait — see its
//! module docs for how.
//!
//! **Spatialisation is a main-thread computation.** Once a frame, after the
//! transforms are propagated, each voice's position against the listener
//! becomes a left and a right gain ([`spatial`]), and only the pair crosses
//! the thread; the mixer ramps to it over a block. That keeps the audio
//! thread free of the world entirely, and the frame rate — far above the 10
//! to 20 Hz at which a moving source's gain is audible as steps — is the
//! update rate of every position.
//!
//! # What plays when
//!
//! Only a running game makes a sound. Edit mode runs no game code, and the
//! frame that leaves play stops every voice, because the world Stop restores
//! is not the one they were playing in. Pause holds the mix where it is.
//!
//! The one exception is the editor's [preview](Audio::preview): a single clip,
//! flat, that plays in any mode so a sound can be auditioned before a source
//! is given it. It belongs to the editor rather than to either world, so any
//! move of the transport ends it.
//! An [`AudioSource`] with `play_on_start` starts the first frame the game
//! runs with it in the world — at Play, or when it is spawned later.
//!
//! The listener is the first [`AudioListener`] in the world, and with none the
//! camera the frame is drawn from, which in play is the active camera.
//!
//! # Clips, music, and the device
//!
//! A sound is decoded whole, once, on a thread of its own ([`load`]), and
//! every voice of it shares the samples. A source in the scene is decoded the
//! frame it is first seen, edit mode included, so pressing Play does not wait
//! on a decode. Music is the exception: it is decoded as it plays, because a
//! track decoded whole is tens of megabytes. A changed file is decoded again;
//! a voice already playing keeps the samples it started with.
//!
//! The device opens on its own thread ([`device`]) and may not exist. Without
//! one, every call still answers, every sound is accepted, and every sound
//! ends the next frame: a game cannot tell the difference except by ear.
//!
//! # Not here
//!
//! No Doppler, no occlusion, no reverb, no HRTF, no buses beyond master and
//! music. Each is a stage after the spatial gain rather than a change to it.

pub mod clip;
mod device;
pub mod load;
pub mod mixer;
pub mod spatial;

use glam::Vec3;
use orrin_ecs::{Entity, FxHashMap, World};
use orrin_registry::AssetId;

use self::load::{ClipState, Clips, Opened};
use self::mixer::{Body, Command, Mixer, Shared};
pub use self::spatial::{Falloff, Listener};
use crate::asset_db::AssetDatabase;
use crate::scene::{AudioListener, AudioSource, Camera, Play, PlayState, WorldTransform};

/// A playing sound, as a script holds it. The bits are never zero, which is
/// how a refusal is spelled across the ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SoundId(u64);

impl SoundId {
    pub fn to_bits(self) -> u64 {
        self.0
    }

    pub fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

/// Where a sound is heard from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Emitter {
    /// In both ears alike, wherever the listener is.
    Flat,
    /// A fixed point in the world.
    At(Vec3),
    /// Wherever this entity is, for as long as it exists.
    On(Entity),
}

/// A request to play a clip.
#[derive(Clone, Copy, Debug)]
pub struct Sound {
    pub clip: AssetId,
    pub emitter: Emitter,
    pub volume: f32,
    pub looping: bool,
    pub falloff: Falloff,
}

impl Sound {
    pub fn flat(clip: AssetId) -> Self {
        Self {
            clip,
            emitter: Emitter::Flat,
            volume: 1.0,
            looping: false,
            falloff: Falloff::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    Script,
    Source(Entity),
    Music,
    Preview,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Stage {
    /// Its clip is decoding, or its stream opening.
    Waiting,
    Opening,
    /// In the mixer, last sent these gains.
    Sounding([f32; 2]),
}

struct Voice {
    clip: AssetId,
    emitter: Emitter,
    origin: Origin,
    /// What the script asked for, on top of a source's own volume.
    volume: f32,
    /// A source's own volume; 1 otherwise.
    base: f32,
    looping: bool,
    falloff: Falloff,
    stage: Stage,
}

enum Output {
    /// Nothing will ever hear it.
    Silent,
    /// A [`Mixer`] handed out by [`Audio::with_mixer`] is pulled by whoever
    /// holds it.
    Pulled,
    Device(device::Device),
}

/// The engine's sound, as a world resource.
pub struct Audio {
    shared: Shared,
    output: Output,
    last_id: u64,
    voices: FxHashMap<u64, Voice>,
    commands: Vec<Command>,
    clips: Clips,
    /// The voice each source entity is playing.
    sources: FxHashMap<Entity, u64>,
    /// Source entities the running game has already started, or chosen not
    /// to: `play_on_start` is a start, not a loop.
    seen: std::collections::HashSet<Entity, orrin_ecs::FxBuildHasher>,
    music: Option<u64>,
    preview: Option<u64>,
    master_volume: f32,
    music_volume: f32,
    state: PlayState,
    opened: Vec<Opened>,
    ended: Vec<u64>,
    /// Swapped with the exchange's, so the vector the audio thread appends to
    /// keeps its capacity.
    retired: Vec<Body>,
}

impl Default for Audio {
    /// Silent until [`open_device`](Self::open_device).
    fn default() -> Self {
        Self::new(mixer::shared(), Output::Silent)
    }
}

impl Audio {
    fn new(shared: Shared, output: Output) -> Self {
        Self {
            shared,
            output,
            last_id: 0,
            voices: FxHashMap::default(),
            commands: Vec::new(),
            clips: Clips::default(),
            sources: FxHashMap::default(),
            seen: Default::default(),
            music: None,
            preview: None,
            master_volume: 1.0,
            music_volume: 1.0,
            state: PlayState::Editing,
            opened: Vec::new(),
            ended: Vec::new(),
            retired: Vec::with_capacity(mixer::VOICES),
        }
    }

    /// Audio whose mix is pulled by the caller rather than a device: a test,
    /// or a benchmark.
    pub fn with_mixer() -> (Self, Mixer) {
        let shared = mixer::shared();
        (
            Self::new(shared.clone(), Output::Pulled),
            Mixer::new(shared),
        )
    }

    /// A sound the engine builds rather than imports, under an id named the
    /// way the demo library's meshes are. Ready at once and never decoded,
    /// since there is no file behind it.
    pub fn insert_clip(&mut self, id: AssetId, clip: clip::Clip) {
        self.clips.insert(id, clip);
    }

    /// Start opening the default output device. Returns at once.
    pub fn open_device(&mut self) {
        self.output = Output::Device(device::Device::open(Mixer::new(self.shared.clone())));
    }

    pub fn play(&mut self, sound: Sound) -> SoundId {
        self.start(Voice {
            clip: sound.clip,
            emitter: sound.emitter,
            origin: Origin::Script,
            volume: sanitised(sound.volume),
            base: 1.0,
            looping: sound.looping,
            falloff: sound.falloff,
            stage: Stage::Waiting,
        })
    }

    /// Play `entity`'s [`AudioSource`] from the start, stopping what it was
    /// playing. `None` for a source with no clip.
    pub fn play_source(&mut self, entity: Entity, source: &AudioSource) -> Option<SoundId> {
        self.stop_source(entity);
        if source.clip.is_nil() {
            return None;
        }
        let id = self.start(Voice {
            clip: source.clip,
            emitter: if source.spatial {
                Emitter::On(entity)
            } else {
                Emitter::Flat
            },
            origin: Origin::Source(entity),
            volume: 1.0,
            base: sanitised(source.volume),
            looping: source.looping,
            falloff: falloff_of(source),
            stage: Stage::Waiting,
        });
        self.sources.insert(entity, id.0);
        Some(id)
    }

    /// Stop what `entity`'s source is playing. False if it was not.
    pub fn stop_source(&mut self, entity: Entity) -> bool {
        self.sources
            .get(&entity)
            .copied()
            .is_some_and(|id| self.stop(SoundId(id)))
    }

    /// Stop `id`, fading it over a few milliseconds. False if it had already
    /// ended.
    pub fn stop(&mut self, id: SoundId) -> bool {
        let Some(voice) = self.voices.remove(&id.0) else {
            return false;
        };
        if let Stage::Sounding(_) = voice.stage {
            self.commands.push(Command::Stop { id: id.0 });
        }
        self.forget(id.0, voice.origin);
        true
    }

    /// A linear gain on top of everything else that sets its loudness.
    pub fn set_volume(&mut self, id: SoundId, volume: f32) -> bool {
        self.voices
            .get_mut(&id.0)
            .map(|voice| voice.volume = sanitised(volume))
            .is_some()
    }

    pub fn is_playing(&self, id: SoundId) -> bool {
        self.voices.contains_key(&id.0)
    }

    /// Whether `entity`'s [`AudioSource`] is playing, or waiting on its clip
    /// to start.
    pub fn is_source_playing(&self, entity: Entity) -> bool {
        self.sources.contains_key(&entity)
    }

    /// Play `clip` on the music channel, replacing what was there.
    pub fn play_music(&mut self, clip: AssetId, looping: bool) -> SoundId {
        self.stop_music();
        let id = self.start(Voice {
            clip,
            emitter: Emitter::Flat,
            origin: Origin::Music,
            volume: 1.0,
            base: 1.0,
            looping,
            falloff: Falloff::default(),
            stage: Stage::Waiting,
        });
        self.music = Some(id.0);
        id
    }

    pub fn stop_music(&mut self) {
        if let Some(id) = self.music {
            self.stop(SoundId(id));
        }
    }

    /// Audition `clip` for the editor, replacing the last preview. Heard
    /// whatever the transport says, and never by the master volume a game set.
    pub fn preview(&mut self, clip: AssetId) {
        self.stop_preview();
        let id = self.start(Voice {
            clip,
            emitter: Emitter::Flat,
            origin: Origin::Preview,
            volume: 1.0,
            base: 1.0,
            looping: false,
            falloff: Falloff::default(),
            stage: Stage::Waiting,
        });
        self.preview = Some(id.0);
    }

    pub fn stop_preview(&mut self) {
        if let Some(id) = self.preview {
            self.stop(SoundId(id));
        }
    }

    /// The clip the preview is playing, or waiting on to decode.
    pub fn previewing(&self) -> Option<AssetId> {
        self.preview
            .and_then(|id| self.voices.get(&id))
            .map(|voice| voice.clip)
    }

    fn start(&mut self, voice: Voice) -> SoundId {
        self.last_id += 1;
        self.voices.insert(self.last_id, voice);
        SoundId(self.last_id)
    }

    /// Drop whatever pointed at voice `id`.
    fn forget(&mut self, id: u64, origin: Origin) {
        match origin {
            Origin::Source(entity) if self.sources.get(&entity) == Some(&id) => {
                self.sources.remove(&entity);
            }
            Origin::Music if self.music == Some(id) => self.music = None,
            Origin::Preview if self.preview == Some(id) => self.preview = None,
            _ => {}
        }
    }

    fn stop_all(&mut self) {
        for (id, voice) in self.voices.drain() {
            if let Stage::Sounding(_) = voice.stage {
                self.commands.push(Command::Stop { id });
            }
        }
        self.sources.clear();
        self.seen.clear();
        self.music = None;
        self.preview = None;
    }

    /// Follow the transport: everything stops when the game does, and the
    /// mix holds while it is paused.
    fn transport(&mut self, state: PlayState) {
        if state == self.state {
            return;
        }
        self.stop_preview();
        match state {
            PlayState::Editing => {
                self.stop_all();
                self.commands.push(Command::Pause(false));
            }
            PlayState::Paused => self.commands.push(Command::Pause(true)),
            PlayState::Playing => self.commands.push(Command::Pause(false)),
        }
        self.state = state;
    }

    /// Whether anything will ever hear what is sent. A device that failed to
    /// open is silence from then on.
    fn heard(&mut self) -> bool {
        match &self.output {
            Output::Silent => false,
            Output::Pulled => true,
            Output::Device(device) if device.failed() => {
                self.output = Output::Silent;
                self.shared
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .commands
                    .clear();
                false
            }
            Output::Device(_) => true,
        }
    }

    /// Take what the mixer has finished with, and drop the bodies here.
    fn collect(&mut self) {
        let mut exchange = self
            .shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ended.append(&mut exchange.finished);
        std::mem::swap(&mut exchange.retired, &mut self.retired);
        drop(exchange);
        self.retired.clear();
        for id in std::mem::take(&mut self.ended) {
            if let Some(voice) = self.voices.remove(&id) {
                self.forget(id, voice.origin);
            }
        }
    }

    /// Start every source the running game has not seen yet, once.
    fn start_sources(&mut self, world: &World) {
        let mut starting = Vec::new();
        world.query::<&AudioSource>().for_each(|entity, source| {
            if self.seen.insert(entity) && source.play_on_start {
                starting.push((entity, *source));
            }
        });
        for (entity, source) in starting {
            self.play_source(entity, &source);
        }
    }

    /// Bring every voice in line with the world and hand the mixer this
    /// frame's gains.
    fn mix(&mut self, world: &World, db: Option<&AssetDatabase>) {
        let listener = listener(world);
        let master = self.master_volume;
        let music = self.music_volume;
        for (&id, voice) in &mut self.voices {
            if let Origin::Source(entity) = voice.origin {
                match world.get::<AudioSource>(entity) {
                    Some(source) => {
                        voice.base = sanitised(source.volume);
                        voice.falloff = falloff_of(&source);
                    }
                    None => {
                        self.ended.push(id);
                        continue;
                    }
                }
            }
            let Some(gains) = voice.gains(world, &listener, master, music) else {
                self.ended.push(id);
                continue;
            };
            match voice.stage {
                Stage::Waiting if voice.origin == Origin::Music => {
                    voice.stage = if self.clips.open(id, voice.clip, voice.looping, db) {
                        Stage::Opening
                    } else {
                        self.ended.push(id);
                        Stage::Waiting
                    };
                }
                Stage::Waiting => match self.clips.state(voice.clip) {
                    Some(ClipState::Ready(clip)) => {
                        self.commands.push(Command::Play {
                            id,
                            body: Body::Clip {
                                clip: clip.clone(),
                                looping: voice.looping,
                            },
                            gains,
                            downmix: voice.emitter != Emitter::Flat,
                        });
                        voice.stage = Stage::Sounding(gains);
                    }
                    Some(ClipState::Loading) => {}
                    Some(ClipState::Failed) => self.ended.push(id),
                    None => self.clips.request(voice.clip, db),
                },
                Stage::Opening => {}
                Stage::Sounding(sent) if sent != gains => {
                    self.commands.push(Command::Gains { id, gains });
                    voice.stage = Stage::Sounding(gains);
                }
                Stage::Sounding(_) => {}
            }
        }
        for id in std::mem::take(&mut self.ended) {
            self.stop(SoundId(id));
        }

        for opened in std::mem::take(&mut self.opened) {
            let voice = self.voices.get_mut(&opened.voice);
            match (voice, opened.stream) {
                (Some(voice), Some(stream)) if voice.stage == Stage::Opening => {
                    let gains = voice
                        .gains(world, &listener, master, music)
                        .unwrap_or_default();
                    self.commands.push(Command::Play {
                        id: opened.voice,
                        body: Body::Stream(stream),
                        gains,
                        downmix: false,
                    });
                    voice.stage = Stage::Sounding(gains);
                }
                (Some(_), None) => {
                    self.stop(SoundId(opened.voice));
                }
                _ => {}
            }
        }
    }

    /// Hand this frame's commands over — or, with nothing to hear them, end
    /// every sound now.
    fn flush(&mut self) {
        if self.heard() {
            let mut exchange = self
                .shared
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            exchange.commands.append(&mut self.commands);
            return;
        }
        self.commands.clear();
        let ended: Vec<u64> = self.voices.keys().copied().collect();
        for id in ended {
            self.stop(SoundId(id));
        }
        self.commands.clear();
    }

    pub fn set_music_volume(&mut self, volume: f32) {
        self.music_volume = sanitised(volume);
    }

    pub fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = sanitised(volume);
    }

    /// Whether sound `id` is decoded, for `AssetRef.State`; `None` for one
    /// nothing has asked to play.
    pub fn clip_state(&self, id: AssetId) -> Option<crate::asset_db::LoadState> {
        use crate::asset_db::LoadState;
        self.clips.state(id).map(|state| match state {
            ClipState::Loading => LoadState::Loading,
            ClipState::Ready(_) => LoadState::Ready,
            ClipState::Failed => LoadState::Failed,
        })
    }
}

/// A volume the mix can use: finite and not negative.
fn sanitised(volume: f32) -> f32 {
    if volume.is_finite() {
        volume.max(0.0)
    } else {
        0.0
    }
}

/// Once a frame, after the last propagation: start, follow and end voices,
/// and hand the mixer this frame's gains.
pub fn update(world: &World) {
    let Some(mut audio) = world.get_resource_mut::<Audio>() else {
        return;
    };
    let state = world
        .get_resource::<Play>()
        .map_or(PlayState::Editing, |play| play.state());
    audio.transport(state);
    audio.collect();

    let mut db = world.get_resource_mut::<AssetDatabase>();
    if let Some(db) = db.as_deref_mut() {
        for id in db.take_audio_changes() {
            audio.clips.changed(id, db);
        }
    }
    let db = db.as_deref();
    let mut opened = std::mem::take(&mut audio.opened);
    audio.clips.receive(&mut opened);
    audio.opened = opened;

    // With nothing to hear it, every sound ends in `flush` this frame, so
    // there is nothing to decode, start or spatialise.
    if audio.heard() {
        world.query::<&AudioSource>().for_each(|_, source| {
            if !source.clip.is_nil() {
                audio.clips.request(source.clip, db);
            }
        });
        if state == PlayState::Playing {
            audio.start_sources(world);
        }
        audio.mix(world, db);
    }
    audio.flush();
}

impl Voice {
    /// This frame's left and right gains, or `None` once what it plays from
    /// is gone.
    fn gains(
        &self,
        world: &World,
        listener: &Listener,
        master: f32,
        music: f32,
    ) -> Option<[f32; 2]> {
        let mut volume = self.volume * self.base;
        match self.origin {
            Origin::Preview => {}
            Origin::Music => volume *= master * music,
            Origin::Script | Origin::Source(_) => volume *= master,
        }
        let position = match self.emitter {
            Emitter::Flat => return Some([volume; 2]),
            Emitter::At(position) => position,
            Emitter::On(entity) => world.get::<WorldTransform>(entity)?.translation(),
        };
        let [left, right] = spatial::gains(listener, position, self.falloff);
        Some([left * volume, right * volume])
    }
}

fn falloff_of(source: &AudioSource) -> Falloff {
    Falloff {
        min_distance: source.min_distance,
        max_distance: source.max_distance,
        rolloff: source.rolloff,
    }
}

/// The first [`AudioListener`], else the camera the frame is drawn from.
fn listener(world: &World) -> Listener {
    let mut found = None;
    world
        .query::<(&AudioListener, &WorldTransform)>()
        .find(|_, (_, transform)| {
            found = Some(Listener::from_matrix(transform.0));
            true
        });
    found
        .or_else(|| {
            world
                .get_resource::<Camera>()
                .map(|camera| Listener::from_camera(&camera))
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat};

    use super::clip::Clip;
    use super::mixer::{BLOCK, RATE};
    use super::*;

    struct Rig {
        world: World,
        mixer: Mixer,
        clip: AssetId,
    }

    /// A world in play, with one looping-friendly clip of constant level
    /// decoded and ready.
    fn rig() -> Rig {
        let (mut audio, mixer) = Audio::with_mixer();
        let clip = AssetId::new();
        audio
            .clips
            .insert(clip, Clip::new(1, RATE, vec![0.5; RATE as usize].into()));
        let mut world = World::new();
        world.insert_resource(audio);
        let mut play = Play::default();
        play.set(PlayState::Playing);
        world.insert_resource(play);
        world.insert_resource(Camera {
            position: Vec3::ZERO,
            target: Vec3::NEG_Z,
            up: Vec3::Y,
            ..Camera::default()
        });
        Rig { world, mixer, clip }
    }

    impl Rig {
        fn audio(&self) -> std::cell::RefMut<'_, Audio> {
            self.world.resource_mut::<Audio>()
        }

        fn frame(&mut self) -> [f32; 2] {
            update(&self.world);
            self.pull(BLOCK)
        }

        /// The last frame of `frames` of mix: what the ears hear once any
        /// ramp has finished.
        fn pull(&mut self, frames: usize) -> [f32; 2] {
            let mut last = [0.0; 2];
            for _ in 0..frames {
                last = [self.mixer.next().unwrap(), self.mixer.next().unwrap()];
            }
            last
        }

        fn set_state(&mut self, state: PlayState) {
            self.world.resource_mut::<Play>().set(state);
        }

        fn spawn_at(&mut self, position: Vec3) -> Entity {
            let entity = self.world.spawn();
            self.world.insert(
                entity,
                WorldTransform::new(Mat4::from_translation(position)),
            );
            entity
        }
    }

    #[test]
    fn a_flat_sound_plays_its_clip_and_ends() {
        let mut rig = rig();
        let id = rig.audio().play(Sound::flat(rig.clip));
        assert_ne!(id.to_bits(), 0);
        assert!(rig.audio().is_playing(id));
        assert_eq!(rig.frame(), [0.5, 0.5]);

        rig.pull(RATE as usize);
        update(&rig.world);
        rig.pull(BLOCK);
        update(&rig.world);
        assert!(!rig.audio().is_playing(id));
        assert!(!rig.audio().stop(id), "an ended sound has nothing to stop");
    }

    #[test]
    fn a_sound_on_an_entity_follows_it_from_ear_to_ear() {
        let mut rig = rig();
        let entity = rig.spawn_at(Vec3::new(1.0, 0.0, 0.0));
        let sound = Sound {
            emitter: Emitter::On(entity),
            looping: true,
            ..Sound::flat(rig.clip)
        };
        rig.audio().play(sound);
        let [left, right] = rig.frame();
        assert!(
            left.abs() < 1e-6 && (right - 0.5).abs() < 1e-6,
            "{left} {right}"
        );

        *rig.world.get_mut::<WorldTransform>(entity).unwrap() =
            WorldTransform::new(Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0)));
        rig.frame();
        let [left, right] = rig.frame();
        assert!(
            (left - 0.5).abs() < 1e-6 && right.abs() < 1e-6,
            "{left} {right}"
        );
    }

    #[test]
    fn an_audio_listener_is_heard_from_rather_than_the_camera() {
        let mut rig = rig();
        let listener = rig.world.spawn();
        rig.world.insert(listener, AudioListener);
        rig.world.insert(
            listener,
            WorldTransform::new(Mat4::from_quat(Quat::from_rotation_y(std::f32::consts::PI))),
        );
        let sound = Sound {
            emitter: Emitter::At(Vec3::new(1.0, 0.0, 0.0)),
            looping: true,
            ..Sound::flat(rig.clip)
        };
        rig.audio().play(sound);
        let [left, right] = rig.frame();
        assert!(left > 0.49 && right.abs() < 1e-6, "{left} {right}");
    }

    #[test]
    fn a_source_with_play_on_start_starts_once_when_the_game_runs() {
        let mut rig = rig();
        rig.set_state(PlayState::Editing);
        let entity = rig.spawn_at(Vec3::ZERO);
        let source = AudioSource {
            clip: rig.clip,
            looping: true,
            spatial: false,
            volume: 0.5,
            ..AudioSource::default()
        };
        rig.world.insert(entity, source);
        assert_eq!(rig.frame(), [0.0; 2], "edit mode makes no sound");

        rig.set_state(PlayState::Playing);
        assert_eq!(rig.frame(), [0.25; 2]);
        rig.frame();
        assert_eq!(rig.audio().voices.len(), 1, "started once, not every frame");

        assert!(rig.audio().stop_source(entity));
        rig.frame();
        assert_eq!(rig.frame(), [0.0; 2]);
        assert!(
            rig.audio().voices.is_empty(),
            "a stopped source stays stopped"
        );
    }

    /// A source's volume is read every frame, so the inspector's slider
    /// moves a sound that is playing; a script's volume multiplies it.
    #[test]
    fn a_sources_volume_and_a_scripts_multiply() {
        let mut rig = rig();
        let entity = rig.spawn_at(Vec3::ZERO);
        let source = AudioSource {
            clip: rig.clip,
            looping: true,
            spatial: false,
            play_on_start: false,
            ..AudioSource::default()
        };
        rig.world.insert(entity, source);
        let id = rig.audio().play_source(entity, &source).unwrap();
        assert_eq!(rig.frame(), [0.5; 2]);

        rig.world.get_mut::<AudioSource>(entity).unwrap().volume = 0.5;
        assert!(rig.audio().set_volume(id, 0.5));
        rig.frame();
        assert_eq!(rig.frame(), [0.125; 2]);
    }

    #[test]
    fn leaving_play_silences_everything_and_pause_holds_it() {
        let mut rig = rig();
        let sound = Sound {
            looping: true,
            ..Sound::flat(rig.clip)
        };
        let id = rig.audio().play(sound);
        assert_eq!(rig.frame(), [0.5; 2]);

        rig.set_state(PlayState::Paused);
        assert_eq!(rig.frame(), [0.0; 2]);
        rig.set_state(PlayState::Playing);
        assert_eq!(rig.frame(), [0.5; 2]);

        rig.set_state(PlayState::Editing);
        rig.frame();
        assert_eq!(rig.frame(), [0.0; 2]);
        assert!(!rig.audio().is_playing(id));
    }

    /// The editor auditions a clip with no game running, and a second preview
    /// replaces the first rather than playing over it.
    #[test]
    fn the_editor_previews_one_clip_at_a_time_without_a_running_game() {
        let mut rig = rig();
        rig.set_state(PlayState::Editing);
        let quieter = AssetId::new();
        rig.audio().clips.insert(
            quieter,
            Clip::new(1, RATE, vec![0.25; RATE as usize].into()),
        );

        rig.audio().preview(rig.clip);
        assert_eq!(rig.audio().previewing(), Some(rig.clip));
        assert_eq!(rig.frame(), [0.5; 2]);

        rig.audio().preview(quieter);
        assert_eq!(rig.audio().previewing(), Some(quieter));
        rig.frame();
        assert_eq!(rig.frame(), [0.25; 2]);

        rig.audio().stop_preview();
        assert_eq!(rig.audio().previewing(), None);
        rig.frame();
        assert_eq!(rig.frame(), [0.0; 2]);
    }

    #[test]
    fn a_preview_that_reaches_its_end_is_over() {
        let mut rig = rig();
        rig.set_state(PlayState::Editing);
        rig.audio().preview(rig.clip);
        rig.frame();
        assert_eq!(rig.audio().previewing(), Some(rig.clip));
        rig.pull(RATE as usize);
        update(&rig.world);
        rig.pull(BLOCK);
        update(&rig.world);
        assert_eq!(rig.audio().previewing(), None);
    }

    /// Play, Pause and Stop each end a preview: it belongs to the editor, and
    /// the game's mix starts without it.
    #[test]
    fn a_preview_stops_when_the_transport_moves() {
        let mut rig = rig();
        rig.set_state(PlayState::Editing);
        rig.audio().preview(rig.clip);
        assert_eq!(rig.frame(), [0.5; 2]);

        rig.set_state(PlayState::Playing);
        rig.frame();
        assert_eq!(rig.frame(), [0.0; 2]);
        assert_eq!(rig.audio().previewing(), None);
    }

    #[test]
    fn a_despawned_entity_takes_its_sound_with_it() {
        let mut rig = rig();
        let entity = rig.spawn_at(Vec3::new(0.0, 0.0, -1.0));
        let sound = Sound {
            emitter: Emitter::On(entity),
            looping: true,
            ..Sound::flat(rig.clip)
        };
        let id = rig.audio().play(sound);
        rig.frame();
        rig.world.despawn(entity);
        rig.frame();
        assert!(!rig.audio().is_playing(id));
        assert_eq!(rig.frame(), [0.0; 2]);
    }

    /// With no device there is nothing to wait for: the sound is accepted and
    /// over by the next frame.
    #[test]
    fn without_a_device_a_sound_ends_the_frame_after_it_starts() {
        let mut world = World::new();
        world.insert_resource(Audio::default());
        let mut play = Play::default();
        play.set(PlayState::Playing);
        world.insert_resource(play);
        let id = world
            .resource_mut::<Audio>()
            .play(Sound::flat(AssetId::new()));
        assert_ne!(id.to_bits(), 0);
        update(&world);
        assert!(!world.resource::<Audio>().is_playing(id));
    }

    /// The real device, which CI has none of: a tenth of a second of silence
    /// ends only if the device is pulling the mix. Run with
    /// `cargo test -p orrin-core --lib audio::tests::the_device -- --ignored`.
    #[test]
    #[ignore = "needs an audio output device"]
    fn the_device_pulls_the_mix_until_a_sound_ends() {
        let mut audio = Audio::default();
        audio.open_device();
        let clip = AssetId::new();
        audio.clips.insert(
            clip,
            Clip::new(1, RATE, vec![0.0; RATE as usize / 10].into()),
        );
        let mut world = World::new();
        world.insert_resource(audio);
        let mut play = Play::default();
        play.set(PlayState::Playing);
        world.insert_resource(play);
        let id = world.resource_mut::<Audio>().play(Sound::flat(clip));

        let start = std::time::Instant::now();
        while world.resource::<Audio>().is_playing(id) {
            assert!(
                start.elapsed() < std::time::Duration::from_secs(5),
                "the device never finished a 100 ms sound"
            );
            update(&world);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let took = start.elapsed();
        assert!(matches!(
            world.resource::<Audio>().output,
            Output::Device(_)
        ));
        assert!(
            took >= std::time::Duration::from_millis(90),
            "ended in {took:?}"
        );
        println!("a 100 ms sound ended after {took:?}");
    }

    /// The whole path a project's file takes: the database names it, the
    /// loading thread opens it, and the mix plays it — and a second track
    /// replaces the first.
    #[test]
    fn music_streams_from_the_project_and_replaces_itself() {
        let dir = crate::asset_db::tests::Dir::new("audio-music");
        dir.write(
            "music/theme.wav",
            &clip::tests::wav(1, RATE, &vec![16384; RATE as usize]),
        );
        dir.write("music/broken.wav", b"not a sound");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let theme = db.find("music/theme.wav").unwrap().id;
        let broken = db.find("music/broken.wav").unwrap().id;
        let mut rig = rig();
        rig.world.insert_resource(db);

        let first = rig.audio().play_music(theme, true);
        let mut heard = [0.0; 2];
        for _ in 0..500 {
            heard = rig.frame();
            if heard[0] > 0.0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!((heard[0] - 0.5).abs() < 1e-3, "{heard:?}");

        let second = rig.audio().play_music(broken, true);
        assert!(!rig.audio().is_playing(first));
        for _ in 0..500 {
            rig.frame();
            if !rig.audio().is_playing(second) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            !rig.audio().is_playing(second),
            "a file that will not open ends"
        );
        assert_eq!(rig.frame(), [0.0; 2]);
    }

    /// A scene's sources are decoded before Play, so the first frame of the
    /// game does not wait on a decode; a sound that will not decode is a
    /// failed asset rather than a voice waiting for ever.
    #[test]
    fn a_scenes_sources_decode_in_edit_mode_and_a_bad_file_fails() {
        let dir = crate::asset_db::tests::Dir::new("audio-preload");
        dir.write("sfx/hum.wav", &clip::tests::wav(1, RATE, &[100; 64]));
        dir.write("sfx/broken.wav", b"not a sound");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let hum = db.find("sfx/hum.wav").unwrap().id;
        let broken = db.find("sfx/broken.wav").unwrap().id;
        let mut rig = rig();
        rig.world.insert_resource(db);
        rig.set_state(PlayState::Editing);
        for clip in [hum, broken] {
            let entity = rig.world.spawn();
            rig.world.insert(
                entity,
                AudioSource {
                    clip,
                    ..AudioSource::default()
                },
            );
        }

        use crate::asset_db::LoadState;
        for _ in 0..500 {
            update(&rig.world);
            let audio = rig.audio();
            if audio.clip_state(hum) != Some(LoadState::Loading)
                && audio.clip_state(broken) != Some(LoadState::Loading)
            {
                break;
            }
            drop(audio);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(rig.audio().clip_state(hum), Some(LoadState::Ready));
        assert_eq!(rig.audio().clip_state(broken), Some(LoadState::Failed));
        assert!(rig.audio().voices.is_empty());
    }
}
