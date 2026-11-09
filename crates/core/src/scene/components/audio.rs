use orrin_registry::{AssetId, Reflect};

/// A sound that plays from an entity: from where it stands, when `spatial`,
/// or into both ears alike when not. It sounds only while the game runs — see
/// [`audio`](crate::audio).
///
/// Distances are metres, like every other length. Inside `min_distance` the
/// source is at its full `volume`; past it the `rolloff` takes over, down to
/// `max_distance`, beyond which nothing changes.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct AudioSource {
    /// A WAV, Ogg Vorbis or MP3 file. Nil plays nothing.
    pub clip: AssetId,
    /// A linear gain, not decibels: 0.5 is half the amplitude.
    pub volume: f32,
    pub looping: bool,
    /// Starts when the game starts, or when the entity first appears in a
    /// running game.
    pub play_on_start: bool,
    pub spatial: bool,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: Rolloff,
}

impl Default for AudioSource {
    fn default() -> Self {
        Self {
            clip: AssetId::NIL,
            volume: 1.0,
            looping: false,
            play_on_start: true,
            spatial: true,
            min_distance: 1.0,
            max_distance: 50.0,
            rolloff: Rolloff::Inverse,
        }
    }
}

/// How a spatial source quietens between its minimum and maximum distance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum Rolloff {
    /// Halves with every doubling of distance, which is what a point source
    /// does in open air. Never reaches silence; it stops falling at the
    /// maximum distance.
    #[default]
    Inverse,
    /// Falls in a straight line to silence at the maximum distance, for a
    /// sound that must end at a known radius.
    Linear,
}

/// Where a scene's spatial sound is heard from. The first one found is the
/// listener; with none, the camera the frame is drawn from is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub struct AudioListener;
