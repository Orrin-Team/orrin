use orrin_ecs::World;

use super::Play;

/// The frame's clock. `delta_time` is game time, scaled by the transport's
/// time scale; the elapsed total and `unscaled_delta_time` are real time,
/// which is what the renderer's adaptation and the editor run on.
pub struct Time {
    // Accumulated as f64: repeatedly adding small f32 deltas to an f32 total
    // loses resolution as the total grows (~2 ms granularity after a few hours),
    // eventually dropping short frames entirely. Exposed as f32 at the getter.
    elapsed_time: f64,
    delta_time: f32,
    unscaled_delta_time: f32,
    frame_count: u64,
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}

impl Time {
    pub fn new() -> Self {
        Time {
            elapsed_time: 0.0,
            delta_time: 0.0,
            unscaled_delta_time: 0.0,
            frame_count: 0,
        }
    }

    pub fn update(&mut self, delta_time: f32, scale: f32) {
        self.elapsed_time += delta_time as f64;
        self.unscaled_delta_time = delta_time;
        self.delta_time = delta_time * scale;
        self.frame_count += 1;
    }

    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }

    pub fn unscaled_delta_time(&self) -> f32 {
        self.unscaled_delta_time
    }

    pub fn elapsed_time(&self) -> f32 {
        self.elapsed_time as f32
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

/// Start a frame `delta` real seconds long and return the game time it is
/// worth, which is what every system that simulates the game advances by.
pub fn advance_time(world: &World, delta: f32) -> f32 {
    let scale = world
        .get_resource::<Play>()
        .map_or(1.0, |play| play.time_scale());
    let mut time = world.resource_mut::<Time>();
    time.update(delta, scale);
    time.delta_time()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::PlayState;

    #[test]
    fn gameplay_sees_the_scaled_delta_and_the_clock_the_real_one() {
        let mut time = Time::new();
        time.update(0.02, 0.5);
        assert_eq!(time.delta_time(), 0.01);
        assert_eq!(time.unscaled_delta_time(), 0.02);
        time.update(0.02, 0.0);
        assert_eq!(time.delta_time(), 0.0);
        assert_eq!(time.unscaled_delta_time(), 0.02);
        assert!((time.elapsed_time() - 0.04).abs() < 1e-6);
        assert_eq!(time.frame_count(), 2);
    }

    #[test]
    fn a_frame_advances_by_the_transports_scale() {
        let mut world = World::new();
        world.insert_resource(Time::new());
        let mut play = Play::default();
        play.set(PlayState::Playing);
        play.set_time_scale(0.5);
        world.insert_resource(play);

        assert_eq!(advance_time(&world, 0.04), 0.02);
        assert_eq!(world.resource::<Time>().delta_time(), 0.02);
        assert_eq!(world.resource::<Time>().unscaled_delta_time(), 0.04);
    }

    /// A world with no transport — a story, a test — runs at one.
    #[test]
    fn a_world_with_no_transport_runs_at_one() {
        let mut world = World::new();
        world.insert_resource(Time::new());
        assert_eq!(advance_time(&world, 0.04), 0.04);
    }
}
