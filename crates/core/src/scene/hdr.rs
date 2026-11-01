use orrin_registry::Reflect;

/// How the frame's radiance becomes a displayable image.
///
/// Exposure is expressed the way a camera expresses it — in EV100, the exposure
/// value a 100-ISO meter would report — rather than as a linear multiplier. That
/// is what lets metering and manual control share one number: auto-exposure
/// measures an EV100 from the scene, manual states one outright, and
/// [`exposure_compensation`](Self::exposure_compensation) shifts either by a
/// count of stops. A stop is a doubling, so the dial behaves the same at noon
/// and at midnight, which a multiplier does not.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct HdrSettings {
    /// Meter the frame and adapt to it. Off makes
    /// [`manual_ev100`](Self::manual_ev100) the whole exposure.
    pub auto_exposure: bool,
    /// Stops added on top of whatever exposure is in force. Positive is
    /// brighter. This is the photographer's ±EV dial, and the one control worth
    /// touching per scene once metering works.
    pub exposure_compensation: f32,
    /// The exposure used when [`auto_exposure`](Self::auto_exposure) is off.
    pub manual_ev100: f32,
    /// The log2-luminance window the histogram spans, in cd/m². Everything
    /// outside it lands in an end bin, so a window that excludes the scene pins
    /// the metering to one end.
    ///
    /// Real luminance, because the frame is: lights are photometric, so a
    /// window stated in physical units means what it says. See
    /// [`Light`](super::Light) for the unit each kind of light carries.
    pub min_log_luminance: f32,
    pub max_log_luminance: f32,
    /// Seconds to cover ~63% of the distance to a newly measured luminance.
    /// Brightening is the faster of the two because eyes are.
    pub adaptation_brighten: f32,
    pub adaptation_darken: f32,
}

impl Default for HdrSettings {
    fn default() -> Self {
        Self {
            auto_exposure: true,
            exposure_compensation: 0.0,
            // Overcast daylight, which is what `Light::default` describes. The
            // sunny-16 exterior is about 15 and a lit interior about 7, so this
            // sits within a few stops of either — the range the compensation
            // dial covers comfortably.
            manual_ev100: 12.0,
            // Twenty-five stops of real luminance, 0.004 to about 130 000 cd/m².
            // Now that lights are photometric the window can be sized to the
            // physical world rather than to an uncalibrated guess: the bottom is
            // below a moonlit surface, the top above a sunlit white one, and the
            // scene sits inside it instead of pegged against an end bin where
            // metering stops responding.
            //
            // A sun disc or an emissive filament is above the top and lands in
            // the last bin, which is correct — those are the pixels metering is
            // supposed to ignore rather than expose for. 254 bins over 25 stops
            // is 0.1 stops each, still far finer than anyone can see.
            min_log_luminance: -8.0,
            max_log_luminance: 17.0,
            adaptation_brighten: 0.4,
            adaptation_darken: 1.2,
        }
    }
}

impl HdrSettings {
    /// The linear multiplier an EV100 stands for, under the saturation-based
    /// speed convention (Lagarde & de Rousiers): the exposure that maps the
    /// saturation luminance `1.2 * 2^EV100` to 1.0.
    ///
    /// The GPU repeats this arithmetic for the metered path — the two must agree
    /// or toggling auto-exposure jumps — but the CPU needs it too, for the
    /// manual path's push constant.
    pub fn exposure_from_ev100(ev100: f32) -> f32 {
        1.0 / (1.2 * ev100.exp2())
    }

    /// The multiplier the manual path applies: the stated EV100, shifted by the
    /// compensation dial.
    pub fn manual_exposure(&self) -> f32 {
        Self::exposure_from_ev100(self.manual_ev100 - self.exposure_compensation)
    }

    /// The fraction of the way to a new measurement the adapted luminance moves
    /// in `dt` seconds, for an exponential approach with time constant `tau`.
    ///
    /// Framerate-independent by construction: halving `dt` and doubling the
    /// frames leaves the same total approach, which a per-frame lerp constant
    /// would not.
    pub fn adaptation_rate(tau: f32, dt: f32) -> f32 {
        if tau <= 0.0 || !dt.is_finite() || dt <= 0.0 {
            return 1.0;
        }
        1.0 - (-dt / tau).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Adaptation must depend on elapsed time, not on how many frames elapsed,
    /// or the scene brightens faster on a fast machine. Sixty steps of a 60 Hz
    /// frame and thirty of a 30 Hz one both cover one second, so both must land
    /// in the same place.
    #[test]
    fn adaptation_is_framerate_independent() {
        let converge = |dt: f32, steps: usize| {
            let mut value = 0.0f32;
            for _ in 0..steps {
                value += (1.0 - value) * HdrSettings::adaptation_rate(0.4, dt);
            }
            value
        };
        let fast = converge(1.0 / 120.0, 120);
        let slow = converge(1.0 / 30.0, 30);
        assert!(
            (fast - slow).abs() < 1e-3,
            "120 Hz reached {fast}, 30 Hz reached {slow}",
        );
    }

    /// A stop is a doubling of light, so +1 EV of compensation must be exactly
    /// twice the exposure — the property that makes the dial behave the same at
    /// every scene brightness, and the one an eyeballed curve would lose.
    #[test]
    fn one_stop_of_compensation_doubles_the_exposure() {
        let base = HdrSettings {
            auto_exposure: false,
            ..HdrSettings::default()
        };
        let brighter = HdrSettings {
            exposure_compensation: 1.0,
            ..base
        };
        assert!((brighter.manual_exposure() / base.manual_exposure() - 2.0).abs() < 1e-5);
    }

    /// A stalled frame or a first frame with no elapsed time must not leave the
    /// adapted value pinned where it started.
    #[test]
    fn a_degenerate_timestep_converges_immediately() {
        assert_eq!(HdrSettings::adaptation_rate(0.4, 0.0), 1.0);
        assert_eq!(HdrSettings::adaptation_rate(0.0, 0.016), 1.0);
        assert_eq!(HdrSettings::adaptation_rate(0.4, f32::NAN), 1.0);
    }
}
