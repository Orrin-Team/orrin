/// Light bleeding from bright areas into their surroundings.
///
/// Thresholdless by design: there is no brightness cutoff feeding the chain, so
/// nothing pops as a pixel crosses one and there is no knee to retune when the
/// lighting changes. The chain filters the whole *exposure-scaled* image, which
/// is what makes it scene-invariant — the same [`strength`](Self::strength)
/// looks the same at noon and at midnight, because metering has already put both
/// at the same working range.
#[derive(Clone, Copy, Debug)]
pub struct BloomSettings {
    /// Off drops the whole chain from the frame rather than running it at zero
    /// strength.
    pub enabled: bool,
    /// How much of the final image is bloom, as a straight blend. Energy is
    /// conserved: this takes light from the scene rather than adding to it, so
    /// raising it cannot blow the image out.
    pub strength: f32,
    /// Spread of the upsample tent filter, in texels of the level being sampled.
    /// Above 1.0 the levels overlap more and the glow reaches further.
    pub radius: f32,
    /// How much weight each upsample step gives the coarser, blurrier level over
    /// the sharper one beside it. Higher spreads the glow wider.
    ///
    /// It is a blend weight rather than a gain, and that is what normalises the
    /// chain: each step is a convex combination, so the finished bloom carries
    /// the energy of one level no matter how many levels the frame's size gave
    /// it. Summing instead would tie the glow's brightness to the window size.
    pub scatter: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            // Low, because a blend is a much stronger control than an additive
            // term: at 0.06 the glow reads clearly without the image looking
            // hazy.
            strength: 0.06,
            radius: 1.0,
            scatter: 0.7,
        }
    }
}

impl BloomSettings {
    /// Weight the finest level ends up carrying in a chain of `mips` levels,
    /// under the upsample's convex blend.
    ///
    /// Exists to be asserted rather than to be called: it is the CPU mirror of
    /// what `bloom_upsample.comp` does, and the property it pins — total weight
    /// one, whatever the chain length — is the one whose absence made bloom look
    /// like fog and made its brightness depend on the window's size.
    pub fn chain_weights(scatter: f32, mips: usize) -> Vec<f32> {
        let mut weights = vec![0.0; mips.max(1)];
        // The coarsest level enters whole; every step down mixes it with the
        // sharper level beside it.
        weights[mips.saturating_sub(1)] = 1.0;
        for level in (0..mips.saturating_sub(1)).rev() {
            for weight in weights.iter_mut().skip(level + 1) {
                *weight *= scatter;
            }
            weights[level] = 1.0 - scatter;
        }
        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chain's total weight must be one however many levels the frame's
    /// extent produced. Without it the glow is as many times too bright as the
    /// chain is long — and because the length comes from the window's size, it
    /// would change brightness as the window is dragged.
    #[test]
    fn the_chain_carries_one_level_of_energy_whatever_its_length() {
        for mips in 1..=6 {
            let total: f32 = BloomSettings::chain_weights(0.7, mips).iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "a {mips}-level chain totals {total}, not 1",
            );
        }
    }

    /// Scatter has to actually move weight toward the blurrier levels, or the
    /// slider does nothing recognisable.
    #[test]
    fn more_scatter_moves_weight_to_the_coarser_levels() {
        let tight = BloomSettings::chain_weights(0.3, 4);
        let wide = BloomSettings::chain_weights(0.9, 4);
        assert!(
            wide[3] > tight[3],
            "coarsest level: {} at 0.9 vs {} at 0.3",
            wide[3],
            tight[3],
        );
        assert!(wide[0] < tight[0], "finest level should lose weight");
    }
}
