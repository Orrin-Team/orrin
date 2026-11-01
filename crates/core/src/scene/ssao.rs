use orrin_registry::Reflect;

#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct SsaoSettings {
    /// When false the SSAO passes are skipped and occlusion is 1.0 everywhere.
    pub enabled: bool,
    /// Hemisphere sample radius in world units.
    pub radius: f32,
    /// Depth bias that fights self-occlusion acne on flat surfaces.
    pub bias: f32,
    /// Contrast applied to the result (`ao = pow(ao, power)`).
    pub power: f32,
    /// Resolve and blur the occlusion at half the frame's width and height,
    /// letting the bilateral blur beside it upsample the result.
    ///
    /// A quarter of the pixels through the most expensive loop in the frame:
    /// SSAO is 32 taps, each of them a dependent and effectively random depth
    /// fetch. On by default, because the term it computes is low-frequency and
    /// then deliberately blurred — there is nothing at full resolution for the
    /// extra three quarters of the work to resolve.
    ///
    /// What half resolution costs is edge quality, and that cost is paid by the
    /// blur being bilateral: a box filter here would drag a near surface's
    /// occlusion across the silhouette onto the far one, and at half resolution
    /// each tap covers four pixels, so the halo would be wide enough to read as
    /// an outline. Weighting each tap by depth removes it. Turning this off is
    /// still the A/B for "is that artefact the AO?".
    ///
    /// Structural, not a uniform. It resizes graph resources, so switching it
    /// recompiles the frame graph — cheaply, because `GraphImages` caches its
    /// allocations across a recompile.
    pub half_resolution: bool,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 1.0,
            bias: 0.025,
            power: 1.0,
            half_resolution: true,
        }
    }
}
