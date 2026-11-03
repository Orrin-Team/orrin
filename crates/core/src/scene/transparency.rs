use orrin_registry::Reflect;

/// Whether the frame draws blended geometry at all.
///
/// Frame *structure*, like SSAO and reflections: off is a different graph, not a
/// flag read at record time. It registers the accumulation and composite nodes,
/// and it is one more thing that keeps the geometry prepass alive — the
/// accumulation depth-tests against the depth that pass writes.
///
/// A setting rather than something derived from whether any blended object is on
/// screen. Deriving it would recompile the graph and reallocate every image the
/// moment the last pane of glass left the frustum, and a toggle is also the A/B
/// for "is this a transparency bug?".
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct TransparencySettings {
    pub enabled: bool,
}

impl Default for TransparencySettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Whether the frame draws refractive geometry at all.
///
/// Frame structure like [`TransparencySettings`], and beside it rather than
/// inside it because the two queues are not the same feature wearing two names.
/// Weighted-blended transparency must not be sorted and screen-space refraction
/// must be; one costs two small targets and the other a full-resolution mip
/// pyramid of the frame. A scene with glass but no smoke wants the second
/// without the first, and separating them is also what makes each its own A/B
/// when a non-opaque surface looks wrong.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct RefractionSettings {
    pub enabled: bool,
}

impl Default for RefractionSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}
