/// Temporal antialiasing: the frame is jittered by a subpixel offset each frame
/// and the result accumulated against a reprojected history.
///
/// It is the only pass whose *input* it also produces — the camera jitter and
/// the motion vectors exist for it — so turning it off changes the projection
/// matrix the whole frame is drawn with, not just which nodes the graph
/// registers.
#[derive(Clone, Copy, Debug)]
pub struct TaaSettings {
    /// Off drops the resolve node *and* stops jittering the projection. A frame
    /// that jittered without resolving would simply shake.
    pub enabled: bool,
    /// Weight the reprojected history keeps in the steady state. Higher is
    /// smoother and slower to respond; the neighbourhood clip is what keeps a
    /// high value from smearing rather than this dial.
    pub feedback: f32,
    /// Multiplier on the Halton offset, in pixels. One covers the whole pixel,
    /// which is what actually antialiases; lower trades edge quality for less
    /// texture softening.
    pub jitter_scale: f32,
}

impl Default for TaaSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            feedback: 0.92,
            jitter_scale: 1.0,
        }
    }
}
