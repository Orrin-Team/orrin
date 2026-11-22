//! The device-free half of the shared geometry arena: where a mesh lands, when
//! a replaced mesh's space may be used again, and when the buffers behind it
//! must grow.
//!
//! The buffers, staging and copies are a driver's business; the arithmetic that
//! decides a mesh's span is not, and holding it here is what lets it be tested
//! without a device and shared by every driver.
//!
//! # Reuse
//!
//! A replaced mesh's span is *retired*, not freed: frames already recorded may
//! still be drawing from it. It becomes free space once [`has_retired`] says
//! every one of them is done — the rule, and the depth, the driver's pools use
//! for every other frame-delayed destroy. An upload takes the first free range
//! that fits, splitting it, and moves the end only when none does; a freed range
//! merges with its neighbours, so a mesh streamed at a size that wanders keeps
//! landing in the space its predecessors left rather than in slivers of it.
//!
//! The end never comes back down, even over free space. A driver that grows its
//! buffers with a copy it does not wait for relies on a span placed at the end
//! being bytes that copy never writes (see `vulkan::mesh`).
//!
//! Vertices and indices are separate address spaces. Positions and surfaces
//! share the vertex one, because one `vertex_offset` addresses both.

use super::pool::has_retired;

/// Where one mesh's geometry sits inside the shared buffers.
///
/// The three fields are `VkDrawIndexedIndirectCommand`'s, in its units: indices
/// for `first_index`, vertices for `vertex_offset`. `vertex_offset` is signed
/// because the command's is, and applies to *both* vertex bindings — which is
/// sound only because a mesh's positions and surfaces are placed together and
/// therefore share a vertex number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshSpan {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
}

/// Free elements `start..end` of one address space.
#[derive(Clone, Copy)]
struct Hole {
    start: u32,
    end: u32,
}

/// One address space: everything below `end` is a mesh's, retiring, or in
/// `holes` — sorted, disjoint, and never touching one another.
#[derive(Default)]
struct Ranges {
    end: u32,
    holes: Vec<Hole>,
}

impl Ranges {
    /// First fit, splitting the hole it lands in.
    fn allocate(&mut self, len: u32) -> u32 {
        if len == 0 {
            return self.end;
        }
        if let Some(at) = self
            .holes
            .iter()
            .position(|hole| hole.end - hole.start >= len)
        {
            let hole = &mut self.holes[at];
            let start = hole.start;
            hole.start += len;
            if hole.start == hole.end {
                self.holes.remove(at);
            }
            return start;
        }
        let start = self.end;
        self.end += len;
        start
    }

    fn free(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }
        let end = start + len;
        let at = self.holes.partition_point(|hole| hole.start < start);
        debug_assert!(
            (at == 0 || self.holes[at - 1].end <= start)
                && self.holes.get(at).is_none_or(|next| end <= next.start),
            "{start}..{end} freed while part of it was already free"
        );
        let joins_before = at > 0 && self.holes[at - 1].end == start;
        let joins_after = self.holes.get(at).is_some_and(|next| next.start == end);
        match (joins_before, joins_after) {
            (true, true) => {
                self.holes[at - 1].end = self.holes[at].end;
                self.holes.remove(at);
            }
            (true, false) => self.holes[at - 1].end = end,
            (false, true) => self.holes[at].start = start,
            (false, false) => self.holes.insert(at, Hole { start, end }),
        }
    }
}

/// A replaced mesh's span, waiting for the frames that may draw from it.
struct Retiring {
    first_vertex: u32,
    vertices: u32,
    first_index: u32,
    indices: u32,
    frame: u64,
}

/// Where meshes go in both address spaces, and the spans on their way back.
///
/// Separate from the buffers so the arithmetic that decides where a mesh lands
/// can be tested without a device.
pub(crate) struct Placement {
    vertices: Ranges,
    indices: Ranges,
    retiring: Vec<Retiring>,
    frames_in_flight: u64,
}

impl Placement {
    pub(crate) fn new(frames_in_flight: u64) -> Self {
        Self {
            vertices: Ranges::default(),
            indices: Ranges::default(),
            retiring: Vec::new(),
            frames_in_flight,
        }
    }

    /// Where a mesh of this size goes: free space where it fits, the end where
    /// it does not.
    pub(crate) fn place(&mut self, vertices: u32, indices: u32) -> MeshSpan {
        MeshSpan {
            vertex_offset: self.vertices.allocate(vertices) as i32,
            first_index: self.indices.allocate(indices),
            index_count: indices,
        }
    }

    /// Give back a replaced mesh's span once the frames up to `frame`, which
    /// may have drawn it, have retired. `vertices` is the mesh's vertex count,
    /// which the span does not carry.
    pub(crate) fn retire(&mut self, span: MeshSpan, vertices: u32, frame: u64) {
        self.retiring.push(Retiring {
            first_vertex: span.vertex_offset as u32,
            vertices,
            first_index: span.first_index,
            indices: span.index_count,
            frame,
        });
    }

    /// Free every retiring span whose frames are done by the time `frame`
    /// begins. Once a frame, so nothing but the emptiness check unless a mesh
    /// was replaced.
    pub(crate) fn collect(&mut self, frame: u64) {
        if self.retiring.is_empty() {
            return;
        }
        let (vertices, indices) = (&mut self.vertices, &mut self.indices);
        let depth = self.frames_in_flight;
        self.retiring.retain(|span| {
            let done = has_retired(span.frame, depth, frame);
            if done {
                vertices.free(span.first_vertex, span.vertices);
                indices.free(span.first_index, span.indices);
            }
            !done
        });
    }

    /// How many vertices and indices the buffers must hold.
    pub(crate) fn ends(&self) -> (u32, u32) {
        (self.vertices.end, self.indices.end)
    }
}

/// The capacity a buffer of `capacity` should grow to in order to hold `needed`,
/// or `None` when it already does.
///
/// Powers of two, so a scene that loads a thousand meshes copies the arena a
/// logarithmic number of times rather than a thousand.
pub(crate) fn grown(capacity: u64, needed: u64) -> Option<u64> {
    (needed > capacity).then(|| needed.next_power_of_two())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The depth the drivers run at.
    const IN_FLIGHT: u64 = 2;

    /// A mesh's span has to keep meaning what it meant after the arena grows,
    /// because handles handed out earlier are never revisited.
    #[test]
    fn growth_preserves_earlier_spans() {
        let mut placement = Placement::new(IN_FLIGHT);
        let first = placement.place(4, 6);
        assert_eq!(grown(8, 4), None);
        assert_eq!(grown(8, 12), Some(16));
        let after = placement.place(2, 2);

        assert_eq!(first.first_index, 0);
        assert_eq!(after.first_index, 6);
        assert_eq!(after.vertex_offset, 4);
    }

    /// A frame recorded before the replace may still be drawing the old span
    /// until `IN_FLIGHT` later frames have begun — the pools' rule.
    #[test]
    fn a_retired_span_is_not_reused_until_its_frames_have_retired() {
        let mut placement = Placement::new(IN_FLIGHT);
        let old = placement.place(4, 6);
        placement.retire(old, 4, 10);

        placement.collect(10);
        placement.collect(11);
        let early = placement.place(4, 6);
        assert_eq!(
            (early.vertex_offset, early.first_index),
            (4, 6),
            "frame 10 may still be on the GPU, so the old span is not handed out"
        );

        placement.collect(12);
        let late = placement.place(4, 6);
        assert_eq!(
            (late.vertex_offset, late.first_index),
            (0, 0),
            "once frame 10 has retired its span is reused"
        );
        assert_eq!(placement.ends(), (8, 12), "and the end did not move");
    }

    #[test]
    fn adjacent_retired_spans_merge_into_one_hole() {
        let mut placement = Placement::new(IN_FLIGHT);
        let a = placement.place(4, 6);
        let b = placement.place(4, 6);
        let c = placement.place(4, 6);
        placement.place(1, 1);
        // Out of address order, so the merge has to join on both sides.
        placement.retire(a, 4, 0);
        placement.retire(c, 4, 0);
        placement.retire(b, 4, 0);
        placement.collect(IN_FLIGHT);

        let whole = placement.place(12, 18);
        assert_eq!((whole.vertex_offset, whole.first_index), (0, 0));
        assert_eq!(placement.ends(), (13, 19));
    }

    #[test]
    fn a_hole_too_small_is_skipped_and_still_used_by_what_fits() {
        let mut placement = Placement::new(IN_FLIGHT);
        let small = placement.place(4, 6);
        placement.place(1, 1);
        placement.retire(small, 4, 0);
        placement.collect(IN_FLIGHT);

        let large = placement.place(5, 7);
        assert_eq!((large.vertex_offset, large.first_index), (5, 7));
        let fits = placement.place(3, 3);
        assert_eq!((fits.vertex_offset, fits.first_index), (0, 0));
        let rest = placement.place(1, 3);
        assert_eq!(
            (rest.vertex_offset, rest.first_index),
            (3, 3),
            "a split hole keeps its remainder"
        );
    }

    /// The same with a size that wanders, which is what an edit in a modelling
    /// tool does to a streamed mesh. Up to `IN_FLIGHT + 1` copies are held at
    /// once — the live one and those still retiring — and first fit can strand
    /// about as much again in holes none of the next sizes fits; what matters is
    /// that it settles there rather than creeping.
    #[test]
    fn a_mesh_whose_size_wanders_settles_into_a_bounded_amount_of_the_arena() {
        let mut placement = Placement::new(IN_FLIGHT);
        let mut seed = 0x9e37_79b9u32;
        let mut size = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            900 + (seed >> 16) % 200
        };
        let first = size();
        let mut live = (placement.place(first, first * 3), first);
        let mut settled = (0, 0);
        for frame in 1..=2_000 {
            placement.collect(frame);
            placement.retire(live.0, live.1, frame);
            let next = size();
            live = (placement.place(next, next * 3), next);
            if frame == 500 {
                settled = placement.ends();
            }
        }
        assert_eq!(placement.ends(), settled, "the end crept after settling");
        let copies = 2 * (IN_FLIGHT as u32 + 1);
        assert!(
            settled.0 <= copies * 1_100 && settled.1 <= copies * 3_300,
            "more than {copies} of the largest size: {settled:?}"
        );
    }
}
