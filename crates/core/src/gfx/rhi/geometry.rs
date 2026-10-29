//! The device-free half of the shared geometry arena: where a mesh lands and
//! when the buffers behind it must grow.
//!
//! The buffers, staging and copies are a driver's business; the arithmetic that
//! decides a mesh's span is not, and holding it here is what lets it be tested
//! without a device and shared by every driver.

/// Where one mesh's geometry sits inside the shared buffers.
///
/// The three fields are `VkDrawIndexedIndirectCommand`'s, in its units: indices
/// for `first_index`, vertices for `vertex_offset`. `vertex_offset` is signed
/// because the command's is, and applies to *both* vertex bindings — which is
/// sound only because a mesh's positions and surfaces are appended together and
/// therefore share a vertex number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshSpan {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
}

/// The bump allocator over both watermarks.
///
/// Separate from the buffers so the arithmetic that decides where a mesh lands
/// can be tested without a device.
#[derive(Default)]
pub(crate) struct Placement {
    pub(crate) vertices: u32,
    pub(crate) indices: u32,
}

impl Placement {
    /// Where a mesh of this size goes, advancing both watermarks past it.
    pub(crate) fn place(&mut self, vertices: u32, indices: u32) -> MeshSpan {
        let span = MeshSpan {
            first_index: self.indices,
            index_count: indices,
            vertex_offset: self.vertices as i32,
        };
        self.vertices += vertices;
        self.indices += indices;
        span
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

    #[test]
    fn spans_are_appended_end_to_end() {
        let mut placement = Placement::default();
        let first = placement.place(4, 6);
        let second = placement.place(3, 3);

        assert_eq!(first.vertex_offset, 0);
        assert_eq!(first.first_index, 0);
        assert_eq!(first.index_count, 6);
        assert_eq!(second.vertex_offset, 4);
        assert_eq!(second.first_index, 6);
        assert_eq!(placement.vertices, 7);
        assert_eq!(placement.indices, 9);
    }

    /// A mesh's span has to keep meaning what it meant after the arena grows,
    /// because handles handed out earlier are never revisited.
    #[test]
    fn growth_preserves_earlier_spans() {
        let mut placement = Placement::default();
        let first = placement.place(4, 6);
        assert_eq!(grown(8, 4), None);
        assert_eq!(grown(8, 12), Some(16));
        let after = placement.place(2, 2);

        assert_eq!(first.first_index, 0);
        assert_eq!(after.first_index, 6);
        assert_eq!(after.vertex_offset, 4);
    }

    #[test]
    fn an_empty_mesh_takes_no_room() {
        let mut placement = Placement::default();
        placement.place(0, 0);
        let next = placement.place(1, 3);

        assert_eq!(next.first_index, 0);
        assert_eq!(next.vertex_offset, 0);
    }
}
