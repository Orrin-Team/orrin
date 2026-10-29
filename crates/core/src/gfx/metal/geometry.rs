//! Every mesh's geometry, in three shared buffers: positions, surfaces and
//! indices, with a mesh as three numbers into them.
//!
//! The arithmetic that places a mesh is `rhi::geometry`'s, shared with Vulkan.
//! Memory is unified here, so an upload is a write into the buffer and growth is
//! a larger buffer with the old contents copied across; a frame still reading the
//! old one is safe because a command buffer retains what it bound.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::MtlDriver;
use crate::geom::Aabb;
use crate::gfx::rhi::{MeshSpan, Placement, grown};
use crate::gfx::{PositionVertex, SurfaceVertex, Vertex};

const INITIAL_VERTICES: u64 = 1 << 14;
const INITIAL_INDICES: u64 = 1 << 15;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub(super) struct Geometry {
    pub(super) positions: Buffer,
    pub(super) surfaces: Buffer,
    pub(super) indices: Buffer,
    placement: Placement,
}

impl Geometry {
    pub(super) fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            positions: allocate(
                device,
                INITIAL_VERTICES * size_of::<PositionVertex>() as u64,
            ),
            surfaces: allocate(device, INITIAL_VERTICES * size_of::<SurfaceVertex>() as u64),
            indices: allocate(device, INITIAL_INDICES * 4),
            placement: Placement::default(),
        }
    }

    fn upload(
        &mut self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> (MeshSpan, Aabb) {
        let bounds = Aabb::from_points(vertices.iter().map(|v| glam::Vec3::from(v.position)));
        let (positions, surfaces): (Vec<_>, Vec<_>) = vertices.iter().map(Vertex::split).unzip();

        let held_vertices = u64::from(self.placement.vertices);
        let held_indices = u64::from(self.placement.indices);
        self.reserve(
            device,
            held_vertices + positions.len() as u64,
            held_indices + indices.len() as u64,
        );
        let span = self
            .placement
            .place(positions.len() as u32, indices.len() as u32);

        let vertex_offset = span.vertex_offset as usize;
        write(&self.positions, vertex_offset, &positions);
        write(&self.surfaces, vertex_offset, &surfaces);
        write(&self.indices, span.first_index as usize, indices);
        (span, bounds)
    }

    fn reserve(
        &mut self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        vertices: u64,
        indices: u64,
    ) {
        let grow = |buffer: &mut Buffer, stride: u64, needed: u64| {
            if let Some(capacity) = grown(buffer.length() as u64 / stride, needed) {
                let bigger = allocate(device, capacity * stride);
                // SAFETY: both buffers are shared, the old is no longer than the
                // new, and nothing writes the old while it is copied.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffer.contents().as_ptr().cast::<u8>(),
                        bigger.contents().as_ptr().cast::<u8>(),
                        buffer.length(),
                    );
                }
                *buffer = bigger;
            }
        };
        grow(
            &mut self.positions,
            size_of::<PositionVertex>() as u64,
            vertices,
        );
        grow(
            &mut self.surfaces,
            size_of::<SurfaceVertex>() as u64,
            vertices,
        );
        grow(&mut self.indices, 4, indices);
    }
}

fn allocate(device: &Retained<ProtocolObject<dyn MTLDevice>>, bytes: u64) -> Buffer {
    device
        .newBufferWithLength_options(bytes as usize, MTLResourceOptions::StorageModeShared)
        .expect("failed to allocate the geometry arena")
}

/// Write `items` at element `at` of `buffer`.
fn write<T: Copy>(buffer: &Buffer, at: usize, items: &[T]) {
    let stride = size_of::<T>();
    assert!(
        (at + items.len()) * stride <= buffer.length(),
        "the arena was sized for this"
    );
    // SAFETY: the range was just checked to lie inside a shared buffer, and `T`
    // is one of the repr(C) all-float vertex halves or `u32`, so its bytes are
    // its value.
    unsafe {
        std::ptr::copy_nonoverlapping(
            items.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>().add(at * stride),
            std::mem::size_of_val(items),
        );
    }
}

impl MtlDriver {
    pub fn upload_mesh(&self, vertices: &[Vertex], indices: &[u32]) -> (MeshSpan, Aabb) {
        self.geometry
            .lock()
            .expect("a pass panicked while using the device")
            .0
            .upload(&self.device, vertices, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(x: f32) -> Vertex {
        Vertex {
            position: [x, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [1.0; 3],
            uv: [0.0; 2],
            tangent: [1.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn meshes_land_end_to_end_and_survive_growth() {
        let driver = MtlDriver::new().expect("a Metal device");
        let (first, bounds) = driver.upload_mesh(&[vertex(1.0), vertex(2.0)], &[0, 1, 0]);
        assert_eq!(
            first,
            MeshSpan {
                first_index: 0,
                index_count: 3,
                vertex_offset: 0
            }
        );
        assert_eq!(bounds.max.x, 2.0);

        // Enough vertices to force the positions buffer to grow.
        let many: Vec<_> = (0..INITIAL_VERTICES + 5)
            .map(|i| vertex(i as f32))
            .collect();
        let indices: Vec<u32> = (0..many.len() as u32).collect();
        let (second, _) = driver.upload_mesh(&many, &indices);
        assert_eq!(second.vertex_offset, 2);
        assert_eq!(second.first_index, 3);

        // The first mesh's second vertex is still where it was.
        let geometry = driver.geometry.lock().unwrap();
        let geometry = &geometry.0;
        // SAFETY: a shared buffer holding at least two position vertices.
        let positions = unsafe {
            std::slice::from_raw_parts(
                geometry
                    .positions
                    .contents()
                    .as_ptr()
                    .cast::<PositionVertex>(),
                2,
            )
        };
        assert_eq!(positions[1].position, [2.0, 0.0, 0.0]);
    }
}
