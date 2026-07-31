//! Broadphase: a bounding volume hierarchy over collider AABBs.
//!
//! Rebuilt from scratch every frame. That sounds wasteful, but a top-down
//! build is O(n log n), branchless-simple, and needs none of the incremental
//! machinery (fat AABBs, refitting, tree rotations) a persistent tree does —
//! Box2D-style dynamic trees only pay off once thousands of mostly-static
//! bodies exist. The seam is `build` + `query_pairs`; a persistent tree can
//! replace the internals later without touching `collision::run`.

use glam::Vec3;
use super::Aabb;

type NodeIndex = u32;

enum NodeKind {
    // `body` indexes the `bounds` slice passed to `build` (which
    // `collision::run` keeps parallel to its body list).
    Leaf { body: u32 },
    Internal { left: NodeIndex, right: NodeIndex },
}

struct Node {
    // Its own AABB for a leaf, the union of both children for an internal node.
    aabb: Aabb,
    kind: NodeKind,
}

#[derive(Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    root: Option<NodeIndex>,
    /// Scratch for the partition in `build_ranges`, held here rather than
    /// created per build for the same reason as `nodes` — see
    /// [`rebuild`](Bvh::rebuild). Never read outside a build.
    indices: Vec<usize>,
}

impl Bvh {
    /// Build a tree over `bounds`; leaf `body` values are indices into it.
    ///
    /// Allocates. A caller that rebuilds every frame should keep one `Bvh` and
    /// call [`rebuild`](Bvh::rebuild) instead.
    pub fn build(bounds: &[Aabb]) -> Self {
        let mut bvh = Bvh::default();
        bvh.rebuild(bounds);
        bvh
    }

    /// Rebuild over `bounds` in place, keeping the buffers the last build grew.
    ///
    /// Rebuilding from scratch every frame is the design (see the module docs)
    /// and stays; the two `Vec`s it filled were the incidental part. Clearing
    /// instead of dropping keeps the capacity the previous frame reached, so a
    /// scene whose body count is stable stops calling the allocator after the
    /// first frame — which is most scenes, most frames.
    pub fn rebuild(&mut self, bounds: &[Aabb]) {
        self.nodes.clear();
        self.indices.clear();
        self.indices.extend(0..bounds.len());

        // Disjoint fields, so the partition scratch and the node arena can be
        // borrowed at once.
        let Bvh { nodes, indices, root } = self;
        *root = if indices.is_empty() {
            None
        } else {
            Some(Self::build_ranges(nodes, bounds, indices))
        };
    }

    fn build_ranges(nodes: &mut Vec<Node>, bounds: &[Aabb], indices: &mut [usize]) -> NodeIndex {
        if indices.len() == 1 {
            let index = nodes.len() as u32;
            nodes.push(Node { aabb: bounds[indices[0]], kind: NodeKind::Leaf { body: indices[0] as u32 } });
            return index;
        }

        // Split on the axis where the centers spread widest; centers (not box
        // edges) keep the partition meaningful when bounds overlap heavily.
        let mut min_c = Vec3::splat(f32::INFINITY);
        let mut max_c = Vec3::splat(f32::NEG_INFINITY);

        for &i in indices.iter() {
            let center = bounds[i].center();
            min_c = min_c.min(center);
            max_c = max_c.max(center);
        }

        let span = max_c - min_c;
        let axis: usize = if span.x >= span.y && span.x >= span.z { 0 }
        else if span.y >= span.z { 1 }
        else { 2 };

        // Median by count, not by value: both halves stay non-empty even when
        // every center coincides, so the recursion always terminates.
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&i, &j|
            bounds[i].center()[axis].total_cmp(&bounds[j].center()[axis]));

        let (left_half, right_half) = indices.split_at_mut(mid);
        let left = Self::build_ranges(nodes, bounds, left_half);
        let right = Self::build_ranges(nodes, bounds, right_half);

        let aabb = nodes[left as usize].aabb.union(&nodes[right as usize].aabb);

        let index = nodes.len() as u32;
        nodes.push(Node { aabb, kind: NodeKind::Internal { left, right } });
        index
    }

    /// Append every overlapping body pair `(i, j)` with `i < j` to `out`.
    pub fn query_pairs(&self, out: &mut Vec<(u32, u32)>) {
        // Hoisted out of the leaf loop: a stack per leaf is one allocation per
        // body, and reusing it keeps the capacity reached by the deepest
        // traversal for every leaf after the first.
        let mut stack: Vec<NodeIndex> = Vec::new();

        for node in &self.nodes {
            let NodeKind::Leaf { body } = node.kind else { continue };

            stack.clear();
            if let Some(root) = self.root {
                stack.push(root);
            }

            while let Some(i) = stack.pop() {
                let other = &self.nodes[i as usize];
                if !other.aabb.overlaps(&node.aabb) {
                    continue;
                }
                match other.kind {
                    NodeKind::Leaf { body: other_body } => {
                        // Every pair is discovered from both of its leaves;
                        // `<` keeps exactly one and drops self-pairs.
                        if body < other_body {
                            out.push((body, other_body));
                        }
                    }
                    NodeKind::Internal { left, right } => {
                        stack.push(left);
                        stack.push(right);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn aabb(min: [f32; 3], max: [f32; 3]) -> Aabb {
        Aabb { min: Vec3::from_array(min), max: Vec3::from_array(max) }
    }

    fn pairs_of(bounds: &[Aabb]) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        Bvh::build(bounds).query_pairs(&mut pairs);
        pairs.sort_unstable();
        pairs
    }

    /// A reused tree has to answer exactly what a fresh one does, including
    /// after a rebuild over *fewer* bodies — where stale nodes left in the
    /// arena, or stale indices in the partition scratch, would show up as
    /// phantom pairs.
    #[test]
    fn rebuild_matches_a_fresh_build() {
        let crowded: Vec<Aabb> = (0..12)
            .map(|i| {
                let min = Vec3::new(i as f32 * 0.5, 0.0, 0.0);
                Aabb { min, max: min + Vec3::ONE }
            })
            .collect();
        let sparse = vec![
            aabb([0.0; 3], [1.0; 3]),
            aabb([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]),
        ];

        let mut reused = Bvh::default();
        for bounds in [&crowded, &sparse, &crowded, &Vec::new(), &sparse] {
            let mut from_reused = Vec::new();
            reused.rebuild(bounds);
            reused.query_pairs(&mut from_reused);
            from_reused.sort_unstable();
            assert_eq!(from_reused, pairs_of(bounds));
        }
    }

    #[test]
    fn empty_and_single_produce_no_pairs() {
        assert!(pairs_of(&[]).is_empty());
        assert!(pairs_of(&[aabb([0.0; 3], [1.0; 3])]).is_empty());
    }

    #[test]
    fn overlapping_pair_is_found_once() {
        let pairs = pairs_of(&[
            aabb([0.0; 3], [1.0; 3]),
            aabb([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]),
            aabb([10.0; 3], [11.0; 3]), // far away, matches nothing
        ]);
        assert_eq!(pairs, vec![(0, 1)]);
    }

    #[test]
    fn matches_brute_force_on_a_grid() {
        // 4x4 grid of unit boxes spaced 0.75 apart: lots of overlap, so a
        // traversal bug (missed subtree, double-count) can't hide.
        let mut bounds = Vec::new();
        for x in 0..4 {
            for z in 0..4 {
                let min = Vec3::new(x as f32 * 0.75, 0.0, z as f32 * 0.75);
                bounds.push(Aabb { min, max: min + Vec3::ONE });
            }
        }
        let mut expected = Vec::new();
        for i in 0..bounds.len() {
            for j in (i + 1)..bounds.len() {
                if bounds[i].overlaps(&bounds[j]) {
                    expected.push((i as u32, j as u32));
                }
            }
        }
        assert_eq!(pairs_of(&bounds), expected);
    }
}
