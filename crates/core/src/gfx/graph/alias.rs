//! Which transients can be handed the same allocation.
//!
//! A transient is undefined at the start of every frame and dead at the end of
//! it, so the only thing that keeps its memory reserved is the span of passes
//! between the first that touches it and the last. Two transients whose spans
//! do not overlap are never wanted at once, and one block of memory can serve
//! both — which is the lifetime information the graph has had since it was a
//! graph and has never spent.
//!
//! # What v1 aliases, and what it deliberately does not
//!
//! **Only images whose allocations are interchangeable**: same descriptor, same
//! usage, same sharing mode. That is a stricter rule than Vulkan's, and it is
//! the strictest thing the compiler can *know* — memory requirements come from
//! a `Device`, and this module has none, so identical create-info is how it
//! establishes identical requirements without asking. It costs the pairs that
//! would fit inside a larger block: the whole half-resolution post chain could
//! live in one full-resolution image's memory, and none of it does.
//!
//! The rule still pays, because a post chain is mostly one shape. At 1080p with
//! every effect on it folds 44 transients into 35 allocations and 318 MB into
//! 241 MB, and the groups are the ones a reader would predict — each
//! full-resolution stage output handed on to a later stage's, so that
//! `subsurface_color`, `oit_color` and `dof_color` are one image and
//! `ssr_color`, `refraction_color` and `motion_blur_color` are another.
//!
//! **Nothing memoryless.** Those ask for `LAZILY_ALLOCATED` memory, which is a
//! different memory type and, on the tiler they exist for, no allocation at all.
//!
//! **No packing at an offset.** Every member of a group sits at offset zero of
//! its own block, so a group is a set of equals rather than a heap with a
//! bump allocator over it. Offsets are what the shape rule above would need to
//! be relaxed, and they are the follow-up if the remaining 241 MB is ever worth
//! attacking.
//!
//! # The dependency it adds
//!
//! Handing B the memory A was using is a write-after-read hazard, and the one
//! the rest of the compiler is built to assume cannot happen (`dependency_edges`
//! orders every reader after every writer *of one resource*, which says nothing
//! about two). `compile` closes it by seeding B's barrier state from A's final
//! one, so B's first access sources A's last — see `State::aliased`. The
//! dependency is therefore derived and emitted exactly like every other one,
//! rather than being a rule the executor has to remember.

use std::ops::Range;

use super::{ImageDesc, PassDecl, PassId, ResourceId, TransientImage};

/// The slots of `order` each resource is live across, `None` for one no live
/// pass touches.
///
/// Inclusive of both ends: a resource is live *during* the pass that last reads
/// it, which is the whole point — the slot after that one is the first that may
/// have its memory.
pub(super) fn lifetimes(
    order: &[PassId],
    passes: &[PassDecl],
    resource_count: usize,
) -> Vec<Option<Range<usize>>> {
    let mut lifetimes: Vec<Option<Range<usize>>> = vec![None; resource_count];
    for (slot, &pass_id) in order.iter().enumerate() {
        for &(resource, _) in &passes[pass_id.index()].accesses {
            let lifetime = &mut lifetimes[resource.index()];
            *lifetime = Some(match lifetime {
                Some(range) => range.start..slot + 1,
                None => slot..slot + 1,
            });
        }
    }
    lifetimes
}

/// Partition the transients into sets that can share one allocation, each in
/// the order the memory changes hands in.
///
/// First fit down the images in lifetime order, which is the standard answer to
/// interval-graph colouring and the optimal one for a chain: a frame's post
/// chain is a sequence of intervals, and greedy is exact on those. Groups of one
/// are dropped — they are what the executor was already doing.
pub(super) fn groups(
    images: &[Option<TransientImage>],
    lifetimes: &[Option<Range<usize>>],
    upscaled: bool,
) -> Vec<Vec<ResourceId>> {
    let mut candidates: Vec<(ResourceId, &TransientImage, Range<usize>)> = images
        .iter()
        .enumerate()
        .filter_map(|(index, image)| {
            let image = image.as_ref()?;
            let lifetime = lifetimes[index].clone()?;
            // A lazily-allocated image is not the kind of memory this shares.
            (!image.memoryless).then_some((ResourceId(index as u32), image, lifetime))
        })
        .collect();
    candidates.sort_by_key(|(id, _, lifetime)| (lifetime.start, *id));

    let mut groups: Vec<(&TransientImage, usize, Vec<ResourceId>)> = Vec::new();
    for (id, image, lifetime) in candidates {
        match groups.iter_mut().find(|(shape, end, _)| {
            *end <= lifetime.start && interchangeable(shape, image, upscaled)
        }) {
            Some((_, end, members)) => {
                *end = lifetime.end;
                members.push(id);
            }
            None => groups.push((image, lifetime.end, vec![id])),
        }
    }

    groups
        .into_iter()
        .filter(|(_, _, members)| members.len() > 1)
        .map(|(_, _, members)| members)
        .collect()
}

/// Whether two images would be allocated identically, which is how this module
/// establishes that their memory requirements match without a `Device` to ask.
/// Their extents are compared as the frame will size them: in a frame that does
/// not upscale, a render-extent and an output-extent image are one size.
fn interchangeable(one: &TransientImage, two: &TransientImage, upscaled: bool) -> bool {
    let folded = |image: &TransientImage| ImageDesc {
        extent: image.desc.extent.folded(upscaled),
        ..image.desc
    };
    folded(one) == folded(two) && one.usage == two.usage && one.concurrent == two.concurrent
}
