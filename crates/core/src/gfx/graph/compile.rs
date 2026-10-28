use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::ops::Range;

use super::sync::{AccessFlags, ImageLayout, ImageUsage, PipelineStages};

use super::{
    Access, Barrier, GraphBuilder, GraphError, ImageDesc, PassDecl, PassId, PassKind, Queue,
    ResourceDecl, ResourceId, ResourceKind, Segment, alias, schedule,
};

/// A graph-owned image, sized and flagged by the compiler. `usage` is the union
/// of what passes declared, so an image is never created with a capability
/// nothing asked for and never missing one something did.
#[derive(Clone, Copy, Debug)]
pub struct TransientImage {
    pub desc: ImageDesc,
    pub usage: ImageUsage,
    /// Every access is an attachment, so the image never leaves the render pass
    /// that wrote it. On a tiler that means it can live in tile memory and cost
    /// no DRAM at all — the property the MSAA color and depth targets rely on
    /// for the 4x HDR buffer to be affordable on Apple hardware.
    pub memoryless: bool,
    /// Reachable from both queues, because both touch it. See
    /// [`FrameGraph::concurrent`].
    pub concurrent: bool,
    /// Which of [`FrameGraph::alias_groups`] this image shares its allocation
    /// with, `None` for one that has its own. See [`alias`](super::alias).
    pub alias: Option<u32>,
}

/// A compiled frame: passes in execution order, the barriers between them, and
/// the images the graph owns.
///
/// Compiled from declarations alone — no `Device` is involved — which is what
/// lets CI assert the barrier plan on a runner with no GPU.
pub struct FrameGraph {
    resources: Vec<ResourceDecl>,
    passes: Vec<PassDecl>,
    order: Vec<PassId>,
    /// Parallel to `order`: what must be recorded before that pass runs.
    barriers: Vec<Vec<Barrier>>,
    final_barriers: Vec<Barrier>,
    culled: Vec<PassId>,
    /// Indexed by `ResourceId`; `None` for imports and buffers.
    images: Vec<Option<TransientImage>>,
    /// Contiguous runs of `order`, one per submission. One segment covering the
    /// whole frame unless it was split — see `schedule.rs`.
    segments: Vec<Segment>,
    /// Resources reached from more than one queue, which therefore cannot be
    /// created in `SharingMode::Exclusive`. Indexed by `ResourceId`.
    concurrent: Vec<bool>,
    /// The slots each resource is live across; `None` for one no live pass
    /// touches. Indexed by `ResourceId`.
    lifetimes: Vec<Option<Range<usize>>>,
    /// Sets of transients that share one allocation, each in the order the
    /// memory changes hands in.
    alias_groups: Vec<Vec<ResourceId>>,
}

impl FrameGraph {
    pub fn order(&self) -> &[PassId] {
        &self.order
    }

    pub fn barriers_before(&self, slot: usize) -> &[Barrier] {
        &self.barriers[slot]
    }

    pub fn final_barriers(&self) -> &[Barrier] {
        &self.final_barriers
    }

    pub fn culled(&self) -> &[PassId] {
        &self.culled
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Which queue the pass in `slot` is submitted to.
    pub fn slot_queue(&self, slot: usize) -> Queue {
        self.segments
            .iter()
            .find(|segment| segment.passes.contains(&slot))
            .map_or(Queue::Graphics, |segment| segment.queue)
    }

    /// The resources more than one queue touches.
    ///
    /// An image in `SharingMode::Exclusive` belongs to one queue family at a
    /// time and reading it from another is undefined without an ownership
    /// transfer; these are the ones the executor has to create reachable from
    /// both instead. Empty for an unsplit frame, which is every frame on a
    /// device with one queue.
    pub fn concurrent(&self) -> impl Iterator<Item = ResourceId> + '_ {
        self.concurrent
            .iter()
            .enumerate()
            .filter(|(_, shared)| **shared)
            .map(|(index, _)| ResourceId(index as u32))
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    pub fn pass_name(&self, pass: PassId) -> &'static str {
        self.passes[pass.index()].name
    }

    pub fn pass_kind(&self, pass: PassId) -> PassKind {
        self.passes[pass.index()].kind
    }

    pub fn resource_name(&self, resource: ResourceId) -> &'static str {
        self.resources[resource.index()].name
    }

    pub(super) fn pass(&self, pass: PassId) -> &PassDecl {
        &self.passes[pass.index()]
    }

    /// The slots of [`order`](Self::order) this resource is live across, from
    /// the first pass that touches it to the last inclusive. `None` for an
    /// import, a buffer, or a resource only culled passes touch.
    pub fn lifetime(&self, resource: ResourceId) -> Option<Range<usize>> {
        self.lifetimes[resource.index()].clone()
    }

    /// The sets of transients that share one allocation, each in the order the
    /// memory changes hands in.
    ///
    /// Members of a group are never alive at the same time — that is what makes
    /// them a group — and the barrier plan already carries the dependency that
    /// hands the memory over, so an executor that honours these needs nothing
    /// else. One that ignores them and allocates separately is correct too, just
    /// larger: see [`alias`](super::alias).
    pub fn alias_groups(&self) -> &[Vec<ResourceId>] {
        &self.alias_groups
    }

    /// The images the executor must allocate, in declaration order.
    pub fn transient_images(&self) -> impl Iterator<Item = (ResourceId, TransientImage)> + '_ {
        self.images
            .iter()
            .enumerate()
            .filter_map(|(index, image)| image.map(|image| (ResourceId(index as u32), image)))
    }
}

/// Order the passes, derive the barriers between them, and size the images.
pub fn compile(builder: GraphBuilder) -> Result<FrameGraph, GraphError> {
    builder.validate_names()?;
    let GraphBuilder {
        resources,
        passes,
        async_compute,
    } = builder;

    for pass in &passes {
        check_single_access_per_resource(pass, &resources)?;
        check_compute_declares_no_attachment(pass, &resources)?;
    }

    let order = topological_order(&passes)?;
    let (order, culled) = cull(order, &passes, &resources);
    check_raw_passes_last(&order, &passes)?;
    // Before the barriers, because which stages one may name depends on the
    // queue its pass is recorded on.
    let segments = schedule::segments(&order, &passes, async_compute);
    let queues = schedule::slot_queues(&segments, order.len());
    let mut images = derive_images(&order, &passes, &resources);
    let concurrent = derive_concurrent(&order, &passes, &resources, &segments);
    for (index, image) in images.iter_mut().enumerate() {
        if let Some(image) = image {
            image.concurrent = concurrent[index];
        }
    }

    // Before the barriers, because sharing an allocation is a dependency
    // between the resources that share it and the plan has to carry it.
    let lifetimes = alias::lifetimes(&order, &passes, resources.len());
    let alias_groups = alias::groups(&images, &lifetimes);
    let mut previous_in_group = vec![None; resources.len()];
    for (index, group) in alias_groups.iter().enumerate() {
        for (position, &member) in group.iter().enumerate() {
            images[member.index()]
                .as_mut()
                .expect("only transients are grouped")
                .alias = Some(index as u32);
            previous_in_group[member.index()] = position.checked_sub(1).map(|before| group[before]);
        }
    }

    let (barriers, final_barriers) =
        derive_barriers(&order, &passes, &resources, &queues, &previous_in_group)?;

    Ok(FrameGraph {
        resources,
        passes,
        order,
        barriers,
        final_barriers,
        culled,
        images,
        segments,
        concurrent,
        lifetimes,
        alias_groups,
    })
}

/// Which resources more than one segment's queue touches.
///
/// Per resource rather than per boundary: a scratch image written and read
/// entirely inside the compute tail never leaves that queue however many
/// segments the frame has, and paying for it would be paying for the split
/// rather than for what crossed it.
fn derive_concurrent(
    order: &[PassId],
    passes: &[PassDecl],
    resources: &[ResourceDecl],
    segments: &[Segment],
) -> Vec<bool> {
    let mut touched: Vec<Option<Queue>> = vec![None; resources.len()];
    let mut concurrent = vec![false; resources.len()];
    for segment in segments {
        for slot in segment.passes.clone() {
            for &(resource, _) in &passes[order[slot].index()].accesses {
                match touched[resource.index()] {
                    Some(queue) if queue != segment.queue => concurrent[resource.index()] = true,
                    Some(_) => {}
                    None => touched[resource.index()] = Some(segment.queue),
                }
            }
        }
    }
    concurrent
}

/// A pass is a single point in the schedule, so two accesses to one resource
/// inside it cannot be ordered against each other.
fn check_single_access_per_resource(
    pass: &PassDecl,
    resources: &[ResourceDecl],
) -> Result<(), GraphError> {
    for (index, &(resource, access)) in pass.accesses.iter().enumerate() {
        if let Some(&(_, first)) = pass.accesses[..index]
            .iter()
            .find(|(other, _)| *other == resource)
        {
            return Err(GraphError::ConflictingAccess {
                pass: pass.name,
                resource: resources[resource.index()].name,
                first,
                second: access,
            });
        }
    }
    Ok(())
}

/// A dispatch runs outside any render pass, so a compute pass has nothing to
/// attach an attachment to. Caught here, where the message can say so, rather
/// than as a framebuffer the executor never builds and a validation layer
/// complaint about the layout it therefore never reached.
fn check_compute_declares_no_attachment(
    pass: &PassDecl,
    resources: &[ResourceDecl],
) -> Result<(), GraphError> {
    if pass.kind != PassKind::Compute {
        return Ok(());
    }
    for &(resource, access) in &pass.accesses {
        if access.is_attachment() {
            return Err(GraphError::AttachmentInComputePass {
                pass: pass.name,
                resource: resources[resource.index()].name,
                access,
            });
        }
    }
    Ok(())
}

/// A raw pass owns its own submission, and v1 records the rest of the frame as a
/// single command buffer submitted before any of them. Splitting that command
/// buffer around a raw pass in the middle is possible, but it then hands the same
/// images to two submissions in one frame and vulkano's per-command-buffer
/// resource tracking rejects the second — so the constraint is checked here, with
/// a message, rather than discovered as a submission failure at runtime.
fn check_raw_passes_last(order: &[PassId], passes: &[PassDecl]) -> Result<(), GraphError> {
    let mut raw: Option<&'static str> = None;
    for &pass_id in order {
        let pass = &passes[pass_id.index()];
        match (pass.kind, raw) {
            (PassKind::Raw, _) => raw = Some(pass.name),
            (PassKind::Inline | PassKind::Compute, Some(raw)) => {
                return Err(GraphError::RawPassNotLast {
                    raw,
                    followed_by: pass.name,
                });
            }
            (PassKind::Inline | PassKind::Compute, None) => {}
        }
    }
    Ok(())
}

/// Edges from data flow: every reader of a resource runs after every writer of
/// it, and writers run in registration order.
///
/// Two rules, and each earns its place:
///
/// *Readers after all writers* is what makes registration order irrelevant to
/// correctness — a pass author can register the shading pass before the shadow
/// pass that feeds it and still get the right schedule. It also means a resource
/// is never read at an intermediate value, only at its final one, which is the
/// only reading an unversioned resource can offer.
///
/// *Writers in registration order* breaks the one tie data flow cannot: the
/// tonemap pass and the egui overlay both write the swapchain image and neither
/// reads the other's result, so nothing but the author's intent says which lands
/// on top. A mature graph versions the resource on each write and lets the data
/// dependency order them; v1 takes registration order instead, which is the same
/// answer for every frame written so far and is two lines instead of a
/// versioning layer.
///
/// Together they mean write-after-read cannot occur inside a frame, which is the
/// invariant `step` asserts.
fn dependency_edges(passes: &[PassDecl], resource_count: usize) -> Vec<Vec<usize>> {
    let mut edges = vec![Vec::new(); passes.len()];
    for index in 0..resource_count {
        let resource = ResourceId(index as u32);
        let touching = |want_write: bool| -> Vec<usize> {
            passes
                .iter()
                .enumerate()
                .filter(|(_, pass)| {
                    pass.accesses
                        .iter()
                        .any(|&(id, access)| id == resource && access.is_write() == want_write)
                })
                .map(|(index, _)| index)
                .collect()
        };
        let writers = touching(true);
        for pair in writers.windows(2) {
            edges[pair[0]].push(pair[1]);
        }
        for &writer in &writers {
            for reader in touching(false) {
                edges[writer].push(reader);
            }
        }
    }
    edges
}

/// Kahn's algorithm, taking the lowest-numbered ready pass each step.
///
/// The tiebreak is registration order, and it has to be *some* total order:
/// a schedule that varies run to run would make the golden barrier plan
/// unassertable, and would move a sync bug's reproduction from "this commit" to
/// "one run in five".
fn topological_order(passes: &[PassDecl]) -> Result<Vec<PassId>, GraphError> {
    let edges = dependency_edges(passes, max_resource(passes));
    let mut indegree = vec![0usize; passes.len()];
    for targets in &edges {
        for &target in targets {
            indegree[target] += 1;
        }
    }

    let mut ready: BinaryHeap<Reverse<usize>> = (0..passes.len())
        .filter(|&index| indegree[index] == 0)
        .map(Reverse)
        .collect();
    let mut order = Vec::with_capacity(passes.len());
    while let Some(Reverse(index)) = ready.pop() {
        order.push(PassId(index as u32));
        for &target in &edges[index] {
            indegree[target] -= 1;
            if indegree[target] == 0 {
                ready.push(Reverse(target));
            }
        }
    }

    if order.len() != passes.len() {
        let scheduled: HashSet<usize> = order.iter().map(|id| id.index()).collect();
        return Err(GraphError::Cycle {
            passes: passes
                .iter()
                .enumerate()
                .filter(|(index, _)| !scheduled.contains(index))
                .map(|(_, pass)| pass.name)
                .collect(),
        });
    }
    Ok(order)
}

fn max_resource(passes: &[PassDecl]) -> usize {
    passes
        .iter()
        .flat_map(|pass| pass.accesses.iter())
        .map(|(id, _)| id.index() + 1)
        .max()
        .unwrap_or(0)
}

/// Drop passes whose results nothing observes.
///
/// "Observed" means written into an imported resource, or read by a pass that is
/// itself observed — imports are the only things that outlive the frame. One
/// reverse sweep suffices because a consumer always follows its producer in the
/// topological order.
fn cull(
    order: Vec<PassId>,
    passes: &[PassDecl],
    resources: &[ResourceDecl],
) -> (Vec<PassId>, Vec<PassId>) {
    let mut needed: HashSet<ResourceId> = resources
        .iter()
        .enumerate()
        .filter(|(_, resource)| resource.is_imported())
        .map(|(index, _)| ResourceId(index as u32))
        .collect();

    let mut live = vec![false; passes.len()];
    for &pass_id in order.iter().rev() {
        let pass = &passes[pass_id.index()];
        let observed = pass
            .accesses
            .iter()
            .any(|(id, access)| access.is_write() && needed.contains(id));
        if observed {
            live[pass_id.index()] = true;
            for (id, access) in &pass.accesses {
                if !access.is_write() {
                    needed.insert(*id);
                }
            }
        }
    }

    order.into_iter().partition(|id| live[id.index()])
}

#[derive(Clone, Copy)]
struct State {
    layout: ImageLayout,
    has_write: bool,
    write_stages: PipelineStages,
    write_access: AccessFlags,
    read_stages: PipelineStages,
    /// What the last write has already been made visible to. A second reader in
    /// the same layout that needs no more than this needs no barrier.
    visible_stages: PipelineStages,
    visible_access: AccessFlags,
    /// The queue every access currently contributing to a barrier's *source*
    /// half ran on, or `None` when they did not all run on one.
    ///
    /// What tells a dependency between two dispatches in the compute tail —
    /// which nothing but its barrier expresses — from one on the segment before
    /// it, which the semaphore between them already carries. See
    /// [`schedule::narrow`].
    src_queue: Option<Queue>,
}

impl State {
    fn new(resource: &ResourceDecl) -> Self {
        Self {
            layout: resource.entry_layout(),
            has_write: false,
            write_stages: PipelineStages::empty(),
            write_access: AccessFlags::empty(),
            read_stages: PipelineStages::empty(),
            visible_stages: PipelineStages::empty(),
            visible_access: AccessFlags::empty(),
            src_queue: None,
        }
    }

    /// The state a resource starts in when it is about to be handed the memory
    /// `previous` has finished with.
    ///
    /// The layout is still the entry one — `Undefined`, because nothing of the
    /// contents survives — but the *memory* is not fresh, and the pass that
    /// dirtied it has to finish before this one starts.
    ///
    /// `previous`'s readers join its writers in `write_stages` rather than
    /// staying reads, for two reasons that point the same way. Aliasing is the
    /// one write-after-read a frame contains, and `step` sources a write from
    /// `write_stages` alone because `dependency_edges` guarantees there are no
    /// others; and a read of the old resource is a read of this memory, so it
    /// belongs in the half that the next write waits for.
    fn aliased(previous: State, resource: &ResourceDecl) -> Self {
        Self {
            layout: resource.entry_layout(),
            has_write: true,
            write_stages: previous.write_stages | previous.read_stages,
            // Only the writes made anything to flush. Kept rather than dropped
            // to an execution dependency because a transition out of
            // `Undefined` may rewrite the image's compression metadata, and a
            // driver doing that wants the previous writes available first.
            write_access: previous.write_access,
            read_stages: PipelineStages::empty(),
            visible_stages: PipelineStages::empty(),
            visible_access: AccessFlags::empty(),
            src_queue: previous.src_queue,
        }
    }
}

fn derive_barriers(
    order: &[PassId],
    passes: &[PassDecl],
    resources: &[ResourceDecl],
    queues: &[Queue],
    previous_in_group: &[Option<ResourceId>],
) -> Result<(Vec<Vec<Barrier>>, Vec<Barrier>), GraphError> {
    let mut states: Vec<State> = resources.iter().map(State::new).collect();
    let mut touched = vec![false; resources.len()];
    let mut barriers = Vec::with_capacity(order.len());

    for (slot, &pass_id) in order.iter().enumerate() {
        let pass = &passes[pass_id.index()];
        let queue = queues[slot];
        let mut pass_barriers = Vec::new();
        for &(resource, access) in &pass.accesses {
            let decl = &resources[resource.index()];

            if !access.is_write() && !states[resource.index()].has_write && !decl.is_imported() {
                return Err(GraphError::ReadBeforeWrite {
                    pass: pass.name,
                    resource: decl.name,
                });
            }

            // Taking over an aliased allocation is seeded into the state rather
            // than emitted as a barrier of its own, so it reaches the driver
            // folded into this pass's first barrier and narrows across a queue
            // boundary exactly as any other dependency does.
            //
            // After the check above rather than before: seeding sets
            // `has_write`, which would make a read-before-write read as a
            // legitimate read of what the previous tenant left.
            if !touched[resource.index()]
                && let Some(before) = previous_in_group[resource.index()]
            {
                states[resource.index()] = State::aliased(states[before.index()], decl);
            }
            touched[resource.index()] = true;
            let state = &mut states[resource.index()];

            // Whether anything the barrier's source half describes ran on the
            // other queue, which is what decides whether the semaphore between
            // the segments has already covered it. Read before `step` advances
            // the state past it.
            let crossed = state.src_queue.is_some_and(|src| src != queue);
            if let Some(mut barrier) = step(state, decl.is_image(), resource, access, queue) {
                if queue == Queue::AsyncCompute {
                    schedule::narrow(&mut barrier, crossed);
                }
                pass_barriers.push(barrier);
            }
        }
        barriers.push(pass_barriers);
    }

    // Leave every import in the layout its owner expects — for the swapchain
    // image, the one the presentation engine requires.
    let mut final_barriers = Vec::new();
    for (index, decl) in resources.iter().enumerate() {
        let Some(exit) = decl.exit_layout() else {
            continue;
        };
        let state = states[index];
        if state.layout == exit {
            continue;
        }
        final_barriers.push(Barrier {
            resource: ResourceId(index as u32),
            src_stages: nonempty(state.write_stages | state.read_stages),
            src_access: state.write_access,
            dst_stages: PipelineStages::BOTTOM_OF_PIPE,
            dst_access: AccessFlags::empty(),
            old_layout: state.layout,
            new_layout: exit,
        });
    }

    Ok((barriers, final_barriers))
}

/// Advance one resource's state by one access, returning the barrier that move
/// requires, if any.
fn step(
    state: &mut State,
    is_image: bool,
    resource: ResourceId,
    access: Access,
    queue: Queue,
) -> Option<Barrier> {
    let layout = if is_image {
        access.layout()
    } else {
        ImageLayout::Undefined
    };
    let transition = layout != state.layout;

    let barrier = if access.is_write() {
        // `dependency_edges` orders every reader after every writer, so a write
        // never follows a read of the same resource and there is no
        // write-after-read hazard to cover. Versioned resources would break
        // that; this is where it would show up first.
        debug_assert!(
            state.read_stages.is_empty(),
            "write-after-read on a resource the schedule said could not have one",
        );
        (transition || state.has_write).then(|| Barrier {
            resource,
            src_stages: nonempty(state.write_stages),
            src_access: state.write_access,
            dst_stages: access.stages(),
            dst_access: access.flags(),
            old_layout: state.layout,
            new_layout: layout,
        })
    } else {
        let already_visible = state.visible_stages.contains(access.stages())
            && state.visible_access.contains(access.flags());
        (transition || (state.has_write && !already_visible)).then(|| Barrier {
            resource,
            // A layout transition rewrites the image, so a reader that moves one
            // out from under an earlier reader has to wait for that reader as
            // well as for the last write — a write-after-read on the transition
            // itself. Only the *stages* need widening: a read dirties nothing,
            // so there is nothing of it to make available, which is why
            // `src_access` stays the write's alone. The frame's closing
            // barriers already source themselves this way; this is the same
            // rule inside it.
            //
            // Unreachable until a resource was read in two different layouts.
            // The prepass depth is the first: five passes sample it and the
            // transparency accumulation attaches it read-only between them.
            src_stages: nonempty(state.write_stages | state.read_stages),
            src_access: state.write_access,
            dst_stages: access.stages(),
            dst_access: access.flags(),
            old_layout: state.layout,
            new_layout: layout,
        })
    };

    state.layout = layout;
    if access.is_write() {
        state.has_write = true;
        state.write_stages = access.stages();
        state.write_access = access.flags();
        state.read_stages = PipelineStages::empty();
        state.visible_stages = PipelineStages::empty();
        state.visible_access = AccessFlags::empty();
        // A write resets the source half, so it also resets whose it is.
        state.src_queue = Some(queue);
    } else {
        state.read_stages |= access.stages();
        // A layout transition is itself a write, so what earlier readers were
        // given visibility of does not carry across one.
        if transition {
            state.visible_stages = access.stages();
            state.visible_access = access.flags();
        } else {
            state.visible_stages |= access.stages();
            state.visible_access |= access.flags();
        }
        // A read joins the source half rather than replacing it — a later
        // transition sources the readers it moves the layout out from under as
        // well as the write — so one reader from the other queue is enough to
        // make the whole of it the semaphore's business.
        if state.src_queue != Some(queue) {
            state.src_queue = None;
        }
    }
    barrier
}

/// `TopOfPipe` is the identity source stage: nothing has happened yet, so
/// nothing has to finish first. An empty source stage mask is not legal.
fn nonempty(stages: PipelineStages) -> PipelineStages {
    if stages.is_empty() {
        PipelineStages::TOP_OF_PIPE
    } else {
        stages
    }
}

fn derive_images(
    order: &[PassId],
    passes: &[PassDecl],
    resources: &[ResourceDecl],
) -> Vec<Option<TransientImage>> {
    let mut images = vec![None; resources.len()];
    for (index, decl) in resources.iter().enumerate() {
        let ResourceKind::Transient(desc) = decl.kind else {
            continue;
        };
        let accesses: Vec<Access> = order
            .iter()
            .flat_map(|id| passes[id.index()].accesses.iter())
            .filter(|(resource, _)| resource.index() == index)
            .map(|&(_, access)| access)
            .collect();
        if accesses.is_empty() {
            continue;
        }

        let memoryless = accesses.iter().all(|access| access.is_attachment());
        let mut usage = accesses.iter().fold(ImageUsage::empty(), |usage, access| {
            usage | access.image_usage()
        });
        if memoryless {
            usage |= ImageUsage::TRANSIENT_ATTACHMENT;
        }
        images[index] = Some(TransientImage {
            desc,
            usage,
            memoryless,
            // Both filled in by `compile` once the frame has been cut into
            // segments and its lifetimes taken; `derive_images` is about what a
            // pass declared, and these are about where the compiler put it.
            concurrent: false,
            alias: None,
        });
    }
    images
}
