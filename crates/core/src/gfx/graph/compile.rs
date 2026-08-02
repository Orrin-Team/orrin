use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use vulkano::image::{ImageLayout, ImageUsage};
use vulkano::sync::{AccessFlags, PipelineStages};

use super::{
    Access, Barrier, GraphBuilder, GraphError, ImageDesc, PassDecl, PassId, PassKind, ResourceDecl,
    ResourceId, ResourceKind,
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
    let GraphBuilder { resources, passes } = builder;

    for pass in &passes {
        check_single_access_per_resource(pass, &resources)?;
    }

    let order = topological_order(&passes)?;
    let (order, culled) = cull(order, &passes, &resources);
    check_raw_passes_last(&order, &passes)?;
    let (barriers, final_barriers) = derive_barriers(&order, &passes, &resources)?;
    let images = derive_images(&order, &passes, &resources);

    Ok(FrameGraph {
        resources,
        passes,
        order,
        barriers,
        final_barriers,
        culled,
        images,
    })
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
            (PassKind::Inline, Some(raw)) => {
                return Err(GraphError::RawPassNotLast {
                    raw,
                    followed_by: pass.name,
                });
            }
            (PassKind::Inline, None) => {}
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
        }
    }
}

fn derive_barriers(
    order: &[PassId],
    passes: &[PassDecl],
    resources: &[ResourceDecl],
) -> Result<(Vec<Vec<Barrier>>, Vec<Barrier>), GraphError> {
    let mut states: Vec<State> = resources.iter().map(State::new).collect();
    let mut barriers = Vec::with_capacity(order.len());

    for &pass_id in order {
        let pass = &passes[pass_id.index()];
        let mut pass_barriers = Vec::new();
        for &(resource, access) in &pass.accesses {
            let decl = &resources[resource.index()];
            let state = &mut states[resource.index()];

            if !access.is_write() && !state.has_write && !decl.is_imported() {
                return Err(GraphError::ReadBeforeWrite {
                    pass: pass.name,
                    resource: decl.name,
                });
            }

            if let Some(barrier) = step(state, decl.is_image(), resource, access) {
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
            src_stages: nonempty(state.write_stages),
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
        });
    }
    images
}
