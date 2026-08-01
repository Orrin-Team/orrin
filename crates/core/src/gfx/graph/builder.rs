use vulkano::image::ImageLayout;

use super::{
    Access, GraphError, ImageDesc, ImportedLayouts, ResourceDecl, ResourceId, ResourceKind,
};

/// Handle to a pass registered on a [`GraphBuilder`].
///
/// It is also the index the executor looks its recording closure up by: passes
/// are registered in one order and their bodies stored in the same one, so the
/// two cannot drift.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PassId(pub(super) u32);

impl PassId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PassKind {
    /// Records into the frame's primary command buffer. The executor puts the
    /// pass's barriers in front of it, so the graph's plan is what runs.
    Inline,
    /// The escape hatch: a pass that builds and submits its own command buffer
    /// (egui, whose renderer owns its submission). The executor closes the
    /// command buffer recorded so far, submits it, hands the pass the resulting
    /// future, and continues from what it returns.
    ///
    /// The cost of the escape hatch is exactly one thing: the graph cannot place
    /// this pass's barriers, so a raw pass is responsible for entering and
    /// leaving its declared resources in the layouts the plan states. Its
    /// declarations still order it and still feed resource creation, so it is
    /// visible to the schedule rather than invisible to it.
    Raw,
}

#[derive(Clone, Debug)]
pub(super) struct PassDecl {
    pub name: &'static str,
    pub kind: PassKind,
    /// In declaration order, which is the order barriers are emitted in. At most
    /// one entry per resource; a second is [`GraphError::ConflictingAccess`].
    pub accesses: Vec<(ResourceId, Access)>,
}

/// Declares the resources and passes of one frame.
///
/// Nothing here touches a `Device`: a graph is compiled from declarations
/// alone, which is why the barrier plan can be asserted on a CI runner that has
/// no GPU.
#[derive(Default)]
pub struct GraphBuilder {
    pub(super) resources: Vec<ResourceDecl>,
    pub(super) passes: Vec<PassDecl>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// An image the graph owns: allocated on compile, undefined at frame start,
    /// freed when the graph is replaced.
    pub fn create_image(&mut self, name: &'static str, desc: ImageDesc) -> ResourceId {
        self.push(name, ResourceKind::Transient(desc))
    }

    /// An image supplied per frame from outside — the acquired swapchain image,
    /// or a texture the asset system owns.
    pub fn import_image(
        &mut self,
        name: &'static str,
        desc: ImageDesc,
        entry: ImageLayout,
        exit: ImageLayout,
    ) -> ResourceId {
        self.push(
            name,
            ResourceKind::Imported(desc, ImportedLayouts { entry, exit }),
        )
    }

    pub fn import_buffer(&mut self, name: &'static str) -> ResourceId {
        self.push(name, ResourceKind::ImportedBuffer)
    }

    fn push(&mut self, name: &'static str, kind: ResourceKind) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceDecl { name, kind });
        id
    }

    pub fn pass(&mut self, name: &'static str, kind: PassKind) -> PassBuilder<'_> {
        PassBuilder {
            graph: self,
            decl: PassDecl {
                name,
                kind,
                accesses: Vec::new(),
            },
        }
    }

    pub(super) fn validate_names(&self) -> Result<(), GraphError> {
        for (index, resource) in self.resources.iter().enumerate() {
            if self.resources[..index].iter().any(|r| r.name == resource.name) {
                return Err(GraphError::DuplicateResource { name: resource.name });
            }
        }
        for (index, pass) in self.passes.iter().enumerate() {
            if self.passes[..index].iter().any(|p| p.name == pass.name) {
                return Err(GraphError::DuplicatePass { name: pass.name });
            }
        }
        Ok(())
    }
}

pub struct PassBuilder<'a> {
    graph: &'a mut GraphBuilder,
    decl: PassDecl,
}

impl PassBuilder<'_> {
    pub fn access(mut self, resource: ResourceId, access: Access) -> Self {
        self.decl.accesses.push((resource, access));
        self
    }

    /// Register the pass. The returned id indexes the executor's parallel list
    /// of recording closures.
    pub fn build(self) -> PassId {
        let id = PassId(self.graph.passes.len() as u32);
        self.graph.passes.push(self.decl);
        id
    }
}
