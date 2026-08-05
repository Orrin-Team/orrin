use std::fmt;

use super::Access;

/// A graph that cannot be compiled.
///
/// Every variant names the pass and resource involved and says what to do about
/// it, per architecture §6: the whole reason to derive barriers from
/// declarations is that a mistake becomes a message here instead of a race that
/// only reproduces on someone else's driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    DuplicateResource {
        name: &'static str,
    },
    DuplicatePass {
        name: &'static str,
    },
    /// One pass declared two accesses to the same resource. The compiler models
    /// a pass as a single point in time, so it has no way to order them.
    ConflictingAccess {
        pass: &'static str,
        resource: &'static str,
        first: Access,
        second: Access,
    },
    /// A transient read by a pass that no writer precedes: the pass would sample
    /// undefined memory.
    ReadBeforeWrite {
        pass: &'static str,
        resource: &'static str,
    },
    Cycle {
        passes: Vec<&'static str>,
    },
    /// An inline pass was scheduled after a [`PassKind::Raw`](super::PassKind)
    /// one. See the variant's message for why v1 cannot run that.
    RawPassNotLast {
        raw: &'static str,
        followed_by: &'static str,
    },
    /// A [`PassKind::Compute`](super::PassKind) pass declared an attachment.
    /// Attachments only exist inside a render pass, and a dispatch cannot be.
    AttachmentInComputePass {
        pass: &'static str,
        resource: &'static str,
        access: Access,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::DuplicateResource { name } => write!(
                f,
                "render graph: two resources are named `{name}`; names identify \
                 resources in the frame inspector and the barrier plan, so they \
                 must be unique"
            ),
            GraphError::DuplicatePass { name } => write!(
                f,
                "render graph: two passes are named `{name}`; pass names key GPU \
                 timings, so they must be unique"
            ),
            GraphError::ConflictingAccess {
                pass,
                resource,
                first,
                second,
            } => write!(
                f,
                "render graph: pass `{pass}` declares `{resource}` as both \
                 {first:?} and {second:?}. A pass is one point in the schedule, \
                 so the two cannot be ordered against each other — split it into \
                 two passes, or declare the single access that covers both."
            ),
            GraphError::ReadBeforeWrite { pass, resource } => write!(
                f,
                "render graph: pass `{pass}` reads `{resource}`, which no earlier \
                 pass writes. A transient's contents do not survive a frame, so \
                 this samples undefined memory — register the producing pass, or \
                 import `{resource}` if something outside the graph fills it."
            ),
            GraphError::Cycle { passes } => write!(
                f,
                "render graph: passes {passes:?} form a dependency cycle — one of \
                 them reads what a later one writes. Break the cycle by \
                 splitting the shared resource into a written one and a read one."
            ),
            GraphError::RawPassNotLast { raw, followed_by } => write!(
                f,
                "render graph: raw pass `{raw}` is scheduled before `{followed_by}`, \
                 but v1 submits the frame as one command buffer and runs raw \
                 passes on the future after it. Give `{followed_by}` a dependency \
                 that places it before `{raw}`, or fold its work into an inline \
                 pass."
            ),
            GraphError::AttachmentInComputePass {
                pass,
                resource,
                access,
            } => write!(
                f,
                "render graph: compute pass `{pass}` declares `{resource}` as \
                 {access:?}, but a dispatch runs outside any render pass and an \
                 attachment only exists inside one. Read it as Sampled or write \
                 it as StorageWrite, or make `{pass}` an inline pass."
            ),
        }
    }
}

impl std::error::Error for GraphError {}
