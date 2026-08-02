use std::fmt;

use vulkano::image::ImageLayout;
use vulkano::sync::{AccessFlags, PipelineStages};

use super::{FrameGraph, ResourceId};

/// One derived dependency: what must finish, what may then start, and the layout
/// the image changes to on the way.
///
/// Nothing in here was written by a pass author. It is the whole output of the
/// compiler that a driver would care about, which is why it is also the thing CI
/// asserts — a barrier that silently loosens is a race, and a race that only
/// shows up on one vendor's scheduler is exactly the bug class the graph exists
/// to remove.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Barrier {
    pub resource: ResourceId,
    pub src_stages: PipelineStages,
    pub src_access: AccessFlags,
    pub dst_stages: PipelineStages,
    pub dst_access: AccessFlags,
    /// Equal when no transition is needed, and both `Undefined` for buffers.
    pub old_layout: ImageLayout,
    pub new_layout: ImageLayout,
}

impl Barrier {
    pub fn is_transition(&self) -> bool {
        self.old_layout != self.new_layout
    }
}

/// The compiled frame, rendered as text.
///
/// Written by hand rather than derived so the golden file is stable against
/// vulkano's `Debug` impls: a dependency bump should not rewrite the baseline
/// that guards synchronisation.
impl fmt::Display for FrameGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (slot, &pass_id) in self.order().iter().enumerate() {
            let pass = self.pass(pass_id);
            writeln!(f, "{slot:02} pass {} [{:?}]", pass.name, pass.kind)?;
            for barrier in self.barriers_before(slot) {
                self.write_barrier(f, barrier)?;
            }
            for &(resource, access) in &pass.accesses {
                writeln!(f, "     {} {:?}", self.resource_name(resource), access)?;
            }
        }
        if !self.final_barriers().is_empty() {
            writeln!(f, "-- end of frame")?;
            for barrier in self.final_barriers() {
                self.write_barrier(f, barrier)?;
            }
        }
        for &pass_id in self.culled() {
            writeln!(f, "-- culled pass {}", self.pass(pass_id).name)?;
        }
        Ok(())
    }
}

/// A compiled graph has one useful rendering, so `{:?}` gives the same one as
/// `{}` — an assertion that fails on the plan should print the plan.
impl fmt::Debug for FrameGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl FrameGraph {
    fn write_barrier(&self, f: &mut fmt::Formatter<'_>, barrier: &Barrier) -> fmt::Result {
        write!(f, "  ! {}", self.resource_name(barrier.resource))?;
        if barrier.is_transition() {
            write!(f, " {:?}->{:?}", barrier.old_layout, barrier.new_layout)?;
        }
        writeln!(
            f,
            " src {}/{} dst {}/{}",
            stages(barrier.src_stages),
            access(barrier.src_access),
            stages(barrier.dst_stages),
            access(barrier.dst_access),
        )
    }
}

/// Names every listed flag the value carries, and reports anything left over
/// rather than dropping it — a flag the table has never heard of is a hole in
/// the golden baseline, so it has to be visible.
macro_rules! flag_names {
    ($value:expr, $($flag:expr => $name:literal),+ $(,)?) => {{
        let value = $value;
        let mut rest = value;
        let mut out = String::new();
        $(
            if value.contains($flag) {
                if !out.is_empty() {
                    out.push('|');
                }
                out.push_str($name);
                rest = rest.difference($flag);
            }
        )+
        if !rest.is_empty() {
            if !out.is_empty() {
                out.push('|');
            }
            out.push_str("<unnamed>");
        }
        if out.is_empty() { "-".to_string() } else { out }
    }};
}

fn stages(value: PipelineStages) -> String {
    flag_names!(
        value,
        PipelineStages::TOP_OF_PIPE => "TopOfPipe",
        PipelineStages::DRAW_INDIRECT => "DrawIndirect",
        PipelineStages::VERTEX_INPUT => "VertexInput",
        PipelineStages::VERTEX_SHADER => "VertexShader",
        PipelineStages::FRAGMENT_SHADER => "FragmentShader",
        PipelineStages::EARLY_FRAGMENT_TESTS => "EarlyFragmentTests",
        PipelineStages::LATE_FRAGMENT_TESTS => "LateFragmentTests",
        PipelineStages::COLOR_ATTACHMENT_OUTPUT => "ColorAttachmentOutput",
        PipelineStages::COMPUTE_SHADER => "ComputeShader",
        PipelineStages::ALL_TRANSFER => "AllTransfer",
        PipelineStages::BOTTOM_OF_PIPE => "BottomOfPipe",
    )
}

fn access(value: AccessFlags) -> String {
    flag_names!(
        value,
        AccessFlags::INDIRECT_COMMAND_READ => "IndirectCommandRead",
        AccessFlags::INDEX_READ => "IndexRead",
        AccessFlags::VERTEX_ATTRIBUTE_READ => "VertexAttributeRead",
        AccessFlags::UNIFORM_READ => "UniformRead",
        AccessFlags::SHADER_SAMPLED_READ => "ShaderSampledRead",
        AccessFlags::SHADER_STORAGE_READ => "ShaderStorageRead",
        AccessFlags::SHADER_STORAGE_WRITE => "ShaderStorageWrite",
        AccessFlags::COLOR_ATTACHMENT_READ => "ColorAttachmentRead",
        AccessFlags::COLOR_ATTACHMENT_WRITE => "ColorAttachmentWrite",
        AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ => "DepthStencilAttachmentRead",
        AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE => "DepthStencilAttachmentWrite",
        AccessFlags::TRANSFER_READ => "TransferRead",
        AccessFlags::TRANSFER_WRITE => "TransferWrite",
    )
}
