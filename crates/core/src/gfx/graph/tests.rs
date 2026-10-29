use crate::gfx::format::Format;

use super::*;

fn image() -> ImageDesc {
    ImageDesc::new(Format::Rgba8Unorm)
}

fn names(graph: &FrameGraph) -> Vec<&'static str> {
    graph
        .order()
        .iter()
        .map(|&id| graph.pass_name(id))
        .collect()
}

#[test]
fn a_reader_is_scheduled_after_its_writer_whatever_order_they_were_registered_in() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let intermediate = builder.create_image("intermediate", image());

    builder
        .pass("consumer", PassKind::Inline)
        .access(intermediate, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();
    builder
        .pass("producer", PassKind::Inline)
        .access(intermediate, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    assert_eq!(names(&graph), vec!["producer", "consumer"]);
}

/// Registration order is the tiebreak, and it has to be honoured exactly: the
/// tonemap pass and the egui overlay both write the swapchain image with no data
/// dependency between them, so nothing else decides which one lands on top.
#[test]
fn two_writers_of_one_resource_run_in_registration_order() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    builder
        .pass("scene", PassKind::Inline)
        .access(target, Access::ColorAttachment)
        .build();
    builder
        .pass("overlay", PassKind::Raw)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    assert_eq!(names(&graph), vec!["scene", "overlay"]);
}

#[test]
fn independent_passes_keep_a_stable_order() {
    let compile_once = || {
        let mut builder = GraphBuilder::new();
        let a = builder.import_image("a", image(), ImageLayout::Undefined, ImageLayout::General);
        let b = builder.import_image("b", image(), ImageLayout::Undefined, ImageLayout::General);
        builder
            .pass("first", PassKind::Inline)
            .access(a, Access::ColorAttachment)
            .build();
        builder
            .pass("second", PassKind::Inline)
            .access(b, Access::ColorAttachment)
            .build();
        names(&compile(builder).unwrap())
    };
    assert_eq!(compile_once(), vec!["first", "second"]);
    assert_eq!(compile_once(), compile_once());
}

#[test]
fn a_cycle_names_the_passes_in_it() {
    let mut builder = GraphBuilder::new();
    let a = builder.import_image("a", image(), ImageLayout::Undefined, ImageLayout::General);
    let b = builder.import_image("b", image(), ImageLayout::Undefined, ImageLayout::General);
    builder
        .pass("left", PassKind::Inline)
        .access(a, Access::ColorAttachment)
        .access(b, Access::Sampled)
        .build();
    builder
        .pass("right", PassKind::Inline)
        .access(b, Access::ColorAttachment)
        .access(a, Access::Sampled)
        .build();

    let error = compile(builder).unwrap_err();
    let GraphError::Cycle { mut passes } = error else {
        panic!("expected a cycle, got {error}");
    };
    passes.sort_unstable();
    assert_eq!(passes, vec!["left", "right"]);
}

#[test]
fn sampling_a_transient_nothing_writes_is_rejected_by_name() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let never_written = builder.create_image("shadow_map", image());
    builder
        .pass("shading", PassKind::Inline)
        .access(never_written, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    assert_eq!(
        compile(builder).unwrap_err(),
        GraphError::ReadBeforeWrite {
            pass: "shading",
            resource: "shadow_map",
        }
    );
}

/// Reading an import that no pass writes is fine — something outside the frame
/// filled it. Only a transient can be read before anything wrote it.
#[test]
fn reading_an_unwritten_import_is_allowed() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let objects = builder.import_buffer("objects");
    builder
        .pass("shading", PassKind::Inline)
        .access(objects, Access::StorageRead)
        .access(target, Access::ColorAttachment)
        .build();

    assert!(compile(builder).is_ok());
}

#[test]
fn two_accesses_to_one_resource_in_one_pass_are_rejected_by_name() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    builder
        .pass("feedback", PassKind::Inline)
        .access(target, Access::ColorAttachment)
        .access(target, Access::Sampled)
        .build();

    assert_eq!(
        compile(builder).unwrap_err(),
        GraphError::ConflictingAccess {
            pass: "feedback",
            resource: "target",
            first: Access::ColorAttachment,
            second: Access::Sampled,
        }
    );
}

#[test]
fn a_pass_whose_output_nothing_reads_is_culled() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let orphan = builder.create_image("orphan", image());
    builder
        .pass("wasted", PassKind::Inline)
        .access(orphan, Access::ColorAttachment)
        .build();
    builder
        .pass("present", PassKind::Inline)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    assert_eq!(names(&graph), vec!["present"]);
    assert_eq!(graph.culled().len(), 1);
    assert_eq!(graph.pass_name(graph.culled()[0]), "wasted");
    // Nothing runs that touches it, so nothing allocates it either.
    assert!(
        graph
            .transient_images()
            .all(|(id, _)| graph.resource_name(id) != "orphan")
    );
}

/// The usage flags an image is created with come from what passes declared, so
/// forgetting to declare a read is a creation-time failure rather than a
/// validation-layer surprise later.
#[test]
fn image_usage_is_the_union_of_declared_accesses() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let color = builder.create_image("color", image());
    builder
        .pass("produce", PassKind::Inline)
        .access(color, Access::ColorAttachment)
        .build();
    builder
        .pass("consume", PassKind::Inline)
        .access(color, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let (_, created) = graph.transient_images().next().unwrap();
    assert_eq!(
        created.usage,
        ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED
    );
    assert!(!created.memoryless);
}

/// An image only ever used as an attachment never leaves the render pass that
/// wrote it, so it can stay in tile memory. This is what keeps the 4x MSAA HDR
/// target free of DRAM cost on Apple hardware.
#[test]
fn an_attachment_only_image_is_marked_memoryless() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let msaa = builder.create_image("msaa", image());
    builder
        .pass("draw", PassKind::Inline)
        .access(msaa, Access::ColorAttachment)
        .access(target, Access::ResolveAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let (_, created) = graph.transient_images().next().unwrap();
    assert!(created.memoryless);
    assert!(created.usage.contains(ImageUsage::TRANSIENT_ATTACHMENT));
}

#[test]
fn a_write_then_read_produces_one_transition_and_one_dependency() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let color = builder.create_image("color", image());
    builder
        .pass("produce", PassKind::Inline)
        .access(color, Access::ColorAttachment)
        .build();
    builder
        .pass("consume", PassKind::Inline)
        .access(color, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();

    let [produce] = graph.barriers_before(0) else {
        panic!("expected one barrier before `produce`");
    };
    assert_eq!(produce.old_layout, ImageLayout::Undefined);
    assert_eq!(produce.new_layout, ImageLayout::ColorAttachmentOptimal);
    assert_eq!(produce.src_stages, PipelineStages::TOP_OF_PIPE);

    let consume = graph.barriers_before(1);
    assert_eq!(consume.len(), 2);
    assert_eq!(consume[0].old_layout, ImageLayout::ColorAttachmentOptimal);
    assert_eq!(consume[0].new_layout, ImageLayout::ShaderReadOnlyOptimal);
    assert_eq!(
        consume[0].src_stages,
        PipelineStages::COLOR_ATTACHMENT_OUTPUT
    );
}

/// Two passes sampling the same image in the same layout are unordered and need
/// nothing between them — the whole reason reads are declared separately from
/// writes.
#[test]
fn a_second_reader_in_the_same_layout_needs_no_barrier() {
    let mut builder = GraphBuilder::new();
    let left = builder.import_image(
        "left",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let right = builder.import_image(
        "right",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let color = builder.create_image("color", image());
    builder
        .pass("produce", PassKind::Inline)
        .access(color, Access::ColorAttachment)
        .build();
    builder
        .pass("read_once", PassKind::Inline)
        .access(color, Access::Sampled)
        .access(left, Access::ColorAttachment)
        .build();
    builder
        .pass("read_twice", PassKind::Inline)
        .access(color, Access::Sampled)
        .access(right, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let slot = graph
        .order()
        .iter()
        .position(|&id| graph.pass_name(id) == "read_twice")
        .unwrap();
    assert!(
        graph
            .barriers_before(slot)
            .iter()
            .all(|barrier| graph.resource_name(barrier.resource) != "color")
    );
}

/// A layout transition rewrites the image, so a reader that moves one out from
/// under an earlier reader has to wait for that reader too — a write-after-read
/// on the transition itself, and the one hazard `dependency_edges` cannot rule
/// out, because both passes really are readers.
///
/// The frame that made this reachable is transparency's: five passes sample the
/// prepass depth, and the accumulation attaches it read-only between them. A
/// barrier sourced from the prepass's write alone would let the transition run
/// while SSAO was still sampling.
#[test]
fn a_transition_waits_for_the_readers_it_moves_the_layout_out_from_under() {
    let mut builder = GraphBuilder::new();
    let present = builder.import_image(
        "present",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    // Imported so the sampling pass is observed and survives culling — a
    // culled reader is not in the schedule and has no layout to be moved out
    // from under.
    let probe = builder.import_buffer("probe");
    let depth = builder.create_image("depth", ImageDesc::new(Format::D32Float));
    builder
        .pass("write_depth", PassKind::Inline)
        .access(depth, Access::DepthAttachment)
        .build();
    builder
        .pass("sample_depth", PassKind::Compute)
        .access(depth, Access::Sampled)
        .access(probe, Access::StorageWrite)
        .build();
    builder
        .pass("attach_depth", PassKind::Inline)
        .access(depth, Access::DepthAttachmentRead)
        .access(present, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let slot = graph
        .order()
        .iter()
        .position(|&id| graph.pass_name(id) == "attach_depth")
        .unwrap();
    let barrier = graph
        .barriers_before(slot)
        .iter()
        .find(|barrier| graph.resource_name(barrier.resource) == "depth")
        .expect("moving to a read-only depth layout needs a barrier");

    assert_eq!(barrier.old_layout, ImageLayout::ShaderReadOnlyOptimal);
    assert_eq!(barrier.new_layout, ImageLayout::DepthStencilReadOnlyOptimal);
    assert!(
        barrier.src_stages.contains(PipelineStages::COMPUTE_SHADER),
        "the transition must wait for the sampling pass: {:?}",
        barrier.src_stages,
    );
}

/// An acquired swapchain image arrives `Undefined` and must be handed back as
/// `PresentSrc`; nothing in the frame declares that, so the compiler owes it.
#[test]
fn an_import_is_left_in_its_declared_exit_layout() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "swapchain",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    builder
        .pass("present", PassKind::Inline)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let [barrier] = graph.final_barriers() else {
        panic!("expected one closing barrier");
    };
    assert_eq!(barrier.old_layout, ImageLayout::ColorAttachmentOptimal);
    assert_eq!(barrier.new_layout, ImageLayout::PresentSrc);
    assert_eq!(barrier.src_stages, PipelineStages::COLOR_ATTACHMENT_OUTPUT);
}

/// v1 records the frame as one command buffer and runs raw passes on the future
/// after it, so an inline pass scheduled behind one has nowhere to go. The
/// constraint is checked rather than assumed: discovering it as a submission
/// failure means reading a vulkano resource-tracking error instead of a sentence
/// naming both passes.
#[test]
fn an_inline_pass_after_a_raw_one_is_rejected_by_name() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scratch = builder.create_image("scratch", image());
    builder
        .pass("overlay", PassKind::Raw)
        .access(target, Access::ColorAttachment)
        .access(scratch, Access::ColorAttachment)
        .build();
    builder
        .pass("after", PassKind::Inline)
        .access(scratch, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    assert_eq!(
        compile(builder).unwrap_err(),
        GraphError::RawPassNotLast {
            raw: "overlay",
            followed_by: "after",
        }
    );
}

/// A dispatch runs outside any render pass, so a compute pass has nothing to
/// attach an attachment to. Left unchecked this compiles into a plan whose
/// framebuffer the executor never builds, and surfaces as a validation message
/// about a layout nothing reached.
#[test]
fn an_attachment_declared_by_a_compute_pass_is_rejected_by_name() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    builder
        .pass("histogram", PassKind::Compute)
        .access(target, Access::ColorAttachment)
        .build();

    assert_eq!(
        compile(builder).unwrap_err(),
        GraphError::AttachmentInComputePass {
            pass: "histogram",
            resource: "target",
            access: Access::ColorAttachment,
        }
    );
}

/// A compute pass is scheduled and synchronised exactly like an inline one — the
/// kind changes how the executor brackets it, not where it lands or what has to
/// finish first. Worth pinning, because the alternative reading (compute is a
/// separate queue, or floats to the front) is the one someone will assume.
#[test]
fn a_compute_pass_is_ordered_by_data_flow_like_any_other() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let measured = builder.import_buffer("measured");
    let scene = builder.create_image("scene", image());

    builder
        .pass("present", PassKind::Inline)
        .access(measured, Access::StorageRead)
        .access(target, Access::ColorAttachment)
        .build();
    builder
        .pass("measure", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(measured, Access::StorageWrite)
        .build();
    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    assert_eq!(names(&graph), vec!["draw", "measure", "present"]);
}

#[test]
fn duplicate_names_are_rejected() {
    let mut builder = GraphBuilder::new();
    builder.create_image("color", image());
    builder.create_image("color", image());
    assert_eq!(
        compile(builder).unwrap_err(),
        GraphError::DuplicateResource { name: "color" }
    );
}

/// A downsample chain cannot add its upsampled result back into the level it
/// came from, and this is the test that says why.
///
/// `down1` reads `mip0` and `up0` writes it, so "readers after all writers"
/// puts `down1` after `up0` — while `up0` reads `mip1`, which `down1` produces.
/// That is a cycle, and it is a consequence of resources being unversioned:
/// with a version per write the two would be different resources and the data
/// dependency would order them. Until then a bloom chain needs a separate
/// image to accumulate into, which is why `frame::declare` registers an up
/// chain alongside the down chain rather than blending in place.
#[test]
fn an_upsample_cannot_accumulate_into_the_level_it_read() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scene = builder.create_image("scene", image());
    let mip0 = builder.create_image("mip0", image());
    let mip1 = builder.create_image("mip1", image());

    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("down0", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(mip0, Access::StorageWrite)
        .build();
    builder
        .pass("down1", PassKind::Compute)
        .access(mip0, Access::Sampled)
        .access(mip1, Access::StorageWrite)
        .build();
    builder
        .pass("up0", PassKind::Compute)
        .access(mip1, Access::Sampled)
        .access(mip0, Access::StorageWrite)
        .build();
    builder
        .pass("composite", PassKind::Inline)
        .access(mip0, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let error = compile(builder).unwrap_err();
    let GraphError::Cycle { passes } = error else {
        panic!("expected a cycle, got {error}");
    };
    assert!(
        passes.contains(&"down1") && passes.contains(&"up0"),
        "{passes:?}"
    );
}

/// The default, and what every test above compiles: one queue, one segment,
/// covering the whole order. A frame that asked for nothing else must be
/// planned exactly as it was before there was anything to ask for.
#[test]
fn a_frame_that_did_not_ask_for_a_second_queue_is_one_graphics_segment() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scene = builder.create_image("scene", image());
    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("post", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(target, Access::StorageWrite)
        .build();

    let graph = compile(builder).unwrap();
    assert_eq!(graph.segments().len(), 1);
    assert_eq!(graph.segments()[0].queue, Queue::Graphics);
    assert_eq!(graph.segments()[0].passes, 0..2);
}

/// The shape the split exists to produce: the frame's graphics head, the
/// compute tail behind it, and the one graphics pass that puts the result on
/// the swapchain. Three segments and exactly one queue change in the middle,
/// because every queue change costs a semaphore and an ownership transfer.
#[test]
fn the_compute_tail_is_split_onto_the_async_queue() {
    let mut builder = GraphBuilder::new();
    builder.request_async_compute();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scene = builder.create_image("scene", image());
    let blurred = builder.create_image("blurred", image());
    let graded = builder.create_image("graded", image());
    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("blur", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(blurred, Access::StorageWrite)
        .build();
    builder
        .pass("grade", PassKind::Compute)
        .access(blurred, Access::Sampled)
        .access(graded, Access::StorageWrite)
        .build();
    builder
        .pass("tonemap", PassKind::Inline)
        .access(graded, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let queues: Vec<_> = graph.segments().iter().map(|s| s.queue).collect();
    assert_eq!(
        queues,
        vec![Queue::Graphics, Queue::AsyncCompute, Queue::Graphics],
    );
    assert_eq!(graph.segments()[0].passes, 0..1);
    assert_eq!(graph.segments()[1].passes, 1..3);
    assert_eq!(graph.segments()[2].passes, 3..4);
}

/// A compute pass with graphics work still to come after it stays on the
/// graphics queue.
///
/// The frame's interleaved dispatches — the depth pyramid, the fog grid, the
/// transparency composite — are each one queue change out and one back for a
/// pass the graphics queue would have run anyway. The tail is the only place
/// where crossing pays, because it is the only place where what follows on the
/// graphics queue belongs to the *next* frame.
#[test]
fn a_compute_pass_with_graphics_work_after_it_stays_on_the_graphics_queue() {
    let mut builder = GraphBuilder::new();
    builder.request_async_compute();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let depth = builder.create_image("depth", image());
    let pyramid = builder.create_image("pyramid", image());
    let scene = builder.create_image("scene", image());
    let graded = builder.create_image("graded", image());
    builder
        .pass("prepass", PassKind::Inline)
        .access(depth, Access::ColorAttachment)
        .build();
    builder
        .pass("pyramid", PassKind::Compute)
        .access(depth, Access::Sampled)
        .access(pyramid, Access::StorageWrite)
        .build();
    builder
        .pass("forward", PassKind::Inline)
        .access(pyramid, Access::Sampled)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("grade", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(graded, Access::StorageWrite)
        .build();
    builder
        .pass("tonemap", PassKind::Inline)
        .access(graded, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    assert_eq!(
        graph.segments().iter().map(|s| s.queue).collect::<Vec<_>>(),
        vec![Queue::Graphics, Queue::AsyncCompute, Queue::Graphics],
    );
    // `pyramid` is inside the head, not a segment of its own.
    assert_eq!(graph.segments()[0].passes, 0..3);
    assert_eq!(graph.segments()[1].passes, 3..4);
    assert_eq!(graph.segments()[2].passes, 4..5);
}

/// A resource written on one queue and read on the other has to be created
/// able to be: an image in `SharingMode::Exclusive` belongs to one queue family
/// at a time, and a read from the other is undefined without an ownership
/// transfer nobody records. The compiler names them so the executor can create
/// them concurrent.
///
/// Concurrent rather than a derived release/acquire pair, which is the more
/// precise answer and was the first design: ownership is per resource *and*
/// wraps across the frame boundary, so the pair only stays balanced if the
/// frame also hands every resource back to whichever queue touches it first
/// next frame — an invariant that fails silently, on contents rather than on a
/// validation message, and fails on drivers this machine cannot run. What
/// concurrent costs instead is colour compression, on these resources only, and
/// that is a number rather than a risk.
#[test]
fn a_resource_read_from_a_second_queue_is_named_as_concurrent() {
    let mut builder = GraphBuilder::new();
    builder.request_async_compute();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scene = builder.create_image("scene", image());
    let scratch = builder.create_image("scratch", image());
    let graded = builder.create_image("graded", image());
    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("blur", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(scratch, Access::StorageWrite)
        .build();
    builder
        .pass("grade", PassKind::Compute)
        .access(scratch, Access::Sampled)
        .access(graded, Access::StorageWrite)
        .build();
    builder
        .pass("tonemap", PassKind::Inline)
        .access(graded, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let shared: Vec<&str> = graph
        .concurrent()
        .map(|id| graph.resource_name(id))
        .collect();
    // `scene` is written on graphics and sampled on compute; `graded` is
    // written on compute and sampled on graphics; `target` is graphics only.
    assert!(shared.contains(&"scene"), "{shared:?}");
    assert!(shared.contains(&"graded"), "{shared:?}");
    assert!(!shared.contains(&"target"), "{shared:?}");
    // `scratch` never leaves the async segment.
    assert!(!shared.contains(&"scratch"), "{shared:?}");
}

/// The barriers the frame already derived are unchanged by the split.
///
/// A semaphore between two segments makes every write before the signal
/// available to everything after the wait, so the only thing a queue boundary
/// adds over an ordinary pass boundary is that execution dependency. The
/// layout transition the reader needs is the one the compiler always emitted,
/// in the place it always emitted it — which is also why the golden plan does
/// not move when a frame gains a second queue.
#[test]
fn the_split_does_not_move_a_barrier() {
    let build = |async_compute: bool| {
        let mut builder = GraphBuilder::new();
        if async_compute {
            builder.request_async_compute();
        }
        let target = builder.import_image(
            "target",
            image(),
            ImageLayout::Undefined,
            ImageLayout::PresentSrc,
        );
        let scene = builder.create_image("scene", image());
        let graded = builder.create_image("graded", image());
        builder
            .pass("draw", PassKind::Inline)
            .access(scene, Access::ColorAttachment)
            .build();
        builder
            .pass("grade", PassKind::Compute)
            .access(scene, Access::Sampled)
            .access(graded, Access::StorageWrite)
            .build();
        builder
            .pass("tonemap", PassKind::Inline)
            .access(graded, Access::Sampled)
            .access(target, Access::ColorAttachment)
            .build();
        compile(builder).unwrap()
    };

    let one = build(false);
    let two = build(true);
    assert_eq!(one.order(), two.order());

    // Same resources, same layouts, same place, whichever queue the pass landed
    // on. Only the stage masks of the async segment differ.
    let transitions =
        |graph: &FrameGraph, slot: usize| -> Vec<(&'static str, ImageLayout, ImageLayout)> {
            graph
                .barriers_before(slot)
                .iter()
                .map(|barrier| {
                    (
                        graph.resource_name(barrier.resource),
                        barrier.old_layout,
                        barrier.new_layout,
                    )
                })
                .collect()
        };
    for slot in 0..one.order().len() {
        assert_eq!(transitions(&one, slot), transitions(&two, slot), "{slot}");
        if two.slot_queue(slot) == Queue::Graphics {
            assert_eq!(
                one.barriers_before(slot),
                two.barriers_before(slot),
                "{slot}"
            );
        }
    }
    assert_eq!(one.final_barriers(), two.final_barriers());
}

/// Every slot is covered exactly once, in order — the same property the
/// recording partition has, and for the same reason: a segment that dropped a
/// slot would drop a pass.
#[test]
fn segments_cover_every_slot_in_order() {
    let mut builder = GraphBuilder::new();
    builder.request_async_compute();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scene = builder.create_image("scene", image());
    let graded = builder.create_image("graded", image());
    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("grade", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(graded, Access::StorageWrite)
        .build();
    builder
        .pass("tonemap", PassKind::Inline)
        .access(graded, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();
    builder
        .pass("overlay", PassKind::Raw)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let mut next = 0;
    for segment in graph.segments() {
        assert_eq!(segment.passes.start, next);
        next = segment.passes.end;
    }
    assert_eq!(next, graph.order().len());
    // The raw pass belongs to the trailing graphics segment: it submits itself,
    // and it submits on the queue the tonemap it draws over ran on.
    assert_eq!(graph.segments().last().unwrap().queue, Queue::Graphics);
}

/// A frame with nothing to put on the second queue gets one segment, even
/// having asked for one. The split is derived from the frame, not switched on.
#[test]
fn a_frame_with_no_compute_tail_is_not_split() {
    let mut builder = GraphBuilder::new();
    builder.request_async_compute();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scene = builder.create_image("scene", image());
    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("tonemap", PassKind::Inline)
        .access(scene, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    assert_eq!(graph.segments().len(), 1);
    assert_eq!(graph.segments()[0].queue, Queue::Graphics);
}

/// A barrier recorded on the async queue may only name stages that queue
/// supports.
///
/// `vkCmdPipelineBarrier2` on a compute-only family rejects
/// `ColorAttachmentOutput` outright — VUID-vkCmdPipelineBarrier2-srcStageMask-03849
/// — and the first barrier of the compute tail sources exactly that, because
/// what it waits for is the forward pass's colour write. The semaphore between
/// the two segments is what really carries that dependency: a wait makes every
/// write submitted before the signal both available and visible, so the barrier
/// is left with the layout transition and the visibility half, and sources
/// itself at `TopOfPipe` with no access.
///
/// Narrowed here rather than where it is recorded, because it is a property of
/// the plan — the golden file should show the frame the driver is actually
/// told about.
#[test]
fn a_barrier_on_the_async_queue_names_only_stages_that_queue_has() {
    let mut builder = GraphBuilder::new();
    builder.request_async_compute();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scene = builder.create_image("scene", image());
    let graded = builder.create_image("graded", image());
    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("grade", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(graded, Access::StorageWrite)
        .build();
    builder
        .pass("tonemap", PassKind::Inline)
        .access(graded, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let compute_only = PipelineStages::TOP_OF_PIPE
        | PipelineStages::DRAW_INDIRECT
        | PipelineStages::COMPUTE_SHADER
        | PipelineStages::ALL_TRANSFER
        | PipelineStages::BOTTOM_OF_PIPE
        | PipelineStages::HOST
        | PipelineStages::ALL_COMMANDS;

    for slot in 0..graph.order().len() {
        if graph.slot_queue(slot) != Queue::AsyncCompute {
            continue;
        }
        for barrier in graph.barriers_before(slot) {
            assert!(
                compute_only.contains(barrier.src_stages),
                "slot {slot}: {:?}",
                barrier.src_stages,
            );
            assert!(
                compute_only.contains(barrier.dst_stages),
                "slot {slot}: {:?}",
                barrier.dst_stages,
            );
        }
    }

    // And specifically: the one that used to source the colour write now
    // sources nothing, because the semaphore already did.
    let scene_barrier = graph
        .barriers_before(1)
        .iter()
        .find(|barrier| graph.resource_name(barrier.resource) == "scene")
        .expect("the tail still transitions the image it samples");
    assert_eq!(scene_barrier.src_stages, PipelineStages::TOP_OF_PIPE);
    assert_eq!(scene_barrier.src_access, AccessFlags::empty());
    assert_eq!(
        scene_barrier.old_layout,
        ImageLayout::ColorAttachmentOptimal
    );
    assert_eq!(scene_barrier.new_layout, ImageLayout::ShaderReadOnlyOptimal);
    assert_eq!(scene_barrier.dst_stages, PipelineStages::COMPUTE_SHADER);
}

/// The graphics segments keep every stage they had. Narrowing is for the queue
/// that cannot express them, not a general loosening.
#[test]
fn a_barrier_on_the_graphics_queue_keeps_its_stages() {
    let mut builder = GraphBuilder::new();
    builder.request_async_compute();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scene = builder.create_image("scene", image());
    let graded = builder.create_image("graded", image());
    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("grade", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(graded, Access::StorageWrite)
        .build();
    builder
        .pass("tonemap", PassKind::Inline)
        .access(graded, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    let graded_barrier = graph
        .barriers_before(2)
        .iter()
        .find(|barrier| graph.resource_name(barrier.resource) == "graded")
        .expect("the tonemap transitions what it samples");
    assert_eq!(graded_barrier.src_stages, Access::StorageWrite.stages());
    assert_eq!(graded_barrier.dst_stages, Access::Sampled.stages());
}

/// A dependency between two dispatches inside the compute tail survives the
/// narrowing.
///
/// The bloom chain is a run of these: each level reads the one before it, on
/// the same queue, with nothing between them but the barrier. Nothing else
/// expresses that ordering — there is no semaphore inside a segment — so
/// dropping it is a race, and it is the exact race the first narrowing this
/// module grew would have caused, because `Access::stages` widens every shader
/// access to vertex | fragment | compute and every one of those barriers
/// therefore names two stages the queue does not have.
#[test]
fn a_dependency_inside_the_async_tail_is_kept() {
    let mut builder = GraphBuilder::new();
    builder.request_async_compute();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let scene = builder.create_image("scene", image());
    let down = builder.create_image("down", image());
    let up = builder.create_image("up", image());
    builder
        .pass("draw", PassKind::Inline)
        .access(scene, Access::ColorAttachment)
        .build();
    builder
        .pass("downsample", PassKind::Compute)
        .access(scene, Access::Sampled)
        .access(down, Access::StorageWrite)
        .build();
    builder
        .pass("upsample", PassKind::Compute)
        .access(down, Access::Sampled)
        .access(up, Access::StorageWrite)
        .build();
    builder
        .pass("tonemap", PassKind::Inline)
        .access(up, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    assert_eq!(graph.slot_queue(2), Queue::AsyncCompute);
    let barrier = graph
        .barriers_before(2)
        .iter()
        .find(|barrier| graph.resource_name(barrier.resource) == "down")
        .expect("the upsample waits for the level it reads");

    // Narrowed to the one stage this queue has, and not collapsed: the write it
    // waits for happened here, so this barrier is the only thing that orders
    // the two.
    assert_eq!(barrier.src_stages, PipelineStages::COMPUTE_SHADER);
    assert_eq!(
        barrier.src_access,
        Access::StorageWrite.flags(),
        "the write must still be made available",
    );
    assert_eq!(barrier.dst_stages, PipelineStages::COMPUTE_SHADER);
    assert_eq!(barrier.dst_access, Access::Sampled.flags());
    assert_eq!(barrier.old_layout, ImageLayout::General);
    assert_eq!(barrier.new_layout, ImageLayout::ShaderReadOnlyOptimal);
}

/// A four-pass chain where `first` is finished with before `second` is touched,
/// so the two are never alive at once and one allocation can serve both.
fn aliasable_chain(second: ImageDesc) -> FrameGraph {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let first = builder.create_image("first", image());
    let second = builder.create_image("second", second);

    builder
        .pass("write_first", PassKind::Inline)
        .access(first, Access::ColorAttachment)
        .build();
    builder
        .pass("read_first", PassKind::Inline)
        .access(first, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();
    builder
        .pass("write_second", PassKind::Inline)
        .access(second, Access::ColorAttachment)
        .build();
    builder
        .pass("read_second", PassKind::Inline)
        .access(second, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    compile(builder).unwrap()
}

fn group_names(graph: &FrameGraph) -> Vec<Vec<&'static str>> {
    graph
        .alias_groups()
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|&id| graph.resource_name(id))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The whole feature: two images the frame is never holding at the same time
/// are one allocation, and the graph is the only thing that knows it.
#[test]
fn two_transients_that_are_never_alive_at_once_share_an_allocation() {
    let graph = aliasable_chain(image());
    assert_eq!(group_names(&graph), vec![vec!["first", "second"]]);
    // In lifetime order, because that is the order the memory changes hands in.
    let (_, second) = graph
        .transient_images()
        .find(|(id, _)| graph.resource_name(*id) == "second")
        .unwrap();
    assert_eq!(second.alias, Some(0));
}

/// A resource is live from the first slot that touches it to the last, and
/// nothing outside that window may be handed its memory.
#[test]
fn a_lifetime_spans_the_slots_that_touch_the_resource() {
    let graph = aliasable_chain(image());
    let lifetime = |name: &str| {
        let (id, _) = graph
            .transient_images()
            .find(|(id, _)| graph.resource_name(*id) == name)
            .unwrap();
        graph.lifetime(id).unwrap()
    };
    assert_eq!(lifetime("first"), 0..2);
    assert_eq!(lifetime("second"), 2..4);
}

/// The reason the two halves of a ping-pong stay two allocations: `blur_x` is
/// still being read in the slot `blur_y` is first written.
#[test]
fn transients_alive_in_the_same_slot_are_never_aliased() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let blur_x = builder.create_image("blur_x", image());
    let blur_y = builder.create_image("blur_y", image());

    builder
        .pass("horizontal", PassKind::Inline)
        .access(blur_x, Access::ColorAttachment)
        .build();
    builder
        .pass("vertical", PassKind::Inline)
        .access(blur_x, Access::Sampled)
        .access(blur_y, Access::ColorAttachment)
        .build();
    builder
        .pass("composite", PassKind::Inline)
        .access(blur_y, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    assert!(group_names(&graph).is_empty());
}

/// Only images whose allocations are interchangeable share one, because
/// "interchangeable" is the only thing the compiler can establish without a
/// `Device` to ask for memory requirements.
#[test]
fn transients_of_different_shape_are_never_aliased() {
    let graph = aliasable_chain(ImageDesc::new(Format::Rgba16Float));
    assert!(group_names(&graph).is_empty());
}

/// A memoryless target asks for a memory type the aliased block would not be
/// allocated from, and on the tiler it exists for it costs nothing to leave
/// alone.
#[test]
fn a_memoryless_target_is_never_aliased() {
    let mut builder = GraphBuilder::new();
    let target = builder.import_image(
        "target",
        image(),
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );
    let msaa = builder.create_image("msaa", image());
    let later = builder.create_image("later", image());

    builder
        .pass("draw", PassKind::Inline)
        .access(msaa, Access::ColorAttachment)
        .access(target, Access::ResolveAttachment)
        .build();
    builder
        .pass("write_later", PassKind::Inline)
        .access(later, Access::ColorAttachment)
        .build();
    builder
        .pass("read_later", PassKind::Inline)
        .access(later, Access::Sampled)
        .access(target, Access::ColorAttachment)
        .build();

    let graph = compile(builder).unwrap();
    assert!(group_names(&graph).is_empty());
}

/// The dependency aliasing adds, and the only thing that makes it safe: the
/// first pass to write the shared memory waits for the last pass that read what
/// was in it, even though the two name different resources.
#[test]
fn the_first_write_of_an_aliased_image_waits_for_the_reader_before_it() {
    let graph = aliasable_chain(image());
    let [barrier] = graph.barriers_before(2) else {
        panic!("expected one barrier before `write_second`");
    };
    assert_eq!(graph.resource_name(barrier.resource), "second");
    // Its contents are gone, so it enters undefined — and the memory under it
    // is still being sampled by `read_first` until that pass finishes.
    assert_eq!(barrier.old_layout, ImageLayout::Undefined);
    assert_eq!(barrier.new_layout, ImageLayout::ColorAttachmentOptimal);
    // Both halves of what the memory was doing: the pass that wrote `first` and
    // the pass that sampled it. A read of the old resource is a read of this
    // memory, so it is part of what the new write waits for.
    assert_eq!(
        barrier.src_stages,
        Access::ColorAttachment.stages() | Access::Sampled.stages()
    );
    assert_eq!(barrier.dst_stages, PipelineStages::COLOR_ATTACHMENT_OUTPUT);
}

/// Without aliasing that same barrier sources nothing, which is what makes the
/// one above a dependency the graph added rather than one it already had.
#[test]
fn the_first_write_of_an_unaliased_image_waits_for_nothing() {
    let graph = aliasable_chain(ImageDesc::new(Format::Rgba16Float));
    let [barrier] = graph.barriers_before(2) else {
        panic!("expected one barrier before `write_second`");
    };
    assert_eq!(barrier.src_stages, PipelineStages::TOP_OF_PIPE);
}
