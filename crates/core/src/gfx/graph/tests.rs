use vulkano::format::Format;
use vulkano::image::ImageLayout;
use vulkano::sync::PipelineStages;

use super::*;

fn image() -> ImageDesc {
    ImageDesc::new(Format::R8G8B8A8_UNORM)
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
    assert!(graph
        .transient_images()
        .all(|(id, _)| graph.resource_name(id) != "orphan"));
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
        vulkano::image::ImageUsage::COLOR_ATTACHMENT | vulkano::image::ImageUsage::SAMPLED
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
    assert!(created
        .usage
        .contains(vulkano::image::ImageUsage::TRANSIENT_ATTACHMENT));
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
    assert!(graph
        .barriers_before(slot)
        .iter()
        .all(|barrier| graph.resource_name(barrier.resource) != "color"));
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
    assert_eq!(
        barrier.src_stages,
        PipelineStages::COLOR_ATTACHMENT_OUTPUT
    );
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
