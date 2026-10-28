//! The render graph and the frame declaration speak the engine's own
//! vocabulary, so a second backend can consume them.
//!
//! Source tests, because the property is an absence: nothing in these files names
//! a vulkano type. Comments are exempt — several explain what vulkano does and
//! why the graph is shaped around it.

const DEVICE_FREE: &[(&str, &str)] = &[
    (
        "gfx/graph/access.rs",
        include_str!("../src/gfx/graph/access.rs"),
    ),
    (
        "gfx/graph/alias.rs",
        include_str!("../src/gfx/graph/alias.rs"),
    ),
    (
        "gfx/graph/builder.rs",
        include_str!("../src/gfx/graph/builder.rs"),
    ),
    (
        "gfx/graph/compile.rs",
        include_str!("../src/gfx/graph/compile.rs"),
    ),
    (
        "gfx/graph/error.rs",
        include_str!("../src/gfx/graph/error.rs"),
    ),
    ("gfx/graph/mod.rs", include_str!("../src/gfx/graph/mod.rs")),
    (
        "gfx/graph/plan.rs",
        include_str!("../src/gfx/graph/plan.rs"),
    ),
    (
        "gfx/graph/resource.rs",
        include_str!("../src/gfx/graph/resource.rs"),
    ),
    (
        "gfx/graph/schedule.rs",
        include_str!("../src/gfx/graph/schedule.rs"),
    ),
    (
        "gfx/graph/tests.rs",
        include_str!("../src/gfx/graph/tests.rs"),
    ),
    (
        "gfx/vulkan/frame.rs",
        include_str!("../src/gfx/vulkan/frame.rs"),
    ),
];

#[test]
fn the_graph_and_the_frame_declaration_name_no_vulkano_type() {
    let offenders: Vec<String> = DEVICE_FREE
        .iter()
        .flat_map(|(path, source)| {
            source.lines().enumerate().filter_map(move |(index, line)| {
                let code = line.split("//").next().unwrap_or("");
                code.contains("vulkano")
                    .then(|| format!("{path}:{}: {}", index + 1, line.trim()))
            })
        })
        .collect();
    assert!(
        offenders.is_empty(),
        "{} line(s) name vulkano:\n{}",
        offenders.len(),
        offenders.join("\n"),
    );
}
