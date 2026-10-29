//! Compares two directories of offscreen captures and prints the largest
//! per-channel difference in each file, so "pixel-identical" is a number a
//! reviewer reads rather than a claim.
//!
//! ```bash
//! ORRIN_CAPTURE_A=~/orrin-metal-baseline/captures ORRIN_CAPTURE_B=target/capture \
//!     cargo test -p orrin-core --test capture_diff -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d because it needs two directories the caller produced. A file
//! missing from either side, a size mismatch, or any difference above
//! `ORRIN_CAPTURE_MAX_DIFF` (default 0) fails.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn pngs(dir: &Path) -> BTreeSet<String> {
    fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            name.ends_with(".png").then_some(name)
        })
        .collect()
}

fn dir(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a capture directory")))
}

#[test]
#[ignore = "compares two capture directories the caller produced"]
fn two_capture_directories_agree() {
    let (a, b) = (dir("ORRIN_CAPTURE_A"), dir("ORRIN_CAPTURE_B"));
    let allowed: u8 = std::env::var("ORRIN_CAPTURE_MAX_DIFF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let (names_a, names_b) = (pngs(&a), pngs(&b));
    assert_eq!(names_a, names_b, "the directories hold different captures");

    let mut failures = Vec::new();
    println!("{:<28} max channel diff", "capture");
    for name in &names_a {
        let left = image::open(a.join(name)).expect("decodes").to_rgba8();
        let right = image::open(b.join(name)).expect("decodes").to_rgba8();
        if left.dimensions() != right.dimensions() {
            println!(
                "{name:<28} size {:?} vs {:?}",
                left.dimensions(),
                right.dimensions()
            );
            failures.push(name.clone());
            continue;
        }
        let max = left
            .as_raw()
            .iter()
            .zip(right.as_raw())
            .map(|(x, y)| x.abs_diff(*y))
            .max()
            .unwrap_or(0);
        println!("{name:<28} {max}");
        if max > allowed {
            failures.push(name.clone());
        }
    }
    assert!(
        failures.is_empty(),
        "differs beyond {allowed}: {failures:?}"
    );
}
