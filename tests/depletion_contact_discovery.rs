//! Independent controls for native-blind contact discovery. The overlap
//! reference is a sphere-union volume, and refinement is replayed with the
//! production pair kernel rather than an optimizer-score acceptance rule.
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, f64::consts::PI};
use tetramer_mc::{
    contact_discovery::{DiscoveryConfig, discover, estimate_pair_overlap, slot_seeds},
    contact_memory::{MemoryCounts, MemoryState},
    depletion::GateOptions,
    geometry::{Atom, Shape, SphereTree},
    math::Pose,
};

fn sphere(copies: usize) -> SphereTree {
    SphereTree::new(Shape {
        name: "discovery sphere-union reference".into(),
        volume: 4. * PI / 3.,
        atoms: vec![
            Atom {
                center: [0.; 3],
                radius: 1.
            };
            copies
        ],
    })
    .unwrap()
}

fn pose(distance: f64) -> Pose {
    Pose {
        position: [distance, 0., 0.],
        orientation: [1., 0., 0., 0.],
    }
}

fn options() -> GateOptions {
    GateOptions {
        max_cells: 255,
        max_depth: 10,
        min_width: 0.,
    }
}

fn lens(radius: f64, distance: f64) -> f64 {
    if distance >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + distance) * (2. * radius - distance).powi(2) / 12.
    }
}

fn small_config() -> DiscoveryConfig {
    DiscoveryConfig {
        starts: 3,
        search_steps: 4,
        search_points: 256,
        validation_points: 512,
        refine_steps: 12,
        burn: 2,
        save_every: 2,
        rd: 0.4,
        activity: 0.1,
        seed: 260926113,
        gap: 0.01,
        refine_translation_std_a: 0.2,
        envelope_max_cells: 63,
        ..Default::default()
    }
}

#[test]
fn exclusion_overlap_matches_analytic_sphere_lens() -> Result<()> {
    let tree = sphere(1);
    for (i, distance) in [2.05, 2.4, 3.3].into_iter().enumerate() {
        let estimate = estimate_pair_overlap(
            &tree,
            pose(distance),
            0.6,
            20_000,
            918271 + i as u64,
            options(),
        )?;
        let exact = lens(1.6, distance);
        assert!(estimate.volume.is_finite() && estimate.standard_error.is_finite());
        assert!(estimate.standard_error >= 0.);
        assert!(estimate.lower_volume <= exact + 1e-10);
        assert!(exact <= estimate.lower_volume + estimate.uncertain_volume + 1e-10);
        // This is a fixed-seed statistical control, not a guarantee that a
        // nominal 95% interval must cover every individual realization.
        assert!(
            (estimate.volume - exact).abs() <= 6. * estimate.standard_error + 1e-10,
            "distance={distance}, estimate={}, analytic={exact}, se={}",
            estimate.volume,
            estimate.standard_error
        );
        if exact == 0. {
            assert_eq!(estimate.volume, 0.);
        }
    }
    Ok(())
}

#[test]
fn overlapping_atomic_duplicates_are_counted_once() -> Result<()> {
    let single = sphere(1);
    let duplicated = sphere(7);
    let exact = lens(1.5, 2.1);
    for tree in [&single, &duplicated] {
        let estimate = estimate_pair_overlap(tree, pose(2.1), 0.5, 20_000, 892621, options())?;
        assert!((estimate.volume - exact).abs() <= 6. * estimate.standard_error + 1e-10);
        // Summing atomic pair lenses would multiply this result by 49.
        assert!(estimate.volume < 1.2 * exact);
    }
    Ok(())
}

#[test]
fn hard_valid_zero_radius_has_zero_exclusion_overlap() -> Result<()> {
    let tree = sphere(3);
    for distance in [2., 2.2, 8.] {
        let estimate = estimate_pair_overlap(&tree, pose(distance), 0., 1024, 556123, options())?;
        assert_eq!(estimate.volume, 0.);
        assert_eq!(estimate.standard_error, 0.);
        assert_eq!(estimate.hits, 0);
    }
    Ok(())
}

#[test]
fn discovery_repeats_deterministically_and_retains_every_start() -> Result<()> {
    let tree = sphere(1);
    let config = small_config();
    let first = discover(&tree, &config)?;
    let second = discover(&tree, &config)?;
    assert_eq!(
        serde_json::to_value(&first.slots)?,
        serde_json::to_value(&second.slots)?
    );
    assert_eq!(first.slots.len(), config.starts);
    for (index, slot) in first.slots.iter().enumerate() {
        assert_eq!(slot.slot, index);
        assert!(slot.search_optimized_score.volume >= slot.search_initial_score.volume);
        assert_eq!(slot.search_attempts.len(), config.search_steps);
        assert_eq!(slot.refinement_trace.len(), config.refine_steps);
        assert_eq!(
            slot.refinement_samples.len(),
            (config.refine_steps - config.burn) / config.save_every
        );
        for sample in &slot.refinement_samples {
            assert!(sample.step > config.burn);
            assert_eq!((sample.step - config.burn) % config.save_every, 0);
            let update = &slot.refinement_trace[sample.step - 1];
            assert_eq!(sample.pose, update.retained_pose);
            assert_eq!(sample.accepted, update.accepted);
        }
    }
    // Rotationally equivalent sphere basins must not cause starts to be
    // discarded by clustering, winner filtering, or score thresholds.
    Ok(())
}

#[test]
fn heldout_cloud_budget_cannot_change_search_or_physical_refinement() -> Result<()> {
    let tree = sphere(1);
    let config = small_config();
    let first = discover(&tree, &config)?;
    let changed = DiscoveryConfig {
        validation_points: 791,
        ..config.clone()
    };
    let second = discover(&tree, &changed)?;
    for (a, b) in first.slots.iter().zip(&second.slots) {
        assert_eq!(a.initial_pose, b.initial_pose);
        assert_eq!(a.optimized_pose, b.optimized_pose);
        assert_eq!(a.search_initial_score, b.search_initial_score);
        assert_eq!(a.search_optimized_score, b.search_optimized_score);
        assert_eq!(
            serde_json::to_value(&a.search_attempts)?,
            serde_json::to_value(&b.search_attempts)?
        );
        assert_eq!(
            serde_json::to_value(&a.refinement_trace)?,
            serde_json::to_value(&b.refinement_trace)?
        );
        assert_eq!(
            serde_json::to_value(&a.refinement_samples)?,
            serde_json::to_value(&b.refinement_samples)?
        );
    }
    Ok(())
}

#[test]
fn zero_radius_discovery_does_not_invent_a_score_based_physical_force() -> Result<()> {
    let tree = sphere(1);
    let config = DiscoveryConfig {
        rd: 0.,
        activity: 0.8,
        burn: 0,
        save_every: 1,
        refine_steps: 24,
        ..small_config()
    };
    let result = discover(&tree, &config)?;
    let mut rejected = 0;
    let mut accepted = 0;
    for slot in &result.slots {
        assert_eq!(slot.search_initial_score.volume, 0.);
        assert_eq!(slot.search_optimized_score.volume, 0.);
        for estimate in slot
            .initial_validation
            .iter()
            .chain(&slot.optimized_validation)
        {
            assert_eq!(estimate.volume, 0.);
        }
        for update in &slot.refinement_trace {
            if update.hard_valid && !update.outside_ball {
                assert!(update.accepted);
                assert_eq!(update.gate.as_ref().unwrap().log_weight, 0.);
                accepted += 1;
            } else {
                assert!(!update.accepted);
                assert_eq!(update.retained_pose, update.old_pose);
                rejected += 1;
            }
        }
    }
    assert!(
        accepted > 0 && rejected > 0,
        "exercise retained states and rejected repeats"
    );
    Ok(())
}

#[test]
fn refinement_is_bitwise_replay_of_the_exact_pair_kernel() -> Result<()> {
    let tree = sphere(1);
    let config = small_config();
    let result = discover(&tree, &config)?;
    let mut seed_keys = BTreeSet::new();
    for slot in &result.slots {
        for seed in [
            slot.seeds.initialization,
            slot.seeds.search,
            slot.seeds.search_score,
            slot.seeds.validation_initial[0],
            slot.seeds.validation_initial[1],
            slot.seeds.validation_optimized[0],
            slot.seeds.validation_optimized[1],
            slot.seeds.refinement,
        ] {
            assert!(
                seed_keys.insert(seed),
                "all slot/stage RNG seeds must be distinct"
            );
        }
        let mut state = MemoryState {
            poses: vec![slot.optimized_pose],
        };
        let mut rng = StdRng::seed_from_u64(slot.seeds.refinement);
        let mut counts = MemoryCounts::default();
        for saved in &slot.refinement_trace {
            let replay = state.update(
                &tree,
                &slot.resolved_memory_config,
                config.gate_options(),
                &mut rng,
            )?;
            assert_eq!(serde_json::to_value(&replay)?, serde_json::to_value(saved)?);
            counts.record(&replay);
        }
        assert_eq!(counts, slot.refinement_counts);
    }
    Ok(())
}

#[test]
fn zero_activity_accepts_every_valid_move_even_with_positive_discovery_scores() -> Result<()> {
    let tree = sphere(1);
    let config = DiscoveryConfig {
        activity: 0.,
        refine_steps: 24,
        burn: 0,
        save_every: 1,
        ..small_config()
    };
    let result = discover(&tree, &config)?;
    let mut accepted = 0;
    for slot in &result.slots {
        assert!(slot.search_optimized_score.volume > 0.);
        for update in &slot.refinement_trace {
            if update.hard_valid {
                assert!(update.accepted);
                assert_eq!(update.gate.as_ref().unwrap().log_weight, 0.);
                accepted += 1;
            } else {
                assert!(!update.accepted);
                assert_eq!(update.retained_pose, update.old_pose);
            }
        }
    }
    assert!(accepted > 0);
    Ok(())
}

#[test]
fn adjacent_master_seeds_do_not_reuse_other_slots_or_stage_streams() {
    let mut keys = BTreeSet::new();
    for master in 81732..81735 {
        for slot in 0..6 {
            let s = slot_seeds(master, slot);
            for key in [
                s.initialization,
                s.search,
                s.search_score,
                s.validation_initial[0],
                s.validation_initial[1],
                s.validation_optimized[0],
                s.validation_optimized[1],
                s.refinement,
            ] {
                assert!(
                    keys.insert(key),
                    "stream reuse across nearby independent runs"
                );
            }
        }
    }
}

#[test]
fn command_line_archives_complete_hashes_and_refuses_overwrite() -> Result<()> {
    use std::{
        fs,
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };
    let temporary = std::env::temp_dir().join(format!(
        "depletion-discovery-test-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&temporary)?;
    let shape_path = temporary.join("shape.json");
    let output = temporary.join("discovery");
    let shape_bytes = serde_json::to_vec(&sphere(1).shape)?;
    fs::write(&shape_path, &shape_bytes)?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_depletion-contact-discovery"));
    command
        .arg("--shape")
        .arg(&shape_path)
        .arg("--out")
        .arg(&output)
        .args([
            "--starts",
            "2",
            "--search-steps",
            "2",
            "--search-points",
            "64",
            "--validation-points",
            "128",
            "--refine-steps",
            "4",
            "--burn",
            "0",
            "--save-every",
            "1",
            "--rd",
            "0.4",
            "--activity",
            "0.1",
            "--gap",
            "0.01",
            "--envelope-max-cells",
            "31",
            "--seed",
            "713893",
        ]);
    let first = command.output()?;
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let manifest_bytes = fs::read(output.join("manifest.json"))?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    assert_eq!(manifest["complete"], true);
    assert_eq!(manifest["native_information"], false);
    assert_eq!(manifest["slots"], 2);
    assert_eq!(manifest["retained_samples"], 8);
    assert_eq!(
        manifest["shape_sha256"],
        format!("{:x}", Sha256::digest(&shape_bytes))
    );
    let hashes = manifest["outputs_sha256"].as_object().unwrap();
    assert_eq!(hashes.len(), 4);
    for name in [
        "shape.json",
        "source-bundle.json",
        "provenance.json",
        "discovery.json",
    ] {
        let bytes = fs::read(output.join(name))?;
        assert_eq!(hashes[name], format!("{:x}", Sha256::digest(&bytes)));
    }
    let discovery_bytes = fs::read(output.join("discovery.json"))?;
    let rerun = command.output()?;
    assert!(!rerun.status.success());
    assert!(String::from_utf8_lossy(&rerun.stderr).contains("already exists"));
    assert_eq!(fs::read(output.join("manifest.json"))?, manifest_bytes);
    assert_eq!(fs::read(output.join("discovery.json"))?, discovery_bytes);
    fs::remove_dir_all(temporary)?;
    Ok(())
}
