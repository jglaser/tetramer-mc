//! Synthetic fixed-context balance, support and replay checks; no protein data.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use tetramer_mc::{
    defensive_dimer_proposal::DefensiveBranch,
    docking::{DockingMethod, DockingProposal},
    geometry::{Atom, Shape, SphereTree},
    math::{IDENTITY, Pose, Vec3, add, matmul, matvec, quaternion, rotation},
    oligomer_proposal::OligomerConfig,
    proposal::FrozenRelativePoseProposal,
    spherical::Container,
    two_neighbor_singleton::{SingletonStatus, TrialDisposition, TwoNeighborSingleton},
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn proposal(uniform: f64, correlation: f64) -> Result<DockingProposal> {
    proposal_variances(uniform, correlation, [0.07, 0.19])
}
fn proposal_variances(
    uniform: f64,
    correlation: f64,
    variances: [f64; 2],
) -> Result<DockingProposal> {
    let cov = |s: f64| -> [[f64; 6]; 6] {
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { s } else { 0. }))
    };
    let model = json!({"schema":"reciprocal-pose-mixture-v1",
        "reciprocal_components":[true,false],"base_model":{
        "coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
        "angular_length":1.,"weights":[0.35,0.65],
        "anchors":[{"position":[0.9,0.,0.],"rotation":IDENTITY},
                   {"position":[0.,0.9,0.],"rotation":IDENTITY}],
        "means":vec![[0.;6];2],"covariances":[cov(variances[0]),cov(variances[1])]}});
    DockingProposal::new(
        FrozenRelativePoseProposal::from_json_str_open(&model.to_string(), [6.; 3], uniform, SHA)?,
        DockingMethod::PosteriorInvolution,
        correlation,
        [0.; 3],
    )
}
fn tree() -> SphereTree {
    SphereTree::new(Shape {
        name: "singleton synthetic sphere".into(),
        volume: 4. * std::f64::consts::PI * 0.1_f64.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 0.1,
        }],
    })
    .unwrap()
}
fn state() -> [Pose; 4] {
    [
        pose([0.; 3]),
        pose([-0.9, 0., 0.]),
        pose([0., -0.9, 0.]),
        pose([0., 0., -1.5]),
    ]
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-11 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
fn compose(frame: Pose, pose: Pose) -> Pose {
    let r = rotation(frame.orientation);
    Pose {
        position: add(frame.position, matvec(r, pose.position)),
        orientation: quaternion(matmul(r, rotation(pose.orientation))),
    }
}

#[test]
fn two_neighbor_singleton_catalogue_ignores_source_and_rebuilds_for_other_mobile() -> Result<()> {
    let model = proposal(0.01, 0.9)?;
    let core = tree();
    let wall = Container::new(4., &core)?;
    let config = OligomerConfig::default();
    let original = state();
    let mut changed_source = original;
    // A poisoned moving slot cannot enter any catalogue operation.
    changed_source[0] = pose([f64::NAN; 3]);
    let first =
        TwoNeighborSingleton::new(&model, &core, &wall, &original, 0, [1, 2], 4., 8, &config)?;
    let second = TwoNeighborSingleton::new(
        &model,
        &core,
        &wall,
        &changed_source,
        0,
        [1, 2],
        4.,
        8,
        &config,
    )?;
    assert!(!first.mixture().fused.is_empty());
    assert_eq!(first.mixture().key(), second.mixture().key());
    assert_eq!(
        first.mixture().log_weights(),
        second.mixture().log_weights()
    );
    assert_eq!(
        serde_json::to_value(&first.mixture().fused)?,
        serde_json::to_value(&second.mixture().fused)?
    );
    assert_eq!(
        first.mixture().label_logs(original[0]),
        second.mixture().label_logs(original[0])
    );
    let mut changed_neighbor = original;
    changed_neighbor[1].position[0] -= 0.3;
    let rebuilt = TwoNeighborSingleton::new(
        &model,
        &core,
        &wall,
        &changed_neighbor,
        0,
        [1, 2],
        4.,
        8,
        &config,
    )?;
    assert_ne!(
        first.density(original[0])?.log_full,
        rebuilt.density(original[0])?.log_full
    );
    assert!(
        first
            .evaluate(pose([-0.9, 0., 0.]))?
            .spectator_core_collisions
            .contains(&1)
    );
    assert!(
        !rebuilt
            .evaluate(pose([-0.9, 0., 0.]))?
            .spectator_core_collisions
            .contains(&1)
    );
    Ok(())
}

#[test]
fn two_neighbor_singleton_full_density_and_correction_use_both_branches() -> Result<()> {
    let model = proposal(0.7, 0.2)?;
    let core = tree();
    let wall = Container::new(4., &core)?;
    let initial = state();
    let kernel = TwoNeighborSingleton::new(
        &model,
        &core,
        &wall,
        &initial,
        0,
        [1, 2],
        4.,
        30,
        &OligomerConfig::default(),
    )?;
    let old = initial[0];
    let d = kernel.density(old)?;
    let learned = kernel
        .mixture()
        .label_logs(old)
        .iter()
        .map(|x| x.exp())
        .sum::<f64>();
    close(d.log_uniform.exp(), 1. / 8_f64.powi(3));
    close(d.log_learned.exp(), learned);
    close(d.log_full.exp(), 0.5 / 8_f64.powi(3) + 0.5 * learned);
    // Outside the cube the .5 learned weight still belongs in the density.
    let outside = kernel.density(pose([4.1, 0., 0.]))?;
    assert_eq!(outside.log_uniform, f64::NEG_INFINITY);
    close(
        outside.log_full,
        outside.log_learned - std::f64::consts::LN_2,
    );
    let out = kernel.propose(&mut StdRng::seed_from_u64(91), old)?;
    assert_eq!(out.status, SingletonStatus::Candidate);
    let new = kernel.density(out.candidate.unwrap())?;
    close(out.complete_log_correction()?, d.log_full - new.log_full);
    assert!((out.complete_log_correction()? - (d.log_learned - new.log_learned)).abs() > 1e-8);
    // Reverse and forward successful subdensities have the same finite-cap factor.
    for p in [0.001_f64, 0.3, 1.] {
        let factor = (0..30).map(|r| (1. - p).powi(r)).sum::<f64>();
        close(
            (d.log_full.exp() * factor / (new.log_full.exp() * factor)).ln(),
            out.complete_log_correction()?,
        );
    }
    Ok(())
}

#[test]
fn two_neighbor_singleton_retries_redraw_complete_mixture_and_replay_all_randomness() -> Result<()>
{
    let model = proposal(0.2, 0.7)?;
    let core = tree();
    let wall = Container::new(0.31, &core)?;
    let initial = [pose([0.; 3]), pose([-0.2, 0., 0.]), pose([0.2, 0., 0.])];
    let kernel = TwoNeighborSingleton::new(
        &model,
        &core,
        &wall,
        &initial,
        0,
        [1, 2],
        3.,
        12,
        &OligomerConfig::default(),
    )?;
    let mut rng = StdRng::seed_from_u64(876);
    let mut replay = StdRng::seed_from_u64(876);
    let out = kernel.propose(&mut rng, initial[0])?;
    assert_eq!(out.status, SingletonStatus::CapExhausted);
    assert_eq!(out.trials.len(), 12);
    assert!(out.counts.uniform > 0 && out.counts.learned > 0);
    assert_eq!(out.counts.uniform + out.counts.learned, 12);
    assert_eq!(out.counts.geometric_rejections, 12);
    assert!(out.candidate.is_none());
    for (index, trial) in out.trials.iter().enumerate() {
        assert_eq!(trial.index, index + 1);
        assert_eq!(trial.disposition, TrialDisposition::GeometricRejection);
        assert_eq!(trial.branch_uniform, replay.random::<f64>());
        let (expected, trace) = if trial.branch_uniform < 0.5 {
            assert_eq!(trial.branch, DefensiveBranch::Uniform);
            let u: [f64; 3] = std::array::from_fn(|_| replay.random());
            let z: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(&mut replay));
            let n = z.iter().map(|x| x * x).sum::<f64>().sqrt();
            (
                Some(compose(
                    initial[2],
                    Pose {
                        position: u.map(|v| (2. * v - 1.) * 3.),
                        orientation: z.map(|v| v / n),
                    },
                )),
                json!({"translation_uniforms":u,"quaternion_normals":z}),
            )
        } else {
            assert_eq!(trial.branch, DefensiveBranch::Learned);
            kernel.mixture().draw_singleton_independent(&mut replay)?
        };
        assert_eq!(
            serde_json::to_value(expected)?,
            serde_json::to_value(trial.proposed_pose)?
        );
        assert_eq!(trace, trial.trace);
        let candidate = expected.unwrap();
        assert_eq!(
            kernel.density(candidate)?.log_full,
            trial.density.unwrap().log_full
        );
        assert_eq!(
            kernel.evaluate(candidate)?,
            trial.feasibility.clone().unwrap()
        );
    }
    assert_eq!(rng.random::<u64>(), replay.random::<u64>());
    serde_json::to_vec(&out)?;
    Ok(())
}

#[test]
fn two_neighbor_singleton_zero_cap_and_zero_reverse_flow_use_no_rng() -> Result<()> {
    let model = proposal(0.1, 0.9)?;
    let core = tree();
    let wall = Container::new(6., &core)?;
    let initial = state();
    let config = OligomerConfig {
        multi_contact_mass: 0.,
        ..OligomerConfig::default()
    };
    let zero =
        TwoNeighborSingleton::new(&model, &core, &wall, &initial, 0, [1, 2], 1., 0, &config)?;
    let kernel =
        TwoNeighborSingleton::new(&model, &core, &wall, &initial, 0, [1, 2], 1., 10, &config)?;
    let mut rng = StdRng::seed_from_u64(51);
    let mut untouched = StdRng::seed_from_u64(51);
    let out = zero.propose(&mut rng, initial[0])?;
    assert_eq!(out.status, SingletonStatus::CapZero);
    assert!(out.trials.is_empty() && out.candidate.is_none());
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    // All atlas anchors have identity orientation, hence their Cayley charts
    // exclude this exact half-turn; the source also lies outside the cube.
    let old = Pose {
        position: [3., 0., 0.],
        orientation: [0., 1., 0., 0.],
    };
    assert_eq!(kernel.density(old)?.log_full, f64::NEG_INFINITY);
    let mut rng = StdRng::seed_from_u64(52);
    let mut untouched = StdRng::seed_from_u64(52);
    let out = kernel.propose(&mut rng, old)?;
    assert_eq!(out.status, SingletonStatus::SourceZeroReverseFlow);
    assert!(out.trials.is_empty() && out.candidate.is_none());
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    assert_eq!(
        serde_json::to_value(out)?["old_density"]["log_full"],
        "-inf"
    );
    Ok(())
}

#[test]
fn two_neighbor_singleton_checks_all_spectators_wall_and_detached_support() -> Result<()> {
    let model = proposal(0.1, 0.9)?;
    let core = tree();
    let wall = Container::new(4., &core)?;
    let initial = state();
    let kernel = TwoNeighborSingleton::new(
        &model,
        &core,
        &wall,
        &initial,
        0,
        [1, 2],
        4.,
        8,
        &OligomerConfig::default(),
    )?;
    for label in 1..4 {
        assert_eq!(
            kernel.evaluate(initial[label])?.spectator_core_collisions,
            vec![label]
        );
    }
    assert!(kernel.evaluate(pose([0., 0., 3.]))?.hard_valid());
    assert!(kernel.evaluate(pose([3.9, 0., 0.]))?.wall_valid);
    assert!(!kernel.evaluate(pose([3.91, 0., 0.]))?.wall_valid);
    let mut rng = StdRng::seed_from_u64(1);
    let mut untouched = StdRng::seed_from_u64(1);
    let error = kernel.propose(&mut rng, initial[3]).unwrap_err();
    assert_eq!(error.outcome.status, SingletonStatus::Fatal);
    assert!(error.outcome.trials.is_empty());
    assert_eq!(
        error
            .outcome
            .source_feasibility
            .as_ref()
            .unwrap()
            .spectator_core_collisions,
        vec![3]
    );
    serde_json::to_vec(&error)?;
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    Ok(())
}

#[test]
fn two_neighbor_singleton_draws_ignore_old_pose_and_atlas_kernel_settings() -> Result<()> {
    let model = proposal(0.01, 0.1)?;
    let alternate_model = proposal(0.8, 0.95)?;
    let core = tree();
    let wall = Container::new(4., &core)?;
    let initial = state();
    let config = OligomerConfig::default();
    let first =
        TwoNeighborSingleton::new(&model, &core, &wall, &initial, 0, [1, 2], 4., 20, &config)?;
    let second = TwoNeighborSingleton::new(
        &alternate_model,
        &core,
        &wall,
        &initial,
        0,
        [1, 2],
        4.,
        20,
        &config,
    )?;
    let a = first.propose(&mut StdRng::seed_from_u64(33), initial[0])?;
    let b = second.propose(&mut StdRng::seed_from_u64(33), pose([0., 0., 0.6]))?;
    assert_eq!(
        serde_json::to_value(&a.trials)?,
        serde_json::to_value(&b.trials)?
    );
    assert_eq!(
        serde_json::to_value(a.candidate)?,
        serde_json::to_value(b.candidate)?
    );
    assert_eq!(a.status, SingletonStatus::Candidate);
    assert_ne!(a.complete_log_correction()?, b.complete_log_correction()?);
    Ok(())
}

#[test]
fn two_neighbor_singleton_rejects_invalid_context_labels_and_cube() -> Result<()> {
    let model = proposal(0.1, 0.9)?;
    let core = tree();
    let wall = Container::new(4., &core)?;
    let initial = state();
    for neighbors in [[1, 1], [0, 1], [1, 4]] {
        assert!(
            TwoNeighborSingleton::new(
                &model,
                &core,
                &wall,
                &initial,
                0,
                neighbors,
                4.,
                2,
                &OligomerConfig::default()
            )
            .is_err()
        );
    }
    for width in [0., -1., f64::NAN, f64::INFINITY, 1e308] {
        assert!(
            TwoNeighborSingleton::new(
                &model,
                &core,
                &wall,
                &initial,
                0,
                [1, 2],
                width,
                2,
                &OligomerConfig::default()
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn two_neighbor_singleton_defense_uses_fixed_translated_rotated_anchor_frame() -> Result<()> {
    let model = proposal(0.1, 0.9)?;
    let core = tree();
    let wall = Container::new(6., &core)?;
    let frame = Pose {
        position: [1.2, -0.7, 0.4],
        orientation: [0.5_f64.sqrt(), 0., 0., 0.5_f64.sqrt()],
    };
    let mut initial = state();
    initial[2] = frame;
    let kernel = TwoNeighborSingleton::new(
        &model,
        &core,
        &wall,
        &initial,
        0,
        [1, 2],
        1.,
        1,
        &OligomerConfig::default(),
    )?;
    let inside = compose(frame, pose([0.8, 0.7, 0.6]));
    close(kernel.density(inside)?.log_uniform, -3. * 2_f64.ln());
    let outside = compose(frame, pose([1.1, 0., 0.]));
    assert_eq!(kernel.density(outside)?.log_uniform, f64::NEG_INFINITY);
    // Select a reproducible uniform first trial without relying on a fitted
    // catalogue or acceptance event to choose the branch.
    let seed = (0..100)
        .find(|s| StdRng::seed_from_u64(*s).random::<f64>() < 0.5)
        .unwrap();
    let out = kernel.propose(&mut StdRng::seed_from_u64(seed), initial[0])?;
    assert_eq!(
        serde_json::to_value(out.uniform_frame)?,
        serde_json::to_value(frame)?
    );
    assert_eq!(out.uniform_center, frame.position);
    let trial = &out.trials[0];
    assert_eq!(trial.branch, DefensiveBranch::Uniform);
    let u: [f64; 3] = serde_json::from_value(trial.trace["translation_uniforms"].clone())?;
    let z: [f64; 4] = serde_json::from_value(trial.trace["quaternion_normals"].clone())?;
    let n = z.iter().map(|x| x * x).sum::<f64>().sqrt();
    let expected = compose(
        frame,
        Pose {
            position: u.map(|v| 2. * v - 1.),
            orientation: z.map(|v| v / n),
        },
    );
    assert_eq!(
        serde_json::to_value(expected)?,
        serde_json::to_value(trial.proposed_pose.unwrap())?
    );
    close(trial.density.unwrap().log_uniform, -3. * 2_f64.ln());
    Ok(())
}

#[test]
fn two_neighbor_singleton_fatal_score_retains_begun_trial_without_retry() -> Result<()> {
    let model = proposal_variances(0.1, 0.9, [1e-300; 2])?;
    let core = tree();
    let wall = Container::new(4., &core)?;
    let initial = state();
    let kernel = TwoNeighborSingleton::new(
        &model,
        &core,
        &wall,
        &initial,
        0,
        [1, 2],
        1e6,
        20,
        &OligomerConfig {
            multi_contact_mass: 0.,
            ..OligomerConfig::default()
        },
    )?;
    assert!(kernel.density(initial[0])?.log_full.is_finite());
    let seed = (0..100)
        .find(|s| StdRng::seed_from_u64(*s).random::<f64>() < 0.5)
        .unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut replay = StdRng::seed_from_u64(seed);
    let error = kernel.propose(&mut rng, initial[0]).unwrap_err();
    assert_eq!(error.outcome.status, SingletonStatus::Fatal);
    assert_eq!(error.outcome.trials.len(), 1);
    let trial = &error.outcome.trials[0];
    assert_eq!(trial.branch, DefensiveBranch::Uniform);
    assert_eq!(trial.disposition, TrialDisposition::Fatal);
    assert!(trial.proposed_pose.is_some() && trial.trace["quaternion_normals"].is_array());
    assert!(trial.density.is_none() && trial.feasibility.is_none());
    assert_eq!(error.outcome.counts.uniform, 1);
    assert_eq!(error.outcome.counts.fatal_trials, 1);
    assert_eq!(error.outcome.counts.geometric_rejections, 0);
    let _: f64 = replay.random();
    for _ in 0..3 {
        let _: f64 = replay.random();
    }
    for _ in 0..4 {
        let _: f64 = StandardNormal.sample(&mut replay);
    }
    assert_eq!(rng.random::<u64>(), replay.random::<u64>());
    serde_json::to_vec(&error)?;
    Ok(())
}
