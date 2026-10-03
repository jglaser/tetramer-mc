//! Fixed-cloud threshold, independent RNG, strict frame and baseline controls; no bath.
use anyhow::Result;
use rand::{
    RngExt, SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    auxiliary_overlap_threshold::AuxiliaryOverlapThreshold,
    capped_dimer::FixedDimerContext,
    defensive_dimer_proposal::DefensiveDimerProposal,
    docking::{DockingMethod, DockingProposal},
    factorized_dimer::{
        FactorizedAttemptStatus, FactorizedDimerCaps, FactorizedDimerOrder,
        FactorizedDimerProposal, FactorizedDimerStatus,
    },
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, rotation},
    proposal::FrozenRelativePoseProposal,
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const ORDERS: [FactorizedDimerOrder; 2] = [
    FactorizedDimerOrder::RootFirst,
    FactorizedDimerOrder::InternalFirst,
];

fn pose(x: f64) -> Pose {
    Pose {
        position: [x, 0., 0.],
        orientation: [1., 0., 0., 0.],
    }
}
fn sphere(radius: f64) -> SphereTree {
    SphereTree::new(Shape {
        name: "factorized sphere".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
    .unwrap()
}
fn atlas(center: f64, variance: f64, broken: bool) -> Result<DockingProposal> {
    let anchor = if broken { pose(1e308) } else { pose(center) };
    let mean = if broken {
        [1e308, 0., 0., 0., 0., 0.]
    } else {
        [0.; 6]
    };
    let covariance: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { variance } else { 0. }));
    let model = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
        "angular_length":1.,"weights":[1.],"anchors":[{"position":anchor.position,"rotation":rotation(anchor.orientation)}],
        "means":[mean],"covariances":[covariance]}).to_string();
    DockingProposal::new(
        FrozenRelativePoseProposal::from_json_str_open(&model, [80.; 3], 0.1, SHA)?,
        DockingMethod::PosteriorInvolution,
        0.7,
        [0.; 3],
    )
}

fn remove_guidance(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(fields) => {
            fields.remove("guidance");
            fields.remove("guidance_count");
            for child in fields.values_mut() {
                remove_guidance(child);
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                remove_guidance(child);
            }
        }
        _ => (),
    }
}

#[test]
fn cloud_validation_closed_boundaries_counts_and_correction() -> Result<()> {
    let exclusion = sphere(1.);
    assert!(AuxiliaryOverlapThreshold::new(&exclusion, vec![], 0).is_err());
    assert!(AuxiliaryOverlapThreshold::new(&exclusion, vec![[f64::NAN, 0., 0.]], 1).is_err());
    assert!(AuxiliaryOverlapThreshold::new(&exclusion, vec![[1.0001, 0., 0.]], 1).is_err());
    let guide = AuxiliaryOverlapThreshold::new(
        &exclusion,
        vec![[1., 0., 0.], [0.5, 0., 0.], [0.5, 0., 0.], [-1., 0., 0.]],
        4,
    )?;
    assert_eq!(guide.count_relative(pose(0.))?, 4);
    assert_eq!(guide.count_relative(pose(1.))?, 3);
    assert_eq!(guide.count_relative(pose(2.))?, 1); // child boundary included
    assert_eq!(guide.count_world([pose(5.), pose(6.)])?, 3);
    // Exact quarter-turn carries the fixed root-body cloud in world space.
    let root = Pose {
        position: [5., 6., 7.],
        orientation: [0.5, 0.5, 0.5, 0.5],
    };
    let child = Pose {
        position: [5., 7., 7.],
        ..root
    };
    assert_eq!(guide.count_world([root, child])?, 3);
    assert!((guide.log_correction(3, 1)? - 4. * 2f64.ln()).abs() < 1e-14);
    assert_eq!(guide.log_correction(3, 1)?, -guide.log_correction(1, 3)?);
    assert!(guide.log_correction(5, 1).is_err());
    assert!(guide.count_relative(pose(f64::INFINITY)).is_err());
    Ok(())
}

#[test]
fn zero_count_guide_preserves_baseline_proposal_trace_schema_and_rng() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0., 1., false)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![], 4)?;
    for order in ORDERS {
        let proposal = FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(&base, 2.5, 0.5)?,
            FactorizedDimerCaps {
                root: 8,
                internal: 8,
                joint: 3,
            },
            order,
        );
        for seed in 1..=8 {
            let (mut rng, mut baseline_rng, mut aux, mut aux_replay) = (
                StdRng::seed_from_u64(seed),
                StdRng::seed_from_u64(seed),
                StdRng::seed_from_u64(900 + seed),
                StdRng::seed_from_u64(900 + seed),
            );
            let baseline = proposal.propose(&mut baseline_rng, &context, [state[0], state[1]])?;
            let guided = proposal.propose_guided(
                &mut rng,
                &mut aux,
                &context,
                [state[0], state[1]],
                &guide,
            )?;
            let d = guided.guidance.as_ref().unwrap();
            assert_eq!(d.old_count, Some(0));
            assert_eq!(d.threshold, Some(0));
            let uniform = Uniform::new_inclusive(0usize, 0)?;
            assert_eq!(
                d.integer_draws,
                (0..4)
                    .map(|_| uniform.sample(&mut aux_replay))
                    .collect::<Vec<_>>()
            );
            assert_eq!(aux.random::<u64>(), aux_replay.random::<u64>());
            assert_eq!(rng.random::<u64>(), baseline_rng.random::<u64>());
            let baseline_json = serde_json::to_value(&baseline)?;
            assert!(!serde_json::to_string(&baseline)?.contains("guidance"));
            let mut guided_json = serde_json::to_value(&guided)?;
            remove_guidance(&mut guided_json);
            assert_eq!(guided_json, baseline_json);
            if guided.candidate.is_some() {
                assert_eq!(d.aux_log_correction, Some(0.));
            }
        }
    }
    Ok(())
}

#[test]
fn zero_caps_and_source_outside_domain_do_not_consume_either_rng() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0., 1., false)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.5, 0., 0.]], 4)?;
    for order in ORDERS {
        for zero in 0..4 {
            let mut caps = FactorizedDimerCaps {
                root: 3,
                internal: 3,
                joint: 2,
            };
            match zero {
                0 => caps.root = 0,
                1 => caps.internal = 0,
                2 => caps.joint = 0,
                _ => (),
            }
            let proposal = FactorizedDimerProposal::new(
                DefensiveDimerProposal::new(&base, 2.5, 0.5)?,
                caps,
                order,
            );
            let old = if zero == 3 {
                [pose(0.), pose(-2.5)]
            } else {
                [state[0], state[1]]
            };
            let (mut rng, mut aux, mut expected_rng, mut expected_aux) = (
                StdRng::seed_from_u64(9),
                StdRng::seed_from_u64(19),
                StdRng::seed_from_u64(9),
                StdRng::seed_from_u64(19),
            );
            let result = proposal.propose_guided(&mut rng, &mut aux, &context, old, &guide)?;
            assert_eq!(
                result.status,
                if zero == 3 {
                    FactorizedDimerStatus::SourceOutsideDomain
                } else {
                    FactorizedDimerStatus::CapExhausted
                }
            );
            assert!(result.attempts.is_empty());
            let d = result.guidance.unwrap();
            assert!(d.integer_draws.is_empty() && d.threshold.is_none());
            assert_eq!(d.count_queries, 4);
            assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
            assert_eq!(aux.random::<u64>(), expected_aux.random::<u64>());
        }
    }
    Ok(())
}

#[test]
fn maximum_threshold_is_fixed_across_all_failed_edges_and_joint_caps() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(0.5), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(1.5, 1e-12, false)?;
    let cloud = vec![[0.1, 0., 0.]; 8];
    let mut first_integer = None;
    for m in [1, 4] {
        for order in ORDERS {
            let guide = AuxiliaryOverlapThreshold::new(&exclusion, cloud.clone(), m)?;
            let proposal = FactorizedDimerProposal::new(
                DefensiveDimerProposal::new(&base, 4., 0.)?,
                FactorizedDimerCaps {
                    root: 2,
                    internal: 3,
                    joint: 2,
                },
                order,
            );
            let (mut rng, mut aux, mut aux_replay) = (
                StdRng::seed_from_u64(8),
                StdRng::seed_from_u64(44),
                StdRng::seed_from_u64(44),
            );
            let result = proposal.propose_guided(
                &mut rng,
                &mut aux,
                &context,
                [state[0], state[1]],
                &guide,
            )?;
            let d = result.guidance.as_ref().unwrap();
            let uniform = Uniform::new_inclusive(0usize, 8)?;
            let integers = (0..m)
                .map(|_| uniform.sample(&mut aux_replay))
                .collect::<Vec<_>>();
            assert_eq!(d.integer_draws, integers);
            assert_eq!(d.threshold, integers.iter().copied().max());
            assert!(d.threshold.unwrap() > 0); // frozen seed, not an adaptive search
            if let Some(first) = first_integer {
                assert_eq!(integers[0], first);
            } else {
                first_integer = Some(integers[0]);
            }
            assert_eq!(aux.random::<u64>(), aux_replay.random::<u64>());
            assert_eq!(result.status, FactorizedDimerStatus::CapExhausted);
            assert_eq!(d.count_queries, 10);
            assert_eq!(d.point_tests, 80);
            assert_eq!(result.attempts.len(), 2);
            for attempt in &result.attempts {
                assert_eq!(
                    attempt.status,
                    FactorizedAttemptStatus::InternalCapExhausted
                );
                assert_eq!(attempt.internal_draws.len(), 3);
                assert_eq!(
                    attempt.root_draws.len(),
                    usize::from(order == FactorizedDimerOrder::RootFirst)
                );
                for draw in &attempt.internal_draws {
                    assert!(draw.feasibility.as_ref().unwrap().feasible());
                    assert_eq!(draw.guidance_count, Some(0));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn accepted_candidate_keeps_full_f_and_nonunit_auxiliary_ratio_separate() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(1.5), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0.6, 1e-12, false)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.1, 0., 0.]], 4)?;
    for order in ORDERS {
        let raw = DefensiveDimerProposal::new(&base, 4., 0.)?;
        let proposal = FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(&base, 4., 0.)?,
            FactorizedDimerCaps {
                root: 2,
                internal: 2,
                joint: 2,
            },
            order,
        );
        let result = proposal.propose_guided(
            &mut StdRng::seed_from_u64(8),
            &mut StdRng::seed_from_u64(44),
            &context,
            [state[0], state[1]],
            &guide,
        )?;
        assert_eq!(result.status, FactorizedDimerStatus::Candidate);
        let d = result.guidance.as_ref().unwrap();
        assert_eq!(
            (d.old_count, d.new_count, d.threshold),
            (Some(0), Some(1), Some(0))
        );
        assert!((d.aux_log_correction.unwrap() + 4. * 2f64.ln()).abs() < 1e-14);
        assert_eq!(d.integer_draws, vec![0; 4]);
        assert_eq!(d.count_queries, 9);
        assert_eq!(d.point_tests, 9);
        let frame = result.attempts[0]
            .frame
            .as_ref()
            .unwrap()
            .guidance
            .as_ref()
            .unwrap();
        assert_eq!(frame.relative_count, Some(1));
        assert_eq!(frame.world_count, Some(1));
        assert_eq!(frame.recovered_count, Some(1));
        assert_eq!(frame.reconstructed_world_count, Some(1));
        let candidate = result.candidate.unwrap();
        let correction = raw.correction(
            state[2],
            state[0],
            state[1],
            candidate.root,
            candidate.child,
        )?;
        assert_eq!(
            serde_json::to_value(&candidate.diagnostics)?,
            serde_json::to_value(correction)?
        );
        assert!(candidate.diagnostics.log_reverse_forward.abs() > 1.);
    }
    Ok(())
}

#[test]
fn frame_mismatch_is_fatal_with_all_counts_before_any_rng() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(1e14), pose(1e14 + 1.), pose(1e14 + 10.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0., 1., false)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![[-0.003, 0., 0.]], 4)?;
    let proposal = FactorizedDimerProposal::new(
        DefensiveDimerProposal::new(&base, 4., 0.5)?,
        FactorizedDimerCaps {
            root: 3,
            internal: 3,
            joint: 2,
        },
        FactorizedDimerOrder::RootFirst,
    );
    let (mut rng, mut aux, mut expected_rng, mut expected_aux) = (
        StdRng::seed_from_u64(9),
        StdRng::seed_from_u64(19),
        StdRng::seed_from_u64(9),
        StdRng::seed_from_u64(19),
    );
    let failure = proposal
        .propose_guided(&mut rng, &mut aux, &context, [state[0], state[1]], &guide)
        .unwrap_err();
    assert!(
        failure
            .fatal_error
            .contains("guidance count frame mismatch")
    );
    let outcome = failure.outcome.unwrap();
    assert!(outcome.attempts.is_empty());
    let counts = outcome.source_frame.unwrap().guidance.unwrap();
    assert_eq!(counts.relative_count, Some(0));
    assert_eq!(counts.recovered_count, Some(0));
    assert_eq!(counts.world_count, Some(1));
    assert_eq!(counts.reconstructed_world_count, Some(1));
    let d = outcome.guidance.unwrap();
    assert_eq!(d.old_count, Some(0));
    assert_eq!(d.count_queries, 4);
    assert!(d.integer_draws.is_empty());
    assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
    assert_eq!(aux.random::<u64>(), expected_aux.random::<u64>());
    Ok(())
}

#[test]
fn mismatched_shape_and_numerical_edge_failure_retain_guidance_without_retry() -> Result<()> {
    let (core, exclusion, wrong) = (sphere(0.2), sphere(1.), sphere(1.1));
    let state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0., 1., true)?;
    for wrong_shape in [true, false] {
        let guide = AuxiliaryOverlapThreshold::new(
            if wrong_shape { &wrong } else { &exclusion },
            vec![[0.5, 0., 0.]],
            4,
        )?;
        let proposal = FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(&base, 4., 0.)?,
            FactorizedDimerCaps {
                root: 3,
                internal: 3,
                joint: 2,
            },
            FactorizedDimerOrder::InternalFirst,
        );
        let (mut rng, mut aux) = (StdRng::seed_from_u64(33), StdRng::seed_from_u64(44));
        let failure = proposal
            .propose_guided(&mut rng, &mut aux, &context, [state[0], state[1]], &guide)
            .unwrap_err();
        let outcome = failure.outcome.unwrap();
        if wrong_shape {
            assert!(failure.fatal_error.contains("exclusion shape differs"));
            assert!(outcome.attempts.is_empty());
            assert!(outcome.guidance.unwrap().integer_draws.is_empty());
            assert_eq!(
                rng.random::<u64>(),
                StdRng::seed_from_u64(33).random::<u64>()
            );
            assert_eq!(
                aux.random::<u64>(),
                StdRng::seed_from_u64(44).random::<u64>()
            );
        } else {
            assert_eq!(outcome.guidance.unwrap().integer_draws.len(), 4);
            assert_eq!(outcome.attempts.len(), 1);
            let attempt = &outcome.attempts[0];
            assert!(attempt.root_draws.is_empty());
            assert_eq!(attempt.internal_draws.len(), 1);
            assert!(attempt.internal_draws[0].draw.null_reason.is_some());
            assert!(attempt.internal_draws[0].guidance_count.is_none());
        }
    }
    Ok(())
}

#[test]
fn both_streams_replay_prefix_and_continue_deterministically() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0., 1., false)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.3, 0., 0.], [0.8, 0.2, 0.]], 4)?;
    let proposal = FactorizedDimerProposal::new(
        DefensiveDimerProposal::new(&base, 2.5, 0.5)?,
        FactorizedDimerCaps {
            root: 4,
            internal: 4,
            joint: 2,
        },
        FactorizedDimerOrder::InternalFirst,
    );
    let (mut rng, mut aux, mut restarted, mut restarted_aux) = (
        StdRng::seed_from_u64(81),
        StdRng::seed_from_u64(181),
        StdRng::seed_from_u64(81),
        StdRng::seed_from_u64(181),
    );
    for _ in 0..3 {
        proposal.propose_guided(&mut rng, &mut aux, &context, [state[0], state[1]], &guide)?;
    }
    for _ in 0..3 {
        proposal.propose_guided(
            &mut restarted,
            &mut restarted_aux,
            &context,
            [state[0], state[1]],
            &guide,
        )?;
    }
    for _ in 0..5 {
        let a =
            proposal.propose_guided(&mut rng, &mut aux, &context, [state[0], state[1]], &guide)?;
        let b = proposal.propose_guided(
            &mut restarted,
            &mut restarted_aux,
            &context,
            [state[0], state[1]],
            &guide,
        )?;
        assert_eq!(serde_json::to_value(a)?, serde_json::to_value(b)?);
    }
    assert_eq!(rng.random::<u64>(), restarted.random::<u64>());
    assert_eq!(aux.random::<u64>(), restarted_aux.random::<u64>());
    Ok(())
}

#[test]
fn checked_complete_correction_cannot_silently_drop_auxiliary_terms() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(1.5), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0.6, 1e-12, false)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.1, 0., 0.]], 4)?;
    let proposal = FactorizedDimerProposal::new(
        DefensiveDimerProposal::new(&base, 4., 0.)?,
        FactorizedDimerCaps {
            root: 2,
            internal: 2,
            joint: 2,
        },
        FactorizedDimerOrder::RootFirst,
    );
    let outcome = proposal.propose_guided(
        &mut StdRng::seed_from_u64(8),
        &mut StdRng::seed_from_u64(44),
        &context,
        [state[0], state[1]],
        &guide,
    )?;
    let full = outcome
        .candidate
        .as_ref()
        .unwrap()
        .diagnostics
        .log_reverse_forward;
    assert_eq!(
        outcome.complete_log_correction()?,
        full + outcome
            .guidance
            .as_ref()
            .unwrap()
            .aux_log_correction
            .unwrap()
    );
    assert_ne!(outcome.complete_log_correction()?, full);
    for missing in 0..3 {
        let mut invalid = outcome.clone();
        let d = invalid.guidance.as_mut().unwrap();
        match missing {
            0 => d.aux_log_correction = None,
            1 => d.new_count = None,
            _ => d.aux_log_correction = Some(0.),
        }
        assert!(invalid.complete_log_correction().is_err());
    }
    for bad in [f64::INFINITY, f64::NAN] {
        let mut invalid = outcome.clone();
        invalid
            .candidate
            .as_mut()
            .unwrap()
            .diagnostics
            .log_reverse_forward = bad;
        assert!(invalid.complete_log_correction().is_err());
    }
    let mut negative_infinity = outcome.clone();
    negative_infinity
        .candidate
        .as_mut()
        .unwrap()
        .diagnostics
        .log_reverse_forward = f64::NEG_INFINITY;
    assert_eq!(
        negative_infinity.complete_log_correction()?,
        f64::NEG_INFINITY
    );
    let baseline = proposal.propose(
        &mut StdRng::seed_from_u64(8),
        &context,
        [state[0], state[1]],
    )?;
    assert_eq!(
        baseline.complete_log_correction()?,
        baseline
            .candidate
            .as_ref()
            .unwrap()
            .diagnostics
            .log_reverse_forward
    );
    let mut incomplete = outcome;
    incomplete.status = FactorizedDimerStatus::CapExhausted;
    assert!(incomplete.complete_log_correction().is_err());
    Ok(())
}
