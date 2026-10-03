//! Frozen anchor/root auxiliary guidance: independent sphere limits and RNG controls.
//! No proteins, physical bath clouds, production kernels, or statistical fitting.
use anyhow::Result;
use rand::{
    RngExt, SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};
use serde_json::{Value, json};
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
    math::{Pose, matmul, rotation},
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
        name: "root guidance test sphere".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
    .unwrap()
}
fn atlas(variance: f64) -> Result<DockingProposal> {
    let covariance: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { variance } else { 0. }));
    let model = json!({"coordinate_convention":"anchor-body-relative", "shape_sha256":SHA,
        "angular_length":1., "weights":[1.], "anchors":[{"position":[0.6,0.,0.],"rotation":rotation(pose(0.).orientation)}],
        "means":[[0.,0.,0.,0.,0.,0.]], "covariances":[covariance]}).to_string();
    DockingProposal::new(
        FrozenRelativePoseProposal::from_json_str_open(&model, [80.; 3], 0.1, SHA)?,
        DockingMethod::PosteriorInvolution,
        0.7,
        [0.; 3],
    )
}
fn caps() -> FactorizedDimerCaps {
    FactorizedDimerCaps {
        root: 4,
        internal: 4,
        joint: 3,
    }
}
fn proposal(
    base: &DockingProposal,
    order: FactorizedDimerOrder,
    cap: FactorizedDimerCaps,
) -> Result<FactorizedDimerProposal<'_>> {
    Ok(FactorizedDimerProposal::new(
        DefensiveDimerProposal::new(base, 4., 0.)?,
        cap,
        order,
    ))
}
fn strip_guidance(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for key in ["guidance", "root_guidance", "guidance_count"] {
                map.remove(key);
            }
            for child in map.values_mut() {
                strip_guidance(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                strip_guidance(item);
            }
        }
        _ => (),
    }
}

#[test]
fn both_nonzero_edge_corrections_are_added_once_to_complete_f() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(1.5), pose(3.), pose(0.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(1e-12)?;
    let root = AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.1, 0., 0.]], 2)?;
    let internal = AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.1, 0., 0.]], 4)?;
    for order in ORDERS {
        let p = proposal(&base, order, caps())?;
        let outcome = p.propose_with_guides(
            &mut StdRng::seed_from_u64(8),
            &context,
            [state[0], state[1]],
            Some(&root),
            Some(&mut StdRng::seed_from_u64(44)),
            Some(&internal),
            Some(&mut StdRng::seed_from_u64(45)),
        )?;
        assert_eq!(outcome.status, FactorizedDimerStatus::Candidate);
        let r = outcome.root_guidance.as_ref().unwrap();
        let i = outcome.guidance.as_ref().unwrap();
        for d in [r, i] {
            assert_eq!(
                (d.old_count, d.new_count, d.threshold),
                (Some(0), Some(1), Some(0))
            );
            assert_eq!(d.integer_draws, vec![0; d.m]);
            assert!((d.aux_log_correction.unwrap() + d.m as f64 * 2f64.ln()).abs() < 1e-14);
        }
        let candidate = outcome.candidate.as_ref().unwrap();
        let independently_scored = DefensiveDimerProposal::new(&base, 4., 0.)?.correction(
            state[2],
            state[0],
            state[1],
            candidate.root,
            candidate.child,
        )?;
        assert_eq!(
            serde_json::to_value(&candidate.diagnostics)?,
            serde_json::to_value(independently_scored)?
        );
        let full = candidate.diagnostics.log_reverse_forward;
        assert_eq!(
            outcome.complete_log_correction()?,
            (full + i.aux_log_correction.unwrap()) + r.aux_log_correction.unwrap()
        );
        assert_ne!(
            outcome.complete_log_correction()?,
            full + i.aux_log_correction.unwrap()
        );
        let attempt = outcome.attempts.last().unwrap();
        assert_eq!(attempt.root_draws.last().unwrap().guidance_count, Some(1));
        assert_eq!(
            attempt.internal_draws.last().unwrap().guidance_count,
            Some(1)
        );
        for frame in [
            attempt
                .frame
                .as_ref()
                .unwrap()
                .root_guidance
                .as_ref()
                .unwrap(),
            attempt.frame.as_ref().unwrap().guidance.as_ref().unwrap(),
        ] {
            assert_eq!(frame.relative_count, Some(1));
            assert_eq!(frame.recovered_count, Some(1));
            assert_eq!(frame.world_count, Some(1));
            assert_eq!(frame.reconstructed_world_count, Some(1));
        }
        for root_term in [false, true] {
            for missing in [false, true] {
                let mut bad = outcome.clone();
                let d = if root_term {
                    bad.root_guidance.as_mut().unwrap()
                } else {
                    bad.guidance.as_mut().unwrap()
                };
                if missing {
                    d.new_count = None;
                } else {
                    d.aux_log_correction = Some(0.);
                }
                assert!(bad.complete_log_correction().is_err());
            }
        }
    }
    Ok(())
}

#[test]
fn root_only_has_no_internal_guidance_or_internal_rng_consumption() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(1.5), pose(3.), pose(0.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(1e-12)?;
    let root = AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.1, 0., 0.]], 3)?;
    for order in ORDERS {
        let mut unused = StdRng::seed_from_u64(99);
        let outcome = proposal(&base, order, caps())?.propose_with_guides(
            &mut StdRng::seed_from_u64(8),
            &context,
            [state[0], state[1]],
            Some(&root),
            Some(&mut StdRng::seed_from_u64(44)),
            None,
            Some(&mut unused),
        )?;
        assert_eq!(outcome.status, FactorizedDimerStatus::Candidate);
        assert!(outcome.guidance.is_none());
        assert_eq!(
            unused.random::<u64>(),
            StdRng::seed_from_u64(99).random::<u64>()
        );
        let d = outcome.root_guidance.as_ref().unwrap();
        assert!((d.aux_log_correction.unwrap() + 3. * 2f64.ln()).abs() < 1e-14);
        let full = outcome
            .candidate
            .as_ref()
            .unwrap()
            .diagnostics
            .log_reverse_forward;
        assert_eq!(
            outcome.complete_log_correction()?,
            full + d.aux_log_correction.unwrap()
        );
        for attempt in &outcome.attempts {
            assert!(
                attempt
                    .internal_draws
                    .iter()
                    .all(|v| v.guidance_count.is_none())
            );
            if let Some(frame) = &attempt.frame {
                assert!(frame.guidance.is_none());
            }
        }
    }
    Ok(())
}

#[test]
fn disabled_extension_exactly_preserves_legacy_outcomes_and_both_rng_routes() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(1.5), pose(3.), pose(0.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0.15)?;
    let internal =
        AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.1, 0., 0.], [0.8, 0., 0.]], 2)?;
    for order in ORDERS {
        for seed in 1..=8 {
            for guided in [false, true] {
                let p = proposal(&base, order, caps())?;
                let (mut old_rng, mut new_rng) =
                    (StdRng::seed_from_u64(seed), StdRng::seed_from_u64(seed));
                let (mut old_aux, mut new_aux, mut unused_root) = (
                    StdRng::seed_from_u64(seed + 100),
                    StdRng::seed_from_u64(seed + 100),
                    StdRng::seed_from_u64(999),
                );
                let old = if guided {
                    p.propose_guided(
                        &mut old_rng,
                        &mut old_aux,
                        &context,
                        [state[0], state[1]],
                        &internal,
                    )?
                } else {
                    p.propose(&mut old_rng, &context, [state[0], state[1]])?
                };
                let new = p.propose_with_guides(
                    &mut new_rng,
                    &context,
                    [state[0], state[1]],
                    None,
                    Some(&mut unused_root),
                    if guided { Some(&internal) } else { None },
                    Some(&mut new_aux),
                )?;
                assert_eq!(serde_json::to_value(&new)?, serde_json::to_value(&old)?);
                assert!(!serde_json::to_string(&new)?.contains("root_guidance"));
                assert_eq!(old_rng.random::<u64>(), new_rng.random::<u64>());
                assert_eq!(old_aux.random::<u64>(), new_aux.random::<u64>());
                assert_eq!(
                    unused_root.random::<u64>(),
                    StdRng::seed_from_u64(999).random::<u64>()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn empty_clouds_preserve_baseline_draws_and_use_only_their_own_threshold_streams() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(1.5), pose(3.), pose(0.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0.15)?;
    let root = AuxiliaryOverlapThreshold::new(&exclusion, vec![], 2)?;
    let internal = AuxiliaryOverlapThreshold::new(&exclusion, vec![], 3)?;
    for order in ORDERS {
        for seed in 1..=8 {
            let p = proposal(&base, order, caps())?;
            let (mut plain, mut guided) =
                (StdRng::seed_from_u64(seed), StdRng::seed_from_u64(seed));
            let (mut root_rng, mut internal_rng) = (
                StdRng::seed_from_u64(seed + 100),
                StdRng::seed_from_u64(seed + 200),
            );
            let a = p.propose(&mut plain, &context, [state[0], state[1]])?;
            let b = p.propose_with_guides(
                &mut guided,
                &context,
                [state[0], state[1]],
                Some(&root),
                Some(&mut root_rng),
                Some(&internal),
                Some(&mut internal_rng),
            )?;
            let mut value = serde_json::to_value(&b)?;
            strip_guidance(&mut value);
            assert_eq!(value, serde_json::to_value(&a)?);
            assert_eq!(plain.random::<u64>(), guided.random::<u64>());
            for (d, rng, seed, m) in [
                (
                    b.root_guidance.as_ref().unwrap(),
                    &mut root_rng,
                    seed + 100,
                    2,
                ),
                (
                    b.guidance.as_ref().unwrap(),
                    &mut internal_rng,
                    seed + 200,
                    3,
                ),
            ] {
                assert_eq!(d.old_count, Some(0));
                assert_eq!(d.threshold, Some(0));
                let mut replay = StdRng::seed_from_u64(seed);
                let uniform = Uniform::new_inclusive(0usize, 0)?;
                let expected = (0..m)
                    .map(|_| uniform.sample(&mut replay))
                    .collect::<Vec<_>>();
                assert_eq!(d.integer_draws, expected);
                assert_eq!(rng.random::<u64>(), replay.random::<u64>());
            }
            if b.candidate.is_some() {
                assert_eq!(b.complete_log_correction()?, a.complete_log_correction()?);
            }
        }
    }
    Ok(())
}

#[test]
fn zero_caps_and_source_outside_preserve_all_three_rngs_and_no_thresholds() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(1.5), pose(3.), pose(0.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0.1)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.1, 0., 0.]], 2)?;
    for order in ORDERS {
        for mode in 0..4 {
            let mut cap = caps();
            match mode {
                0 => cap.root = 0,
                1 => cap.internal = 0,
                2 => cap.joint = 0,
                _ => (),
            };
            let old = if mode == 3 {
                [pose(1.5), pose(4.)]
            } else {
                [state[0], state[1]]
            };
            let (mut rng, mut a, mut b) = (
                StdRng::seed_from_u64(1),
                StdRng::seed_from_u64(2),
                StdRng::seed_from_u64(3),
            );
            let result = proposal(&base, order, cap)?.propose_with_guides(
                &mut rng,
                &context,
                old,
                Some(&guide),
                Some(&mut a),
                Some(&guide),
                Some(&mut b),
            )?;
            assert_eq!(
                result.status,
                if mode == 3 {
                    FactorizedDimerStatus::SourceOutsideDomain
                } else {
                    FactorizedDimerStatus::CapExhausted
                }
            );
            assert!(result.attempts.is_empty());
            for d in [
                result.root_guidance.as_ref().unwrap(),
                result.guidance.as_ref().unwrap(),
            ] {
                assert!(d.threshold.is_none() && d.integer_draws.is_empty());
                assert_eq!(d.count_queries, 4);
            }
            assert_eq!(
                rng.random::<u64>(),
                StdRng::seed_from_u64(1).random::<u64>()
            );
            assert_eq!(a.random::<u64>(), StdRng::seed_from_u64(2).random::<u64>());
            assert_eq!(b.random::<u64>(), StdRng::seed_from_u64(3).random::<u64>());
        }
    }
    Ok(())
}

#[test]
fn global_quarter_turn_preserves_anchor_frame_counts_and_candidates() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(1.5), pose(3.), pose(0.)];
    let transform = Pose {
        position: [5., 6., 7.],
        orientation: [0.5, 0.5, 0.5, 0.5],
    };
    let moved = state.map(|p| Pose {
        position: transform.apply(p.position),
        orientation: transform.orientation,
    });
    let base = atlas(1e-4)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![[0.1, 0., 0.]], 2)?;
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let transformed = FixedDimerContext::new(
        &core,
        &exclusion,
        &moved,
        [0, 1],
        2,
        None,
        transform.position,
    )?;
    for order in ORDERS {
        let p = proposal(&base, order, caps())?;
        let run = |c: &FixedDimerContext<'_>, old: [Pose; 2]| {
            p.propose_with_guides(
                &mut StdRng::seed_from_u64(8),
                c,
                old,
                Some(&guide),
                Some(&mut StdRng::seed_from_u64(10)),
                Some(&guide),
                Some(&mut StdRng::seed_from_u64(11)),
            )
        };
        let a = run(&context, [state[0], state[1]])?;
        let b = run(&transformed, [moved[0], moved[1]])?;
        assert_eq!(a.status, FactorizedDimerStatus::Candidate);
        assert_eq!(a.status, b.status);
        for (x, y) in [
            (
                a.root_guidance.as_ref().unwrap(),
                b.root_guidance.as_ref().unwrap(),
            ),
            (a.guidance.as_ref().unwrap(), b.guidance.as_ref().unwrap()),
        ] {
            assert_eq!(serde_json::to_value(x)?, serde_json::to_value(y)?);
        }
        let ac = a.candidate.as_ref().unwrap();
        let bc = b.candidate.as_ref().unwrap();
        for (x, y) in [(ac.root, bc.root), (ac.child, bc.child)] {
            let pos = transform.apply(x.position);
            let rot = matmul(rotation(transform.orientation), rotation(x.orientation));
            for k in 0..3 {
                assert!((pos[k] - y.position[k]).abs() < 1e-12);
                for j in 0..3 {
                    assert!((rot[k][j] - rotation(y.orientation)[k][j]).abs() < 1e-12);
                }
            }
        }
        assert!((a.complete_log_correction()? - b.complete_log_correction()?).abs() < 1e-8);
    }
    Ok(())
}

#[test]
fn root_frame_disagreement_is_fatal_before_any_rng_or_retry() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(1e14 + 1.), pose(1e14 + 2.), pose(1e14)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0.1)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![[-0.003, 0., 0.]], 2)?;
    let (mut rng, mut aux) = (StdRng::seed_from_u64(1), StdRng::seed_from_u64(2));
    let failure = proposal(&base, FactorizedDimerOrder::RootFirst, caps())?
        .propose_with_guides(
            &mut rng,
            &context,
            [state[0], state[1]],
            Some(&guide),
            Some(&mut aux),
            None,
            None,
        )
        .unwrap_err();
    assert!(
        failure
            .fatal_error
            .contains("guidance count frame mismatch")
    );
    let result = failure.outcome.unwrap();
    assert!(result.attempts.is_empty());
    let counts = result.source_frame.unwrap().root_guidance.unwrap();
    assert_eq!(counts.relative_count, Some(0));
    assert_eq!(counts.recovered_count, Some(0));
    assert_eq!(counts.world_count, Some(1));
    assert_eq!(counts.reconstructed_world_count, Some(1));
    let d = result.root_guidance.unwrap();
    assert_eq!(d.old_count, Some(0));
    assert_eq!(d.count_queries, 4);
    assert!(d.integer_draws.is_empty() && d.threshold.is_none());
    assert_eq!(
        rng.random::<u64>(),
        StdRng::seed_from_u64(1).random::<u64>()
    );
    assert_eq!(
        aux.random::<u64>(),
        StdRng::seed_from_u64(2).random::<u64>()
    );
    Ok(())
}

#[test]
fn invalid_root_shape_or_missing_required_stream_is_fatal_not_an_mh_self_loop() -> Result<()> {
    let (core, exclusion, wrong) = (sphere(0.2), sphere(1.), sphere(1.1));
    let state = [pose(1.5), pose(3.), pose(0.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0.1)?;
    for mismatch in [true, false] {
        let guide = AuxiliaryOverlapThreshold::new(
            if mismatch { &wrong } else { &exclusion },
            vec![[0.1, 0., 0.]],
            2,
        )?;
        let (mut rng, mut aux) = (StdRng::seed_from_u64(1), StdRng::seed_from_u64(2));
        let failure = proposal(&base, FactorizedDimerOrder::RootFirst, caps())?
            .propose_with_guides(
                &mut rng,
                &context,
                [state[0], state[1]],
                Some(&guide),
                if mismatch { Some(&mut aux) } else { None },
                None,
                None,
            )
            .unwrap_err();
        assert!(failure.fatal_error.contains(if mismatch {
            "exclusion shape differs"
        } else {
            "RNG"
        }));
        let outcome = failure.outcome.unwrap();
        assert!(outcome.attempts.is_empty());
        assert!(outcome.root_guidance.unwrap().integer_draws.is_empty());
        assert_eq!(
            rng.random::<u64>(),
            StdRng::seed_from_u64(1).random::<u64>()
        );
        assert_eq!(
            aux.random::<u64>(),
            StdRng::seed_from_u64(2).random::<u64>()
        );
    }
    Ok(())
}

#[test]
fn positive_root_threshold_rejects_hard_valid_roots_until_every_cap_is_exhausted() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(-1.5), pose(-3.), pose(0.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(1e-12)?;
    let guide = AuxiliaryOverlapThreshold::new(&exclusion, vec![[-0.9, 0., 0.]], 4)?;
    let cap = FactorizedDimerCaps {
        root: 3,
        internal: 2,
        joint: 2,
    };
    for order in ORDERS {
        let (mut rng, mut root_rng, mut expected_rng, mut expected_aux) = (
            StdRng::seed_from_u64(8),
            StdRng::seed_from_u64(44),
            StdRng::seed_from_u64(8),
            StdRng::seed_from_u64(44),
        );
        let expected_integers = (0..4)
            .map(|_| {
                Uniform::new_inclusive(0usize, 1)
                    .unwrap()
                    .sample(&mut expected_aux)
            })
            .collect::<Vec<_>>();
        assert_eq!(expected_integers.iter().max(), Some(&1));
        let result = proposal(&base, order, cap)?.propose_with_guides(
            &mut rng,
            &context,
            [state[0], state[1]],
            Some(&guide),
            Some(&mut root_rng),
            None,
            None,
        )?;
        assert_eq!(result.status, FactorizedDimerStatus::CapExhausted);
        assert!(result.candidate.is_none());
        assert_eq!(result.attempts.len(), cap.joint);
        let diagnostics = result.root_guidance.as_ref().unwrap();
        assert_eq!(diagnostics.old_count, Some(1));
        assert_eq!(diagnostics.threshold, Some(1));
        assert_eq!(diagnostics.integer_draws, expected_integers);
        assert!(diagnostics.new_count.is_none() && diagnostics.aux_log_correction.is_none());
        assert_eq!(diagnostics.count_queries, 4 + cap.root * cap.joint);
        let raw = DefensiveDimerProposal::new(&base, 4., 0.)?;
        for attempt in &result.attempts {
            assert_eq!(attempt.status, FactorizedAttemptStatus::RootCapExhausted);
            assert_eq!(attempt.root_draws.len(), cap.root);
            assert!(attempt.proposed.is_none() && attempt.frame.is_none());
            let internal_count = if order == FactorizedDimerOrder::InternalFirst {
                1
            } else {
                0
            };
            assert_eq!(attempt.internal_draws.len(), internal_count);
            for _ in 0..internal_count {
                raw.draw_edge(&mut expected_rng, pose(-1.5));
            }
            for record in &attempt.root_draws {
                assert!(record.feasibility.as_ref().unwrap().hard_valid());
                assert_eq!(record.guidance_count, Some(0));
                // Independent single-sphere geometry: the fixed anchor point
                // lies more than one exclusion radius from the sampled root.
                let world = record.world_pose.unwrap().position;
                let squared_distance =
                    (world[0] + 0.9).powi(2) + world[1].powi(2) + world[2].powi(2);
                assert!(squared_distance > 1.);
                raw.draw_edge(&mut expected_rng, pose(-1.5));
            }
        }
        assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
        assert_eq!(root_rng.random::<u64>(), expected_aux.random::<u64>());
    }
    Ok(())
}
