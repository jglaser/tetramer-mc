//! Staged stopping, replay, full-density and strict frame controls; no bath.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    capped_dimer::FixedDimerContext,
    defensive_dimer_proposal::DefensiveDimerProposal,
    dimer_tree_proposal::{tree_coordinates, tree_members},
    docking::{DockingMethod, DockingProposal},
    factorized_dimer::{
        FactorizedAttemptStatus, FactorizedDimerCaps, FactorizedDimerOrder,
        FactorizedDimerProposal, FactorizedDimerStatus,
    },
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, norm, rotation, sub},
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

#[test]
fn cheap_edges_reproduce_existing_joint_draws_and_rng_exactly() -> Result<()> {
    let base = atlas(0., 1., false)?;
    let raw = DefensiveDimerProposal::new(&base, 2.5, 0.5)?;
    let (anchor, root, child) = (pose(3.), pose(0.), pose(1.));
    let old = tree_coordinates(anchor, root, child)?;
    let mut joint_rng = StdRng::seed_from_u64(615);
    let mut edge_rng = StdRng::seed_from_u64(615);
    for _ in 0..24 {
        let joint = raw.propose(&mut joint_rng, anchor, root, child)?;
        let edges = [
            raw.draw_edge(&mut edge_rng, old[0]),
            raw.draw_edge(&mut edge_rng, old[1]),
        ];
        assert_eq!(
            serde_json::to_value(&joint.edges)?,
            serde_json::to_value(edges)?
        );
    }
    assert_eq!(joint_rng.random::<u64>(), edge_rng.random::<u64>());
    // The old coordinate annotates a draw; it never influences its random law.
    let a = raw.draw_edge(&mut StdRng::seed_from_u64(93), pose(0.));
    let b = raw.draw_edge(&mut StdRng::seed_from_u64(93), pose(100.));
    assert_eq!(a.trace, b.trace);
    assert_eq!(a.proposed_relative_pose, b.proposed_relative_pose);
    Ok(())
}

#[test]
fn zero_caps_and_source_outside_domain_preserve_rng_in_both_orders() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(1.), pose(5.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0., 1., false)?;
    for order in ORDERS {
        for caps in [
            FactorizedDimerCaps {
                root: 0,
                internal: 3,
                joint: 4,
            },
            FactorizedDimerCaps {
                root: 3,
                internal: 0,
                joint: 4,
            },
            FactorizedDimerCaps {
                root: 3,
                internal: 3,
                joint: 0,
            },
        ] {
            let proposal = FactorizedDimerProposal::new(
                DefensiveDimerProposal::new(&base, 2.5, 0.5)?,
                caps,
                order,
            );
            let mut rng = StdRng::seed_from_u64(22);
            let mut untouched = StdRng::seed_from_u64(22);
            let result = proposal.propose(&mut rng, &context, [state[0], state[1]])?;
            assert_eq!(result.status, FactorizedDimerStatus::CapExhausted);
            assert!(result.attempts.is_empty() && result.candidate.is_none());
            assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
        }
        let proposal = FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(&base, 2.5, 0.5)?,
            FactorizedDimerCaps {
                root: 3,
                internal: 3,
                joint: 4,
            },
            order,
        );
        let mut rng = StdRng::seed_from_u64(28);
        let mut untouched = StdRng::seed_from_u64(28);
        let result = proposal.propose(&mut rng, &context, [pose(0.), pose(2.)])?;
        assert_eq!(result.status, FactorizedDimerStatus::SourceOutsideDomain);
        assert!(result.attempts.is_empty());
        assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    }
    Ok(())
}

#[test]
fn first_stage_exhaustion_is_bounded_and_short_circuits_other_edge() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0., 1., false)?;
    // The entire tiny uniform cube overlaps its anchor and the identity root.
    for order in ORDERS {
        let proposal = FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(&base, 0.1, 1.)?,
            FactorizedDimerCaps {
                root: 4,
                internal: 4,
                joint: 3,
            },
            order,
        );
        let result = proposal.propose(
            &mut StdRng::seed_from_u64(41),
            &context,
            [state[0], state[1]],
        )?;
        assert_eq!(result.status, FactorizedDimerStatus::CapExhausted);
        assert_eq!(result.attempts.len(), 3);
        for (index, attempt) in result.attempts.iter().enumerate() {
            assert_eq!(attempt.index, index + 1);
            assert!(attempt.proposed.is_none() && attempt.final_feasibility.is_none());
            match order {
                FactorizedDimerOrder::RootFirst => {
                    assert_eq!(attempt.status, FactorizedAttemptStatus::RootCapExhausted);
                    assert_eq!(attempt.root_draws.len(), 4);
                    assert!(attempt.internal_draws.is_empty());
                    assert!(
                        attempt.root_draws.iter().all(|d| !d
                            .feasibility
                            .as_ref()
                            .unwrap()
                            .hard_valid())
                    );
                }
                FactorizedDimerOrder::InternalFirst => {
                    assert_eq!(
                        attempt.status,
                        FactorizedAttemptStatus::InternalCapExhausted
                    );
                    assert_eq!(attempt.internal_draws.len(), 4);
                    assert!(attempt.root_draws.is_empty());
                    assert!(
                        attempt.internal_draws.iter().all(|d| d
                            .feasibility
                            .as_ref()
                            .unwrap()
                            .internal_core_overlap)
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn final_child_wall_and_spectator_failures_discard_both_edges() -> Result<()> {
    let (core, exclusion) = (sphere(0.05), sphere(1.));
    let base = atlas(1.5, 1e-8, false)?;
    for order in ORDERS {
        for child_wall in [true, false] {
            let mut state = vec![pose(-1.), pose(-0.3), pose(0.)];
            if !child_wall {
                state.push(pose(3.));
            }
            let context = FixedDimerContext::new(
                &core,
                &exclusion,
                &state,
                [0, 1],
                2,
                if child_wall { Some(2.) } else { None },
                [0.; 3],
            )?;
            // Old selected poses are NOT fixed obstacles in the root prefilter.
            assert!(context.evaluate_fixed_body(state[0])?.hard_valid());
            let proposal = FactorizedDimerProposal::new(
                DefensiveDimerProposal::new(&base, 4., 0.)?,
                FactorizedDimerCaps {
                    root: 2,
                    internal: 2,
                    joint: 3,
                },
                order,
            );
            let result = proposal.propose(
                &mut StdRng::seed_from_u64(8),
                &context,
                [state[0], state[1]],
            )?;
            assert_eq!(result.status, FactorizedDimerStatus::CapExhausted);
            assert_eq!(result.attempts.len(), 3);
            for attempt in &result.attempts {
                assert_eq!(attempt.status, FactorizedAttemptStatus::FinalRejected);
                assert_eq!(attempt.root_draws.len(), 1);
                assert_eq!(attempt.internal_draws.len(), 1);
                assert!(
                    attempt.root_draws[0]
                        .feasibility
                        .as_ref()
                        .unwrap()
                        .hard_valid()
                );
                assert!(
                    attempt.internal_draws[0]
                        .feasibility
                        .as_ref()
                        .unwrap()
                        .feasible()
                );
                let full = attempt.final_feasibility.as_ref().unwrap();
                if child_wall {
                    assert_eq!(full.wall_valid, [true, false]);
                } else {
                    assert_eq!(full.spectator_core_collisions[1], vec![3]);
                }
            }
            assert_ne!(
                result.attempts[0].root_draws[0].draw.trace,
                result.attempts[1].root_draws[0].draw.trace
            );
            assert_ne!(
                result.attempts[0].internal_draws[0].draw.trace,
                result.attempts[1].internal_draws[0].draw.trace
            );
        }
    }
    Ok(())
}

#[test]
fn all_draws_replay_with_exact_coordinates_and_complete_final_correction() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, Some(8.), [0.; 3])?;
    let base = atlas(0., 1., false)?;
    let raw = DefensiveDimerProposal::new(&base, 2.5, 0.5)?;
    let old_edges = tree_coordinates(state[2], state[0], state[1])?;
    let mut successes = 0;
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
        for seed in 1..=16 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut replay = StdRng::seed_from_u64(seed);
            let result = proposal.propose(&mut rng, &context, [state[0], state[1]])?;
            for attempt in &result.attempts {
                let root_first = order == FactorizedDimerOrder::RootFirst;
                let edge_order = if root_first { [0, 1] } else { [1, 0] };
                for edge in edge_order {
                    if edge == 0 {
                        for (i, record) in attempt.root_draws.iter().enumerate() {
                            assert_eq!(record.index, i + 1);
                            let expected = raw.draw_edge(&mut replay, old_edges[0]);
                            assert_eq!(
                                serde_json::to_value(&record.draw)?,
                                serde_json::to_value(expected)?
                            );
                            if i + 1 < attempt.root_draws.len() {
                                assert!(!record.feasibility.as_ref().unwrap().hard_valid());
                            }
                        }
                    } else {
                        for (i, record) in attempt.internal_draws.iter().enumerate() {
                            assert_eq!(record.index, i + 1);
                            let expected = raw.draw_edge(&mut replay, old_edges[1]);
                            assert_eq!(
                                serde_json::to_value(&record.draw)?,
                                serde_json::to_value(expected)?
                            );
                            if i + 1 < attempt.internal_draws.len() {
                                assert!(!record.feasibility.as_ref().unwrap().feasible());
                            }
                        }
                    }
                }
                if let Some(members) = attempt.proposed {
                    let a = attempt.root_draws.last().unwrap();
                    let b = attempt.internal_draws.last().unwrap();
                    assert_eq!(Some(members[0]), a.world_pose);
                    assert_eq!(
                        members,
                        tree_members(
                            state[2],
                            [
                                a.draw.proposed_relative_pose.unwrap(),
                                b.draw.proposed_relative_pose.unwrap()
                            ]
                        )?
                    );
                    let frame = attempt.frame.as_ref().unwrap();
                    assert_eq!(
                        frame.reconstructed_feasibility,
                        *attempt.final_feasibility.as_ref().unwrap()
                    );
                    assert_eq!(frame.internal_relative, *b.feasibility.as_ref().unwrap());
                }
            }
            assert_eq!(rng.random::<u64>(), replay.random::<u64>());
            if let Some(candidate) = result.candidate {
                successes += 1;
                assert_eq!(
                    result.attempts.last().unwrap().status,
                    FactorizedAttemptStatus::Candidate
                );
                let expected = raw.correction(
                    state[2],
                    state[0],
                    state[1],
                    candidate.root,
                    candidate.child,
                )?;
                assert_eq!(
                    serde_json::to_value(&candidate.diagnostics)?,
                    serde_json::to_value(expected)?
                );
                let reverse = raw.correction(
                    state[2],
                    candidate.root,
                    candidate.child,
                    state[0],
                    state[1],
                )?;
                assert!(
                    (reverse.log_reverse_forward + candidate.diagnostics.log_reverse_forward).abs()
                        < 1e-10
                );
            }
        }
    }
    assert!(successes > 0);
    Ok(())
}

#[test]
fn numerical_null_is_fatal_and_retains_first_edge_without_retry() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(0., 1., true)?;
    let raw = DefensiveDimerProposal::new(&base, 2.5, 0.)?;
    let old_edges = tree_coordinates(state[2], state[0], state[1])?;
    for order in ORDERS {
        let proposal = FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(&base, 2.5, 0.)?,
            FactorizedDimerCaps {
                root: 8,
                internal: 8,
                joint: 3,
            },
            order,
        );
        let mut rng = StdRng::seed_from_u64(33);
        let mut replay = StdRng::seed_from_u64(33);
        let failure = proposal
            .propose(&mut rng, &context, [state[0], state[1]])
            .unwrap_err();
        let result = failure.outcome.as_ref().unwrap();
        assert_eq!(result.attempts.len(), 1);
        let attempt = &result.attempts[0];
        assert_eq!(attempt.status, FactorizedAttemptStatus::InProgress);
        let edge = if order == FactorizedDimerOrder::RootFirst {
            assert!(attempt.internal_draws.is_empty());
            assert_eq!(attempt.root_draws.len(), 1);
            &attempt.root_draws[0].draw
        } else {
            assert!(attempt.root_draws.is_empty());
            assert_eq!(attempt.internal_draws.len(), 1);
            &attempt.internal_draws[0].draw
        };
        let expected = raw.draw_edge(
            &mut replay,
            old_edges[usize::from(order == FactorizedDimerOrder::InternalFirst)],
        );
        assert!(edge.proposed_relative_pose.is_none());
        assert_eq!(serde_json::to_value(edge)?, serde_json::to_value(expected)?);
        assert_eq!(rng.random::<u64>(), replay.random::<u64>());
        assert!(serde_json::to_string(&failure)?.contains("fatal_error"));
    }
    Ok(())
}

#[test]
fn source_frame_predicate_mismatch_is_fatal_before_rng() -> Result<()> {
    let (core, exclusion) = (sphere(0.0001), sphere(1.));
    let scale = 0.98_f64.sqrt();
    let anchor = Pose {
        position: [-4e14, 5e14, -6e14],
        orientation: [0.7 / scale, 0.2 / scale, 0.3 / scale, 0.6 / scale],
    };
    let root = Pose {
        position: [1e14, -2e14, 3e14],
        ..pose(0.)
    };
    let child = Pose {
        position: [1e14 + 1., -2e14, 3e14],
        ..pose(0.)
    };
    let reconstructed = tree_members(anchor, tree_coordinates(anchor, root, child)?)?;
    // Fixed adversarial geometry: coordinate recomposition moves the root by
    // much more than the small hard diameter. Place a fixed witness there.
    assert!(norm(sub(reconstructed[0].position, root.position)) > 0.001);
    let state = [root, child, anchor, reconstructed[0]];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    assert!(context.evaluate([root, child])?.feasible());
    let base = atlas(0., 1., false)?;
    let proposal = FactorizedDimerProposal::new(
        DefensiveDimerProposal::new(&base, 2.5, 0.5)?,
        FactorizedDimerCaps {
            root: 3,
            internal: 3,
            joint: 3,
        },
        FactorizedDimerOrder::RootFirst,
    );
    let mut rng = StdRng::seed_from_u64(91);
    let mut untouched = StdRng::seed_from_u64(91);
    let failure = proposal
        .propose(&mut rng, &context, [root, child])
        .unwrap_err();
    assert!(failure.fatal_error.contains("frame predicate mismatch"));
    let result = failure.outcome.unwrap();
    assert!(result.source_frame.is_some() && result.attempts.is_empty());
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    Ok(())
}

#[test]
fn invalid_source_and_seed_prefix_restart_preserve_failure_and_snapshot() -> Result<()> {
    let (core, exclusion) = (sphere(0.2), sphere(1.));
    let mut state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
    state[2] = pose(90.);
    assert_eq!(context.anchor(), pose(3.));
    let base = atlas(0., 1., false)?;
    let proposal = FactorizedDimerProposal::new(
        DefensiveDimerProposal::new(&base, 2.5, 0.5)?,
        FactorizedDimerCaps {
            root: 4,
            internal: 4,
            joint: 2,
        },
        FactorizedDimerOrder::RootFirst,
    );
    let mut rng = StdRng::seed_from_u64(11);
    let mut untouched = StdRng::seed_from_u64(11);
    let failure = proposal
        .propose(&mut rng, &context, [pose(0.), pose(0.1)])
        .unwrap_err();
    assert!(failure.outcome.unwrap().attempts.is_empty());
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    let mut continuous = StdRng::seed_from_u64(81);
    for _ in 0..3 {
        proposal.propose(&mut continuous, &context, [state[0], state[1]])?;
    }
    let mut resumed = StdRng::seed_from_u64(81);
    for _ in 0..3 {
        proposal.propose(&mut resumed, &context, [state[0], state[1]])?;
    }
    // Seed/prefix replay only; no RNG serialization/checkpoint API is claimed.
    for _ in 0..8 {
        let a = proposal.propose(&mut continuous, &context, [state[0], state[1]])?;
        let b = proposal.propose(&mut resumed, &context, [state[0], state[1]])?;
        assert_eq!(serde_json::to_value(a)?, serde_json::to_value(b)?);
    }
    Ok(())
}
