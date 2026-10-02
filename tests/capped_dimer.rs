//! Conditional-joint geometry, stopped-trial, failure and finite-flow controls.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    capped_dimer::{CappedDimerProposal, CappedDimerStatus, FixedDimerContext},
    defensive_dimer_proposal::DefensiveDimerProposal,
    docking::{DockingMethod, DockingProposal},
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, rotation},
    proposal::FrozenRelativePoseProposal,
};
const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
fn pose(x: f64) -> Pose {
    Pose {
        position: [x, 0., 0.],
        orientation: [1., 0., 0., 0.],
    }
}
fn sphere(r: f64) -> SphereTree {
    SphereTree::new(Shape {
        name: "capped sphere".into(),
        volume: 4. * PI * r.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: r,
        }],
    })
    .unwrap()
}
fn atlas(broken: bool) -> Result<DockingProposal> {
    let mean = if broken {
        [1e308, 0., 0., 0., 0., 0.]
    } else {
        [0.; 6]
    };
    let anchor = if broken { pose(1e308) } else { pose(0.) };
    let cov: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let raw=json!({"coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
        "angular_length":1.,"weights":[1.],"anchors":[{"position":anchor.position,"rotation":rotation(anchor.orientation)}],
        "means":[mean],"covariances":[cov]}).to_string();
    DockingProposal::new(
        FrozenRelativePoseProposal::from_json_str_open(&raw, [80.; 3], 0.1, SHA)?,
        DockingMethod::PosteriorInvolution,
        0.7,
        [0.; 3],
    )
}
#[test]
fn full_predicate_includes_members_spectators_atomic_wall_and_strict_contact() -> Result<()> {
    let core = sphere(0.2);
    let exc = sphere(1.);
    let state = [pose(0.), pose(1.), pose(3.5), pose(-2.)];
    let context = FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, Some(4.), [0.; 3])?;
    assert!(context.evaluate([pose(0.), pose(1.)])?.feasible());
    // Core tangency is allowed; exclusion tangency is outside strict contact.
    assert!(context.evaluate([pose(0.), pose(0.4)])?.feasible());
    assert!(
        !context
            .evaluate([pose(0.), pose(2.)])?
            .internal_exclusion_contact
    );
    assert!(
        context
            .evaluate([pose(0.), pose(0.39)])?
            .internal_core_overlap
    );
    let external = context.evaluate([pose(-2.), pose(-1.)])?;
    assert_eq!(external.spectator_core_collisions[0], vec![3]);
    assert!(!external.feasible());
    let edge = context.evaluate([pose(3.), pose(3.81)])?;
    assert!(!edge.wall_valid[1]);
    // A shifted sphere uses actual atomic radius, not a center-only wall.
    let shifted = FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, Some(4.), [10., 0., 0.])?;
    assert!(shifted.evaluate([pose(10.), pose(13.79)])?.wall_valid[1]);
    assert!(!shifted.evaluate([pose(10.), pose(13.81)])?.wall_valid[1]);
    Ok(())
}
#[test]
fn source_outside_contact_and_zero_cap_are_true_rng_preserving_self_loops() -> Result<()> {
    let core = sphere(0.2);
    let exc = sphere(1.);
    let state = [pose(0.), pose(1.), pose(5.)];
    let context = FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(false)?;
    for (cap, old, status) in [
        (
            16,
            [pose(0.), pose(2.1)],
            CappedDimerStatus::SourceOutsideContact,
        ),
        (0, [pose(0.), pose(1.)], CappedDimerStatus::CapExhausted),
    ] {
        let proposal = CappedDimerProposal::new(DefensiveDimerProposal::new(&base, 2.5, 0.5)?, cap);
        let mut rng = StdRng::seed_from_u64(738);
        let mut untouched = StdRng::seed_from_u64(738);
        let out = proposal.propose(&mut rng, &context, old)?;
        assert_eq!(out.status, status);
        assert!(out.candidate.is_none());
        assert!(out.trials.is_empty());
        assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    }
    Ok(())
}
#[test]
fn cap_exhaustion_records_every_whole_joint_trial_without_a_candidate() -> Result<()> {
    let core = sphere(0.2);
    let exc = sphere(1.);
    let state = [pose(0.), pose(1.), pose(20.)];
    let context = FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(false)?;
    // Every root uniform draw is within sqrt(3)*0.1 < 0.4 of its hard anchor.
    let proposal = CappedDimerProposal::new(DefensiveDimerProposal::new(&base, 0.1, 1.)?, 8);
    let out = proposal.propose(
        &mut StdRng::seed_from_u64(42),
        &context,
        [state[0], state[1]],
    )?;
    assert_eq!(out.status, CappedDimerStatus::CapExhausted);
    assert!(out.candidate.is_none());
    assert_eq!(out.trials.len(), 8);
    for (i, t) in out.trials.iter().enumerate() {
        assert_eq!(t.index, i + 1);
        assert!(t.draw.as_ref().unwrap().candidate.is_some());
        let f = t.feasibility.as_ref().unwrap();
        assert!(!f.feasible());
        assert_eq!(f.spectator_core_collisions[0], vec![2]);
    }
    Ok(())
}
#[test]
fn stopped_draw_is_exact_raw_prefix_and_keeps_complete_density_ratio() -> Result<()> {
    let core = sphere(0.2);
    let exc = sphere(1.);
    let state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, Some(8.), [0.; 3])?;
    let base = atlas(false)?;
    let raw = DefensiveDimerProposal::new(&base, 2.5, 0.5)?;
    let capped = CappedDimerProposal::new(DefensiveDimerProposal::new(&base, 2.5, 0.5)?, 16);
    let mut successes = 0;
    let mut failures_before_success = 0;
    for seed in 1..=32 {
        let mut a = StdRng::seed_from_u64(seed);
        let mut b = StdRng::seed_from_u64(seed);
        let out = capped.propose(&mut a, &context, [state[0], state[1]])?;
        for (i, t) in out.trials.iter().enumerate() {
            let reference = raw.propose(&mut b, context.anchor(), state[0], state[1])?;
            assert_eq!(
                serde_json::to_value(t.draw.as_ref().unwrap())?,
                serde_json::to_value(&reference)?
            );
            if i + 1 < out.trials.len() {
                assert!(!t.feasibility.as_ref().unwrap().feasible());
            }
        }
        assert_eq!(a.random::<u64>(), b.random::<u64>());
        if let Some(c) = out.candidate {
            successes += 1;
            failures_before_success += out.trials.len() - 1;
            assert!(
                out.trials
                    .last()
                    .unwrap()
                    .feasibility
                    .as_ref()
                    .unwrap()
                    .feasible()
            );
            let back = raw.correction(context.anchor(), c.root, c.child, state[0], state[1])?;
            assert!((back.log_reverse_forward + c.diagnostics.log_reverse_forward).abs() < 1e-10);
        } else {
            assert_eq!(out.trials.len(), 16);
        }
    }
    assert!(successes > 0 && failures_before_success > 0);
    Ok(())
}
#[test]
fn computational_null_is_fatal_with_its_raw_trace_and_never_retried() -> Result<()> {
    let core = sphere(0.2);
    let exc = sphere(1.);
    let state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, None, [0.; 3])?;
    let broken = atlas(true)?;
    let raw = DefensiveDimerProposal::new(&broken, 2.5, 0.)?;
    let capped = CappedDimerProposal::new(DefensiveDimerProposal::new(&broken, 2.5, 0.)?, 16);
    let mut a = StdRng::seed_from_u64(33);
    let mut b = StdRng::seed_from_u64(33);
    let failure = capped
        .propose(&mut a, &context, [state[0], state[1]])
        .unwrap_err();
    assert_eq!(failure.trials.len(), 1);
    let expected = raw.propose(&mut b, context.anchor(), state[0], state[1])?;
    assert!(expected.candidate.is_none());
    assert_eq!(
        serde_json::to_value(&failure.trials[0].draw)?,
        serde_json::to_value(Some(expected))?
    );
    assert_eq!(a.random::<u64>(), b.random::<u64>());
    assert!(serde_json::to_string(&failure)?.contains("fatal_error"));
    Ok(())
}
#[test]
fn invalid_context_source_and_shifted_arithmetic_fail_explicitly() -> Result<()> {
    let core = sphere(0.2);
    let exc = sphere(1.);
    let state = [pose(0.), pose(1.), pose(3.)];
    for (members, anchor) in [([0, 0], 2), ([0, 9], 2), ([0, 1], 1)] {
        assert!(
            FixedDimerContext::new(&core, &exc, &state, members, anchor, None, [0.; 3]).is_err()
        );
    }
    assert!(FixedDimerContext::new(&exc, &core, &state, [0, 1], 2, None, [0.; 3]).is_err());
    assert!(FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, Some(0.1), [0.; 3]).is_err());
    let context = FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, None, [0.; 3])?;
    let base = atlas(false)?;
    let capped = CappedDimerProposal::new(DefensiveDimerProposal::new(&base, 2.5, 0.5)?, 16);
    let mut rng = StdRng::seed_from_u64(7);
    let mut untouched = StdRng::seed_from_u64(7);
    let failure = capped
        .propose(&mut rng, &context, [pose(0.), pose(0.1)])
        .unwrap_err();
    assert!(failure.trials.is_empty());
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    let huge = FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, Some(4.), [1e308, 0., 0.])?;
    assert!(huge.evaluate([pose(0.), pose(1.)]).is_err());
    Ok(())
}
#[test]
fn stream_continuation_and_spectator_snapshot_are_deterministic() -> Result<()> {
    let core = sphere(0.2);
    let exc = sphere(1.);
    let mut state = [pose(0.), pose(1.), pose(3.)];
    let context = FixedDimerContext::new(&core, &exc, &state, [0, 1], 2, Some(8.), [0.; 3])?;
    state[2] = pose(90.); // immutable context must not silently follow external state.
    assert_eq!(context.anchor(), pose(3.));
    let base = atlas(false)?;
    let capped = CappedDimerProposal::new(DefensiveDimerProposal::new(&base, 2.5, 0.5)?, 4);
    let mut continuous = StdRng::seed_from_u64(81);
    for _ in 0..3 {
        capped.propose(&mut continuous, &context, [state[0], state[1]])?;
    }
    // StdRng intentionally has no serialized/clone state. Replay the frozen
    // event prefix to reconstruct its stream; this is not a checkpoint format.
    let mut resumed = StdRng::seed_from_u64(81);
    for _ in 0..3 {
        capped.propose(&mut resumed, &context, [state[0], state[1]])?;
    }
    for _ in 0..8 {
        let a = capped.propose(&mut continuous, &context, [state[0], state[1]])?;
        let b = capped.propose(&mut resumed, &context, [state[0], state[1]])?;
        assert_eq!(serde_json::to_value(a)?, serde_json::to_value(b)?);
    }
    Ok(())
}
#[test]
fn finite_state_full_mixture_flow_and_rejection_completion_include_outside_region() {
    // Explicit stopped sequences on a discrete analogue, with unequal target
    // and mixture weights. No unknown feasibility normalizer is used in MH.
    let pi: [f64; 4] = [0.05, 0.2, 0.3, 0.45];
    let f: [f64; 4] = [0.1, 0.3, 0.2, 0.4];
    let allowed = [true, true, true, false];
    fn walk(f: &[f64; 4], d: &[bool; 4], left: usize, mass: f64, q: &mut [f64; 5]) {
        if left == 0 {
            q[4] += mass;
            return;
        }
        for y in 0..4 {
            if d[y] {
                q[y] += mass * f[y];
            } else {
                walk(f, d, left - 1, mass * f[y], q);
            }
        }
    }
    for cap in 0..=4 {
        let mut stopped = [0.; 5];
        walk(&f, &allowed, cap, 1., &mut stopped);
        assert!((stopped.iter().sum::<f64>() - 1.).abs() < 1e-14);
        let mut kernel = [[0.; 4]; 4];
        for x in 0..4 {
            if !allowed[x] {
                kernel[x][x] = 1.;
                continue;
            }
            for y in 0..4 {
                if x != y {
                    let mh = (pi[y] * f[x] / (pi[x] * f[y])).min(1.);
                    kernel[x][y] = stopped[y] * mh;
                }
            }
            kernel[x][x] = 1. - kernel[x].iter().sum::<f64>();
        }
        for x in 0..4 {
            for y in 0..4 {
                assert!((pi[x] * kernel[x][y] - pi[y] * kernel[y][x]).abs() < 1e-14);
            }
        }
        for y in 0..4 {
            assert!(((0..4).map(|x| pi[x] * kernel[x][y]).sum::<f64>() - pi[y]).abs() < 1e-14);
        }
    }
}
