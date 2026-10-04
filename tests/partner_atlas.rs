//! Deterministic toy checks only: partner-conditional atlas proposals, full
//! helper correction, unchanged default RNG law, and one physical endpoint gate.
use anyhow::Result;
use rand::{RngExt, SeedableRng, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use tetramer_mc::{
    bounded_singleton_path::{Budget, Limits},
    depletion::GateOptions,
    depletion_surrogate::DimerDepletionSurrogate,
    docking::{DockingMethod, DockingProposal, MemberLabel},
    flexible_subset::FlexibleSubset,
    flexible_surrogate_chain::{FlexibleSurrogateConfig, FlexibleSurrogateKernel},
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, add, cayley, matmul, norm, quaternion, rotation, sub},
    proposal::FrozenRelativePoseProposal,
    spherical::Container,
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
fn pose(x: f64) -> Pose {
    Pose {
        position: [x, 0., 0.],
        orientation: [1., 0., 0., 0.],
    }
}
// Reconstruct the documented Gaussian/Cayley law independently; the runtime
// helper stays private, so this also checks the candidate's actual draw order.
fn local_pose(rng: &mut StdRng, old: Pose, translation: f64, rotation_half: f64) -> Pose {
    let displacement = std::array::from_fn(|_| {
        let z: f64 = StandardNormal.sample(rng);
        translation * z
    });
    let increment = std::array::from_fn(|_| {
        let z: f64 = StandardNormal.sample(rng);
        rotation_half * z
    });
    Pose {
        position: add(old.position, displacement),
        orientation: quaternion(matmul(cayley(increment), rotation(old.orientation))),
    }
}
fn tree(radius: f64) -> SphereTree {
    SphereTree::new(Shape {
        name: "toy".into(),
        volume: 1.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
    .unwrap()
}
fn points() -> Vec<[f64; 3]> {
    (0..8)
        .map(|i| std::array::from_fn(|j| if i & (1 << j) == 0 { -0.5 } else { 0.5 }))
        .collect()
}
fn model(cube: f64, uniform: f64, seam_chart: bool) -> Result<FrozenRelativePoseProposal> {
    let raw = if seam_chart {
        let covariance: [[f64; 6]; 6] =
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 0.25 } else { 0. }));
        json!({"coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
            "angular_length":1.3,"weights":[1.],
            "anchors":[{"position":[1.,0.,0.],"rotation":cayley([0.;3])}],
            "means":vec![[0.;6]],"covariances":vec![covariance]})
    } else {
        let mut lower = [[0.; 6]; 6];
        for (i, row) in lower.iter_mut().enumerate() {
            row[i] = 0.4 + i as f64 * 0.05;
        }
        lower[3][0] = 0.15;
        lower[5][1] = -0.12;
        let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
        });
        let base = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":SHA,
            "angular_length":1.3,"weights":[0.3,0.7],
            "anchors":[{"position":[2.4,-0.5,0.3],"rotation":cayley([0.3,-0.1,0.4])},
                {"position":[-0.6,2.2,1.0],"rotation":cayley([-0.2,0.3,0.1])}],
            "means":vec![[0.05,0.03,-0.04,0.1,-0.05,0.04];2],
            "covariances":[covariance,covariance]});
        json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,
            "reciprocal_components":[true,false]})
    };
    FrozenRelativePoseProposal::from_json_str_open(&raw.to_string(), [cube; 3], uniform, SHA)
}
fn atlas(cube: f64, uniform: f64, seam_chart: bool) -> Result<DockingProposal> {
    DockingProposal::new(
        model(cube, uniform, seam_chart)?,
        DockingMethod::PosteriorInvolution,
        0.6,
        [0.; 3],
    )
}
fn kernel<'a>(
    core: &'a SphereTree,
    exclusion: &'a SphereTree,
    wall: &'a Container,
    points: &'a [[f64; 3]],
) -> FlexibleSurrogateKernel<'a> {
    FlexibleSurrogateKernel {
        core,
        exclusion,
        wall: Some(wall),
        wall_center: [0.; 3],
        members: [2, 0],
        rd: 0.8,
        activity: 0.3,
        lambda: 2.,
        envelope: GateOptions {
            max_cells: 31,
            max_depth: 4,
            min_width: 0.,
        },
        body_points: points,
        point_volume: 1.,
        config: FlexibleSurrogateConfig {
            inner_steps: 8,
            translation_std: 0.3,
            rotation_std_degrees: 12.,
            guidance_strength: 1.,
        },
    }
}
fn initial() -> [Pose; 3] {
    [
        Pose {
            orientation: quaternion(cayley([0.1, -0.2, 0.05])),
            ..pose(1.)
        },
        pose(-5.),
        Pose {
            orientation: quaternion(cayley([-0.1, 0.05, 0.2])),
            ..pose(2.)
        },
    ]
}
fn rngs(seed: u64) -> [StdRng; 4] {
    std::array::from_fn(|i| StdRng::seed_from_u64(seed + 101 * i as u64))
}
fn words(mut rngs: [StdRng; 4]) -> [u64; 4] {
    std::array::from_fn(|i| rngs[i].random())
}
fn budget() -> Budget {
    Budget::new(Limits {
        raw_per_leg: 100_000,
        raw_per_outer: 200_000,
        raw_campaign: 1_000_000,
        retained_per_leg: 100_000,
        retained_per_outer: 200_000,
        retained_campaign: 1_000_000,
        cpu_seconds: 120.,
    })
    .unwrap()
}
fn call(
    k: &FlexibleSurrogateKernel,
    atlas: &DockingProposal,
    direct: bool,
    state: &mut [Pose],
    rngs: &mut [StdRng; 4],
    budget: &mut Budget,
    record: &mut Value,
) -> Result<()> {
    let [proposal, inner, bath, accept] = rngs;
    if direct {
        k.step_partner_atlas_direct(atlas, state, proposal, inner, bath, accept, budget, record)
    } else {
        k.step_partner_atlas(atlas, state, proposal, inner, bath, accept, budget, record)
    }
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-9 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
fn same_pose(a: Pose, b: Pose) {
    assert!(norm(sub(a.position, b.position)) < 1e-8);
    close(
        a.orientation
            .iter()
            .zip(b.orientation)
            .map(|(x, y)| x * y)
            .sum::<f64>()
            .abs(),
        1.,
    );
}

#[test]
fn current_partner_trace_reverse_density_and_inner_outer_corrections() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(8., &core)?;
    let points = points();
    let k = kernel(&core, &exclusion, &wall, &points);
    let atlas = atlas(12., 0.1, false)?;
    let mut selections = [0; 2];
    let mut modes = [0; 2];
    let mut learned = 0;
    let mut asymmetric = 0;
    let mut changed_partner = false;
    let mut hard = 0;
    let mut mh = 0;
    let mut physical = 0;
    for seed in 0..32 {
        let old = initial();
        let mut state = old;
        let mut streams = rngs(seed);
        let [mut proposal, mut inner, _, _] = rngs(seed);
        let mut record = Value::Null;
        call(
            &k,
            &atlas,
            false,
            &mut state,
            &mut streams,
            &mut budget(),
            &mut record,
        )?;
        let scorer = DimerDepletionSurrogate::new(&exclusion, points.clone(), &[old[1]], 1., 0.3)?;
        let mut current = k.members.map(|i| old[i]);
        let steps = record["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 8);
        let mut accepted_count = 0;
        for trace in steps {
            let slot = usize::from(proposal.random::<bool>());
            selections[slot] += 1;
            assert_eq!(trace["selected_slot"], slot);
            assert_eq!(trace["selected_label"], k.members[slot]);
            assert_eq!(trace["old"], json!(current));
            let mut target = current;
            let mut correction = 0.;
            if proposal.random::<f64>() < 0.25 {
                modes[1] += 1;
                assert_eq!(trace["mode"], "partner_atlas");
                assert_eq!(trace["partner_pose"], json!(current[1 - slot]));
                changed_partner |= current[1 - slot] != old[k.members[1 - slot]];
                let (candidate, info) = atlas.propose_members(
                    &mut proposal,
                    &[current[slot]],
                    0,
                    &[current[1 - slot]],
                )?;
                assert_eq!(trace["proposal_trace"], info);
                if let Some(candidate) = candidate {
                    target[slot] = candidate;
                    correction = info["log_reverse_forward"].as_f64().unwrap();
                    if info["branch"] == "involution" {
                        learned += 1;
                        asymmetric += usize::from(correction.abs() > 1e-6);
                        close(
                            correction,
                            atlas.members_log_density(&[current[slot]], &[current[1 - slot]])?
                                - atlas.members_log_density(&[candidate], &[current[1 - slot]])?,
                        );
                        close(
                            correction,
                            info["expanded_log_reverse_forward"].as_f64().unwrap(),
                        );
                        close(
                            correction,
                            info["step"]["log_correction"].as_f64().unwrap()
                                + info["label_log_reverse_forward"].as_f64().unwrap(),
                        );
                        let source: MemberLabel =
                            serde_json::from_value(info["labels"]["source"].clone())?;
                        let destination: MemberLabel =
                            serde_json::from_value(info["labels"]["target"].clone())?;
                        let noise: [f64; 6] =
                            serde_json::from_value(info["step"]["inverse_trace"]["noise"].clone())?;
                        let reverse = atlas.apply_member_trace(
                            &[candidate],
                            0,
                            &[current[1 - slot]],
                            destination,
                            source,
                            noise,
                        )?;
                        same_pose(reverse.handle, current[slot]);
                        close(reverse.log_reverse_forward, -correction);
                    }
                }
            } else {
                modes[0] += 1;
                assert_eq!(trace["mode"], "local");
                target[slot] =
                    local_pose(&mut proposal, current[slot], 0.3, 12_f64.to_radians() / 2.);
            }
            assert_eq!(trace["proposed"], json!(target));
            assert_eq!(target[1 - slot], current[1 - slot]);
            let old_score = scorer.score_unpruned(current)?.log_surrogate;
            close(trace["old_score"].as_f64().unwrap(), old_score);
            match trace["status"].as_str().unwrap() {
                "null_proposal" | "zero_reverse_support" => assert!(trace.get("log_u").is_none()),
                "hard_rejected" => {
                    hard += 1;
                    assert!(
                        !FlexibleSubset::new(&core, &old, &k.members, &target, 0.8)?
                            .hard_valid(Some(&wall), [0.; 3])
                    );
                    assert!(trace.get("log_u").is_none());
                }
                "completed" => {
                    let delta =
                        scorer.score_unpruned(target)?.log_surrogate - old_score + correction;
                    close(trace["log_acceptance_ratio"].as_f64().unwrap(), delta);
                    let log_u = inner.sample::<f64, _>(Open01).ln();
                    assert_eq!(trace["log_u"], json!(log_u));
                    assert_eq!(trace["accepted"], log_u < delta.min(0.));
                    if log_u < delta.min(0.) {
                        current = target;
                        accepted_count += 1;
                    } else {
                        mh += 1;
                    }
                }
                status => panic!("unexpected status {status}"),
            }
            assert_eq!(trace["retained"], json!(current));
        }
        assert_eq!(record["inner_counts"]["attempted"], 8);
        assert_eq!(record["inner_counts"]["accepted"], accepted_count);
        assert_eq!(streams[0].random::<u64>(), proposal.random::<u64>());
        assert_eq!(streams[1].random::<u64>(), inner.random::<u64>());
        close(
            record["complete_log_correction"].as_f64().unwrap(),
            scorer
                .score_unpruned(k.members.map(|i| old[i]))?
                .log_surrogate
                - scorer.score_unpruned(current)?.log_surrogate,
        );
        assert_eq!(state[1], old[1]);
        if record["physical_decisions"] == 1 {
            physical += 1;
            let ratio = record["bath"]["aggregate"]["log_weight"].as_f64().unwrap()
                + record["complete_log_correction"].as_f64().unwrap();
            close(record["log_acceptance_ratio"].as_f64().unwrap(), ratio);
            assert_eq!(
                record["accepted"],
                record["log_u"].as_f64().unwrap() < ratio.min(0.)
            );
            assert_eq!(
                k.members.map(|i| state[i]),
                if record["accepted"] == true {
                    current
                } else {
                    k.members.map(|i| old[i])
                }
            );
        }
    }
    assert!(selections.iter().all(|n| *n > 0) && modes.iter().all(|n| *n > 0));
    assert!(learned > 5 && asymmetric > 5 && changed_partner && hard > 0 && mh > 0 && physical > 0);
    Ok(())
}

#[test]
fn exterior_uniform_source_is_explicit_null_without_retry_or_bath() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(8., &core)?;
    let points = points();
    let mut k = kernel(&core, &exclusion, &wall, &points);
    k.config.inner_steps = 32;
    k.config.translation_std = 0.;
    k.config.rotation_std_degrees = 0.;
    let atlas = atlas(0.4, 1. - 1e-12, false)?;
    let mut state = [pose(1.), pose(-5.), pose(2.)];
    let old = state;
    let mut rng = rngs(71);
    let expected = words(rngs(71));
    let mut record = Value::Null;
    call(
        &k,
        &atlas,
        false,
        &mut state,
        &mut rng,
        &mut budget(),
        &mut record,
    )?;
    assert_eq!(state, old);
    assert_eq!(record["status"], "identity_self_loop");
    assert_eq!(record["steps"].as_array().unwrap().len(), 32);
    let mut rejected = 0;
    for trace in record["steps"].as_array().unwrap() {
        if trace["mode"] == "partner_atlas" {
            rejected += 1;
            assert_eq!(trace["status"], "zero_reverse_support");
            assert_eq!(trace["uniform_source_support"], false);
            assert_eq!(trace["uniform_candidate_support"], true);
            assert!(trace.get("log_u").is_none());
            assert_eq!(trace["retained"], trace["old"]);
        }
    }
    assert!(rejected > 0);
    assert_eq!(record["inner_counts"]["zero_reverse_support"], rejected);
    assert_eq!(record["inner_counts"]["attempted"], 32);
    assert_eq!(record["physical_decisions"], 0);
    assert_eq!(&words(rng)[2..], &expected[2..]);
    assert!(atlas.member_uniform_contains(pose(0.2))?);
    assert!(!atlas.member_uniform_contains(pose(0.200001))?);
    Ok(())
}

#[test]
fn learned_seam_null_consumes_horizon_without_inner_coin() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(8., &core)?;
    let points = points();
    let mut k = kernel(&core, &exclusion, &wall, &points);
    k.config.inner_steps = 32;
    k.config.translation_std = 0.;
    k.config.rotation_std_degrees = 0.;
    let atlas = atlas(12., 1e-100, true)?;
    let mut state = [
        pose(1.),
        pose(-5.),
        Pose {
            orientation: [0., 1., 0., 0.],
            ..pose(2.)
        },
    ];
    let old = state;
    let mut streams = rngs(29);
    let [mut proposal, mut inner, _, _] = rngs(29);
    let mut record = Value::Null;
    call(
        &k,
        &atlas,
        false,
        &mut state,
        &mut streams,
        &mut budget(),
        &mut record,
    )?;
    let mut nulls = 0;
    for trace in record["steps"].as_array().unwrap() {
        let slot = usize::from(proposal.random::<bool>());
        if proposal.random::<f64>() < 0.25 {
            let (candidate, _) = atlas.propose_members(
                &mut proposal,
                &[old[k.members[slot]]],
                0,
                &[old[k.members[1 - slot]]],
            )?;
            assert!(candidate.is_none());
            nulls += 1;
            assert_eq!(trace["status"], "null_proposal");
            assert!(trace.get("log_u").is_none());
        } else {
            local_pose(&mut proposal, old[k.members[slot]], 0., 0.);
            inner.sample::<f64, _>(Open01);
        }
    }
    assert!(nulls > 0);
    assert_eq!(record["inner_counts"]["null_proposals"], nulls);
    assert_eq!(record["inner_counts"]["attempted"], 32);
    assert_eq!(state, old);
    assert_eq!(record["physical_decisions"], 0);
    assert_eq!(streams[0].random::<u64>(), proposal.random::<u64>());
    assert_eq!(streams[1].random::<u64>(), inner.random::<u64>());
    Ok(())
}

#[test]
fn direct_candidate_uses_q_once_and_no_surrogate_or_inner_rng() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(8., &core)?;
    // Deliberately invalid score quadrature: direct control must never evaluate it.
    let bad_points = [[f64::NAN, 0., 0.]];
    let mut k = kernel(&core, &exclusion, &wall, &bad_points);
    k.config.inner_steps = 1;
    k.point_volume = f64::NAN;
    let atlas = atlas(12., 0.1, false)?;
    let mut physical = 0;
    let mut learned = 0;
    for seed in 0..64 {
        let old = initial();
        let mut state = old;
        let mut streams = rngs(seed);
        let expected = words(rngs(seed));
        let mut record = Value::Null;
        call(
            &k,
            &atlas,
            true,
            &mut state,
            &mut streams,
            &mut budget(),
            &mut record,
        )?;
        assert_eq!(record["kind"], "flexible_partner_atlas_direct");
        assert_eq!(record["inner_filter"], false);
        assert!(record.get("old_score").is_none());
        assert!(record.get("proposed_score").is_none());
        assert!(record.get("inner_counts").is_none());
        assert_eq!(record["steps"].as_array().unwrap().len(), 1);
        assert_eq!(streams[1].random::<u64>(), expected[1]);
        let trace = &record["steps"][0];
        assert!(trace.get("log_u").is_none());
        assert!(trace.get("proposed_score").is_none());
        if record["physical_decisions"] == 1 {
            physical += 1;
            assert_eq!(trace["status"], "direct_candidate");
            assert!(trace["accepted"].is_null());
            let q = trace["proposal_log_reverse_forward"].as_f64().unwrap();
            close(record["complete_log_correction"].as_f64().unwrap(), q);
            close(
                record["log_acceptance_ratio"].as_f64().unwrap(),
                q + record["bath"]["aggregate"]["log_weight"].as_f64().unwrap(),
            );
            if trace["proposal_trace"]["branch"] == "involution" {
                learned += 1;
            }
            let candidate: [Pose; 2] = serde_json::from_value(record["proposed"].clone())?;
            let slot = trace["selected_slot"].as_u64().unwrap() as usize;
            assert_eq!(candidate[1 - slot], old[k.members[1 - slot]]);
            assert_eq!(
                k.members.map(|i| state[i]),
                if record["accepted"] == true {
                    candidate
                } else {
                    k.members.map(|i| old[i])
                }
            );
        }
        assert_eq!(state[1], old[1]);
    }
    assert!(physical > 10 && learned > 0);
    Ok(())
}

#[test]
fn fatal_validation_and_bath_errors_are_atomic_and_preserve_trace() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(8., &core)?;
    let points = points();
    let mut k = kernel(&core, &exclusion, &wall, &points);
    let atlas = atlas(12., 0.1, false)?;
    let mut state = initial();
    let old = state;
    let mut streams = rngs(17);
    let mut record = Value::Null;
    assert!(
        call(
            &k,
            &atlas,
            true,
            &mut state,
            &mut streams,
            &mut budget(),
            &mut record
        )
        .is_err()
    );
    assert_eq!(record["status"], "fatal");
    assert_eq!(state, old);
    assert_eq!(words(streams), words(rngs(17)));
    let invalid = DockingProposal::new(
        model(12., 0.1, false)?,
        DockingMethod::Mixture,
        0.6,
        [0.; 3],
    )?;
    let mut streams = rngs(18);
    assert!(
        call(
            &k,
            &invalid,
            false,
            &mut state,
            &mut streams,
            &mut budget(),
            &mut record
        )
        .is_err()
    );
    assert_eq!(words(streams), words(rngs(18)));
    assert_eq!(state, old);
    k.config.inner_steps = 1;
    k.lambda = 1000.;
    let mut failures = 0;
    for seed in 0..16 {
        let mut state = old;
        let mut streams = rngs(seed);
        let expected = words(rngs(seed));
        let mut budget = budget();
        budget.limits.raw_per_leg = 0;
        budget.limits.raw_per_outer = 0;
        budget.limits.raw_campaign = 0;
        let result = call(
            &k,
            &atlas,
            true,
            &mut state,
            &mut streams,
            &mut budget,
            &mut record,
        );
        if result.is_err() {
            failures += 1;
            assert_eq!(record["status"], "fatal");
            assert_eq!(state, old);
            assert_eq!(record["steps"].as_array().unwrap().len(), 1);
            assert_eq!(record["steps"][0]["status"], "direct_candidate");
            assert!(record["bath_failure"].is_object());
            assert_eq!(record["physical_decisions"], 0);
            assert_eq!(streams[3].random::<u64>(), expected[3]);
        }
    }
    assert!(failures > 0);
    Ok(())
}

#[test]
fn one_step_delayed_gate_cannot_dominate_the_same_direct_proposal() {
    let mut strict = false;
    for score_delta in [-5_f64, -0.2, 0., 0.7, 4.] {
        for proposal_ratio in [-3_f64, 0., 2.] {
            for bath in [-2_f64, 0., 4.] {
                let delayed = (score_delta + proposal_ratio).min(0.).exp()
                    * (bath - score_delta).min(0.).exp();
                let direct = (bath + proposal_ratio).min(0.).exp();
                assert!(delayed <= direct + 1e-15);
                strict |= delayed < direct - 0.1;
            }
        }
    }
    assert!(strict);
    // Flat m1 still filters an asymmetric proposal before a compensating bath.
    assert!((-3_f64).exp() < 1.);
    assert_eq!((4_f64 - 3.).min(0.).exp(), 1.);
}
