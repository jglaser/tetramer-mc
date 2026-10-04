use anyhow::Result;
use rand::{RngExt, SeedableRng, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use tetramer_mc::{
    bounded_singleton_path::{Budget, Limits, bounded_path},
    depletion::GateOptions,
    depletion_surrogate::DimerDepletionSurrogate,
    flexible_subset::FlexibleSubset,
    flexible_surrogate_chain::{FlexibleSurrogateConfig, FlexibleSurrogateKernel},
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, add, cayley, matmul, norm, quaternion, rotation, sub, transpose},
    singleton_path::{SingletonOrder, SingletonPath},
    spherical::Container,
};

fn pose(x: f64) -> Pose {
    Pose {
        position: [x, 0., 0.],
        orientation: [1., 0., 0., 0.],
    }
}
fn tree(radius: f64) -> SphereTree {
    SphereTree::new(Shape {
        name: "sphere".into(),
        volume: 1.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
    .unwrap()
}
fn cloud() -> Vec<[f64; 3]> {
    let mut points = vec![];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let point = [i, j, k].map(|v| -0.75 + 0.5 * v as f64);
                if norm(point) <= 1. {
                    points.push(point);
                }
            }
        }
    }
    points
}
fn budget() -> Budget {
    Budget::new(Limits {
        raw_per_leg: 1_000_000,
        raw_per_outer: 1_000_000,
        raw_campaign: 1_000_000_000,
        retained_per_leg: 1_000_000,
        retained_per_outer: 1_000_000,
        retained_campaign: 1_000_000_000,
        cpu_seconds: 120.,
    })
    .unwrap()
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
        members: [0, 1],
        rd: 0.8,
        activity: 0.5,
        lambda: 8.,
        envelope: GateOptions {
            max_cells: 31,
            max_depth: 4,
            min_width: 0.,
        },
        body_points: points,
        point_volume: 0.125,
        config: FlexibleSurrogateConfig {
            inner_steps: 8,
            translation_std: 0.3,
            rotation_std_degrees: 15.,
            guidance_strength: 1.,
        },
    }
}
fn rngs(seed: u64) -> [StdRng; 4] {
    std::array::from_fn(|i| StdRng::seed_from_u64(seed + 101 * i as u64))
}
fn step(
    kernel: &FlexibleSurrogateKernel,
    state: &mut [Pose],
    rngs: &mut [StdRng; 4],
    budget: &mut Budget,
    record: &mut Value,
) -> Result<()> {
    let [proposal, inner, bath, accept] = rngs;
    kernel.step(state, proposal, inner, bath, accept, budget, record)
}
fn call(
    kernel: &FlexibleSurrogateKernel,
    state: &mut [Pose],
    seed: u64,
    budget: &mut Budget,
) -> Result<Value> {
    let mut record = Value::Null;
    step(kernel, state, &mut rngs(seed), budget, &mut record)?;
    Ok(record)
}
fn next_words(mut rngs: [StdRng; 4]) -> [u64; 4] {
    std::array::from_fn(|i| rngs[i].random())
}

#[test]
fn zero_horizon_preserves_state_and_all_rngs() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(5., &core)?;
    let points = cloud();
    let mut kernel = kernel(&core, &exclusion, &wall, &points);
    kernel.config.inner_steps = 0;
    let mut state = [pose(1.), pose(2.), pose(0.)];
    let old = state;
    let mut rng = rngs(77);
    let mut budget = budget();
    let mut record = Value::Null;
    step(&kernel, &mut state, &mut rng, &mut budget, &mut record)?;
    assert_eq!(state, old);
    assert_eq!(record["status"], "identity_self_loop");
    assert_eq!(record["physical_decisions"], 0);
    assert_eq!(record["steps"], json!([]));
    assert_eq!(record["inner_counts"]["attempted"], 0);
    assert_eq!(record["budget_before"], record["budget_after"]);
    assert_eq!(next_words(rng), next_words(rngs(77)));
    Ok(())
}

#[test]
fn random_scan_records_every_residence_and_changes_internal_geometry() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(3., &core)?;
    let points = cloud();
    let mut kernel = kernel(&core, &exclusion, &wall, &points);
    // Noncanonical order makes confusing slots with labels observable.
    kernel.members = [1, 0];
    let mut hard = 0;
    let mut rejected = 0;
    let mut accepted = 0;
    let mut physical = [0; 2];
    let mut selected = [0; 2];
    let mut distance_changed = false;
    let mut orientation_changed = false;
    for seed in 200..232 {
        let mut state = [
            Pose {
                orientation: quaternion(cayley([0.2, -0.1, 0.3])),
                ..pose(1.)
            },
            Pose {
                position: [1.8, 0.4, 0.],
                orientation: quaternion(cayley([-0.1, 0.3, 0.2])),
            },
            pose(0.),
        ];
        let old = state;
        let spectators = [old[2]];
        let score = DimerDepletionSurrogate::new(
            &exclusion,
            points.clone(),
            &spectators,
            kernel.point_volume,
            kernel.activity * kernel.config.guidance_strength,
        )?;
        let mut rng = rngs(seed);
        let [mut proposal, mut inner, _, _] = rngs(seed);
        let mut record = Value::Null;
        let mut budget = budget();
        step(&kernel, &mut state, &mut rng, &mut budget, &mut record)?;
        let steps = record["steps"].as_array().unwrap();
        assert_eq!(steps.len(), kernel.config.inner_steps);
        let mut previous = kernel.members.map(|i| old[i]);
        let mut counts = [0; 3];
        for trace in steps {
            let slot = usize::from(proposal.random::<bool>());
            selected[slot] += 1;
            assert_eq!(trace["selected_slot"], slot);
            assert_eq!(trace["selected_label"], kernel.members[slot]);
            assert_eq!(trace["old"], json!(previous));
            // Independently reconstruct the documented Gaussian/Cayley law,
            // including the selector draw before the six proposal variates.
            let displacement = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(&mut proposal);
                kernel.config.translation_std * z
            });
            let cayley_vector = std::array::from_fn(|_| {
                let z: f64 = StandardNormal.sample(&mut proposal);
                kernel.config.rotation_std_degrees.to_radians() / 2. * z
            });
            let mut proposed = previous;
            proposed[slot] = Pose {
                position: add(previous[slot].position, displacement),
                orientation: quaternion(matmul(
                    cayley(cayley_vector),
                    rotation(previous[slot].orientation),
                )),
            };
            assert_eq!(trace["proposed"], json!(proposed));
            assert_eq!(proposed[1 - slot], previous[1 - slot]);
            proposed[slot].validate()?;
            let retained: [Pose; 2] = serde_json::from_value(trace["retained"].clone())?;
            let endpoint = FlexibleSubset::new(&core, &old, &kernel.members, &proposed, kernel.rd)?;
            let old_score = score.score_unpruned(previous)?.log_surrogate;
            assert_eq!(trace["old_score"], json!(old_score));
            if trace["status"] == "hard_rejected" {
                hard += 1;
                counts[0] += 1;
                assert!(!endpoint.hard_valid(Some(&wall), [0.; 3]));
                assert!(trace.get("proposed_score").is_none());
                assert!(trace.get("log_u").is_none());
                assert_eq!(retained, previous);
            } else {
                assert!(endpoint.hard_valid(Some(&wall), [0.; 3]));
                let delta = score.score_unpruned(proposed)?.log_surrogate - old_score;
                assert_eq!(trace["log_acceptance_ratio"], json!(delta));
                let log_u = inner.sample::<f64, _>(Open01).ln();
                assert_eq!(trace["log_u"], json!(log_u));
                let accept = log_u < delta.min(0.);
                assert_eq!(trace["accepted"], accept);
                if accept {
                    accepted += 1;
                    counts[1] += 1;
                    assert_eq!(retained, proposed);
                    distance_changed |= (norm(sub(retained[0].position, retained[1].position))
                        - norm(sub(previous[0].position, previous[1].position)))
                    .abs()
                        > 1e-10;
                    let before = matmul(
                        transpose(rotation(previous[0].orientation)),
                        rotation(previous[1].orientation),
                    );
                    let after = matmul(
                        transpose(rotation(retained[0].orientation)),
                        rotation(retained[1].orientation),
                    );
                    orientation_changed |=
                        (0..3).any(|i| (0..3).any(|j| (before[i][j] - after[i][j]).abs() > 1e-10));
                } else {
                    rejected += 1;
                    counts[2] += 1;
                    assert_eq!(retained, previous);
                }
            }
            assert_eq!(
                trace["retained_score"],
                json!(score.score_unpruned(retained)?.log_surrogate)
            );
            previous = retained;
        }
        assert_eq!(
            record["inner_counts"],
            json!({"attempted":8,
            "hard_rejected":counts[0],"accepted":counts[1],"mh_rejected":counts[2]})
        );
        assert_eq!(record["proposed"], json!(previous));
        assert_eq!(rng[0].random::<u64>(), proposal.random::<u64>());
        assert_eq!(rng[1].random::<u64>(), inner.random::<u64>());
        if record["status"] == "completed" {
            assert_eq!(record["physical_decisions"], 1);
            let correction = record["old_score"]["log_surrogate"].as_f64().unwrap()
                - record["proposed_score"]["log_surrogate"].as_f64().unwrap();
            assert_eq!(record["complete_log_correction"], json!(correction));
            let ratio = record["bath"]["aggregate"]["log_weight"].as_f64().unwrap() + correction;
            assert_eq!(record["log_acceptance_ratio"], json!(ratio));
            let accept = record["log_u"].as_f64().unwrap() < ratio.min(0.);
            assert_eq!(record["accepted"], accept);
            physical[usize::from(accept)] += 1;
            if accept {
                assert_eq!(kernel.members.map(|i| state[i]), previous);
            } else {
                assert_eq!(state, old);
            }
            assert_eq!(record["bath"]["aggregate"]["raw_points"], budget.raw);
            assert_eq!(
                record["bath"]["aggregate"]["retained_points"],
                budget.retained
            );
        }
        assert_eq!(state[2], old[2]);
    }
    assert!(hard > 0 && rejected > 0 && accepted > 0);
    assert!(selected.iter().all(|&n| n > 0));
    assert!(physical.iter().all(|&n| n > 0));
    assert!(distance_changed && orientation_changed);
    Ok(())
}

#[test]
fn isolated_pair_uses_nonzero_internal_bath_and_matches_reference_path() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(5., &core)?;
    let points = cloud();
    let mut kernel = kernel(&core, &exclusion, &wall, &points);
    kernel.config.guidance_strength = 0.;
    let mut nonzero = false;
    let mut orders = std::collections::BTreeSet::new();
    for members in [[0, 1], [1, 0]] {
        kernel.members = members;
        for seed in 80..88 {
            let mut state = [pose(0.), pose(1.)];
            let old = state;
            let record = call(&kernel, &mut state, seed, &mut budget())?;
            assert_eq!(record["status"], "completed");
            let proposed: [Pose; 2] = serde_json::from_value(record["proposed"].clone())?;
            let path = SingletonPath::new(&core, &old, &members, &proposed, kernel.rd)?;
            assert!(path.hard_valid(Some(&wall), [0.; 3]));
            // Public reference path has no resource wrapper; its fair coin,
            // two independent clouds, intermediate and every count must agree.
            let mut rng = StdRng::seed_from_u64(seed + 202);
            let expected =
                path.sample(&mut rng, kernel.lambda, kernel.activity, kernel.envelope)?;
            assert_eq!(record["bath"], json!(expected));
            nonzero |=
                expected.aggregate.retained_points > 0 && expected.aggregate.log_weight != 0.;
            orders.insert(record["bath"]["order"].as_str().unwrap().to_owned());
            assert_eq!(record["complete_log_correction"], 0.);
        }
    }
    assert!(
        nonzero,
        "isolated flexible pair must not get the rigid zero-spectator cancellation"
    );
    assert_eq!(orders.len(), 2);
    Ok(())
}

#[test]
fn identity_endpoint_skips_bath_but_counts_all_inner_attempts() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(5., &core)?;
    let points = cloud();
    let mut kernel = kernel(&core, &exclusion, &wall, &points);
    kernel.config.translation_std = 0.;
    kernel.config.rotation_std_degrees = 0.;
    let mut state = [pose(0.), pose(1.)];
    let old = state;
    let mut rng = rngs(1);
    let controls = rngs(1);
    let mut record = Value::Null;
    let mut budget = budget();
    step(&kernel, &mut state, &mut rng, &mut budget, &mut record)?;
    assert_eq!(state, old);
    assert_eq!(record["status"], "identity_self_loop");
    assert_eq!(record["inner_counts"]["accepted"], 8);
    assert_eq!(record["steps"].as_array().unwrap().len(), 8);
    assert_eq!(record["physical_decisions"], 0);
    assert!(record.get("bath").is_none());
    let actual = next_words(rng);
    let control = next_words(controls);
    assert_eq!(&actual[2..], &control[2..]);
    assert_eq!(budget.raw, 0);
    Ok(())
}

#[test]
fn fatal_bath_preserves_full_inner_trace_state_and_outer_coin() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(5., &core)?;
    let points = cloud();
    let mut kernel = kernel(&core, &exclusion, &wall, &points);
    kernel.config.guidance_strength = 0.;
    let mut state = [pose(0.), pose(1.)];
    let old = state;
    let mut budget = budget();
    budget.limits.raw_per_leg = 0;
    budget.limits.raw_per_outer = 0;
    budget.limits.raw_campaign = 0;
    let mut rng = rngs(41);
    let mut controls = rngs(41);
    let mut record = Value::Null;
    assert!(step(&kernel, &mut state, &mut rng, &mut budget, &mut record).is_err());
    assert_eq!(state, old);
    assert_eq!(record["status"], "fatal");
    assert_eq!(record["steps"].as_array().unwrap().len(), 8);
    assert_eq!(record["inner_counts"]["attempted"], 8);
    assert_eq!(record["physical_decisions"], 0);
    assert!(
        record["bath_failure"]["failed_progress"]["gate"]["raw_points"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        record["bath_failure"]["failed_progress"]["processed_points"],
        0
    );
    assert!(record["bath_failure"]["order"].is_string());
    assert_eq!(record["bath_failure"]["failed_leg"], 0);
    assert_eq!(record["bath_failure"]["completed_legs"], json!([]));
    assert_eq!(rng[3].random::<u64>(), controls[3].random::<u64>());
    assert_eq!(budget.raw, 0);
    assert_eq!(record["budget_before"], record["budget_after"]);
    Ok(())
}

#[test]
fn bounded_path_allows_core_invalid_copied_intermediates() -> Result<()> {
    let core = tree(0.2);
    let wall = Container::new(3., &core)?;
    let old = [pose(0.), pose(1.)];
    let proposed = [old[1], old[0]];
    let path = SingletonPath::new(&core, &old, &[0, 1], &proposed, 0.8)?;
    assert!(path.hard_valid(Some(&wall), [0.; 3]));
    let opts = GateOptions {
        max_cells: 31,
        max_depth: 4,
        min_width: 0.,
    };
    for order in [
        SingletonOrder::FirstThenSecond,
        SingletonOrder::SecondThenFirst,
    ] {
        let middle = path.intermediate_selected(order);
        assert!(
            !FlexibleSubset::new(&core, &old, &[0, 1], &middle, 0.8)?
                .hard_valid(Some(&wall), [0.; 3])
        );
        let mut reference_rng = StdRng::seed_from_u64(808213);
        let expected = path.sample_with_order(&mut reference_rng, 20., 0.5, opts, order)?;
        let mut rng = StdRng::seed_from_u64(808213);
        let mut budget = budget();
        let result = bounded_path(
            &core,
            &old,
            [0, 1],
            proposed,
            &path,
            &mut rng,
            0.8,
            20.,
            0.5,
            opts,
            &mut budget,
            Some(order),
        )?;
        assert_eq!(json!(result), json!(expected));
        assert_eq!(rng.random::<u64>(), reference_rng.random::<u64>());
        assert!(
            result
                .legs
                .iter()
                .all(|leg| leg.raw_points > 0 && leg.retained_points > 0)
        );
        assert_eq!(budget.raw, result.aggregate.raw_points);
        assert_eq!(budget.retained, result.aggregate.retained_points);
    }
    Ok(())
}

#[test]
fn later_bounded_leg_failure_keeps_completed_nonzero_leg_and_budget() -> Result<()> {
    let core = tree(0.2);
    let old = [pose(0.), pose(1.)];
    let proposed = [old[1], old[0]];
    let path = SingletonPath::new(&core, &old, &[0, 1], &proposed, 0.8)?;
    let opts = GateOptions {
        max_cells: 31,
        max_depth: 4,
        min_width: 0.,
    };
    let order = SingletonOrder::FirstThenSecond;
    let mut reference_rng = StdRng::seed_from_u64(808214);
    let expected = path.sample_with_order(&mut reference_rng, 20., 0.5, opts, order)?;
    assert!(expected.legs[0].raw_points > 0 && expected.legs[0].retained_points > 0);
    assert!(expected.legs[0].log_weight != 0. && expected.legs[1].raw_points > 0);
    let mut budget = budget();
    budget.raw = 17;
    budget.retained = 9;
    // An exact first-leg allowance forces failure on the second planned count,
    // after a genuinely nonzero cloud has completed and entered the budget.
    budget.limits.raw_per_outer = expected.legs[0].raw_points;
    let mut rng = StdRng::seed_from_u64(808214);
    let error = bounded_path(
        &core,
        &old,
        [0, 1],
        proposed,
        &path,
        &mut rng,
        0.8,
        20.,
        0.5,
        opts,
        &mut budget,
        Some(order),
    )
    .unwrap_err();
    assert_eq!(error.failed_leg, 1);
    assert_eq!(error.order, order);
    assert_eq!(
        error.intermediate_selected,
        path.intermediate_selected(order)
    );
    assert_eq!(json!(error.completed_legs), json!([expected.legs[0]]));
    assert!(error.reason.contains("planned raw-point"));
    assert!(!error.failed_progress.complete);
    assert_eq!(error.failed_progress.processed_points, 0);
    assert_eq!(
        error.failed_progress.gate.raw_points,
        expected.legs[1].raw_points
    );
    assert_eq!(budget.raw, 17 + expected.legs[0].raw_points);
    assert_eq!(budget.retained, 9 + expected.legs[0].retained_points);
    Ok(())
}

#[test]
fn saved_boundary_and_rng_states_continue_identically() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(5., &core)?;
    let points = cloud();
    let kernel = kernel(&core, &exclusion, &wall, &points);
    let mut state = vec![pose(1.), pose(2.), pose(0.)];
    let mut rng = rngs(10000);
    let mut budget = budget();
    // StdRng deliberately has no Clone/serialization in rand 0.10. Replaying
    // the same fixed prefix reconstructs each live RNG state independently.
    let mut continued_rng = rngs(10000);
    let mut replay_state = state.clone();
    let mut replay_budget = Budget::new(budget.limits)?;
    for _ in 0..5 {
        let mut a = Value::Null;
        let mut b = Value::Null;
        step(&kernel, &mut state, &mut rng, &mut budget, &mut a)?;
        step(
            &kernel,
            &mut replay_state,
            &mut continued_rng,
            &mut replay_budget,
            &mut b,
        )?;
        assert_eq!(a, b);
    }
    // The physical boundary/counters roundtrip through the checkpoint format;
    // reconstructed live RNG states continue without reseeding at each call.
    let cp: Value = serde_json::from_slice(&serde_json::to_vec(&json!({
        "poses":state,"raw":budget.raw,"retained":budget.retained,
    }))?)?;
    let mut continued: Vec<Pose> = serde_json::from_value(cp["poses"].clone())?;
    assert_eq!(continued, replay_state);
    let mut continued_budget = Budget::new(budget.limits)?;
    continued_budget.raw = cp["raw"].as_u64().unwrap();
    continued_budget.retained = cp["retained"].as_u64().unwrap();
    for _ in 0..8 {
        let mut a = Value::Null;
        let mut b = Value::Null;
        step(&kernel, &mut state, &mut rng, &mut budget, &mut a)?;
        step(
            &kernel,
            &mut continued,
            &mut continued_rng,
            &mut continued_budget,
            &mut b,
        )?;
        assert_eq!(a, b);
        assert_eq!(state, continued);
    }
    assert_eq!(budget.raw, continued_budget.raw);
    assert_eq!(budget.retained, continued_budget.retained);
    assert_eq!(next_words(rng), next_words(continued_rng));
    Ok(())
}

#[test]
fn invalid_inputs_or_exhausted_budget_fail_before_any_draw() -> Result<()> {
    let (core, exclusion) = (tree(0.2), tree(1.));
    let wall = Container::new(5., &core)?;
    let points = cloud();
    for bad in 0..10 {
        let mut kernel = kernel(&core, &exclusion, &wall, &points);
        let mut state = [pose(1.), pose(2.), pose(0.)];
        let mut budget = budget();
        match bad {
            0 => kernel.config.inner_steps = 1025,
            1 => kernel.rd = 0.7,
            2 => kernel.members = [1, 1],
            3 => kernel.members = [0, 3],
            4 => state[0] = state[2],
            5 => state[0] = pose(8.),
            6 => kernel.config.guidance_strength = f64::INFINITY,
            7 => kernel.lambda = 0.,
            8 => budget.raw = budget.limits.raw_campaign + 1,
            _ => budget.started -= budget.limits.cpu_seconds + 1.,
        }
        let old = state;
        let mut rng = rngs(7);
        let mut record = Value::Null;
        assert!(step(&kernel, &mut state, &mut rng, &mut budget, &mut record).is_err());
        assert_eq!(state, old);
        assert_eq!(record["status"], "fatal");
        assert_eq!(record["physical_decisions"], 0);
        assert_eq!(next_words(rng), next_words(rngs(7)));
    }
    Ok(())
}
