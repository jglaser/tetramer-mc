use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use tetramer_mc::{
    bounded_singleton_path::{Budget, Limits, bounded_singleton},
    depletion::GateOptions,
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, cayley, matmul, norm, quaternion, rotation, sub, transpose},
    rigid_subset::{RigidSubset, transport_members},
    rigid_surrogate_chain::{RigidSurrogateConfig, RigidSurrogateKernel},
    simulation::cpu_seconds,
    spherical::Container,
};
fn pose(x: f64) -> Pose {
    Pose {
        position: [x, 0., 0.],
        orientation: [1., 0., 0., 0.],
    }
}
fn tree(r: f64) -> SphereTree {
    SphereTree::new(Shape {
        name: "sphere".into(),
        volume: 1.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: r,
        }],
    })
    .unwrap()
}
fn cloud() -> Vec<[f64; 3]> {
    let mut p = vec![];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let x = [i, j, k].map(|v| -0.75 + 0.5 * v as f64);
                if norm(x) <= 1. {
                    p.push(x);
                }
            }
        }
    }
    p
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
) -> RigidSurrogateKernel<'a> {
    RigidSurrogateKernel {
        core,
        exclusion,
        wall: Some(wall),
        wall_center: [0.; 3],
        members: [0, 1],
        handle: 0,
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
        config: RigidSurrogateConfig {
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
fn call(k: &RigidSurrogateKernel, state: &mut [Pose], seed: u64, b: &mut Budget) -> Result<Value> {
    let [mut p, mut i, mut bath, mut accept] = rngs(seed);
    let mut trace = Value::Null;
    k.step(state, &mut p, &mut i, &mut bath, &mut accept, b, &mut trace)?;
    Ok(trace)
}
fn same_pose(a: Pose, b: Pose) {
    assert!(norm(sub(a.position, b.position)) < 2e-12);
    let a = rotation(a.orientation);
    let b = rotation(b.orientation);
    for i in 0..3 {
        for j in 0..3 {
            assert!((a[i][j] - b[i][j]).abs() < 2e-12);
        }
    }
}

#[test]
fn zero_horizon_preserves_state_and_all_rngs() -> Result<()> {
    let (c, e) = (tree(0.2), tree(1.));
    let w = Container::new(5., &c)?;
    let points = cloud();
    let mut k = kernel(&c, &e, &w, &points);
    k.config.inner_steps = 0;
    let mut state = [pose(1.), pose(2.), pose(0.)];
    let old = state;
    let [mut p, mut i, mut b, mut a] = rngs(77);
    let mut controls = rngs(77);
    let mut record = Value::Null;
    let mut limits = budget();
    k.step(
        &mut state,
        &mut p,
        &mut i,
        &mut b,
        &mut a,
        &mut limits,
        &mut record,
    )?;
    assert_eq!(state, old);
    assert_eq!(record["status"], "identity_self_loop");
    assert_eq!(record["physical_decisions"], 0);
    assert_eq!(limits.raw, 0);
    assert_eq!(record["steps"], json!([]));
    for (mut actual, control) in [p, i, b, a].into_iter().zip(&mut controls) {
        assert_eq!(actual.random::<u64>(), control.random::<u64>());
    }
    Ok(())
}

#[test]
fn every_inner_residence_and_one_outer_correction_are_retained() -> Result<()> {
    let (c, e) = (tree(0.2), tree(1.));
    let w = Container::new(3., &c)?;
    let points = cloud();
    let k = kernel(&c, &e, &w, &points);
    let mut hard = 0;
    let mut rejected = 0;
    let mut physical_rejected = 0;
    let mut physical_accepted = 0;
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
        let record = call(&k, &mut state, seed, &mut budget())?;
        let steps = record["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 8);
        let mut previous = [old[0], old[1]];
        for step in steps {
            let retained: [Pose; 2] = serde_json::from_value(step["retained"].clone())?;
            assert_eq!(step["old_handle"], json!(previous[0]));
            if step["status"] == "hard_rejected" {
                hard += 1;
                assert!(step.get("proposed_score").is_none());
                assert_eq!(retained, previous);
            } else {
                let delta = step["proposed_score"]["log_surrogate"].as_f64().unwrap()
                    - step["old_score"].as_f64().unwrap();
                assert_eq!(step["log_acceptance_ratio"], json!(delta));
                let accepted = step["log_u"].as_f64().unwrap() < delta.min(0.);
                assert_eq!(step["accepted"], accepted);
                if !accepted {
                    rejected += 1;
                    assert_eq!(retained, previous);
                }
            }
            let carried = transport_members(&old, &[0, 1], 0, retained[0])?;
            assert_eq!(retained.as_slice(), carried);
            assert!(
                (norm(sub(retained[0].position, retained[1].position))
                    - norm(sub(old[0].position, old[1].position)))
                .abs()
                    < 2e-12
            );
            previous = retained;
        }
        assert_eq!(record["proposed"], json!(previous));
        if record["status"] == "completed" {
            assert_eq!(record["physical_decisions"], 1);
            let correction = record["old_score"]["log_surrogate"].as_f64().unwrap()
                - record["proposed_score"]["log_surrogate"].as_f64().unwrap();
            assert_eq!(record["complete_log_correction"], json!(correction));
            let logratio = record["bath"]["log_weight"].as_f64().unwrap() + correction;
            assert_eq!(record["log_acceptance_ratio"], json!(logratio));
            let accept = record["log_u"].as_f64().unwrap() < logratio.min(0.);
            assert_eq!(record["accepted"], accept);
            if accept {
                physical_accepted += 1;
                assert_eq!([state[0], state[1]], previous);
            } else {
                physical_rejected += 1;
                assert_eq!(state, old);
            }
        }
        assert_eq!(state[2], old[2]);
    }
    assert!(hard > 0 && rejected > 0 && physical_rejected > 0 && physical_accepted > 0);
    Ok(())
}

#[test]
fn outer_gate_matches_independent_rigid_subset_reconstruction_for_both_handles() -> Result<()> {
    let (c, e) = (tree(0.2), tree(1.));
    let w = Container::new(5., &c)?;
    let points = cloud();
    for (members, handle) in [([0, 1], 0), ([0, 1], 1), ([1, 0], 0), ([1, 0], 1)] {
        let mut k = kernel(&c, &e, &w, &points);
        k.members = members;
        k.handle = handle;
        let mut state = [pose(1.), pose(2.), pose(0.)];
        let old = state;
        let trace = call(&k, &mut state, 83, &mut budget())?;
        let new: [Pose; 2] = serde_json::from_value(trace["proposed"].clone())?;
        let slot = members.iter().position(|&i| i == handle).unwrap();
        let gate = RigidSubset::new(&c, &old, &members, handle, new[slot], 0.8)?;
        if trace["status"] == "completed" {
            let mut b = budget();
            let mut rng = StdRng::seed_from_u64(83 + 202);
            let expected =
                bounded_singleton(&gate, &mut rng, k.lambda, k.activity, k.envelope, &mut b)
                    .unwrap();
            assert_eq!(trace["bath"], json!(expected));
        }
        let mut end = old;
        for (i, &label) in members.iter().enumerate() {
            end[label] = new[i];
        }
        let inverse = transport_members(&end, &members, handle, old[handle])?;
        for (i, &label) in members.iter().enumerate() {
            same_pose(inverse[i], old[label]);
        }
        let r0 = matmul(
            transpose(rotation(old[0].orientation)),
            rotation(old[1].orientation),
        );
        let r1 = matmul(
            transpose(rotation(end[0].orientation)),
            rotation(end[1].orientation),
        );
        for i in 0..3 {
            for j in 0..3 {
                assert!((r0[i][j] - r1[i][j]).abs() < 2e-12);
            }
        }
    }
    Ok(())
}

#[test]
fn fatal_bath_retains_inner_trace_and_never_mutates_or_consumes_outer_coin() -> Result<()> {
    let (c, e) = (tree(0.2), tree(1.));
    let w = Container::new(5., &c)?;
    let points = cloud();
    let mut k = kernel(&c, &e, &w, &points);
    k.config.guidance_strength = 0.;
    let mut state = [pose(1.), pose(2.), pose(0.)];
    let old = state;
    let mut b = budget();
    b.limits.raw_per_leg = 0;
    b.limits.raw_per_outer = 0;
    b.limits.raw_campaign = 0;
    let [mut p, mut i, mut bath, mut a] = rngs(41);
    let mut control = StdRng::seed_from_u64(41 + 303);
    let mut trace = Value::Null;
    assert!(
        k.step(
            &mut state, &mut p, &mut i, &mut bath, &mut a, &mut b, &mut trace
        )
        .is_err()
    );
    assert_eq!(state, old);
    assert_eq!(trace["status"], "fatal");
    assert_eq!(trace["steps"].as_array().unwrap().len(), 8);
    assert!(trace.get("bath_failure").is_some());
    assert_eq!(trace["physical_decisions"], 0);
    assert_eq!(a.random::<u64>(), control.random::<u64>());
    assert_eq!(b.raw, 0);
    Ok(())
}

#[test]
fn disk_continuation_from_completed_outer_boundaries_matches_uninterrupted() -> Result<()> {
    let (c, e) = (tree(0.2), tree(1.));
    let w = Container::new(5., &c)?;
    let points = cloud();
    let k = kernel(&c, &e, &w, &points);
    let initial = vec![pose(1.), pose(2.), pose(0.)];
    let mut state = initial.clone();
    let mut b = budget();
    let mut reference = vec![];
    for block in 0..16 {
        reference.push(call(&k, &mut state, 10000 + block * 997, &mut b)?);
    }
    let mut split = initial;
    let mut sb = budget();
    let mut records = vec![];
    for block in 0..7 {
        records.push(call(&k, &mut split, 10000 + block * 997, &mut sb)?);
    }
    let path = std::env::temp_dir().join(format!(
        "rigid-surrogate-restart-{}.json",
        std::process::id()
    ));
    std::fs::write(
        &path,
        serde_json::to_vec(
            &json!({"poses":split,"completed":7,"raw":sb.raw,"retained":sb.retained}),
        )?,
    )?;
    let cp: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    std::fs::remove_file(path)?;
    split = serde_json::from_value(cp["poses"].clone())?;
    let mut resumed = budget();
    resumed.raw = cp["raw"].as_u64().unwrap();
    resumed.retained = cp["retained"].as_u64().unwrap();
    for block in cp["completed"].as_u64().unwrap()..16 {
        records.push(call(&k, &mut split, 10000 + block * 997, &mut resumed)?);
    }
    assert_eq!(records, reference);
    assert_eq!(state, split);
    assert_eq!(b.raw, resumed.raw);
    assert_eq!(b.retained, resumed.retained);
    assert!(cpu_seconds().is_finite());
    Ok(())
}

#[test]
fn invalid_source_or_configuration_fails_before_draws() -> Result<()> {
    let (c, e) = (tree(0.2), tree(1.));
    let w = Container::new(5., &c)?;
    let points = cloud();
    for bad in 0..4 {
        let mut k = kernel(&c, &e, &w, &points);
        let mut state = [pose(1.), pose(2.), pose(0.)];
        match bad {
            0 => k.config.inner_steps = 1025,
            1 => k.rd = 0.7,
            2 => k.handle = 2,
            _ => state[0] = state[2],
        }
        let old = state;
        let [mut p, mut i, mut bath, mut a] = rngs(7);
        let mut control = StdRng::seed_from_u64(7);
        let mut trace = Value::Null;
        assert!(
            k.step(
                &mut state,
                &mut p,
                &mut i,
                &mut bath,
                &mut a,
                &mut budget(),
                &mut trace
            )
            .is_err()
        );
        assert_eq!(state, old);
        assert_eq!(p.random::<u64>(), control.random::<u64>());
        assert_eq!(trace["status"], "fatal");
    }
    Ok(())
}
