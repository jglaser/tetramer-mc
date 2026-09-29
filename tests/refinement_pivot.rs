//! Independent checks of the opt-in fixed material-pivot pair refinement.
//! A pivot is frozen per basin. Moving p=t+Rc by an even Cartesian increment
//! and R by an inverse-symmetric rotation is reversible in dt dHaar(R).
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use tetramer_mc::{
    contact_discovery::{DiscoveryConfig, discover},
    contact_memory::{MemoryConfig, MemoryState, pivoted_local_pose},
    depletion::GateOptions,
    geometry::{Atom, Shape, SphereTree},
    math::*,
};

fn normal(rng: &mut StdRng) -> Vec3 {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}

fn close_vec(a: Vec3, b: Vec3, tolerance: f64) {
    assert!(norm(sub(a, b)) < tolerance, "{a:?} != {b:?}");
}

fn close_pose(a: Pose, b: Pose, tolerance: f64) {
    close_vec(a.position, b.position, tolerance);
    for (left, right) in rotation(a.orientation)
        .into_iter()
        .zip(rotation(b.orientation))
    {
        close_vec(left, right, tolerance);
    }
}

fn sphere() -> SphereTree {
    SphereTree::new(Shape {
        name: "fixed-pivot sphere reference".into(),
        volume: 4. * std::f64::consts::PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
    .unwrap()
}

fn determinant(mut matrix: [[f64; 6]; 6]) -> f64 {
    let mut result = 1.;
    for column in 0..6 {
        let pivot = (column..6)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        if pivot != column {
            matrix.swap(pivot, column);
            result = -result;
        }
        let value = matrix[column][column];
        result *= value;
        for row in column + 1..6 {
            let ratio = matrix[row][column] / value;
            for entry in column + 1..6 {
                matrix[row][entry] -= ratio * matrix[column][entry];
            }
        }
    }
    result
}

#[test]
fn pivot_increment_and_inverse_are_exact_rigid_operations() {
    let mut rng = StdRng::seed_from_u64(202609280021);
    for _ in 0..128 {
        let old = Pose {
            position: scale(normal(&mut rng), 40.),
            orientation: quaternion(cayley(normal(&mut rng))),
        };
        let pivot = scale(normal(&mut rng), 20.);
        let displacement = scale(normal(&mut rng), 0.3);
        let increment = scale(normal(&mut rng), 0.4);
        let new = pivoted_local_pose(old, displacement, increment, pivot);
        close_vec(new.apply(pivot), add(old.apply(pivot), displacement), 2e-12);
        let recovered =
            pivoted_local_pose(new, scale(displacement, -1.), scale(increment, -1.), pivot);
        close_pose(recovered, old, 2e-12);
    }
}

#[test]
fn fixed_increment_preserves_translation_times_haar_volume() {
    let mut rng = StdRng::seed_from_u64(202609280022);
    let epsilon = 1e-5;
    for _ in 0..12 {
        let old = Pose {
            position: normal(&mut rng),
            orientation: quaternion(cayley(normal(&mut rng))),
        };
        let pivot = scale(normal(&mut rng), 15.);
        let displacement = scale(normal(&mut rng), 0.4);
        let increment = scale(normal(&mut rng), 0.3);
        let reference = pivoted_local_pose(old, displacement, increment, pivot);
        let inverse_reference = transpose(rotation(reference.orientation));
        let mut derivative = [[0.; 6]; 6];
        for column in 0..6 {
            let mut pair = [[0.; 6]; 2];
            for (index, sign) in [-1., 1.].into_iter().enumerate() {
                let mut perturbed = old;
                if column < 3 {
                    perturbed.position[column] += sign * epsilon;
                } else {
                    let mut angular = [0.; 3];
                    angular[column - 3] = sign * epsilon / 2.;
                    perturbed.orientation =
                        quaternion(matmul(cayley(angular), rotation(old.orientation)));
                }
                let moved = pivoted_local_pose(perturbed, displacement, increment, pivot);
                let relative = matmul(rotation(moved.orientation), inverse_reference);
                pair[index][..3].copy_from_slice(&sub(moved.position, reference.position));
                pair[index][3..].copy_from_slice(&[
                    (relative[2][1] - relative[1][2]) / 2.,
                    (relative[0][2] - relative[2][0]) / 2.,
                    (relative[1][0] - relative[0][1]) / 2.,
                ]);
            }
            for row in 0..6 {
                derivative[row][column] = (pair[1][row] - pair[0][row]) / (2. * epsilon);
            }
        }
        assert!((determinant(derivative) - 1.).abs() < 2e-8);
    }
}

#[test]
fn zero_bath_accepts_all_valid_pivot_moves_and_retains_rejections() -> Result<()> {
    let tree = sphere();
    let config = MemoryConfig {
        slots: 1,
        radius: Some(3.),
        global_probability: 0.,
        local_translation_std_a: 0.2,
        local_small_angle_std_degrees: 5.,
        ..Default::default()
    }
    .resolve(&tree, 0.5, 0.)?;
    let mut state = MemoryState {
        poses: vec![Pose {
            position: [2.1, 0., 0.],
            orientation: [1., 0., 0., 0.],
        }],
    };
    let mut rng = StdRng::seed_from_u64(202609280023);
    let mut accepted = 0;
    for _ in 0..256 {
        let update = state.update_with_local_pivot(
            &tree,
            &config,
            GateOptions::default(),
            Some([-1., 0., 0.]),
            &mut rng,
        )?;
        if update.hard_valid && !update.outside_ball {
            assert!(update.accepted);
            assert_eq!(update.gate.as_ref().unwrap().log_weight, 0.);
            accepted += 1;
        } else {
            assert!(!update.accepted);
            assert_eq!(update.old_pose, update.retained_pose);
        }
    }
    assert!(accepted > 0 && accepted < 256);
    Ok(())
}

#[test]
fn warmup_adaptation_freezes_before_every_retained_sample_and_replays() -> Result<()> {
    let tree = sphere();
    let config = DiscoveryConfig {
        starts: 2,
        search_steps: 0,
        search_points: 32,
        validation_points: 32,
        refine_steps: 96,
        burn: 32,
        save_every: 4,
        activity: 0.2,
        rd: 0.4,
        refine_contact_pivot: true,
        refine_adapt: true,
        refine_adapt_window: 8,
        envelope_max_cells: 31,
        seed: 202609280024,
        ..Default::default()
    };
    let result = discover(&tree, &config)?;
    for slot in &result.slots {
        let protocol = &slot.refinement_protocol;
        assert!(protocol.contact_pivot_body.is_some());
        assert_eq!(protocol.frozen_after_step, config.burn);
        assert!(
            protocol
                .updates
                .iter()
                .all(|update| update.after_step <= config.burn)
        );
        let mut state = MemoryState {
            poses: vec![slot.optimized_pose],
        };
        let mut rng = StdRng::seed_from_u64(slot.seeds.refinement);
        let mut active = slot.resolved_memory_config.clone();
        for (index, saved) in slot.refinement_trace.iter().enumerate() {
            let step = index + 1;
            if step > config.burn {
                assert_eq!(active, protocol.frozen_memory_config);
            }
            let replay = state.update_with_local_pivot(
                &tree,
                &active,
                config.gate_options(),
                protocol.contact_pivot_body,
                &mut rng,
            )?;
            assert_eq!(serde_json::to_value(&replay)?, serde_json::to_value(saved)?);
            for update in protocol
                .updates
                .iter()
                .filter(|update| update.after_step == step)
            {
                active.local_translation_std_a =
                    slot.resolved_memory_config.local_translation_std_a * update.scale_after;
                active.local_small_angle_std_degrees =
                    slot.resolved_memory_config.local_small_angle_std_degrees * update.scale_after;
            }
        }
        assert!(
            slot.refinement_samples
                .iter()
                .all(|sample| sample.step > config.burn)
        );
        assert_eq!(slot.refinement_samples.len(), 16);
    }
    Ok(())
}
