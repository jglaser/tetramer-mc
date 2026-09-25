//! Independent finite-system physical equilibrium check of the production
//! fixed-time collective phase, composed with uniform single-body Poisson MH.
//! Three equal hard spheres, rd < (2/sqrt(3)-1)*core_radius: three exclusion
//! spheres cannot share a point in any hard-valid configuration. The physical
//! many-body ideal-depletion target therefore has an exact analytic pair-lens
//! reference, independently importance sampled in the protein-only spherical
//! wall. The bath itself has no wall exclusion. Saved samples retain rejections.
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    cluster_phase::{ClusterPhase, ClusterPhaseConfig, ClusterPhaseCounts},
    depletion::GateOptions,
    docking::{DockingMethod, DockingProposal},
    geometry::{Atom, Shape, SphereTree},
    math::{IDENTITY, Pose, cayley, norm, sub},
    proposal::FrozenRelativePoseProposal,
    rigid_subset::RigidSubset,
    simulation::hash_bytes,
    spherical::Container,
};

const RD: f64 = 0.14;
const Z: f64 = 5.;
const LAMBDA: f64 = 5.;
const WALL: f64 = 3.5;
const CENTER_RADIUS: f64 = WALL - 1.;
const OBS: usize = 4;

fn pose(position: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn uniform_pose(rng: &mut StdRng) -> Pose {
    let position = loop {
        let p = std::array::from_fn(|_| rng.random_range(-CENTER_RADIUS..CENTER_RADIUS));
        if norm(p) <= CENTER_RADIUS {
            break p;
        }
    };
    let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
    let length = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    Pose {
        position,
        orientation: q.map(|v| v / length),
    }
}
fn lens(distance: f64) -> f64 {
    let r = 1. + RD;
    if distance >= 2. * r {
        0.
    } else {
        PI * (4. * r + distance) * (2. * r - distance).powi(2) / 12.
    }
}
fn exact(poses: &[Pose]) -> Option<(f64, [f64; OBS])> {
    let mut weight = 0.;
    let mut contacts = 0;
    let mut distance2 = 0.;
    for i in 0..3 {
        for j in i + 1..3 {
            let d = norm(sub(poses[i].position, poses[j].position));
            if d < 2. {
                return None;
            }
            weight += Z * lens(d);
            contacts += usize::from(d < 2. * (1. + RD));
            distance2 += d * d;
        }
    }
    let largest = match contacts {
        0 => 1,
        1 => 2,
        _ => 3,
    };
    Some((
        weight.exp(),
        [
            contacts as f64,
            largest as f64,
            distance2 / 3.,
            poses.iter().map(|p| p.orientation[0].powi(2)).sum::<f64>() / 3.,
        ],
    ))
}

#[derive(Clone, Debug)]
struct Reference {
    mean: [f64; OBS],
    se: [f64; OBS],
    ess: f64,
    valid: usize,
}
fn reference(draws: usize) -> Reference {
    let mut rng = StdRng::seed_from_u64(10459359);
    let mut sumw = 0.;
    let mut sumw2 = 0.;
    let mut valid = 0;
    let mut sumwf = [0.; OBS];
    let mut sumw2f = [0.; OBS];
    let mut sumw2f2 = [0.; OBS];
    for _ in 0..draws {
        let poses = [
            uniform_pose(&mut rng),
            uniform_pose(&mut rng),
            uniform_pose(&mut rng),
        ];
        let Some((w, f)) = exact(&poses) else {
            continue;
        };
        valid += 1;
        sumw += w;
        sumw2 += w * w;
        for k in 0..OBS {
            sumwf[k] += w * f[k];
            sumw2f[k] += w * w * f[k];
            sumw2f2[k] += w * w * f[k] * f[k];
        }
    }
    let mean = sumwf.map(|v| v / sumw);
    let se = std::array::from_fn(|k| {
        ((sumw2f2[k] - 2. * mean[k] * sumw2f[k] + mean[k] * mean[k] * sumw2).max(0.)
            / (sumw * sumw))
            .sqrt()
    });
    Reference {
        mean,
        se,
        ess: sumw * sumw / sumw2,
        valid,
    }
}

fn learned_proposal(tree: &SphereTree) -> DockingProposal {
    let shape_hash = hash_bytes(&serde_json::to_vec(&tree.shape).unwrap());
    let covariance = |diagonal: [f64; 6], cross: f64| {
        let mut l = [[0.; 6]; 6];
        for k in 0..6 {
            l[k][k] = diagonal[k];
        }
        l[4][0] = cross;
        let matrix: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|k| l[i][k] * l[j][k]).sum())
        });
        matrix
    };
    // Deliberately unequal weights, off-diagonal translation/rotation
    // covariance, angular centers and scales unrelated to the isotropic target.
    let model = json!({
        "angular_length":1.,"shape_sha256":shape_hash,
        "coordinate_convention":"anchor-body-relative",
        "anchors":[{"position":[2.1,0.,0.],"rotation":IDENTITY},
            {"position":[0.,2.7,0.],"rotation":cayley([0.3,-0.5,0.2])}],
        "means":[[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]],
        "covariances":[covariance([0.35,0.8,0.6,0.65,1.2,0.9],0.3),
            covariance([0.9,0.45,0.7,1.3,0.6,0.8],-0.25)],
        "weights":[0.35,0.65]
    });
    let frozen = FrozenRelativePoseProposal::from_json_str_open(
        &model.to_string(),
        [2. * (WALL + tree.bound); 3],
        0.15,
        &shape_hash,
    )
    .unwrap();
    DockingProposal::new(frozen, DockingMethod::PosteriorInvolution, 0.45, [0.; 3]).unwrap()
}

#[derive(Debug)]
struct Chain {
    mean: [f64; OBS],
    se: [f64; OBS],
    cluster: ClusterPhaseCounts,
    single_accepted: u64,
    single_attempted: u64,
    corrected_involutions: u64,
    accepted_involutions: u64,
    accepted_corrected_involutions: u64,
}
fn run_chain(
    tree: &SphereTree,
    wall: &Container,
    mut poses: Vec<Pose>,
    seed: u64,
    learned: bool,
) -> Chain {
    let cfg = ClusterPhaseConfig {
        duration: 0.5,
        dimer_rate: 4.,
        trimer_rate: 1.,
        transport_probability: if learned { 0.7 } else { 0. },
        local_translation_std_a: 0.5,
        local_small_angle_std_degrees: 20.,
        ..ClusterPhaseConfig::default()
    };
    let gate = GateOptions {
        max_cells: 1,
        max_depth: 0,
        min_width: 0.,
    };
    let proposal = learned.then(|| learned_proposal(tree));
    let engine = ClusterPhase::new(tree, wall, RD, Z, LAMBDA, gate, cfg, proposal).unwrap();
    let mut clock = StdRng::seed_from_u64(seed + 1);
    let mut proposal_rng = StdRng::seed_from_u64(seed + 2);
    let mut gate_rng = StdRng::seed_from_u64(seed + 3);
    let mut accept_rng = StdRng::seed_from_u64(seed + 4);
    let mut bias_rng = StdRng::seed_from_u64(seed + 5);
    let mut single_rng = StdRng::seed_from_u64(seed + 6);
    let mut single_gate_rng = StdRng::seed_from_u64(seed + 7);
    let mut bias_state = None;
    let burn = 3_000;
    let block_size = 500;
    let blocks = 48;
    let mut block_means = vec![[0.; OBS]; blocks];
    let mut totals = ClusterPhaseCounts::default();
    let mut single_accepted = 0;
    let mut single_attempted = 0;
    let mut corrected_involutions = 0;
    let mut accepted_involutions = 0;
    let mut accepted_corrected_involutions = 0;
    for sweep in 0..burn + block_size * blocks {
        // Fixed three single-body attempts; proposals are uniform in the same
        // sphere-center domain in both directions. Orientations are Haar.
        for _ in 0..3 {
            let i = single_rng.random_range(0..3);
            let proposed = uniform_pose(&mut single_rng);
            let trial = RigidSubset::new(tree, &poses, &[i], i, proposed, RD).unwrap();
            if sweep >= burn {
                single_attempted += 1;
            }
            if trial.hard_valid(Some(wall), [0.; 3]) {
                let sampled = trial.sample(&mut single_gate_rng, LAMBDA, Z, gate).unwrap();
                if single_rng.random::<f64>().max(f64::MIN_POSITIVE).ln()
                    < sampled.log_weight.min(0.)
                {
                    poses[i] = proposed;
                    if sweep >= burn {
                        single_accepted += 1;
                    }
                }
            }
        }
        // Physical production implementation, no test substitute for the
        // event schedule or the collective many-body depletion gate.
        let (counts, records) = engine
            .run(
                &mut poses,
                &mut clock,
                &mut proposal_rng,
                &mut gate_rng,
                &mut accept_rng,
                &mut bias_rng,
                None,
                &mut bias_state,
                learned,
            )
            .unwrap();
        if !learned {
            assert!(records.is_empty());
        }
        if sweep >= burn {
            for row in &records {
                if row["kind"] == "cluster_event" && row["proposal"]["branch"] == "involution" {
                    let corrected = row["proposal"]["log_reverse_forward"]
                        .as_f64()
                        .is_some_and(|x| x.abs() > 1e-6);
                    corrected_involutions += u64::from(corrected);
                    accepted_involutions += u64::from(row["accepted"] == true);
                    accepted_corrected_involutions +=
                        u64::from(corrected && row["accepted"] == true);
                }
            }
        }
        let (_, f) = exact(&poses).expect("physical kernel produced hard overlap");
        assert!(poses.iter().all(|&p| wall.contains(p)));
        if sweep >= burn {
            totals.add(&counts);
            let block = (sweep - burn) / block_size;
            for k in 0..OBS {
                block_means[block][k] += f[k] / block_size as f64;
            }
        }
    }
    let mean =
        std::array::from_fn(|k| block_means.iter().map(|b| b[k]).sum::<f64>() / blocks as f64);
    let se = std::array::from_fn(|k| {
        (block_means
            .iter()
            .map(|b| (b[k] - mean[k]).powi(2))
            .sum::<f64>()
            / (blocks * (blocks - 1)) as f64)
            .sqrt()
    });
    Chain {
        mean,
        se,
        cluster: totals,
        single_accepted,
        single_attempted,
        corrected_involutions,
        accepted_involutions,
        accepted_corrected_involutions,
    }
}

#[test]
fn fixed_duration_cluster_phase_matches_independent_sphere_equilibrium() {
    assert!(RD < 2. / 3.0_f64.sqrt() - 1.);
    let tree = SphereTree::new(Shape {
        name: "analytic sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
    .unwrap();
    let wall = Container::new(WALL, &tree).unwrap();
    let reference = reference(250_000);
    assert!(reference.ess > 60_000. && reference.valid > 60_000);
    let dispersed = vec![
        pose([-1.8, -1.1, 0.]),
        pose([1.8, -1.1, 0.]),
        pose([0., 2., 0.]),
    ];
    let side = 2.05;
    let r = side / 3.0_f64.sqrt();
    let aggregated = vec![
        pose([r, 0., 0.]),
        pose([-r / 2., side / 2., 0.]),
        pose([-r / 2., -side / 2., 0.]),
    ];
    assert_eq!(exact(&dispersed).unwrap().1[0], 0.);
    assert_eq!(exact(&aggregated).unwrap().1[0], 3.);
    let chains = [
        run_chain(&tree, &wall, dispersed, 820001, false),
        run_chain(&tree, &wall, aggregated.clone(), 820002, false),
        run_chain(&tree, &wall, aggregated, 820003, true),
    ];
    eprintln!("independent_reference: {reference:?}");
    for (i, c) in chains.iter().enumerate() {
        eprintln!("initialization_{i}: {c:?}");
        if i == 2 {
            assert!(c.cluster.transport_events > 5_000);
            assert!(
                c.corrected_involutions > 1_000
                    && c.accepted_involutions > 50
                    && c.accepted_corrected_involutions > 50,
                "need accepted nontrivially corrected map trials"
            );
        }
        assert!(c.cluster.events > 5_000 && c.cluster.accepted > 1_000);
        assert!(
            c.cluster.accepted_attachments > 100 && c.cluster.accepted_detachments > 100,
            "collective chain must actually change aggregation state"
        );
        assert!(c.single_accepted > 10_000 && c.single_attempted == 72_000);
        assert_eq!(c.cluster.phases, 24_000);
        for k in 0..OBS {
            let tolerance = 5. * (c.se[k].powi(2) + reference.se[k].powi(2)).sqrt() + 0.002;
            assert!(
                (c.mean[k] - reference.mean[k]).abs() < tolerance,
                "initialization {i}, observable {k}: chain {} +/- {}, reference {} +/- {}",
                c.mean[k],
                c.se[k],
                reference.mean[k],
                reference.se[k]
            );
        }
    }
    for k in 0..OBS {
        let tolerance = 5. * (chains[0].se[k].powi(2) + chains[1].se[k].powi(2)).sqrt() + 0.002;
        assert!(
            (chains[0].mean[k] - chains[1].mean[k]).abs() < tolerance,
            "initialization disagreement in observable {k}"
        );
    }
}
