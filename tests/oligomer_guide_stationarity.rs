//! Independent finite-system physical equilibrium check of the production
//! fixed-time collective phase with the oligomer-conditioned guide, composed
//! with uniform single-body Poisson MH. Four independent streams per guide
//! length (two dispersed, two aggregated) retain every attempted sweep.
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
    oligomer_guide::OligomerGuideConfig,
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
// Frozen independent reference from the completed 250,000-draw calculation in
// runs/cluster-phase-validation-20260925/stationarity.log. That calculation used
// seed 10459359 and independently sampled Haar orientations and uniform centers
// in exactly this wall; it evaluated analytic pair lenses, not the Poisson gate.
// Its hard-valid ESS is 82673.1313357483. Reusing it avoids rerunning completed
// physical calculations. A changed shape, bath, wall, or observable requires a
// new reference and must not silently reuse these constants.
fn archived_reference() -> Reference {
    Reference {
        mean: [
            0.5078716417323399,
            1.502843616463136,
            9.604587403498265,
            0.25021435304297224,
        ],
        se: [
            0.0024812894354681292,
            0.0023839506628228926,
            0.008758773488373017,
            0.0005013158501871876,
        ],
        ess: 82673.1313357483,
        valid: 87123,
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
    elapsed_seconds: f64,
}
fn run_chain(
    tree: &SphereTree,
    wall: &Container,
    mut poses: Vec<Pose>,
    seed: u64,
    guide_steps: usize,
) -> Chain {
    let started = std::time::Instant::now();
    let cfg = ClusterPhaseConfig {
        duration: 0.5,
        dimer_rate: 4.,
        trimer_rate: 1.,
        transport_probability: 0.7,
        guide: Some(OligomerGuideConfig {
            steps: guide_steps,
            anchor_count: 2,
            score_power: 1.,
        }),
        local_translation_std_a: 0.5,
        local_small_angle_std_degrees: 20.,
        ..ClusterPhaseConfig::default()
    };
    let gate = GateOptions {
        max_cells: 1,
        max_depth: 0,
        min_width: 0.,
    };
    let proposal = Some(learned_proposal(tree));
    let engine = ClusterPhase::new(tree, wall, RD, Z, LAMBDA, gate, cfg, proposal).unwrap();
    let mut clock = StdRng::seed_from_u64(seed + 1);
    let mut proposal_rng = StdRng::seed_from_u64(seed + 2);
    let mut gate_rng = StdRng::seed_from_u64(seed + 3);
    let mut accept_rng = StdRng::seed_from_u64(seed + 4);
    let mut bias_rng = StdRng::seed_from_u64(seed + 5);
    let mut single_rng = StdRng::seed_from_u64(seed + 6);
    let mut single_gate_rng = StdRng::seed_from_u64(seed + 7);
    let mut bias_state = None;
    let burn = 2_000;
    let block_size = 500;
    let blocks = 16;
    let mut block_means = vec![[0.; OBS]; blocks];
    let mut totals = ClusterPhaseCounts::default();
    let mut single_accepted = 0;
    let mut single_attempted = 0;
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
                false,
            )
            .unwrap();
        assert!(records.is_empty());
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
        elapsed_seconds: started.elapsed().as_secs_f64(),
    }
}

#[test]
fn oligomer_conditioned_guide_matches_independent_physical_sphere_equilibrium() {
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
    let reference = archived_reference();
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
    eprintln!("archived_independent_reference: {reference:?}");

    let mut summaries: Vec<([f64; OBS], [f64; OBS])> = Vec::new();
    for steps in [1, 4, 16] {
        // The seed allocation is frozen before sampling. All arms receive the
        // same sweep allocation, ordinary move sizes, and phase duration/rates.
        // Inner proposals consume separate deterministic streams naturally; no
        // assertion pretends trajectories remain paired after different moves.
        let chains: Vec<_> = (0..4)
            .map(|stream| {
                let initial = if stream < 2 {
                    dispersed.clone()
                } else {
                    aggregated.clone()
                };
                run_chain(
                    &tree,
                    &wall,
                    initial,
                    960250000 + 1000 * steps as u64 + 10 * stream,
                    steps,
                )
            })
            .collect();
        let mut totals = ClusterPhaseCounts::default();
        for (stream, c) in chains.iter().enumerate() {
            eprintln!("guide_steps_{steps}_stream_{stream}: {c:?}");
            assert!(c.elapsed_seconds > 0.);
            assert_eq!(c.single_attempted, 24_000);
            assert!(c.single_accepted > 2_000);
            assert_eq!(c.cluster.phases, 8_000);
            assert!(c.cluster.events > 1_000 && c.cluster.accepted > 100);
            assert!(c.cluster.guide_events > 500);
            assert!(
                c.cluster.guide_inner_attempts
                    >= c.cluster
                        .guide_events
                        .saturating_sub(c.cluster.guide_endpoint_nulls)
            );
            assert!(c.cluster.guide_inner_accepted > 20);
            totals.add(&c.cluster);
            for k in 0..OBS {
                let tolerance = 5. * (c.se[k].powi(2) + reference.se[k].powi(2)).sqrt() + 0.002;
                assert!(
                    (c.mean[k] - reference.mean[k]).abs() < tolerance,
                    "m={steps}, stream={stream}, observable={k}: {} +/- {}, ref {} +/- {}",
                    c.mean[k],
                    c.se[k],
                    reference.mean[k],
                    reference.se[k]
                );
            }
        }
        // These conditions rule out a vacuous success caused solely by the
        // unchanged single-body/local kernels. guide_accepted must count actual
        // accepted nonidentity endpoints, excluding empty/no-op inner paths.
        assert!(
            totals.guide_accepted > 20,
            "m={steps}: insufficient accepted nonidentity guide moves: {totals:?}"
        );
        assert!(
            totals.accepted_attachments > 50 && totals.accepted_detachments > 50,
            "m={steps}: insufficient aggregation-state changes"
        );
        assert!(
            totals.completed_exchanges > 0,
            "m={steps}: physical collective contact-exchange coverage absent"
        );

        let mean: [f64; OBS] =
            std::array::from_fn(|k| chains.iter().map(|c| c.mean[k]).sum::<f64>() / 4.);
        let se: [f64; OBS] = std::array::from_fn(|k| {
            let blocked = chains.iter().map(|c| c.se[k].powi(2)).sum::<f64>().sqrt() / 4.;
            let between = (chains
                .iter()
                .map(|c| (c.mean[k] - mean[k]).powi(2))
                .sum::<f64>()
                / 12.)
                .sqrt();
            blocked.max(between)
        });
        eprintln!("guide_steps_{steps}_pooled: mean={mean:?}, se={se:?}, cluster={totals:?}");
        for k in 0..OBS {
            let tolerance = 5. * (se[k].powi(2) + reference.se[k].powi(2)).sqrt() + 0.002;
            assert!(
                (mean[k] - reference.mean[k]).abs() < tolerance,
                "m={steps}: pooled reference disagreement in observable {k}"
            );
            let from_dispersed = (chains[0].mean[k] + chains[1].mean[k]) / 2.;
            let from_aggregate = (chains[2].mean[k] + chains[3].mean[k]) / 2.;
            let initialization_se = chains.iter().map(|c| c.se[k].powi(2)).sum::<f64>().sqrt() / 2.;
            assert!(
                (from_dispersed - from_aggregate).abs() < 5. * initialization_se + 0.002,
                "m={steps}: initialization disagreement in observable {k}"
            );
        }
        summaries.push((mean, se));
    }
    for a in 0..summaries.len() {
        for b in a + 1..summaries.len() {
            for k in 0..OBS {
                let tolerance =
                    5. * (summaries[a].1[k].powi(2) + summaries[b].1[k].powi(2)).sqrt() + 0.002;
                assert!(
                    (summaries[a].0[k] - summaries[b].0[k]).abs() < tolerance,
                    "guide-length arms {a}/{b} disagree in observable {k}"
                );
            }
        }
    }
}
