//! A fixed, nonunit proposal-ratio witness for the composed accepted flow.
//! The anchor is a coordinate frame, not a third physical sphere. No trajectory,
//! witness search, proposal redraw, or production integration is performed.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::f64::consts::PI;
use tetramer_mc::{
    basin_involution::BasinTrace,
    depletion::GateOptions,
    dimer_tree_proposal::{DimerTreeProposal, DimerTreeTrace},
    docking::{DockingMethod, DockingProposal},
    flexible_subset::FlexibleSubset,
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, norm, sub},
    proposal::FrozenRelativePoseProposal,
};

fn pose(position: [f64; 3]) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}

#[derive(Default)]
struct Moments {
    sum: f64,
    squares: f64,
}
impl Moments {
    fn add(&mut self, x: f64) {
        self.sum += x;
        self.squares += x * x;
    }
    fn mean(&self, n: usize) -> f64 {
        self.sum / n as f64
    }
    fn variance(&self, n: usize) -> f64 {
        ((self.squares - self.sum * self.sum / n as f64) / (n - 1) as f64).max(0.)
    }
}

fn lens(radius: f64, distance: f64) -> f64 {
    if distance >= 2. * radius {
        0.
    } else {
        PI * (4. * radius + distance) * (2. * radius - distance).powi(2) / 12.
    }
}

#[test]
fn actual_two_edge_map_and_world_gate_have_correct_nonunit_accepted_flow() -> Result<()> {
    // One centered Gaussian, Cartesian standard deviation 3. The fixed c=0
    // traces move root (0,0,0)->(0,0.5,0), internal offset (1,0,0)->(3,0,0).
    // Both orientations remain identity, so their Haar factors cancel. The
    // independently known joint log G(old)/G(new) is (0.25+9-1)/(2*9)=11/24.
    let sha = "0000000000000000000000000000000000000000000000000000000000000000";
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j {
                if i < 3 { 9. } else { 1. }
            } else {
                0.
            }
        })
    });
    let model = json!({
        "coordinate_convention":"anchor-body-relative", "shape_sha256":sha,
        "angular_length":1., "weights":[1.],
        "anchors":[{"position":[0.,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
        "means":vec![[0.;6]], "covariances":[covariance]
    });
    let frozen =
        FrozenRelativePoseProposal::from_json_str_open(&model.to_string(), [20.; 3], 0.1, sha)?;
    let docking = DockingProposal::new(frozen, DockingMethod::PosteriorInvolution, 0., [0.; 3])?;
    let proposal = DimerTreeProposal::new(&docking)?;
    let anchor = pose([0.; 3]);
    let old = [pose([0.; 3]), pose([1., 0., 0.])];
    let trace = DimerTreeTrace {
        edges: [
            BasinTrace {
                source: 0,
                target: 0,
                noise: [0., 1. / 6., 0., 0., 0., 0.],
            },
            BasinTrace {
                source: 0,
                target: 0,
                noise: [1., 0., 0., 0., 0., 0.],
            },
        ],
    };
    let forward = proposal.apply(anchor, old[0], old[1], &trace)?;
    let candidate = forward.candidate.expect("predeclared finite forward trace");
    let new = [candidate.root, candidate.child];
    assert!(norm(sub(new[0].position, [0., 0.5, 0.])) < 1e-13);
    assert!(norm(sub(new[1].position, [3., 0.5, 0.])) < 1e-13);
    let backward = proposal.apply(anchor, new[0], new[1], &candidate.inverse_trace)?;
    let returned = backward
        .candidate
        .expect("predeclared finite inverse trace");
    for (actual, expected) in [returned.root, returned.child].into_iter().zip(old) {
        assert!(norm(sub(actual.position, expected.position)) < 1e-13);
        assert_eq!(actual.orientation, expected.orientation);
    }
    let log_r = candidate.diagnostics.log_reverse_forward;
    let inverse_log_r = returned.diagnostics.log_reverse_forward;
    assert!((log_r - 11. / 24.).abs() < 1e-12);
    assert!((log_r + inverse_log_r).abs() < 1e-12);
    assert!((candidate.diagnostics.expanded_log_reverse_forward - log_r).abs() < 1e-12);
    assert!((returned.diagnostics.expanded_log_reverse_forward - inverse_log_r).abs() < 1e-12);
    assert!((1.4..1.8).contains(&log_r.exp()));

    let core = 0.2_f64;
    let rd = 0.8;
    let tree = SphereTree::new(Shape {
        name: "composed dimer-flow sphere".into(),
        volume: 4. * PI * core.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: core,
        }],
    })?;
    let gates = [
        FlexibleSubset::new(&tree, &old, &[0, 1], &new, rd)?,
        FlexibleSubset::new(&tree, &new, &[0, 1], &old, rd)?,
    ];
    let opts = GateOptions {
        max_cells: 1,
        max_depth: 0,
        min_width: 0.,
    };
    for gate in &gates {
        assert_eq!(gate.spectator_count(), 0, "virtual anchor entered bath");
        assert!(gate.hard_valid(None, [0.; 3]));
    }
    assert_eq!(
        gates[0].envelope(opts)?.volume.to_bits(),
        gates[1].envelope(opts)?.volume.to_bits()
    );
    let lambda = 0.35;
    let mut zero_rng = StdRng::seed_from_u64(808207);
    let mut zero_untouched = StdRng::seed_from_u64(808207);
    let zero_forward = (log_r + gates[0].sample(&mut zero_rng, lambda, 0., opts)?.log_weight)
        .min(0.)
        .exp();
    let zero_reverse = (inverse_log_r
        + gates[1].sample(&mut zero_rng, lambda, 0., opts)?.log_weight)
        .min(0.)
        .exp();
    assert_eq!(zero_forward, 1.);
    assert!(
        zero_reverse < 0.7,
        "zero activity still needs proposal correction"
    );
    assert!((zero_forward - log_r.exp() * zero_reverse).abs() < 1e-12);
    assert_eq!(zero_rng.random::<u64>(), zero_untouched.random::<u64>());

    let z = 0.45;
    let old_overlap = lens(core + rd, norm(sub(old[1].position, old[0].position)));
    let new_overlap = lens(core + rd, norm(sub(new[1].position, new[0].position)));
    assert!((old_overlap - 5. * PI / 12.).abs() < 1e-13);
    assert_eq!(new_overlap, 0.);
    // The physical target uses Cartesian translation x Haar. There is no d^2
    // in this endpoint ratio, nor an additional analytic bath term inside MH.
    let weighted_reverse = (log_r + z * (new_overlap - old_overlap)).exp();
    let n = 12_000;
    let mut correct = [Moments::default(), Moments::default()];
    let mut omitted = [Moments::default(), Moments::default()];
    let mut total_raw = 0;
    for index in 0..2 {
        let mut rng = StdRng::seed_from_u64([808208, 808209][index]);
        let correction = [log_r, inverse_log_r][index];
        for _ in 0..n {
            let draw = gates[index].sample(&mut rng, lambda, z, opts)?;
            total_raw += draw.raw_points;
            assert_eq!(draw.retained_points, draw.gained + draw.lost);
            correct[index].add((correction + draw.log_weight).min(0.).exp());
            omitted[index].add(draw.log_weight.min(0.).exp());
        }
    }
    let residual = |moments: &[Moments; 2]| {
        let difference = moments[0].mean(n) - weighted_reverse * moments[1].mean(n);
        let se = ((moments[0].variance(n) + weighted_reverse.powi(2) * moments[1].variance(n))
            / n as f64)
            .sqrt();
        (difference, se)
    };
    let (difference, se) = residual(&correct);
    assert!(
        difference.abs() < 6. * se + 1e-5,
        "composed flow residual {difference}, SE {se}, forward {}, weighted reverse {}",
        correct[0].mean(n),
        weighted_reverse * correct[1].mean(n)
    );
    // The same fixed draws must expose omission of the map correction. This is
    // a sensitivity control, not another run or selection of a favorable seed.
    let (wrong_difference, wrong_se) = residual(&omitted);
    assert!(wrong_difference.abs() > 6. * wrong_se + 1e-5);
    assert!(total_raw > 100_000);
    Ok(())
}
