//! Independent target-mixture and source-label checks for posterior transport.
//! Physical many-body tests use independent exact starts in involution_depletion.rs.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use tetramer_mc::{
    basin_involution::{BasinPair, BasinTrace, FixedBasinInvolution},
    docking::{DockingMethod, DockingProposal},
    math::*,
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
};

const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const WEIGHTS: [f64; 2] = [0.23, 0.77];

fn parameters() -> Vec<GaussianComponentParameters> {
    (0..2)
        .map(|j| {
            let scales = if j == 0 {
                [0.7, 1.1, 0.9, 0.35, 0.5, 0.4]
            } else {
                [1.4, 0.8, 1.3, 0.9, 0.65, 1.1]
            };
            let mut lower = [[0.; 6]; 6];
            for i in 0..6 {
                lower[i][i] = scales[i];
            }
            lower[1][0] = 0.2;
            lower[3][0] = 0.17;
            lower[5][2] = -0.15;
            lower[4][3] = 0.12;
            GaussianComponentParameters {
                anchor_position: if j == 0 {
                    [-0.5, 0.1, 0.]
                } else {
                    [0.7, -0.2, 0.3]
                },
                anchor_rotation: if j == 0 {
                    cayley([0.1, -0.2, 0.3])
                } else {
                    cayley([-0.5, 0.7, 0.2])
                },
                mean: [0.1, 0., -0.1, 0.05, -0.1, 0.02],
                covariance: std::array::from_fn(|i| {
                    std::array::from_fn(|k| (0..6).map(|d| lower[i][d] * lower[k][d]).sum())
                }),
                weight: WEIGHTS[j],
            }
        })
        .collect()
}
fn model() -> Result<FrozenRelativePoseProposal> {
    FrozenRelativePoseProposal::from_components_open(parameters(), 1.3, [20.; 3], 0.1, SHA, SHA)
}
fn map(c: f64) -> Result<FixedBasinInvolution> {
    FixedBasinInvolution::new(
        parameters(),
        1.3,
        c,
        vec![
            BasinPair {
                first: 0,
                second: 0,
                weight: WEIGHTS[0].powi(2),
            },
            BasinPair {
                first: 0,
                second: 1,
                weight: 2. * WEIGHTS[0] * WEIGHTS[1],
            },
            BasinPair {
                first: 1,
                second: 1,
                weight: WEIGHTS[1].powi(2),
            },
        ],
    )
}
fn normal<const N: usize>(rng: &mut StdRng) -> [f64; N] {
    std::array::from_fn(|_| StandardNormal.sample(rng))
}
fn anchor() -> Pose {
    Pose {
        position: [7., -4., 3.],
        orientation: quaternion(cayley([0.3, -0.5, 0.2])),
    }
}
fn lab(p: Pose) -> Pose {
    Pose {
        position: anchor().apply(p.position),
        orientation: quaternion(matmul(
            rotation(anchor().orientation),
            rotation(p.orientation),
        )),
    }
}
fn relative(p: Pose) -> Pose {
    Pose {
        position: anchor().inverse(p.position),
        orientation: quaternion(matmul(
            transpose(rotation(anchor().orientation)),
            rotation(p.orientation),
        )),
    }
}
fn log_g(model: &FrozenRelativePoseProposal, p: Pose) -> Result<f64> {
    model.relative_log_density(p.position, rotation(p.orientation))
}
#[derive(Clone, Copy, Default)]
struct Moment {
    n: usize,
    sum: f64,
    sq: f64,
}
impl Moment {
    fn add(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.sq += x * x;
    }
    fn z(self, expected: f64) -> f64 {
        let n = self.n as f64;
        (self.sum / n - expected)
            / ((self.sq / n - (self.sum / n).powi(2)).max(0.) / (n - 1.))
                .sqrt()
                .max(1e-15)
    }
}
fn observable(p: Pose) -> [f64; 8] {
    let r = rotation(p.orientation);
    [
        p.position[0],
        p.position[0].powi(2),
        p.position[1],
        p.position[2],
        r[0][0],
        r[0][0].powi(2),
        p.position[0] * r[1][1],
        r[2][0],
    ]
}

#[test]
fn posterior_transport_preserves_an_exact_mixture_and_expanded_trace_balances() -> Result<()> {
    let model = model()?;
    for (i, c) in [-0.6, 0., 0.9, 1.].into_iter().enumerate() {
        let map = map(c)?;
        let proposal = DockingProposal::new(
            model.clone(),
            DockingMethod::PosteriorInvolution,
            c,
            [0.; 3],
        )?;
        let mut starts = StdRng::seed_from_u64(912304);
        let mut rng = StdRng::seed_from_u64(773210 + i as u64);
        let mut changes = [Moment::default(); 8];
        let mut maximum_cancellation = 0_f64;
        let mut mapped = 0;
        for _ in 0..8000 {
            let source = usize::from(starts.random::<f64>() >= WEIGHTS[0]);
            let old = map.decode(source, normal(&mut starts))?;
            let (candidate, info) = proposal.propose(&mut rng, lab(old), &[anchor()])?;
            // This test concerns the Gaussian branch, selected independently of
            // the state. Uniform-branch physical correction is tested separately.
            if info["branch"] == "uniform" {
                continue;
            }
            let new = relative(candidate.expect("finite Gaussian draw"));
            let correction = info["log_reverse_forward"].as_f64().unwrap();
            let expanded = info["expanded_log_reverse_forward"].as_f64().unwrap();
            let residual = correction + log_g(&model, new)? - log_g(&model, old)?;
            assert!(
                residual.abs() < 2e-10,
                "target G must have unit acceptance: {residual}"
            );
            maximum_cancellation = maximum_cancellation.max((expanded - correction).abs());
            let trace: BasinTrace = serde_json::from_value(info["trace"].clone())?;
            let step = map.apply(old, &trace)?;
            let backward = map.apply(step.pose, &step.inverse_trace)?;
            assert!(norm(sub(old.position, backward.pose.position)) < 1e-10);
            let a = rotation(old.orientation);
            let b = rotation(backward.pose.orientation);
            assert!(
                a.iter()
                    .flatten()
                    .zip(b.iter().flatten())
                    .all(|(x, y)| (x - y).abs() < 1e-10)
            );
            for (m, (a, b)) in changes
                .iter_mut()
                .zip(observable(new).into_iter().zip(observable(old)))
            {
                m.add(a - b);
            }
            mapped += 1;
        }
        let max_z = changes.iter().map(|m| m.z(0.).abs()).fold(0., f64::max);
        eprintln!(
            "Posterior c={c}: {mapped} Gaussian draws, maximum paired z={max_z:.3}, trace cancellation={maximum_cancellation:.3e}"
        );
        assert!(mapped > 6800 && max_z < 5.5 && maximum_cancellation < 2e-10);
    }
    Ok(())
}

#[test]
fn source_labels_have_posterior_law_and_c0_is_branch_matched_independent_redraw() -> Result<()> {
    let model = model()?;
    let map = map(0.)?;
    let proposal = DockingProposal::new(
        model.clone(),
        DockingMethod::PosteriorInvolution,
        0.,
        [0.; 3],
    )?;
    let components = parameters()
        .into_iter()
        .map(|mut p| {
            p.weight = 1.;
            FrozenRelativePoseProposal::from_components_open(vec![p], 1.3, [20.; 3], 0.1, SHA, SHA)
        })
        .collect::<Result<Vec<_>>>()?;
    for start_label in 0..2 {
        let old = map.decode(start_label, [0.5, -0.3, 0.4, 0.2, -0.1, 0.3])?;
        let expected = (WEIGHTS[0].ln() + log_g(&components[0], old)? - log_g(&model, old)?).exp();
        let mut rng = StdRng::seed_from_u64(784110 + start_label as u64);
        let mut n = 0usize;
        let mut sources = 0usize;
        let mut targets = 0usize;
        let mut latent = [[Moment::default(); 2]; 6];
        for _ in 0..24000 {
            let (new, info) = proposal.propose(&mut rng, lab(old), &[anchor()])?;
            if info["branch"] == "uniform" {
                continue;
            }
            let trace: BasinTrace = serde_json::from_value(info["trace"].clone())?;
            let coordinates = map.encode(trace.target, relative(new.unwrap()))?;
            for k in 0..6 {
                assert!((coordinates[k] - trace.noise[k]).abs() < 1e-11);
                latent[k][0].add(coordinates[k]);
                latent[k][1].add(coordinates[k].powi(2));
            }
            n += 1;
            sources += usize::from(trace.source == 0);
            targets += usize::from(trace.target == 0);
        }
        for (observed, p, label) in [
            (sources, expected, "posterior source"),
            (targets, WEIGHTS[0], "independent destination"),
        ] {
            let z = (observed as f64 - n as f64 * p) / (n as f64 * p * (1. - p)).sqrt().max(1.);
            assert!(z.abs() < 5.5, "{label}: {z}");
        }
        for axis in latent {
            assert!(axis[0].z(0.).abs() < 5.5 && axis[1].z(1.).abs() < 5.5);
        }
        eprintln!(
            "Start {start_label}: source fraction {:.6}, expected {expected:.6}; target {:.6}",
            sources as f64 / n as f64,
            targets as f64 / n as f64
        );
    }
    Ok(())
}
