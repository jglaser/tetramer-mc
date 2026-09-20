//! Independent complete-native-region integration with geometric finite-volume covers.
//! An optional frozen Gaussian guide shares the same full-density correction.
use crate::{
    depletion::GateOptions,
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    overlap_weight::{self, OverlapEnvelope},
    proposal::{FrozenRelativePoseProposal, ProposalBranch},
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
};
use anyhow::{Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    f64::consts::PI,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NativeMetric {
    pub native_poses: Vec<Pose>,
    pub rigid_members: Vec<Pose>,
    pub member_error_scale: f64,
    pub angle_error_scale_deg: f64,
}

impl NativeMetric {
    pub fn q(&self, pose: Pose) -> f64 {
        self.native_poses
            .iter()
            .map(|reference| {
                let error = self
                    .rigid_members
                    .iter()
                    .map(|member| {
                        norm(sub(
                            pose.apply(member.position),
                            reference.apply(member.position),
                        ))
                    })
                    .fold(0., f64::max)
                    / self.member_error_scale;
                error.max(
                    relative_angle(pose.orientation, reference.orientation)
                        / self.angle_error_scale_deg.to_radians(),
                )
            })
            .fold(f64::INFINITY, f64::min)
    }
}

pub fn relative_angle(a: [f64; 4], b: [f64; 4]) -> f64 {
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    2. * (a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>().abs() / (na * nb))
        .clamp(0., 1.)
        .acos()
}

/// Stable SO(3) angle CDF numerator, including small caps.
pub fn theta_minus_sin(theta: f64) -> f64 {
    if theta.abs() < 0.05 {
        let s = theta * theta;
        theta
            * s
            * (1. / 6. + s * (-1. / 120. + s * (1. / 5040. + s * (-1. / 362880. + s / 39916800.))))
    } else {
        theta - theta.sin()
    }
}

pub fn inverse_angle_cdf(u: f64, cap: f64) -> f64 {
    let target = u * theta_minus_sin(cap);
    let (mut lo, mut hi) = (0., cap);
    for _ in 0..58 {
        let mid = 0.5 * (lo + hi);
        if theta_minus_sin(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeCover {
    pub reference: Pose,
    pub centroid: Vec3,
    pub member_covariance: Mat3,
    pub moment_trace: f64,
    pub lambda_max_upper: f64,
    pub l_lower: f64,
    pub ball_radius: f64,
    pub nominal_angle_cap: f64,
    pub angle_cap: f64,
    pub volume: f64,
}

impl NativeCover {
    pub fn new(metric: &NativeMetric) -> Result<Self> {
        ensure!(
            metric.native_poses.len() == 1,
            "Exactly one native reference required; covers around multiple references are not implemented"
        );
        ensure!(
            !metric.rigid_members.is_empty(),
            "Empty rigid-member metric"
        );
        for p in metric.native_poses.iter().chain(&metric.rigid_members) {
            p.validate()?;
        }
        let a = metric.member_error_scale;
        let nominal = metric.angle_error_scale_deg.to_radians();
        ensure!(
            a.is_finite() && a > 0. && nominal.is_finite() && nominal > 0. && nominal <= PI,
            "Invalid metric tolerances"
        );
        let n = metric.rigid_members.len() as f64;
        let centroid = metric
            .rigid_members
            .iter()
            .fold([0.; 3], |sum, p| add(sum, scale(p.position, 1. / n)));
        let mut covariance = [[0.; 3]; 3];
        let mut magnitude = 0.;
        for p in &metric.rigid_members {
            let d = sub(p.position, centroid);
            magnitude += dot(p.position, p.position) / n;
            for i in 0..3 {
                for j in 0..3 {
                    covariance[i][j] += d[i] * d[j] / n;
                }
            }
        }
        let trace = (0..3).map(|i| covariance[i][i]).sum::<f64>();
        let frobenius = covariance
            .iter()
            .flatten()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let row_bound = covariance
            .iter()
            .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0., f64::max);
        // Both norms bound lambda_max. Guard subtraction outward. If finite
        // bounds cannot be constructed, nominal angular support is always safe.
        let slack = 4096. * f64::EPSILON * (1. + magnitude) * n;
        let upper = frobenius.min(row_bound) + slack;
        let lower = if upper.is_finite() && trace.is_finite() && slack.is_finite() {
            (trace - slack - upper).max(0.)
        } else {
            0.
        };
        let geometric = if lower > 0. {
            2. * (a / (2. * lower.sqrt())).min(1.).asin()
        } else {
            PI
        };
        let cap = nominal.min(geometric);
        let volume = 4. * a.powi(3) / 3. * theta_minus_sin(cap);
        ensure!(
            volume.is_finite() && volume > 0.,
            "Invalid or underflowed cover volume"
        );
        ensure!(centroid.iter().all(|x| x.is_finite()), "Nonfinite centroid");
        Ok(Self {
            reference: metric.native_poses[0],
            centroid,
            member_covariance: covariance,
            moment_trace: trace,
            lambda_max_upper: if upper.is_finite() { upper } else { f64::MAX },
            l_lower: lower,
            ball_radius: a,
            nominal_angle_cap: nominal,
            angle_cap: cap,
            volume,
        })
    }

    pub fn sample(&self, rng: &mut StdRng) -> Pose {
        let axis = direction(rng);
        let theta = inverse_angle_cdf(rng.random(), self.angle_cap);
        let (s, c) = (0.5 * theta).sin_cos();
        let relative = rotation([c, s * axis[0], s * axis[1], s * axis[2]]);
        let r0 = rotation(self.reference.orientation);
        let r = matmul(r0, relative);
        let w = scale(
            direction(rng),
            self.ball_radius * rng.random::<f64>().cbrt(),
        );
        Pose {
            position: add(
                sub(
                    self.reference.position,
                    sub(matvec(r, self.centroid), matvec(r0, self.centroid)),
                ),
                w,
            ),
            orientation: quaternion(r),
        }
    }

    pub fn contains(&self, pose: Pose) -> bool {
        let w = add(
            sub(pose.position, self.reference.position),
            sub(
                matvec(rotation(pose.orientation), self.centroid),
                matvec(rotation(self.reference.orientation), self.centroid),
            ),
        );
        norm(w) <= self.ball_radius * (1. + 1e-12)
            && relative_angle(pose.orientation, self.reference.orientation)
                <= self.angle_cap + 1e-12
    }

    /// Membership for a probability density: no tolerance-expanded shell.
    /// The atan2 form retains small angular differences that acos can erase.
    pub fn contains_support(&self, pose: Pose) -> bool {
        let w = add(
            sub(pose.position, self.reference.position),
            sub(
                matvec(rotation(pose.orientation), self.centroid),
                matvec(rotation(self.reference.orientation), self.centroid),
            ),
        );
        let [a, b, c, d] = self.reference.orientation;
        let [e, f, g, h] = pose.orientation;
        let scalar = a * e + b * f + c * g + d * h;
        let vector = [
            a * f - b * e - c * h + d * g,
            a * g + b * h - c * e - d * f,
            a * h - b * g + c * f - d * e,
        ];
        let angle = 2. * norm(vector).atan2(scalar.abs());
        norm(w) <= self.ball_radius && angle <= self.angle_cap
    }
}

/// A normalized geometric proposal; its components do not redefine the target.
#[derive(Clone, Debug, Serialize)]
pub struct NativeCoverMixture {
    pub scales: Vec<f64>,
    pub weights: Vec<f64>,
    pub covers: Vec<NativeCover>,
}

impl NativeCoverMixture {
    pub fn new(base: &NativeCover, scales: Vec<f64>, mut weights: Vec<f64>) -> Result<Self> {
        ensure!(
            !scales.is_empty()
                && scales.iter().all(|s| s.is_finite() && *s > 0. && *s <= 1.)
                && scales.contains(&1.),
            "Cover scales must be in (0,1] and include 1 for complete target support"
        );
        if weights.is_empty() {
            weights = vec![1.; scales.len()];
        }
        ensure!(
            weights.len() == scales.len() && weights.iter().all(|w| w.is_finite() && *w > 0.),
            "Need one finite positive weight per cover"
        );
        let sum: f64 = weights.iter().sum();
        ensure!(sum.is_finite() && sum > 0., "Invalid weight sum");
        for weight in &mut weights {
            *weight /= sum;
            ensure!(*weight > 0., "Normalized weight underflow");
        }
        let mut covers = Vec::new();
        for &s in &scales {
            let mut cover = base.clone();
            if s != 1. {
                cover.ball_radius *= s;
                cover.angle_cap *= s;
                cover.nominal_angle_cap *= s;
                cover.volume =
                    4. * cover.ball_radius.powi(3) / 3. * theta_minus_sin(cover.angle_cap);
            }
            ensure!(
                cover.volume.is_finite() && cover.volume > 0.,
                "Underflowed cover volume"
            );
            covers.push(cover);
        }
        Ok(Self {
            scales,
            weights,
            covers,
        })
    }

    pub fn is_legacy(&self) -> bool {
        self.scales == [1.] && self.weights == [1.]
    }

    pub fn sample(&self, pose_rng: &mut StdRng, component_rng: &mut StdRng) -> (usize, Pose) {
        let mut selected = 0;
        if self.covers.len() > 1 {
            let u = component_rng.random::<f64>();
            let mut sum = 0.;
            selected = self.covers.len() - 1;
            for (index, weight) in self.weights.iter().enumerate() {
                sum += weight;
                if u < sum {
                    selected = index;
                    break;
                }
            }
        }
        (selected, self.covers[selected].sample(pose_rng))
    }

    pub fn log_density(&self, pose: Pose) -> f64 {
        self.covers
            .iter()
            .zip(&self.weights)
            .filter(|(cover, _)| cover.contains_support(pose))
            .map(|(cover, weight)| weight.ln() - cover.volume.ln())
            .fold(f64::NEG_INFINITY, log_add)
    }
}

/// A normalized optional guide, evaluated in the same laboratory measure as
/// the geometric cover. Its selected proposal anchor does not restrict the
/// physical neighbor list used by the hard and depletion tests.
pub struct NativeGuide {
    model: FrozenRelativePoseProposal,
    weight: f64,
    capture_center: Vec3,
    centered_anchor: Pose,
}

pub struct NativeGuideDraw {
    pub pose: Pose,
    pub branch: ProposalBranch,
    pub component: Option<usize>,
}

impl NativeGuide {
    pub fn new(
        model: FrozenRelativePoseProposal,
        weight: f64,
        capture_center: Vec3,
        anchor: Pose,
    ) -> Result<Self> {
        ensure!(
            !model.is_periodic(),
            "Native guide must be an open-space density"
        );
        ensure!(
            weight.is_finite() && weight > 0. && weight < 1.,
            "Guide weight must lie in (0,1)"
        );
        ensure!(
            capture_center.iter().all(|x| x.is_finite()),
            "Nonfinite guide cube center"
        );
        anchor.validate()?;
        Ok(Self {
            model,
            weight,
            capture_center,
            centered_anchor: Pose {
                position: sub(anchor.position, capture_center),
                orientation: anchor.orientation,
            },
        })
    }

    pub fn log_density(&self, pose: Pose) -> Result<f64> {
        self.model.log_density(
            &Pose {
                position: sub(pose.position, self.capture_center),
                orientation: pose.orientation,
            },
            &self.centered_anchor,
        )
    }

    pub fn full_log_density(&self, covers: &NativeCoverMixture, pose: Pose) -> Result<f64> {
        Ok(log_add(
            (1. - self.weight).ln() + covers.log_density(pose),
            self.weight.ln() + self.log_density(pose)?,
        ))
    }

    pub fn sample(&self, rng: &mut StdRng) -> Result<NativeGuideDraw> {
        // This is an independent draw. The dummy old pose is in the uniform
        // cube and is used only to satisfy the existing proposal interface.
        let dummy = Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        };
        let outcome = self.model.propose(rng, &[dummy, self.centered_anchor], 0)?;
        ensure!(
            outcome.candidate.is_some(),
            "Native guide numerical null: {:?}; no retry",
            outcome.null_reason
        );
        let centered = outcome.candidate.unwrap();
        Ok(NativeGuideDraw {
            pose: Pose {
                position: add(centered.position, self.capture_center),
                orientation: centered.orientation,
            },
            branch: outcome.branch,
            component: outcome.component_index,
        })
    }
}

fn direction(rng: &mut StdRng) -> Vec3 {
    loop {
        let x: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let n = norm(x);
        if n > 0. && n.is_finite() {
            return scale(x, 1. / n);
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeRegionOptions {
    pub config: PathBuf,
    pub out: PathBuf,
    pub samples: u64,
    pub seed: u64,
    pub cloud_replicates: usize,
    pub activity: Option<f64>,
    pub lambda_ratio: Option<f64>,
    pub cover_scales: Vec<f64>,
    pub cover_weights: Vec<f64>,
    pub model: Option<PathBuf>,
    pub model_weight: f64,
    pub model_uniform_probability: f64,
    pub model_anchor_index: usize,
}
#[derive(Deserialize)]
struct Config {
    shape: PathBuf,
    fixed_poses: Vec<Pose>,
    capture_center: Vec3,
    capture_radius: f64,
    depletant_radius: f64,
    reservoir_density: f64,
    poisson_lambda_ratio: f64,
    #[serde(default)]
    endpoint_gate: GateOptions,
    metadata: Value,
}
fn stream(seed: u64, draw: u64, cloud: usize, label: &str) -> StdRng {
    let mut h = Sha256::new();
    h.update(b"native-region-independent-v1");
    h.update(seed.to_le_bytes());
    h.update(draw.to_le_bytes());
    h.update((cloud as u64).to_le_bytes());
    h.update(label);
    StdRng::from_seed(h.finalize().into())
}
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    a.max(b) + (a.min(b) - a.max(b)).exp().ln_1p()
}
struct Moments {
    sum: f64,
    square: f64,
    max: f64,
    nonzero: u64,
}
impl Moments {
    fn new() -> Self {
        Self {
            sum: f64::NEG_INFINITY,
            square: f64::NEG_INFINITY,
            max: f64::NEG_INFINITY,
            nonzero: 0,
        }
    }
    fn add(&mut self, x: f64) {
        self.sum = log_add(self.sum, x);
        self.square = log_add(self.square, 2. * x);
        self.max = self.max.max(x);
        self.nonzero += 1;
    }
    fn value(&self, n: u64) -> Value {
        if self.nonzero == 0 {
            return json!({"unconditional_draws":n,"nonzero":0,"log_normalizer":null,"ess":0.});
        }
        let ess = (2. * self.sum - self.square).exp().min(n as f64);
        json!({"unconditional_draws":n,"nonzero":self.nonzero,"log_normalizer":self.sum-(n as f64).ln(),
            "ess":ess,"observed_relative_standard_error":if n>1 {Some(((n as f64/ess-1.)/(n-1)as f64).max(0.).sqrt())}else{None},
            "maximum_weight_fraction":(self.max-self.sum).exp(),"log_sum_weights":self.sum,"log_sum_squared_weights":self.square})
    }
}

pub fn run(options: NativeRegionOptions) -> Result<Value> {
    ensure!(
        options.samples > 0 && options.cloud_replicates > 0,
        "Positive draw/cloud counts required"
    );
    let raw = fs::read(&options.config)?;
    let mut cfg: Config = serde_json::from_slice(&raw)?;
    if cfg.shape.is_relative() {
        cfg.shape = options.config.parent().unwrap().join(&cfg.shape);
    }
    if let Some(z) = options.activity {
        cfg.reservoir_density = z;
    }
    if let Some(r) = options.lambda_ratio {
        cfg.poisson_lambda_ratio = r;
    }
    ensure!(
        cfg.capture_radius.is_finite()
            && cfg.capture_radius > 0.
            && cfg.capture_center.iter().all(|x| x.is_finite()),
        "Invalid capture"
    );
    ensure!(
        cfg.depletant_radius.is_finite()
            && cfg.depletant_radius >= 0.
            && cfg.reservoir_density.is_finite()
            && cfg.reservoir_density >= 0.
            && cfg.poisson_lambda_ratio.is_finite()
            && cfg.poisson_lambda_ratio > 0.,
        "Invalid bath"
    );
    cfg.endpoint_gate.validate()?;
    for p in &cfg.fixed_poses {
        p.validate()?;
    }
    let metric: NativeMetric = serde_json::from_value(cfg.metadata.clone())?;
    let cover = NativeCover::new(&metric)?;
    let mixture = NativeCoverMixture::new(
        &cover,
        options.cover_scales.clone(),
        options.cover_weights.clone(),
    )?;
    let shape_raw = fs::read(&cfg.shape)?;
    let shape_sha = hash_bytes(&shape_raw);
    let model_raw = options.model.as_ref().map(fs::read).transpose()?;
    let guide = if let Some(raw_model) = &model_raw {
        ensure!(
            options.model_anchor_index < cfg.fixed_poses.len(),
            "Guide anchor index outside physical neighbor list"
        );
        let model = FrozenRelativePoseProposal::from_json_str_open(
            std::str::from_utf8(raw_model)?,
            [2. * cfg.capture_radius; 3],
            options.model_uniform_probability,
            &shape_sha,
        )?;
        Some(NativeGuide::new(
            model,
            options.model_weight,
            cfg.capture_center,
            cfg.fixed_poses[options.model_anchor_index],
        )?)
    } else {
        None
    };
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?;
    let env = Environment {
        tree: &tree,
        fixed: cfg.fixed_poses.iter().copied().map(Placed::new).collect(),
        labels: cfg
            .fixed_poses
            .iter()
            .enumerate()
            .map(|(j, _)| (j, [0; 3]))
            .collect(),
        rd: cfg.depletant_radius,
    };
    for i in 0..env.fixed.len() {
        for j in 0..i {
            ensure!(
                !tree.overlaps(&env.fixed[i], &env.fixed[j]),
                "Fixed neighbors overlap"
            );
        }
    }
    ensure!(!options.out.exists(), "Refuse existing output directory");
    fs::create_dir_all(options.out.join("provenance"))?;
    fs::write(options.out.join("provenance/config.json"), &raw)?;
    fs::write(options.out.join("provenance/shape.json"), &shape_raw)?;
    if let Some(raw_model) = &model_raw {
        fs::write(options.out.join("provenance/guide-model.json"), raw_model)?;
    }
    let source = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(options.out.join("provenance/source-bundle.json"), source)?;
    save(&options.out.join("cover.json"), &cover)?;
    save(&options.out.join("cover-mixture.json"), &mixture)?;
    let lambda = if cfg.reservoir_density > 0. {
        cfg.poisson_lambda_ratio * cfg.reservoir_density
    } else {
        1.
    };
    ensure!(lambda.is_finite() && lambda > 0., "Invalid intensity");
    let mut manifest = json!({"schema":if guide.is_some(){3}else{2},"samples":options.samples,"seed":options.seed,
        "cloud_replicates":options.cloud_replicates,"lambda":lambda,"lambda_ratio":cfg.poisson_lambda_ratio,"activity":cfg.reservoir_density,
        "depletant_radius":cfg.depletant_radius,"config_sha256":hash_bytes(&raw),"shape_sha256":shape_sha,
        "executable_sha256":hash_file(&std::env::current_exe()?)?,"source_bundle_sha256":hash_bytes(source.as_bytes()),
        "proposal":"Normalized geometric covers with an optional frozen Gaussian/cube guide; original target unchanged",
        "estimator":"1/N sum hard*capture*(original_q<=1)*mean_independent_cloud_W/full_mixture_density; all invalid draws zero",
        "numerical_scope":"Analytically conservative spectral-norm bound with FP64 outward guard; not formal interval arithmetic",
        "cover":cover,"cover_mixture":mixture,"metric":metric});
    if let Some(raw_model) = &model_raw {
        manifest["guide"] = json!({"model_sha256":hash_bytes(raw_model),"weight":options.model_weight,
            "uniform_probability":options.model_uniform_probability,"anchor_index":options.model_anchor_index,
            "anchor_pose":cfg.fixed_poses[options.model_anchor_index],"cube_lengths":([2. * cfg.capture_radius;3]),
            "capture_center":cfg.capture_center});
    }
    save(&options.out.join("manifest.json"), &manifest)?;
    let mut output = BufWriter::new(File::create(options.out.join("samples.jsonl"))?);
    let start = Instant::now();
    let cpu = cpu_seconds();
    let (mut total, mut core, mut shell) = (Moments::new(), Moments::new(), Moments::new());
    let (mut hard, mut hard_core, mut hard_shell) =
        (Moments::new(), Moments::new(), Moments::new());
    let mut cross_sum = f64::NEG_INFINITY;
    let mut component_draws = vec![0u64; mixture.covers.len()];
    let mut guide_component_draws =
        vec![0u64; guide.as_ref().map_or(0, |g| g.model.component_count())];
    let (mut cover_draws, mut guide_draws, mut guide_uniform_draws) = (0u64, 0u64, 0u64);
    let (mut q_rejected, mut capture_rejected, mut hard_rejected, mut raw_points) =
        (0u64, 0u64, 0u64, 0u64);
    for draw in 0..options.samples {
        let use_guide = guide
            .as_ref()
            .is_some_and(|g| stream(options.seed, draw, 0, "family").random::<f64>() < g.weight);
        let (component, guide_branch, guide_component, pose) = if use_guide {
            let selected =
                guide
                    .as_ref()
                    .unwrap()
                    .sample(&mut stream(options.seed, draw, 0, "guide"))?;
            guide_draws += 1;
            if let Some(index) = selected.component {
                guide_component_draws[index] += 1;
            }
            if selected.branch == ProposalBranch::Uniform {
                guide_uniform_draws += 1;
            }
            (
                None,
                Some(selected.branch),
                selected.component,
                selected.pose,
            )
        } else {
            let (index, pose) = mixture.sample(
                &mut stream(options.seed, draw, 0, "pose"),
                &mut stream(options.seed, draw, 0, "component"),
            );
            component_draws[index] += 1;
            cover_draws += 1;
            (Some(index), None, None, pose)
        };
        let log_density = if let Some(g) = &guide {
            g.full_log_density(&mixture, pose)?
        } else {
            mixture.log_density(pose)
        };
        ensure!(
            log_density.is_finite(),
            "Generated pose outside mixture support; no retry"
        );
        let q = metric.q(pose);
        ensure!(q.is_finite(), "Nonfinite q");
        let reason = if q > 1. {
            q_rejected += 1;
            Some("q")
        } else if norm(sub(pose.position, cfg.capture_center)) > cfg.capture_radius {
            capture_rejected += 1;
            Some("capture")
        } else if !env.hard_valid(pose) {
            hard_rejected += 1;
            Some("hard")
        } else {
            None
        };
        let mut row = if let Some(reason) = reason {
            json!({"draw":draw,"q":q,"zero":reason})
        } else {
            let envelope = OverlapEnvelope::build(&env, pose, cfg.endpoint_gate)?;
            let mut logs = Vec::new();
            let mut counts = Vec::new();
            let mut cloud_raw = Vec::new();
            for cloud in 0..options.cloud_replicates {
                let w = overlap_weight::sample_with_envelope(
                    &mut stream(options.seed, draw, cloud, "cloud"),
                    &env,
                    pose,
                    lambda,
                    cfg.reservoir_density,
                    &envelope,
                )?;
                logs.push(w.log_weight);
                counts.push(w.overlap_points);
                cloud_raw.push(w.raw_points);
                raw_points += w.raw_points;
            }
            let log_mean = logs.iter().copied().fold(f64::NEG_INFINITY, log_add)
                - (options.cloud_replicates as f64).ln();
            let log_hard = -log_density;
            let log_weight = log_hard + log_mean;
            total.add(log_weight);
            hard.add(log_hard);
            cross_sum = log_add(cross_sum, log_weight + log_hard);
            if q <= 0.8 {
                core.add(log_weight);
                hard_core.add(log_hard);
            } else {
                shell.add(log_weight);
                hard_shell.add(log_hard);
            };
            json!({"draw":draw,"q":q,"pose":pose,"log_importance_weight":log_weight,"log_boltzmann_mean":log_mean,
                "cloud_log_weights":logs,"cloud_overlap_counts":counts,"cloud_raw_points":cloud_raw,
                "lower_volume":envelope.lower_volume,"upper_volume":envelope.upper_volume()})
        };
        if guide.is_some() || !mixture.is_legacy() {
            row["pose"] = json!(pose);
            row["proposal_component"] = json!(component);
            row["log_proposal_density"] = json!(log_density);
            if reason.is_none() {
                row["log_hard_weight"] = json!(-log_density);
            }
        }
        if guide.is_some() {
            row["proposal_family"] = json!(if use_guide { "guide" } else { "cover" });
            row["guide_branch"] = json!(guide_branch);
            row["guide_component"] = json!(guide_component);
        }
        serde_json::to_writer(&mut output, &row)?;
        writeln!(output)?;
        if (draw + 1) % 4096 == 0 || draw + 1 == options.samples {
            output.flush()?;
            save(
                &options.out.join("progress.json"),
                &json!({"completed":draw+1,"requested":options.samples,"valid":total.nonzero,
                "q_rejected":q_rejected,"capture_rejected":capture_rejected,"hard_rejected":hard_rejected,"wall_seconds":start.elapsed().as_secs_f64()}),
            )?;
        }
    }
    output.flush()?;
    let result = json!({"complete":true,"samples":options.samples,"native":total.value(options.samples),"native_core":core.value(options.samples),
        "native_shell":shell.value(options.samples),"q_rejected":q_rejected,"capture_rejected":capture_rejected,"hard_rejected":hard_rejected,
        "hard_native":hard.value(options.samples),"hard_native_core":hard_core.value(options.samples),"hard_native_shell":hard_shell.value(options.samples),
        "physical_hard_log_cross_sum":if cross_sum.is_finite(){Some(cross_sum)}else{None},"component_draws":component_draws,
        "family_draws":{"cover":cover_draws,"guide":guide_draws},"guide_component_draws":guide_component_draws,"guide_uniform_draws":guide_uniform_draws,
        "cover_volume":cover.volume,"raw_cloud_points":raw_points,"wall_seconds":start.elapsed().as_secs_f64(),"cpu_seconds":cpu_seconds()-cpu,
        "samples_sha256":hash_file(&options.out.join("samples.jsonl"))?,
        "uncertainty":"Independent fixed-N estimator; observed ESS and uncertainty do not certify absence of unresolved within-region weight concentration"});
    save(&options.out.join("summary.json"), &result)?;
    Ok(result)
}
