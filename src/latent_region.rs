//! Direct integration of a frozen six-dimensional chart ball or shell.
//!
//! Uniform latent ball draws have density 1/V6. The exact pose Jacobian is
//! det(L)/(ell^3*pi^2*(1+|c|^2)^2), relative to center volume and normalized
//! SO(3) Haar measure. Thus V6 H I_region J W is unbiased for THIS REGION'S
//! physical mass when E[W|pose]=exp(z*C). Invalid draws remain zero; no retry.
use crate::{
    docking::DockingConfig,
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    overlap_weight::{self, OverlapEnvelope},
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    f64::consts::PI,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Clone, Debug)]
pub struct LatentRegionOptions {
    pub config: PathBuf,
    pub region: PathBuf,
    pub out: PathBuf,
    pub samples: u64,
    pub seed: u64,
    pub cloud_replicates: usize,
    /// Auxiliary intensity only: does not alter the frozen physical activity.
    pub lambda_ratio: f64,
}

#[derive(Clone, Deserialize)]
struct Metric {
    native_poses: Vec<Pose>,
    rigid_members: Vec<Pose>,
    member_error_scale: f64,
    angle_error_scale_deg: f64,
}
impl Metric {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.native_poses.is_empty() && !self.rigid_members.is_empty(),
            "Empty registration definition"
        );
        ensure!(
            self.member_error_scale.is_finite()
                && self.member_error_scale > 0.
                && self.angle_error_scale_deg.is_finite()
                && self.angle_error_scale_deg > 0.,
            "Invalid registration scales"
        );
        for pose in self.native_poses.iter().chain(&self.rigid_members) {
            pose.validate()?;
        }
        Ok(())
    }
    fn q(&self, pose: Pose) -> f64 {
        self.native_poses
            .iter()
            .map(|native| {
                let displacement = self
                    .rigid_members
                    .iter()
                    .map(|member| {
                        norm(sub(
                            pose.apply(member.position),
                            native.apply(member.position),
                        ))
                    })
                    .fold(0., f64::max)
                    / self.member_error_scale;
                let a = rotation(pose.orientation);
                let b = rotation(native.orientation);
                let delta = quaternion(matmul(a, transpose(b)));
                let angle = 2. * delta[0].abs().clamp(0., 1.).acos();
                displacement.max(angle / self.angle_error_scale_deg.to_radians())
            })
            .fold(f64::INFINITY, f64::min)
    }
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

fn q_in_interval(
    q: f64,
    minimum: f64,
    maximum: Option<f64>,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> bool {
    q.is_finite()
        && (if lower_inclusive {
            q >= minimum
        } else {
            q > minimum
        })
        && maximum.is_none_or(|maximum| {
            if upper_inclusive {
                q <= maximum
            } else {
                q < maximum
            }
        })
}

#[derive(Default)]
struct Moments {
    count: u64,
    log_sum: Option<f64>,
    log_sum2: Option<f64>,
    log_max: Option<f64>,
}
impl Moments {
    fn add(&mut self, w: f64) {
        self.count += 1;
        self.log_sum = Some(log_add(self.log_sum.unwrap_or(f64::NEG_INFINITY), w));
        self.log_sum2 = Some(log_add(self.log_sum2.unwrap_or(f64::NEG_INFINITY), 2. * w));
        self.log_max = Some(self.log_max.unwrap_or(f64::NEG_INFINITY).max(w));
    }
    fn value(&self, n: u64) -> Value {
        let Some(sum) = self.log_sum else {
            return json!({"draws":n,"nonzero":0,"logQ":null,"ess":0.});
        };
        let ess = (2. * sum - self.log_sum2.unwrap()).exp().min(n as f64);
        json!({"draws":n,"nonzero":self.count,"logQ":sum-(n as f64).ln(),"ess":ess,
            "relative_se":if n>1 {Some(((n as f64/ess-1.).max(0.)/(n-1) as f64).sqrt())} else {None},
            "max_fraction":(self.log_max.unwrap()-sum).exp(),
            "scope":"Observed-sample regional uncertainty; does not establish global mass or unobserved tails"})
    }
}

fn stream(seed: u64, draw: u64, cloud: usize, role: &str) -> StdRng {
    let mut h = Sha256::new();
    h.update(b"tetramer-uniform-latent-region-v1");
    h.update(seed.to_le_bytes());
    h.update(draw.to_le_bytes());
    h.update((cloud as u64).to_le_bytes());
    h.update(role.as_bytes());
    StdRng::from_seed(h.finalize().into())
}

struct Chart {
    lower: [[f64; 6]; 6],
    mean: [f64; 6],
    anchor_position: Vec3,
    anchor_rotation: Mat3,
    fixed: Pose,
    ell: f64,
    log_det: f64,
}
impl Chart {
    fn new(region: &Value, shape_hash: &str, capture_radius: f64, fixed: Pose) -> Result<Self> {
        let raw = serde_json::to_string(&region["gaussian_chart"])?;
        // Reuse the atlas validator for proper rotations, finite coordinates,
        // symmetric positive definite covariance and exact shape identity.
        let model = FrozenRelativePoseProposal::from_json_str_open(
            &raw,
            [2. * capture_radius; 3],
            0.1,
            shape_hash,
        )?;
        let parameters = model.component_parameters();
        ensure!(
            parameters.len() == 1,
            "Frozen region must specify exactly one Gaussian chart"
        );
        let p = &parameters[0];
        let ell = region["gaussian_chart"]["angular_length"]
            .as_f64()
            .context("Missing angular length")?;
        let mut lower = [[0.; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let remainder = 0.5 * (p.covariance[i][j] + p.covariance[j][i])
                    - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                lower[i][j] = if i == j {
                    ensure!(
                        remainder.is_finite() && remainder > 0.,
                        "Nonpositive covariance pivot"
                    );
                    remainder.sqrt()
                } else {
                    remainder / lower[j][j]
                };
            }
        }
        let log_det = (0..6).map(|i| lower[i][i].ln()).sum();
        Ok(Self {
            lower,
            mean: p.mean,
            anchor_position: p.anchor_position,
            anchor_rotation: p.anchor_rotation,
            fixed,
            ell,
            log_det,
        })
    }
    fn decode(&self, u: [f64; 6]) -> (Pose, f64) {
        let x: [f64; 6] = std::array::from_fn(|i| {
            self.mean[i] + (0..=i).map(|j| self.lower[i][j] * u[j]).sum::<f64>()
        });
        let c = [x[3] / self.ell, x[4] / self.ell, x[5] / self.ell];
        let relative_t = add(self.anchor_position, [x[0], x[1], x[2]]);
        let relative_r = matmul(cayley(c), self.anchor_rotation);
        let pose = Pose {
            position: self.fixed.apply(relative_t),
            orientation: quaternion(matmul(rotation(self.fixed.orientation), relative_r)),
        };
        let log_jacobian =
            self.log_det - 3. * self.ell.ln() - 2. * PI.ln() - 2. * dot(c, c).ln_1p();
        (pose, log_jacobian)
    }
    fn encode(&self, pose: Pose) -> Result<[f64; 6]> {
        let relative_t = self.fixed.inverse(pose.position);
        let relative_r = matmul(
            transpose(rotation(self.fixed.orientation)),
            rotation(pose.orientation),
        );
        let q = quaternion(matmul(relative_r, transpose(self.anchor_rotation)));
        ensure!(q[0].abs() > 0., "Decoded pose lies on Cayley seam");
        let t = sub(relative_t, self.anchor_position);
        let x = [
            t[0],
            t[1],
            t[2],
            self.ell * q[1] / q[0],
            self.ell * q[2] / q[0],
            self.ell * q[3] / q[0],
        ];
        let mut u = [0.; 6];
        for i in 0..6 {
            u[i] = (x[i] - self.mean[i] - (0..i).map(|j| self.lower[i][j] * u[j]).sum::<f64>())
                / self.lower[i][i];
        }
        ensure!(
            u.iter().all(|x| x.is_finite()),
            "Nonfinite backmapped latent"
        );
        Ok(u)
    }
}

pub fn run(options: LatentRegionOptions) -> Result<Value> {
    ensure!(
        options.samples > 0 && options.cloud_replicates > 0,
        "Positive sample and cloud counts required"
    );
    ensure!(
        options.lambda_ratio.is_finite() && options.lambda_ratio > 0.,
        "Invalid auxiliary lambda ratio"
    );
    let config_raw = fs::read(&options.config)?;
    let mut cfg: DockingConfig = serde_json::from_slice(&config_raw)?;
    cfg.validate()?;
    ensure!(
        cfg.target_region.is_none(),
        "latent-region-normalizer does not implement DockingConfig.target_region; use a configuration without that docking-only constraint and the explicit frozen region definition"
    );
    if cfg.shape.is_relative() {
        cfg.shape = options
            .config
            .parent()
            .context("Missing config parent")?
            .join(&cfg.shape);
    }
    let region_raw = fs::read(&options.region)?;
    let region: Value = serde_json::from_slice(&region_raw)?;
    let fixed: Pose = serde_json::from_value(region["fixed_neighbor"].clone())?;
    let physical_fixed: Vec<Pose> = if let Some(value) = region.get("physical_fixed_neighbors") {
        serde_json::from_value(value.clone())?
    } else {
        vec![fixed]
    };
    ensure!(
        cfg.fixed_poses == physical_fixed && physical_fixed.contains(&fixed),
        "Frozen physical neighbors or chart anchor differ from configuration"
    );
    ensure!(
        region["capture_center"] == json!(cfg.capture_center)
            && region["capture_radius"] == json!(cfg.capture_radius),
        "Frozen capture domain mismatch"
    );
    ensure!(
        region["activity"] == json!(cfg.reservoir_density)
            && region["depletant_radius"] == json!(cfg.depletant_radius),
        "Frozen physical bath mismatch"
    );
    ensure!(
        region["physical_metric"] == cfg.metadata,
        "Frozen registration metric mismatch"
    );
    let metric: Metric = serde_json::from_value(cfg.metadata.clone())?;
    metric.validate()?;
    let radius = region["mahalanobis_radius"]
        .as_f64()
        .context("Missing frozen latent radius")?;
    let minimum_q = region["minimum_original_q"]
        .as_f64()
        .context("Missing original q cutoff")?;
    let inner_radius = region
        .get("minimum_mahalanobis_radius")
        .map(|value| value.as_f64().context("Invalid inner latent radius"))
        .transpose()?
        .unwrap_or(0.);
    let maximum_q = region
        .get("maximum_original_q")
        .map(|value| value.as_f64().context("Invalid maximum original q"))
        .transpose()?;
    let minimum_q_inclusive = region
        .get("minimum_original_q_inclusive")
        .map(|value| {
            value
                .as_bool()
                .context("Invalid minimum q endpoint inclusion")
        })
        .transpose()?
        .unwrap_or(true);
    let maximum_q_inclusive = region
        .get("maximum_original_q_inclusive")
        .map(|value| {
            value
                .as_bool()
                .context("Invalid maximum q endpoint inclusion")
        })
        .transpose()?
        .unwrap_or(true);
    ensure!(
        radius.is_finite()
            && radius > 0.
            && minimum_q.is_finite()
            && minimum_q >= 0.
            && inner_radius.is_finite()
            && inner_radius >= 0.
            && inner_radius < radius
            && maximum_q.is_none_or(|q| q.is_finite() && q >= minimum_q),
        "Invalid frozen region"
    );
    // Keep the exact legacy arithmetic when the inner radius is zero. In a
    // shell, the scaled sixth power is uniform and its volume is the difference
    // of two six-ball volumes. expm1 avoids cancellation for narrow shells.
    let shell_fraction = if inner_radius == 0. {
        1.
    } else {
        -(6. * ((inner_radius - radius) / radius).ln_1p()).exp_m1()
    };
    ensure!(
        shell_fraction.is_finite() && shell_fraction > 0.,
        "Unresolved shell volume"
    );
    let mut log_volume = 3. * PI.ln() + 6. * radius.ln() - 6_f64.ln();
    if inner_radius > 0. {
        log_volume += shell_fraction.ln();
    }
    let shape_raw = fs::read(&cfg.shape)?;
    let shape_hash = hash_bytes(&shape_raw);
    ensure!(
        region["shape_sha256"] == json!(shape_hash),
        "Frozen region shape mismatch"
    );
    let chart = Chart::new(&region, &shape_hash, cfg.capture_radius, fixed)?;
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?;
    let env = Environment {
        tree: &tree,
        fixed: physical_fixed.iter().copied().map(Placed::new).collect(),
        labels: physical_fixed
            .iter()
            .enumerate()
            .map(|(i, _)| (i, [0; 3]))
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
    let lambda = if cfg.reservoir_density > 0. {
        options.lambda_ratio * cfg.reservoir_density
    } else {
        1.
    };
    ensure!(
        lambda.is_finite() && lambda > 0.,
        "Invalid Poisson intensity"
    );
    if options.out.exists() {
        ensure!(
            fs::read_dir(&options.out)?.next().is_none(),
            "Output must be empty"
        );
    }
    fs::create_dir_all(options.out.join("provenance"))?;
    for (name, bytes) in [
        ("input-config.json", config_raw.as_slice()),
        ("region.json", region_raw.as_slice()),
        ("shape.json", shape_raw.as_slice()),
    ] {
        fs::write(options.out.join("provenance").join(name), bytes)?;
    }
    let source = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(options.out.join("provenance/source-bundle.json"), source)?;
    save(&options.out.join("config.json"), &cfg)?;
    let extended = inner_radius > 0.
        || maximum_q.is_some()
        || region.get("physical_fixed_neighbors").is_some()
        || !minimum_q_inclusive
        || !maximum_q_inclusive;
    let mut manifest = json!({"schema":if extended {"uniform-latent-region-normalizer-v2"} else {"uniform-latent-region-normalizer-v1"},"samples":options.samples,"seed":options.seed,
        "cloud_replicates":options.cloud_replicates,"activity":cfg.reservoir_density,"lambda":lambda,"lambda_ratio":options.lambda_ratio,
        "config_sha256":hash_bytes(&config_raw),"region_sha256":hash_bytes(&region_raw),"shape_sha256":shape_hash,
        "source_bundle_sha256":hash_bytes(source.as_bytes()),"executable_sha256":hash_file(&std::env::current_exe()?)?,
        "latent_radius":radius,"log_latent_ball_volume":log_volume,"minimum_original_q":minimum_q,
        "target":"REGION ONLY: integral of H_capture H_hard I(q>=q_min) exp(z*C) over the frozen latent ellipsoid; center volume times normalized SO(3) Haar",
        "estimator":"V6 * mean over all unconditional uniform6-ball draws of H Iq J * independent-cloud average W; invalid draws zero",
        "jacobian":"det(L)/(ell^3*pi^2*(1+|Cayley|^2)^2)","inference":"No global normalizer or global basin coverage claim"});
    if extended {
        manifest["minimum_latent_radius"] = json!(inner_radius);
        manifest["maximum_original_q"] = json!(maximum_q);
        manifest["minimum_original_q_inclusive"] = json!(minimum_q_inclusive);
        manifest["maximum_original_q_inclusive"] = json!(maximum_q_inclusive);
        manifest["physical_fixed_neighbors"] = json!(physical_fixed);
        manifest["chart_anchor"] = json!(fixed);
        manifest["log_latent_shell_volume"] = json!(log_volume);
        manifest["target"] = json!(
            "REGION ONLY: full physical-neighbor union, frozen latent shell and declared original-q endpoint inclusions; capture/hard-invalid draws zero"
        );
        manifest["estimator"] = json!(
            "V6_shell * mean over unconditional uniform6-shell draws of H Iq J * independent-cloud average W"
        );
    }
    save(&options.out.join("manifest.json"), &manifest)?;
    let start = Instant::now();
    let cpu_start = cpu_seconds();
    let mut writer = BufWriter::new(File::create(options.out.join("samples.jsonl"))?);
    let mut weighted = Moments::default();
    let mut hard = Moments::default();
    let mut capture_rejected = 0;
    let mut hard_rejected = 0;
    let mut region_rejected = 0;
    let mut maximum_backmap_error = 0_f64;
    let mut raw_points = 0_u64;
    for draw in 0..options.samples {
        let mut rng = stream(options.seed, draw, 0, "latent");
        let direction: [f64; 6] = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
        let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        ensure!(
            norm.is_finite() && norm > 0.,
            "Invalid normal direction (stop; never silently redraw)"
        );
        let uniform = rng.random::<f64>();
        let radial = if inner_radius == 0. {
            radius * uniform.powf(1. / 6.)
        } else {
            radius * (1. - shell_fraction * (1. - uniform)).powf(1. / 6.)
        };
        let latent = direction.map(|x| x * radial / norm);
        let (pose, log_jacobian) = chart.decode(latent);
        pose.validate()?;
        let backmap = chart.encode(pose)?;
        let backmap_radius = backmap.iter().map(|x| x * x).sum::<f64>().sqrt();
        let error = latent
            .iter()
            .zip(backmap)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f64, f64::max);
        maximum_backmap_error = maximum_backmap_error.max(error);
        ensure!(
            error <= 2e-7 * (1. + radius) && backmap_radius <= radius + 2e-7 * (1. + radius),
            "Chart inverse failed; stop instead of censoring a draw"
        );
        ensure!(log_jacobian.is_finite(), "Nonfinite physical Jacobian");
        let q = metric.q(pose);
        ensure!(q.is_finite(), "Invalid physical q");
        let capture_valid = cfg.contains(pose);
        let hard_valid = capture_valid && env.hard_valid(pose);
        let region_valid = q_in_interval(
            q,
            minimum_q,
            maximum_q,
            minimum_q_inclusive,
            maximum_q_inclusive,
        );
        capture_rejected += u64::from(!capture_valid);
        hard_rejected += u64::from(capture_valid && !hard_valid);
        region_rejected += u64::from(hard_valid && !region_valid);
        let mut clouds = Vec::new();
        let mut log_weight = None;
        let mut log_hard_weight = None;
        if hard_valid && region_valid {
            let envelope = OverlapEnvelope::build(&env, pose, cfg.endpoint_gate)?;
            let mut log_cloud_sum = f64::NEG_INFINITY;
            for cloud in 0..options.cloud_replicates {
                let w = overlap_weight::sample_with_envelope(
                    &mut stream(options.seed, draw, cloud, "cloud"),
                    &env,
                    pose,
                    lambda,
                    cfg.reservoir_density,
                    &envelope,
                )?;
                raw_points += w.raw_points;
                log_cloud_sum = log_add(log_cloud_sum, w.log_weight);
                clouds.push(w);
            }
            let h = log_volume + log_jacobian;
            let w = h + log_cloud_sum - (options.cloud_replicates as f64).ln();
            ensure!(h.is_finite() && w.is_finite(), "Invalid regional weight");
            weighted.add(w);
            hard.add(h);
            log_weight = Some(w);
            log_hard_weight = Some(h);
        }
        serde_json::to_writer(
            &mut writer,
            &json!({"draw":draw,"latent":latent,"latent_radius":radial,
            "pose":pose,"backmapped_latent":backmap,"backmapped_radius":backmap_radius,
            "log_physical_jacobian":log_jacobian,"physical_jacobian":log_jacobian.exp(),"q":q,
            "capture_valid":capture_valid,"hard_valid":hard_valid,"region_valid":region_valid,
            "log_importance_weight":log_weight,"log_hard_weight":log_hard_weight,"clouds":clouds}),
        )?;
        writer.write_all(b"\n")?;
        if (draw + 1) % 100 == 0 || draw + 1 == options.samples {
            writer.flush()?;
            save(
                &options.out.join("progress.json"),
                &json!({"completed_draws":draw+1,"requested_draws":options.samples,
                "nonzero":weighted.count,"sampler_cpu_seconds":cpu_seconds()-cpu_start,"complete":draw+1==options.samples}),
            )?;
        }
    }
    writer.flush()?;
    let mut summary = json!({"complete":true,"samples":options.samples,"estimates":{"region":weighted.value(options.samples),"hard_region":hard.value(options.samples)},
        "capture_rejected":capture_rejected,"hard_rejected":hard_rejected,"region_rejected":region_rejected,
        "maximum_backmap_error":maximum_backmap_error,"raw_points":raw_points,
        "sampler_cpu_seconds":cpu_seconds()-cpu_start,"wall_seconds":start.elapsed().as_secs_f64(),"manifest":manifest});
    if extended {
        summary["samples_sha256"] = json!(hash_file(&options.out.join("samples.jsonl"))?);
    }
    save(&options.out.join("summary.json"), &summary)?;
    Ok(summary)
}

#[cfg(test)]
mod q_window_tests {
    use super::q_in_interval;

    #[test]
    fn exact_open_closed_and_unbounded_endpoints() {
        assert!(!q_in_interval(1., 1., Some(2.), false, false));
        assert!(q_in_interval(1.5, 1., Some(2.), false, false));
        assert!(!q_in_interval(2., 1., Some(2.), false, false));
        assert!(q_in_interval(1., 1., Some(2.), true, false));
        assert!(q_in_interval(2., 1., Some(2.), false, true));
        assert!(q_in_interval(20., 1., None, true, true));
        assert!(!q_in_interval(f64::NAN, 1., None, true, true));
        assert!(!q_in_interval(f64::INFINITY, 1., None, true, true));
    }
}
