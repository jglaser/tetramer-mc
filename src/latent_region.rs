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
    path::{Path, PathBuf},
    time::Instant,
};

/// A frozen guide changes only the latent proposal, never the physical region.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportanceFile {
    schema: String,
    region_sha256: String,
    defensive_uniform_shell_probability: f64,
    gaussian_components: Vec<ImportanceComponentFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportanceComponentFile {
    weight: f64,
    mean: [f64; 6],
    covariance: [[f64; 6]; 6],
}

struct ImportanceComponent {
    weight: f64,
    mean: [f64; 6],
    lower: [[f64; 6]; 6],
    log_normalizer: f64,
}

struct ImportanceGuide {
    alpha: f64,
    components: Vec<ImportanceComponent>,
}

impl ImportanceGuide {
    fn from_bytes(raw: &[u8], region_hash: &str) -> Result<Self> {
        let parsed: ImportanceFile = serde_json::from_slice(raw)?;
        ensure!(
            parsed.schema == "defensive-latent-shell-guide-v1",
            "Unknown latent guide schema"
        );
        ensure!(
            parsed.region_sha256 == region_hash,
            "Latent guide targets another frozen region"
        );
        let alpha = parsed.defensive_uniform_shell_probability;
        ensure!(
            alpha.is_finite() && alpha > 0. && alpha <= 1.,
            "Uniform defensive mass must lie in (0,1]"
        );
        ensure!(
            alpha == 1. || !parsed.gaussian_components.is_empty(),
            "Missing latent Gaussian components"
        );
        let total: f64 = parsed.gaussian_components.iter().map(|c| c.weight).sum();
        ensure!(
            parsed.gaussian_components.is_empty() || (total.is_finite() && total > 0.),
            "Invalid Gaussian mixture mass"
        );
        let mut components = Vec::new();
        for c in parsed.gaussian_components {
            ensure!(
                c.weight.is_finite() && c.weight > 0. && c.mean.iter().all(|x| x.is_finite()),
                "Invalid latent Gaussian mass or mean"
            );
            let mut lower = [[0.; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    let a = c.covariance[i][j];
                    let b = c.covariance[j][i];
                    ensure!(
                        a.is_finite()
                            && b.is_finite()
                            && (a - b).abs() <= 1e-12 * (1. + a.abs().max(b.abs())),
                        "Nonfinite or asymmetric latent covariance"
                    );
                }
                for j in 0..=i {
                    let value = 0.5 * (c.covariance[i][j] + c.covariance[j][i])
                        - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                    lower[i][j] = if i == j {
                        ensure!(
                            value.is_finite() && value > 0.,
                            "Latent covariance is not positive definite"
                        );
                        value.sqrt()
                    } else {
                        value / lower[j][j]
                    };
                }
            }
            let log_normalizer =
                -3. * (2. * PI).ln() - (0..6).map(|i| lower[i][i].ln()).sum::<f64>();
            ensure!(
                log_normalizer.is_finite(),
                "Unrepresentable Gaussian density"
            );
            ensure!(
                c.weight / total > 0.,
                "Unrepresentable normalized mixture mass"
            );
            components.push(ImportanceComponent {
                weight: c.weight / total,
                mean: c.mean,
                lower,
                log_normalizer,
            });
        }
        Ok(Self { alpha, components })
    }

    fn log_density(&self, u: [f64; 6], in_shell: bool, log_volume: f64) -> f64 {
        let mut result = if in_shell {
            self.alpha.ln() - log_volume
        } else {
            f64::NEG_INFINITY
        };
        if self.alpha < 1. {
            for c in &self.components {
                let mut whitened = [0.; 6];
                for i in 0..6 {
                    whitened[i] = (u[i]
                        - c.mean[i]
                        - (0..i).map(|j| c.lower[i][j] * whitened[j]).sum::<f64>())
                        / c.lower[i][i];
                }
                let log_g = c.log_normalizer - 0.5 * whitened.iter().map(|x| x * x).sum::<f64>();
                result = log_add(result, (1. - self.alpha).ln() + c.weight.ln() + log_g);
            }
        }
        result
    }

    fn draw(
        &self,
        rng: &mut StdRng,
        outer: f64,
        inner: f64,
        shell_fraction: f64,
    ) -> Result<([f64; 6], f64, Option<usize>)> {
        // Alpha=1 consumes precisely the legacy uniform stream.
        if self.alpha == 1. || rng.random::<f64>() < self.alpha {
            let (u, radius) = draw_uniform(rng, outer, inner, shell_fraction)?;
            return Ok((u, radius, None));
        }
        let p = rng.random::<f64>();
        let mut cumulative = 0.;
        let mut index = self.components.len() - 1;
        for (i, c) in self.components.iter().enumerate() {
            cumulative += c.weight;
            if p < cumulative {
                index = i;
                break;
            }
        }
        let component = &self.components[index];
        let standard: [f64; 6] = std::array::from_fn(|_| StandardNormal.sample(rng));
        let u: [f64; 6] = std::array::from_fn(|i| {
            component.mean[i]
                + (0..=i)
                    .map(|j| component.lower[i][j] * standard[j])
                    .sum::<f64>()
        });
        let radius = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        ensure!(
            u.iter().all(|x| x.is_finite()) && radius.is_finite(),
            "Nonfinite Gaussian draw; never silently redraw"
        );
        Ok((u, radius, Some(index)))
    }
}

fn draw_uniform(
    rng: &mut StdRng,
    radius: f64,
    inner_radius: f64,
    shell_fraction: f64,
) -> Result<([f64; 6], f64)> {
    let direction: [f64; 6] = std::array::from_fn(|_| StandardNormal.sample(rng));
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
    Ok((direction.map(|x| x * radial / norm), radial))
}

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
        ensure!(
            !model.has_reciprocal_components(),
            "A latent region requires an unwrapped Gaussian chart"
        );
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
    run_inner(options, None)
}

/// Unbiased fixed-region integration under a frozen defensive latent proposal.
pub fn run_with_importance(options: LatentRegionOptions, guide: &Path) -> Result<Value> {
    run_inner(options, Some(guide))
}

fn run_inner(options: LatentRegionOptions, guide_path: Option<&Path>) -> Result<Value> {
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
    let guide_raw = guide_path.map(fs::read).transpose()?;
    let guide = guide_raw
        .as_ref()
        .map(|raw| ImportanceGuide::from_bytes(raw, &hash_bytes(&region_raw)))
        .transpose()?;
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
    if let Some(raw) = &guide_raw {
        fs::write(options.out.join("provenance/importance-guide.json"), raw)?;
    }
    let source = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(options.out.join("provenance/source-bundle.json"), source)?;
    save(&options.out.join("config.json"), &cfg)?;
    let extended = guide.is_some()
        || inner_radius > 0.
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
    if let Some(g) = &guide {
        manifest["schema"] = json!("importance-latent-region-normalizer-v1");
        manifest["importance_guide_sha256"] = json!(hash_bytes(guide_raw.as_ref().unwrap()));
        manifest["importance_uniform_probability"] = json!(g.alpha);
        manifest["importance_component_count"] = json!(g.components.len());
        manifest["proposal_density_measure"] =
            json!("Lebesgue measure in the original six-dimensional whitened region chart");
        manifest["estimator"] = json!(
            "mean over all unconditional draws of I_shell H_capture H_hard I_q J * independent-cloud average W / complete latent mixture density; outside-shell and invalid draws zero, no retries"
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
    let mut shell_rejected = 0;
    let mut maximum_backmap_error = 0_f64;
    let mut raw_points = 0_u64;
    for draw in 0..options.samples {
        let mut rng = stream(options.seed, draw, 0, "latent");
        let (latent, radial, selected_component) = if let Some(g) = &guide {
            g.draw(&mut rng, radius, inner_radius, shell_fraction)?
        } else {
            let (u, radial) = draw_uniform(&mut rng, radius, inner_radius, shell_fraction)?;
            (u, radial, None)
        };
        // Preserve legacy uniform boundary arithmetic. The guide branch has an
        // explicit target-shell indicator and never retries an exterior draw.
        let shell_valid =
            guide.is_none() || ((inner_radius == 0. || radial > inner_radius) && radial <= radius);
        let log_proposal_density = guide.as_ref().map_or(-log_volume, |g| {
            g.log_density(latent, shell_valid, log_volume)
        });
        ensure!(
            log_proposal_density.is_finite(),
            "Unrepresentable latent proposal density"
        );
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
        let inverse_scale = if guide.is_some() {
            radial.max(radius)
        } else {
            radius
        };
        ensure!(
            error <= 2e-7 * (1. + inverse_scale)
                && (guide.is_some() || backmap_radius <= radius + 2e-7 * (1. + radius)),
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
        shell_rejected += u64::from(!shell_valid);
        let mut clouds = Vec::new();
        let mut log_weight = None;
        let mut log_hard_weight = None;
        if hard_valid && region_valid && shell_valid {
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
            let h = if guide.is_some() {
                log_jacobian - log_proposal_density
            } else {
                log_volume + log_jacobian
            };
            let w = h + log_cloud_sum - (options.cloud_replicates as f64).ln();
            ensure!(h.is_finite() && w.is_finite(), "Invalid regional weight");
            weighted.add(w);
            hard.add(h);
            log_weight = Some(w);
            log_hard_weight = Some(h);
        }
        let mut row = json!({"draw":draw,"latent":latent,"latent_radius":radial,
            "pose":pose,"backmapped_latent":backmap,"backmapped_radius":backmap_radius,
            "log_physical_jacobian":log_jacobian,"physical_jacobian":log_jacobian.exp(),"q":q,
            "capture_valid":capture_valid,"hard_valid":hard_valid,"region_valid":region_valid,
            "log_importance_weight":log_weight,"log_hard_weight":log_hard_weight,"clouds":clouds});
        if guide.is_some() {
            row["shell_valid"] = json!(shell_valid);
            row["log_proposal_density"] = json!(log_proposal_density);
            row["proposal_branch"] = json!(if selected_component.is_some() {
                "gaussian"
            } else {
                "uniform-shell"
            });
            row["proposal_component"] = json!(selected_component);
        }
        serde_json::to_writer(&mut writer, &row)?;
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
    if guide.is_some() {
        summary["shell_rejected"] = json!(shell_rejected);
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

#[cfg(test)]
mod importance_tests {
    use super::*;

    fn raw(alpha: f64) -> Value {
        let mut covariance = [[0.; 6]; 6];
        for (i, row) in covariance.iter_mut().enumerate() {
            row[i] = 1.;
        }
        covariance[1][0] = 0.4;
        covariance[0][1] = 0.4;
        json!({"schema":"defensive-latent-shell-guide-v1", "region_sha256":"region",
            "defensive_uniform_shell_probability":alpha,
            "gaussian_components":[{"weight":3., "mean":[1.,0.,0.,0.,0.,0.], "covariance":covariance}]})
    }

    #[test]
    fn latent_importance_density_support_and_correlated_gaussian() -> Result<()> {
        let guide = ImportanceGuide::from_bytes(raw(0.5).to_string().as_bytes(), "region")?;
        let u = [1.3, -0.7, 0.2, 0.6, -0.2, 0.1];
        let exponent = (0.3_f64.powi(2) - 2. * 0.4 * 0.3 * (-0.7) + 0.7_f64.powi(2))
            / (1. - 0.4_f64.powi(2))
            + 0.2_f64.powi(2)
            + 0.6_f64.powi(2)
            + 0.2_f64.powi(2)
            + 0.1_f64.powi(2);
        let log_g = -3. * (2. * PI).ln() - 0.5 * (1. - 0.4_f64.powi(2)).ln() - 0.5 * exponent;
        assert!((guide.log_density(u, false, 10.) - (0.5_f64.ln() + log_g)).abs() < 1e-13);
        let inside = guide.log_density(u, true, 10.);
        assert!((inside - (0.5 * (-10_f64).exp() + 0.5 * log_g.exp()).ln()).abs() < 1e-13);
        assert!(inside >= 0.5_f64.ln() - 10.);
        Ok(())
    }

    #[test]
    fn latent_importance_rejects_missing_floor_wrong_target_and_bad_covariance() {
        for alpha in [0., -0.1, 1.1] {
            assert!(
                ImportanceGuide::from_bytes(raw(alpha).to_string().as_bytes(), "region").is_err()
            );
        }
        assert!(ImportanceGuide::from_bytes(raw(0.5).to_string().as_bytes(), "wrong").is_err());
        let mut v = raw(0.5);
        v["gaussian_components"][0]["covariance"][0][1] = json!(0.2);
        assert!(ImportanceGuide::from_bytes(v.to_string().as_bytes(), "region").is_err());
        v = raw(0.5);
        v["gaussian_components"][0]["covariance"][0][0] = json!(-1.);
        assert!(ImportanceGuide::from_bytes(v.to_string().as_bytes(), "region").is_err());
        v = raw(0.5);
        v["gaussian_components"] = json!([]);
        assert!(ImportanceGuide::from_bytes(v.to_string().as_bytes(), "region").is_err());
        v["defensive_uniform_shell_probability"] = json!(1.);
        assert!(ImportanceGuide::from_bytes(v.to_string().as_bytes(), "region").is_ok());
    }

    #[test]
    fn latent_importance_multiple_components_use_normalized_complete_mixture() -> Result<()> {
        let mut data = raw(0.3);
        let mut second = data["gaussian_components"][0].clone();
        second["weight"] = json!(7.);
        second["mean"] = json!([-2., 1., 0., 0., 0., 0.]);
        data["gaussian_components"]
            .as_array_mut()
            .unwrap()
            .push(second);
        let guide = ImportanceGuide::from_bytes(data.to_string().as_bytes(), "region")?;
        let u = [0.4, -0.1, 0., 0., 0., 0.];
        let normal = |dx: f64, dy: f64| {
            let md = (dx * dx - 0.8 * dx * dy + dy * dy) / 0.84;
            (-3. * (2. * PI).ln() - 0.5 * 0.84_f64.ln() - 0.5 * md).exp()
        };
        let expected =
            0.3 * (-8_f64).exp() + 0.7 * (0.3 * normal(-0.6, -0.1) + 0.7 * normal(2.4, -1.1));
        assert!((guide.log_density(u, true, 8.) - expected.ln()).abs() < 1e-13);
        let mut counts = [0_usize; 3];
        let mut rng = StdRng::seed_from_u64(815711);
        for _ in 0..100_000 {
            let (_, _, index) = guide.draw(&mut rng, 2., 1., 1. - 1_f64 / 64.)?;
            counts[index.map_or(0, |i| i + 1)] += 1;
        }
        for (count, expected) in counts.into_iter().zip([30_000, 21_000, 49_000]) {
            assert!(count.abs_diff(expected) < 1200);
        }
        Ok(())
    }

    #[test]
    fn latent_importance_uniform_limit_preserves_rng_and_unconditional_volume() -> Result<()> {
        let outer: f64 = 2.3;
        let inner: f64 = 1.2;
        let fraction = 1. - (inner / outer).powi(6);
        let log_v = (PI.powi(3) * (outer.powi(6) - inner.powi(6)) / 6.).ln();
        let mut r0 = StdRng::seed_from_u64(27182);
        let mut r1 = StdRng::seed_from_u64(27182);
        let all_uniform = ImportanceGuide::from_bytes(raw(1.).to_string().as_bytes(), "region")?;
        for _ in 0..128 {
            let (u, r) = draw_uniform(&mut r0, outer, inner, fraction)?;
            let (v, s, component) = all_uniform.draw(&mut r1, outer, inner, fraction)?;
            assert_eq!((u, r), (v, s));
            assert_eq!(component, None);
            assert_eq!(all_uniform.log_density(u, true, log_v), -log_v);
        }
        let guide = ImportanceGuide::from_bytes(raw(0.5).to_string().as_bytes(), "region")?;
        let mut sum = 0.;
        let mut square = 0.;
        let mut outside = 0;
        let mut normals = 0;
        for _ in 0..100_000 {
            let (u, r, k) = guide.draw(&mut r0, outer, inner, fraction)?;
            let inside = r > inner && r <= outer;
            normals += usize::from(k.is_some());
            outside += usize::from(!inside);
            let w = if inside {
                (-guide.log_density(u, true, log_v) - log_v).exp()
            } else {
                0.
            };
            sum += w;
            square += w * w;
        }
        let mean = sum / 100_000.;
        let se = ((square / 100_000. - mean * mean) / 99_999.).sqrt();
        assert!((mean - 1.).abs() < 6. * se);
        assert!(normals > 48_000 && normals < 52_000 && outside > 10_000);
        Ok(())
    }
}
