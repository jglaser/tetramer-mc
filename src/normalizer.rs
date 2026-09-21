//! Fixed-budget independent importance estimates of physical contact-region mass.
//! Invalid proposals contribute zero, and the denominator is the full pose law.
use crate::{
    docking::DockingConfig,
    geometry::{Environment, Placed, Shape, SphereTree},
    math::*,
    overlap_weight::{self, OverlapEnvelope},
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    spherical::Container,
};
use anyhow::{Context, Result, ensure};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Clone, Debug)]
pub struct NormalizerOptions {
    pub config: PathBuf,
    pub model: PathBuf,
    pub out: PathBuf,
    pub samples: u64,
    pub seed: u64,
    /// Multiply Gaussian standard deviations, retaining means and chart anchors.
    pub covariance_scale: f64,
    pub uniform_probability: Option<f64>,
    /// Optional fixed-neighbor index for the proposal coordinate frame only.
    /// All fixed neighbors remain in the hard and depletion environment.
    pub proposal_anchor_index: Option<usize>,
    pub cloud_replicates: usize,
    pub activity: Option<f64>,
}

/// A protein-only wall. Ideal depletants continue to permeate the wall.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NormalizerWall {
    pub center: Vec3,
    pub radius: f64,
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
            "Empty region definition"
        );
        for p in self.native_poses.iter().chain(&self.rigid_members) {
            p.validate()?;
        }
        ensure!(
            self.member_error_scale.is_finite()
                && self.member_error_scale > 0.
                && self.angle_error_scale_deg.is_finite()
                && self.angle_error_scale_deg > 0.,
            "Invalid region scales"
        );
        Ok(())
    }
    fn q(&self, p: Pose) -> f64 {
        self.native_poses
            .iter()
            .map(|r| {
                let error = self
                    .rigid_members
                    .iter()
                    .map(|m| norm(sub(p.apply(m.position), r.apply(m.position))))
                    .fold(0., f64::max)
                    / self.member_error_scale;
                let dot: f64 = p
                    .orientation
                    .iter()
                    .zip(r.orientation)
                    .map(|(a, b)| a * b)
                    .sum();
                let normp = p.orientation.iter().map(|a| a * a).sum::<f64>().sqrt();
                let normr = r.orientation.iter().map(|a| a * a).sum::<f64>().sqrt();
                let angle = 2. * (dot.abs() / (normp * normr)).clamp(0., 1.).acos();
                error.max(angle / self.angle_error_scale_deg.to_radians())
            })
            .fold(f64::INFINITY, f64::min)
    }
}

pub fn region_name(q: f64, contact: bool) -> String {
    let name = if q <= 0.8 {
        "native_core"
    } else if q <= 1. {
        "native_shell"
    } else if q < 2. {
        "shoulder"
    } else if q < 5. {
        "intermediate"
    } else {
        "distant"
    };
    format!("{name}_{}", if contact { "bound" } else { "unbound" })
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

#[derive(Clone, Debug)]
struct Moments {
    log_sum: f64,
    log_sum2: f64,
    log_max: f64,
    nonzero: u64,
}
impl Default for Moments {
    fn default() -> Self {
        Self {
            log_sum: f64::NEG_INFINITY,
            log_sum2: f64::NEG_INFINITY,
            log_max: f64::NEG_INFINITY,
            nonzero: 0,
        }
    }
}
impl Moments {
    fn add(&mut self, log_weight: f64) {
        self.log_sum = log_add(self.log_sum, log_weight);
        self.log_sum2 = log_add(self.log_sum2, 2. * log_weight);
        self.log_max = self.log_max.max(log_weight);
        self.nonzero += 1;
    }
    fn value(&self, n: u64) -> Value {
        if self.nonzero == 0 {
            return json!({"unconditional_draws":n,"nonzero":0,"log_normalizer":null,"ess":0.,
                "observed_relative_standard_error":null,"maximum_weight_fraction":null,
                "coverage":"No nonzero observations; no upper bound on physical mass"});
        }
        let ess = (2. * self.log_sum - self.log_sum2).exp().min(n as f64);
        json!({"unconditional_draws":n,"nonzero":self.nonzero,
            "log_normalizer":self.log_sum-(n as f64).ln(),"ess":ess,
            "observed_relative_standard_error":if n>1 {Some(((n as f64/ess-1.)/(n-1) as f64).max(0.).sqrt())} else {None},
            "maximum_weight_fraction":(self.log_max-self.log_sum).exp(),
            "log_sum_weights":self.log_sum,"log_sum_squared_weights":self.log_sum2,
            "coverage":"Observed-sample uncertainty; does not certify unseen modes"})
    }
}

fn stream(seed: u64, draw: u64, cloud: usize, name: &str) -> StdRng {
    let mut h = Sha256::new();
    h.update(b"tetramer-normalizer-independent-v1");
    h.update(seed.to_le_bytes());
    h.update(draw.to_le_bytes());
    h.update((cloud as u64).to_le_bytes());
    h.update(name.as_bytes());
    StdRng::from_seed(h.finalize().into())
}

pub fn run(options: NormalizerOptions) -> Result<Value> {
    run_impl(options, None)
}

/// Integrate the full atomic-wall domain, with capture used only as a
/// conservative proposal/support enclosure. A truncated enclosure is refused.
pub fn run_with_wall(options: NormalizerOptions, wall: NormalizerWall) -> Result<Value> {
    run_impl(options, Some(wall))
}

fn run_impl(options: NormalizerOptions, wall_spec: Option<NormalizerWall>) -> Result<Value> {
    ensure!(
        options.samples > 0 && options.cloud_replicates > 0,
        "Positive sample and cloud counts required"
    );
    ensure!(
        options.covariance_scale.is_finite() && options.covariance_scale > 0.,
        "Invalid covariance scale"
    );
    let raw = fs::read(&options.config)?;
    let mut cfg: DockingConfig = serde_json::from_slice(&raw)?;
    cfg.validate()?;
    ensure!(
        cfg.target_region.is_none(),
        "basin-normalizer does not implement DockingConfig.target_region; use a configuration without that docking-only constraint and an explicitly supported integration region"
    );
    if let Some(index) = options.proposal_anchor_index {
        ensure!(
            index < cfg.fixed_poses.len(),
            "Proposal anchor index outside fixed-neighbor list"
        );
    }
    if cfg.shape.is_relative() {
        cfg.shape = options.config.parent().unwrap().join(&cfg.shape);
    }
    if let Some(z) = options.activity {
        ensure!(z.is_finite() && z >= 0., "Invalid activity");
        cfg.reservoir_density = z;
    }
    let epsilon = options
        .uniform_probability
        .unwrap_or(cfg.uniform_probability);
    ensure!(
        epsilon.is_finite() && epsilon > 0. && epsilon <= 1.,
        "Invalid uniform weight"
    );
    let metric: Metric = serde_json::from_value(cfg.metadata.clone())
        .context("Missing complete fixed region definition in config metadata")?;
    metric.validate()?;
    let shape_raw = fs::read(&cfg.shape)?;
    let shape_sha = hash_bytes(&shape_raw);
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?;
    let wall = if let Some(spec) = wall_spec {
        ensure!(
            spec.center.iter().all(|x| x.is_finite()),
            "Invalid wall center"
        );
        let container = Container::new(spec.radius, &tree)?;
        // For any atom, |body center - wall center| <= wall radius +
        // |atom body-frame center|. The shape bound therefore encloses ALL
        // feasible centers, even for offset or highly concave sphere unions.
        let required = norm(sub(spec.center, cfg.capture_center)) + spec.radius + tree.bound;
        let guard = 256. * f64::EPSILON * (1. + required);
        ensure!(
            required.is_finite() && cfg.capture_radius >= required + guard,
            "Capture must enclose the full atomic-wall domain: radius >= {}",
            required + guard
        );
        for (i, pose) in cfg.fixed_poses.iter().enumerate() {
            ensure!(
                container.contains(Pose {
                    position: sub(pose.position, spec.center),
                    ..*pose
                }),
                "Fixed neighbor {i} is outside atomic wall"
            );
        }
        Some(container)
    } else {
        None
    };
    let mut contact_shape = tree.shape.clone();
    for a in &mut contact_shape.atoms {
        a.radius += cfg.depletant_radius;
    }
    let contact_tree = SphereTree::new(contact_shape)?;
    let env = Environment {
        tree: &tree,
        fixed: cfg.fixed_poses.iter().copied().map(Placed::new).collect(),
        labels: (0..cfg.fixed_poses.len()).map(|j| (j, [0; 3])).collect(),
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
    let model_raw = fs::read(&options.model)?;
    let original = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&model_raw)?,
        [2. * cfg.capture_radius; 3],
        epsilon,
        &shape_sha,
    )?;
    let reciprocal = original.has_reciprocal_components();
    // Exact nonlinear inverse branches are already normalized proposal laws.
    // Keep the immutable model intact instead of exporting its base Gaussians
    // and silently dropping those branches during covariance replacement.
    let model = if reciprocal {
        ensure!(
            options.covariance_scale == 1.,
            "Reciprocal normalizer proposals require covariance scale 1"
        );
        original
    } else {
        let mut parameters = original.component_parameters();
        for p in &mut parameters {
            for row in &mut p.covariance {
                for v in row {
                    *v *= options.covariance_scale.powi(2);
                }
            }
        }
        original.with_component_parameters(parameters)?
    };
    let centered = |p: Pose| Pose {
        position: sub(p.position, cfg.capture_center),
        orientation: p.orientation,
    };
    let mut poses = vec![centered(cfg.initial_pose)];
    if let Some(index) = options.proposal_anchor_index {
        poses.push(centered(cfg.fixed_poses[index]));
    } else {
        // Preserve the original anchor-selection order and RNG stream.
        poses.extend(cfg.fixed_poses.iter().copied().map(centered));
    }
    let lambda = if cfg.reservoir_density > 0. {
        cfg.poisson_lambda_ratio * cfg.reservoir_density
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
            "Output directory must be empty"
        );
    }
    fs::create_dir_all(options.out.join("provenance"))?;
    fs::write(options.out.join("provenance/input-config.json"), &raw)?;
    fs::write(options.out.join("provenance/model.json"), &model_raw)?;
    fs::write(options.out.join("provenance/shape.json"), &shape_raw)?;
    let source = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(options.out.join("provenance/source-bundle.json"), source)?;
    save(&options.out.join("config.json"), &cfg)?;
    let mut manifest = json!({"schema":if options.proposal_anchor_index.is_some(){2}else{1},"samples":options.samples,"seed":options.seed,
        "cloud_replicates":options.cloud_replicates,"covariance_scale":options.covariance_scale,
        "uniform_probability":epsilon,"lambda":lambda,"activity":cfg.reservoir_density,
        "config_sha256":hash_bytes(&raw),"model_sha256":hash_bytes(&model_raw),"shape_sha256":shape_sha,
        "source_bundle_sha256":hash_bytes(source.as_bytes()),"executable_sha256":hash_file(&std::env::current_exe()?)?,
        "proposal":if options.proposal_anchor_index.is_some(){"Full normalized atlas+uniform cube/Haar density at the selected fixed anchor; no retry"}else{"Full normalized atlas+uniform cube/Haar density averaged over all fixed anchors; no retry"},
        "proposal_anchor_index":options.proposal_anchor_index,
        "physical_fixed_neighbor_count":cfg.fixed_poses.len(),
        "target":"hard(x,S) capture(x) exp[z |E(x) intersect union E(S)|] with Lebesgue and normalized Haar measure",
        "partition":"q<=.8; .8<q<=1; 1<q<2; 2<=q<5; q>=5, each split by exclusion contact; exhaustive within target support",
        "estimator":"zero for invalid; exp(z lower_volume)*(1+z/lambda)^K / full_proposal_density; average independent clouds in linear scale",
        "inference":"Independent importance draws. Observed ESS/RSE cannot rule out unobserved high-weight modes."});
    if reciprocal {
        manifest["schema"] = json!(3);
        manifest["proposal_model_kind"] = json!("reciprocal-pose-mixture-v1");
        manifest["base_component_count"] = json!(model.component_count());
        manifest["virtual_component_count"] = json!(model.virtual_branches().len());
        manifest["reciprocal_components"] = json!(model.reciprocal_components());
    }
    if let Some(spec) = wall_spec {
        manifest["pose_proposal_schema"] = manifest["schema"].clone();
        manifest["schema"] = json!(4);
        manifest["atomic_wall"] = json!(spec);
        manifest["bath_wall_permeable"] = json!(true);
        manifest["shape_bound"] = json!(tree.bound);
        manifest["target"] = json!(
            "hard(x,S) atomic_wall(x) exp[z |E(x) intersect union E(S)|] with Lebesgue and normalized Haar measure; capture encloses all feasible centers"
        );
        manifest["partition"] = json!(
            "q<=.8; .8<q<=1; 1<q<2; 2<=q<5; q>=5, each split by exclusion contact; exhaustive within the full atomic-wall domain"
        );
    }
    save(&options.out.join("manifest.json"), &manifest)?;
    let start = Instant::now();
    let cpu_start = cpu_seconds();
    let mut writer = BufWriter::new(File::create(options.out.join("samples.jsonl"))?);
    let mut moments: BTreeMap<String, Moments> = BTreeMap::new();
    for name in [
        "total",
        "native",
        "other",
        "hard_total",
        "hard_native",
        "hard_other",
    ] {
        moments.insert(name.into(), Moments::default());
    }
    for q in [
        "native_core",
        "native_shell",
        "shoulder",
        "intermediate",
        "distant",
    ] {
        for contact in ["bound", "unbound"] {
            moments.insert(format!("{q}_{contact}"), Moments::default());
        }
    }
    let mut valid = 0u64;
    let mut capture_rejected = 0u64;
    let mut hard_rejected = 0u64;
    let mut wall_rejected = 0u64;
    let mut numerical_nulls = 0u64;
    let mut raw_points = 0u64;
    let mut proposal_cpu = 0.;
    let mut geometry_cpu = 0.;
    let mut envelope_cpu = 0.;
    let mut cloud_cpu = 0.;
    for draw in 0..options.samples {
        let before = cpu_seconds();
        let outcome = model.propose(&mut stream(options.seed, draw, 0, "pose"), &poses, 0)?;
        ensure!(
            (!reciprocal && wall.is_none() && options.proposal_anchor_index.is_none())
                || outcome.candidate.is_some(),
            "{} importance draw produced a numerical null: {:?}; stop instead of censoring",
            if reciprocal {
                "Reciprocal"
            } else {
                "Selected-anchor"
            },
            outcome.null_reason
        );
        let candidate = outcome.candidate.map(|p| Pose {
            position: add(p.position, cfg.capture_center),
            orientation: p.orientation,
        });
        let log_proposal = if let Some(p) = outcome.candidate {
            let mut total = f64::NEG_INFINITY;
            for anchor in &poses[1..] {
                total = log_add(total, model.log_density(&p, anchor)?);
            }
            let density = total - ((poses.len() - 1) as f64).ln();
            ensure!(density.is_finite(), "Nonfinite proposal density");
            Some(density)
        } else {
            None
        };
        proposal_cpu += cpu_seconds() - before;
        let mut capture_valid = false;
        let mut hard_valid = false;
        let mut wall_valid = false;
        let mut q = None;
        let mut contact = None;
        let mut region = None;
        let mut clouds = Vec::new();
        let mut log_weight = None;
        let mut log_hard = None;
        if let Some(p) = candidate {
            let before = cpu_seconds();
            capture_valid = cfg.contains(p);
            wall_valid = wall.as_ref().is_none_or(|container| {
                container.contains(Pose {
                    position: sub(p.position, wall_spec.unwrap().center),
                    ..p
                })
            });
            if capture_valid && wall_valid {
                hard_valid = env.hard_valid(p);
            }
            if !capture_valid {
                capture_rejected += 1;
            } else if !wall_valid {
                wall_rejected += 1;
            } else if !hard_valid {
                hard_rejected += 1;
            }
            if hard_valid {
                let score = metric.q(p);
                ensure!(score.is_finite(), "Invalid native coordinate");
                q = Some(score);
                let placed = Placed::new(p);
                let bound = env.fixed.iter().any(|f| contact_tree.overlaps(&placed, f));
                contact = Some(bound);
                region = Some(region_name(score, bound));
            }
            geometry_cpu += cpu_seconds() - before;
            if hard_valid {
                valid += 1;
                let before = cpu_seconds();
                let envelope = OverlapEnvelope::build(&env, p, cfg.endpoint_gate)?;
                envelope_cpu += cpu_seconds() - before;
                let before = cpu_seconds();
                let mut log_cloud_sum = f64::NEG_INFINITY;
                for cloud in 0..options.cloud_replicates {
                    let weight = overlap_weight::sample_with_envelope(
                        &mut stream(options.seed, draw, cloud, "cloud"),
                        &env,
                        p,
                        lambda,
                        cfg.reservoir_density,
                        &envelope,
                    )?;
                    raw_points += weight.raw_points;
                    log_cloud_sum = log_add(log_cloud_sum, weight.log_weight);
                    clouds.push(weight);
                }
                cloud_cpu += cpu_seconds() - before;
                let inverse_q = -log_proposal.unwrap();
                let importance = log_cloud_sum - (options.cloud_replicates as f64).ln() + inverse_q;
                ensure!(importance.is_finite(), "Nonfinite importance weight");
                log_weight = Some(importance);
                log_hard = Some(inverse_q);
                let basin = if q.unwrap() <= 1. { "native" } else { "other" };
                for key in ["total", basin, region.as_ref().unwrap().as_str()] {
                    moments.get_mut(key).unwrap().add(importance);
                }
                moments.get_mut("hard_total").unwrap().add(inverse_q);
                moments
                    .get_mut(&format!("hard_{basin}"))
                    .unwrap()
                    .add(inverse_q);
            }
        } else {
            numerical_nulls += 1;
        }
        let mut row = json!({"draw":draw,"pose":candidate,"proposal":outcome,
            "capture_valid":capture_valid,"hard_valid":hard_valid,"q":q,"depletion_contact":contact,"region":region,
            "log_proposal_density":log_proposal,"log_importance_weight":log_weight,"log_hard_weight":log_hard,"clouds":clouds});
        if wall.is_some() {
            row["wall_valid"] = json!(wall_valid);
        }
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
        if (draw + 1) % 100 == 0 || draw + 1 == options.samples {
            writer.flush()?;
            save(
                &options.out.join("progress.json"),
                &json!({"completed_draws":draw+1,"requested_draws":options.samples,
                "hard_valid":valid,"sampler_cpu_seconds":cpu_seconds()-cpu_start,"complete":draw+1==options.samples}),
            )?;
        }
    }
    writer.flush()?;
    let estimates: BTreeMap<_, _> = moments
        .into_iter()
        .map(|(k, m)| (k, m.value(options.samples)))
        .collect();
    let mut summary = json!({"complete":true,"samples":options.samples,"hard_valid":valid,"capture_rejected":capture_rejected,
        "hard_rejected":hard_rejected,"numerical_nulls":numerical_nulls,"raw_points":raw_points,"estimates":estimates,
        "sampler_cpu_seconds":cpu_seconds()-cpu_start,"wall_seconds":start.elapsed().as_secs_f64(),
        "cost":{"proposal_cpu_seconds":proposal_cpu,"geometry_cpu_seconds":geometry_cpu,"envelope_cpu_seconds":envelope_cpu,"cloud_cpu_seconds":cloud_cpu},
        "manifest":manifest});
    if wall.is_some() {
        summary["wall_rejected"] = json!(wall_rejected);
    }
    save(&options.out.join("summary.json"), &summary)?;
    Ok(summary)
}
