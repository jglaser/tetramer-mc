//! Fixed-schedule random-potential SMC on the unchanged full R4 domain.
//!
//! This is a separate estimator, not continuation of any independent-draw run.
//! See docs/smc-r4-control.md for the unnormalized-measure identity and limits.
use super::{Chart, Metric, draw_uniform, log_add};
use crate::{
    depletion,
    docking::{self, DockingConfig},
    geometry::{Environment, Placed, Shape, SphereTree},
    math::Pose,
    overlap_weight::{self, OverlapEnvelope},
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    f64::consts::PI,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

const RADIUS: f64 = 4.;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SmcBridge {
    PhysicalActivity,
    ProposalDensity,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmcOptions {
    pub config: PathBuf,
    pub region: PathBuf,
    /// Optional complete UNFILTERED chart ball; its old q/capture/shell masks are ignored.
    pub initial_reference_region: Option<PathBuf>,
    /// Current-ball probability; must equal one when no reference is supplied.
    pub initial_current_probability: f64,
    pub out: PathBuf,
    pub initial_draws: usize,
    pub population: usize,
    pub seed: u64,
    pub bridge: SmcBridge,
    /// Fixed, strictly increasing activity fractions, starting at 0 and ending at 1.
    pub schedule: Vec<f64>,
    /// Each sweep makes one symmetric physical-pose proposal per offspring.
    pub sweeps_per_stage: usize,
    pub cloud_replicates: usize,
    /// Incremental clouds use lambda = lambda_ratio * delta_activity.
    pub lambda_ratio: f64,
}

#[derive(Clone, Debug, Serialize)]
struct Particle {
    pose: Pose,
    latent: [f64; 6],
    /// The unconditional initialization draw; never replaced by an offspring index.
    initial_ancestor: usize,
}

#[derive(Default, Serialize)]
struct MutationCounts {
    attempted: usize,
    accepted: usize,
    accepted_translation_changes: usize,
    accepted_rotation_changes: usize,
    capture_rejected: usize,
    region_rejected: usize,
    hard_rejected: usize,
    gate_rejected: usize,
    raw_points: u64,
}

fn stream(seed: u64, stage: usize, index: usize, subindex: usize, role: &str) -> StdRng {
    let mut hash = Sha256::new();
    hash.update(b"latent-region-smc-v1\0");
    hash.update(seed.to_le_bytes());
    for value in [stage, index, subindex] {
        hash.update((value as u64).to_le_bytes());
    }
    hash.update(role.as_bytes());
    StdRng::from_seed(hash.finalize().into())
}

fn jsonline(writer: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn scaled_weights(log_weights: &[f64]) -> Result<(Vec<f64>, f64, f64)> {
    ensure!(
        !log_weights.is_empty() && log_weights.iter().all(|x| x.is_finite()),
        "Nonfinite or empty SMC weights"
    );
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = log_weights.iter().map(|x| (x - maximum).exp()).collect();
    ensure!(
        weights.iter().all(|w| *w > 0.),
        "Positive SMC weight underflowed during rescaling"
    );
    let sum: f64 = weights.iter().sum();
    ensure!(sum.is_finite() && sum > 0., "Invalid scaled SMC weight sum");
    let ess = sum * sum / weights.iter().map(|w| w * w).sum::<f64>();
    Ok((weights, maximum + sum.ln(), ess))
}

/// The offset is uniform in [0,1/N); strict CDF comparison skips zero weights,
/// including at offset zero. Offspring counts have expectation N*w_i/sum(w).
fn systematic_at_offset(weights: &[f64], population: usize, offset: f64) -> Result<Vec<usize>> {
    ensure!(
        population > 0 && offset >= 0. && offset < 1. / population as f64,
        "Invalid systematic offset"
    );
    ensure!(
        !weights.is_empty() && weights.iter().all(|x| x.is_finite() && *x >= 0.),
        "Invalid resampling weights"
    );
    let sum: f64 = weights.iter().sum();
    ensure!(
        sum.is_finite() && sum > 0.,
        "Cannot resample a zero population"
    );
    let last = weights.iter().rposition(|x| *x > 0.).unwrap();
    let mut parent = 0;
    let mut cumulative = weights[0];
    let mut parents = Vec::with_capacity(population);
    for i in 0..population {
        let target = (offset + i as f64 / population as f64) * sum;
        while parent < last && cumulative <= target {
            parent += 1;
            cumulative += weights[parent];
        }
        ensure!(
            weights[parent] > 0.,
            "Systematic resampling selected zero mass"
        );
        parents.push(parent);
    }
    Ok(parents)
}

// A finite Cayley ball has no support on its opposite-orientation seam.
// All other inverse-map failures remain explicit numerical failures.
fn encode_if_off_seam(chart: &Chart, pose: Pose) -> Result<Option<[f64; 6]>> {
    use crate::math::{matmul, quaternion, rotation, transpose};
    let residual = matmul(
        matmul(
            transpose(rotation(chart.fixed.orientation)),
            rotation(pose.orientation),
        ),
        transpose(chart.anchor_rotation),
    );
    if quaternion(residual)[0].abs() == 0. {
        Ok(None)
    } else {
        Ok(Some(chart.encode(pose)?))
    }
}

fn proposal_log_density(
    chart: &Chart,
    reference: Option<&(Chart, f64, f64)>,
    alpha: f64,
    log_volume: f64,
    pose: Pose,
) -> Result<f64> {
    let current = encode_if_off_seam(chart, pose)?;
    let mut density =
        if current.is_some_and(|u| u.iter().map(|x| x * x).sum::<f64>() <= RADIUS * RADIUS) {
            alpha.ln() - log_volume - chart.decode(current.unwrap()).1
        } else {
            f64::NEG_INFINITY
        };
    if let Some((c, r, v)) = reference {
        let u = encode_if_off_seam(c, pose)?;
        if alpha < 1. && u.is_some_and(|u| u.iter().map(|x| x * x).sum::<f64>() <= r * r) {
            density = log_add(density, (1. - alpha).ln() - v - c.decode(u.unwrap()).1);
        }
    }
    Ok(density)
}

fn ancestry(particles: &[Particle]) -> Value {
    let mut counts = BTreeMap::<usize, usize>::new();
    for p in particles {
        *counts.entry(p.initial_ancestor).or_default() += 1;
    }
    let n = particles.len() as f64;
    let sum_sq = counts.values().map(|&c| (c as f64).powi(2)).sum::<f64>();
    json!({"distinct_initial_ancestors":counts.len(),
        "initial_family_ESS":if sum_sq > 0. { n*n/sum_sq } else { 0. },
        "largest_initial_family_fraction":counts.values().copied().max().map(|c| c as f64/n),
        "initial_ancestor_counts":counts,
        "scope":"Descendant concentration, not independent terminal-pose ESS or convergence."})
}

fn mutate(
    particles: &mut [Particle],
    cfg: &DockingConfig,
    env: &Environment,
    chart: &Chart,
    options: &SmcOptions,
    stage: usize,
    activity: f64,
    beta: f64,
    reference: Option<&(Chart, f64, f64)>,
    log_volume: f64,
) -> Result<(MutationCounts, Vec<Value>)> {
    let mut counts = MutationCounts::default();
    let mut density_records = Vec::new();
    let lambda = if activity > 0. {
        cfg.poisson_lambda_ratio * activity
    } else {
        1.
    };
    ensure!(
        lambda.is_finite() && lambda > 0.,
        "Invalid mutation count intensity"
    );
    for (index, particle) in particles.iter_mut().enumerate() {
        let mut log_g = proposal_log_density(
            chart,
            reference,
            options.initial_current_probability,
            log_volume,
            particle.pose,
        )?;
        ensure!(log_g.is_finite(), "Retained SMC pose lost proposal support");
        for sweep in 0..options.sweeps_per_stage {
            counts.attempted += 1;
            let new = docking::local(
                &mut stream(options.seed, stage, index, sweep, "local-proposal"),
                particle.pose,
                cfg,
            );
            new.validate()?;
            if !cfg.contains(new) {
                counts.capture_rejected += 1;
                continue;
            }
            let latent = chart.encode(new)?;
            if latent.iter().map(|x| x * x).sum::<f64>() > RADIUS * RADIUS {
                counts.region_rejected += 1;
                continue;
            }
            if !env.hard_valid(new) {
                counts.hard_rejected += 1;
                continue;
            }
            // The proposal is symmetric in physical translation/Haar measure.
            // This is the gained/lost-count MH gate, not a ratio of noisy Zs.
            let gate = depletion::sample(
                &mut stream(options.seed, stage, index, sweep, "mutation-gate"),
                env,
                particle.pose,
                new,
                lambda,
                activity,
                cfg.endpoint_gate,
            )?;
            counts.raw_points += gate.raw_points;
            ensure!(gate.log_weight.is_finite(), "Nonfinite mutation gate");
            let new_log_g = proposal_log_density(
                chart,
                reference,
                options.initial_current_probability,
                log_volume,
                new,
            )?;
            ensure!(new_log_g.is_finite(), "SMC candidate lost proposal support");
            let correction = if options.bridge == SmcBridge::ProposalDensity {
                (1. - beta) * (new_log_g - log_g)
            } else {
                0.
            };
            let log_acceptance = (correction + gate.log_weight).min(0.);
            let log_uniform = stream(options.seed, stage, index, sweep, "mutation-accept")
                .random::<f64>()
                .ln();
            let accepted = log_uniform < log_acceptance;
            if options.bridge == SmcBridge::ProposalDensity {
                density_records.push(json!({"particle":index,"sweep":sweep,"old_pose":particle.pose,"proposed_pose":new,
                    "log_g_old":log_g,"log_g_new":new_log_g,"beta":beta,"deterministic_log_correction":correction,
                    "gate":gate,"log_acceptance":log_acceptance,"log_uniform":log_uniform,"accepted":accepted}));
            }
            if accepted {
                log_g = new_log_g;
                counts.accepted_translation_changes +=
                    usize::from(new.position != particle.pose.position);
                counts.accepted_rotation_changes +=
                    usize::from(new.orientation != particle.pose.orientation);
                particle.pose = new;
                particle.latent = latent;
                counts.accepted += 1;
            } else {
                counts.gate_rejected += 1;
            }
        }
    }
    Ok((counts, density_records))
}

pub fn run(options: SmcOptions) -> Result<Value> {
    let alpha = options.initial_current_probability;
    ensure!(
        alpha.is_finite()
            && alpha > 0.
            && alpha <= 1.
            && (options.initial_reference_region.is_some() || alpha == 1.),
        "Initialization current-ball probability must be positive; reference required below one"
    );
    ensure!(
        options.initial_draws > 0 && options.population > 0 && options.cloud_replicates > 0,
        "Positive SMC budgets required"
    );
    ensure!(
        options.lambda_ratio.is_finite() && options.lambda_ratio > 0.,
        "Invalid incremental lambda ratio"
    );
    ensure!(
        options.schedule.len() >= 2
            && options.schedule.first() == Some(&0.)
            && options.schedule.last() == Some(&1.)
            && options.schedule.iter().all(|t| t.is_finite())
            && options.schedule.windows(2).all(|w| w[0] < w[1]),
        "Fixed schedule must strictly increase from zero to one"
    );
    ensure!(
        !options.out.exists(),
        "Fresh SMC output directory required; no overwrite or resume"
    );
    let config_raw = fs::read(&options.config)?;
    let mut cfg: DockingConfig = serde_json::from_slice(&config_raw)?;
    cfg.validate()?;
    ensure!(
        cfg.target_region.is_none(),
        "SMC full R4 does not accept docking target_region"
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
    let physical_fixed: Vec<Pose> = region
        .get("physical_fixed_neighbors")
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()?
        .unwrap_or_else(|| vec![fixed]);
    ensure!(
        physical_fixed == cfg.fixed_poses && physical_fixed.contains(&fixed),
        "Frozen physical neighbors/anchor mismatch"
    );
    ensure!(
        region["capture_center"] == json!(cfg.capture_center)
            && region["capture_radius"] == json!(cfg.capture_radius),
        "Frozen capture mismatch"
    );
    ensure!(
        region["activity"] == json!(cfg.reservoir_density)
            && region["depletant_radius"] == json!(cfg.depletant_radius),
        "Frozen physical bath mismatch"
    );
    ensure!(
        region["physical_metric"] == cfg.metadata,
        "Frozen original metric mismatch"
    );
    let metric: Metric = serde_json::from_value(cfg.metadata.clone())?;
    metric.validate()?;
    ensure!(
        region["mahalanobis_radius"] == json!(RADIUS)
            && region
                .get("minimum_mahalanobis_radius")
                .is_none_or(|v| v == &json!(0.))
            && region["minimum_original_q"] == json!(0.)
            && region
                .get("minimum_original_q_inclusive")
                .is_none_or(|v| v == &json!(true))
            && region.get("maximum_original_q").is_none(),
        "SMC requires the full original R4, without q/class/shell restrictions"
    );
    let shape_raw = fs::read(&cfg.shape)?;
    let shape_hash = hash_bytes(&shape_raw);
    ensure!(
        region["shape_sha256"] == json!(shape_hash),
        "Frozen shape mismatch"
    );
    let chart = Chart::new(&region, &shape_hash, cfg.capture_radius, fixed)?;
    let reference_raw = options
        .initial_reference_region
        .as_ref()
        .map(fs::read)
        .transpose()?;
    let reference = reference_raw
        .as_ref()
        .map(|bytes| -> Result<(Chart, f64, f64)> {
            let spec: Value = serde_json::from_slice(bytes)?;
            let anchor: Pose = serde_json::from_value(spec["fixed_neighbor"].clone())?;
            let neighbors: Vec<Pose> = spec
                .get("physical_fixed_neighbors")
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()?
                .unwrap_or_else(|| vec![anchor]);
            ensure!(
                spec["shape_sha256"] == json!(shape_hash)
                    && neighbors == physical_fixed
                    && neighbors.contains(&anchor),
                "Reference chart shape/physical-neighbor binding mismatch"
            );
            let radius = spec["mahalanobis_radius"]
                .as_f64()
                .context("Missing reference-ball radius")?;
            ensure!(
                radius.is_finite() && radius > 0.,
                "Invalid reference-ball radius"
            );
            let log_v = 3. * PI.ln() + 6. * radius.ln() - 6_f64.ln();
            let reference_chart = Chart::new(&spec, &shape_hash, cfg.capture_radius, anchor)?;
            // Deliberately do not read the old q, inner-radius or capture restrictions.
            Ok((reference_chart, radius, log_v))
        })
        .transpose()?;
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
                "Fixed physical neighbors overlap"
            );
        }
    }
    // Creating the leaf atomically prevents a competing invocation from reusing it.
    if let Some(parent) = options.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&options.out)?;
    fs::create_dir(options.out.join("provenance"))?;
    let source_bundle = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    for (name, bytes) in [
        ("config.json", config_raw.as_slice()),
        ("region.json", region_raw.as_slice()),
        ("shape.json", shape_raw.as_slice()),
        ("source-bundle.json", source_bundle.as_slice()),
    ] {
        fs::write(options.out.join("provenance").join(name), bytes)?;
    }
    if let Some(bytes) = &reference_raw {
        fs::write(
            options.out.join("provenance/initial-reference-region.json"),
            bytes,
        )?;
    }
    let log_volume = 3. * PI.ln() + 6. * RADIUS.ln() - 6_f64.ln();
    let manifest = json!({"schema":"latent-region-smc-v1", "options":options,
        "config_sha256":hash_bytes(&config_raw), "region_sha256":hash_bytes(&region_raw), "shape_sha256":shape_hash,
        "source_bundle_sha256":hash_bytes(source_bundle), "executable_sha256":hash_file(&std::env::current_exe()?)?,
        "target":"H_capture H_hard I_original_R4 exp(z C) d3t dHaar(R); C uses the full fixed-neighbor exclusion union",
        "initialization":"M unconditional chart-ball mixture draws; physical-activity bridge a=H_current/g, proposal-density bridge a=H_current; all M retained",
        "bridge":options.bridge,
        "bridge_density":"physical_activity: gamma_beta=H exp(beta*z*C); proposal_density: gamma_beta=H g_physical^(1-beta) exp(beta*z*C), relative to fixed d3t dHaar",
        "final_target_unchanged":true,
        "initial_current_probability":alpha,
        "initial_reference_sha256":reference_raw.as_ref().map(|b|hash_bytes(b)),
        "initial_reference_radius":reference.as_ref().map(|(_,r,_)|r),
        "initial_density":"alpha I_current_ball/(V4 J_current) + (1-alpha) I_unfiltered_reference_ball/(Vref J_reference); ignore reference q/capture/inner-radius masks",
        "activity_path":options.schedule.iter().map(|t| t*cfg.reservoir_density).collect::<Vec<_>>(),
        "incremental_potential":"fresh independent nonnegative overlap factors at delta_z, averaged in linear space; proposal-density bridge additionally multiplies by g_physical^(-delta_beta); lambda=lambda_ratio*delta_z",
        "mutation":"fixed symmetric physical proposals and gained/lost gate at beta*z; proposal-density bridge adds (1-beta)*(log_g_new-log_g_old); no extra Jacobian factor",
        "terminal_estimator":"Zhat * mean_N(f(terminal_pose)); report fixed-region indicators with the unchanged independent complete classifier",
        "worker_count":1, "log_latent_ball_volume":log_volume,
        "scope":"Separate fixed-budget conditional normalizer. No protein convergence claim, automatic extension, classifier change, or existing Lean SMC theorem."});
    save(&options.out.join("manifest.json"), &manifest)?;
    let start = Instant::now();
    let cpu_start = cpu_seconds();
    let mut completed_stage = 0usize;
    let result = (|| -> Result<Value> {
        save(
            &options.out.join("status.json"),
            &json!({"complete":false,"phase":"initialization"}),
        )?;
        let mut initial_file =
            BufWriter::new(File::create(options.out.join("initialization.jsonl"))?);
        let mut stage_file = BufWriter::new(File::create(options.out.join("stages.jsonl"))?);
        let mut initial = Vec::new();
        let mut initial_logs = Vec::new();
        for draw in 0..options.initial_draws {
            let mut rng = stream(options.seed, 0, draw, 0, "initial-pose");
            let from_current = alpha == 1. || rng.random::<f64>() < alpha;
            let (selected_chart, selected_radius, _) = if from_current {
                (&chart, RADIUS, log_volume)
            } else {
                let (c, r, v) = reference.as_ref().context("Missing initial reference")?;
                (c, *r, *v)
            };
            let (selected_latent, _) = draw_uniform(&mut rng, selected_radius, 0., 1.)?;
            let (pose, selected_log_jacobian) = selected_chart.decode(selected_latent);
            pose.validate()?;
            let latent = if from_current {
                Some(selected_latent)
            } else {
                encode_if_off_seam(&chart, pose)?
            };
            let radius = latent.map(|u| u.iter().map(|x| x * x).sum::<f64>().sqrt());
            let current_ball = radius.is_some_and(|r| r <= RADIUS);
            let log_jacobian = latent.map(|u| chart.decode(u).1);
            let mut log_density = if current_ball {
                alpha.ln() - log_volume - log_jacobian.context("Missing current Jacobian")?
            } else {
                f64::NEG_INFINITY
            };
            let mut reference_latent = None;
            let mut reference_log_jacobian = None;
            let mut reference_ball = false;
            if let Some((reference_chart, r, log_v)) = &reference {
                reference_latent = if from_current {
                    encode_if_off_seam(reference_chart, pose)?
                } else {
                    Some(selected_latent)
                };
                reference_log_jacobian = reference_latent.map(|u| reference_chart.decode(u).1);
                reference_ball =
                    reference_latent.is_some_and(|u| u.iter().map(|x| x * x).sum::<f64>() <= r * r);
                if reference_ball && alpha < 1. {
                    log_density = log_add(
                        log_density,
                        (1. - alpha).ln()
                            - log_v
                            - reference_log_jacobian.context("Missing reference Jacobian")?,
                    );
                }
            }
            ensure!(
                selected_log_jacobian.is_finite()
                    && log_jacobian.is_none_or(|v| v.is_finite())
                    && reference_log_jacobian.is_none_or(|v| v.is_finite())
                    && log_density.is_finite(),
                "Initial draw has nonfinite Jacobian or no representable mixture support"
            );
            let capture_valid = cfg.contains(pose);
            let hard_valid = capture_valid && current_ball && env.hard_valid(pose);
            let log_hard_weight = hard_valid.then_some(-log_density);
            let log_weight =
                hard_valid.then_some(if options.bridge == SmcBridge::ProposalDensity {
                    0.
                } else {
                    -log_density
                });
            jsonline(
                &mut initial_file,
                &json!({"draw":draw,"pose":pose,"latent":latent,"latent_radius":radius,
                "selected_initial_chart":if from_current {"current"} else {"reference"},
                "selected_latent":selected_latent,"selected_log_physical_jacobian":selected_log_jacobian,
                "reference_latent":reference_latent,"reference_log_physical_jacobian":reference_log_jacobian,
                "reference_ball_valid":reference_ball,"current_ball_valid":current_ball,
                "log_proposal_density":if alpha==1. {Some(-log_volume)} else {None},
                "log_physical_proposal_density":log_density,"log_physical_jacobian":log_jacobian,
                "capture_valid":capture_valid,"hard_valid":hard_valid,"log_hard_weight":log_hard_weight,"log_initial_weight":log_weight}),
            )?;
            if let Some(weight) = log_weight {
                initial.push(Particle {
                    pose,
                    latent: latent.context("Valid target missing chart coordinates")?,
                    initial_ancestor: draw,
                });
                initial_logs.push(weight);
            }
        }
        initial_file.flush()?;
        if initial.is_empty() {
            jsonline(
                &mut stage_file,
                &json!({"stage":0,"activity":0.,"log_Z":null,"Z":0.,"zero_estimate":true,
                "initial_draws":options.initial_draws,"initial_hits":0,"parents":[],"particles":[]}),
            )?;
            stage_file.flush()?;
            return Ok(
                json!({"schema":"latent-region-smc-summary-v1","complete":true,"zero_estimate":true,
                "log_Z":null,"Z":0.,"initial_draws":options.initial_draws,"initial_hits":0,"completed_stage":0,
                "terminal_particles":[],"ancestry":ancestry(&[]),"reason":"Zero initial mass in the fixed unconditional draw budget; retained without retry."}),
            );
        }
        let (weights, log_sum, initial_ess) = scaled_weights(&initial_logs)?;
        let mut log_z = log_sum - (options.initial_draws as f64).ln();
        let offset =
            stream(options.seed, 0, 0, 0, "resampling").random::<f64>() / options.population as f64;
        let parents = systematic_at_offset(&weights, options.population, offset)?;
        let mut particles: Vec<_> = parents.iter().map(|&i| initial[i].clone()).collect();
        jsonline(
            &mut stage_file,
            &json!({"stage":0,"activity":0.,"initial_draws":options.initial_draws,
            "initial_hits":initial.len(),"initial_weight_ESS":initial_ess,"log_Z_increment":log_z,"log_Z":log_z,
            "resampling_offset":offset,"parent_initial_draws":parents.iter().map(|&i| initial[i].initial_ancestor).collect::<Vec<_>>(),
            "particles":particles,"ancestry":ancestry(&particles)}),
        )?;
        stage_file.flush()?;
        for stage in 1..options.schedule.len() {
            let beta = options.schedule[stage];
            let delta_beta = beta - options.schedule[stage - 1];
            let previous_activity = cfg.reservoir_density * options.schedule[stage - 1];
            let activity = cfg.reservoir_density * options.schedule[stage];
            let delta = activity - previous_activity;
            let lambda = if delta > 0. {
                options.lambda_ratio * delta
            } else {
                1.
            };
            ensure!(
                lambda.is_finite() && lambda > 0.,
                "Invalid incremental count intensity"
            );
            let mut logs = Vec::with_capacity(options.population);
            let mut potential_rows = Vec::with_capacity(options.population);
            for (index, particle) in particles.iter().enumerate() {
                let envelope = if delta > 0. {
                    Some(OverlapEnvelope::build(
                        &env,
                        particle.pose,
                        cfg.endpoint_gate,
                    )?)
                } else {
                    None
                };
                let mut clouds = Vec::with_capacity(options.cloud_replicates);
                for cloud in 0..options.cloud_replicates {
                    clouds.push(if let Some(envelope) = &envelope {
                        overlap_weight::sample_with_envelope(
                            &mut stream(options.seed, stage, index, cloud, "incremental-cloud"),
                            &env,
                            particle.pose,
                            lambda,
                            delta,
                            envelope,
                        )?
                    } else {
                        overlap_weight::OverlapWeight::default()
                    });
                }
                let cloud_logs: Vec<_> = clouds.iter().map(|c| c.log_weight).collect();
                let (_, cloud_log_sum, _) = scaled_weights(&cloud_logs)?;
                let log_g = proposal_log_density(
                    &chart,
                    reference.as_ref(),
                    alpha,
                    log_volume,
                    particle.pose,
                )?;
                ensure!(
                    log_g.is_finite(),
                    "Potential pose lost frozen proposal support"
                );
                let density_correction = if options.bridge == SmcBridge::ProposalDensity {
                    -delta_beta * log_g
                } else {
                    0.
                };
                let log_weight =
                    cloud_log_sum - (options.cloud_replicates as f64).ln() + density_correction;
                logs.push(log_weight);
                potential_rows.push(json!({"particle":index,"pose":particle.pose,"latent":particle.latent,
                    "initial_ancestor":particle.initial_ancestor,"clouds":clouds,"log_g":log_g,"deterministic_log_correction":density_correction,"log_incremental_weight":log_weight}));
            }
            let (weights, log_sum, weight_ess) = scaled_weights(&logs)?;
            let increment = log_sum - (options.population as f64).ln();
            log_z += increment;
            ensure!(log_z.is_finite(), "Nonfinite SMC normalizer");
            let offset = stream(options.seed, stage, 0, 0, "resampling").random::<f64>()
                / options.population as f64;
            let parents = systematic_at_offset(&weights, options.population, offset)?;
            particles = parents.iter().map(|&i| particles[i].clone()).collect();
            let (counts, density_records) = mutate(
                &mut particles,
                &cfg,
                &env,
                &chart,
                &options,
                stage,
                activity,
                beta,
                reference.as_ref(),
                log_volume,
            )?;
            jsonline(
                &mut stage_file,
                &json!({"stage":stage,"beta":beta,"delta_beta":delta_beta,"activity":activity,"previous_activity":previous_activity,
                "delta_activity":delta,"incremental_lambda":lambda,"log_Z_increment":increment,"log_Z":log_z,
                "pre_resampling_weight_ESS":weight_ess,"potentials":potential_rows,"resampling_offset":offset,
                "parents":parents,"mutation_counts":counts,"mutation_density_records":density_records,"particles":particles,"ancestry":ancestry(&particles)}),
            )?;
            stage_file.flush()?;
            completed_stage = stage;
            save(
                &options.out.join("status.json"),
                &json!({"complete":false,"phase":"annealing","completed_stage":stage,"log_Z":log_z}),
            )?;
        }
        let linear_z = log_z.exp();
        Ok(
            json!({"schema":"latent-region-smc-summary-v1","complete":true,"zero_estimate":false,
            "log_Z":log_z,"Z":(linear_z.is_finite() && linear_z > 0.).then_some(linear_z),
            "linear_Z_representable":linear_z.is_finite() && linear_z > 0.,"completed_stage":completed_stage,
            "initial_draws":options.initial_draws,"initial_hits":initial.len(),"initial_weight_ESS":initial_ess,
            "terminal_particles":particles,"ancestry":ancestry(&particles)}),
        )
    })();
    match result {
        Ok(mut summary) => {
            summary["manifest_sha256"] = json!(hash_file(&options.out.join("manifest.json"))?);
            summary["initialization_sha256"] =
                json!(hash_file(&options.out.join("initialization.jsonl"))?);
            summary["stages_sha256"] = json!(hash_file(&options.out.join("stages.jsonl"))?);
            summary["sampler_cpu_seconds"] = json!(cpu_seconds() - cpu_start);
            summary["wall_seconds"] = json!(start.elapsed().as_secs_f64());
            save(&options.out.join("summary.json"), &summary)?;
            save(
                &options.out.join("status.json"),
                &json!({"complete":true,"phase":"complete","completed_stage":completed_stage,
                "summary_sha256":hash_file(&options.out.join("summary.json"))?}),
            )?;
            Ok(summary)
        }
        Err(error) => {
            save(
                &options.out.join("status.json"),
                &json!({"complete":false,"phase":"failed","completed_stage":completed_stage,"error":format!("{error:#}")}),
            )?;
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn systematic_skips_zero_at_zero_offset_and_has_unbiased_counts() -> Result<()> {
        assert_eq!(
            systematic_at_offset(&[0., 1., 0., 1., 0.], 2, 0.)?,
            vec![1, 3]
        );
        let weights = [0., 0.13, 0.27, 0., 0.60];
        let mut counts = [0.; 5];
        for i in 0..1000 {
            for parent in systematic_at_offset(&weights, 7, (i as f64 + 0.5) / 7000.)? {
                counts[parent] += 1.;
            }
        }
        for i in 0..weights.len() {
            assert!((counts[i] / 7000. - weights[i]).abs() < 1. / 7000.);
        }
        assert!(systematic_at_offset(&[0., 0.], 2, 0.).is_err());
        assert!(scaled_weights(&[0., -1000.]).is_err());
        Ok(())
    }
}
