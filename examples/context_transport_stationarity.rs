//! Fixed synthetic IID stationarity control; no protein, atom, or bath queries.
use anyhow::{Result, ensure};
use clap::Parser;
use rand::{SeedableRng, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
use tetramer_mc::{
    basin_involution::{BasinPair, BasinTrace, FixedBasinInvolution},
    context_transport::{OriginalLabelCorrection, apply_selected_charts},
    math::{Pose, add, cayley, invert_relative_pose, matmul, rotation},
    proposal::GaussianComponentParameters,
    simulation::cpu_seconds,
};

const RHO: f64 = 0.65;
const ELL: f64 = 1.3;
const DRAWS: usize = 4096;
const SEEDS: [u64; 4] = [202610040501, 202610040502, 202610040503, 202610040504];
const PRIORS: [f64; 3] = [0.275, 0.275, 0.45];
const ARMS: [&str; 3] = ["adjusted_full", "unadjusted_full", "adjusted_old_shortcut"];
const OBSERVABLES: [&str; 12] = [
    "tx<0",
    "ty<0",
    "tz<0",
    "R00>0",
    "R11>0",
    "R22>0",
    "R01>0",
    "R12>0",
    "R20>0",
    "tx<0_and_R00>0",
    "ty<0_and_R11>0",
    "tz<0_and_R22>0",
];
const POSE_TOLERANCE: f64 = 2e-10;
const FLOW_TOLERANCE: f64 = 2e-9;
const RELATIVE_TOLERANCE: f64 = 2e-11;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    out: PathBuf,
}

fn parameter(index: usize) -> GaussianComponentParameters {
    let lower: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j {
                0.68 + 0.075 * i as f64 + 0.06 * index as f64
            } else if i > j {
                0.018 * (i + j + 1) as f64
            } else {
                0.
            }
        })
    });
    GaussianComponentParameters {
        anchor_position: if index == 0 {
            [-0.8, 0.3, -0.2]
        } else {
            [1., -0.4, 0.3]
        },
        anchor_rotation: cayley(if index == 0 {
            [0.2, -0.4, 0.1]
        } else {
            [-0.5, 0.3, 0.2]
        }),
        mean: [0.; 6],
        covariance: std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
        }),
        weight: 1.,
    }
}

fn chart(parameter: GaussianComponentParameters) -> Result<FixedBasinInvolution> {
    FixedBasinInvolution::new(
        vec![parameter],
        ELL,
        RHO,
        vec![BasinPair {
            first: 0,
            second: 0,
            weight: 1.,
        }],
    )
}

fn branch(label: usize) -> (usize, bool) {
    match label {
        0 => (0, false),
        1 => (0, true),
        2 => (1, false),
        _ => unreachable!(),
    }
}

fn original_logs(maps: &[FixedBasinInvolution], pose: Pose) -> Result<Vec<f64>> {
    (0..3)
        .map(|b| {
            let (k, inverse) = branch(b);
            Ok(PRIORS[b].ln()
                + maps[k].checked_log_density(
                    0,
                    if inverse {
                        invert_relative_pose(pose)
                    } else {
                        pose
                    },
                )?)
        })
        .collect()
}

fn total(logs: &[f64]) -> Result<f64> {
    ensure!(
        logs.iter().all(|v| v.is_finite()),
        "Toy panel reached nonfinite original support"
    );
    let m = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(m + logs.iter().map(|v| (v - m).exp()).sum::<f64>().ln())
}

fn draw_label(rng: &mut StdRng, logs: &[f64]) -> Result<usize> {
    let mut best = f64::NEG_INFINITY;
    let mut selected = None;
    for (i, &log) in logs.iter().enumerate() {
        ensure!(log.is_finite(), "Toy labels require finite support");
        let u: f64 = Open01.sample(rng);
        let score = log - (-u.ln()).ln();
        if score > best {
            best = score;
            selected = Some(i);
        }
    }
    selected.ok_or_else(|| anyhow::anyhow!("Empty original label law"))
}

fn observables(p: Pose) -> [bool; 12] {
    let r = rotation(p.orientation);
    let t = p.position;
    [
        t[0] < 0.,
        t[1] < 0.,
        t[2] < 0.,
        r[0][0] > 0.,
        r[1][1] > 0.,
        r[2][2] > 0.,
        r[0][1] > 0.,
        r[1][2] > 0.,
        r[2][0] > 0.,
        t[0] < 0. && r[0][0] > 0.,
        t[1] < 0. && r[1][1] > 0.,
        t[2] < 0. && r[2][2] > 0.,
    ]
}

fn line(out: &mut BufWriter<fs::File>, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

fn write(path: &std::path::Path, value: &Value) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    writeln!(f)?;
    f.sync_all()?;
    Ok(())
}

#[derive(Clone, Default, Serialize)]
struct Counts {
    attempted: u64,
    accepted: u64,
    up: [u64; 12],
    down: [u64; 12],
    old_true: [u64; 12],
    retained_true: [u64; 12],
    maximum_recovery_error: f64,
    maximum_flow_residual: f64,
    maximum_antisymmetry_error: f64,
    maximum_recovery_ratio: f64,
    maximum_flow_ratio: f64,
    maximum_antisymmetry_ratio: f64,
}

fn run(out: &std::path::Path) -> Result<()> {
    let clock = Instant::now();
    let cpu = cpu_seconds();
    let original_parameters: Vec<_> = (0..2).map(parameter).collect();
    let original: Vec<_> = original_parameters
        .iter()
        .cloned()
        .map(chart)
        .collect::<Result<_>>()?;
    let unadjusted_parameters: Vec<_> = (0..3)
        .map(|b| {
            let (k, inverse) = branch(b);
            let mut center = original[k].decode(0, [0.; 6])?;
            if inverse {
                center = invert_relative_pose(center);
            }
            Ok(GaussianComponentParameters {
                anchor_position: center.position,
                anchor_rotation: rotation(center.orientation),
                mean: [0.; 6],
                covariance: original_parameters[k].covariance,
                weight: 1.,
            })
        })
        .collect::<Result<_>>()?;
    let shifts = [[1.3, -0.4, 0.3], [-0.8, 1.1, 0.2], [0.3, -0.5, 1.2]];
    let angular_shifts = [[0.25, -0.1, 0.15], [-0.2, 0.3, -0.15], [0.1, 0.2, -0.3]];
    let scales = [0.75_f64, 1.2, 1.05];
    let adjusted_parameters: Vec<_> = unadjusted_parameters
        .iter()
        .enumerate()
        .map(|(b, p)| {
            let mut p = p.clone();
            p.anchor_position = add(p.anchor_position, shifts[b]);
            p.anchor_rotation = matmul(cayley(angular_shifts[b]), p.anchor_rotation);
            for row in &mut p.covariance {
                for v in row {
                    *v *= scales[b] * scales[b];
                }
            }
            p
        })
        .collect();
    let unadjusted: Vec<_> = unadjusted_parameters
        .iter()
        .cloned()
        .map(chart)
        .collect::<Result<_>>()?;
    let adjusted: Vec<_> = adjusted_parameters
        .iter()
        .cloned()
        .map(chart)
        .collect::<Result<_>>()?;
    write(
        &out.join("fixture.json"),
        &json!({"schema":"selected-chart-iid-fixture-v1","correlation":RHO,"angular_length":ELL,
        "original_parameters":original_parameters,"branches":(0..3).map(|b|{let(k,inverted)=branch(b);json!({"component_index":k,"inverted":inverted,"prior":PRIORS[b]})}).collect::<Vec<_>>(),
        "adjusted_parameters":adjusted_parameters,"unadjusted_parameters":unadjusted_parameters,"seeds":SEEDS,"draws_per_population":DRAWS,
        "arms":ARMS,"observables":OBSERVABLES,"pose_tolerance":POSE_TOLERANCE,"flow_tolerance":FLOW_TOLERANCE,
        "relative_tolerance":RELATIVE_TOLERANCE,"numerical_checks":"Absolute plus declared intermediate-magnitude scale; diagnostics, not certified roundoff bounds. Every attempt retained; no seam filtering.",
        "source_points":SEEDS.len()*DRAWS,"arm_attempts":SEEDS.len()*DRAWS*3,"event_count":SEEDS.len()*DRAWS*8,
        "rng":"One StdRng::seed_from_u64 stream per population; each draw consumes source Gumbel labels, six source normals, posterior-source Gumbels, prior-target Gumbels, six shared map normals, then one shared Open01 acceptance variate. IID source is always freshly generated, never the retained previous point.",
        "statistical_tests":{"valid_arms":["adjusted_full","unadjusted_full"],"observables":12,"family_alpha":0.05,"bonferroni_tests":24,"per_test_alpha":0.05/24.,"method":"pooled exact conditional two-sided binomial(up,up+down,0.5); retain every stream"},
        "physical_target":"Original MAP-factor normalized mixture G0 in d^3t times normalized Haar",
        "factor_convention":"Every supplied covariance is passed directly once to FixedBasinInvolution; no additional LLT export",
        "center_convention":"All children ordinary physical charts; inverse-origin children do not equal the original reciprocal law",
        "geometry_queries":0,"protein_queries":0,"scope":"Synthetic IID one-step stationarity control; no assembly or mixing-time conclusion"}),
    )?;
    let log_priors = PRIORS.map(f64::ln);
    let mut counts = vec![vec![Counts::default(); 3]; SEEDS.len()];
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("events.jsonl"))?;
    let mut journal = BufWriter::new(file);
    for (population, seed) in SEEDS.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(*seed);
        for draw in 0..DRAWS {
            line(
                &mut journal,
                &json!({"kind":"source_begun","population":population,"draw":draw}),
            )?;
            let generating_branch = draw_label(&mut rng, &log_priors)?;
            let generating_latent = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
            let (k, inverse) = branch(generating_branch);
            let mut old = original[k].decode(0, generating_latent)?;
            if inverse {
                old = invert_relative_pose(old);
            }
            let old_logs = original_logs(&original, old)?;
            let gx = total(&old_logs)?;
            let trace = BasinTrace {
                source: draw_label(&mut rng, &old_logs)?,
                target: draw_label(&mut rng, &log_priors)?,
                noise: std::array::from_fn(|_| StandardNormal.sample(&mut rng)),
            };
            let acceptance_uniform: f64 = Open01.sample(&mut rng);
            let old_observed = observables(old);
            line(
                &mut journal,
                &json!({"kind":"source","population":population,"draw":draw,"generating_branch":generating_branch,
                "generating_latent":generating_latent,"old":old,"old_weighted_logs":old_logs,"trace":trace,
                "acceptance_uniform":acceptance_uniform,"observations_old":old_observed}),
            )?;
            for (arm, name) in ARMS.iter().enumerate() {
                line(
                    &mut journal,
                    &json!({"kind":"arm_begun","population":population,"draw":draw,"arm":name}),
                )?;
                let maps = if arm == 1 { &unadjusted } else { &adjusted };
                let step = apply_selected_charts(
                    (&maps[trace.source], trace.source),
                    (&maps[trace.target], trace.target),
                    old,
                    trace.noise,
                )?;
                let proposed = step.map_step().pose;
                let new_logs = original_logs(&original, proposed)?;
                let gy = total(&new_logs)?;
                let correction = step.correction(&log_priors, &old_logs, &new_logs)?;
                let OriginalLabelCorrection::Finite {
                    log_forward_label_probability: f,
                    log_reverse_label_probability: r,
                    ..
                } = &correction
                else {
                    anyhow::bail!(
                        "Toy reached exact zero reverse support; retain attempted prefix"
                    );
                };
                let used = if arm == 2 {
                    gx - gy
                } else {
                    correction.log_reverse_forward()
                };
                let log_pi_ratio = gy - gx;
                let raw_log_acceptance = log_pi_ratio + used;
                let log_acceptance = raw_log_acceptance.min(0.);
                ensure!(log_acceptance.is_finite(), "Nonfinite toy acceptance");
                let accepted = acceptance_uniform.ln() < log_acceptance;
                let retained = if accepted { proposed } else { old };
                let inverse = &step.map_step().inverse_trace;
                let back = apply_selected_charts(
                    (&maps[inverse.source], inverse.source),
                    (&maps[inverse.target], inverse.target),
                    proposed,
                    inverse.noise,
                )?;
                let reverse_correction = back
                    .correction(&log_priors, &new_logs, &old_logs)?
                    .log_reverse_forward();
                let pose_error = old
                    .position
                    .iter()
                    .zip(back.map_step().pose.position)
                    .map(|(a, b)| (a - b).abs())
                    .chain(
                        rotation(old.orientation)
                            .into_iter()
                            .flatten()
                            .zip(
                                rotation(back.map_step().pose.orientation)
                                    .into_iter()
                                    .flatten(),
                            )
                            .map(|(a, b)| (a - b).abs()),
                    )
                    .fold(0_f64, f64::max);
                let noise_error = trace
                    .noise
                    .into_iter()
                    .zip(back.map_step().inverse_trace.noise)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0_f64, f64::max);
                let antisymmetry = (correction.log_reverse_forward() + reverse_correction).abs();
                let log_phi = |v: [f64; 6]| -0.5 * v.iter().map(|v| v * v).sum::<f64>();
                let forward = gx + f + log_phi(trace.noise);
                let reverse =
                    gy + r + log_phi(inverse.noise) + step.map_step().log_extended_jacobian;
                let flow_residual = (forward + (log_pi_ratio + used).min(0.)
                    - reverse
                    - (-log_pi_ratio - used).min(0.))
                .abs();
                let recovery_scale = 1.
                    + old
                        .position
                        .into_iter()
                        .chain(proposed.position)
                        .chain(back.map_step().pose.position)
                        .chain(step.map_step().source_latent)
                        .chain(step.map_step().target_latent)
                        .chain(trace.noise)
                        .chain(inverse.noise)
                        .chain(back.map_step().inverse_trace.noise)
                        .map(f64::abs)
                        .fold(0_f64, f64::max);
                let flow_scale = 1.
                    + [
                        gx,
                        gy,
                        *f,
                        *r,
                        log_phi(trace.noise),
                        log_phi(inverse.noise),
                        step.map_step().log_extended_jacobian,
                        correction.log_reverse_forward(),
                        reverse_correction,
                    ]
                    .into_iter()
                    .map(f64::abs)
                    .sum::<f64>();
                let recovery_tolerance = POSE_TOLERANCE + RELATIVE_TOLERANCE * recovery_scale;
                let flow_tolerance = FLOW_TOLERANCE + RELATIVE_TOLERANCE * flow_scale;
                ensure!(
                    recovery_tolerance.is_finite() && flow_tolerance.is_finite(),
                    "Unrepresentable numerical diagnostic scale"
                );
                let retained_observed = observables(retained);
                let c = &mut counts[population][arm];
                c.attempted += 1;
                c.accepted += u64::from(accepted);
                c.maximum_recovery_error =
                    c.maximum_recovery_error.max(pose_error.max(noise_error));
                c.maximum_antisymmetry_error = c.maximum_antisymmetry_error.max(antisymmetry);
                c.maximum_flow_residual = c.maximum_flow_residual.max(flow_residual);
                c.maximum_recovery_ratio = c
                    .maximum_recovery_ratio
                    .max(pose_error.max(noise_error) / recovery_tolerance);
                c.maximum_flow_ratio = c.maximum_flow_ratio.max(flow_residual / flow_tolerance);
                c.maximum_antisymmetry_ratio = c
                    .maximum_antisymmetry_ratio
                    .max(antisymmetry / flow_tolerance);
                for k in 0..12 {
                    c.old_true[k] += u64::from(old_observed[k]);
                    c.retained_true[k] += u64::from(retained_observed[k]);
                    c.up[k] += u64::from(!old_observed[k] && retained_observed[k]);
                    c.down[k] += u64::from(old_observed[k] && !retained_observed[k]);
                }
                line(
                    &mut journal,
                    &json!({"kind":"arm_result","population":population,"draw":draw,"arm":name,
                    "step":step,"correction":correction,"new_weighted_logs":new_logs,"log_pi_ratio":log_pi_ratio,
                    "used_log_correction":used,"raw_log_acceptance":raw_log_acceptance,"log_acceptance":log_acceptance,"accepted":accepted,"retained":retained,
                    "observations_retained":retained_observed,"reverse_recovered_pose":back.map_step().pose,
                    "reverse_recovered_noise":back.map_step().inverse_trace.noise,"reverse_log_correction":reverse_correction,
                    "pose_recovery_error":pose_error,"noise_recovery_error":noise_error,"correction_antisymmetry_error":antisymmetry,
                    "pointwise_flow_residual":flow_residual,"recovery_scale":recovery_scale,"flow_scale":flow_scale,
                    "recovery_tolerance":recovery_tolerance,"flow_tolerance":flow_tolerance}),
                )?;
                ensure!(
                    pose_error.max(noise_error) <= recovery_tolerance
                        && antisymmetry <= flow_tolerance,
                    "Toy inverse consistency failed"
                );
                ensure!(
                    arm == 2 || flow_residual <= flow_tolerance,
                    "Corrected toy pointwise flow failed"
                );
            }
        }
    }
    journal.flush()?;
    journal.get_ref().sync_all()?;
    write(
        &out.join("summary.json"),
        &json!({"schema":"selected-chart-iid-summary-v1","complete":true,"passed":true,
        "populations":SEEDS.len(),"draws_per_population":DRAWS,"source_points":SEEDS.len()*DRAWS,"arm_attempts":SEEDS.len()*DRAWS*3,
        "events":SEEDS.len()*DRAWS*8,"arms":ARMS,"counts_by_population":counts,"cpu_seconds":cpu_seconds()-cpu,"wall_seconds":clock.elapsed().as_secs_f64(),
        "geometry_queries":0,"protein_queries":0,"statistical_admission":"Pending independent reconstruction and exact paired-binomial reducer; lifecycle passed is not statistical admission"}),
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "Fresh synthetic output required");
    fs::create_dir(&args.out)?;
    if let Err(error) = run(&args.out) {
        write(
            &args.out.join("failure.json"),
            &json!({"complete":false,"passed":false,"error":format!("{error:#}"),"partial_attempts_retained":true}),
        )?;
        return Err(error);
    }
    Ok(())
}
