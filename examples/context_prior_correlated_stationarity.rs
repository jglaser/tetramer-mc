//! Fixed rho=.95 cached-IID physical reference for the unchanged shared singleton step.
//! No equilibrium source generation and no trajectory extension are performed.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use tetramer_mc::{
    bounded_singleton_path::{Budget, Limits},
    context_docking::{PhysicalStep, StepRngs, physical_step},
    depletion::GateOptions,
    docking::{DockingConfig, DockingMethod, DockingProposal, defensive_virtual_branch_log_prior},
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, cayley, rotation},
    proposal::FrozenRelativePoseProposal,
    simulation::cpu_seconds,
    spherical::Container,
};

const CORRELATION: f64 = 0.95;
const STREAMS: usize = 4;
const PER_STREAM: usize = 2048;
const ARMS: [&str; 2] = ["original_prior", "context_prior"];
const ORIGIN: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};
const NAMES: [&str; 12] = [
    "moving_x_negative",
    "moving_y_negative",
    "moving_z_negative",
    "moving_radius_squared_below_one",
    "pair_separation_squared_below_one",
    "position_dot_positive",
    "moving_R00_positive",
    "moving_R11_positive",
    "moving_R22_positive",
    "relative_R00_positive",
    "x_negative_and_R00_positive",
    "pair_close_and_R11_positive",
];
const DESIGN_SHA: &str = "d61dd358a0b59a50648c9dc286ec71b79ffb43639d6be48b7f19c9d7e55c3657";

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    source_cache: PathBuf,
    #[arg(long)]
    source_cache_sha256: String,
    #[arg(long)]
    source_receipt: PathBuf,
    #[arg(long)]
    source_receipt_sha256: String,
    #[arg(long)]
    design: PathBuf,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Source {
    kind: String,
    stream: usize,
    source_index: usize,
    source_attempt: u64,
    old: [Pose; 2],
}
#[derive(Default, Serialize)]
struct Counts {
    attempted: u64,
    learned: u64,
    uniform: u64,
    hard_rejected: u64,
    physical_decisions: u64,
    accepted: u64,
    learned_nonzero_ratio: u64,
    accepted_learned_nonzero_ratio: u64,
    gained: u64,
    lost: u64,
    raw_points: u64,
    retained_points: u64,
}
fn sha(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}
fn write(path: &Path, value: &Value) -> Result<()> {
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    writeln!(f)?;
    f.sync_all()?;
    Ok(())
}
fn emit(journal: &mut BufWriter<File>, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *journal, value)?;
    writeln!(journal)?;
    journal.flush()?;
    Ok(())
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}
fn observed(pair: [Pose; 2]) -> [bool; 12] {
    let x = pair[0].position;
    let y = pair[1].position;
    let d = std::array::from_fn(|k| x[k] - y[k]);
    let a = rotation(pair[0].orientation);
    let b = rotation(pair[1].orientation);
    let close = dot(d, d) < 1.;
    [
        x[0] < 0.,
        x[1] < 0.,
        x[2] < 0.,
        dot(x, x) < 1.,
        close,
        dot(x, y) > 0.,
        a[0][0] > 0.,
        a[1][1] > 0.,
        a[2][2] > 0.,
        (a[0][0] * b[0][0] + a[1][0] * b[1][0]) + a[2][0] * b[2][0] > 0.,
        x[0] < 0. && a[0][0] > 0.,
        close && a[1][1] > 0.,
    ]
}
fn hard_valid(pair: [Pose; 2]) -> bool {
    pair.iter().all(|p| {
        dot(p.position, p.position) <= 1.5_f64.powi(2)
            && dot(p.position, p.position) >= 0.2_f64.powi(2)
    }) && dot(
        std::array::from_fn(|k| pair[0].position[k] - pair[1].position[k]),
        std::array::from_fn(|k| pair[0].position[k] - pair[1].position[k]),
    ) >= 0.2_f64.powi(2)
}
fn role_seed(stream: usize, index: usize, role: &str) -> [u8; 32] {
    Sha256::digest(
        format!("context-prior-cached-iid-correlated095-reference-v1\0{stream}\0{index}\0{role}")
            .as_bytes(),
    )
    .into()
}
fn rngs(stream: usize, index: usize) -> StepRngs {
    StepRngs {
        proposal: StdRng::from_seed(role_seed(stream, index, "proposal")),
        gate: StdRng::from_seed(role_seed(stream, index, "bath")),
        accept: StdRng::from_seed(role_seed(stream, index, "accept")),
    }
}
fn atlas_model(shape_sha: &str) -> Value {
    // Same fixed toy as partner_atlas_stationarity::atlas_model. No fitting.
    let mut lower = [[0.; 6]; 6];
    for (i, s) in [0.4, 0.35, 0.45, 0.6, 0.5, 0.65].into_iter().enumerate() {
        lower[i][i] = s;
    }
    lower[3][0] = 0.12;
    lower[5][1] = -0.1;
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    let base = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":shape_sha,
        "angular_length":1.,"weights":[0.35,0.65],"anchors":[
        {"position":[0.8,0.1,-0.05],"rotation":cayley([0.1,-0.05,0.15])},
        {"position":[-0.3,0.8,0.15],"rotation":cayley([-0.1,0.15,0.05])}],
        "means":vec![[0.;6];2],"covariances":[covariance,covariance]});
    json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,"reciprocal_components":[true,false]})
}
fn load_sources(cli: &Cli) -> Result<(Vec<Source>, Value, Value)> {
    ensure!(
        cli.source_cache.is_absolute()
            && cli.source_receipt.is_absolute()
            && cli.design.is_absolute(),
        "Absolute inputs required"
    );
    ensure!(
        sha(&cli.source_cache)? == cli.source_cache_sha256
            && sha(&cli.source_receipt)? == cli.source_receipt_sha256,
        "Changed source authority"
    );
    ensure!(
        sha(&cli.design)? == DESIGN_SHA,
        "Changed prospective design"
    );
    ensure!(
        cli.source_cache.metadata()?.len() <= 32 * 1024 * 1024,
        "Source byte cap"
    );
    let receipt = read(&cli.source_receipt)?;
    ensure!(
        receipt["schema"] == "partner-atlas-iid-source-cache-v1"
            && receipt["complete"] == true
            && receipt["passed"] == true
            && receipt["streams"] == STREAMS
            && receipt["sources_per_stream"] == PER_STREAM
            && receipt["total_sources"] == STREAMS * PER_STREAM
            && receipt["sources"]
                == json!({"path":cli.source_cache,"sha256":cli.source_cache_sha256}),
        "Invalid cached IID source receipt"
    );
    let binding = &receipt["original_protocol"];
    let path = Path::new(binding["path"].as_str().context("Source protocol path")?);
    ensure!(
        sha(path)? == binding["sha256"].as_str().context("Source protocol hash")?,
        "Changed original source protocol"
    );
    let protocol = read(path)?;
    ensure!(
        protocol["schema"] == "flexible-surrogate-independent-reference-protocol-v1"
            && protocol["source_sampler_exact"] == true
            && protocol["streams"] == 4
            && protocol["sources_per_stream"] == 2048
            && protocol["fixed_spectator"] == json!(ORIGIN),
        "Wrong IID source law"
    );
    for (k, v) in [
        ("core_radius", 0.1),
        ("rd", 0.9),
        ("inflated_radius", 1.),
        ("wall_radius", 1.6),
        ("center_radius", 1.5),
        ("activity", 0.5),
        ("lambda", 2.),
    ] {
        ensure!(
            protocol[k].as_f64() == Some(v),
            "Wrong physical source field {k}"
        );
    }
    let mut rows = Vec::new();
    let mut previous = [0; STREAMS];
    for line in BufReader::new(File::open(&cli.source_cache)?).lines() {
        let line = line?;
        ensure!(
            line.len() <= 4096 && rows.len() < STREAMS * PER_STREAM,
            "Source row cap"
        );
        let r: Source = serde_json::from_str(&line)?;
        let n = rows.len();
        ensure!(
            r.kind == "cached_iid_source"
                && r.stream == n / PER_STREAM
                && r.source_index == n % PER_STREAM
                && r.source_attempt > previous[r.stream],
            "Changed source identity/order"
        );
        r.old[0].validate()?;
        r.old[1].validate()?;
        ensure!(hard_valid(r.old), "Invalid cached physical source");
        previous[r.stream] = r.source_attempt;
        rows.push(r);
    }
    ensure!(rows.len() == 8192, "Incomplete source allocation");
    Ok((rows, receipt, protocol))
}
fn add_counts(c: &mut Counts, step: &PhysicalStep) -> Result<()> {
    c.attempted += 1;
    let learned = step.proposal["branch"] == "involution";
    if learned {
        c.learned += 1;
    } else {
        ensure!(step.proposal["branch"] == "uniform", "Unexpected branch");
        c.uniform += 1;
    }
    if step.status == "wall_rejected" || step.status == "core_rejected" {
        c.hard_rejected += 1;
    }
    if let Some(g) = step.gate {
        c.physical_decisions += 1;
        c.accepted += u64::from(step.accepted);
        c.gained += g.gained;
        c.lost += g.lost;
        c.raw_points += g.raw_points;
        c.retained_points += g.retained_points;
        if learned
            && step
                .log_proposal_reverse_forward
                .context("Missing ratio")?
                .abs()
                > 1e-9
        {
            c.learned_nonzero_ratio += 1;
            c.accepted_learned_nonzero_ratio += u64::from(step.accepted);
        }
    }
    Ok(())
}
fn exercises(c: &Counts) -> bool {
    c.attempted == 8192
        && c.learned > 100
        && c.uniform > 100
        && c.hard_rejected > 0
        && c.physical_decisions > 100
        && c.accepted > 50
        && c.learned_nonzero_ratio > 20
        && c.accepted_learned_nonzero_ratio > 0
        && c.gained > 0
        && c.lost > 0
}
fn run(cli: Cli) -> Result<()> {
    let started = cpu_seconds();
    let wall_clock = std::time::Instant::now();
    let (sources, authority, source_protocol) = load_sources(&cli)?;
    ensure!(
        cli.out.is_absolute() && !cli.out.exists(),
        "Fresh absolute output required"
    );
    fs::create_dir(&cli.out)?;
    let shape = Shape {
        name: "reference sphere".into(),
        volume: 4. * std::f64::consts::PI * 0.1_f64.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 0.1,
        }],
    };
    let shape_bytes = serde_json::to_vec(&shape)?;
    let shape_sha = format!("{:x}", Sha256::digest(&shape_bytes));
    let tree = SphereTree::new(shape)?;
    let wall = Container::new(1.6, &tree)?;
    let model_value = atlas_model(&shape_sha);
    let text = serde_json::to_string(&model_value)?;
    let mut proposals = Vec::new();
    let mut prior_vectors = Vec::new();
    for mask in [None, Some([true, false, false]), Some([false, true, false])] {
        let model =
            FrozenRelativePoseProposal::from_json_str_open(&text, [3.; 3], 0.5, &shape_sha)?;
        let mut p = DockingProposal::new(
            model,
            DockingMethod::PosteriorInvolution,
            CORRELATION,
            [0.; 3],
        )?
        .with_anchor_index(Some(0));
        if let Some(flags) = mask {
            let logs =
                defensive_virtual_branch_log_prior(p.virtual_branch_log_prior(), &flags, 0.1)?;
            p = p.with_virtual_branch_log_prior(&logs)?;
        }
        prior_vectors.push(p.virtual_branch_log_prior().to_vec());
        proposals.push(p);
    }
    let mut cfg = DockingConfig {
        shape: cli.out.join("shape.json"),
        fixed_poses: vec![ORIGIN, ORIGIN],
        initial_pose: ORIGIN,
        capture_center: [0.; 3],
        capture_radius: 1.6,
        depletant_radius: 0.9,
        reservoir_density: 0.5,
        poisson_lambda_ratio: 4.,
        translation_steps: vec![0.25],
        rotation_steps_deg: vec![30.],
        rotation_probability: 0.5,
        local_attempts_per_cycle: 4,
        uniform_probability: 0.5,
        seed: 0,
        endpoint_gate: GateOptions {
            max_cells: 255,
            max_depth: 8,
            min_width: 0.,
        },
        metadata: Value::Null,
        proposal_anchor_index: Some(0),
        target_region: None,
    };
    let limits = Limits {
        raw_per_leg: 100000,
        raw_per_outer: 100000,
        raw_campaign: 10000000,
        retained_per_leg: 100000,
        retained_per_outer: 100000,
        retained_campaign: 10000000,
        cpu_seconds: 300.,
    };
    let mut budget = Budget::new(limits)?;
    let compiled = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    write(
        &cli.out.join("shape.json"),
        &serde_json::from_slice(&shape_bytes)?,
    )?;
    write(&cli.out.join("model.json"), &model_value)?;
    fs::write(cli.out.join("source-bundle.json"), compiled)?;
    let protocol = json!({"schema":"context-prior-correlated095-physical-reference-v1","correlation":CORRELATION,"complete":true,
        "design":read(&cli.design)?,"design_sha256":DESIGN_SHA,"source_cache":{"path":cli.source_cache,"sha256":cli.source_cache_sha256},
        "source_receipt":{"path":cli.source_receipt,"sha256":cli.source_receipt_sha256},"source_authority":authority,
        "original_protocol":source_protocol,"atlas_model":model_value,"effective_log_priors":prior_vectors,
        "source_example_sha256":format!("{:x}",Sha256::digest(include_bytes!("context_prior_correlated_stationarity.rs"))),
        "source_bundle_sha256":format!("{:x}",Sha256::digest(compiled.as_bytes())),"observable_names":NAMES,
        "fixed_vector_order":["origin label2","unchanged cached slot1"],"model_anchor_index":0,
        "null_policy":"Unexpected numerical or support null is fatal with partial trace, no replacement.",
        "event_count":49152,"source_points":8192,"physical_calls":16384,"negative_algebraic_decisions":8192,
        "limits":limits,"all_attempts_retained":true,"new_source_draws":0});
    write(&cli.out.join("protocol.json"), &protocol)?;
    let mut journal = BufWriter::new(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(cli.out.join("events.jsonl"))?,
    );
    let mut counts: [Counts; 2] = std::array::from_fn(|_| Counts::default());
    // [arm including negative][stream][observable][source0/source1/up/down].
    let mut paired = [[[[0_u64; 4]; 12]; 4]; 3];
    let mut events = 0_u64;
    for source in &sources {
        budget.check_cpu()?;
        ensure!(
            wall_clock.elapsed().as_secs_f64() < 600.,
            "Wall budget exceeded"
        );
        emit(
            &mut journal,
            &json!({"kind":"source_begin","stream":source.stream,"source_index":source.source_index,"source_attempt":source.source_attempt}),
        )?;
        events += 1;
        let before = observed(source.old);
        let prior_index = if source.old[1].position[0] >= 0. {
            1
        } else {
            2
        };
        emit(
            &mut journal,
            &json!({"kind":"source","source":source,"observations":before,"context_prior_index":prior_index}),
        )?;
        events += 1;
        cfg.fixed_poses = vec![ORIGIN, source.old[1]];
        cfg.initial_pose = source.old[0];
        cfg.validate()?;
        for (arm_index, arm) in ARMS.iter().enumerate() {
            emit(
                &mut journal,
                &json!({"kind":"arm_begin","stream":source.stream,"source_index":source.source_index,"arm":arm}),
            )?;
            events += 1;
            let mut rng = rngs(source.stream, source.source_index);
            let proposer = &proposals[if arm_index == 0 { 0 } else { prior_index }];
            let step = match physical_step(
                &tree,
                &wall,
                &cfg.fixed_poses,
                source.old[0],
                &cfg,
                Some(proposer),
                &mut rng,
                &mut budget,
            ) {
                Ok(s) => s,
                Err(error) => {
                    emit(
                        &mut journal,
                        &json!({"kind":"arm_fatal","stream":source.stream,"source_index":source.source_index,"arm":arm,"failure":error}),
                    )?;
                    write(
                        &cli.out.join("failure.json"),
                        &json!({"complete":false,"passed":false,"events_before_fatal":events,
                        "stream":source.stream,"source_index":source.source_index,"arm":arm,"reason":error.to_string(),"raw_completed":budget.raw,"retained_completed":budget.retained}),
                    )?;
                    anyhow::bail!(error);
                }
            };
            add_counts(&mut counts[arm_index], &step)?;
            let retained = [step.retained_pose, source.old[1]];
            ensure!(hard_valid(retained), "Invalid retained physical source");
            let after = observed(retained);
            let mut negative = Value::Null;
            let mut neg_after = before;
            if arm_index == 1 {
                let mut take = false;
                let mut pose = source.old[0];
                let mut ratio = None;
                if let Some(g) = step.gate {
                    let u = step
                        .acceptance_uniform
                        .context("Complete gate missing uniform")?;
                    ratio = Some(g.log_weight.min(0.));
                    take = u.ln() < g.log_weight.min(0.);
                    if take {
                        pose = step
                            .proposed_pose
                            .context("Complete gate missing candidate")?;
                    }
                }
                neg_after = observed([pose, source.old[1]]);
                negative = json!({"id":"context_omitted_proposal_ratio","accepted":take,"retained_pose":pose,"log_acceptance":ratio,"observations":neg_after});
            }
            for (index, values) in [(arm_index, after), (2, neg_after)] {
                if index == 2 && arm_index == 0 {
                    continue;
                }
                for k in 0..12 {
                    let c = &mut paired[index][source.stream][k];
                    c[usize::from(before[k])] += 1;
                    c[2] += u64::from(!before[k] && values[k]);
                    c[3] += u64::from(before[k] && !values[k]);
                }
            }
            emit(
                &mut journal,
                &json!({"kind":"arm_result","stream":source.stream,"source_index":source.source_index,"arm":arm,
                "step":step,"observations_retained":after,"negative":negative,
                "completed_raw_points":budget.raw,"completed_retained_points":budget.retained}),
            )?;
            events += 1;
        }
    }
    journal.flush()?;
    journal.get_ref().sync_all()?;
    ensure!(
        events == 49152 && counts.iter().all(|c| c.attempted == 8192),
        "Incomplete fixed allocation"
    );
    ensure!(
        sha(&cli.source_cache)? == cli.source_cache_sha256
            && sha(&cli.source_receipt)? == cli.source_receipt_sha256
            && sha(&cli.design)? == DESIGN_SHA,
        "Inputs changed during execution"
    );
    write(
        &cli.out.join("summary.json"),
        &json!({"schema":"context-prior-correlated095-physical-summary-v1","correlation":CORRELATION,"complete":true,"passed":true,
        "execution_passed":true,"exercise_passed":counts.iter().all(exercises),"statistical_assessment":"Independent audit required; execution pass alone is not statistical pass",
        "arms":ARMS,"counts":counts,"paired_counts":paired,"paired_layout":["source0","source1","up","down"],
        "events":events,"source_points":8192,"physical_calls":16384,"negative_algebraic_decisions":8192,
        "raw_points":budget.raw,"retained_points":budget.retained,"cpu_seconds":cpu_seconds()-started,"wall_seconds":wall_clock.elapsed().as_secs_f64(),
        "new_source_draws":0,"retries":0,"replacements":0,"extensions":0,
        "protocol_sha256":sha(&cli.out.join("protocol.json"))?,"events_sha256":sha(&cli.out.join("events.jsonl"))?}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    run(Cli::parse())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetramer_mc::{
        basin_involution::{BasinPair, BasinTrace, FixedBasinInvolution},
        math::invert_relative_pose,
    };

    fn close(a: f64, b: f64, scale: f64) {
        assert!(a.is_finite() && b.is_finite());
        assert!(
            (a - b).abs() <= 2e-8 + 2e-10 * scale,
            "{a} differs from {b}, scale {scale}"
        );
    }

    fn norm2(v: &[f64; 6]) -> f64 {
        v.iter().map(|x| x * x).sum()
    }

    fn logsum(v: &[f64]) -> f64 {
        let maximum = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        maximum + v.iter().map(|x| (x - maximum).exp()).sum::<f64>().ln()
    }

    #[test]
    fn correlated095_every_virtual_pair_reverses_and_has_posterior_correction() -> Result<()> {
        let shape_sha = "a".repeat(64);
        let text = serde_json::to_string(&atlas_model(&shape_sha))?;
        let model =
            FrozenRelativePoseProposal::from_json_str_open(&text, [3.; 3], 0.5, &shape_sha)?;
        let branches = model.virtual_branches();
        assert_eq!(branches.len(), 3);
        assert_eq!(
            branches.iter().map(|b| b.inverted).collect::<Vec<_>>(),
            vec![false, true, false]
        );
        let base = model.component_parameters();
        let parameters = branches
            .iter()
            .map(|b| {
                let mut p = base[b.component_index].clone();
                p.weight = b.weight;
                p
            })
            .collect();
        let mut pairs = Vec::new();
        for i in 0..3 {
            for j in i..3 {
                pairs.push(BasinPair {
                    first: i,
                    second: j,
                    weight: branches[i].weight * branches[j].weight * if i == j { 1. } else { 2. },
                });
            }
        }
        let map = FixedBasinInvolution::new(parameters, 1., CORRELATION, pairs)?;
        let mut proposals = Vec::new();
        for mask in [None, Some([true, false, false]), Some([false, true, false])] {
            let model =
                FrozenRelativePoseProposal::from_json_str_open(&text, [3.; 3], 0.5, &shape_sha)?;
            let mut p = DockingProposal::new(
                model,
                DockingMethod::PosteriorInvolution,
                CORRELATION,
                [0.; 3],
            )?;
            if let Some(eligible) = mask {
                let weights = defensive_virtual_branch_log_prior(
                    p.virtual_branch_log_prior(),
                    &eligible,
                    0.1,
                )?;
                p = p.with_virtual_branch_log_prior(&weights)?;
            }
            proposals.push(p);
        }
        let s = ((1. - CORRELATION) * (1. + CORRELATION)).sqrt();
        let sources = [
            [0.; 6],
            [0.4, -0.3, 0.2, 0.1, 0.2, -0.1],
            [8., -6., 4., 3., -2., 1.],
        ];
        let noises = [
            [0.2, -0.6, 1.1, -0.3, 0.9, 0.4],
            [-1., 0.2, 0.4, 0.7, -0.5, 0.3],
        ];
        let mut traces = 0;
        for i in 0..3 {
            for j in 0..3 {
                for u in sources {
                    let decoded = map.decode(i, u)?;
                    let old = if branches[i].inverted {
                        invert_relative_pose(decoded)
                    } else {
                        decoded
                    };
                    for noise in noises {
                        let trace = BasinTrace {
                            source: i,
                            target: j,
                            noise,
                        };
                        let step = proposals[0].apply_relative_trace(old, &trace)?;
                        let v = std::array::from_fn(|k| CORRELATION * u[k] + s * noise[k]);
                        let reverse_noise =
                            std::array::from_fn(|k| s * u[k] - CORRELATION * noise[k]);
                        let scale =
                            1. + norm2(&u) + norm2(&noise) + norm2(&v) + norm2(&reverse_noise);
                        for k in 0..6 {
                            close(step.source_latent[k], u[k], scale);
                            close(step.target_latent[k], v[k], scale);
                            close(step.inverse_trace.noise[k], reverse_noise[k], scale);
                        }
                        close(
                            norm2(&u) + norm2(&noise),
                            norm2(&v) + norm2(&reverse_noise),
                            scale,
                        );
                        // This re-encodes the actual generated destination, then
                        // applies the reverse noise and reciprocal wrapper.
                        let back =
                            proposals[0].apply_relative_trace(step.pose, &step.inverse_trace)?;
                        for k in 0..6 {
                            close(back.source_latent[k], v[k], scale);
                            close(back.target_latent[k], u[k], scale);
                            close(back.inverse_trace.noise[k], noise[k], scale);
                        }
                        for k in 0..3 {
                            close(back.pose.position[k], old.position[k], scale);
                            for l in 0..3 {
                                close(
                                    rotation(back.pose.orientation)[k][l],
                                    rotation(old.orientation)[k][l],
                                    scale,
                                );
                            }
                        }
                        close(
                            step.log_correction,
                            -back.log_correction,
                            1. + step.log_correction.abs() + back.log_correction.abs(),
                        );
                        for p in &proposals {
                            let a = p.branch_log_densities(old, ORIGIN)?;
                            let b = p.branch_log_densities(step.pose, ORIGIN)?;
                            let w = p.virtual_branch_log_prior();
                            let (gx, gy) = (logsum(&a), logsum(&b));
                            let labels = (b[j] - gy) + w[i] - (a[i] - gx) - w[j];
                            let bound = 1.
                                + gx.abs()
                                + gy.abs()
                                + a[i].abs()
                                + b[j].abs()
                                + w[i].abs()
                                + w[j].abs()
                                + step.log_correction.abs()
                                + labels.abs();
                            close(step.log_correction, (a[i] - w[i]) - (b[j] - w[j]), bound);
                            close(step.log_correction + labels, gx - gy, bound);
                        }
                        traces += 1;
                    }
                }
            }
        }
        assert_eq!(traces, 54);
        println!(
            "54 forward,54 inverse maps;162 prior correction checks; no geometry or bath calls"
        );
        Ok(())
    }

    #[test]
    fn correlated095_uses_new_role_namespace_without_reseeding_sources() {
        let previous: [u8; 32] =
            Sha256::digest(b"context-prior-cached-iid-physical-reference-v1\x000\x000\x00proposal")
                .into();
        assert_ne!(role_seed(0, 0, "proposal"), previous);
        assert_ne!(role_seed(0, 0, "proposal"), role_seed(0, 0, "bath"));
        assert_eq!(role_seed(0, 0, "proposal"), role_seed(0, 0, "proposal"));
        assert_ne!(role_seed(0, 0, "proposal"), role_seed(0, 1, "proposal"));
    }
}
