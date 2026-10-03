//! Independent hard-only sphere sources; guided one-step stationarity control.
//! Fixed bounded allocation and an explicit omitted-correction negative control.
use anyhow::{Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use std::{
    f64::consts::PI,
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    auxiliary_overlap_threshold::AuxiliaryOverlapThreshold,
    capped_dimer::FixedDimerContext,
    defensive_dimer_proposal::DefensiveDimerProposal,
    dimer_tree_proposal::tree_members,
    docking::{DockingMethod, DockingProposal},
    factorized_dimer::{FactorizedDimerCaps, FactorizedDimerOrder, FactorizedDimerProposal},
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, rotation},
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
};

const SOURCE: &[u8] = include_bytes!("auxiliary_overlap_sphere_control.rs");
const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const SHAPE_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const ID: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};
const MASTER: u64 = 6100300201;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn seed(pop: usize, slot: usize, role: &str) -> u64 {
    let h =
        hash_bytes(format!("auxiliary-overlap-sphere-v1/{MASTER}/{pop}/{slot}/{role}").as_bytes());
    u64::from_str_radix(&h[..16], 16).unwrap()
}
fn sphere(radius: f64) -> Result<SphereTree> {
    SphereTree::new(Shape {
        name: "auxiliary threshold reference sphere".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    })
}
fn atlas() -> Result<DockingProposal> {
    let covariance: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let model=json!({"coordinate_convention":"anchor-body-relative","shape_sha256":SHAPE_SHA,
        "angular_length":1.,"weights":[1.],"anchors":[{"position":[0.,0.,0.],"rotation":rotation(ID.orientation)}],
        "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance]}).to_string();
    DockingProposal::new(
        FrozenRelativePoseProposal::from_json_str_open(&model, [80.; 3], 0.1, SHAPE_SHA)?,
        DockingMethod::PosteriorInvolution,
        0.7,
        [0.; 3],
    )
}
fn normalize<const N: usize>(x: [f64; N]) -> Result<[f64; N]> {
    let n = x.iter().map(|a| a * a).sum::<f64>().sqrt();
    ensure!(n.is_finite() && n > 0., "invalid reference normal vector");
    Ok(x.map(|a| a / n))
}
fn cloud() -> Vec<[f64; 3]> {
    let mut points = vec![];
    for i in -4..=4 {
        for j in -4..=4 {
            for k in -4..=4 {
                let p = [i as f64 * 0.1, j as f64 * 0.1, k as f64 * 0.1];
                if p.iter().map(|x| x * x).sum::<f64>() <= 0.45f64.powi(2) {
                    points.push(p);
                }
            }
        }
    }
    points
}

fn source(
    rng: &mut StdRng,
    core: &SphereTree,
    exclusion: &SphereTree,
    trace: &mut Vec<Value>,
) -> Result<[Pose; 2]> {
    for _ in 0..1000 {
        let root_uniform: [f64; 3] = std::array::from_fn(|_| rng.random());
        let root_normals: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
        let direction_normals: [f64; 3] = std::array::from_fn(|_| StandardNormal.sample(rng));
        let radial_uniform: f64 = rng.random();
        let relative_normals: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
        trace.push(json!({"root_uniform":root_uniform,"root_normals":root_normals,"direction_normals":direction_normals,
            "radial_uniform":radial_uniform,"relative_normals":relative_normals,"poses":null,"feasibility":null,"status":"in_progress"}));
        let radius = (0.1f64.powi(3) + radial_uniform * (0.9f64.powi(3) - 0.1f64.powi(3))).cbrt();
        let root = Pose {
            position: root_uniform.map(|u| 2. * u - 1.),
            orientation: normalize(root_normals)?,
        };
        let relative = Pose {
            position: normalize(direction_normals)?.map(|x| x * radius),
            orientation: normalize(relative_normals)?,
        };
        let poses = tree_members(ID, [root, relative])?;
        let state = [poses[0], poses[1], ID];
        let context = FixedDimerContext::new(core, exclusion, &state, [0, 1], 2, None, [0.; 3])?;
        let feasible = context.evaluate(poses)?;
        let record = trace.last_mut().unwrap();
        record["poses"] = json!(poses);
        record["feasibility"] = json!(feasible);
        record["status"] = json!("complete");
        if feasible.feasible() {
            return Ok(poses);
        }
    }
    anyhow::bail!("fatal reference source budget; no replacement outer");
}

fn run(out: &PathBuf, config: &Value, writer: &mut impl Write) -> Result<Value> {
    ensure!(
        config["schema"] == "auxiliary-overlap-sphere-control-v1"
            && config["master_seed"] == MASTER
            && config["populations"] == 4
            && config["draws_per_population"] == 4096
            && config["m"] == json!([1, 4])
            && config["core_radius"] == 0.05
            && config["exclusion_radius"] == 0.45
            && config["uniform_half_width"] == 1.
            && config["activity"] == 0.
            && config["caps"] == json!({"root":8,"internal":8,"joint":1})
            && config["order"] == "root_first",
        "changed fixed reference protocol"
    );
    let core = sphere(0.05)?;
    let exclusion = sphere(0.45)?;
    let points = cloud();
    save(&out.join("cloud.json"), &points)?;
    let guides = [
        AuxiliaryOverlapThreshold::new(&exclusion, points.clone(), 1)?,
        AuxiliaryOverlapThreshold::new(&exclusion, points, 4)?,
    ];
    let learned = atlas()?;
    let proposal = FactorizedDimerProposal::new(
        DefensiveDimerProposal::new(&learned, 1., 1.)?,
        FactorizedDimerCaps {
            root: 8,
            internal: 8,
            joint: 1,
        },
        FactorizedDimerOrder::RootFirst,
    );
    let mut trials = 0;
    let mut decisions = 0;
    let mut candidates = [0usize; 2];
    let mut accepted = [0usize; 2];
    let mut wrong_accepted = 0;
    for population in 0..4 {
        for slot in 0..4096 {
            let mut source_trace = vec![];
            let mut row = json!({"population":population,"slot":slot,"source_seed":seed(population,slot,"source"),
            "source":null,"arms":[],"status":"in_progress"});
            let result = (|| -> Result<()> {
                let old = source(
                    &mut StdRng::seed_from_u64(seed(population, slot, "source")),
                    &core,
                    &exclusion,
                    &mut source_trace,
                )?;
                row["source"] = json!(old);
                let state = [old[0], old[1], ID];
                let context =
                    FixedDimerContext::new(&core, &exclusion, &state, [0, 1], 2, None, [0.; 3])?;
                for (j, guide) in guides.iter().enumerate() {
                    let m = guide.m();
                    // Shared proposal and auxiliary prefixes are intentional; each arm has the proper marginal.
                    let mut prng = StdRng::seed_from_u64(seed(population, slot, "proposal"));
                    let mut arng = StdRng::seed_from_u64(seed(population, slot, "auxiliary"));
                    let mut mrng = StdRng::seed_from_u64(seed(population, slot, "mh"));
                    let uniform: f64 = mrng.sample(Open01);
                    let log_uniform = uniform.ln();
                    row["current_arm"] = json!({"m":m,"log_uniform":log_uniform,
                        "proposal_seed":seed(population,slot,"proposal"),
                        "auxiliary_seed":seed(population,slot,"auxiliary"),
                        "mh_seed":seed(population,slot,"mh"),"status":"in_progress"});
                    let before = cpu_seconds();
                    let sampled =
                        proposal.propose_guided(&mut prng, &mut arng, &context, old, guide);
                    let elapsed = cpu_seconds() - before;
                    row["current_arm"]["proposal_cpu_seconds"] = json!(elapsed);
                    let outcome = match sampled {
                        Ok(value) => value,
                        Err(failure) => {
                            row["fatal_proposal"] = json!(failure);
                            anyhow::bail!("fatal reference guided proposal");
                        }
                    };
                    row["current_outcome"] = json!(&outcome);
                    let mut retained = old;
                    let mut wrong_retained = old;
                    let mut accept = false;
                    let mut wrong = false;
                    let mut correction = None;
                    if let Some(candidate) = outcome.candidate.as_ref() {
                        candidates[j] += 1;
                        ensure!(
                            candidate.diagnostics.log_reverse_forward == 0.,
                            "uniform reference has nonzero full-F correction"
                        );
                        let log_ratio = outcome.complete_log_correction()?;
                        ensure!(
                            log_ratio.is_finite(),
                            "nonfinite reference complete correction"
                        );
                        correction = Some(log_ratio);
                        accept = log_uniform < log_ratio.min(0.);
                        wrong = log_uniform < candidate.diagnostics.log_reverse_forward.min(0.);
                        if accept {
                            retained = [candidate.root, candidate.child];
                            accepted[j] += 1;
                        }
                        if wrong {
                            wrong_retained = [candidate.root, candidate.child];
                        }
                    }
                    if m == 4 && wrong {
                        wrong_accepted += 1;
                    }
                    row["arms"].as_array_mut().unwrap().push(json!({"m":m,"outcome":outcome,"log_uniform":log_uniform,
                    "complete_log_correction":correction,"accepted":accept,"retained":retained,
                    "wrong_without_auxiliary":if m==4 {json!({"accepted":wrong,"retained":wrong_retained})} else {Value::Null},
                    "proposal_seed":seed(population,slot,"proposal"),"auxiliary_seed":seed(population,slot,"auxiliary"),
                    "mh_seed":seed(population,slot,"mh"),"proposal_cpu_seconds":elapsed,
                    "proposal_rng_after":prng.random::<u64>(),"auxiliary_rng_after":arng.random::<u64>()}));
                    row.as_object_mut().unwrap().remove("current_outcome");
                    row.as_object_mut().unwrap().remove("current_arm");
                    decisions += 1;
                }
                Ok(())
            })();
            row["source_trace"] = json!(source_trace);
            row["status"] = json!(if result.is_ok() { "complete" } else { "fatal" });
            if let Err(ref error) = result {
                row["error"] = json!(format!("{error:#}"));
            }
            serde_json::to_writer(&mut *writer, &row)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
            result?;
            trials += 1;
        }
    }
    Ok(
        json!({"independent_sources":trials,"corrected_decisions":decisions,"candidate_counts":candidates,
        "accepted_counts":accepted,"negative_control_accepted":wrong_accepted,"new_physical_clouds":0}),
    )
}

fn main() -> Result<()> {
    let started = cpu_seconds();
    let args = Args::parse();
    let config_bytes = fs::read(&args.config)?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    fs::create_dir(&args.out)?;
    for (name, bytes) in [
        ("config.json", config_bytes.as_slice()),
        ("example.rs", SOURCE),
        ("source-bundle.json", BUNDLE),
    ] {
        fs::write(args.out.join(name), bytes)?;
    }
    save(
        &args.out.join("binding.json"),
        &json!({"config_sha256":hash_bytes(&config_bytes),
        "source_sha256":hash_bytes(SOURCE),"source_bundle_sha256":hash_bytes(BUNDLE),
        "executable_sha256":hash_file(&std::env::current_exe()?)?}),
    )?;
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(args.out.join("attempts.jsonl"))?;
    let mut writer = BufWriter::new(file);
    let result = run(&args.out, &config, &mut writer);
    writer.flush()?;
    let summary = match &result {
        Ok(value) => json!({"complete":true,"result":value}),
        Err(error) => json!({"complete":false,"error":format!("{error:#}")} ),
    };
    save(
        &args.out.join("terminal.json"),
        &json!({"summary":summary,"cpu_seconds":cpu_seconds()-started,
        "attempts_sha256":hash_file(&args.out.join("attempts.jsonl"))?}),
    )?;
    result.map(|_| ())
}
