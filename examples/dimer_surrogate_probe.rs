//! Passive score/cost probe of a fixed inventory of saved endpoints. No RNG,
//! physical cloud, hard-state selection, native classification, or MC update.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
use tetramer_mc::{
    depletion_surrogate::DimerDepletionSurrogate,
    geometry::{Shape, SphereTree},
    math::{Pose, Vec3},
    simulation::cpu_seconds,
};

#[derive(Deserialize)]
struct File {
    path: PathBuf,
    sha256: String,
}
impl File {
    fn read(&self) -> Result<Vec<u8>> {
        ensure!(
            fs::metadata(&self.path)?.len() <= 64 * 1024 * 1024,
            "input exceeds 64 MiB"
        );
        let b = fs::read(&self.path)?;
        ensure!(
            format!("{:x}", Sha256::digest(&b)) == self.sha256,
            "input hash mismatch: {}",
            self.path.display()
        );
        Ok(b)
    }
}
#[derive(Deserialize)]
struct Cloud {
    raw: File,
    low: Vec3,
    high: Vec3,
    raw_count: usize,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    old: [Pose; 2],
    proposed: Option<[Pose; 2]>,
}
#[derive(Deserialize)]
struct Group {
    id: String,
    cloud: Cloud,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Plan {
    schema: String,
    shape: File,
    radius: f64,
    activity: f64,
    spectators: Vec<Pose>,
    prefixes: Vec<usize>,
    groups: Vec<Group>,
    unpruned_first_case: bool,
}
fn write(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn run(p: &Plan, out: &Path) -> Result<()> {
    let start = cpu_seconds();
    ensure!(
        p.schema == "dimer-surrogate-saved-endpoints-v1",
        "wrong schema"
    );
    ensure!(p.radius.is_finite() && p.radius >= 0., "bad radius");
    ensure!(
        !p.prefixes.is_empty() && p.prefixes.len() <= 4 && p.groups.len() <= 32,
        "allocation too large"
    );
    let mut shape: Shape = serde_json::from_slice(&p.shape.read()?)?;
    for atom in &mut shape.atoms {
        atom.radius += p.radius;
    }
    let tree = SphereTree::new(shape)?;
    let mut log = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("scores.jsonl"))?;
    let mut rows = 0;
    let mut queries = 0;
    let mut checks = 0;
    let mut score_calls = 0;
    let mut progress = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("progress.jsonl"))?;
    for group in &p.groups {
        ensure!(group.cases.len() <= 128, "too many cases");
        let cloud = &group.cloud;
        let raw = cloud.raw.read()?;
        ensure!(
            cloud.raw_count <= 2_000_000 && raw.len() == 24 * cloud.raw_count,
            "cloud length differs"
        );
        ensure!(
            (0..3).all(|k| cloud.low[k].is_finite()
                && cloud.high[k].is_finite()
                && cloud.low[k] < cloud.high[k]),
            "invalid box"
        );
        let volume = (0..3)
            .map(|k| cloud.high[k] - cloud.low[k])
            .product::<f64>();
        ensure!(volume.is_finite() && volume > 0., "invalid box volume");
        for &n in &p.prefixes {
            ensure!(n > 0 && n <= cloud.raw_count, "invalid prefix");
            let started = cpu_seconds();
            let mut points = vec![];
            for bytes in raw[..24 * n].chunks_exact(24) {
                let u: Vec3 = std::array::from_fn(|k| {
                    f64::from_le_bytes(bytes[8 * k..8 * k + 8].try_into().unwrap())
                });
                ensure!(
                    u.iter().all(|x| x.is_finite() && (0. ..1.).contains(x)),
                    "invalid raw uniform"
                );
                let point =
                    std::array::from_fn(|k| cloud.low[k] + (cloud.high[k] - cloud.low[k]) * u[k]);
                if tree.contains(point, 0.) {
                    points.push(point);
                }
            }
            let g = DimerDepletionSurrogate::new(
                &tree,
                points,
                &p.spectators,
                volume / n as f64,
                p.activity,
            )?;
            let setup = cpu_seconds() - started;
            for (index, case) in group.cases.iter().enumerate() {
                let mut score = |role: &str, poses, unpruned| -> Result<_> {
                    serde_json::to_writer(
                        &mut progress,
                        &json!({"phase":"begin",
                        "group":group.id,"id":case.id,"raw_prefix":n,"role":role}),
                    )?;
                    progress.write_all(b"\n")?;
                    progress.flush()?;
                    let value = if unpruned {
                        g.score_unpruned(poses)?
                    } else {
                        g.score(poses)?
                    };
                    score_calls += 1;
                    serde_json::to_writer(
                        &mut progress,
                        &json!({"phase":"complete",
                        "group":group.id,"id":case.id,"raw_prefix":n,"role":role,"score":value}),
                    )?;
                    progress.write_all(b"\n")?;
                    progress.flush()?;
                    Ok(value)
                };
                let before = cpu_seconds();
                let old = score("old", case.old, false)?;
                let new = case
                    .proposed
                    .map(|y| score("proposed", y, false))
                    .transpose()?;
                let score_cpu = cpu_seconds() - before;
                let delta = new.as_ref().map(|y| y.log_surrogate - old.log_surrogate);
                queries += old.spectator_membership_queries + old.internal_membership_queries;
                if let Some(y) = &new {
                    queries += y.spectator_membership_queries + y.internal_membership_queries;
                }
                let check = if index == 0 && p.unpruned_first_case {
                    let before = cpu_seconds();
                    let b = score("old_unpruned", case.old, true)?;
                    ensure!(
                        b.twice_overlap_units == old.twice_overlap_units,
                        "pruning changed source score"
                    );
                    if let (Some(y), Some(a)) = (case.proposed, new.as_ref()) {
                        ensure!(
                            score("proposed_unpruned", y, true)?.twice_overlap_units
                                == a.twice_overlap_units,
                            "pruning changed target score"
                        );
                    }
                    checks += 1;
                    Some(cpu_seconds() - before)
                } else {
                    None
                };
                serde_json::to_writer(
                    &mut log,
                    &json!({"group":group.id,"id":case.id,"raw_prefix":n,
                    "old":old,"new":new,"delta_log_surrogate":delta,"score_cpu_seconds":score_cpu,
                    "cloud_setup_cpu_seconds":setup,"unpruned_check_cpu_seconds":check}),
                )?;
                log.write_all(b"\n")?;
                log.flush()?;
                rows += 1;
            }
        }
    }
    write(
        &out.join("summary.json"),
        &json!({"complete":true,"rows":rows,
        "pruned_membership_queries":queries,"unpruned_case_prefix_checks":checks,
        "completed_score_calls_including_unpruned":score_calls,"cpu_seconds":cpu_seconds()-start,
        "new_pose_draws":0,"new_physical_clouds":0,"physical_acceptances":0,
        "scope":"Frozen-cloud surrogate diagnostic; no exact energy, equilibrium, or sampling-efficiency claim."}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 4,
        "usage: dimer_surrogate_probe PLAN PLAN_SHA OUTPUT"
    );
    let input = File {
        path: args[1].clone().into(),
        sha256: args[2].clone(),
    };
    let p: Plan = serde_json::from_slice(&input.read()?)?;
    let out = Path::new(&args[3]);
    fs::create_dir(out)?;
    let result = run(&p, out);
    if let Err(e) = &result {
        write(
            &out.join("failure.json"),
            &json!({"complete":false,"error":format!("{e:#}")}),
        )?;
    }
    result.context("surrogate diagnostic failed; retained partial output is not a sample")
}
