//! Prepared, conditional two-mobile-body trajectories. Frozen spectators, NOT
//! finite-system assembly. No adaptive selection, basin filtering or retries
//! after an outer null/failure. Every elementary attempt and retained block is
//! journaled; independent named streams give deterministic graceful continuation.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
use tetramer_mc::{
    auxiliary_overlap_threshold::AuxiliaryOverlapThreshold,
    bounded_singleton_path::{Budget, Limits},
    defensive_dimer_proposal::DefensiveDimerProposal,
    docking::{DockingMethod, DockingProposal},
    evolving_dimer::FixedLabelUpdates,
    factorized_dimer::{FactorizedDimerCaps, FactorizedDimerOrder, FactorizedDimerProposal},
    geometry::{Shape, SphereTree},
    math::{Pose, norm, sub},
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    spherical::{Container, validate_state},
};
const BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
const SOURCE: &[u8] = include_bytes!("evolving_dimer_benchmark.rs");
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    binding: PathBuf,
    #[arg(long)]
    mode: String,
    #[arg(long)]
    job: Option<usize>,
    /// Graceful deterministic continuation control; changes no allocation.
    #[arg(long)]
    stop_after_block: Option<usize>,
    #[arg(long)]
    resume: Option<PathBuf>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BoundFile {
    path: PathBuf,
    sha256: String,
}
impl BoundFile {
    fn read(&self) -> Result<Vec<u8>> {
        let bytes = fs::read(&self.path)?;
        ensure!(
            hash_bytes(&bytes) == self.sha256,
            "changed input {}",
            self.path.display()
        );
        Ok(bytes)
    }
    fn json(&self) -> Result<Value> {
        Ok(serde_json::from_slice(&self.read()?)?)
    }
    fn make(path: &Path) -> Result<Self> {
        Ok(Self {
            path: fs::canonicalize(path)?,
            sha256: hash_file(path)?,
        })
    }
}
fn bound(value: &Value) -> Result<BoundFile> {
    Ok(serde_json::from_value(value.clone())?)
}
fn last_journal_row(path: &Path) -> Result<Value> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    ensure!(size > 0, "empty checkpoint journal");
    let start = size.saturating_sub(65536);
    file.seek(SeekFrom::Start(start))?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail)?;
    ensure!(
        tail.last() == Some(&b'\n'),
        "incomplete checkpoint journal line"
    );
    tail.pop();
    let begin = tail.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    ensure!(
        start == 0 || begin > 0,
        "checkpoint terminal row exceeds audit buffer"
    );
    Ok(serde_json::from_slice(&tail[begin..])?)
}
fn usize_at(v: &Value, k: &str) -> Result<usize> {
    usize::try_from(
        v[k].as_u64()
            .with_context(|| format!("missing integer {k}"))?,
    )
    .map_err(Into::into)
}
fn num(v: &Value, k: &str) -> Result<f64> {
    v[k].as_f64().with_context(|| format!("missing number {k}"))
}
fn rng(master: u64, context: usize, init: &str, stream: usize, block: usize, role: &str) -> StdRng {
    let digest = Sha256::digest(
        format!("evolving-dimer-v1/{master}/{context}/{init}/{stream}/{block}/{role}").as_bytes(),
    );
    StdRng::from_seed(digest.into())
}
fn far_enough(old: [Pose; 2], new: [Pose; 2], distance: f64, angle: f64) -> bool {
    (0..2).any(|i| {
        let a = old[i].orientation;
        let b = new[i].orientation;
        let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        norm(sub(old[i].position, new[i].position)) >= distance
            || 2. * (dot / (na * nb)).abs().min(1.).acos().to_degrees() >= angle
    })
}
struct Journal {
    out: BufWriter<File>,
    digest: Sha256,
    bytes: u64,
    rows: u64,
}
impl Journal {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            out: BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?),
            digest: Sha256::new(),
            bytes: 0,
            rows: 0,
        })
    }
    fn resume(path: &Path, bytes: u64, rows: u64, sha: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        ensure!(
            file.metadata()?.len() == bytes,
            "journal has an uncheckpointed tail; preserve and audit it before recovery"
        );
        let mut digest = Sha256::new();
        let mut buffer = [0; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            digest.update(&buffer[..n]);
        }
        ensure!(
            format!("{:x}", digest.clone().finalize()) == sha,
            "checkpoint journal hash mismatch"
        );
        Ok(Self {
            out: BufWriter::new(OpenOptions::new().append(true).open(path)?),
            digest,
            bytes,
            rows,
        })
    }
    fn line(&mut self, value: &Value) -> Result<()> {
        let mut bytes = serde_json::to_vec(value)?;
        bytes.push(b'\n');
        self.out.write_all(&bytes)?;
        self.out.flush()?;
        self.digest.update(&bytes);
        self.bytes += bytes.len() as u64;
        self.rows += 1;
        Ok(())
    }
    fn sha(&self) -> String {
        format!("{:x}", self.digest.clone().finalize())
    }
}
struct Geometry {
    core: SphereTree,
    exclusion: SphereTree,
    wall: Container,
    state: Vec<Pose>,
    model: DockingProposal,
    load_cpu_seconds: f64,
}
impl Geometry {
    fn load(plan: &Value) -> Result<Self> {
        let started = cpu_seconds();
        let source = bound(&plan["source_config"])?.json()?;
        let frame = bound(&plan["source_frame"])?.json()?;
        ensure!(
            source["initial_poses"] == frame["poses"],
            "source/frame state mismatch"
        );
        let state: Vec<Pose> = serde_json::from_value(source["initial_poses"].clone())?;
        let shape_file = bound(&plan["shape"])?;
        let shape: Shape = serde_json::from_slice(&shape_file.read()?)?;
        let core = SphereTree::new(shape.clone())?;
        let mut inflated = shape;
        for atom in &mut inflated.atoms {
            atom.radius += num(&plan["physical"], "depletant_radius")?;
        }
        let exclusion = SphereTree::new(inflated)?;
        let radius = num(&plan["physical"], "wall_radius")?;
        let wall = Container::new(radius, &core)?;
        validate_state(&core, &wall, &state)?;
        let bytes = bound(&plan["atlas"])?.read()?;
        let model = DockingProposal::new(
            FrozenRelativePoseProposal::from_json_str_open(
                std::str::from_utf8(&bytes)?,
                [2. * (radius + core.bound); 3],
                0.1,
                &shape_file.sha256,
            )?,
            DockingMethod::PosteriorInvolution,
            0.,
            [0.; 3],
        )?;
        Ok(Self {
            core,
            exclusion,
            wall,
            state,
            model,
            load_cpu_seconds: cpu_seconds() - started,
        })
    }
    fn engine(&self, plan: &Value, ci: usize) -> Result<FixedLabelUpdates<'_>> {
        let case = &plan["contexts"][ci];
        let physical = &plan["physical"];
        let activity = num(physical, "activity")?;
        Ok(FixedLabelUpdates {
            core: &self.core,
            exclusion: &self.exclusion,
            wall: &self.wall,
            radius: num(physical, "wall_radius")?,
            members: [usize_at(case, "root")?, usize_at(case, "child")?],
            anchor: usize_at(case, "anchor")?,
            rd: num(physical, "depletant_radius")?,
            activity,
            lambda: if activity == 0. {
                1.
            } else {
                activity * num(physical, "lambda_ratio")?
            },
            envelope: serde_json::from_value(plan["envelope"].clone())?,
        })
    }
    fn proposal(&self, plan: &Value) -> Result<FactorizedDimerProposal<'_>> {
        let f = &plan["factorized"];
        ensure!(f["order"] == "root_first", "unsupported stage order");
        Ok(FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(
                &self.model,
                num(f, "uniform_half_width")?,
                num(f, "uniform_probability")?,
            )?,
            FactorizedDimerCaps {
                root: usize_at(f, "root_cap")?,
                internal: usize_at(f, "internal_cap")?,
                joint: usize_at(f, "joint_cap")?,
            },
            FactorizedDimerOrder::RootFirst,
        ))
    }
}
fn validate_binding(args: &Args, plan: &Value) -> Result<Value> {
    let binding: Value = serde_json::from_slice(&fs::read(&args.binding)?)?;
    ensure!(
        binding["config_sha256"] == hash_file(&args.config)?,
        "config binding differs"
    );
    let pre = if args.mode == "run" {
        let file = bound(&binding["prelaunch_binding"])?;
        file.json()?
    } else {
        binding.clone()
    };
    ensure!(
        pre["config_sha256"] == hash_file(&args.config)?
            && pre["protocol_sha256"] == bound(&plan["protocol"])?.sha256
            && pre["example_source_sha256"] == hash_bytes(SOURCE)
            && pre["compiled_source_bundle_sha256"] == hash_bytes(BUNDLE)
            && pre["executable_sha256"] == hash_file(&std::env::current_exe()?)?,
        "execution closure differs"
    );
    for key in [
        "protocol",
        "scientific_allocation",
        "source_config",
        "source_frame",
        "shape",
        "panel",
        "atlas",
        "patch_map",
    ] {
        bound(&plan[key])?.read()?;
    }
    for key in [
        "reference_config",
        "source_freeze_manifest",
        "original_atlas",
    ] {
        if !plan[key].is_null() {
            bound(&plan[key])?.read()?;
        }
    }
    ensure!(
        plan["schema"] == "evolving-dimer-benchmark-v1",
        "unexpected plan schema"
    );
    ensure!(
        plan["allocation"]["local_member_order"] == json!([0, 1, 0, 1]),
        "local schedule differs"
    );
    Ok(binding)
}
fn prepare(args: &Args, plan: &Value, geometry: &Geometry) -> Result<()> {
    let output = PathBuf::from(
        plan["preparation_output"]
            .as_str()
            .context("preparation_output")?,
    );
    fs::create_dir(&output)?;
    let result = prepare_inner(args, plan, geometry, &output);
    if let Err(error) = &result {
        save(
            &output.join("failure.json"),
            &json!({"complete":false,"error":format!("{error:#}")}),
        )?;
    }
    result
}
fn prepare_inner(args: &Args, plan: &Value, geometry: &Geometry, output: &Path) -> Result<()> {
    let preparation_started=cpu_seconds();
    let master = plan["master_seed"].as_u64().context("master_seed")?;
    let contexts = plan["contexts"].as_array().context("contexts")?;
    let streams = usize_at(&plan["allocation"], "streams")?;
    let mut files = Vec::<BoundFile>::new();
    let mut banks = vec![];
    let mut starts = vec![];
    let bounds = geometry.exclusion.bounds(0.);
    let raw_count = usize_at(&plan["cloud"], "raw_count")?;
    // Complete independent bank generated BEFORE inspecting any prepared start.
    for ci in 0..contexts.len() {
        for init in ["source", "proposal_prepared"] {
            for stream in 0..streams {
                let before = cpu_seconds();
                let mut cloud_rng = rng(master, ci, init, stream, 0, "cloud");
                let mut raw = Vec::with_capacity(raw_count * 24);
                let mut kept = vec![];
                for index in 0..raw_count {
                    let u: [f64; 3] = std::array::from_fn(|_| cloud_rng.random());
                    raw.extend(u.into_iter().flat_map(f64::to_le_bytes));
                    let p = std::array::from_fn(|k| {
                        bounds.lo[k] + (bounds.hi[k] - bounds.lo[k]) * u[k]
                    });
                    if geometry.exclusion.contains(p, 0.) {
                        kept.push(index);
                    }
                }
                let raw_path = output.join(format!("cloud-{ci}-{init}-{stream}.bin"));
                fs::write(&raw_path, raw)?;
                let meta_path = output.join(format!("cloud-{ci}-{init}-{stream}.json"));
                save(
                    &meta_path,
                    &json!({"raw_count":raw_count,"low":bounds.lo,"high":bounds.hi,
            "kept_indices":kept,"cpu_seconds":cpu_seconds()-before}),
                )?;
                let raw_file = BoundFile::make(&raw_path)?;
                let meta_file = BoundFile::make(&meta_path)?;
                banks.push(json!({"context_index":ci,"initialization":init,"stream":stream,"raw":raw_file,"metadata":meta_file}));
                files.extend([raw_file, meta_file]);
            }
        }
    }
    let proposal = geometry.proposal(plan)?;
    for ci in 0..contexts.len() {
        for stream in 0..streams {
            let before = cpu_seconds();
            let engine = geometry.engine(plan, ci)?;
            let context = engine.context(&geometry.state)?;
            let old = engine.selected(&geometry.state);
            let ledger_path = output.join(format!("start-{ci}-{stream}-attempts.jsonl"));
            let mut ledger = Journal::new(&ledger_path)?;
            let mut chosen = None;
            for attempt in 1..=usize_at(&plan["allocation"], "prep_outer_cap")? {
                let mut prng = rng(
                    master,
                    ci,
                    "proposal_prepared",
                    stream,
                    attempt,
                    "preparation",
                );
                let outcome = match proposal.propose(&mut prng, &context, old) {
                    Ok(v) => v,
                    Err(e) => {
                        ledger.line(&json!({"attempt":attempt,"failure":e}))?;
                        anyhow::bail!("preparation failed");
                    }
                };
                let selected = outcome
                    .candidate
                    .as_ref()
                    .map(|c| [c.root, c.child])
                    .filter(|p| {
                        far_enough(
                            old,
                            *p,
                            num(&plan["preparation"], "minimum_max_center_displacement_A").unwrap(),
                            num(&plan["preparation"], "minimum_max_body_orientation_degrees")
                                .unwrap(),
                        )
                    });
                ledger.line(
                    &json!({"attempt":attempt,"outcome":outcome,"selected":selected.is_some()}),
                )?;
                if let Some(p) = selected {
                    chosen = Some((attempt, p));
                    break;
                }
            }
            let Some((attempts, selected)) = chosen else {
                anyhow::bail!(
                    "preparation exhausted for context {ci} stream {stream}; no replacement/fallback"
                );
            };
            let record_path = output.join(format!("start-{ci}-{stream}.json"));
            save(
                &record_path,
                &json!({"context_index":ci,"stream":stream,"source":old,"selected":selected,
            "attempts":attempts,"status":"prepared","is_equilibrium_sample":false,
            "preparation_cpu_seconds":cpu_seconds()-before}),
            )?;
            let record = BoundFile::make(&record_path)?;
            let ledger_file = BoundFile::make(&ledger_path)?;
            starts.push(json!({"context_index":ci,"stream":stream,"status":"prepared","attempts":attempts,"record":record,"ledger":ledger_file}));
            files.extend([record, ledger_file]);
        }
    }
    save(
        &output.join("manifest.json"),
        &json!({"schema":"evolving-dimer-prepared-starts-v1","complete":true,"passed":true,
        "config_sha256":hash_file(&args.config)?,"binding_sha256":hash_file(&args.binding)?,
        "geometry_load_cpu_seconds":geometry.load_cpu_seconds,
        "preparation_cpu_seconds":cpu_seconds()-preparation_started,
        "all_attempts_retained":true,"files":files,"cloud_banks":banks,"alternative_starts":starts}),
    )?;
    Ok(())
}
fn checkpoint(
    path: &Path,
    config_hash: &str,
    binding_hash: &str,
    job: &Value,
    block: usize,
    selected: [Pose; 2],
    journal: &Journal,
    budget: &Budget,
    counts: &Value,
) -> Result<()> {
    save(
        path,
        &json!({"schema":"evolving-dimer-checkpoint-v1","config_sha256":config_hash,
        "binding_sha256":binding_hash,"job":job,"block":block,"selected":selected,
        "journal_bytes":journal.bytes,"journal_rows":journal.rows,"journal_sha256":journal.sha(),
        "cpu_seconds":cpu_seconds()-budget.started,"raw":budget.raw,"retained":budget.retained,"counts":counts}),
    )
}
fn run(args: &Args, plan: &Value, binding: &Value, geometry: &Geometry) -> Result<()> {
    let job_id = args.job.context("--job required")?;
    let job = plan["jobs"]
        .as_array()
        .context("jobs")?
        .iter()
        .find(|j| j["id"].as_u64() == Some(job_id as u64))
        .context("unknown job")?;
    let ci = usize_at(job, "context_index")?;
    let stream = usize_at(job, "stream")?;
    let init = job["initialization"].as_str().context("initialization")?;
    let arm = job["arm"].as_str().context("arm")?;
    ensure!(["local", "unguided", "m4"].contains(&arm), "unknown arm");
    let parent = PathBuf::from(plan["output"].as_str().context("output")?);
    fs::create_dir_all(&parent)?;
    let output = parent.join(format!("job-{job_id:03}"));
    if args.resume.is_none() {
        fs::create_dir(&output)?;
    } else {
        ensure!(output.is_dir(), "missing continuation directory");
        ensure!(
            !output.join("terminal.json").exists(),
            "completed run cannot be resumed"
        );
        ensure!(
            !output.join("failure.json").exists(),
            "fatal run receipt must be preserved; cannot resume"
        );
    }
    let result = run_inner(
        args, plan, binding, geometry, job, &output, ci, stream, init, arm,
    );
    if let Err(error) = &result {
        save(
            &output.join("failure.json"),
            &json!({"job":job,"complete":false,"error":format!("{error:#}")}),
        )?;
    }
    result
}
fn run_inner(
    args: &Args,
    plan: &Value,
    binding: &Value,
    geometry: &Geometry,
    job: &Value,
    output: &Path,
    ci: usize,
    stream: usize,
    init: &str,
    arm: &str,
) -> Result<()> {
    let config_hash = hash_file(&args.config)?;
    let binding_hash = hash_file(&args.binding)?;
    let started = cpu_seconds();
    let prepared = bound(&binding["prepared_manifest"])?.json()?;
    ensure!(
        prepared["complete"] == true && prepared["passed"] == true,
        "incomplete preparation"
    );
    for file in prepared["files"].as_array().context("prepared files")? {
        bound(file)?.read()?;
    }
    let bank = prepared["cloud_banks"]
        .as_array()
        .context("cloud_banks")?
        .iter()
        .find(|b| {
            b["context_index"].as_u64() == Some(ci as u64)
                && b["initialization"] == init
                && b["stream"].as_u64() == Some(stream as u64)
        })
        .context("missing cloud bank")?;
    let raw = bound(&bank["raw"])?.read()?;
    let meta = bound(&bank["metadata"])?.json()?;
    let low: [f64; 3] = serde_json::from_value(meta["low"].clone())?;
    let high: [f64; 3] = serde_json::from_value(meta["high"].clone())?;
    ensure!(
        raw.len() == 24 * usize_at(&plan["cloud"], "raw_count")?,
        "cloud length differs"
    );
    let indices: Vec<usize> = serde_json::from_value(meta["kept_indices"].clone())?;
    let points: Vec<[f64; 3]> = indices
        .iter()
        .map(|&i| {
            std::array::from_fn(|k| {
                let u =
                    f64::from_le_bytes(raw[i * 24 + k * 8..i * 24 + k * 8 + 8].try_into().unwrap());
                low[k] + (high[k] - low[k]) * u
            })
        })
        .collect();
    let guide = if arm == "m4" {
        Some(AuxiliaryOverlapThreshold::new(
            &geometry.exclusion,
            points,
            4,
        )?)
    } else {
        None
    };
    let mut state = geometry.state.clone();
    let engine = geometry.engine(plan, ci)?;
    if init == "proposal_prepared" {
        let start = prepared["alternative_starts"]
            .as_array()
            .context("starts")?
            .iter()
            .find(|s| {
                s["context_index"].as_u64() == Some(ci as u64)
                    && s["stream"].as_u64() == Some(stream as u64)
            })
            .context("missing prepared start")?;
        let record = bound(&start["record"])?.json()?;
        let poses: [Pose; 2] = serde_json::from_value(record["selected"].clone())?;
        for i in 0..2 {
            state[engine.members[i]] = poses[i];
        }
    } else {
        ensure!(init == "source", "unknown initialization");
    }
    let limits: Limits = serde_json::from_value(plan["limits"].clone())?;
    let mut budget = Budget::new(limits)?;
    budget.started = started;
    let mut counts = json!({"local_attempted":0u64,"local_accepted":0u64,"dimer_attempted":0u64,"dimer_accepted":0u64,"dimer_self_loop":0u64});
    let mut completed = 0usize;
    let journal_path = output.join("trajectory.jsonl");
    let mut journal = if let Some(path) = &args.resume {
        ensure!(
            !output.join("failure.json").exists(),
            "fatal runs cannot be resumed as ordinary rejections"
        );
        let cp: Value = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            cp["schema"] == "evolving-dimer-checkpoint-v1"
                && cp["config_sha256"] == config_hash
                && cp["binding_sha256"] == binding_hash
                && cp["job"] == *job,
            "checkpoint provenance differs"
        );
        completed = usize_at(&cp, "block")?;
        ensure!(
            cp["journal_rows"].as_u64()
                == Some(1 + (completed as u64) * if arm == "local" { 5 } else { 6 }),
            "checkpoint row count differs from attempted schedule"
        );
        let last = last_journal_row(&journal_path)?;
        ensure!(
            last["kind"] == "retained_block"
                && last["block"] == cp["block"]
                && last["selected"] == cp["selected"]
                && last["raw"] == cp["raw"]
                && last["retained"] == cp["retained"]
                && last["counts"] == cp["counts"]
                && num(&cp, "cpu_seconds")? >= num(&last, "sampler_cpu_seconds")?,
            "checkpoint disagrees with bound retained trajectory"
        );
        let selected: [Pose; 2] = serde_json::from_value(cp["selected"].clone())?;
        for i in 0..2 {
            state[engine.members[i]] = selected[i];
        }
        budget.raw = cp["raw"].as_u64().context("raw counter")?;
        budget.retained = cp["retained"].as_u64().context("retained counter")?;
        ensure!(
            budget.raw <= limits.raw_campaign && budget.retained <= limits.retained_campaign,
            "checkpoint exceeds completed bath allocation"
        );
        budget.started -= num(&cp, "cpu_seconds")?;
        counts = cp["counts"].clone();
        Journal::resume(
            &journal_path,
            cp["journal_bytes"].as_u64().context("bytes")?,
            cp["journal_rows"].as_u64().context("rows")?,
            cp["journal_sha256"].as_str().context("hash")?,
        )?
    } else {
        let mut journal = Journal::new(&journal_path)?;
        journal.line(&json!({"kind":"initial","block":0,"job":job,"selected":engine.selected(&state),
            "fixed_source":plan["source_frame"],"cloud":bank,"cloud_cpu_seconds":meta["cpu_seconds"],
            "geometry_load_cpu_seconds":geometry.load_cpu_seconds,
            "conditional_target":true,"sampler_cpu_seconds":cpu_seconds()-budget.started}))?;
        journal
    };
    ensure!(
        engine
            .context(&state)?
            .evaluate(engine.selected(&state))?
            .hard_valid(),
        "invalid evolving start"
    );
    let proposal = geometry.proposal(plan)?;
    let warmup = usize_at(&plan["allocation"], "warmup_blocks")?;
    let total = warmup + usize_at(&plan["allocation"], "production_blocks")?;
    ensure!(completed <= total, "checkpoint block exceeds allocation");
    let end = args.stop_after_block.unwrap_or(total).min(total);
    ensure!(end >= completed, "stop precedes checkpoint");
    let master = plan["master_seed"].as_u64().context("master_seed")?;
    for block in completed + 1..=end {
        for (attempt, slot) in [0, 1, 0, 1].into_iter().enumerate() {
            let mut record = Value::Null;
            let mut prng = rng(
                master,
                ci,
                init,
                stream,
                block,
                &format!("local/{attempt}/proposal"),
            );
            let mut brng = rng(
                master,
                ci,
                init,
                stream,
                block,
                &format!("local/{attempt}/bath"),
            );
            let mut arng = rng(
                master,
                ci,
                init,
                stream,
                block,
                &format!("local/{attempt}/accept"),
            );
            let result = engine.local(
                &mut state,
                slot,
                num(&plan["local"], "translation_std_A")?,
                num(&plan["local"], "rotation_std_degrees")?,
                &mut prng,
                &mut brng,
                &mut arng,
                &mut budget,
                &mut record,
            );
            if record.is_null() {
                record = json!({"kind":"local","status":"failed_before_proposal"});
            }
            record["block"] = json!(block);
            record["attempt"] = json!(attempt);
            record["retained"] = json!(engine.selected(&state));
            record["sampler_cpu_seconds"] = json!(cpu_seconds() - budget.started);
            if let Err(error) = &result {
                record["fatal_error"] = json!(format!("{error:#}"));
            }
            journal.line(&record)?;
            result?;
            increment(&mut counts, "local_attempted");
            if record["accepted"] == true {
                increment(&mut counts, "local_accepted");
            }
        }
        if arm != "local" {
            let mut record = Value::Null;
            let mut prng = rng(master, ci, init, stream, block, &format!("{arm}/proposal"));
            let mut trng = rng(master, ci, init, stream, block, &format!("{arm}/threshold"));
            let mut brng = rng(master, ci, init, stream, block, &format!("{arm}/bath"));
            let mut arng = rng(master, ci, init, stream, block, &format!("{arm}/accept"));
            let result = engine.dimer(
                &mut state,
                &proposal,
                guide.as_ref(),
                &mut prng,
                &mut trng,
                &mut brng,
                &mut arng,
                &mut budget,
                &mut record,
            );
            if record.is_null() {
                record = json!({"kind":"factorized_dimer","status":"failed_before_proposal"});
            }
            record["block"] = json!(block);
            record["retained"] = json!(engine.selected(&state));
            record["sampler_cpu_seconds"] = json!(cpu_seconds() - budget.started);
            if let Err(error) = &result {
                record["fatal_error"] = json!(format!("{error:#}"));
            }
            journal.line(&record)?;
            result?;
            increment(&mut counts, "dimer_attempted");
            if record["accepted"] == true {
                increment(&mut counts, "dimer_accepted");
            }
            if record["status"] == "proposal_self_loop" {
                increment(&mut counts, "dimer_self_loop");
            }
        }
        journal.line(
            &json!({"kind":"retained_block","block":block,"production":block>warmup,
            "selected":engine.selected(&state),"sampler_cpu_seconds":cpu_seconds()-budget.started,
            "raw":budget.raw,"retained":budget.retained,"counts":counts}),
        )?;
        checkpoint(
            &output.join("checkpoint.json"),
            &config_hash,
            &binding_hash,
            job,
            block,
            engine.selected(&state),
            &journal,
            &budget,
            &counts,
        )?;
    }
    save(
        &output.join(if end == total {
            "terminal.json"
        } else {
            "paused.json"
        }),
        &json!({
        "complete":end==total,"job":job,"blocks":end,"counts":counts,"raw":budget.raw,"retained":budget.retained,
        "cpu_seconds":cpu_seconds()-budget.started,"trajectory":BoundFile::make(&journal_path)?,
        "geometry_load_cpu_seconds":geometry.load_cpu_seconds,
        "config_sha256":config_hash,"binding_sha256":binding_hash,"conditional_target":true}),
    )?;
    Ok(())
}
fn increment(counts: &mut Value, key: &str) {
    counts[key] = json!(counts[key].as_u64().unwrap() + 1);
}
fn main() -> Result<()> {
    let args = Args::parse();
    let plan: Value = serde_json::from_slice(&fs::read(&args.config)?)?;
    let binding = validate_binding(&args, &plan)?;
    let geometry = Geometry::load(&plan)?;
    match args.mode.as_str() {
        "prepare-starts" => prepare(&args, &plan, &geometry),
        "run" => run(&args, &plan, &binding, &geometry),
        _ => anyhow::bail!("mode must be prepare-starts or run"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preparation_uses_quaternion_sign_invariant_distance() {
        let a = Pose {
            position: [0.; 3],
            orientation: [1., 0., 0., 0.],
        };
        let minus = Pose {
            orientation: [-1., 0., 0., 0.],
            ..a
        };
        assert!(!far_enough([a, a], [minus, minus], 5., 10.));
        let b = Pose {
            position: [5., 0., 0.],
            ..a
        };
        assert!(far_enough([a, a], [b, a], 5., 10.));
    }
    #[test]
    fn named_stream_continuation_and_role_separation() {
        let a = (1..=8)
            .map(|b| rng(77, 1, "source", 2, b, "local/0/proposal").random::<u64>())
            .collect::<Vec<_>>();
        let b = (1..=3)
            .chain(4..=8)
            .map(|b| rng(77, 1, "source", 2, b, "local/0/proposal").random::<u64>())
            .collect::<Vec<_>>();
        assert_eq!(a, b);
        assert_ne!(
            a[0],
            rng(77, 1, "source", 2, 1, "m4/proposal").random::<u64>()
        );
    }
    #[test]
    fn journal_rejects_uncheckpointed_tail() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("evolving-journal-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let path = dir.join("journal.jsonl");
        let mut j = Journal::new(&path)?;
        j.line(&json!({"x":1}))?;
        let (bytes, rows, sha) = (j.bytes, j.rows, j.sha());
        drop(j);
        let mut j = Journal::resume(&path, bytes, rows, &sha)?;
        j.line(&json!({"x":2}))?;
        drop(j);
        assert!(Journal::resume(&path, bytes, rows, &sha).is_err());
        fs::remove_dir_all(dir)?;
        Ok(())
    }
}
