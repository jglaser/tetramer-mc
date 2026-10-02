//! Matched defensive proposal with analytic, world and path bath controls.
//! Each physical attempt starts from an EXACT rejection draw of the bounded
//! sphere equilibrium law. No previous endpoint is reused and no orientation
//! refresh follows the one physical MH step. This is not a production kernel.
//! Exactness is mathematical: RNG, FP64, geometry predicates and transformed
//! coordinates remain implementation obligations, not a floating-point proof.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    f64::consts::PI,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    defensive_dimer_proposal::{DefensiveDimerOutcome, DefensiveDimerProposal},
    depletion::{GateOptions, GateResult},
    dimer_tree_proposal::DimerTreeOutcome,
    docking::{DockingMethod, DockingProposal},
    flexible_subset::FlexibleSubset,
    geometry::{Atom, Shape, SphereTree},
    math::{Pose, norm, rotation, sub},
    proposal::FrozenRelativePoseProposal,
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
    singleton_path::{SingletonPath, SingletonPathResult},
};
const FRAME: Pose = Pose {
    position: [0.; 3],
    orientation: [1., 0., 0., 0.],
};
const RNG_DOMAIN: &[u8] = b"dimer-one-step-stationarity-v1";
const ANALYSIS_PROTOCOL: &str = "matched-defensive-bath-bonferroni5-v1";

#[derive(Parser)]
#[command(about = "Exact-IID-source, one-step two-sphere stationarity control")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Arm {
    Analytic,
    Poisson,
    SingletonPath,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum ProposalKind {
    Tree,
    DefensiveIndependent,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Population {
    id: String,
    proposal_kind: ProposalKind,
    arm: Arm,
    seed: u64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Gate {
    max_cells: usize,
    max_depth: u32,
    min_width: f64,
}
impl Gate {
    fn options(self) -> GateOptions {
        GateOptions {
            max_cells: self.max_cells,
            max_depth: self.max_depth,
            min_width: self.min_width,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: u32,
    core_radius: f64,
    exclusion_radius: f64,
    activity: f64,
    auxiliary_intensity: f64,
    root_radius: f64,
    min_separation: f64,
    max_separation: f64,
    rho: f64,
    uniform_probability: f64,
    uniform_half_width: f64,
    gate: Gate,
    draws_per_population: usize,
    populations: Vec<Population>,
    analysis_protocol: String,
}
impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == 1 && self.analysis_protocol == ANALYSIS_PROTOCOL,
            "unsupported protocol"
        );
        ensure!(
            [
                self.core_radius,
                self.exclusion_radius,
                self.auxiliary_intensity,
                self.root_radius,
                self.min_separation,
                self.max_separation,
                self.uniform_half_width
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.),
            "invalid positive geometry/intensity"
        );
        ensure!(
            self.exclusion_radius >= self.core_radius
                && self.min_separation >= 2. * self.core_radius
                && self.max_separation > self.min_separation,
            "invalid hard-core shell"
        );
        ensure!(
            self.max_separation.powi(3).is_finite() && self.root_radius.powi(3).is_finite(),
            "unrepresentable source volumes"
        );
        ensure!(
            self.activity.is_finite()
                && self.activity >= 0.
                && (self.activity + self.auxiliary_intensity).is_finite(),
            "invalid activity"
        );
        ensure!(
            (self.activity * lens(self.exclusion_radius, self.min_separation)).is_finite(),
            "unrepresentable radial envelope"
        );
        ensure!(
            self.rho.is_finite() && (0. ..=1.).contains(&self.rho),
            "invalid correlation"
        );
        ensure!(
            self.uniform_probability.is_finite()
                && self.uniform_probability > 0.
                && self.uniform_probability < 1.,
            "control needs a nondegenerate defensive mixture"
        );
        ensure!(
            self.uniform_half_width >= self.root_radius
                && self.uniform_half_width >= self.max_separation,
            "defensive cube must cover the toy domain"
        );
        self.gate.options().validate()?;
        ensure!(
            self.draws_per_population == 65536 && self.populations.len() == 8,
            "empty allocation"
        );
        self.draws_per_population
            .checked_mul(self.populations.len())
            .context("attempt overflow")?;
        let mut ids = BTreeSet::new();
        let mut seeds = BTreeSet::new();
        for p in &self.populations {
            ensure!(
                !p.id.is_empty()
                    && p.id
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
                "invalid population id"
            );
            ensure!(
                ids.insert(&p.id) && seeds.insert(p.seed),
                "duplicate population id/seed"
            );
            ensure!(
                p.proposal_kind == ProposalKind::DefensiveIndependent && p.arm == Arm::Poisson,
                "matched control requires defensive proposal metadata"
            );
        }
        Ok(())
    }
}
fn lens(r: f64, d: f64) -> f64 {
    if d >= 2. * r {
        0.
    } else {
        PI * (4. * r + d) * (2. * r - d).powi(2) / 12.
    }
}
fn separation(state: [Pose; 2]) -> f64 {
    norm(sub(state[1].position, state[0].position))
}
fn domain(c: &Config, state: [Pose; 2]) -> bool {
    let d = separation(state);
    norm(state[0].position) <= c.root_radius && d >= c.min_separation && d <= c.max_separation
}
fn seed(master: u64, step: usize, label: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(RNG_DOMAIN);
    h.update(master.to_le_bytes());
    h.update((step as u64).to_le_bytes());
    h.update(label.as_bytes());
    h.finalize().into()
}
fn rng(master: u64, step: usize, label: &str) -> StdRng {
    StdRng::from_seed(seed(master, step, label))
}
fn unit<const N: usize>(normal: [f64; N]) -> Result<[f64; N]> {
    let length = normal.iter().map(|v| v * v).sum::<f64>().sqrt();
    ensure!(
        length.is_finite() && length > 0.,
        "invalid unit-vector draw"
    );
    Ok(normal.map(|v| v / length))
}
#[derive(Clone, Debug, Serialize)]
struct SourceTrace {
    radial_trials: usize,
    radial_uniform: f64,
    radial_log_uniform: f64,
    radial_log_acceptance: f64,
    sampled_distance: f64,
    root_radius_uniform: f64,
    root_direction_normals: [f64; 3],
    relative_direction_normals: [f64; 3],
    quaternion_normals: [[f64; 4]; 2],
}
#[derive(Clone, Debug, Serialize)]
struct Source {
    state: [Pose; 2],
    trace: SourceTrace,
}
fn exact_source(c: &Config, rng: &mut StdRng) -> Result<Source> {
    // Baseline is uniform shell volume, so d^2 is already in the proposal.
    // Monotone sphere-lens volume makes V(d_min) a global bound. Rejection
    // sampling needs no quadrature, approximate normalization, or reweighting.
    let low = c.min_separation.powi(3);
    let width = c.max_separation.powi(3) - low;
    let maximum = lens(c.exclusion_radius, c.min_separation);
    let mut trials = 0usize;
    let (u, log_u, d, log_acceptance) = loop {
        trials = trials
            .checked_add(1)
            .context("source rejection counter overflow")?;
        let u: f64 = rng.sample(Open01);
        let threshold: f64 = rng.sample(Open01);
        let d = (low + u * width).cbrt();
        let log_acceptance = c.activity * (lens(c.exclusion_radius, d) - maximum);
        ensure!(
            log_acceptance.is_finite() && log_acceptance <= 0.,
            "invalid exact radial envelope"
        );
        if threshold.ln() < log_acceptance {
            break (u, threshold.ln(), d, log_acceptance);
        }
    };
    let root_radius_uniform: f64 = rng.sample(Open01);
    let root_direction_normals = std::array::from_fn(|_| StandardNormal.sample(rng));
    let relative_direction_normals = std::array::from_fn(|_| StandardNormal.sample(rng));
    let root_direction = unit(root_direction_normals)?;
    let relative_direction = unit(relative_direction_normals)?;
    let root = root_direction.map(|v| v * c.root_radius * root_radius_uniform.cbrt());
    let child = std::array::from_fn(|i| root[i] + d * relative_direction[i]);
    let quaternion_normals: [[_; 4]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| StandardNormal.sample(rng)));
    let state = [
        Pose {
            position: root,
            orientation: unit(quaternion_normals[0])?,
        },
        Pose {
            position: child,
            orientation: unit(quaternion_normals[1])?,
        },
    ];
    for p in state {
        p.validate()?;
    }
    ensure!(domain(c, state), "exact source outside declared domain");
    Ok(Source {
        state,
        trace: SourceTrace {
            radial_trials: trials,
            radial_uniform: u,
            radial_log_uniform: log_u,
            radial_log_acceptance: log_acceptance,
            sampled_distance: d,
            root_radius_uniform,
            root_direction_normals,
            relative_direction_normals,
            quaternion_normals,
        },
    })
}
fn shape(radius: f64) -> Shape {
    Shape {
        name: "dimer-tree-equilibrium hard sphere".into(),
        volume: 4. * PI * radius.powi(3) / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius,
        }],
    }
}
fn atlas(sha: &str) -> Value {
    let covariance: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
    let q = [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ];
    json!({"coordinate_convention":"anchor-body-relative","shape_sha256":sha,"angular_length":1.,
        "weights":vec![0.25;4],"anchors":q.map(|v|json!({"position":vec![0.;3],"rotation":rotation(v)})),
        "means":vec![[0.;6];4],"covariances":vec![covariance;4]})
}
#[derive(Serialize)]
struct Attempt {
    population_id: String,
    step: usize,
    source_seed: [u8; 32],
    source: SourceTrace,
    old_state: [Pose; 2],
    proposed_state: Option<[Pose; 2]>,
    state: [Pose; 2],
    old_d: f64,
    proposed_d: Option<f64>,
    d: f64,
    proposal: Option<DimerTreeOutcome>,
    defensive_proposal: Option<DefensiveDimerOutcome>,
    proposal_null: bool,
    domain_valid: bool,
    hard_valid: bool,
    q_correction: Option<f64>,
    analytic_log_weight: Option<f64>,
    gate: Option<GateResult>,
    path_gate: Option<SingletonPathResult>,
    log_ratio: Option<f64>,
    log_uniform: f64,
    accepted: bool,
    orientation_refresh_count: usize,
    attempt_cpu_seconds: f64,
    cpu_seconds: f64,
    error: Option<String>,
}
#[derive(Default, Serialize)]
struct Counts {
    attempts: usize,
    accepted: usize,
    proposal_null: usize,
    domain_rejected: usize,
    hard_rejected: usize,
    gate_calls: usize,
    raw_points: u64,
    source_radial_trials: usize,
    errors: usize,
}
fn write_line(w: &mut BufWriter<File>, row: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *w, row)?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(())
}

#[derive(Clone, Debug, Serialize)]
struct Endpoint {
    arm: Arm,
    state: [Pose; 2],
    d: f64,
    gate: Option<GateResult>,
    path_gate: Option<SingletonPathResult>,
    log_ratio: Option<f64>,
    accepted: bool,
}
#[derive(Serialize)]
struct MatchedAttempt {
    common: Attempt,
    outcomes: Vec<Endpoint>,
}
fn attempt(
    c: &Config,
    p: &Population,
    step: usize,
    tree: &SphereTree,
    defensive: &DefensiveDimerProposal<'_>,
) -> Result<MatchedAttempt> {
    let start = cpu_seconds();
    let source_seed = seed(p.seed, step, "source");
    let source = exact_source(c, &mut StdRng::from_seed(source_seed))?;
    let uniform: f64 = rng(p.seed, step, "mh").sample(Open01);
    let old = source.state;
    let mut row = MatchedAttempt {
        common: Attempt {
            population_id: p.id.clone(),
            step,
            source_seed,
            source: source.trace,
            old_state: old,
            proposed_state: None,
            state: old,
            old_d: separation(old),
            proposed_d: None,
            d: separation(old),
            proposal: None,
            defensive_proposal: None,
            proposal_null: true,
            domain_valid: false,
            hard_valid: false,
            q_correction: None,
            analytic_log_weight: None,
            gate: None,
            path_gate: None,
            log_ratio: None,
            log_uniform: uniform.ln(),
            accepted: false,
            orientation_refresh_count: 0,
            attempt_cpu_seconds: 0.,
            cpu_seconds: 0.,
            error: None,
        },
        outcomes: Vec::new(),
    };
    let result = (|| -> Result<()> {
        let proposal =
            defensive.propose(&mut rng(p.seed, step, "proposal"), FRAME, old[0], old[1])?;
        let candidate = proposal
            .candidate
            .as_ref()
            .map(|x| ([x.root, x.child], x.diagnostics.log_reverse_forward));
        row.common.defensive_proposal = Some(proposal);
        row.common.proposal_null = candidate.is_none();
        if let Some((new, q)) = candidate {
            row.common.proposed_state = Some(new);
            let d = separation(new);
            row.common.proposed_d = Some(d);
            row.common.q_correction = Some(q);
            row.common.analytic_log_weight = Some(
                c.activity
                    * (lens(c.exclusion_radius, d) - lens(c.exclusion_radius, row.common.old_d)),
            );
            row.common.domain_valid = domain(c, new);
            let gate = FlexibleSubset::new(
                tree,
                &old,
                &[0, 1],
                &new,
                c.exclusion_radius - c.core_radius,
            )?;
            ensure!(gate.spectator_count() == 0, "virtual frame entered bath");
            row.common.hard_valid = gate.hard_valid(None, [0.; 3]);
            for arm in [Arm::Analytic, Arm::Poisson, Arm::SingletonPath] {
                let mut endpoint = Endpoint {
                    arm,
                    state: old,
                    d: row.common.old_d,
                    gate: None,
                    path_gate: None,
                    log_ratio: None,
                    accepted: false,
                };
                // Each arm starts at the SAME source. No arm's accepted state feeds another.
                if row.common.domain_valid && row.common.hard_valid {
                    let bath = match arm {
                        Arm::Analytic => row.common.analytic_log_weight.unwrap(),
                        Arm::Poisson => {
                            let g = gate.sample(
                                &mut rng(p.seed, step, "world_gate"),
                                c.auxiliary_intensity,
                                c.activity,
                                c.gate.options(),
                            )?;
                            let log = g.log_weight;
                            endpoint.gate = Some(g);
                            log
                        }
                        Arm::SingletonPath => {
                            let path = SingletonPath::new(
                                tree,
                                &old,
                                &[0, 1],
                                &new,
                                c.exclusion_radius - c.core_radius,
                            )?;
                            let g = path.sample(
                                &mut rng(p.seed, step, "path_gate"),
                                c.auxiliary_intensity,
                                c.activity,
                                c.gate.options(),
                            )?;
                            let log = g.aggregate.log_weight;
                            endpoint.gate = Some(g.aggregate);
                            endpoint.path_gate = Some(g);
                            log
                        }
                    };
                    let log = q + bath;
                    ensure!(log.is_finite(), "nonfinite matched MH factor");
                    endpoint.log_ratio = Some(log);
                    endpoint.accepted = row.common.log_uniform < log.min(0.);
                    if endpoint.accepted {
                        endpoint.state = new;
                        endpoint.d = d;
                    }
                }
                ensure!(domain(c, endpoint.state), "matched endpoint outside domain");
                row.outcomes.push(endpoint);
            }
        } else {
            for arm in [Arm::Analytic, Arm::Poisson, Arm::SingletonPath] {
                row.outcomes.push(Endpoint {
                    arm,
                    state: old,
                    d: row.common.old_d,
                    gate: None,
                    path_gate: None,
                    log_ratio: None,
                    accepted: false,
                });
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        row.common.error = Some(format!("{error:#}"));
    }
    row.common.attempt_cpu_seconds = cpu_seconds() - start;
    ensure!(
        row.common.attempt_cpu_seconds.is_finite(),
        "nonfinite CPU clock"
    );
    Ok(row)
}
fn run_population(
    c: &Config,
    p: &Population,
    tree: &SphereTree,
    docking: &DockingProposal,
    w: &mut BufWriter<File>,
) -> Result<Value> {
    let defensive =
        DefensiveDimerProposal::new(docking, c.uniform_half_width, c.uniform_probability)?;
    let start = cpu_seconds();
    let mut counts = Counts::default();
    let mut accepted = [0usize; 3];
    for step in 0..c.draws_per_population {
        let mut row = match attempt(c, p, step, tree, &defensive) {
            Ok(row) => row,
            Err(error) => {
                write_line(
                    w,
                    &json!({"population_id":p.id,"step":step,"source_seed":seed(p.seed,step,"source"),"error":format!("source/attempt failure: {error:#}")}),
                )?;
                return Err(error);
            }
        };
        row.common.cpu_seconds = cpu_seconds() - start;
        counts.attempts += 1;
        counts.proposal_null += usize::from(row.common.proposal_null);
        counts.domain_rejected +=
            usize::from(!row.common.proposal_null && !row.common.domain_valid);
        counts.hard_rejected += usize::from(!row.common.proposal_null && !row.common.hard_valid);
        counts.source_radial_trials += row.common.source.radial_trials;
        counts.errors += usize::from(row.common.error.is_some());
        for (i, endpoint) in row.outcomes.iter().enumerate() {
            accepted[i] += usize::from(endpoint.accepted);
            counts.accepted += usize::from(endpoint.accepted);
            if let Some(g) = endpoint.gate {
                counts.gate_calls += 1;
                counts.raw_points += g.raw_points;
            }
        }
        write_line(w, &row)?;
        ensure!(
            row.common.error.is_none(),
            "matched pair {step} failed; partial outcomes retained: {:?}",
            row.common.error
        );
        ensure!(row.outcomes.len() == 3, "incomplete matched outcomes");
    }
    Ok(
        json!({"population":p,"counts":counts,"accepted_by_arm":accepted,"cpu_seconds":cpu_seconds()-start}),
    )
}
fn execute(args: &Args, c: &Config, bytes: &[u8]) -> Result<()> {
    let shape = shape(c.core_radius);
    let shape_sha = hash_bytes(&serde_json::to_vec(&shape)?);
    let atlas = atlas(&shape_sha);
    let model = FrozenRelativePoseProposal::from_json_str_open(
        &atlas.to_string(),
        [80.; 3],
        0.1,
        &shape_sha,
    )?;
    let docking = DockingProposal::new(model, DockingMethod::PosteriorInvolution, c.rho, [0.; 3])?;
    let tree = SphereTree::new(shape.clone())?;
    let exe = std::env::current_exe()?;
    save(
        &args.out.join("manifest.json"),
        &json!({"schema":1,"protocol":c,"config_sha256":hash_bytes(bytes),
        "executable":exe,"executable_sha256":hash_file(&exe)?,"example_source_sha256":hash_bytes(include_bytes!("dimer_matched_stationarity.rs")),
        "shape":shape,"shape_sha256":shape_sha,"atlas":atlas,"atlas_sha256":hash_bytes(&serde_json::to_vec(&atlas)?),
        "virtual_frame":FRAME,"physical_particles":2,"tree_order":[0,1],"allocation_attempts":c.draws_per_population*c.populations.len(),
        "rng_domain":"dimer-one-step-stationarity-v1","rng_seed_layout":"SHA256(domain || master_seed LE u64 || zero-based attempt LE u64 || role ASCII)",
        "rng_roles":["source","proposal","world_gate","path_gate","mh"],"orientation_refresh_count":0,
        "source":"independent exact rejection from radial d^3 baseline with exp[z(V(d)-V(d_min))] acceptance; root uniform ball; directions and quaternions normalized independent normals",
        "endpoint_decisions":3*c.draws_per_population*c.populations.len(),"scope":"matched one-step analytic/world/path decisions from the same independent source, candidate and MH uniform; no trajectory continuation, adaptation, protein or assembly inference"}),
    )?;
    let mut summaries = Vec::new();
    let started = cpu_seconds();
    for p in &c.populations {
        save(
            &args.out.join("status.json"),
            &json!({"phase":"running","population_id":p.id,"completed_populations":summaries.len()}),
        )?;
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(args.out.join(format!("{}.jsonl", p.id)))?,
        );
        let summary = run_population(c, p, &tree, &docking, &mut writer)?;
        eprintln!(
            "completed {}: {} independent pairs",
            p.id, c.draws_per_population
        );
        summaries.push(summary);
        save(
            &args.out.join("summary.json"),
            &json!({"schema":1,"populations":summaries,"cpu_seconds":cpu_seconds()-started}),
        )?;
    }
    save(
        &args.out.join("status.json"),
        &json!({"phase":"completed","completed_populations":summaries.len(),
        "attempts":c.draws_per_population*c.populations.len(),"cpu_seconds":cpu_seconds()-started}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = fs::read(&args.config)?;
    let c: Config = serde_json::from_slice(&bytes)?;
    c.validate()?;
    fs::create_dir(&args.out).context("output must not exist; parent must exist")?;
    let result = execute(&args, &c, &bytes);
    if let Err(error) = &result {
        save(
            &args.out.join("status.json"),
            &json!({"phase":"failed","error":format!("{error:#}")}),
        )?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config {
        serde_json::from_str(include_str!("dimer-matched-stationarity.json")).unwrap()
    }
    #[test]
    fn frozen_allocation_and_independent_streams() {
        let c = config();
        c.validate().unwrap();
        assert_eq!(c.populations.len() * c.draws_per_population, 524288);
        let mut all = BTreeSet::new();
        for p in &c.populations {
            for step in 0..4 {
                for role in ["source", "proposal", "world_gate", "path_gate", "mh"] {
                    assert!(all.insert(seed(p.seed, step, role)));
                }
            }
        }
        let mut bad = config();
        bad.populations[1].seed = bad.populations[0].seed;
        assert!(bad.validate().is_err());
        let mut bad = config();
        bad.draws_per_population = 16384;
        assert!(bad.validate().is_err());
    }
    #[test]
    fn source_math_and_replay_are_unchanged() {
        let c = config();
        let a = exact_source(&c, &mut rng(610027000, 0, "source")).unwrap();
        let b = exact_source(&c, &mut rng(610027000, 0, "source")).unwrap();
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
        for d in [0.4, 0.6, 1., 1.5, 2., 2.5] {
            assert!(lens(1., d) <= lens(1., 0.4));
        }
    }
    #[test]
    fn zero_activity_shared_decisions_and_no_hidden_refresh() -> Result<()> {
        let mut c = config();
        c.activity = 0.;
        c.validate()?;
        let shape = shape(c.core_radius);
        let sha = hash_bytes(&serde_json::to_vec(&shape)?);
        let model = FrozenRelativePoseProposal::from_json_str_open(
            &atlas(&sha).to_string(),
            [80.; 3],
            0.1,
            &sha,
        )?;
        let docking =
            DockingProposal::new(model, DockingMethod::PosteriorInvolution, c.rho, [0.; 3])?;
        let defensive =
            DefensiveDimerProposal::new(&docking, c.uniform_half_width, c.uniform_probability)?;
        let tree = SphereTree::new(shape)?;
        for step in 0..8 {
            let r = attempt(&c, &c.populations[0], step, &tree, &defensive)?;
            assert!(r.common.error.is_none());
            assert_eq!(r.outcomes.len(), 3);
            let expected = r.common.domain_valid
                && r.common.hard_valid
                && r.common.log_uniform < r.common.q_correction.unwrap().min(0.);
            for e in &r.outcomes {
                assert_eq!(e.accepted, expected);
                assert_eq!(e.state, r.outcomes[0].state);
                if let Some(g) = e.gate {
                    assert_eq!(g.raw_points, 0);
                    assert_eq!(g.log_weight, 0.);
                }
            }
            assert_eq!(r.common.orientation_refresh_count, 0);
            assert_eq!(r.common.state, r.common.old_state);
        }
        Ok(())
    }
}
