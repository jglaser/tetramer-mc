//! Passive source-chart coverage at frozen configurations. No proposals, bath
//! realizations, native classifications, or physical updates are performed.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use tetramer_mc::{
    cluster_phase::{ClusterPhaseConfig, ContactGraph, anchor_pool},
    docking::{DockingMethod, DockingProposal},
    geometry::{Shape, SphereTree},
    math::Pose,
    oligomer_proposal::{OligomerConfig, OligomerMixture},
    proposal::FrozenRelativePoseProposal,
    simulation::{Boundary, Config, cpu_seconds, hash_bytes, hash_file, save},
    spherical::Container,
};

#[derive(Parser, Serialize)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// Immutable JSONL snapshot; the complete input must not change while read.
    #[arg(long)]
    trajectory: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_delimiter = ',', required = true)]
    sweeps: Vec<u64>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 20261002001)]
    seed: u64,
    /// Uniformly selected singleton/dimer/trimer fusion contexts per frame and
    /// size. All singleton pair-neighbor diagnostics are always evaluated.
    #[arg(long, default_value_t = 8)]
    max_contexts_per_size: usize,
    /// Skip expensive fused catalogues; retain all pair/member coverage tests.
    #[arg(long, default_value_t = false)]
    no_fusion: bool,
}
#[derive(Deserialize)]
struct Frame {
    sweep: u64,
    poses: Vec<Pose>,
    boundary: String,
}

fn log_sum(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + xs.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}
fn stream(seed: u64, sweep: u64, label: &str) -> StdRng {
    let sha = hash_bytes(format!("source-coverage-v1:{seed}:{sweep}:{label}").as_bytes());
    StdRng::seed_from_u64(u64::from_str_radix(&sha[..16], 16).expect("hex hash"))
}
fn norm6(z: &[f64; 6]) -> f64 {
    z.iter().fold(0., |n, x| n.hypot(*x))
}
fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

/// Keep every label, including seam failures and zero-weight terms. Posterior
/// statistics refer to learned labels at this one pose, never target weights.
fn summarize(logs: Vec<f64>, coordinates: Vec<Result<[f64; 6]>>, labels: Vec<Value>) -> Value {
    let log_g = log_sum(&logs);
    let mut entries = Vec::with_capacity(logs.len());
    let mut best = None;
    let mut min_norm = None::<f64>;
    let mut posterior_norm = 0.;
    let mut posterior_norm2 = 0.;
    let mut represented_mass = [0.; 3];
    let mut resolved_posterior = 0.;
    let mut errors = 0;
    for (i, ((&log_joint, coordinate), label)) in
        logs.iter().zip(coordinates).zip(labels).enumerate()
    {
        if log_joint.is_finite() && best.is_none_or(|j| log_joint > logs[j]) {
            best = Some(i);
        }
        let posterior = if log_g.is_finite() {
            (log_joint - log_g).exp()
        } else {
            0.
        };
        let (z_norm, error) = match coordinate {
            Ok(z) => (finite(norm6(&z)), None),
            Err(e) => {
                errors += 1;
                (None, Some(e.to_string()))
            }
        };
        if let Some(n) = z_norm {
            min_norm = Some(min_norm.map_or(n, |old| old.min(n)));
            resolved_posterior += posterior;
            posterior_norm += posterior * n;
            posterior_norm2 += posterior * n * n;
            for (j, threshold) in [3., 6., 10.].iter().enumerate() {
                if n <= *threshold {
                    represented_mass[j] += posterior;
                }
            }
        }
        entries.push(
            json!({"label":i,"description":label,"log_joint_density":finite(log_joint),
            "posterior_probability":posterior,"standardized_norm":z_norm,"encode_error":error}),
        );
    }
    json!({"log_learned_density":finite(log_g),"labels":entries,"highest_density_label":best,
        "highest_density_label_norm":best.and_then(|i| entries[i]["standardized_norm"].as_f64()),
        "minimum_chart_norm":min_norm,"posterior_mean_norm":finite(posterior_norm),
        "posterior_rms_norm":finite(posterior_norm2.sqrt()),"posterior_mass_norm_le_3_6_10":represented_mass,
        "resolved_posterior_mass":resolved_posterior,"encoding_failures":errors,
        "density_status":if log_g.is_finite() {"finite"}else{"no_finite_learned_density"}})
}
fn pair_summary(proposal: &DockingProposal, pose: Pose, anchor: Pose) -> Result<Value> {
    let logs = proposal.branch_log_densities(pose, anchor)?;
    let coordinates = (0..logs.len())
        .map(|i| proposal.source_chart_coordinates(i, pose, anchor))
        .collect();
    let labels = (0..logs.len()).map(|i| json!({"branch":i})).collect();
    Ok(summarize(logs, coordinates, labels))
}
fn member_summary(proposal: &DockingProposal, members: &[Pose], pool: &[Pose]) -> Result<Value> {
    ensure!(
        !pool.is_empty(),
        "empty pool has no conditional source density"
    );
    let offset = ((members.len() * pool.len()) as f64).ln();
    let mut logs = vec![];
    let mut coordinates = vec![];
    let mut labels = vec![];
    for (mi, &m) in members.iter().enumerate() {
        for (ai, &a) in pool.iter().enumerate() {
            for (b, l) in proposal.branch_log_densities(m, a)?.into_iter().enumerate() {
                logs.push(l - offset);
                coordinates.push(proposal.source_chart_coordinates(b, m, a));
                labels.push(json!({"kind":"single","member":mi,"anchor":ai,"branch":b}));
            }
        }
    }
    let summary = summarize(logs, coordinates, labels);
    let explicit = proposal.members_log_density(members, pool)?;
    if explicit.is_finite() {
        ensure!(
            (summary["log_learned_density"].as_f64().unwrap() - explicit).abs() < 1e-8,
            "independent member-density reconstruction disagrees"
        );
    }
    Ok(summary)
}
fn fusion_summary(
    proposal: &DockingProposal,
    tree: &SphereTree,
    wall: &Container,
    members: &[Pose],
    spectators: &[Pose],
    pool: &[Pose],
    config: &OligomerConfig,
) -> Result<Value> {
    let start = Instant::now();
    let cpu = cpu_seconds();
    let mixture = OligomerMixture::build(proposal, tree, wall, members, spectators, pool, config)?;
    let build_cpu = cpu_seconds() - cpu;
    let logs = mixture.label_logs(members[0]);
    let coordinates = (0..logs.len())
        .map(|i| mixture.source_chart_coordinates(i, members[0]))
        .collect();
    let labels = (0..logs.len()).map(|i| mixture.describe(i)).collect();
    let mut out = summarize(logs, coordinates, labels);
    out["build_cpu_seconds"] = json!(build_cpu);
    out["wall_seconds"] = json!(start.elapsed().as_secs_f64());
    out["fit_candidates"] = json!(mixture.fit_candidates());
    out["hard_checks"] = json!(mixture.hard_checks());
    out["label_count"] = json!(mixture.label_count());
    out["fused_count"] = json!(
        (0..mixture.label_count())
            .filter(|&i| mixture.describe(i)["kind"] == "fused")
            .count()
    );
    Ok(out)
}
fn neighbors(graph: &ContactGraph, members: &[usize]) -> Vec<usize> {
    (0..graph.adjacency.len())
        .filter(|i| !members.contains(i) && members.iter().any(|&m| graph.adjacency[m][*i]))
        .collect()
}
fn write_line(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    out.write_all(b"\n")?;
    Ok(())
}
fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.parent().unwrap_or(Path::new(".")).join(path)
    }
}
fn read_immutable(path: &Path) -> Result<Vec<u8>> {
    let before = fs::metadata(path)?;
    let raw = fs::read(path)?;
    let after = fs::metadata(path)?;
    ensure!(
        before.len() == after.len()
            && before.modified()? == after.modified()?
            && raw.len() as u64 == before.len(),
        "input changed during read; use frozen snapshot: {}",
        path.display()
    );
    Ok(raw)
}
/// Jacobi eigenvalues of a symmetric 6x6 covariance; only diagnostic, never a
/// regularization or a change to the chart. Rank uses relative 1e-12 threshold.
fn covariance_summary(mut c: [[f64; 6]; 6]) -> Value {
    let diagonal: Vec<_> = (0..6).map(|i| c[i][i].sqrt()).collect();
    let scale = (0..6).map(|i| c[i][i].abs()).fold(0., f64::max);
    for _ in 0..512 {
        let mut maximum = 0.;
        let mut p = 0;
        let mut q = 1;
        for i in 0..6 {
            for j in i + 1..6 {
                if c[i][j].abs() > maximum {
                    maximum = c[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if maximum <= scale * 1e-14 {
            break;
        }
        let angle = 0.5 * (2. * c[p][q]).atan2(c[q][q] - c[p][p]);
        let (s, t) = angle.sin_cos();
        let app = c[p][p];
        let aqq = c[q][q];
        let apq = c[p][q];
        for k in 0..6 {
            if k != p && k != q {
                let x = c[k][p];
                let y = c[k][q];
                c[k][p] = t * x - s * y;
                c[p][k] = c[k][p];
                c[k][q] = s * x + t * y;
                c[q][k] = c[k][q];
            }
        }
        c[p][p] = t * t * app - 2. * s * t * apq + s * s * aqq;
        c[q][q] = s * s * app + 2. * s * t * apq + t * t * aqq;
        c[p][q] = 0.;
        c[q][p] = 0.;
    }
    let mut values: Vec<_> = (0..6).map(|i| c[i][i]).collect();
    values.sort_by(f64::total_cmp);
    let max = values[5];
    json!({"coordinate_standard_deviations":diagonal,"covariance_eigenvalues":values,
        "numerical_rank_relative_1e_12":values.iter().filter(|&&x|x>max*1e-12).count(),
        "log_covariance_determinant":finite(values.iter().map(|x|x.ln()).sum()),
        "condition_number":finite(max/values[0])})
}

fn run(args: &Args) -> Result<()> {
    let start = Instant::now();
    let cpu_start = cpu_seconds();
    ensure!(!args.out.exists(), "fresh output directory required");
    let wanted: BTreeSet<_> = args.sweeps.iter().copied().collect();
    ensure!(
        !wanted.is_empty() && wanted.len() == args.sweeps.len(),
        "provide unique selected sweeps"
    );
    let cfg_raw = read_immutable(&args.config)?;
    let cfg: Config = serde_json::from_slice(&cfg_raw)?;
    cfg.validate()?;
    let radius = match cfg.boundary {
        Boundary::Spherical { radius } => radius,
        _ => anyhow::bail!(
            "spherical snapshots only; periodic anchor/image handling is not implemented"
        ),
    };
    let shape_path = resolve(&args.config, &cfg.shape);
    let shape_raw = read_immutable(&shape_path)?;
    let shape: Shape = serde_json::from_slice(&shape_raw)?;
    let tree = SphereTree::new(shape.clone())?;
    let wall = Container::new(radius, &tree)?;
    let mut inflated = shape;
    for atom in &mut inflated.atoms {
        atom.radius += cfg.depletant_radius;
    }
    let exclusion = SphereTree::new(inflated)?;
    let model_raw = read_immutable(&args.model)?;
    let model = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&model_raw)?,
        [2. * (radius + tree.bound); 3],
        cfg.learned_uniform_weight,
        &hash_bytes(&shape_raw),
    )?;
    let branches = model.virtual_branches();
    let parameters = model.component_parameters();
    let branch_catalogue: Vec<_> = branches
        .iter()
        .enumerate()
        .map(|(b, v)| {
            json!({
        "branch":b,"component_index":v.component_index,"inverted":v.inverted,"weight":v.weight,
        "covariance":covariance_summary(parameters[v.component_index].covariance)})
        })
        .collect();
    let cluster_cfg = cfg.cluster_phase.clone().unwrap_or_default();
    let oligomer = cluster_cfg.oligomer.clone().unwrap_or_default();
    let correlation = cfg
        .frozen_posterior
        .map_or(cluster_cfg.correlation, |x| x.correlation);
    let proposal = DockingProposal::new(
        model,
        DockingMethod::PosteriorInvolution,
        correlation,
        [0.; 3],
    )?;
    let trajectory_raw = read_immutable(&args.trajectory)?;
    let mut frames = vec![];
    let mut selected_raw = Vec::new();
    let mut previous = None;
    for (lineno, line) in std::str::from_utf8(&trajectory_raw)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let frame: Frame = serde_json::from_str(line)
            .with_context(|| format!("trajectory line {}", lineno + 1))?;
        ensure!(
            previous.is_none_or(|s| frame.sweep > s),
            "non-increasing frame order"
        );
        previous = Some(frame.sweep);
        if !wanted.contains(&frame.sweep) {
            continue;
        }
        ensure!(
            frame.boundary == "spherical" && frame.poses.len() == cfg.initial_poses.len(),
            "frame domain/count mismatch"
        );
        for p in &frame.poses {
            p.validate()?;
        }
        selected_raw.extend_from_slice(line.as_bytes());
        selected_raw.push(b'\n');
        frames.push(frame);
    }
    ensure!(
        frames.len() == wanted.len(),
        "one or more requested sweeps missing"
    );
    fs::create_dir_all(&args.out)?;
    for (name, raw) in [
        ("config.json", &cfg_raw),
        ("shape.json", &shape_raw),
        ("model.json", &model_raw),
        ("selected-frames.jsonl", &selected_raw),
    ] {
        fs::write(args.out.join(name), raw)?;
    }
    let source = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(args.out.join("source-bundle.json"), source)?;
    save(
        &args.out.join("plan.json"),
        &json!({"schema":"source-coverage-v1","arguments":args,
        "config_sha256":hash_bytes(&cfg_raw),"shape_sha256":hash_bytes(&shape_raw),"model_sha256":hash_bytes(&model_raw),
        "trajectory_sha256":hash_bytes(&trajectory_raw),"selected_frames_sha256":hash_bytes(&selected_raw),
        "executable_sha256":hash_file(&std::env::current_exe()?)?,"source_bundle_sha256":hash_bytes(source),
        "depletant_radius":cfg.depletant_radius,"activity":cfg.reservoir_density,"spherical_radius":radius,
        "scope":"read-only geometric and learned-density diagnostics, no physical draws, native labels or selection",
        "frame_coordinates":"already sphere centered; coordinate_wall_center is not subtracted",
        "density_measure":"translation volume times normalized rotational Haar; learned density excludes defensive uniform branch",
        "selection":"all singletons; one independent uniform spectator per body; every directed contact pair; uniformly shuffled connected subset contexts within each size",
        "context_primary_law":"cluster-phase anchor law for all context sizes (singleton is a control, not current global anchor law)",
        "empty_contact_pool":"record absent; never replace with a selected nearby anchor",
        "context_config":cluster_cfg,"fusion_config":oligomer,"correlation":correlation,
        "branch_catalogue":branch_catalogue}),
    )?;
    let mut singles = BufWriter::new(fs::File::create(args.out.join("singletons.jsonl"))?);
    let mut contexts = BufWriter::new(fs::File::create(args.out.join("contexts.jsonl"))?);
    let mut frame_rows = vec![];
    let mut failures = 0;
    let mut singleton_count = 0;
    let mut context_count = 0;
    for frame in frames {
        let frame_cpu = cpu_seconds();
        let graph = ContactGraph::build(&exclusion, &frame.poses);
        let n = frame.poses.len();
        let edges: Vec<_> = (0..n)
            .flat_map(|i| {
                (i + 1..n)
                    .filter(|&j| graph.adjacency[i][j])
                    .map(move |j| [i, j])
                    .collect::<Vec<_>>()
            })
            .collect();
        for i in 0..n {
            let c = graph.subset_context(&[i])?;
            let touching = neighbors(&graph, &[i]);
            let mut rng = stream(args.seed, frame.sweep, &format!("singleton:{i}"));
            let draw = rng.random_range(0..n - 1);
            let a = if draw >= i { draw + 1 } else { draw };
            let mut row = json!({"sweep":frame.sweep,"body":i,"context":c,"uniform_anchor":a,
                "uniform_anchor_contact":touching.contains(&a),"touching_anchors":touching});
            match pair_summary(&proposal, frame.poses[i], frame.poses[a]) {
                Ok(v) => row["uniform_anchor_fit"] = v,
                Err(e) => {
                    failures += 1;
                    row["uniform_anchor_error"] = json!(format!("{e:#}"));
                }
            }
            let mut pairs = vec![];
            for &a in &touching {
                match pair_summary(&proposal, frame.poses[i], frame.poses[a]) {
                    Ok(v) => pairs.push(json!({"anchor":a,"fit":v})),
                    Err(e) => {
                        failures += 1;
                        pairs.push(json!({"anchor":a,"error":format!("{e:#}")}));
                    }
                }
            }
            row["contact_fits"] = json!(pairs);
            write_line(&mut singles, &row)?;
            singleton_count += 1;
        }
        singles.flush()?;
        let eligibility = ClusterPhaseConfig {
            dimer_rate: 1.,
            trimer_rate: 1.,
            ..ClusterPhaseConfig::default()
        };
        let channels = graph.channels(&eligibility);
        let mut allocation = vec![];
        for size in 1..=3 {
            let mut choices: Vec<Vec<usize>> = if size == 1 {
                (0..n).map(|i| vec![i]).collect()
            } else {
                channels
                    .iter()
                    .filter(|c| c.members.len() == size)
                    .map(|c| c.members.clone())
                    .collect()
            };
            let eligible = choices.len();
            let mut rng = stream(args.seed, frame.sweep, &format!("contexts:{size}"));
            choices.shuffle(&mut rng);
            choices.truncate(args.max_contexts_per_size);
            allocation.push(json!({"size":size,"eligible":eligible,"selected":choices.len()}));
            for members in choices {
                let context = graph.subset_context(&members)?;
                let weights = graph.anchor_probabilities(
                    &members,
                    cluster_cfg.anchor_contact_uniform_probability.unwrap_or(1.),
                )?;
                let mut anchor_rng =
                    stream(args.seed, frame.sweep, &format!("primary:{members:?}"));
                let u = anchor_rng.random::<f64>();
                let mut sum = 0.;
                let primary = weights
                    .iter()
                    .enumerate()
                    .find_map(|(i, w)| {
                        sum += w;
                        (u < sum).then_some(i)
                    })
                    .unwrap_or_else(|| weights.iter().rposition(|w| *w > 0.).unwrap());
                let selected_pool =
                    anchor_pool(&frame.poses, &members, primary, cluster_cfg.anchor_count)?;
                let actual_pool = neighbors(&graph, &members);
                let m: Vec<_> = members.iter().map(|&i| frame.poses[i]).collect();
                let spectators: Vec<_> = (0..n)
                    .filter(|i| !members.contains(i))
                    .map(|i| frame.poses[i])
                    .collect();
                let mut row = json!({"sweep":frame.sweep,"members":members,"context":context,"primary_anchor":primary,
                    "primary_probability":weights[primary],"production_pool":selected_pool,"complete_contact_pool":actual_pool,
                    "production_pool_contact_coverage":actual_pool.iter().filter(|a|selected_pool.contains(a)).count(),
                    "singleton_context_is_control":size==1});
                for (label, indices) in [
                    ("production", &selected_pool),
                    ("all_contacts", &actual_pool),
                ] {
                    if indices.is_empty() {
                        row[label] = json!({"status":"no_external_contact"});
                        continue;
                    }
                    let pool: Vec<_> = indices.iter().map(|&i| frame.poses[i]).collect();
                    let mut r = json!({"status":"evaluated"});
                    match member_summary(&proposal, &m, &pool) {
                        Ok(v) => r["members_only"] = v,
                        Err(e) => {
                            failures += 1;
                            r["members_only_error"] = json!(format!("{e:#}"));
                        }
                    }
                    if !args.no_fusion {
                        match fusion_summary(
                            &proposal,
                            &tree,
                            &wall,
                            &m,
                            &spectators,
                            &pool,
                            &oligomer,
                        ) {
                            Ok(v) => r["fused"] = v,
                            Err(e) => {
                                failures += 1;
                                r["fused_error"] = json!(format!("{e:#}"));
                            }
                        }
                    }
                    row[label] = r;
                }
                write_line(&mut contexts, &row)?;
                contexts.flush()?;
                context_count += 1;
            }
        }
        frame_rows.push(json!({"sweep":frame.sweep,"bodies":n,"exclusion_edges":edges,"contexts":allocation,"cpu_seconds":cpu_seconds()-frame_cpu}));
        eprintln!(
            "source coverage sweep {} complete; {} contexts, cpu {:.2}s",
            frame.sweep,
            context_count,
            cpu_seconds() - cpu_start
        );
    }
    singles.flush()?;
    contexts.flush()?;
    save(
        &args.out.join("summary.json"),
        &json!({"schema":"source-coverage-v1","completed":true,
        "singleton_records":singleton_count,"context_records":context_count,"diagnostic_errors":failures,"frames":frame_rows,
        "cpu_seconds":cpu_seconds()-cpu_start,"wall_seconds":start.elapsed().as_secs_f64(),
        "singletons_sha256":hash_file(&args.out.join("singletons.jsonl"))?,"contexts_sha256":hash_file(&args.out.join("contexts.jsonl"))?,
        "warning":"Norm and source density are proposal diagnostics, not physical basin masses. Full-contact anchor pools depend on moving poses and are an observer only; not a validated production proposal."}),
    )?;
    ensure!(
        failures == 0,
        "diagnostic completed and drained all records, but {failures} contexts/pairs failed; inspect errors"
    );
    Ok(())
}
fn main() -> Result<()> {
    run(&Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetramer_mc::{
        geometry::Atom,
        math::{IDENTITY, cayley, matmul, quaternion, rotation},
        proposal::GaussianComponentParameters,
    };
    const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    fn pose(x: f64) -> Pose {
        Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        }
    }
    fn proposal() -> DockingProposal {
        let parameters = vec![GaussianComponentParameters {
            anchor_position: [2.2, 0., 0.],
            anchor_rotation: IDENTITY,
            mean: [0.; 6],
            covariance: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { 0.01 } else { 0. })
            }),
            weight: 1.,
        }];
        DockingProposal::new(
            FrozenRelativePoseProposal::from_components_open(
                parameters, 1., [200.; 3], 0.1, SHA, SHA,
            )
            .unwrap(),
            DockingMethod::PosteriorInvolution,
            0.9,
            [0.; 3],
        )
        .unwrap()
    }
    #[test]
    fn represented_contact_bad_anchor_and_absent_basin_are_distinct() {
        let p = proposal();
        let tree = SphereTree::new(Shape {
            name: "exclusion sphere".into(),
            volume: 0.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 1.2,
            }],
        })
        .unwrap();
        let poses = vec![pose(2.2), pose(0.), pose(40.), pose(4.4)];
        let g = ContactGraph::build(&tree, &poses);
        assert_eq!(neighbors(&g, &[0]), vec![1, 3]);
        let good = pair_summary(&p, poses[0], poses[1]).unwrap();
        let far = pair_summary(&p, poses[0], poses[2]).unwrap();
        let absent = pair_summary(&p, poses[0], poses[3]).unwrap();
        assert!(good["minimum_chart_norm"].as_f64().unwrap() < 1e-10);
        assert!(far["minimum_chart_norm"].as_f64().unwrap() > 300.);
        assert!(absent["minimum_chart_norm"].as_f64().unwrap() > 40.);
        assert!(
            good["log_learned_density"].as_f64().unwrap()
                > absent["log_learned_density"].as_f64().unwrap()
        );
    }
    #[test]
    fn rigid_coordinate_change_preserves_source_diagnostics() {
        let p = proposal();
        let a = pose(4.);
        let mut x = pose(6.25);
        x.orientation = quaternion(cayley([0.1, 0.2, -0.1]));
        let global = Pose {
            position: [20., -10., 3.],
            orientation: quaternion(cayley([0.7, -0.2, 0.8])),
        };
        let transform = |p: Pose| Pose {
            position: global.apply(p.position),
            orientation: quaternion(matmul(
                rotation(global.orientation),
                rotation(p.orientation),
            )),
        };
        let before = pair_summary(&p, x, a).unwrap();
        let after = pair_summary(&p, transform(x), transform(a)).unwrap();
        for key in [
            "log_learned_density",
            "minimum_chart_norm",
            "posterior_mean_norm",
        ] {
            assert!(
                (before[key].as_f64().unwrap() - after[key].as_f64().unwrap()).abs() < 1e-9,
                "{key}"
            );
        }
    }
    #[test]
    fn covariance_eigenvalues_and_density_reconstruction() {
        let mut c = std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
        c[0][1] = 0.5;
        c[1][0] = 0.5;
        let d = covariance_summary(c);
        assert!((d["covariance_eigenvalues"][0].as_f64().unwrap() - 0.5).abs() < 1e-10);
        assert!((d["covariance_eigenvalues"][5].as_f64().unwrap() - 1.5).abs() < 1e-10);
        let p = proposal();
        let a = vec![pose(0.), pose(4.4)];
        let m = vec![pose(2.2)];
        let summary = member_summary(&p, &m, &a).unwrap();
        assert!(
            (summary["log_learned_density"].as_f64().unwrap()
                - p.members_log_density(&m, &a).unwrap())
            .abs()
                < 1e-10
        );
    }
    #[test]
    fn empty_contacts_and_cayley_seam_remain_explicit() {
        let p = proposal();
        let seam = Pose {
            position: [2.2, 0., 0.],
            orientation: [0., 1., 0., 0.],
        };
        let d = pair_summary(&p, seam, pose(0.)).unwrap();
        assert_eq!(d["encoding_failures"], 1);
        assert_eq!(d["density_status"], "no_finite_learned_density");
        assert!(d["log_learned_density"].is_null());
        assert!(member_summary(&p, &[pose(2.2)], &[]).is_err());
    }
}
