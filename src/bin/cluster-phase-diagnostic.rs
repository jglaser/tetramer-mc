//! Passive replay of a frozen cluster-phase run. No RNG and no trajectory edits.
//! Topology uses the production atomic-union exclusion predicate, independently
//! of native registry. Every attempted event, including rejections, is retained.
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    cluster_phase::{ClusterPhaseConfig, ContactGraph},
    geometry::{Shape, SphereTree},
    math::{Pose, add, sub},
    simulation::{Boundary, Config, cpu_seconds, hash_bytes},
    spherical::HalfTurn,
};

#[derive(Parser, Serialize)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    trajectory: PathBuf,
    #[arg(long)]
    moves: PathBuf,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Clone, Deserialize)]
struct Frame {
    sweep: u64,
    poses: Vec<Pose>,
    seed_labels: Vec<usize>,
    coordinate_wall_center: [f64; 3],
    counts: Value,
}
#[derive(Default, Serialize)]
struct Counts {
    attempted: u64,
    proposal_nulls: u64,
    hard_valid: u64,
    hard_rejected: u64,
    internal_geometry_nulls: u64,
    physical_accepted: u64,
    accepted: u64,
    proposed_attachments: u64,
    proposed_detachments: u64,
    proposed_exchanges: u64,
    accepted_attachments: u64,
    accepted_detachments: u64,
    accepted_exchanges: u64,
    accepted_contact_preserving: u64,
    proposed_gained_contacts: u64,
    proposed_lost_contacts: u64,
    accepted_gained_contacts: u64,
    accepted_lost_contacts: u64,
}
impl Counts {
    fn record(&mut self, row: &Value, gained: usize, lost: usize) {
        self.attempted += 1;
        let accepted = row["accepted"].as_bool().unwrap();
        let hard = row["hard_valid"].as_bool().unwrap();
        let null = row["proposed_poses"].is_null();
        self.proposal_nulls += u64::from(null);
        self.hard_valid += u64::from(hard);
        self.hard_rejected += u64::from(!null && !hard);
        self.internal_geometry_nulls +=
            u64::from(hard && row["internal_contact_graph_preserved"] == false);
        self.physical_accepted += u64::from(row["physical_accepted"] == true);
        self.accepted += u64::from(accepted);
        self.proposed_attachments += u64::from(gained > 0 && lost == 0);
        self.proposed_detachments += u64::from(gained == 0 && lost > 0);
        self.proposed_exchanges += u64::from(gained > 0 && lost > 0);
        self.proposed_gained_contacts += gained as u64;
        self.proposed_lost_contacts += lost as u64;
        if accepted {
            self.accepted_attachments += u64::from(gained > 0 && lost == 0);
            self.accepted_detachments += u64::from(gained == 0 && lost > 0);
            self.accepted_exchanges += u64::from(gained > 0 && lost > 0);
            self.accepted_contact_preserving += u64::from(gained == 0 && lost == 0);
            self.accepted_gained_contacts += gained as u64;
            self.accepted_lost_contacts += lost as u64;
        }
    }
}
fn components(graph: &ContactGraph) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = graph.adjacency.len();
    let mut components = vec![];
    let mut labels = vec![usize::MAX; n];
    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        let mut members = vec![];
        labels[start] = components.len();
        while let Some(i) = stack.pop() {
            members.push(i);
            for (j, &edge) in graph.adjacency[i].iter().enumerate() {
                if edge && labels[j] == usize::MAX {
                    labels[j] = components.len();
                    stack.push(j);
                }
            }
        }
        members.sort_unstable();
        components.push(members);
    }
    (components, labels)
}
fn subset_info(
    graph: &ContactGraph,
    members: &[usize],
    seeds: &[usize],
    cs: &[Vec<usize>],
    labels: &[usize],
) -> Result<Value> {
    ensure!(!members.is_empty(), "empty subset");
    let parent = labels[members[0]];
    ensure!(
        members.iter().all(|&i| labels[i] == parent),
        "selected subset not connected to same parent"
    );
    let selected_set: BTreeSet<_> = members.iter().copied().collect();
    ensure!(
        selected_set.len() == members.len(),
        "duplicate selected label"
    );
    let boundary = graph.boundary_edges(members);
    let parent_seed = cs[parent].iter().filter(|i| seeds.contains(i)).count();
    let selected_seed = members.iter().filter(|i| seeds.contains(i)).count();
    let internal_edges = members
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            members[k + 1..]
                .iter()
                .filter(|&&j| graph.adjacency[i][j])
                .count()
        })
        .sum::<usize>();
    // Edge count alone is insufficient once larger subsets are considered.
    let mut reached = BTreeSet::from([members[0]]);
    let mut stack = vec![members[0]];
    while let Some(i) = stack.pop() {
        for &j in members {
            if graph.adjacency[i][j] && reached.insert(j) {
                stack.push(j);
            }
        }
    }
    ensure!(
        reached.len() == members.len(),
        "selected subset not internally connected"
    );
    Ok(json!({
        "size": members.len(), "members":members, "parent_members":cs[parent],
        "parent_component_size":cs[parent].len(), "whole_component":cs[parent].len()==members.len(),
        "topology":if boundary.is_empty(){"whole"}else{"embedded"},
        "parent_seed_connected":parent_seed>0, "parent_original_seed_count":parent_seed,
        "selected_original_seed_count":selected_seed,
        "selected_seed_class":if selected_seed==0{"no_original_seed"}else if selected_seed==members.len(){"original_seed_only"}else{"mixed_original_seed_and_solution"},
        "boundary_edges":boundary,"boundary_contact_count":boundary.len(),
        "external_neighbors":boundary.iter().flat_map(|e| e.iter().copied()).filter(|i| !selected_set.contains(i)).collect::<BTreeSet<_>>(),
        "internal_edges":internal_edges
    }))
}
fn validate_context(info: &Value, context: &Value) -> Result<()> {
    for (ours, theirs) in [
        ("size", "subset_size"),
        ("parent_members", "parent_component_members"),
        ("whole_component", "whole_component"),
        ("internal_edges", "internal_contacts"),
        ("boundary_contact_count", "external_contacts"),
    ] {
        ensure!(
            info[ours] == context[theirs],
            "logged subset context mismatch for {theirs}"
        );
    }
    ensure!(
        info["external_neighbors"]
            .as_array()
            .context("invalid neighbor array")?
            .len() as u64
            == context["external_neighbors"]
                .as_u64()
                .context("invalid neighbor count")?,
        "logged subset context mismatch for external_neighbors"
    );
    Ok(())
}
fn census(
    graph: &ContactGraph,
    cfg: &ClusterPhaseConfig,
    seeds: &[usize],
    sweep: u64,
) -> Result<Value> {
    let channels = graph.channels(cfg);
    let (cs, labels) = components(graph);
    let mut groups: BTreeMap<String, Value> = BTreeMap::new();
    for ch in &channels {
        let info = subset_info(graph, &ch.members, seeds, &cs, &labels)?;
        let key = format!(
            "{}|{}|{}",
            ch.members.len(),
            info["topology"].as_str().unwrap(),
            if info["parent_seed_connected"] == true {
                "seed_connected"
            } else {
                "solution"
            }
        );
        let g = groups
            .entry(key)
            .or_insert_with(|| json!({"channels":0,"rate":0.}));
        g["channels"] = json!(g["channels"].as_u64().unwrap() + 1);
        g["rate"] = json!(g["rate"].as_f64().unwrap() + ch.rate);
    }
    let mut sizes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut solution_sizes: BTreeMap<usize, usize> = BTreeMap::new();
    let mut seed_components = vec![];
    for c in &cs {
        *sizes.entry(c.len()).or_default() += 1;
        if c.iter().any(|i| seeds.contains(i)) {
            seed_components.push(c.clone());
        } else {
            *solution_sizes.entry(c.len()).or_default() += 1;
        }
    }
    Ok(
        json!({"sweep":sweep,"eligible_channels":channels.len(),"total_rate":channels.iter().map(|c|c.rate).sum::<f64>(),
        "groups":groups,"component_size_histogram":sizes,"solution_component_size_histogram":solution_sizes,
        "seed_components":seed_components,"components":cs}),
    )
}
fn write_row(out: &mut BufWriter<fs::File>, row: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, row)?;
    out.write_all(b"\n")?;
    Ok(())
}
fn validate_frame(
    frame: &Frame,
    poses: &[Pose],
    center: [f64; 3],
    counts: &Counts,
    phases: u64,
) -> Result<()> {
    ensure!(
        frame.poses == poses,
        "replayed poses differ from snapshot at sweep {}",
        frame.sweep
    );
    ensure!(
        frame.coordinate_wall_center == center,
        "replayed coordinate wall center differs at {}",
        frame.sweep
    );
    let c = &frame.counts["cluster_phase"];
    for (field, v) in [
        ("events", counts.attempted),
        ("accepted", counts.accepted),
        ("hard_valid", counts.hard_valid),
        ("phases", phases),
        ("accepted_attachments", counts.accepted_attachments),
        ("accepted_detachments", counts.accepted_detachments),
        ("completed_exchanges", counts.accepted_exchanges),
        ("gained_contacts", counts.accepted_gained_contacts),
        ("lost_contacts", counts.accepted_lost_contacts),
    ] {
        ensure!(
            c[field].as_u64() == Some(v),
            "cluster counter {field} differs at sweep {}: {} vs {v}",
            frame.sweep,
            c[field]
        );
    }
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "output already exists");
    let cfg_raw = fs::read(&args.config)?;
    let cfg: Config = serde_json::from_slice(&cfg_raw)?;
    ensure!(
        matches!(cfg.boundary, Boundary::Spherical { .. }),
        "requires spherical run"
    );
    let phase = cfg
        .cluster_phase
        .as_ref()
        .context("missing cluster phase configuration")?;
    let shape_raw = fs::read(&cfg.shape)?;
    let mut shape: Shape = serde_json::from_slice(&shape_raw)?;
    for atom in &mut shape.atoms {
        atom.radius += cfg.depletant_radius;
    }
    let tree = SphereTree::new(shape)?;
    let trajectory_raw = fs::read(&args.trajectory)?;
    let mut frames = BTreeMap::new();
    for line in std::str::from_utf8(&trajectory_raw)?.lines() {
        let f: Frame = serde_json::from_str(line)?;
        ensure!(f.seed_labels == cfg.seed_labels, "seed labels differ");
        ensure!(frames.insert(f.sweep, f).is_none(), "duplicate frame");
    }
    let first = frames
        .get(&0)
        .context("frozen trajectory needs initial frame")?;
    let last_sweep = *frames.last_key_value().unwrap().0;
    let mut poses = first.poses.clone();
    let seeds = first.seed_labels.clone();
    let mut center = first.coordinate_wall_center;
    ensure!(
        poses == cfg.initial_poses,
        "initial poses differ from config"
    );
    fs::create_dir_all(&args.out)?;
    let mut events = BufWriter::new(fs::File::create(args.out.join("events.jsonl"))?);
    let mut censuses = BufWriter::new(fs::File::create(args.out.join("phase-census.jsonl"))?);
    let mut checks = BufWriter::new(fs::File::create(args.out.join("replay-checks.jsonl"))?);
    let mut overall = Counts::default();
    let mut groups: BTreeMap<String, Counts> = BTreeMap::new();
    let mut counts_by_kind: BTreeMap<String, u64> = BTreeMap::new();
    let mut old_pose_checks = 0u64;
    let mut logged_context_checks = 0u64;
    let mut snapshot_checks = 1u64;
    let mut phase_checks = 0u64;
    let mut last_census = Value::Null;
    let mut whole_attachment_proposals = Vec::new();
    let mut active_sweep = 0u64;
    let mut graph: Option<ContactGraph> = None;
    let mut phase_ended = false;
    let mut line_count = 0u64;
    let mut sha = Sha256::new();
    let mut input = BufReader::new(fs::File::open(&args.moves)?);
    let mut line = String::new();
    let start = cpu_seconds();
    loop {
        line.clear();
        if input.read_line(&mut line)? == 0 {
            break;
        }
        ensure!(line.ends_with('\n'), "incomplete frozen move line");
        sha.update(line.as_bytes());
        line_count += 1;
        let row: Value =
            serde_json::from_str(&line).with_context(|| format!("move line {line_count}"))?;
        let sweep = row["sweep"].as_u64().context("missing sweep")?;
        ensure!(
            sweep > 0 && sweep <= last_sweep,
            "move outside frozen trajectory"
        );
        if sweep != active_sweep {
            ensure!(sweep == active_sweep + 1, "missing/unsorted sweep");
            if active_sweep > 0 {
                ensure!(phase_ended, "missing phase end for {active_sweep}");
                if let Some(f) = frames.get(&active_sweep) {
                    validate_frame(f, &poses, center, &overall, phase_checks)?;
                    snapshot_checks += 1;
                    write_row(
                        &mut checks,
                        &json!({"sweep":active_sweep,"bit_exact_poses":true,"bit_exact_wall_center":true,"cluster_counters_match":true}),
                    )?;
                }
            }
            active_sweep = sweep;
            graph = None;
            phase_ended = false;
        }
        let kind = row["kind"].as_str().context("missing kind")?;
        *counts_by_kind.entry(kind.into()).or_default() += 1;
        match kind {
            "local" | "global" => {
                ensure!(graph.is_none(), "single move after cluster phase began");
                let i = row["moving_index"]
                    .as_u64()
                    .context("missing moving index")? as usize;
                let old: Pose = serde_json::from_value(row["old_pose"].clone())?;
                ensure!(
                    poses[i] == old,
                    "old pose differs at move {line_count}, sweep {sweep}"
                );
                old_pose_checks += 1;
                let retained: Pose = serde_json::from_value(row["retained_pose"].clone())?;
                if row["accepted"] == true {
                    let proposed: Pose = serde_json::from_value(row["proposed_pose"].clone())?;
                    ensure!(retained == proposed, "accepted move differs from proposal");
                } else {
                    ensure!(retained == old, "rejection changed pose");
                }
                poses[i] = retained;
            }
            "gca" => {
                ensure!(graph.is_none(), "GCA after phase began");
                if row["accepted"] == true {
                    let h = HalfTurn::new(serde_json::from_value(row["axis"].clone())?)?;
                    let ids: Vec<usize> =
                        serde_json::from_value(row["result"]["flipped_indices"].clone())?;
                    for i in ids {
                        poses[i] = h.apply(poses[i]);
                    }
                }
            }
            "center_shift" => {
                ensure!(graph.is_none(), "shift after phase began");
                if row["accepted"] == true {
                    let displacement: [f64; 3] =
                        serde_json::from_value(row["result"]["displacement"].clone())?;
                    for p in &mut poses {
                        p.position = add(p.position, displacement);
                    }
                    center = sub(center, displacement);
                }
            }
            "cluster_event" | "cluster_phase_end" => {
                ensure!(!phase_ended, "cluster event after phase end");
                if graph.is_none() {
                    let g = ContactGraph::build(&tree, &poses);
                    last_census = census(&g, phase, &seeds, sweep)?;
                    write_row(&mut censuses, &last_census)?;
                    graph = Some(g);
                }
                let g = graph.as_ref().unwrap();
                let channels = g.channels(phase);
                let total_rate: f64 = channels.iter().map(|c| c.rate).sum();
                ensure!(
                    row["total_rate"].as_f64() == Some(total_rate),
                    "rate differs at {sweep}"
                );
                if kind == "cluster_phase_end" {
                    ensure!(
                        row["eligible_channels"].as_u64() == Some(channels.len() as u64),
                        "channel count differs"
                    );
                    phase_checks += 1;
                    phase_ended = true;
                } else {
                    let members: Vec<usize> = serde_json::from_value(row["members"].clone())?;
                    ensure!(
                        channels.iter().any(|c| c.members == members),
                        "ineligible selected channel"
                    );
                    let old: Vec<Pose> = serde_json::from_value(row["old_poses"].clone())?;
                    ensure!(
                        old == members.iter().map(|&i| poses[i]).collect::<Vec<_>>(),
                        "cluster old pose mismatch at {sweep}"
                    );
                    old_pose_checks += old.len() as u64;
                    let (cs, labels) = components(g);
                    let mut event = subset_info(g, &members, &seeds, &cs, &labels)?;
                    if !row["subset_context"].is_null() {
                        validate_context(&event, &row["subset_context"])?;
                        logged_context_checks += 1;
                    }
                    let retained: Vec<Pose> =
                        serde_json::from_value(row["retained_poses"].clone())?;
                    let gained: Vec<[usize; 2]> =
                        serde_json::from_value(row["proposed_gained_contacts"].clone())?;
                    let lost: Vec<[usize; 2]> =
                        serde_json::from_value(row["proposed_lost_contacts"].clone())?;
                    let mut next_graph = None;
                    if row["hard_valid"] == true {
                        let proposed: Vec<Pose> =
                            serde_json::from_value(row["proposed_poses"].clone())?;
                        let mut next = poses.clone();
                        for (&i, &p) in members.iter().zip(&proposed) {
                            next[i] = p;
                        }
                        let ng = g.updated(&tree, &next, &members);
                        let same = g.internal_equal(&ng, &members);
                        ensure!(
                            row["internal_contact_graph_preserved"].as_bool() == Some(same),
                            "internal graph mismatch"
                        );
                        if same {
                            let old_edges = g.boundary_edges(&members);
                            let new_edges = ng.boundary_edges(&members);
                            ensure!(
                                gained
                                    == new_edges
                                        .difference(&old_edges)
                                        .copied()
                                        .collect::<Vec<_>>(),
                                "gained edges mismatch"
                            );
                            ensure!(
                                lost == old_edges
                                    .difference(&new_edges)
                                    .copied()
                                    .collect::<Vec<_>>(),
                                "lost edges mismatch"
                            );
                        }
                        next_graph = Some(ng);
                    }
                    let branch = if row["proposal"]["branch"] == "local_rigid" {
                        "local"
                    } else {
                        "transport"
                    };
                    for field in [
                        "event",
                        "event_time",
                        "waiting_time",
                        "total_rate",
                        "selected_rate",
                        "handle",
                        "hard_valid",
                        "physical_accepted",
                        "accepted",
                        "log_acceptance",
                        "internal_contact_graph_preserved",
                        "sampler_cpu_seconds",
                    ] {
                        event[field] = row[field].clone();
                    }
                    event["sweep"] = json!(sweep);
                    event["source_line"] = json!(line_count);
                    event["branch"] = json!(branch);
                    event["proposal_branch"] = row["proposal"]["branch"].clone();
                    event["proposal_null"] = json!(row["proposed_poses"].is_null());
                    event["gained_contacts"] = json!(gained);
                    event["lost_contacts"] = json!(lost);
                    event["map_log_reverse_forward"] =
                        row["proposal"]["log_reverse_forward"].clone();
                    event["gate"] = row["gate"].clone();
                    overall.record(&row, gained.len(), lost.len());
                    let key = format!(
                        "{}|{}|{}|{}",
                        members.len(),
                        event["topology"].as_str().unwrap(),
                        if event["parent_seed_connected"] == true {
                            "seed_connected"
                        } else {
                            "solution"
                        },
                        branch
                    );
                    groups
                        .entry(key)
                        .or_default()
                        .record(&row, gained.len(), lost.len());
                    if event["whole_component"] == true && !gained.is_empty() && lost.is_empty() {
                        whole_attachment_proposals.push(event.clone());
                    }
                    write_row(&mut events, &event)?;
                    if row["accepted"] == true {
                        let proposed: Vec<Pose> =
                            serde_json::from_value(row["proposed_poses"].clone())?;
                        ensure!(
                            retained == proposed,
                            "accepted cluster differs from proposed"
                        );
                        graph = Some(next_graph.context("accepted without evaluated graph")?);
                    } else {
                        ensure!(retained == old, "rejected cluster changed poses");
                    }
                    for (&i, &p) in members.iter().zip(&retained) {
                        poses[i] = p;
                    }
                }
            }
            _ => bail!(
                "unsupported move kind {kind}; replay will not silently ignore physical changes"
            ),
        }
        if phase_ended && sweep % 100 == 0 {
            eprintln!(
                "replayed sweep {sweep}/{last_sweep}, {} cluster attempts, CPU {:.2}s",
                overall.attempted,
                cpu_seconds() - start
            );
        }
    }
    ensure!(
        active_sweep == last_sweep && phase_ended,
        "frozen moves do not end at final saved frame"
    );
    validate_frame(
        frames.get(&last_sweep).unwrap(),
        &poses,
        center,
        &overall,
        phase_checks,
    )?;
    snapshot_checks += 1;
    write_row(
        &mut checks,
        &json!({"sweep":last_sweep,"bit_exact_poses":true,"bit_exact_wall_center":true,"cluster_counters_match":true}),
    )?;
    ensure!(
        snapshot_checks as usize == frames.len(),
        "not every snapshot validated"
    );
    events.flush()?;
    censuses.flush()?;
    checks.flush()?;
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_vec_pretty(&json!({
            "complete":true,"last_sweep":last_sweep,"bodies":poses.len(),"seed_labels":seeds,"move_lines":line_count,
            "move_counts":counts_by_kind,"overall":overall,"groups":groups,
        "last_phase_start_census":last_census,
        "whole_component_attachment_proposals":whole_attachment_proposals,
            "validation":{"snapshot_checks":snapshot_checks,"old_pose_checks":old_pose_checks,"phase_rate_checks":phase_checks,
                "bit_exact_replay":true,"all_cluster_attempts_preserved":true,"logged_context_checks":logged_context_checks,"exact_boundary_edge_changes_checked":true},
            "cpu_seconds":cpu_seconds()-start,
            "interpretation":"Whole means no exclusion contacts to particles outside S. Seed-connected means any physical-contact path to an original seed label, not native registry. Attempts, including rejection/null events, are denominators. Proposed contact changes are evaluated only for hard-valid internally preserved candidates."
        }))?,
    )?;
    fs::write(
        args.out.join("provenance.json"),
        serde_json::to_vec_pretty(&json!({
            "args":args,"config_sha256":hash_bytes(&cfg_raw),"shape_sha256":hash_bytes(&shape_raw),
            "trajectory_sha256":hash_bytes(&trajectory_raw),"moves_sha256":format!("{:x}",sha.finalize()),
            "source_sha256":hash_bytes(include_bytes!("cluster-phase-diagnostic.rs")),
            "geometry_source_sha256":hash_bytes(include_bytes!("../geometry.rs")),
            "cluster_source_sha256":hash_bytes(include_bytes!("../cluster_phase.rs")),
            "executable_sha256":hash_bytes(&fs::read(std::env::current_exe()?)?),
            "depletant_radius_A":cfg.depletant_radius,"depletant_activity_A_minus3":cfg.reservoir_density,
            "definition":"Production SphereTree::overlaps of atomic unions with radius inflated by rd; strict overlap, no centroid-contact proxy.",
            "no_simulation_evolution":true
        }))?,
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn g(n: usize, edges: &[[usize; 2]]) -> ContactGraph {
        let mut g = ContactGraph {
            adjacency: vec![vec![false; n]; n],
        };
        for &[i, j] in edges {
            g.adjacency[i][j] = true;
            g.adjacency[j][i] = true;
        }
        g
    }
    #[test]
    fn whole_dimer_and_embedded_trimer() {
        let graph = g(7, &[[0, 1], [1, 2], [2, 3], [4, 5]]);
        let (cs, labels) = components(&graph);
        let d = subset_info(&graph, &[4, 5], &[0], &cs, &labels).unwrap();
        assert_eq!(d["whole_component"], true);
        assert_eq!(d["parent_seed_connected"], false);
        let t = subset_info(&graph, &[1, 2, 3], &[0], &cs, &labels).unwrap();
        assert_eq!(t["whole_component"], false);
        assert_eq!(t["parent_component_size"], 4);
        assert_eq!(t["boundary_contact_count"], 1);
        assert_eq!(t["parent_seed_connected"], true);
        assert_eq!(t["selected_original_seed_count"], 0);
    }
    #[test]
    fn triangle_counted_once_and_rejections_retained() {
        let graph = g(3, &[[0, 1], [1, 2], [0, 2]]);
        let c = census(&graph, &ClusterPhaseConfig::default(), &[], 1).unwrap();
        assert_eq!(c["eligible_channels"], 4);
        assert_eq!(c["total_rate"], 3.25);
        let mut counts = Counts::default();
        counts.record(&json!({"accepted":false,"hard_valid":true,"physical_accepted":false,"proposed_poses":[1],"internal_contact_graph_preserved":true}),1,1);
        assert_eq!(counts.attempted, 1);
        assert_eq!(counts.hard_valid, 1);
        assert_eq!(counts.proposed_exchanges, 1);
        assert_eq!(counts.accepted_exchanges, 0);
    }
    #[test]
    fn production_context_matches_independent_classifier() {
        let graph = g(7, &[[0, 1], [1, 2], [0, 2], [2, 3], [4, 5], [4, 6], [5, 6]]);
        let (cs, labels) = components(&graph);
        for ch in graph.channels(&ClusterPhaseConfig::default()) {
            let info = subset_info(&graph, &ch.members, &[0], &cs, &labels).unwrap();
            let context = serde_json::to_value(graph.subset_context(&ch.members).unwrap()).unwrap();
            validate_context(&info, &context).unwrap();
        }
    }
    #[test]
    fn disconnected_subset_rejected() {
        let graph = g(4, &[[0, 1], [1, 2], [2, 3]]);
        let (cs, l) = components(&graph);
        assert!(subset_info(&graph, &[0, 3], &[], &cs, &l).is_err());
        let graph = g(5, &[[0, 1], [1, 2], [0, 2], [2, 3], [3, 4]]);
        let (cs, l) = components(&graph);
        assert!(subset_info(&graph, &[0, 1, 2, 4], &[], &cs, &l).is_err());
    }
}
