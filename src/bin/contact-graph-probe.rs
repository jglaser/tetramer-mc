//! Exact atomic-union exclusion-contact graphs of a frozen spherical trajectory.
//! Does not evolve configurations or infer native registry. Positive-volume
//! intersections only: touching exclusion surfaces are not graph edges.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{collections::VecDeque, fs, io::Write, path::PathBuf};
use tetramer_mc::{
    geometry::{Placed, Shape, SphereTree},
    math::Pose,
    simulation::{Boundary, Config, cpu_seconds, hash_bytes},
};

#[derive(Parser, Serialize)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    trajectory: PathBuf,
    #[arg(long)]
    out: PathBuf,
}
#[derive(Deserialize)]
struct Frame {
    sweep: u64,
    poses: Vec<Pose>,
    seed_labels: Vec<usize>,
    boundary: String,
}
#[derive(Serialize)]
struct Graph {
    sweep: u64,
    bodies: usize,
    edges: Vec<[usize; 2]>,
    degrees: Vec<usize>,
    components: Vec<Vec<usize>>,
    isolated_bodies: Vec<usize>,
    seed_contact_component: Vec<usize>,
    seed_labels: Vec<usize>,
}
fn graph(tree: &SphereTree, poses: &[Pose], sweep: u64, seeds: &[usize]) -> Graph {
    let placed: Vec<_> = poses.iter().copied().map(Placed::new).collect();
    let mut edges = vec![];
    let mut adjacent = vec![vec![]; poses.len()];
    for i in 0..poses.len() {
        for j in i + 1..poses.len() {
            if tree.overlaps(&placed[i], &placed[j]) {
                edges.push([i, j]);
                adjacent[i].push(j);
                adjacent[j].push(i);
            }
        }
    }
    let degrees: Vec<_> = adjacent.iter().map(Vec::len).collect();
    let isolated_bodies = degrees
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| (d == 0).then_some(i))
        .collect();
    let mut visited = vec![false; poses.len()];
    let mut components = vec![];
    for i in 0..poses.len() {
        if !visited[i] {
            visited[i] = true;
            let mut queue = VecDeque::from([i]);
            let mut component = vec![];
            while let Some(k) = queue.pop_front() {
                component.push(k);
                for &j in &adjacent[k] {
                    if !visited[j] {
                        visited[j] = true;
                        queue.push_back(j);
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
    }
    // Union of every component containing an original seed label. This does not
    // assume the seed remains connected and does not assert native order.
    let mut seed_contact_component: Vec<_> = components
        .iter()
        .filter(|c| seeds.iter().any(|i| c.binary_search(i).is_ok()))
        .flatten()
        .copied()
        .collect();
    seed_contact_component.sort_unstable();
    Graph {
        sweep,
        bodies: poses.len(),
        edges,
        degrees,
        components,
        isolated_bodies,
        seed_contact_component,
        seed_labels: seeds.to_vec(),
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "output already exists");
    let cfg_raw = fs::read(&args.config)?;
    let cfg: Config = serde_json::from_slice(&cfg_raw)?;
    ensure!(
        matches!(cfg.boundary, Boundary::Spherical { .. }),
        "only spherical input supported"
    );
    ensure!(
        cfg.depletant_radius.is_finite() && cfg.depletant_radius >= 0.,
        "invalid depletant radius"
    );
    let shape_raw = fs::read(&cfg.shape)?;
    let mut shape: Shape = serde_json::from_slice(&shape_raw)?;
    for atom in &mut shape.atoms {
        atom.radius += cfg.depletant_radius;
    }
    let tree = SphereTree::new(shape)?;
    let trajectory_raw = fs::read(&args.trajectory)?;
    let trajectory = std::str::from_utf8(&trajectory_raw)?;
    let expected_frames = trajectory.lines().filter(|s| !s.trim().is_empty()).count();
    fs::create_dir_all(&args.out)?;
    fs::write(
        args.out.join("provenance.json"),
        serde_json::to_vec_pretty(&json!({
            "args": args, "config":cfg, "config_sha256":hash_bytes(&cfg_raw),
            "shape_sha256":hash_bytes(&shape_raw), "trajectory_sha256":hash_bytes(&trajectory_raw),
            "source_sha256":hash_bytes(include_bytes!("contact-graph-probe.rs")),
            "geometry_source_sha256":hash_bytes(include_bytes!("../geometry.rs")),
            "executable_sha256":hash_bytes(&fs::read(std::env::current_exe()?)?),
            "expected_frames":expected_frames,
            "predicate":"strict interbody atomic-sphere overlap after every atomic radius is increased by depletant_radius; no bounding sphere edges; no periodic images",
            "hard_or_wall_validity_checked":false,
            "interpretation":"exclusion-contact connectivity, not native registry or thermodynamic weighting"
        }))?,
    )?;
    let mut out = fs::File::create(args.out.join("graphs.jsonl"))?;
    let start = cpu_seconds();
    let mut frames = 0;
    let mut last_sweep = None;
    for (line_number, line) in trajectory.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let f: Frame = serde_json::from_str(line)
            .with_context(|| format!("frame line {}", line_number + 1))?;
        ensure!(f.boundary == "spherical", "frame boundary mismatch");
        ensure!(
            f.poses.len() == cfg.initial_poses.len(),
            "body count mismatch"
        );
        ensure!(f.seed_labels == cfg.seed_labels, "seed labels mismatch");
        ensure!(
            f.seed_labels.iter().all(|&i| i < f.poses.len()),
            "invalid seed label"
        );
        ensure!(
            last_sweep.is_none_or(|s| f.sweep > s),
            "nonincreasing sweep"
        );
        for p in &f.poses {
            p.validate()?;
        }
        let g = graph(&tree, &f.poses, f.sweep, &f.seed_labels);
        serde_json::to_writer(&mut out, &g)?;
        out.write_all(b"\n")?;
        out.flush()?;
        frames += 1;
        last_sweep = Some(f.sweep);
        if frames % 50 == 0 {
            eprintln!(
                "completed {frames}/{expected_frames} frames, cpu {:.3}s",
                cpu_seconds() - start
            );
        }
    }
    ensure!(frames == expected_frames, "frame denominator mismatch");
    fs::write(
        args.out.join("summary.json"),
        serde_json::to_vec_pretty(&json!({
            "complete":true,"frames":frames,"expected_frames":expected_frames,
            "last_sweep":last_sweep,"cpu_seconds":cpu_seconds()-start
        }))?,
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use tetramer_mc::geometry::Atom;
    fn sphere(rd: f64) -> SphereTree {
        SphereTree::new(Shape {
            name: "sphere".into(),
            volume: 0.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 1. + rd,
            }],
        })
        .unwrap()
    }
    fn pose(x: f64) -> Pose {
        Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        }
    }
    #[test]
    fn contact_threshold_and_no_self_edges() {
        let g = graph(&sphere(0.5), &[pose(0.), pose(2.5), pose(5.5)], 0, &[0]);
        assert_eq!(g.edges, vec![[0, 1]]); // second pair touches at exactly 3 A
        assert_eq!(g.degrees, vec![1, 1, 0]);
        assert_eq!(g.isolated_bodies, vec![2]);
        assert_eq!(g.components, vec![vec![0, 1], vec![2]]);
        assert_eq!(g.seed_contact_component, vec![0, 1]);
    }
    #[test]
    fn zero_radius_is_core_overlap_and_disconnected_seeds_are_unioned() {
        let g = graph(
            &sphere(0.),
            &[pose(0.), pose(1.5), pose(4.), pose(5.5)],
            0,
            &[0, 2],
        );
        assert_eq!(g.edges, vec![[0, 1], [2, 3]]);
        assert_eq!(g.seed_contact_component, vec![0, 1, 2, 3]);
        assert!(
            graph(&sphere(0.), &[pose(0.), pose(2.)], 0, &[])
                .edges
                .is_empty()
        );
    }
}
