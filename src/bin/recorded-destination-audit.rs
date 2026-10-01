//! Deterministic geometry replay of archived singleton control candidates.
//! No proposal, bath draw or physical update is sampled by this observer.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};
use tetramer_mc::{
    docking::{DockingMethod, DockingProposal},
    geometry::{Placed, Shape, SphereTree},
    math::*,
    oligomer_proposal::OligomerMixture,
    proposal::FrozenRelativePoseProposal,
    simulation::{Boundary, Config, cpu_seconds, hash_bytes, hash_file, save},
    spherical::Container,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    run: PathBuf,
    #[arg(long)]
    out: PathBuf,
}
fn angle(a: Pose, b: Pose) -> f64 {
    2. * a
        .orientation
        .iter()
        .zip(b.orientation)
        .map(|(x, y)| x * y)
        .sum::<f64>()
        .abs()
        .clamp(0., 1.)
        .acos()
        .to_degrees()
}
fn compose(a: Pose, b: Pose) -> Pose {
    Pose {
        position: add(a.position, matvec(rotation(a.orientation), b.position)),
        orientation: quaternion(matmul(rotation(a.orientation), rotation(b.orientation))),
    }
}
fn relative(a: Pose, b: Pose) -> Pose {
    let ri = transpose(rotation(a.orientation));
    Pose {
        position: matvec(ri, sub(b.position, a.position)),
        orientation: quaternion(matmul(ri, rotation(b.orientation))),
    }
}
fn inspect(
    p: Pose,
    moving: usize,
    state: &[Pose],
    tree: &SphereTree,
    exclusion: &SphereTree,
    wall: &Container,
) -> Value {
    let b = Placed::new(p);
    let clash: Vec<_> = state
        .iter()
        .enumerate()
        .filter(|(i, s)| *i != moving && tree.overlaps(&b, &Placed::new(**s)))
        .map(|(i, _)| i)
        .collect();
    let contact: Vec<_> = state
        .iter()
        .enumerate()
        .filter(|(i, s)| *i != moving && exclusion.overlaps(&b, &Placed::new(**s)))
        .map(|(i, _)| i)
        .collect();
    json!({"wall_valid":wall.contains(p),"hard_clash_bodies":clash,"contact_bodies":contact,"hard_valid":wall.contains(p)&&clash.is_empty()})
}
fn main() -> Result<()> {
    let args = Args::parse();
    let clock = cpu_seconds();
    let config: Config =
        serde_json::from_slice(&fs::read(args.run.join("effective-config.json"))?)?;
    let manifest: Value = serde_json::from_slice(&fs::read(args.run.join("manifest.json"))?)?;
    let shape_raw = fs::read(&config.shape)?;
    ensure!(
        hash_bytes(&shape_raw) == manifest["shape_sha256"],
        "shape mismatch"
    );
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?;
    let mut exclusion_shape = tree.shape.clone();
    for atom in &mut exclusion_shape.atoms {
        atom.radius += config.depletant_radius;
    }
    let exclusion = SphereTree::new(exclusion_shape)?;
    let radius = match config.boundary {
        Boundary::Spherical { radius } => radius,
        _ => anyhow::bail!("spherical only"),
    };
    let wall = Container::new(radius, &tree)?;
    let model_raw = fs::read(manifest["model_path"].as_str().context("model path")?)?;
    ensure!(
        hash_bytes(&model_raw) == manifest["model_sha256"],
        "model mismatch"
    );
    let model = FrozenRelativePoseProposal::from_json_str_open(
        std::str::from_utf8(&model_raw)?,
        [2. * (radius + tree.bound); 3],
        config.learned_uniform_weight,
        &hash_bytes(&shape_raw),
    )?;
    let settings = config.cluster_phase.as_ref().context("settings")?;
    let proposal = DockingProposal::new(
        model,
        DockingMethod::PosteriorInvolution,
        settings.correlation,
        [0.; 3],
    )?;
    let (map, inverted, _) = proposal.member_chart_parts();
    fs::create_dir(&args.out)?;
    let mut output = BufWriter::new(fs::File::create(args.out.join("audit.jsonl"))?);
    let input = args.run.join("events.jsonl");
    let input_sha = hash_file(&input)?;
    let mut state = config.initial_poses.clone();
    let mut phase = 0u64;
    let mut count = 0usize;
    for line in BufReader::new(fs::File::open(&input)?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        if row["kind"] != "cluster_event" {
            continue;
        }
        let p = row["phase"].as_u64().context("phase")?;
        if p != phase {
            state = config.initial_poses.clone();
            phase = p;
        }
        let members: Vec<usize> = serde_json::from_value(row["members"].clone())?;
        ensure!(members.len() == 1, "singleton only");
        let moving = members[0];
        let old: Vec<Pose> = serde_json::from_value(row["old_poses"].clone())?;
        ensure!(state[moving] == old[0], "old pose replay mismatch");
        let retained: Vec<Pose> = serde_json::from_value(row["retained_poses"].clone())?;
        let proposed: Option<Vec<Pose>> = serde_json::from_value(row["proposed_poses"].clone())?;
        let mut result = json!({"phase":phase,"event":row["event"],"population_seed":row["population_seed"],"moving":moving,"branch":row["proposal"]["branch"],"accepted":row["accepted"],"old":inspect(old[0],moving,&state,&tree,&exclusion,&wall),"diagnostic":row["diagnostic"],"gate":row["gate"],"log_acceptance":row["log_acceptance"],"log_proposal":row["proposal"]["log_reverse_forward"]});
        if let Some(candidate) = proposed {
            let candidate = candidate[0];
            let observation = inspect(candidate, moving, &state, &tree, &exclusion, &wall);
            ensure!(
                observation["hard_valid"] == row["hard_valid"],
                "hard predicate disagrees"
            );
            result["candidate"] = observation;
            result["translation_a"] = json!(norm(sub(old[0].position, candidate.position)));
            result["rotation_degrees"] = json!(angle(old[0], candidate));
        }
        if row["proposal"]["branch"] == "involution" {
            let source_z: [f64; 6] =
                serde_json::from_value(row["proposal"]["step"]["source_latent"].clone())?;
            result["source_norm"] = json!(source_z.iter().map(|x| x * x).sum::<f64>().sqrt());
            let pool: Vec<usize> = serde_json::from_value(row["proposal"]["anchor_pool"].clone())?;
            result["anchor_pool"] = json!(pool);
            let label = if row["proposal"]["oligomer"].is_object() {
                row["proposal"]["oligomer"]["target_label"].clone()
            } else {
                row["proposal"]["labels"]["target"].clone()
            };
            result["source_label"] = if row["proposal"]["oligomer"].is_object() {
                row["proposal"]["oligomer"]["source_label"].clone()
            } else {
                row["proposal"]["labels"]["source"].clone()
            };
            result["target_label"] = label.clone();
            result["fused_available"] = row["proposal"]["oligomer"]["fused_components"].clone();
            let target_z: [f64; 6] =
                serde_json::from_value(row["proposal"]["step"]["target_latent"].clone())?;
            let mut contracted = Vec::new();
            let mean = if label["kind"] == "fused" {
                let spectators: Vec<_> = state
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != moving)
                    .map(|(_, p)| *p)
                    .collect();
                let anchors: Vec<_> = pool.iter().map(|&i| state[i]).collect();
                let mix = OligomerMixture::build(
                    &proposal,
                    &tree,
                    &wall,
                    &old,
                    &spectators,
                    &anchors,
                    settings.oligomer.as_ref().context("oligomer")?,
                )?;
                let source = row["proposal"]["source"].as_u64().context("source")? as usize;
                let target = row["proposal"]["target"].as_u64().context("target")? as usize;
                ensure!(
                    mix.describe(target) == label,
                    "fused reconstruction mismatch"
                );
                let scale = -settings.correlation / (1. - settings.correlation.powi(2)).sqrt();
                let step = mix.apply(&old, 0, source, target, source_z.map(|x| scale * x))?;
                ensure!(
                    step.step.target_latent.iter().all(|x| x.abs() < 1e-8),
                    "mean not zero latent"
                );
                for factor in [0., 0.125, 0.25, 0.5, 1.] {
                    let noise = std::array::from_fn(|i| {
                        (factor * target_z[i] - settings.correlation * source_z[i])
                            / (1. - settings.correlation.powi(2)).sqrt()
                    });
                    let p = mix.apply(&old, 0, source, target, noise)?.handle;
                    let mut v = inspect(p, moving, &state, &tree, &exclusion, &wall);
                    v["factor"] = json!(factor);
                    v["translation_from_old_a"] = json!(norm(sub(p.position, old[0].position)));
                    v["rotation_from_old_degrees"] = json!(angle(p, old[0]));
                    contracted.push(v);
                }
                step.handle
            } else {
                let branch = label["branch"].as_u64().context("branch")? as usize;
                let anchor = label["anchor"].as_u64().context("anchor")? as usize;
                let mut relative = map.decode(branch, [0.; 6])?;
                if inverted[branch] {
                    relative = invert_relative_pose(relative);
                }
                result["target_mean_relative_rotation_degrees"] = json!(angle(
                    Pose {
                        position: [0.; 3],
                        orientation: [1., 0., 0., 0.]
                    },
                    relative
                ));
                for factor in [0., 0.125, 0.25, 0.5, 1.] {
                    let mut y = map.decode(branch, target_z.map(|x| factor * x))?;
                    if inverted[branch] {
                        y = invert_relative_pose(y);
                    }
                    let p = compose(state[pool[anchor]], y);
                    let mut v = inspect(p, moving, &state, &tree, &exclusion, &wall);
                    v["factor"] = json!(factor);
                    v["translation_from_old_a"] = json!(norm(sub(p.position, old[0].position)));
                    v["rotation_from_old_degrees"] = json!(angle(p, old[0]));
                    contracted.push(v);
                }
                compose(state[pool[anchor]], relative)
            };
            ensure!(
                contracted.last().unwrap()["hard_valid"] == result["candidate"]["hard_valid"],
                "decoded full-displacement validity differs"
            );
            ensure!(
                contracted.last().unwrap()["contact_bodies"]
                    == result["candidate"]["contact_bodies"],
                "decoded full-displacement contacts differ"
            );
            result["contracted_target_displacement"] = json!(contracted);
            result["mean_pose"] = serde_json::to_value(mean)?;
            result["mean"] = inspect(mean, moving, &state, &tree, &exclusion, &wall);
            if let Some(candidate) = result.get("candidate") {
                let mean_valid = result["mean"]["hard_valid"] == true;
                result["mean_valid_candidate_invalid"] =
                    json!(mean_valid && candidate["hard_valid"] == false);
            }
            let old_local = relative(state[pool[0]], old[0]);
            result["old_primary_relative_rotation_degrees"] = json!(angle(
                Pose {
                    position: [0.; 3],
                    orientation: [1., 0., 0., 0.]
                },
                old_local
            ));
        }
        serde_json::to_writer(&mut output, &result)?;
        output.write_all(b"\n")?;
        state[moving] = retained[0];
        count += 1;
    }
    output.flush()?;
    ensure!(hash_file(&input)? == input_sha, "source changed");
    save(
        &args.out.join("manifest.json"),
        &json!({"run":args.run,"input_sha256":input_sha,"attempts":count,"cpu_seconds":cpu_seconds()-clock,"scope":"deterministic replay of all archived candidates and latent-zero destination centers; no new sampling","source_bundle":include_str!(concat!(env!("OUT_DIR"),"/source-bundle.json")),"executable_sha256":hash_file(&std::env::current_exe()?)?}),
    )?;
    println!(
        "audited {count} candidates in {} CPU seconds",
        cpu_seconds() - clock
    );
    Ok(())
}
