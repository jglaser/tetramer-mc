//! Read-only hard-core geometry on a frozen list of contact circles.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::{fs, io::Write, path::PathBuf};
use tetramer_mc::{
    circle_geometry::{azimuth_allowed_mass, circle_forbidden_arcs},
    contact_distances::{AzimuthLaw, ContactFrame},
    docking::DockingConfig,
    geometry::{Placed, SphereTree},
    math::{Pose, Vec3, matvec, norm, rotation, sub},
    simulation::{cpu_seconds, hash_bytes, hash_file, save},
};

#[derive(Parser)]
#[command(about = "Audit existing contact circles; no pose draws or Poisson sampling")]
struct Cli {
    #[arg(long)]
    cases: PathBuf,
    #[arg(long)]
    expected_cases_sha256: String,
    #[arg(long)]
    out: PathBuf,
}

fn run(cli: Cli) -> Result<()> {
    ensure!(!cli.out.exists(), "Fresh output directory required");
    let bytes = fs::read(&cli.cases)?;
    ensure!(
        hash_bytes(&bytes) == cli.expected_cases_sha256,
        "Case hash changed"
    );
    let input: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        input["schema"] == "contact-circle-feasibility-cases-v1",
        "Wrong cases schema"
    );
    let config_path = PathBuf::from(input["config"].as_str().context("Missing config")?);
    let config_bytes = fs::read(&config_path)?;
    ensure!(
        hash_bytes(&config_bytes) == input["config_sha256"],
        "Config changed"
    );
    let cfg: DockingConfig = serde_json::from_slice(&config_bytes)?;
    cfg.validate()?;
    let shape_bytes = fs::read(&cfg.shape)?;
    ensure!(
        hash_bytes(&shape_bytes) == input["shape_sha256"],
        "Shape changed"
    );
    let cases = input["cases"].as_array().context("Missing cases")?;
    ensure!(cases.len() == 240, "Frozen all-circle allocation changed");
    let mut identities = std::collections::BTreeSet::new();
    ensure!(
        cases
            .iter()
            .all(|c| identities.insert(c["id"].as_str().unwrap_or("")))
            && !identities.contains(""),
        "Missing or duplicated case identity"
    );
    let tree = SphereTree::new(serde_json::from_slice(&shape_bytes)?)?;
    let fixed: Vec<_> = cfg.fixed_poses.iter().copied().map(Placed::new).collect();
    fs::create_dir(&cli.out)?;
    fs::create_dir(cli.out.join("provenance"))?;
    for (name, data) in [
        ("cases.json", bytes.as_slice()),
        ("config.json", config_bytes.as_slice()),
        ("shape.json", shape_bytes.as_slice()),
    ] {
        fs::write(cli.out.join("provenance").join(name), data)?;
    }
    let bundle = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(cli.out.join("provenance/source-bundle.json"), bundle)?;
    let manifest = json!({"schema":"contact-circle-feasibility-v1", "cases":cases.len(),
        "cases_sha256":cli.expected_cases_sha256, "config_sha256":hash_bytes(&config_bytes),
        "shape_sha256":hash_bytes(&shape_bytes), "source_bundle_sha256":hash_bytes(bundle),
        "executable_sha256":hash_file(&std::env::current_exe()?)?,
        "new_pose_draws":0, "new_Poisson_clouds":0,
        "scope":"All saved conditioned circles; analytic whole-union hard-free arcs, no R4/capture conditioning or physical mass"});
    save(&cli.out.join("manifest.json"), &manifest)?;
    let mut writer = fs::File::create(cli.out.join("circles.jsonl"))?;
    let mut journal = fs::File::create(cli.out.join("attempts.jsonl"))?;
    let started = cpu_seconds();
    for (ordinal, case) in cases.iter().enumerate() {
        let mut stage = "reconstruction";
        writeln!(
            journal,
            "{}",
            json!({"ordinal":ordinal, "id":case["id"], "state":"begin"})
        )?;
        journal.flush()?;
        let result: Result<Value> = (|| {
            let pose: Pose = serde_json::from_value(case["pose"].clone())?;
            pose.validate()?;
            let mut centers = [[0.; 3]; 2];
            let pairs = case["contact_pairs"]
                .as_array()
                .context("Missing pair labels")?;
            ensure!(
                pairs.len() == 2,
                "Exactly two frozen contact labels required"
            );
            for (j, pair) in pairs.iter().enumerate() {
                let neighbor = pair["neighbor_index"]
                    .as_u64()
                    .context("Missing neighbor")? as usize;
                let moving = pair["moving_atom"]
                    .as_u64()
                    .context("Missing moving atom")? as usize;
                let atom = pair["fixed_atom"].as_u64().context("Missing fixed atom")? as usize;
                ensure!(
                    neighbor < fixed.len()
                        && moving < tree.shape.atoms.len()
                        && atom < tree.shape.atoms.len(),
                    "Invalid frozen pair labels"
                );
                centers[j] = sub(
                    cfg.fixed_poses[neighbor].apply(tree.shape.atoms[atom].center),
                    matvec(rotation(pose.orientation), tree.shape.atoms[moving].center),
                );
            }
            let radii = serde_json::from_value(case["radii"].clone())?;
            let circle = ContactFrame::new(centers[0], centers[1])?.circle(radii)?;
            let phi = case["phi"].as_f64().context("Missing azimuth")?;
            let saved_center: Vec3 = serde_json::from_value(case["circle_center"].clone())?;
            let center_error = norm(sub(circle.center, saved_center));
            let radius_error =
                (circle.radius - case["circle_radius"].as_f64().context("Missing radius")?).abs();
            let position_error = norm(sub(circle.point(phi)?, pose.position));
            ensure!(
                center_error < 1e-9 && radius_error < 1e-9 && position_error < 1e-9,
                "Saved circle failed reconstruction"
            );
            let law = &case["azimuth_law"];
            let law = AzimuthLaw::new(
                law["mode"].as_f64().context("Missing mode")?,
                law["gamma"].as_f64().context("Missing gamma")?,
                law["localized_probability"]
                    .as_f64()
                    .context("Missing azimuth probability")?,
            )?;
            stage = "arc_geometry";
            let tick = cpu_seconds();
            let geometry =
                circle_forbidden_arcs(&tree, &circle, pose.orientation, &tree, &cfg.fixed_poses)?;
            let geometry_cpu = cpu_seconds() - tick;
            stage = "azimuth_mass";
            let mass = azimuth_allowed_mass(&law, &geometry.allowed)?;
            let uniform_mass = geometry.allowed.length() / std::f64::consts::TAU;
            stage = "direct_witnesses";
            let hard_free = |p: Pose| !fixed.iter().any(|f| tree.overlaps(&Placed::new(p), f));
            let original_hard = hard_free(pose);
            ensure!(
                original_hard
                    == case["hard_valid"]
                        .as_bool()
                        .context("Missing original predicate")?,
                "Original hard predicate changed"
            );
            let mut angles = vec![phi];
            angles.extend((0..8).map(|i| (i as f64 + 0.5) * std::f64::consts::TAU / 8.));
            for interval in geometry
                .allowed
                .intervals()
                .iter()
                .chain(geometry.forbidden.intervals())
            {
                if interval.length() > 1e-10 {
                    angles.push(0.5 * (interval.lower + interval.upper));
                }
            }
            let mut witnesses = Vec::new();
            for angle in angles {
                let direct = hard_free(Pose {
                    position: circle.point(angle)?,
                    orientation: pose.orientation,
                });
                let predicted = geometry.allowed.contains(angle);
                ensure!(
                    direct == predicted,
                    "Arc/direct-predicate discrepancy at phi={angle}"
                );
                witnesses.push(json!({"phi":angle,"hard_free":direct}));
            }
            Ok(
                json!({"ordinal":ordinal, "id":case["id"], "arm":case["arm"],
                "population":case["population"], "width_index":case["width_index"],
                "circle":circle,"moving_orientation":pose.orientation,"azimuth_law":law,
                "geometry":geometry,"original_law_allowed_mass":mass,"uniform_allowed_mass":uniform_mass,
                "geometry_cpu_seconds":geometry_cpu,"center_error":center_error,
                "radius_error":radius_error,"position_error":position_error,
                "original_hard_valid":original_hard,"witnesses":witnesses}),
            )
        })();
        match result {
            Ok(row) => {
                writeln!(writer, "{row}")?;
                writer.flush()?;
            }
            Err(error) => {
                let failure = json!({"complete":false,"ordinal":ordinal,"id":case["id"],
                    "stage":stage,"error":format!("{error:#}"),"manifest":manifest});
                let _ = save(&cli.out.join("failure.json"), &failure);
                eprintln!("{failure}");
                return Err(error);
            }
        }
    }
    save(
        &cli.out.join("summary.json"),
        &json!({"complete":true,"cases":cases.len(),
        "cpu_seconds":cpu_seconds()-started,"manifest":manifest,
        "circles_sha256":hash_file(&cli.out.join("circles.jsonl"))?,
        "attempts_sha256":hash_file(&cli.out.join("attempts.jsonl"))?}),
    )?;
    println!("Completed {} saved-circle geometry cases", cases.len());
    Ok(())
}

fn main() -> Result<()> {
    run(Cli::parse())
}
