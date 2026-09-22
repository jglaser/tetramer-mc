//! Independent analytic sphere/Haar checks of random-potential SMC, including
//! the unnormalized terminal-indicator estimator and all-attempt initialization.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use tetramer_mc::{
    latent_region::smc::{self, SmcBridge, SmcOptions},
    simulation::hash_file,
};

const R: f64 = 4.;
const ST: f64 = 0.6;
const SR: f64 = 1.3;
const ELL: f64 = 1.1;
const CAPTURE: f64 = 2.2;
const Z: f64 = 4.;

fn root(name: &str) -> Result<PathBuf> {
    let p = std::env::temp_dir().join(format!("tetramer-smc-r4-{name}-{}", std::process::id()));
    if p.exists() {
        fs::remove_dir_all(&p)?;
    }
    fs::create_dir_all(&p)?;
    Ok(p)
}
fn fixture(root: &Path, z: f64, core: f64) -> Result<()> {
    fs::write(
        root.join("shape.json"),
        json!({"name":"SMC sphere","volume":4.*PI*core.powi(3)/3.,
        "atoms":[{"center":[0.,0.,0.],"radius":core}]})
        .to_string(),
    )?;
    let fixed = json!({"position":[0.,0.,0.],"orientation":[1.,0.,0.,0.]});
    let native = json!({"position":[1.,0.,0.],"orientation":[1.,0.,0.,0.]});
    let metadata = json!({"native_poses":[native],"rigid_members":[fixed],"member_error_scale":1.,"angle_error_scale_deg":15.});
    fs::write(root.join("config.json"),json!({"shape":root.join("shape.json"),"fixed_poses":[fixed],"initial_pose":native,
        "capture_center":[0.,0.,0.],"capture_radius":CAPTURE,"depletant_radius":0.4,"reservoir_density":z,
        "poisson_lambda_ratio":8.,"translation_steps":[0.15,0.4],"rotation_steps_deg":[8.,25.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.1,"seed":1,"metadata":metadata,
        "endpoint_gate":{"max_cells":31,"max_depth":8,"min_width":0.3}}).to_string())?;
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i != j {
                0.
            } else if i < 3 {
                ST * ST
            } else {
                SR * SR
            }
        })
    });
    let hash = hash_file(&root.join("shape.json"))?;
    fs::write(root.join("region.json"),json!({"fixed_neighbor":fixed,"physical_fixed_neighbors":[fixed],
        "capture_center":[0.,0.,0.],"capture_radius":CAPTURE,"shape_sha256":hash,
        "activity":z,"depletant_radius":0.4,"physical_metric":metadata,"minimum_original_q":0.,
        "minimum_mahalanobis_radius":0.,"mahalanobis_radius":R,
        "gaussian_chart":{"shape_sha256":hash,"angular_length":ELL,"coordinate_convention":"anchor-body-relative",
        "anchors":[{"position":[0.,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],
        "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance],"weights":[1.]}}).to_string())?;
    Ok(())
}
fn options(root: &Path, out: &str, seed: u64) -> SmcOptions {
    SmcOptions {
        config: root.join("config.json"),
        region: root.join("region.json"),
        initial_reference_region: None,
        initial_current_probability: 1.,
        out: root.join(out),
        initial_draws: 4096,
        population: 192,
        seed,
        bridge: SmcBridge::PhysicalActivity,
        schedule: (0..=8).map(|i| i as f64 / 8.).collect(),
        sweeps_per_stage: 6,
        cloud_replicates: 2,
        lambda_ratio: 8.,
    }
}
fn read(p: impl AsRef<Path>) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn lines(p: impl AsRef<Path>) -> Result<Vec<Value>> {
    fs::read_to_string(p)?
        .lines()
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 3e-11 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
fn norm(v: &Value) -> f64 {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap().powi(2))
        .sum::<f64>()
        .sqrt()
}
fn lens(r: f64) -> f64 {
    if r < 1.4 {
        PI * (2.8 + r) * (1.4 - r).powi(2) / 12.
    } else {
        0.
    }
}
fn reference(z: f64, upper: f64) -> f64 {
    // Independent radial quadrature: integrate physical translations, and use
    // the exact normalized SO(3) Haar cap of the available Cayley 3-ball.
    let f = |r: f64| {
        let a = SR / ELL * (R * R - (r / ST).powi(2)).max(0.).sqrt();
        8. * r * r * (a.atan() - a / (1. + a * a)) * (z * lens(r)).exp()
    };
    let n = 32768;
    let hi = upper.min(CAPTURE).min(ST * R);
    let h = (hi - 0.6) / n as f64;
    (f(0.6)
        + f(hi)
        + (1..n)
            .map(|i| f(0.6 + h * i as f64) * if i % 2 == 0 { 2. } else { 4. })
            .sum::<f64>())
        * h
        / 3.
}
fn check_populations(x: &[f64], exact: f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let se = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n * (n - 1.))).sqrt();
    eprintln!("SMC estimate={mean}, exact={exact}, independent-population SE={se}");
    assert!(
        (mean - exact).abs() < 6. * se + 1e-6,
        "{mean} != {exact} at SE {se}"
    );
}
fn verify_records(out: &Path, z: f64) -> Result<()> {
    let initial = lines(out.join("initialization.jsonl"))?;
    let stages = lines(out.join("stages.jsonl"))?;
    let v = PI.powi(3) * R.powi(6) / 6.;
    let bridge = read(out.join("manifest.json"))?["bridge"] == "proposal_density";
    let mut sum = 0.;
    for (i, row) in initial.iter().enumerate() {
        assert_eq!(row["draw"], i);
        let u = row["latent"].as_array().unwrap();
        let a2 = u[3..]
            .iter()
            .map(|x| (x.as_f64().unwrap() * SR / ELL).powi(2))
            .sum::<f64>();
        let j = ST.powi(3) * SR.powi(3) / (ELL.powi(3) * PI.powi(2) * (1. + a2).powi(2));
        near(row["log_physical_jacobian"].as_f64().unwrap(), j.ln());
        near(row["log_proposal_density"].as_f64().unwrap(), -v.ln());
        let r = norm(&row["pose"]["position"]);
        let valid = r >= 0.6 && r <= CAPTURE;
        assert_eq!(row["hard_valid"], valid);
        if valid {
            near(row["log_hard_weight"].as_f64().unwrap(), (v * j).ln());
            sum += if bridge { 1. } else { v * j };
        } else {
            assert!(row["log_hard_weight"].is_null());
        }
    }
    let mut logz = (sum / initial.len() as f64).ln();
    near(stages[0]["log_Z"].as_f64().unwrap(), logz);
    for stage in stages.iter().skip(1) {
        let delta = stage["delta_activity"].as_f64().unwrap();
        let lambda = stage["incremental_lambda"].as_f64().unwrap();
        let rows = stage["potentials"].as_array().unwrap();
        let mut weights = Vec::new();
        for row in rows {
            let clouds = row["clouds"].as_array().unwrap();
            let mut w = 0.;
            for cloud in clouds {
                let logw = delta * cloud["lower_volume"].as_f64().unwrap()
                    + cloud["overlap_points"].as_u64().unwrap() as f64 * (delta / lambda).ln_1p();
                near(cloud["log_weight"].as_f64().unwrap(), logw);
                w += logw.exp() / clouds.len() as f64;
            }
            let correction = if bridge {
                -stage["delta_beta"].as_f64().unwrap() * row["log_g"].as_f64().unwrap()
            } else {
                0.
            };
            near(
                row["deterministic_log_correction"].as_f64().unwrap(),
                correction,
            );
            near(
                row["log_incremental_weight"].as_f64().unwrap(),
                w.ln() + correction,
            );
            weights.push(w * correction.exp());
        }
        let sum = weights.iter().sum::<f64>();
        let inc = (sum / weights.len() as f64).ln();
        logz += inc;
        near(stage["log_Z_increment"].as_f64().unwrap(), inc);
        near(stage["log_Z"].as_f64().unwrap(), logz);
        let parents = stage["parents"].as_array().unwrap();
        let offset = stage["resampling_offset"].as_f64().unwrap();
        let mut parent = 0;
        let mut cumulative = weights[0];
        for (i, actual) in parents.iter().enumerate() {
            let target = (offset + i as f64 / parents.len() as f64) * sum;
            while cumulative <= target && parent + 1 < weights.len() {
                parent += 1;
                cumulative += weights[parent];
            }
            assert_eq!(actual.as_u64().unwrap() as usize, parent);
            let child = &stage["particles"][i];
            assert_eq!(child["initial_ancestor"], rows[parent]["initial_ancestor"]);
            assert!(norm(&child["latent"]) <= R + 1e-10);
            let r = norm(&child["pose"]["position"]);
            assert!(r >= 0.6 - 1e-12 && r <= CAPTURE + 1e-12);
        }
        for record in stage["mutation_density_records"].as_array().unwrap() {
            let correction = (1. - stage["beta"].as_f64().unwrap())
                * (record["log_g_new"].as_f64().unwrap() - record["log_g_old"].as_f64().unwrap());
            near(
                record["deterministic_log_correction"].as_f64().unwrap(),
                correction,
            );
            let loga = (correction + record["gate"]["log_weight"].as_f64().unwrap()).min(0.);
            near(record["log_acceptance"].as_f64().unwrap(), loga);
            assert_eq!(
                record["accepted"],
                record["log_uniform"].as_f64().unwrap() < loga
            );
        }
        if z == 0. && !bridge {
            near(inc, 0.);
        }
    }
    near(
        read(out.join("summary.json"))?["log_Z"].as_f64().unwrap(),
        logz,
    );
    Ok(())
}

#[test]
fn positive_sphere_normalizer_and_terminal_contact_indicator() -> Result<()> {
    let root = root("positive")?;
    fixture(&root, Z, 0.3)?;
    let mut total = Vec::new();
    let mut contact = Vec::new();
    for i in 0..8 {
        let out = format!("p{i}");
        let opts = options(&root, &out, 4849107 + 1009 * i);
        let result = smc::run(opts)?;
        let q = result["log_Z"].as_f64().unwrap().exp();
        let particles = result["terminal_particles"].as_array().unwrap();
        let hits = particles
            .iter()
            .filter(|p| norm(&p["pose"]["position"]) < 1.4)
            .count();
        total.push(q);
        contact.push(q * hits as f64 / particles.len() as f64);
        if i == 0 {
            verify_records(&root.join(&out), Z)?;
            let stages = lines(root.join(&out).join("stages.jsonl"))?;
            for field in ["accepted_translation_changes", "accepted_rotation_changes"] {
                assert!(
                    stages
                        .iter()
                        .skip(1)
                        .map(|s| s["mutation_counts"][field].as_u64().unwrap())
                        .sum::<u64>()
                        > 0
                );
            }
        }
    }
    check_populations(&total, reference(Z, CAPTURE));
    check_populations(&contact, reference(Z, 1.4));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn zero_activity_reconstruction_determinism_and_cli() -> Result<()> {
    let root = root("zero")?;
    fixture(&root, 0., 0.3)?;
    let mut opts = options(&root, "api", 1119);
    opts.initial_draws = 256;
    opts.population = 64;
    opts.schedule = vec![0., 0.5, 1.];
    opts.sweeps_per_stage = 2;
    let result = smc::run(opts.clone())?;
    assert!(!result["zero_estimate"].as_bool().unwrap());
    verify_records(&root.join("api"), 0.)?;
    let replay = smc::run(opts.clone());
    assert!(replay.is_err());
    let cli = Command::new(env!("CARGO_BIN_EXE_latent-region-smc"))
        .args([
            "--config",
            root.join("config.json").to_str().unwrap(),
            "--region",
            root.join("region.json").to_str().unwrap(),
            "--out",
            root.join("cli").to_str().unwrap(),
            "--seed",
            "1119",
            "--initial-draws",
            "256",
            "--population",
            "64",
            "--stages",
            "2",
            "--sweeps-per-stage",
            "2",
            "--cloud-replicates",
            "2",
            "--lambda-ratio",
            "8",
        ])
        .output()?;
    assert!(
        cli.status.success(),
        "{}",
        String::from_utf8_lossy(&cli.stderr)
    );
    for file in ["initialization.jsonl", "stages.jsonl"] {
        assert_eq!(
            fs::read(root.join("api").join(file))?,
            fs::read(root.join("cli").join(file))?
        );
    }
    let mut region = read(root.join("region.json"))?;
    region["minimum_original_q"] = json!(1.);
    fs::write(root.join("restricted.json"), region.to_string())?;
    opts.region = root.join("restricted.json");
    opts.out = root.join("bad-region");
    assert!(smc::run(opts).is_err());
    assert!(!root.join("bad-region").exists());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn zero_hit_population_is_preserved_without_refill() -> Result<()> {
    let root = root("nohits")?;
    fixture(&root, 0., 10.)?;
    let mut opts = options(&root, "run", 117);
    opts.initial_draws = 37;
    opts.population = 19;
    let result = smc::run(opts)?;
    assert_eq!(result["complete"], true);
    assert_eq!(result["zero_estimate"], true);
    assert_eq!(result["initial_hits"], 0);
    assert_eq!(result["initial_draws"], 37);
    assert_eq!(result["Z"], 0.);
    assert!(result["log_Z"].is_null());
    assert_eq!(result["terminal_particles"].as_array().unwrap().len(), 0);
    assert_eq!(lines(root.join("run/initialization.jsonl"))?.len(), 37);
    assert_eq!(lines(root.join("run/stages.jsonl"))?.len(), 1);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn unfiltered_two_chart_initialization_has_complete_physical_density() -> Result<()> {
    let root = root("mixture")?;
    fixture(&root, 0., 0.3)?;
    let mut reference_spec = read(root.join("region.json"))?;
    let st = 0.55_f64;
    let sr = 1.6_f64;
    let ell = 1.7_f64;
    let radius = 5_f64;
    let mean = [1.2, 0.15, -0.2, 0.2, -0.1, 0.05];
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i != j {
                0.
            } else if i < 3 {
                st * st
            } else {
                sr * sr
            }
        })
    });
    reference_spec["gaussian_chart"]["means"] = json!([mean]);
    reference_spec["gaussian_chart"]["covariances"] = json!([covariance]);
    reference_spec["gaussian_chart"]["angular_length"] = json!(ell);
    reference_spec["mahalanobis_radius"] = json!(radius);
    // These incompatible OLD restrictions must never filter the proposal.
    reference_spec["capture_radius"] = json!(0.01);
    reference_spec["minimum_original_q"] = json!(999.);
    reference_spec["maximum_original_q"] = json!(1000.);
    reference_spec["minimum_mahalanobis_radius"] = json!(4.99);
    fs::write(root.join("reference.json"), reference_spec.to_string())?;
    let mut opts = options(&root, "run", 184917);
    opts.initial_reference_region = Some(root.join("reference.json"));
    opts.initial_current_probability = 0.5;
    opts.initial_draws = 16384;
    opts.population = 128;
    opts.schedule = vec![0., 1.];
    opts.sweeps_per_stage = 0;
    let result = smc::run(opts)?;
    let mut weights = Vec::new();
    let mut exterior = 0;
    let mut from_reference = 0;
    let mut outside_old_capture = 0;
    let v4 = PI.powi(3) * R.powi(6) / 6.;
    let v5 = PI.powi(3) * radius.powi(6) / 6.;
    for row in lines(root.join("run/initialization.jsonl"))? {
        let p = row["pose"]["position"].as_array().unwrap();
        let quat = row["pose"]["orientation"].as_array().unwrap();
        let c: [f64; 3] =
            std::array::from_fn(|i| quat[i + 1].as_f64().unwrap() / quat[0].as_f64().unwrap());
        let u: [f64; 6] = std::array::from_fn(|i| {
            if i < 3 {
                p[i].as_f64().unwrap() / ST
            } else {
                ELL * c[i - 3] / SR
            }
        });
        let ur: [f64; 6] = std::array::from_fn(|i| {
            if i < 3 {
                (p[i].as_f64().unwrap() - mean[i]) / st
            } else {
                (ell * c[i - 3] - mean[i]) / sr
            }
        });
        let in_current = u.iter().map(|v| v * v).sum::<f64>() <= R * R;
        let in_reference = ur.iter().map(|v| v * v).sum::<f64>() <= radius * radius;
        assert_eq!(row["current_ball_valid"], in_current);
        assert_eq!(row["reference_ball_valid"], in_reference);
        let c2 = c.iter().map(|v| v * v).sum::<f64>();
        let j = ST.powi(3) * SR.powi(3) / (ELL.powi(3) * PI.powi(2) * (1. + c2).powi(2));
        let jr = st.powi(3) * sr.powi(3) / (ell.powi(3) * PI.powi(2) * (1. + c2).powi(2));
        near(row["log_physical_jacobian"].as_f64().unwrap(), j.ln());
        near(
            row["reference_log_physical_jacobian"].as_f64().unwrap(),
            jr.ln(),
        );
        let density = if in_current { 0.5 / (v4 * j) } else { 0. }
            + if in_reference { 0.5 / (v5 * jr) } else { 0. };
        near(
            row["log_physical_proposal_density"].as_f64().unwrap(),
            density.ln(),
        );
        let r = norm(&row["pose"]["position"]);
        let valid = in_current && r >= 0.6 && r <= CAPTURE;
        if valid {
            near(row["log_hard_weight"].as_f64().unwrap(), -density.ln());
            weights.push(1. / density);
        } else {
            assert!(row["log_hard_weight"].is_null());
            weights.push(0.);
        }
        if !in_current {
            exterior += 1;
            assert_eq!(row["selected_initial_chart"], "reference");
        }
        if row["selected_initial_chart"] == "reference" {
            from_reference += 1;
            if r > 0.01 {
                outside_old_capture += 1;
            }
        }
    }
    assert!(exterior > 100);
    assert!(from_reference > 7000);
    assert!(outside_old_capture > 7000);
    let n = weights.len() as f64;
    let mean = weights.iter().sum::<f64>() / n;
    let se = (weights.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n * (n - 1.))).sqrt();
    assert!((mean - reference(0., CAPTURE)).abs() < 6. * se);
    near(result["log_Z"].as_f64().unwrap(), mean.ln());
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn proposal_density_bridge_recovers_physical_mass_and_subset() -> Result<()> {
    let root = root("density-bridge")?;
    fixture(&root, Z, 0.3)?;
    let mut total = Vec::new();
    let mut contact = Vec::new();
    for i in 0..8 {
        let out = format!("p{i}");
        let mut opts = options(&root, &out, 7130107 + 1009 * i);
        opts.bridge = SmcBridge::ProposalDensity;
        let result = smc::run(opts)?;
        let q = result["log_Z"].as_f64().unwrap().exp();
        let particles = result["terminal_particles"].as_array().unwrap();
        total.push(q);
        contact.push(
            q * particles
                .iter()
                .filter(|p| norm(&p["pose"]["position"]) < 1.4)
                .count() as f64
                / particles.len() as f64,
        );
        if i == 0 {
            verify_records(&root.join(&out), Z)?;
        }
    }
    check_populations(&total, reference(Z, CAPTURE));
    check_populations(&contact, reference(Z, 1.4));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn zero_activity_two_chart_density_bridge_keeps_density_corrections() -> Result<()> {
    let root = root("zero-mixture-bridge")?;
    fixture(&root, 0., 0.3)?;
    let mut spec = read(root.join("region.json"))?;
    spec["mahalanobis_radius"] = json!(5.);
    spec["gaussian_chart"]["means"] = json!([[0.9, 0., 0., 0., 0., 0.]]);
    spec["minimum_original_q"] = json!(999.);
    spec["capture_radius"] = json!(0.01);
    fs::write(root.join("reference.json"), spec.to_string())?;
    let physical_density = |pose: &Value| {
        let position = pose["position"].as_array().unwrap();
        let q = pose["orientation"].as_array().unwrap();
        let c2 = (1..4)
            .map(|i| (q[i].as_f64().unwrap() / q[0].as_f64().unwrap()).powi(2))
            .sum::<f64>();
        let j = ST.powi(3) * SR.powi(3) / (ELL.powi(3) * PI.powi(2) * (1. + c2).powi(2));
        let angular2 = c2 * (ELL / SR).powi(2);
        let current2 = position
            .iter()
            .map(|p| (p.as_f64().unwrap() / ST).powi(2))
            .sum::<f64>()
            + angular2;
        let reference2 = position
            .iter()
            .enumerate()
            .map(|(i, p)| ((p.as_f64().unwrap() - if i == 0 { 0.9 } else { 0. }) / ST).powi(2))
            .sum::<f64>()
            + angular2;
        let v4 = PI.powi(3) * 4_f64.powi(6) / 6.;
        let v5 = PI.powi(3) * 5_f64.powi(6) / 6.;
        (if current2 <= 16. { 0.5 / (v4 * j) } else { 0. })
            + (if reference2 <= 25. {
                0.5 / (v5 * j)
            } else {
                0.
            })
    };
    let mut total = Vec::new();
    let mut subset = Vec::new();
    let mut correction_seen = false;
    for i in 0..8 {
        let out = format!("p{i}");
        let mut opts = options(&root, &out, 1871001 + 1009 * i);
        opts.bridge = SmcBridge::ProposalDensity;
        opts.initial_reference_region = Some(root.join("reference.json"));
        opts.initial_current_probability = 0.5;
        let result = smc::run(opts)?;
        let q = result["log_Z"].as_f64().unwrap().exp();
        let particles = result["terminal_particles"].as_array().unwrap();
        total.push(q);
        subset.push(
            q * particles
                .iter()
                .filter(|p| norm(&p["pose"]["position"]) < 1.4)
                .count() as f64
                / particles.len() as f64,
        );
        if i == 0 {
            let rows = lines(root.join(&out).join("initialization.jsonl"))?;
            for row in &rows {
                if row["hard_valid"] == true {
                    assert_eq!(row["log_initial_weight"], 0.);
                } else {
                    assert!(row["log_initial_weight"].is_null());
                }
            }
            let stages = lines(root.join(&out).join("stages.jsonl"))?;
            near(
                stages[0]["log_Z"].as_f64().unwrap(),
                (result["initial_hits"].as_u64().unwrap() as f64 / rows.len() as f64).ln(),
            );
            let mut logz = stages[0]["log_Z"].as_f64().unwrap();
            for stage in stages.iter().skip(1) {
                let db = stage["delta_beta"].as_f64().unwrap();
                let beta = stage["beta"].as_f64().unwrap();
                let mut weights = Vec::new();
                for row in stage["potentials"].as_array().unwrap() {
                    let logg = physical_density(&row["pose"]).ln();
                    near(row["log_g"].as_f64().unwrap(), logg);
                    for cloud in row["clouds"].as_array().unwrap() {
                        assert_eq!(cloud["log_weight"], 0.);
                    }
                    near(row["log_incremental_weight"].as_f64().unwrap(), -db * logg);
                    weights.push((-db * logg).exp());
                }
                logz += (weights.iter().sum::<f64>() / weights.len() as f64).ln();
                near(stage["log_Z"].as_f64().unwrap(), logz);
                for row in stage["mutation_density_records"].as_array().unwrap() {
                    let old = physical_density(&row["old_pose"]).ln();
                    let new = physical_density(&row["proposed_pose"]).ln();
                    near(row["log_g_old"].as_f64().unwrap(), old);
                    near(row["log_g_new"].as_f64().unwrap(), new);
                    let correction = (1. - beta) * (new - old);
                    near(
                        row["deterministic_log_correction"].as_f64().unwrap(),
                        correction,
                    );
                    if correction.abs() > 1e-8 {
                        correction_seen = true;
                    }
                    assert_eq!(row["gate"]["log_weight"], 0.);
                    assert_eq!(
                        row["accepted"],
                        row["log_uniform"].as_f64().unwrap() < correction.min(0.)
                    );
                }
            }
            near(logz, result["log_Z"].as_f64().unwrap());
        }
    }
    assert!(correction_seen);
    check_populations(&total, reference(0., CAPTURE));
    check_populations(&subset, reference(0., 1.4));
    fs::remove_dir_all(root)?;
    Ok(())
}
