//! Independent fixed-budget normalizer checks. The physical sphere reference
//! is radial quadrature with normalized Haar rotations; neither the proposal
//! implementation nor sampled overlap volumes enters that reference.
use anyhow::Result;
use serde_json::{Value, json};
use std::{f64::consts::PI, fs, path::Path, process::Command};
use tetramer_mc::{
    math::*,
    normalizer::{self, NormalizerOptions},
    simulation::hash_file,
};

const RADIUS: f64 = 4.;
const CORE: f64 = 1.;
const RD: f64 = 0.7;
const Z: f64 = 0.4;
const CENTER: Vec3 = [7., -4., 3.];
const ELL: f64 = 1.7;
const EPSILON: f64 = 0.55;
const ANGLE: f64 = PI / 6.;

#[derive(Clone, Copy, Default)]
struct Moment {
    n: u64,
    sum: f64,
    sum_sq: f64,
}

impl Moment {
    fn push(&mut self, value: f64) {
        self.n += 1;
        self.sum += value;
        self.sum_sq += value * value;
    }
    fn check(self, expected: f64, label: &str) {
        let n = self.n as f64;
        let mean = self.sum / n;
        let se = ((self.sum_sq / n - mean * mean).max(0.) / (n - 1.)).sqrt();
        eprintln!("{label}: estimate={mean:.8}, reference={expected:.8}, SE={se:.8}");
        assert!(
            (mean - expected).abs() <= 6.5 * se + 2e-8 * (1. + expected.abs()),
            "{label}: {mean} differs from {expected} by more than 6.5 estimated SE ({se})"
        );
    }
}

fn near(a: f64, b: f64, label: &str) {
    assert!(
        (a - b).abs() <= 2e-8 * (1. + a.abs() + b.abs()),
        "{label}: {a} != {b}"
    );
}

fn lens(r: f64) -> f64 {
    let exclusion = CORE + RD;
    if r >= 2. * exclusion {
        0.
    } else {
        PI * (4. * exclusion + r) * (2. * exclusion - r).powi(2) / 12.
    }
}

fn integrate(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        return 0.;
    }
    let count = 8192;
    let h = (hi - lo) / count as f64;
    let sum = f(lo)
        + f(hi)
        + (1..count)
            .map(|i| if i % 2 == 0 { 2. } else { 4. } * f(lo + h * i as f64))
            .sum::<f64>();
    sum * h / 3.
}

fn haar_cap(angle: f64) -> f64 {
    let a = angle.clamp(0., PI);
    (a - a.sin()) / PI
}

/// Cumulative partition integral q<=threshold, restricted to bound/unbound.
/// With the sole rigid member at the origin, q=max(r/R, theta/ANGLE).
fn reference(activity: f64, threshold: f64, bound: bool) -> f64 {
    let contact = 2. * (CORE + RD);
    let lo = if bound { 2. * CORE } else { contact };
    let hi = (RADIUS * threshold)
        .min(RADIUS)
        .min(if bound { contact } else { RADIUS });
    4. * PI
        * haar_cap(ANGLE * threshold)
        * integrate(|r| r * r * (activity * lens(r)).exp(), lo, hi)
}

fn fixed_poses() -> Vec<Pose> {
    vec![
        Pose {
            position: CENTER,
            orientation: quaternion(cayley([0.2, -0.3, 0.1])),
        },
        // This spectator affects the proposal anchor average but cannot touch
        // any hard or exclusion sphere in the capture domain.
        Pose {
            position: add(CENTER, [14., 0., 0.]),
            orientation: quaternion(cayley([-0.3, 0.2, 0.1])),
        },
    ]
}

fn fixture(root: &Path, duplicate: bool) -> Result<Value> {
    fs::create_dir_all(root)?;
    fs::write(
        root.join("shape.json"),
        json!({"name":"sphere normalizer reference", "volume":4.*PI/3.,
            "atoms":[{"center":[0.,0.,0.],"radius":CORE}]})
        .to_string(),
    )?;
    let means = [
        [0.1, -0.2, 0.05, 0.1, 0.05, -0.1],
        [-0.1, 0.1, -0.1, -0.2, 0.1, 0.05],
    ];
    let sd: [[f64; 6]; 2] = [
        [0.8, 1.8, 1.2, 0.9, 0.7, 1.1],
        [1.8, 1.2, 1.6, 1.3, 0.8, 1.2],
    ];
    let covariances: Vec<[[f64; 6]; 6]> = sd
        .iter()
        .map(|s| {
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { s[i] * s[i] } else { 0. }))
        })
        .collect();
    let mut model = json!({
        "coordinate_convention":"anchor-body-relative",
        "shape_sha256":hash_file(&root.join("shape.json"))?,
        "angular_length":ELL,
        "anchors":[
            {"position":[2.4,0.2,-0.1],"rotation":cayley([0.2,-0.3,0.1])},
            {"position":[-2.2,0.3,0.2],"rotation":cayley([-0.4,0.1,0.2])}],
        "means":means,"covariances":covariances,"weights":[0.4,0.6]
    });
    if duplicate {
        for field in ["anchors", "means", "covariances"] {
            let first = model[field][0].clone();
            model[field].as_array_mut().unwrap().push(first);
        }
        model["weights"] = json!([0.2, 0.6, 0.2]);
    }
    fs::write(root.join("model.json"), model.to_string())?;
    fs::write(
        root.join("config.json"),
        json!({
            "shape":"shape.json","fixed_poses":fixed_poses(),
            "initial_pose":Pose{position:add(CENTER,[2.6,0.,0.]),orientation:[1.,0.,0.,0.]},
            "capture_center":CENTER,"capture_radius":RADIUS,
            "depletant_radius":RD,"reservoir_density":Z,"poisson_lambda_ratio":4.,
            "translation_steps":[0.1],"rotation_steps_deg":[5.],"rotation_probability":0.5,
            "local_attempts_per_cycle":1,"uniform_probability":EPSILON,"seed":782433,
            "endpoint_gate":{"max_cells":63,"max_depth":8,"min_width":0.1},
            "metadata":{
                "native_poses":[Pose{position:CENTER,orientation:[1.,0.,0.,0.]}],
                "rigid_members":[Pose{position:[0.,0.,0.],orientation:[1.,0.,0.,0.]}],
                "member_error_scale":RADIUS,"angle_error_scale_deg":30.,
                "purpose":"geometric regions for independent radial/Haar integral; reference center need not be an allowed hard pose"
            }
        }).to_string(),
    )?;
    Ok(model)
}

/// Independent diagonal-Gaussian density reconstruction using the matrix
/// inverse Cayley formula and the normalized-Haar Jacobian. No model methods.
fn proposal_density(pose: Pose, model: &Value, epsilon: f64, covariance_scale: f64) -> f64 {
    let mut learned = 0.;
    let anchors = fixed_poses();
    for anchor in &anchors {
        let inverse = transpose(rotation(anchor.orientation));
        let t = matvec(inverse, sub(pose.position, anchor.position));
        let r = matmul(inverse, rotation(pose.orientation));
        for k in 0..model["weights"].as_array().unwrap().len() {
            let a: Vec3 = serde_json::from_value(model["anchors"][k]["position"].clone()).unwrap();
            let ar: Mat3 = serde_json::from_value(model["anchors"][k]["rotation"].clone()).unwrap();
            let delta = matmul(r, transpose(ar));
            let denominator = 1. + delta[0][0] + delta[1][1] + delta[2][2];
            if denominator <= 1e-13 {
                // The Gaussian tends to zero faster than its polynomial
                // Jacobian at this measure-zero chart seam.
                continue;
            }
            let c = [
                (delta[2][1] - delta[1][2]) / denominator,
                (delta[0][2] - delta[2][0]) / denominator,
                (delta[1][0] - delta[0][1]) / denominator,
            ];
            let latent: [f64; 6] =
                std::array::from_fn(|i| if i < 3 { t[i] - a[i] } else { ELL * c[i - 3] });
            let mut log_g = -3. * (2. * PI).ln();
            for (i, &value) in latent.iter().enumerate() {
                let variance =
                    model["covariances"][k][i][i].as_f64().unwrap() * covariance_scale.powi(2);
                let displacement = value - model["means"][k][i].as_f64().unwrap();
                log_g -= 0.5 * variance.ln() + 0.5 * displacement * displacement / variance;
            }
            log_g += 3. * ELL.ln() + 2. * PI.ln() + 2. * dot(c, c).ln_1p();
            learned += model["weights"][k].as_f64().unwrap() * log_g.exp() / anchors.len() as f64;
        }
    }
    epsilon / (2. * RADIUS).powi(3) + (1. - epsilon) * learned
}

fn run_check(
    root: &Path,
    model: &Value,
    samples: u64,
    seed: u64,
    activity: f64,
    covariance_scale: f64,
) -> Result<()> {
    let cloud_replicates = if activity > 0. { 2 } else { 1 };
    let summary = normalizer::run(NormalizerOptions {
        config: root.join("config.json"),
        model: root.join("model.json"),
        out: root.join("output"),
        samples,
        seed,
        covariance_scale,
        uniform_probability: Some(EPSILON),
        cloud_replicates,
        activity: Some(activity),
    })?;
    let mut moments = [Moment::default(); 14];
    let mut counts = [0_u64; 3];
    let names = [
        "native_core",
        "native_shell",
        "shoulder",
        "intermediate",
        "distant",
    ];
    let text = fs::read_to_string(root.join("output/samples.jsonl"))?;
    let mut draws = 0;
    for line in text.lines() {
        draws += 1;
        let row: Value = serde_json::from_str(line)?;
        let mut values = [0.; 14];
        let Some(pose) = row.get("pose").filter(|p| !p.is_null()) else {
            assert!(row["log_importance_weight"].is_null());
            for m in &mut moments {
                m.push(0.);
            }
            continue;
        };
        let pose: Pose = serde_json::from_value(pose.clone())?;
        let radius = norm(sub(pose.position, CENTER));
        let in_capture = radius <= RADIUS;
        assert_eq!(row["capture_valid"], json!(in_capture));
        if !in_capture {
            counts[0] += 1;
            assert!(row["log_importance_weight"].is_null());
        } else if radius < 2. * CORE {
            counts[1] += 1;
            assert_eq!(row["hard_valid"], false);
            assert!(row["log_importance_weight"].is_null());
        } else {
            counts[2] += 1;
            assert_eq!(row["hard_valid"], true);
            let density = proposal_density(pose, model, EPSILON, covariance_scale);
            near(
                row["log_proposal_density"].as_f64().unwrap(),
                density.ln(),
                "full mixture density",
            );
            near(
                row["log_hard_weight"].as_f64().unwrap(),
                -density.ln(),
                "hard importance weight",
            );
            let angle = 2. * pose.orientation[0].abs().clamp(0., 1.).acos();
            let q = (radius / RADIUS).max(angle / ANGLE);
            near(row["q"].as_f64().unwrap(), q, "native registration metric");
            let index = if q <= 0.8 {
                0
            } else if q <= 1. {
                1
            } else if q < 2. {
                2
            } else if q < 5. {
                3
            } else {
                4
            };
            let contact = radius <= 2. * (CORE + RD);
            assert_eq!(row["depletion_contact"], json!(contact));
            assert_eq!(
                row["region"],
                format!(
                    "{}_{}",
                    names[index],
                    if contact { "bound" } else { "unbound" }
                )
            );
            let clouds = row["clouds"].as_array().unwrap();
            assert_eq!(clouds.len(), cloud_replicates);
            let mean_cloud = clouds
                .iter()
                .map(|c| c["log_weight"].as_f64().unwrap().exp())
                .sum::<f64>()
                / cloud_replicates as f64;
            let weight = row["log_importance_weight"].as_f64().unwrap().exp();
            near(
                weight,
                mean_cloud / density,
                "averaged independent Poisson weight",
            );
            values[0] = weight;
            values[1] = density.recip();
            values[2 + 2 * index + usize::from(!contact)] = weight;
            let r00 = rotation(pose.orientation)[0][0];
            values[12] = weight * r00;
            values[13] = weight * r00 * r00;
        }
        for (m, value) in moments.iter_mut().zip(values) {
            m.push(value);
        }
    }
    assert_eq!(draws, samples, "must retain every unconditional draw");
    assert_eq!(summary["samples"], json!(samples));
    let mut summaries = vec![
        ("total".to_string(), moments[0]),
        ("hard_total".to_string(), moments[1]),
    ];
    for (i, name) in names.iter().enumerate() {
        summaries.push((format!("{name}_bound"), moments[2 + 2 * i]));
        summaries.push((format!("{name}_unbound"), moments[3 + 2 * i]));
    }
    for (key, moment) in summaries {
        let reported = &summary["estimates"][&key];
        assert_eq!(reported["unconditional_draws"], json!(samples));
        if moment.sum == 0. {
            assert!(reported["log_normalizer"].is_null());
        } else {
            near(
                reported["log_normalizer"].as_f64().unwrap(),
                (moment.sum / samples as f64).ln(),
                "summary includes zero draws",
            );
        }
    }
    assert!(
        counts.iter().all(|&n| n > samples / 100),
        "exercise outside, hard and accepted branches: {counts:?}"
    );
    let total = reference(activity, 100., true) + reference(activity, 100., false);
    moments[0].check(total, "physical Z");
    moments[1].check(
        4. * PI / 3. * (RADIUS.powi(3) - (2. * CORE).powi(3)),
        "hard Z",
    );
    let thresholds = [0., 0.8, 1., 2., 5., 100.];
    for index in 0..5 {
        for (j, bound) in [true, false].into_iter().enumerate() {
            let expected = reference(activity, thresholds[index + 1], bound)
                - reference(activity, thresholds[index], bound);
            moments[2 + 2 * index + j].check(
                expected,
                &format!(
                    "{} {}",
                    names[index],
                    if bound { "bound" } else { "unbound" }
                ),
            );
        }
    }
    moments[12].check(0., "Haar rotation first moment");
    moments[13].check(total / 3., "Haar rotation second moment");
    Ok(())
}

#[test]
fn fixed_budget_physical_regions_match_radial_and_haar_reference() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "tetramer-normalizer-reference-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root)?;
    let physical = root.join("physical");
    let model = fixture(&physical, false)?;
    run_check(&physical, &model, 24_000, 402731, Z, 1.)?;
    let zero = root.join("zero");
    let model = fixture(&zero, false)?;
    run_check(&zero, &model, 16_000, 840257, 0., 1.)?;
    let duplicated = root.join("duplicated");
    let model = fixture(&duplicated, true)?;
    run_check(&duplicated, &model, 16_000, 910637, 0., 1.)?;
    let inflated = root.join("inflated");
    let model = fixture(&inflated, false)?;
    run_check(&inflated, &model, 16_000, 729813, 0., 2.)?;
    // Exercise every public CLI override and compare the complete draw stream
    // with the library call. A different output path must not alter RNG state.
    let cli = Command::new(env!("CARGO_BIN_EXE_basin-normalizer"))
        .arg("--config")
        .arg(inflated.join("config.json"))
        .arg("--model")
        .arg(inflated.join("model.json"))
        .arg("--out")
        .arg(root.join("cli"))
        .args([
            "--samples",
            "64",
            "--seed",
            "927451",
            "--covariance-scale",
            "2",
            "--uniform-probability",
            "0.73",
            "--cloud-replicates",
            "1",
            "--activity",
            "0",
        ])
        .output()?;
    assert!(
        cli.status.success(),
        "{}",
        String::from_utf8_lossy(&cli.stderr)
    );
    let cli_summary: Value = serde_json::from_slice(&cli.stdout)?;
    let api_summary = normalizer::run(NormalizerOptions {
        config: inflated.join("config.json"),
        model: inflated.join("model.json"),
        out: root.join("api"),
        samples: 64,
        seed: 927451,
        covariance_scale: 2.,
        uniform_probability: Some(0.73),
        cloud_replicates: 1,
        activity: Some(0.),
    })?;
    assert_eq!(cli_summary["estimates"], api_summary["estimates"]);
    assert_eq!(
        fs::read(root.join("cli/samples.jsonl"))?,
        fs::read(root.join("api/samples.jsonl"))?
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
