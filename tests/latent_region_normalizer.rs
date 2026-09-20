//! Independent broad-rotation volume quadrature for the uniform latent ball.
use anyhow::Result;
use serde_json::{Value, json};
use std::{f64::consts::PI, fs, path::Path, process::Command};
use tetramer_mc::{
    latent_region::{self, LatentRegionOptions},
    math::*,
    simulation::hash_file,
};

const R: f64 = 2.4;
const ST: f64 = 0.7;
const SR: f64 = 1.9;
const ELL: f64 = 1.3;
const CENTER: Vec3 = [2., -3., 4.];

#[derive(Default)]
struct Moment {
    n: usize,
    sum: f64,
    sq: f64,
}
impl Moment {
    fn push(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.sq += x * x;
    }
    fn check(&self, expected: f64, label: &str) {
        let n = self.n as f64;
        let mean = self.sum / n;
        let se = ((self.sq / n - mean * mean).max(0.) / (n - 1.)).sqrt();
        eprintln!("{label}: estimate={mean:.8}, quadrature={expected:.8}, SE={se:.8}");
        assert!(
            (mean - expected).abs() < 6.5 * se + 2e-9 * (1. + expected.abs()),
            "{label}: {mean} != {expected}; SE={se}"
        );
    }
}

fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-9 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}
fn fixed() -> Pose {
    Pose {
        position: [100., -80., 120.],
        orientation: quaternion(cayley([0.4, -0.2, 0.3])),
    }
}
fn anchor_rotation() -> Mat3 {
    cayley([-0.3, 0.5, 0.1])
}

fn fixture(root: &Path, capture: f64, minimum_q: f64) -> Result<()> {
    fs::create_dir_all(root)?;
    fs::write(root.join("shape.json"),json!({"name":"remote small sphere","volume":4.*PI*0.1_f64.powi(3)/3.,"atoms":[{"center":[0.,0.,0.],"radius":0.1}]}).to_string())?;
    let native = Pose {
        position: CENTER,
        orientation: quaternion(matmul(rotation(fixed().orientation), anchor_rotation())),
    };
    let metadata = json!({"native_poses":[native],"rigid_members":[{"position":[0.,0.,0.],"orientation":[1.,0.,0.,0.]}],"member_error_scale":1000.,"angle_error_scale_deg":30.});
    let cfg = json!({"shape":root.join("shape.json"),"fixed_poses":[fixed()],"initial_pose":native,
        "capture_center":CENTER,"capture_radius":capture,"depletant_radius":0.4,"reservoir_density":0.,"poisson_lambda_ratio":64.,
        "translation_steps":[0.1],"rotation_steps_deg":[1.],"rotation_probability":0.5,"local_attempts_per_cycle":1,
        "uniform_probability":0.1,"seed":1,"metadata":metadata});
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
    let shape_hash = hash_file(&root.join("shape.json"))?;
    let region = json!({"fixed_neighbor":fixed(),"capture_center":CENTER,"capture_radius":capture,
        "activity":0.,"depletant_radius":0.4,"physical_metric":metadata,"shape_sha256":shape_hash,
        "minimum_original_q":minimum_q,"mahalanobis_radius":R,
        "gaussian_chart":{"coordinate_convention":"anchor-body-relative","shape_sha256":shape_hash,"angular_length":ELL,
            "anchors":[{"position":fixed().inverse(CENTER),"rotation":anchor_rotation()}],
            "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance],"weights":[1.]}});
    fs::write(root.join("config.json"), cfg.to_string())?;
    fs::write(root.join("region.json"), region.to_string())?;
    Ok(())
}

fn quadrature(capture: f64, minimum_q: f64) -> f64 {
    // Rotation radius rho=R sin(t) gives an analytic translation 3-ball
    // section min(sqrt(R²-rho²),capture/ST). Integrate its physical volume
    // against the normalized Haar Cayley factor, independently of the sampler.
    let rho_min = ELL / SR * (minimum_q * PI / 12.).tan();
    let lo = (rho_min / R).clamp(0., 1.).asin();
    let n = 32768;
    let h = (PI / 2. - lo) / n as f64;
    let f = |t: f64| {
        let rho = R * t.sin();
        let translation = (R * t.cos()).max(0.).min(capture / ST);
        16. / 3. * ST.powi(3) * (SR / ELL).powi(3) * rho.powi(2) * translation.powi(3) * R * t.cos()
            / (1. + (SR / ELL * rho).powi(2)).powi(2)
    };
    (f(lo)
        + f(PI / 2.)
        + (1..n)
            .map(|i| {
                if i % 2 == 0 {
                    2. * f(lo + h * i as f64)
                } else {
                    4. * f(lo + h * i as f64)
                }
            })
            .sum::<f64>())
        * h
        / 3.
}

fn verify(root: &Path, out: &str, n: u64, capture: f64, minimum_q: f64) -> Result<()> {
    let summary: Value = serde_json::from_slice(&fs::read(root.join(out).join("summary.json"))?)?;
    assert!(summary["complete"].as_bool().unwrap());
    let rows = fs::read_to_string(root.join(out).join("samples.jsonl"))?;
    let mut mass = Moment::default();
    let mut r2 = Moment::default();
    let mut r4 = Moment::default();
    let mut axis = Moment::default();
    let mut cross = Moment::default();
    let mut small = Moment::default();
    let mut zeros = 0;
    for (i, line) in rows.lines().enumerate() {
        let row: Value = serde_json::from_str(line)?;
        assert_eq!(row["draw"].as_u64(), Some(i as u64));
        let latent: [f64; 6] = serde_json::from_value(row["latent"].clone())?;
        let pose: Pose = serde_json::from_value(row["pose"].clone())?;
        let radius = latent.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(radius <= R * (1. + 1e-12));
        near(radius, row["latent_radius"].as_f64().unwrap());
        r2.push(radius * radius);
        r4.push(radius.powi(4));
        axis.push(latent[0] * latent[0]);
        cross.push(latent[0] * latent[1]);
        small.push(f64::from(radius <= R / 2.));
        let relative_t = fixed().inverse(pose.position);
        let anchor_t = fixed().inverse(CENTER);
        let delta = quaternion(matmul(
            matmul(
                transpose(rotation(fixed().orientation)),
                rotation(pose.orientation),
            ),
            transpose(anchor_rotation()),
        ));
        let c = [
            delta[1] / delta[0],
            delta[2] / delta[0],
            delta[3] / delta[0],
        ];
        for k in 0..3 {
            near((relative_t[k] - anchor_t[k]) / ST, latent[k]);
            near(c[k] * ELL / SR, latent[k + 3]);
        }
        let expected_j =
            ST.powi(3) * SR.powi(3) / (ELL.powi(3) * PI.powi(2) * (1. + dot(c, c)).powi(2));
        near(
            row["log_physical_jacobian"].as_f64().unwrap(),
            expected_j.ln(),
        );
        let q = (2. * norm(c).atan() / (PI / 6.)).max(norm(sub(pose.position, CENTER)) / 1000.);
        near(q, row["q"].as_f64().unwrap());
        let valid = norm(sub(pose.position, CENTER)) <= capture && q >= minimum_q;
        assert_eq!(row["region_valid"].as_bool(), Some(q >= minimum_q));
        if valid {
            let w = row["log_importance_weight"].as_f64().unwrap().exp();
            near(w, PI.powi(3) * R.powi(6) / 6. * expected_j);
            assert_eq!(row["clouds"].as_array().unwrap().len(), 2);
            for cloud in row["clouds"].as_array().unwrap() {
                assert_eq!(cloud["log_weight"].as_f64(), Some(0.));
            }
            mass.push(w);
        } else {
            zeros += 1;
            assert!(row["log_importance_weight"].is_null());
            assert!(row["clouds"].as_array().unwrap().is_empty());
            mass.push(0.);
        }
    }
    assert_eq!(mass.n as u64, n);
    mass.check(quadrature(capture, minimum_q), out);
    r2.check(0.75 * R * R, "E r²");
    r4.check(0.6 * R.powi(4), "E r⁴");
    axis.check(R * R / 8., "E u0²");
    cross.check(0., "E u0u1");
    small.check(1. / 64., "P(r<R/2)");
    near(
        summary["estimates"]["region"]["logQ"]
            .as_f64()
            .unwrap()
            .exp(),
        mass.sum / n as f64,
    );
    if capture < R * ST {
        assert!(zeros > n / 4);
    } else {
        assert_eq!(zeros, 0);
    }
    Ok(())
}

#[test]
fn broad_rotational_volume_and_unconditional_zeros() -> Result<()> {
    let root = std::env::temp_dir().join(format!("tetramer-latent-region-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    for (name, capture, qmin, seed) in [("full", 10., 0., 7491210), ("clipped", 0.8, 1., 7493210)] {
        let dir = root.join(name);
        fixture(&dir, capture, qmin)?;
        latent_region::run(LatentRegionOptions {
            config: dir.join("config.json"),
            region: dir.join("region.json"),
            out: dir.join("api"),
            samples: 32768,
            seed,
            cloud_replicates: 2,
            lambda_ratio: 64.,
        })?;
        verify(&dir, "api", 32768, capture, qmin)?;
    }
    let dir = root.join("cli");
    fixture(&dir, 10., 0.)?;
    latent_region::run(LatentRegionOptions {
        config: dir.join("config.json"),
        region: dir.join("region.json"),
        out: dir.join("api"),
        samples: 64,
        seed: 88311,
        cloud_replicates: 2,
        lambda_ratio: 64.,
    })?;
    let status = Command::new(env!("CARGO_BIN_EXE_latent-region-normalizer"))
        .args([
            "--config",
            dir.join("config.json").to_str().unwrap(),
            "--region",
            dir.join("region.json").to_str().unwrap(),
            "--out",
            dir.join("cli").to_str().unwrap(),
            "--samples",
            "64",
            "--seed",
            "88311",
        ])
        .output()?;
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    assert_eq!(
        fs::read(dir.join("api/samples.jsonl"))?,
        fs::read(dir.join("cli/samples.jsonl"))?
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

fn shell_reference(inner: f64, capture: f64, qmin: f64, qmax: f64, z: f64) -> f64 {
    // Independent radial quadrature of translation volume and normalized Haar.
    // Only the second, central physical sphere interacts (core .3, inflated .7).
    let lo = ELL / SR * (qmin * PI / 12.).tan();
    let hi = (ELL / SR * (qmax * PI / 12.).tan()).min(R);
    let nr = 1024;
    let nt = 512;
    let dh = (hi - lo) / nr as f64;
    let integrand = |rho: f64| {
        let low = (ST * (inner * inner - rho * rho).max(0.).sqrt()).max(0.6);
        let high = (ST * (R * R - rho * rho).max(0.).sqrt()).min(capture);
        if high <= low {
            return 0.;
        }
        let h = (high - low) / nt as f64;
        let f = |d: f64| {
            let overlap = if d < 1.4 {
                PI * (2.8 + d) * (1.4 - d).powi(2) / 12.
            } else {
                0.
            };
            d * d * (z * overlap).exp()
        };
        let radial = (f(low)
            + f(high)
            + (1..nt)
                .map(|i| (if i % 2 == 0 { 2. } else { 4. }) * f(low + i as f64 * h))
                .sum::<f64>())
            * h
            / 3.;
        16. * (SR / ELL).powi(3) * rho * rho * radial / (1. + (SR / ELL * rho).powi(2)).powi(2)
    };
    (integrand(lo)
        + integrand(hi)
        + (1..nr)
            .map(|i| (if i % 2 == 0 { 2. } else { 4. }) * integrand(lo + i as f64 * dh))
            .sum::<f64>())
        * dh
        / 3.
}

#[test]
fn shell_q_window_and_second_neighbor_depletion() -> Result<()> {
    let root = std::env::temp_dir().join(format!("tetramer-latent-shell-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    let (inner, capture, qmin, qmax, z): (f64, f64, f64, f64, f64) = (1.2, 1.0, 1.0, 4.0, 2.0);
    fixture(&root, capture, qmin)?;
    let mut shape: Value = serde_json::from_slice(&fs::read(root.join("shape.json"))?)?;
    shape["atoms"][0]["radius"] = json!(0.3);
    shape["volume"] = json!(4. * PI * 0.3_f64.powi(3) / 3.);
    fs::write(root.join("shape.json"), shape.to_string())?;
    let shape_hash = hash_file(&root.join("shape.json"))?;
    let mut cfg: Value = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
    cfg["fixed_poses"].as_array_mut().unwrap().push(json!(Pose {
        position: CENTER,
        orientation: [1., 0., 0., 0.]
    }));
    cfg["reservoir_density"] = json!(z);
    fs::write(root.join("config.json"), cfg.to_string())?;
    let mut region: Value = serde_json::from_slice(&fs::read(root.join("region.json"))?)?;
    region["physical_fixed_neighbors"] = cfg["fixed_poses"].clone();
    region["activity"] = json!(z);
    region["shape_sha256"] = json!(shape_hash);
    region["gaussian_chart"]["shape_sha256"] = json!(shape_hash);
    region["minimum_mahalanobis_radius"] = json!(inner);
    region["maximum_original_q"] = json!(qmax);
    region["minimum_original_q_inclusive"] = json!(false);
    region["maximum_original_q_inclusive"] = json!(false);
    fs::write(root.join("region.json"), region.to_string())?;
    let options = LatentRegionOptions {
        config: root.join("config.json"),
        region: root.join("region.json"),
        out: root.join("run"),
        samples: 32768,
        seed: 883917,
        cloud_replicates: 2,
        lambda_ratio: 64.,
    };
    let summary = latent_region::run(options.clone())?;
    assert_eq!(summary["manifest"]["minimum_original_q_inclusive"], false);
    assert_eq!(summary["manifest"]["maximum_original_q_inclusive"], false);
    let volume = PI.powi(3) * (R.powi(6) - inner.powi(6)) / 6.;
    near(
        summary["manifest"]["log_latent_shell_volume"]
            .as_f64()
            .unwrap(),
        volume.ln(),
    );
    let (mut mass, mut hard, mut uniform6, mut r2) = (
        Moment::default(),
        Moment::default(),
        Moment::default(),
        Moment::default(),
    );
    let mut zero_hard = 0;
    let mut zero_q = 0;
    for line in fs::read_to_string(root.join("run/samples.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let radius = row["latent_radius"].as_f64().unwrap();
        assert!(radius >= inner * (1. - 1e-12) && radius <= R * (1. + 1e-12));
        uniform6.push((radius.powi(6) - inner.powi(6)) / (R.powi(6) - inner.powi(6)));
        r2.push(radius * radius);
        let pose: Pose = serde_json::from_value(row["pose"].clone())?;
        let d = norm(sub(pose.position, CENTER));
        let q = row["q"].as_f64().unwrap();
        let hard_ok = d <= capture && d >= 0.6;
        let region_ok = q > qmin && q < qmax;
        assert_eq!(row["hard_valid"].as_bool(), Some(hard_ok));
        assert_eq!(row["region_valid"].as_bool(), Some(region_ok));
        zero_hard += usize::from(!hard_ok);
        zero_q += usize::from(!region_ok);
        let h = volume * row["physical_jacobian"].as_f64().unwrap();
        if hard_ok && region_ok {
            near(row["log_hard_weight"].as_f64().unwrap(), h.ln());
            mass.push(row["log_importance_weight"].as_f64().unwrap().exp());
            hard.push(h);
        } else {
            assert!(
                row["log_importance_weight"].is_null()
                    && row["clouds"].as_array().unwrap().is_empty()
            );
            mass.push(0.);
            hard.push(0.);
        }
    }
    assert!(zero_hard > 0 && zero_q > 0);
    mass.check(
        shell_reference(inner, capture, qmin, qmax, z),
        "AO shell with second neighbor",
    );
    hard.check(
        shell_reference(inner, capture, qmin, qmax, 0.),
        "hard shell with second neighbor",
    );
    uniform6.check(0.5, "uniform scaled sixth power");
    r2.check(
        0.75 * (R.powi(8) - inner.powi(8)) / (R.powi(6) - inner.powi(6)),
        "shell E r²",
    );
    // The full frozen neighbor list and nonempty radial interval are mandatory.
    for (label, key, bad_value) in [
        (
            "missing-neighbor",
            "physical_fixed_neighbors",
            json!([fixed()]),
        ),
        ("empty-shell", "minimum_mahalanobis_radius", json!(R)),
        ("inverted-q", "maximum_original_q", json!(0.5)),
        (
            "malformed-endpoint",
            "minimum_original_q_inclusive",
            json!("false"),
        ),
    ] {
        let mut bad = region.clone();
        bad[key] = bad_value;
        let path = root.join(format!("{label}.json"));
        fs::write(&path, bad.to_string())?;
        let mut invalid = options.clone();
        invalid.region = path;
        invalid.out = root.join(label);
        assert!(latent_region::run(invalid).is_err());
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
