//! Full-vessel integrals against independent sphere radial quadrature, and
//! atomic (orientation-dependent) wall predicates for a dumbbell.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tetramer_mc::{
    math::*,
    normalizer::{self, NormalizerOptions, NormalizerWall},
    simulation::hash_file,
};

const CENTER: Vec3 = [7., -4., 3.];
const CORE: f64 = 0.3;
const RD: f64 = 0.5;
const WALL: f64 = 3.;

fn fixture(dumbbell: bool) -> Result<PathBuf> {
    let root = std::env::temp_dir().join(format!(
        "wall-normalizer-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir_all(&root)?;
    let atoms = if dumbbell {
        json!([{"center":[-0.7,0.,0.],"radius":CORE},{"center":[0.7,0.,0.],"radius":CORE}])
    } else {
        json!([{"center":[0.,0.,0.],"radius":CORE}])
    };
    fs::write(root.join("shape.json"), json!({"atoms":atoms}).to_string())?;
    let identity = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let fixed = Pose {
        position: CENTER,
        orientation: quaternion(cayley([0.2, -0.1, 0.3])),
    };
    let base = json!({"shape_sha256":hash_file(&root.join("shape.json"))?,"coordinate_convention":"anchor-body-relative",
        "angular_length":1.,"weights":[1.],"anchors":[{"position":[1.,0.,0.],"rotation":IDENTITY}],
        "means":[([0.;6])],"covariances":[std::array::from_fn::<_,6,_>(|i| std::array::from_fn::<_,6,_>(|j|if i==j{1.}else{0.}))]});
    fs::write(root.join("model.json"),json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,"reciprocal_components":[true]}).to_string())?;
    fs::write(root.join("config.json"),json!({"shape":"shape.json","fixed_poses":[fixed],
        "initial_pose":Pose{position:add(CENTER,[1.5,0.,0.]),..identity},
        "capture_center":add(CENTER,[0.2,0.1,0.]),"capture_radius":5.,
        "depletant_radius":RD,"reservoir_density":0.4,"poisson_lambda_ratio":16.,
        "translation_steps":[0.1],"rotation_steps_deg":[5.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.1,"seed":441,
        "endpoint_gate":{"max_cells":63,"max_depth":8,"min_width":0.1},
        "metadata":{"native_poses":[fixed],"rigid_members":[identity],"member_error_scale":3.,"angle_error_scale_deg":90.}}).to_string())?;
    Ok(root)
}

fn options(root: &Path, out: &str, samples: u64, activity: f64) -> NormalizerOptions {
    NormalizerOptions {
        config: root.join("config.json"),
        model: root.join("model.json"),
        out: root.join(out),
        samples,
        seed: 817913,
        covariance_scale: 1.,
        uniform_probability: Some(0.4),
        proposal_anchor_index: None,
        cloud_replicates: 2,
        activity: Some(activity),
    }
}

fn reference(activity: f64) -> f64 {
    // Fixed and moving sphere centers must be separated by 2*CORE, while
    // every moving atom fits iff its center is within WALL-CORE.
    let a = 2. * CORE;
    let b = WALL - CORE;
    let n = 16384;
    let h = (b - a) / n as f64;
    let f = |r: f64| {
        let e = CORE + RD;
        let overlap = if r < 2. * e {
            PI * (4. * e + r) * (2. * e - r).powi(2) / 12.
        } else {
            0.
        };
        4. * PI * r * r * (activity * overlap).exp()
    };
    (f(a)
        + f(b)
        + (1..n)
            .map(|i| if i % 2 == 0 { 2. } else { 4. } * f(a + h * i as f64))
            .sum::<f64>())
        * h
        / 3.
}

#[test]
fn full_wall_spheres_match_radial_depletion_and_haar_limits() -> Result<()> {
    let root = fixture(false)?;
    for (index, activity) in [0., 0.4].into_iter().enumerate() {
        let name = format!("sphere-{index}");
        let mut opt = options(&root, &name, 24000, activity);
        opt.seed += index as u64;
        let result = normalizer::run_with_wall(
            opt,
            NormalizerWall {
                center: CENTER,
                radius: WALL,
            },
        )?;
        assert_eq!(result["manifest"]["schema"], 4);
        assert_eq!(result["manifest"]["pose_proposal_schema"], 3);
        assert_eq!(result["manifest"]["bath_wall_permeable"], true);
        let mut sum = 0.;
        let mut sq = 0.;
        let mut angular = 0.;
        let mut angular_sq = 0.;
        let mut rejected = 0;
        for line in fs::read_to_string(root.join(&name).join("samples.jsonl"))?.lines() {
            let row: Value = serde_json::from_str(line)?;
            let p: Pose = serde_json::from_value(row["pose"].clone())?;
            let expected_wall = norm(sub(p.position, CENTER)) <= WALL - CORE;
            assert_eq!(row["wall_valid"], expected_wall);
            if !expected_wall {
                assert_eq!(row["hard_valid"], false);
                assert!(row["log_importance_weight"].is_null());
                if row["capture_valid"] == true {
                    rejected += 1;
                }
            }
            let w = row["log_importance_weight"].as_f64().map_or(0., f64::exp);
            sum += w;
            sq += w * w;
            let v = w * rotation(p.orientation)[0][0];
            angular += v;
            angular_sq += v * v;
        }
        let n = 24000.;
        let mean = sum / n;
        let se = ((sq / n - mean * mean) / (n - 1.)).sqrt();
        let expected = reference(activity);
        eprintln!("wall sphere z={activity}: {mean} +/- {se}, radial reference {expected}");
        assert!((mean - expected).abs() < 6.5 * se);
        let angular_mean = angular / n;
        let angular_se = ((angular_sq / n - angular_mean * angular_mean) / (n - 1.)).sqrt();
        assert!(angular_mean.abs() < 6.5 * angular_se);
        assert!(rejected > 0);
        assert_eq!(result["wall_rejected"], rejected);
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn full_wall_checks_rotated_atomic_union_not_only_body_bound() -> Result<()> {
    let root = fixture(true)?;
    normalizer::run_with_wall(
        options(&root, "dumbbell", 4096, 0.),
        NormalizerWall {
            center: CENTER,
            radius: WALL,
        },
    )?;
    let mut beyond_bound = 0;
    let mut rejected = 0;
    for line in fs::read_to_string(root.join("dumbbell/samples.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let p: Pose = serde_json::from_value(row["pose"].clone())?;
        let valid = [[-0.7, 0., 0.], [0.7, 0., 0.]]
            .into_iter()
            .all(|a| norm(sub(p.apply(a), CENTER)) + CORE <= WALL);
        assert_eq!(row["wall_valid"], valid);
        if valid && norm(sub(p.position, CENTER)) + 1. > WALL {
            beyond_bound += 1;
        }
        if !valid {
            rejected += 1;
            assert_eq!(row["hard_valid"], false);
        }
    }
    assert!(beyond_bound > 0);
    assert!(rejected > 0);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn near_wall_sphere_retains_depletion_outside_the_protein_wall() -> Result<()> {
    let root = fixture(false)?;
    let path = root.join("config.json");
    let mut cfg: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let offset = 2.6;
    cfg["fixed_poses"][0] = json!(Pose {
        position: add(CENTER, [offset, 0., 0.]),
        orientation: [1., 0., 0., 0.]
    });
    cfg["initial_pose"] = json!(Pose {
        position: add(CENTER, [offset, 0.7, 0.]),
        orientation: [1., 0., 0., 0.]
    });
    fs::write(&path, cfg.to_string())?;
    // This point is outside the protein wall, but inside both exclusions at
    // a hard-valid, wall-valid pair placement. A clipped bath changes its weight.
    let witness = [3.1, 0.35, 0.];
    assert!(norm(witness) > WALL);
    assert!(norm(sub(witness, [offset, 0., 0.])) < CORE + RD);
    assert!(norm(sub(witness, [offset, 0.7, 0.])) < CORE + RD);
    assert!(norm([offset, 0.7, 0.]) + CORE < WALL);
    let z = 3.;
    normalizer::run_with_wall(
        options(&root, "near-wall", 24000, z),
        NormalizerWall {
            center: CENTER,
            radius: WALL,
        },
    )?;
    let mut sums = [0.; 2];
    let mut squares = [0.; 2];
    for line in fs::read_to_string(root.join("near-wall/samples.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let w = row["log_importance_weight"].as_f64().map_or(0., f64::exp);
        for (i, value) in [
            w,
            if row["depletion_contact"] == true {
                w
            } else {
                0.
            },
        ]
        .into_iter()
        .enumerate()
        {
            sums[i] += value;
            squares[i] += value * value;
        }
    }
    // Radial coordinate is now relative to the off-center fixed sphere.
    // The allowed solid-angle fraction follows from the cosine law for the
    // moving center inside radius WALL-CORE. The depletion lens is uncut.
    let inner = WALL - CORE;
    for (i, upper) in [inner + offset, 2. * (CORE + RD)].into_iter().enumerate() {
        let lower = 2. * CORE;
        let n = 32768;
        let h = (upper - lower) / n as f64;
        let f = |r: f64| {
            let fraction = ((1. + (inner * inner - offset * offset - r * r) / (2. * offset * r))
                / 2.)
                .clamp(0., 1.);
            let e = CORE + RD;
            let lens = if r < 2. * e {
                PI * (4. * e + r) * (2. * e - r).powi(2) / 12.
            } else {
                0.
            };
            4. * PI * r * r * fraction * (z * lens).exp()
        };
        let expected = (f(lower)
            + f(upper)
            + (1..n)
                .map(|k| if k % 2 == 0 { 2. } else { 4. } * f(lower + h * k as f64))
                .sum::<f64>())
            * h
            / 3.;
        let count = 24000.;
        let mean = sums[i] / count;
        let se = ((squares[i] / count - mean * mean) / (count - 1.)).sqrt();
        eprintln!("near-wall region {i}: {mean} +/- {se}, uncut-bath reference {expected}");
        assert!((mean - expected).abs() < 6.5 * se);
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn full_wall_refuses_truncated_capture_and_outside_fixed_neighbor_before_output() -> Result<()> {
    let root = fixture(false)?;
    let path = root.join("config.json");
    let original: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let mut cfg = original.clone();
    cfg["capture_radius"] = json!(2.);
    fs::write(&path, cfg.to_string())?;
    let error = normalizer::run_with_wall(
        options(&root, "bad-support", 1, 0.),
        NormalizerWall {
            center: CENTER,
            radius: WALL,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("enclose the full atomic-wall"));
    assert!(!root.join("bad-support").exists());
    cfg = original;
    cfg["fixed_poses"][0]["position"] = json!(add(CENTER, [3., 0., 0.]));
    fs::write(&path, cfg.to_string())?;
    let error = normalizer::run_with_wall(
        options(&root, "bad-fixed", 1, 0.),
        NormalizerWall {
            center: CENTER,
            radius: WALL,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("Fixed neighbor 0 is outside"));
    assert!(!root.join("bad-fixed").exists());
    fs::remove_dir_all(root)?;
    Ok(())
}
