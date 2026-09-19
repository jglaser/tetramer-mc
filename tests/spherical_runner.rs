//! Runner-level boundary, proposal-density, scheduling, and continuation checks.
use anyhow::Result;
use hoomd_gsd::file_layer::{GsdFile, Mode};
use rand::{SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::{fs, path::Path};
use tetramer_mc::{
    math::{IDENTITY, Pose},
    proposal::FrozenRelativePoseProposal,
    simulation::{Method, RunOptions, hash_file, run},
};

fn model(hash: &str, mean: f64, variance: f64) -> Value {
    let covariance: Vec<Vec<f64>> = (0..6)
        .map(|i| (0..6).map(|j| if i == j { variance } else { 0. }).collect())
        .collect();
    json!({"angular_length":1.,"anchors":[{"position":[0.,0.,0.],"rotation":IDENTITY}],"means":[[mean,0.,0.,0.,0.,0.]],"covariances":[covariance],"weights":[1.],"shape_sha256":hash,"coordinate_convention":"anchor-body-relative"})
}

#[test]
fn open_gaussian_uses_ordinary_relative_displacements_and_no_wrapping() -> Result<()> {
    let hash = "0".repeat(64);
    let text = model(&hash, 8., 0.01).to_string();
    let proposal = FrozenRelativePoseProposal::from_json_str_open(&text, [10.; 3], 0.1, &hash)?;
    let old = Pose {
        position: [1., 0., 0.],
        orientation: [1., 0., 0., 0.],
    };
    let anchor = Pose {
        position: [-1., 0., 0.],
        ..old
    };
    let near = Pose {
        position: [7., 0., 0.],
        ..old
    };
    let far = Pose {
        position: [-3., 0., 0.],
        ..old
    };
    assert!(!proposal.is_periodic());
    assert!(proposal.log_density(&near, &anchor)? > proposal.log_density(&far, &anchor)? + 5.);
    let expected = 0.9_f64.ln() + proposal.relative_log_density([8., 0., 0.], IDENTITY)?;
    assert!(
        (proposal.log_density(&near, &anchor)? - expected).abs() < 1e-12,
        "outside cube only learned density remains"
    );
    let mut rng = StdRng::seed_from_u64(912);
    let mut outside = 0;
    for _ in 0..1000 {
        let draw = proposal.propose(&mut rng, &[old, anchor], 0)?;
        let candidate = draw
            .candidate
            .expect("open Gaussian draws are not box nulls");
        if candidate.position[0] >= 5. {
            outside += 1;
        }
        let q_old = proposal.log_density(&old, &anchor)?;
        let q_new = proposal.log_density(&candidate, &anchor)?;
        assert!((draw.log_reverse_forward.unwrap() - (q_old - q_new)).abs() < 1e-12);
    }
    assert!(
        outside > 800,
        "learned centers must remain outside the cube, pending physical wall rejection"
    );
    Ok(())
}

fn options(
    root: &Path,
    name: &str,
    method: Method,
    sweeps: u64,
    resume: Option<&str>,
) -> RunOptions {
    RunOptions {
        config: root.join("config.json"),
        model: (method == Method::Learned).then(|| root.join("model.json")),
        method,
        out: root.join(name),
        sweeps,
        sample_every: 2,
        resume: resume.map(|s| root.join(s).join("checkpoint.json")),
        write_gsd: true,
        record_moves: true,
    }
}

#[test]
fn spherical_collective_schedules_resume_exactly_and_record_unwrapped_coordinates() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "tetramer-mc-spherical-runner-{}",
        std::process::id()
    ));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fs::create_dir(&root)?;
    fs::write(
        root.join("shape.json"),
        json!({"name":"sphere","atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string(),
    )?;
    let mut config = json!({"shape":"shape.json","box_lengths":[12.,12.,12.],"boundary":{"kind":"spherical","radius":6.},"seed":8819,"depletant_radius":0.4,"reservoir_density":0.08,"poisson_lambda_ratio":4.,"endpoint_gate":{"max_cells":63,"max_depth":6,"min_width":0.25},"gca_probability":1.,"center_shift_probability":1.,"initial_poses":[{"position":[-2.2,0.,0.],"orientation":[1.,0.,0.,0.]},{"position":[0.,0.1,0.],"orientation":[1.,0.,0.,0.]},{"position":[2.2,0.,0.],"orientation":[1.,0.,0.,0.]}]});
    fs::write(
        root.join("model.json"),
        model(&hash_file(&root.join("shape.json"))?, 2.3, 0.3).to_string(),
    )?;
    for (gca, shift, label) in [(1., 0., "gca"), (0., 1., "shift"), (1., 1., "both")] {
        config["gca_probability"] = json!(gca);
        config["center_shift_probability"] = json!(shift);
        fs::write(root.join("config.json"), config.to_string())?;
        for method in [Method::Learned, Method::LocalUniform] {
            let name = format!("{label}-{method:?}");
            let full = format!("{name}-full");
            let part = format!("{name}-part");
            let resumed = format!("{name}-resume");
            let summary = run(options(&root, &full, method, 8, None))?;
            run(options(&root, &part, method, 4, None))?;
            run(options(&root, &resumed, method, 8, Some(&part)))?;
            let read = |folder: &str| -> Result<Value> {
                Ok(serde_json::from_slice(&fs::read(
                    root.join(folder).join("checkpoint.json"),
                )?)?)
            };
            assert_eq!(read(&full)?, read(&resumed)?, "restart differs for {name}");
            assert_eq!(
                summary["counts"]["gca"]["completed"],
                json!((8. * gca) as u64)
            );
            assert_eq!(
                summary["counts"]["center_shift"]["completed"],
                json!((8. * shift) as u64)
            );
            assert_eq!(summary["boundary"], "spherical");
            assert_eq!(summary["bath_wall_permeable"], true);
            let frames: Vec<Value> = fs::read_to_string(root.join(&full).join("trajectory.jsonl"))?
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
            assert_eq!(
                frames[0]["poses"][0]["position"][0], -2.2,
                "spherical positions must not wrap"
            );
            for frame in &frames {
                for p in frame["poses"].as_array().unwrap() {
                    let pose: Pose = serde_json::from_value(p.clone())?;
                    assert!(pose.position.iter().map(|x| x * x).sum::<f64>() <= 25. + 1e-12);
                }
            }
            let gsd = GsdFile::open(&root.join(&full).join("trajectory.gsd"), Mode::Read)?;
            let periodic: Vec<u8> = gsd.iter_scalars(0, "log/tetramer_mc/periodic")?.collect();
            assert_eq!(periodic, vec![0, 0, 0]);
            let positions: Vec<[f32; 3]> = gsd.iter_arrays(0, "particles/position")?.collect();
            assert!((positions[0][0] + 2.2).abs() < 1e-6);
            let images: Vec<[i32; 3]> = gsd.iter_arrays(0, "particles/image")?.collect();
            assert!(images.iter().all(|x| *x == [0; 3]));
        }
    }
    config["initial_poses"][0]["position"] = json!([5.1, 0., 0.]);
    fs::write(root.join("config.json"), config.to_string())?;
    assert!(
        run(options(
            &root,
            "outside-wall",
            Method::LocalUniform,
            2,
            None
        ))
        .is_err()
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
