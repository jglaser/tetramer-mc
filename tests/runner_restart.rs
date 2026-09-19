//! Full-kernel continuation, GSD I/O, and mismatched-model safeguards.
use anyhow::Result;
use serde_json::{Value, json};
use std::{fs, path::Path};
use tetramer_mc::simulation::{Method, RunOptions, hash_file, run};

fn options(
    root: &Path,
    name: &str,
    method: Method,
    sweeps: u64,
    resume: Option<&str>,
) -> RunOptions {
    RunOptions {
        config: root.join("config.json"),
        model: if method == Method::Learned {
            Some(root.join("model.json"))
        } else {
            None
        },
        method,
        out: root.join(name),
        sweeps,
        sample_every: 2,
        resume: resume.map(|name| root.join(name).join("checkpoint.json")),
        write_gsd: true,
        record_moves: true,
    }
}

#[test]
fn checkpoint_continuation_is_exact_and_inputs_checked() -> Result<()> {
    let root = std::env::temp_dir().join(format!("tetramer-mc-restart-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fs::create_dir(&root)?;
    fs::write(root.join("shape.json"),json!({"name":"sphere","volume":4.188790204786391,"atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string())?;
    fs::write(root.join("config.json"),json!({"shape":"shape.json","box_lengths":[14.,14.,14.],"seed":8114,"depletant_radius":0.6,"reservoir_density":0.03,"poisson_lambda_ratio":4.,"endpoint_gate":{"max_cells":63,"max_depth":6,"min_width":0.25},"initial_poses":[{"position":[4.,4.,4.],"orientation":[1.,0.,0.,0.]},{"position":[6.2,4.,4.],"orientation":[1.,0.,0.,0.]},{"position":[8.4,4.3,4.],"orientation":[1.,0.,0.,0.]}]}).to_string())?;
    let covariance: Vec<Vec<f64>> = (0..6)
        .map(|i| (0..6).map(|j| if i == j { 1. } else { 0. }).collect())
        .collect();
    let mut model = json!({"angular_length":1.,"anchors":[{"position":[0.,0.,0.],"rotation":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}],"means":[[2.4,0.,0.,0.,0.,0.]],"covariances":[covariance],"shape_sha256":hash_file(&root.join("shape.json"))?,"coordinate_convention":"anchor-body-relative"});
    fs::write(root.join("model.json"), model.to_string())?;
    for method in [Method::Learned, Method::LocalUniform] {
        let name = if method == Method::Learned {
            "learned"
        } else {
            "uniform"
        };
        let a = format!("{name}-full");
        let b = format!("{name}-part");
        let c = format!("{name}-resumed");
        run(options(&root, &a, method, 12, None))?;
        run(options(&root, &b, method, 4, None))?;
        run(options(&root, &c, method, 12, Some(&b)))?;
        let full: Value =
            serde_json::from_slice(&fs::read(root.join(&a).join("checkpoint.json"))?)?;
        let resumed: Value =
            serde_json::from_slice(&fs::read(root.join(&c).join("checkpoint.json"))?)?;
        assert_eq!(full, resumed, "checkpoint differs after resumed {name} run");
        let parse_moves = |folder: &str| -> Result<Vec<Value>> {
            Ok(fs::read_to_string(root.join(folder).join("moves.jsonl"))?
                .lines()
                .map(|line| {
                    let mut v: Value = serde_json::from_str(line).unwrap();
                    v.as_object_mut().unwrap().remove("sampler_cpu_seconds");
                    v
                })
                .collect())
        };
        let all = parse_moves(&a)?;
        let tail = parse_moves(&c)?;
        assert_eq!(&all[12..], tail.as_slice());
        let summary: Value =
            serde_json::from_slice(&fs::read(root.join(&c).join("summary.json"))?)?;
        assert_eq!(
            summary["segment_counts"]["selected_body_updates"],
            json!(24)
        );
        assert_eq!(
            summary["initial_counts"]["selected_body_updates"],
            json!(12)
        );
        assert!(
            run(options(&root, &a, method, 12, None)).is_err(),
            "must not overwrite an existing run"
        );
    }
    let checkpoint_path = root.join("uniform-part/checkpoint.json");
    let mut legacy: Value = serde_json::from_slice(&fs::read(&checkpoint_path)?)?;
    legacy.as_object_mut().unwrap().remove("shape_sha256");
    fs::write(&checkpoint_path, legacy.to_string())?;
    run(options(
        &root,
        "legacy-resumed",
        Method::LocalUniform,
        12,
        Some("uniform-part"),
    ))?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("uniform-full/checkpoint.json"))?)?;
    let recovered: Value =
        serde_json::from_slice(&fs::read(root.join("legacy-resumed/checkpoint.json"))?)?;
    assert_eq!(
        expected, recovered,
        "provenance-checked legacy continuation differs"
    );
    model["shape_sha256"] = json!("bad hash");
    fs::write(root.join("model.json"), model.to_string())?;
    assert!(run(options(&root, "invalid-model", Method::Learned, 2, None)).is_err());
    let original_shape = fs::read(root.join("shape.json"))?;
    let mut changed: Value = serde_json::from_slice(&original_shape)?;
    changed["atoms"][0]["radius"] = json!(0.9);
    fs::write(root.join("shape.json"), changed.to_string())?;
    assert!(
        run(options(
            &root,
            "changed-shape",
            Method::LocalUniform,
            12,
            Some("uniform-part")
        ))
        .is_err(),
        "baseline restart must reject changed physical shape"
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
