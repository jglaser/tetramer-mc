//! Shared DockingConfig must not make an integrator silently ignore an explicit
//! conditional target. Missing geometry/model inputs prove rejection is early.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tetramer_mc::{
    latent_region::{self, LatentRegionOptions},
    normalizer::{self, NormalizerOptions},
};

fn fixture() -> Result<(PathBuf, Value)> {
    let root = std::env::temp_dir().join(format!(
        "docking-normalizer-guard-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir_all(&root)?;
    let pose = |x: f64| json!({"position":[x,0.,0.],"orientation":[1.,0.,0.,0.]});
    let metric = json!({"native_poses":[pose(0.)],"rigid_members":[pose(0.)],
        "member_error_scale":2.,"angle_error_scale_deg":15.});
    let cfg = json!({"shape":root.join("missing-shape.json"),"fixed_poses":[pose(-10.)],
        "initial_pose":pose(3.),"capture_center":[0.,0.,0.],"capture_radius":18.,
        "depletant_radius":1.5,"reservoir_density":0.035,"poisson_lambda_ratio":64.,
        "translation_steps":[0.2,2.],"rotation_steps_deg":[1.5,15.],"rotation_probability":0.5,
        "local_attempts_per_cycle":2,"uniform_probability":0.05,"seed":115101010,
        "metadata":metric,"proposal_anchor_index":0,
        "target_region":{"metric":metric,"window":{"minimum":1.,"maximum":2.,
            "lower_inclusive":false,"upper_inclusive":false}}});
    Ok((root, cfg))
}

fn save_config(root: &Path, cfg: &Value) -> Result<PathBuf> {
    let path = root.join("config.json");
    fs::write(&path, serde_json::to_vec(cfg)?)?;
    Ok(path)
}

#[test]
fn absolute_normalizer_rejects_docking_region_before_reading_geometry_or_writing_output()
-> Result<()> {
    let (root, mut cfg) = fixture()?;
    let options = NormalizerOptions {
        config: save_config(&root, &cfg)?,
        model: root.join("missing-model.json"),
        out: root.join("output"),
        samples: 1,
        seed: 17,
        covariance_scale: 1.,
        uniform_probability: None,
        proposal_anchor_index: None,
        cloud_replicates: 1,
        activity: None,
    };
    let error = normalizer::run(options.clone()).unwrap_err().to_string();
    assert!(
        error.contains("basin-normalizer does not implement DockingConfig.target_region"),
        "{error}"
    );
    assert!(!options.out.exists());
    assert_eq!(
        fs::read_dir(&root)?.count(),
        1,
        "Only input config may exist; no probes or outputs"
    );
    // Historical absent/null constraints pass this guard and reach the missing
    // geometry input, without making any physical draws or writing an output.
    for absent in [true, false] {
        if absent {
            cfg.as_object_mut().unwrap().remove("target_region");
        } else {
            cfg["target_region"] = Value::Null;
        }
        save_config(&root, &cfg)?;
        let error = normalizer::run(options.clone()).unwrap_err().to_string();
        assert!(!error.contains("target_region"), "{error}");
        assert!(!options.out.exists());
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn latent_normalizer_rejects_docking_region_before_reading_region_or_writing_output() -> Result<()>
{
    let (root, mut cfg) = fixture()?;
    let options = LatentRegionOptions {
        config: save_config(&root, &cfg)?,
        region: root.join("missing-region.json"),
        out: root.join("output"),
        samples: 1,
        seed: 19,
        cloud_replicates: 1,
        lambda_ratio: 64.,
    };
    let error = latent_region::run(options.clone()).unwrap_err().to_string();
    assert!(
        error.contains("latent-region-normalizer does not implement DockingConfig.target_region"),
        "{error}"
    );
    assert!(!options.out.exists());
    assert_eq!(
        fs::read_dir(&root)?.count(),
        1,
        "Only input config may exist; no probes or outputs"
    );
    for absent in [true, false] {
        if absent {
            cfg.as_object_mut().unwrap().remove("target_region");
        } else {
            cfg["target_region"] = Value::Null;
        }
        save_config(&root, &cfg)?;
        let error = latent_region::run(options.clone()).unwrap_err().to_string();
        assert!(!error.contains("target_region"), "{error}");
        assert!(!options.out.exists());
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
