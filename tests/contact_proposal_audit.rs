use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::{fs, process::Command};
use tetramer_mc::{math::*, proposal::FrozenRelativePoseProposal, simulation::hash_bytes};

#[test]
fn neighborhood_score_matches_disjoint_two_neighbor_lenses() -> Result<()> {
    use tetramer_mc::{
        contact_discovery::estimate_environment_overlap,
        depletion::GateOptions,
        geometry::{Atom, Environment, Placed, Shape, SphereTree},
    };
    let tree = SphereTree::new(Shape {
        name: "sphere".into(),
        volume: 4. * std::f64::consts::PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })?;
    let at = |x| Pose {
        position: [x, 0., 0.],
        orientation: [1., 0., 0., 0.],
    };
    let env = Environment {
        tree: &tree,
        fixed: vec![Placed::new(at(0.)), Placed::new(at(4.4))],
        labels: vec![],
        rd: 0.4,
    };
    let result = estimate_environment_overlap(
        &env,
        at(2.2),
        100_000,
        719,
        GateOptions {
            max_cells: 511,
            ..Default::default()
        },
    )?;
    let a = 1.4_f64;
    let d = 2.2_f64;
    let exact = 2. * std::f64::consts::PI * (4. * a + d) * (2. * a - d).powi(2) / 12.;
    assert!((result.volume - exact).abs() < 6. * result.standard_error + 1e-9);
    Ok(())
}

#[test]
fn repeated_exclusion_region_is_counted_once() -> Result<()> {
    use tetramer_mc::{
        contact_discovery::estimate_environment_overlap,
        depletion::GateOptions,
        geometry::{Atom, Environment, Placed, Shape, SphereTree},
    };
    let tree = SphereTree::new(Shape {
        name: "sphere".into(),
        volume: 4. * std::f64::consts::PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })?;
    let at = |x| Pose {
        position: [x, 0., 0.],
        orientation: [1., 0., 0., 0.],
    };
    let one = Environment {
        tree: &tree,
        fixed: vec![Placed::new(at(0.))],
        labels: vec![],
        rd: 0.4,
    };
    // A diagnostic union predicate must also be invariant to duplicated regions.
    // The duplicated scaffold is not used as a physical configuration.
    let repeated = Environment {
        tree: &tree,
        fixed: vec![Placed::new(at(0.)), Placed::new(at(0.))],
        labels: vec![],
        rd: 0.4,
    };
    let opts = GateOptions {
        max_cells: 511,
        ..Default::default()
    };
    let a = estimate_environment_overlap(&one, at(2.2), 10_000, 720, opts)?;
    let b = estimate_environment_overlap(&repeated, at(2.2), 10_000, 720, opts)?;
    assert_eq!(a.volume, b.volume);
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.uncertain_volume, b.uncertain_volume);
    Ok(())
}

fn model(hash: &str, reciprocal: bool) -> Value {
    let covariance: [[f64; 6]; 6] =
        std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1e-8 } else { 0. }));
    let base = json!({"schema":"weighted-pose-mixture-v1","coordinate_convention":"anchor-body-relative",
        "shape_sha256":hash,"angular_length":2.,"anchors":[{"position":[3.,0.,0.],"rotation":IDENTITY}],
        "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance],"weights":[1.]});
    if reciprocal {
        json!({"schema":"reciprocal-pose-mixture-v1","base_model":base,"reciprocal_components":[true]})
    } else {
        base
    }
}

#[test]
fn conditional_branch_samples_have_exact_reciprocal_geometry() -> Result<()> {
    let hash = "a".repeat(64);
    let m = FrozenRelativePoseProposal::from_json_str_open(
        &model(&hash, true).to_string(),
        [100.; 3],
        0.1,
        &hash,
    )?;
    for seed in 0..32 {
        let a = m
            .draw_relative_branch(&mut StdRng::seed_from_u64(seed), 0, false)?
            .unwrap();
        let b = m
            .draw_relative_branch(&mut StdRng::seed_from_u64(seed), 0, true)?
            .unwrap();
        let inverse = transpose(rotation(a.orientation));
        let t = matvec(inverse, a.position).map(|x| -x);
        assert!(norm(sub(t, b.position)) < 1e-12);
        let br = rotation(b.orientation);
        for i in 0..3 {
            for j in 0..3 {
                assert!((inverse[i][j] - br[i][j]).abs() < 1e-12);
            }
        }
        assert!(
            (m.relative_log_density(a.position, rotation(a.orientation))?
                - m.relative_log_density(b.position, br)?)
            .abs()
                < 1e-9
        );
    }
    Ok(())
}

#[test]
fn audit_separates_pair_from_neighbor_obstruction_and_preserves_all_draws() -> Result<()> {
    struct TestDirectory(std::path::PathBuf);
    impl TestDirectory {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let temp = TestDirectory(std::env::temp_dir().join(
        format!("contact-proposal-audit-{}-{}",std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()),
    ));
    fs::create_dir(temp.path())?;
    let shape=json!({"name":"audit sphere","volume":4.*std::f64::consts::PI/3.,"atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string();
    fs::write(temp.path().join("shape.json"), &shape)?;
    fs::write(
        temp.path().join("model.json"),
        model(&hash_bytes(shape.as_bytes()), false).to_string(),
    )?;
    fs::write(
        temp.path().join("fixed.json"),
        json!([
            {"position":[0.,0.,0.],"orientation":[1.,0.,0.,0.]},
            {"position":[3.5,0.,0.],"orientation":[1.,0.,0.,0.]}
        ])
        .to_string(),
    )?;
    let out = temp.path().join("audit");
    let result = Command::new(env!("CARGO_BIN_EXE_contact-proposal-audit"))
        .args([
            "--shape",
            temp.path().join("shape.json").to_str().unwrap(),
            "--model",
            temp.path().join("model.json").to_str().unwrap(),
            "--fixed-poses",
            temp.path().join("fixed.json").to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--draws-per-component",
            "64",
        ])
        .output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let s: Value = serde_json::from_slice(&fs::read(out.join("summary.json"))?)?;
    assert_eq!(s["weighted_pair_hard_valid"], 1.);
    assert_eq!(s["weighted_all_neighbors_hard_valid"], 0.);
    assert_eq!(
        fs::read_to_string(out.join("draws.jsonl"))?.lines().count(),
        64
    );
    Ok(())
}
