//! Proposal-only sphere fixtures for the native-class world-pose bridge.
//! No clouds, physical weights, protein inputs, or production sampling.
use super::*;
use crate::geometry::Shape;
use rand::{RngExt, SeedableRng};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static SERIAL: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    directory: PathBuf,
    region: Vec<u8>,
    guide: Value,
    cfg: DockingConfig,
    tree: SphereTree,
    shape_hash: String,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}
impl Fixture {
    fn new(alpha: f64, beta: f64, axes: &[usize], transformed: bool) -> Result<Self> {
        let directory = std::env::temp_dir().join(format!(
            "physical-class-line-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        ensure!(!directory.exists(), "Fresh proposal fixture required");
        fs::create_dir(&directory)?;
        let shape_raw = serde_json::to_vec(&json!({"atoms":[{"center":[0.,0.,0.],"radius":0.3}]}))?;
        fs::write(directory.join("shape.json"), &shape_raw)?;
        let shape_hash = hash_bytes(&shape_raw);
        let fixed = Pose {
            position: if transformed { [3., -2., 1.] } else { [0.; 3] },
            orientation: if transformed {
                quaternion(cayley([0.2, -0.3, 0.1]))
            } else {
                [1., 0., 0., 0.]
            },
        };
        let eye: [[f64; 6]; 6] =
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
        let mut lower = eye;
        if transformed {
            lower[0][0] = 1.3;
            lower[1][0] = 0.2;
            lower[3][0] = 0.2;
            lower[4][1] = -0.1;
        }
        let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
        });
        let region = serde_json::to_vec(&json!({"shape_sha256":shape_hash,"fixed_neighbor":fixed,
            "physical_fixed_neighbors":[fixed],"capture_center":fixed.position,"capture_radius":1.5,
            "mahalanobis_radius":4.,"minimum_original_q":0.,
            "gaussian_chart":{"shape_sha256":shape_hash,"coordinate_convention":"anchor-body-relative",
            "angular_length":1.2,"anchors":[{"position":[0.,0.,0.],"rotation":IDENTITY}],
            "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance],"weights":[1.]}}))?;
        let native = json!({"schema":"native-entry-compiled-v1","source_definition_sha256":"0".repeat(64),
            "source_input_sha256":{"tetramer-shape.json":shape_hash},
            "criteria":{"body_member_position_entry_A":2.,"body_orientation_entry_deg":15.,
            "monomer_position_entry_A":3.,"monomer_orientation_entry_deg":20.,"contact_entry_A":2.,
            "native_reference_patch_gap_A":1.,"minimum_shared_native_residue_pairs":1.,
            "hard_overlap_tolerance_A":1e-8,"catalogue_cycle_position_tolerance_A":1e-6,
            "catalogue_cycle_angle_tolerance_deg":1e-6},"fixed_poses":[fixed],
            "members":[{"position":[0.,0.,0.],"rotation":IDENTITY}],
            "monomer_atoms":[{"center":[0.,0.,0.],"radius":0.3,"residue":0}],"residue_count":1,
            "references":[{"label":"A","family":"A","position":[0.8,0.,0.],"rotation":IDENTITY,"native_residue_pairs":[0]}],
            "motifs":[{"id":7,"position":[0.8,0.,0.],"rotation":IDENTITY,
            "member_contacts":[{"member_i":0,"member_j":0,"directed_class":"A"}]}]});
        let native_raw = serde_json::to_vec(&native)?;
        fs::write(directory.join("native.json"), &native_raw)?;
        let guide = json!({"schema":"defensive-native-class-line-guide-v1","region_sha256":hash_bytes(&region),
            "defensive_uniform_shell_probability":alpha,"conditional_probability":beta,"minimum_conditional_mass":1e-12,
            "raw_translation_axes":axes,"gaussian_components":[
            {"weight":0.4,"mean":[0.,0.,0.,0.,0.,0.],"covariance":eye},
            {"weight":0.6,"mean":[4.5,0.,0.,0.,0.,0.],"covariance":eye}],
            "class_channels":[{"class":"hard_free","probability":0.2},
            {"class":"native","probability":0.3},{"class":"contact_without_native","probability":0.4},
            {"class":"native","probability":0.1,"orthant":63}],
            "compiled_native":{"path":directory.join("native.json"),"sha256":hash_bytes(&native_raw)},
            "shape_sha256":shape_hash,"fixed_poses":[fixed],"capture_center":fixed.position,
            "capture_radius":1.5,"depletant_radius":0.4});
        let cfg = serde_json::from_value(
            json!({"shape":directory.join("shape.json"),"fixed_poses":[fixed],
            "initial_pose":fixed,"capture_center":[0.,0.,0.],"capture_radius":20.,"depletant_radius":0.4,
            "reservoir_density":2.,"poisson_lambda_ratio":64.,"translation_steps":[0.1],"rotation_steps_deg":[1.],
            "rotation_probability":0.5,"local_attempts_per_cycle":1,"uniform_probability":0.1,"seed":1}),
        )?;
        Ok(Self {
            directory,
            region,
            guide,
            cfg,
            tree: SphereTree::new(serde_json::from_slice::<Shape>(&shape_raw)?)?,
            shape_hash,
        })
    }
    fn parse(&self) -> Result<PhysicalLatentGuide> {
        self.parse_spec(&self.guide)
    }
    fn parse_spec(&self, guide: &Value) -> Result<PhysicalLatentGuide> {
        PhysicalLatentGuide::from_bytes_with_geometry(
            &self.region,
            &serde_json::to_vec(guide)?,
            &self.shape_hash,
            &self.cfg,
            &self.tree,
        )
    }
    fn gaussian(&self) -> Result<PhysicalLatentGuide> {
        let mut g = self.guide.clone();
        g["schema"] = json!("defensive-latent-shell-guide-v1");
        for key in [
            "conditional_probability",
            "minimum_conditional_mass",
            "raw_translation_axes",
            "class_channels",
            "compiled_native",
            "shape_sha256",
            "fixed_poses",
            "capture_center",
            "capture_radius",
            "depletant_radius",
        ] {
            g.as_object_mut().unwrap().remove(key);
        }
        PhysicalLatentGuide::from_bytes(&self.region, &serde_json::to_vec(&g)?, &self.shape_hash)
    }
}
fn near(a: f64, b: f64) {
    assert!(
        a == b || (a - b).abs() <= 2e-11 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

#[test]
fn physical_class_line_source_capture_and_target_context_are_separate() -> Result<()> {
    let mut f = Fixture::new(0.5, 1., &[0, 1, 2], true)?;
    assert!(
        PhysicalLatentGuide::from_bytes(&f.region, &serde_json::to_vec(&f.guide)?, &f.shape_hash)
            .is_err()
    );
    let a = f.parse()?;
    assert!(a.is_native_class_line() && !a.is_hard_free_line());
    f.cfg.capture_center = [-100., 20., 30.];
    f.cfg.capture_radius = 1000.;
    let b = f.parse()?;
    assert_eq!(a.reference_capture(), ([3., -2., 1.], 1.5));
    for u in [
        [0.; 6],
        [1., 0.1, 0., 0., 0., 0.],
        [5., 0., 0., 0., 0., 0.],
        [0., 0., 0., 5., 0., 0.],
    ] {
        let (pose, j) = a.decode(u)?;
        let x = a.evaluate(pose)?;
        let y = b.evaluate(pose)?;
        assert_eq!(x.log_physical_density, y.log_physical_density);
        assert_eq!(x.native_class_line_density, y.native_class_line_density);
        assert!(x.hard_free_line_density.is_none());
        if !x.structural_zero {
            near(x.log_physical_density, x.log_latent_density.unwrap() - j);
        }
    }
    f.cfg.fixed_poses[0].position[0] += 1.;
    assert!(f.parse().is_err());
    Ok(())
}

#[test]
fn physical_class_line_support_zero_fallback_tail_and_exact_seam() -> Result<()> {
    let f = Fixture::new(0.5, 1., &[0], false)?;
    let g = f.parse()?;
    let zero = g.evaluate(g.decode([5., 0., 0., 0., 0., 0.])?.0)?;
    assert!(zero.structural_zero && zero.log_physical_density == f64::NEG_INFINITY);
    assert!(zero.latent.is_some());
    near(
        half_mixture_log_density(-3., zero.log_physical_density)?,
        -3. - 2_f64.ln(),
    );
    let tail = g.evaluate(g.decode([0., 0., 0., 5., 0., 0.])?.0)?;
    assert!(!tail.structural_zero && tail.log_physical_density.is_finite());
    near(
        tail.log_physical_density,
        f.gaussian()?
            .log_density(g.decode([0., 0., 0., 5., 0., 0.])?.0)?,
    );
    let mut native_only = f.guide.clone();
    native_only["class_channels"] = json!([{"class":"native","probability":1.}]);
    let native_only = f.parse_spec(&native_only)?;
    // Orientation misses the native angular cap while the source hard-free
    // line has positive mass, so this must use the intermediate fallback.
    let u = [1., 0., 0., 0.4, 0., 0.];
    if let PhysicalGuideLaw::NativeClassLine(line) = &native_only.guide {
        let (_, detail) =
            line.density_details(u, true, native_only.log_volume, &native_only.chart, 4.)?;
        for component in detail["axes"][0]["components"].as_array().unwrap() {
            assert_eq!(component["channels"][0]["fallback_target"], "hard_free");
        }
    } else {
        unreachable!();
    }
    assert!(
        native_only
            .evaluate(native_only.decode(u)?.0)?
            .log_physical_density
            .is_finite()
    );
    for scalar in [1e-6_f64, 1e-12, 1e-50] {
        assert!(
            g.evaluate(Pose {
                position: [2., 0., 0.],
                orientation: [scalar, (1. - scalar * scalar).sqrt(), 0., 0.]
            })?
            .log_physical_density
            .is_finite()
        );
    }
    let unrepresentable = Pose {
        position: [2., 0., 0.],
        orientation: [1e-200, 1., 0., 0.],
    };
    assert!(g.evaluate(unrepresentable).is_err());
    let seam = g.evaluate(Pose {
        orientation: [0., 1., 0., 0.],
        ..unrepresentable
    })?;
    assert!(
        seam.structural_zero && seam.latent.is_none() && seam.native_class_line_density.is_none()
    );
    Ok(())
}

#[test]
fn physical_class_line_disabled_and_uniform_preserve_gaussian_rng() -> Result<()> {
    for (alpha, beta) in [(0.5, 0.), (1., 1.)] {
        let f = Fixture::new(alpha, beta, &[0, 1, 2], true)?;
        let new = f.parse()?;
        let old = f.gaussian()?;
        let mut a = StdRng::seed_from_u64(6100407101);
        let mut b = StdRng::seed_from_u64(6100407101);
        for _ in 0..64 {
            let x = new.draw_only(&mut a)?;
            let y = old.draw(&mut b)?;
            assert_eq!(x.latent, y.latent);
            assert_eq!(x.pose, y.pose);
            assert_eq!(x.gaussian_component, y.gaussian_component);
            assert!(x.native_class_line_draw.is_some() && x.hard_free_line_draw.is_none());
            assert_eq!(new.log_density(x.pose)?, old.log_density(y.pose)?);
        }
        assert_eq!(a.random::<u64>(), b.random::<u64>());
    }
    Ok(())
}

#[test]
fn physical_class_line_complete_axis_channel_mixture_in_transformed_chart() -> Result<()> {
    for transformed in [false, true] {
        let f = Fixture::new(0.5, 0.8, &[0, 1, 2], transformed)?;
        let all = f.parse()?;
        let channels = f.guide["class_channels"].as_array().unwrap();
        let mut pieces = Vec::new();
        for axis in 0..3 {
            for channel in channels {
                let probability = channel["probability"].as_f64().unwrap() / 3.;
                let mut part = f.guide.clone();
                part["raw_translation_axes"] = json!([axis]);
                let mut only = channel.clone();
                only["probability"] = json!(1.);
                part["class_channels"] = json!([only]);
                pieces.push((probability, f.parse_spec(&part)?));
            }
        }
        for u in [
            [0.; 6],
            [1., 0.2, 0.1, 0., 0., 0.],
            [-1., 0.2, 0., 0., 0., 0.],
            [5., 0., 0., 0., 0., 0.],
            [0., 0., 0., 5., 0., 0.],
        ] {
            let (pose, j) = all.decode(u)?;
            let expected = pieces
                .iter()
                .map(|(p, g)| Ok(p.ln() + g.log_density(pose)?))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .fold(f64::NEG_INFINITY, log_add);
            let q = all.evaluate(pose)?;
            near(q.log_physical_density, expected);
            near(q.log_latent_density.unwrap() - j, expected);
            let axes = q.native_class_line_density.as_ref().unwrap()["axes"]
                .as_array()
                .unwrap();
            assert_eq!(axes.len(), 3);
            for a in axes {
                assert_eq!(a["channels"].as_array().unwrap().len(), 4);
            }
        }
    }
    Ok(())
}

#[test]
fn physical_class_line_draw_trace_retains_five_raw_coordinates() -> Result<()> {
    let f = Fixture::new(0.5, 1., &[0, 1, 2], true)?;
    let g = f.parse()?;
    let mut rng = StdRng::seed_from_u64(6100407102);
    let mut conditional = 0;
    for _ in 0..96 {
        let draw = g.draw_only(&mut rng)?;
        let trace = draw.native_class_line_draw.unwrap();
        if trace["conditional"] == true {
            conditional += 1;
            let axis = trace["axis"].as_u64().unwrap() as usize;
            let old: [f64; 6] = serde_json::from_value(trace["original_latent"].clone())?;
            let a = g.chart.coordinates(old);
            let b = g.chart.coordinates(draw.latent);
            for i in 0..6 {
                if i != axis {
                    near(a[i], b[i]);
                }
            }
            assert!(matches!(
                trace["fallback_target"].as_str(),
                Some("class" | "hard_free" | "unconditional")
            ));
        }
        assert!(g.evaluate(draw.pose)?.log_physical_density.is_finite());
    }
    assert!(conditional > 0);
    Ok(())
}

#[test]
fn physical_class_line_authenticates_compiled_bytes_physical_shape_and_source_geometry()
-> Result<()> {
    let mut f = Fixture::new(0.5, 1., &[0, 1, 2], false)?;
    let report = f.parse()?.native_shape_compatibility().cloned().unwrap();
    assert_eq!(report["compatible"], true);
    assert_eq!(report["matched_atoms"], 1);
    let mut vanished = f.guide.clone();
    vanished["class_channels"] = json!([
        {"class":"hard_free","probability":1.},
        {"class":"native","probability":1e-20}
    ]);
    assert!(f.parse_spec(&vanished).is_err());
    vanished["class_channels"].as_array_mut().unwrap().reverse();
    assert!(f.parse_spec(&vanished).is_err()); // positive first interval below Open01 support
    for (field, value) in [
        ("shape_sha256", json!("0".repeat(64))),
        ("capture_radius", json!(20.)),
        ("capture_center", json!([0., 1., 0.])),
        ("depletant_radius", json!(0.5)),
        ("fixed_poses", json!([])),
    ] {
        let mut bad = f.guide.clone();
        bad[field] = value;
        assert!(f.parse_spec(&bad).is_err(), "{field}");
    }
    let mut bad = f.guide.clone();
    bad["compiled_native"]["sha256"] = json!("0".repeat(64));
    assert!(f.parse_spec(&bad).is_err());
    let native_path = f.directory.join("native.json");
    let saved = fs::read(&native_path)?;
    let mut wrong: Value = serde_json::from_slice(&saved)?;
    wrong["monomer_atoms"][0]["radius"] = json!(0.31);
    let bytes = serde_json::to_vec(&wrong)?;
    fs::write(&native_path, &bytes)?;
    bad = f.guide.clone();
    bad["compiled_native"]["sha256"] = json!(hash_bytes(&bytes));
    assert!(f.parse_spec(&bad).is_err()); // matching file hash is insufficient
    assert!(f.parse().is_err()); // changed native bytes are also rejected
    fs::write(&native_path, &saved)?;
    f.tree = SphereTree::new(serde_json::from_value(
        json!({"atoms":[{"center":[0.,0.,0.],"radius":0.31}]}),
    )?)?;
    assert!(f.parse().is_err()); // bound source hash cannot disguise a wrong tree
    Ok(())
}

#[test]
fn physical_class_line_disabled_active_gaussians_cannot_hide_under_uniform_floor() -> Result<()> {
    let mut f = Fixture::new(0.5, 0., &[0], false)?;
    f.guide["gaussian_components"][0]["mean"] = json!([1e200, 0., 0., 0., 0., 0.]);
    let g = f.parse()?;
    assert!(g.evaluate(g.decode([0.; 6])?.0).is_err());
    // Its zero-weight Gaussian branch really is disabled at alpha=1.
    f.guide["defensive_uniform_shell_probability"] = json!(1.);
    let g = f.parse()?;
    assert!(
        g.evaluate(g.decode([0.; 6])?.0)?
            .log_physical_density
            .is_finite()
    );
    Ok(())
}
