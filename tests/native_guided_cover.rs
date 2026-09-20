//! Independent density and physical-integral controls for the hybrid guide.
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::{f64::consts::PI, fs, path::Path};
use tetramer_mc::{
    math::*,
    native_region::{
        self, NativeCover, NativeCoverMixture, NativeGuide, NativeMetric, NativeRegionOptions,
    },
    proposal::{FrozenRelativePoseProposal, GaussianComponentParameters},
    simulation::hash_file,
};

const CENTER: Vec3 = [7., -4., 3.];
const ELL: f64 = 3.;
const EPSILON: f64 = 0.1;
const BETA: f64 = 0.75;

fn identity(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn reference() -> Pose {
    Pose {
        position: CENTER,
        orientation: quaternion(cayley([0.2, -0.3, 0.1])),
    }
}
fn anchor() -> Pose {
    Pose {
        position: add(CENTER, [15., 0., 0.]),
        orientation: quaternion(cayley([-0.4, 0.2, 0.3])),
    }
}
fn metric(centroid: Vec3, radius: f64) -> NativeMetric {
    NativeMetric {
        native_poses: vec![reference()],
        rigid_members: vec![identity(centroid)],
        member_error_scale: radius,
        angle_error_scale_deg: 40.,
    }
}
fn parameters() -> (GaussianComponentParameters, [[f64; 6]; 6]) {
    let sd = [0.9, 0.7, 1.1, 0.23, 0.19, 0.27];
    let mut lower = [[0.; 6]; 6];
    for i in 0..6 {
        lower[i][i] = sd[i];
    }
    lower[3][0] = 0.18;
    lower[4][1] = -0.11;
    lower[5][2] = 0.13;
    let covariance = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    let inverse = transpose(rotation(anchor().orientation));
    (
        GaussianComponentParameters {
            anchor_position: matvec(inverse, sub(reference().position, anchor().position)),
            anchor_rotation: matmul(inverse, rotation(reference().orientation)),
            mean: [0.3, -0.15, 0.05, 0.02, -0.015, 0.01],
            covariance,
            weight: 1.,
        },
        lower,
    )
}
fn model(cube: f64) -> Result<FrozenRelativePoseProposal> {
    FrozenRelativePoseProposal::from_components_open(
        vec![parameters().0],
        ELL,
        [cube; 3],
        EPSILON,
        &"0".repeat(64),
        &"0".repeat(64),
    )
}
fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 2e-10 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

// Matrix inverse Cayley and an explicitly supplied Cholesky factor provide
// a density reconstruction independent of the production quaternion inverse.
fn gaussian_density(p: Pose) -> f64 {
    let (component, lower) = parameters();
    let inverse = transpose(rotation(anchor().orientation));
    let t = matvec(inverse, sub(p.position, anchor().position));
    let residual = matmul(
        matmul(inverse, rotation(p.orientation)),
        transpose(component.anchor_rotation),
    );
    let d = 1. + residual[0][0] + residual[1][1] + residual[2][2];
    assert!(d > 1e-10);
    let c = [
        (residual[2][1] - residual[1][2]) / d,
        (residual[0][2] - residual[2][0]) / d,
        (residual[1][0] - residual[0][1]) / d,
    ];
    let mut z = [0.; 6];
    let mut log = -3. * (2. * PI).ln();
    for i in 0..6 {
        let x = if i < 3 {
            t[i] - component.anchor_position[i]
        } else {
            ELL * c[i - 3]
        };
        z[i] =
            (x - component.mean[i] - (0..i).map(|j| lower[i][j] * z[j]).sum::<f64>()) / lower[i][i];
        log -= lower[i][i].ln() + 0.5 * z[i] * z[i];
    }
    (log + 3. * ELL.ln() + 2. * PI.ln() + 2. * dot(c, c).ln_1p()).exp()
}

#[test]
fn both_proposal_families_and_cube_are_in_the_density() -> Result<()> {
    let cover = NativeCover::new(&metric([0.; 3], 2.))?;
    let mixture = NativeCoverMixture::new(&cover, vec![0.3, 1.], vec![0.4, 0.6])?;
    let guide = NativeGuide::new(model(16.)?, BETA, CENTER, anchor())?;
    for delta in [[0.; 3], [1.5, 0., 0.], [3., 0., 0.], [9., 0., 0.]] {
        let p = Pose {
            position: add(CENTER, delta),
            orientation: reference().orientation,
        };
        let g = (1. - EPSILON) * gaussian_density(p)
            + if delta.iter().all(|v| *v >= -8. && *v < 8.) {
                EPSILON / 16_f64.powi(3)
            } else {
                0.
            };
        near(guide.log_density(p)?, g.ln());
        let c: f64 = mixture
            .covers
            .iter()
            .zip(&mixture.weights)
            .filter(|(c, _)| norm(delta) <= c.ball_radius)
            .map(|(c, w)| w / c.volume)
            .sum();
        near(
            guide.full_log_density(&mixture, p)?,
            ((1. - BETA) * c + BETA * g).ln(),
        );
    }
    for bad in [0., 1., -0.1, f64::NAN] {
        assert!(NativeGuide::new(model(16.)?, bad, CENTER, anchor()).is_err());
    }
    let mut rng = StdRng::seed_from_u64(99010171);
    for _ in 0..1024 {
        let p = guide.sample(&mut rng)?.pose;
        assert!(guide.log_density(p)?.is_finite());
    }
    Ok(())
}

fn fixture(
    root: &Path,
    m: NativeMetric,
    capture: f64,
    second_neighbor: bool,
    activity: f64,
) -> Result<NativeRegionOptions> {
    fs::create_dir_all(root)?;
    fs::write(
        root.join("shape.json"),
        json!({"name":"sphere","atoms":[{"center":[0.,0.,0.],"radius":1.}]}).to_string(),
    )?;
    let shape_sha = hash_file(&root.join("shape.json"))?;
    let p = parameters().0;
    let raw_model = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":shape_sha,"angular_length":ELL,
        "anchors":[{"position":p.anchor_position,"rotation":p.anchor_rotation}],"means":[p.mean],"covariances":[p.covariance],"weights":[1.]});
    fs::write(root.join("model.json"), raw_model.to_string())?;
    let mut fixed = vec![anchor()];
    if second_neighbor {
        fixed.push(identity(CENTER));
    }
    fs::write(root.join("config.json"), json!({"shape":"shape.json","fixed_poses":fixed,"capture_center":CENTER,
        "capture_radius":capture,"depletant_radius":0.7,"reservoir_density":activity,"poisson_lambda_ratio":16.,
        "endpoint_gate":{"max_cells":63,"max_depth":8,"min_width":0.1},"metadata":m}).to_string())?;
    Ok(NativeRegionOptions {
        config: root.join("config.json"),
        out: root.join("output"),
        samples: 32768,
        seed: 99010193,
        cloud_replicates: 2,
        activity: None,
        lambda_ratio: None,
        cover_scales: vec![1.],
        cover_weights: vec![],
        model: Some(root.join("model.json")),
        model_weight: BETA,
        model_uniform_probability: EPSILON,
        model_anchor_index: 0,
    })
}

fn check(result: &Value, field: &str, expected: f64) {
    let q = result[field]["log_normalizer"].as_f64().unwrap().exp();
    let se = q * result[field]["observed_relative_standard_error"]
        .as_f64()
        .unwrap();
    eprintln!("{field}: {q}, expected {expected}, SE {se}");
    assert!((q - expected).abs() <= 6.5 * se + 2e-9 * (1. + expected.abs()));
}

#[test]
fn coupled_shifted_guide_integrates_known_native_volume_and_capture_zeros() -> Result<()> {
    let root = std::env::temp_dir().join(format!("native-guided-volume-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    let haar = native_region::theta_minus_sin(40_f64.to_radians()) / PI;
    let full = native_region::run(fixture(
        &root.join("full"),
        metric([1., -0.5, 2.], 2.),
        8.,
        false,
        0.,
    )?)?;
    check(&full, "native", 4. * PI * 8. / 3. * haar);
    check(&full, "hard_native", 4. * PI * 8. / 3. * haar);
    assert!(full["family_draws"]["cover"].as_u64().unwrap() > 7000);
    assert!(full["family_draws"]["guide"].as_u64().unwrap() > 23000);
    assert!(full["q_rejected"].as_u64().unwrap() > 0);
    let clipped = native_region::run(fixture(
        &root.join("clipped"),
        metric([0.; 3], 2.),
        1.,
        false,
        0.,
    )?)?;
    check(&clipped, "native", 4. * PI / 3. * haar);
    assert!(clipped["capture_rejected"].as_u64().unwrap() > 1000);
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn guide_anchor_does_not_remove_the_other_physical_neighbor() -> Result<()> {
    let root = std::env::temp_dir().join(format!("native-guided-ao-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    let options = fixture(&root, metric([0.; 3], 4.), 4., true, 0.4)?;
    let result = native_region::run(options)?;
    let haar = native_region::theta_minus_sin(40_f64.to_radians()) / PI;
    let f = |r: f64| {
        let overlap = if r < 3.4 {
            PI * (6.8 + r) * (3.4 - r).powi(2) / 12.
        } else {
            0.
        };
        4. * PI * r * r * (0.4 * overlap).exp()
    };
    let n = 8192;
    let h = 2. / n as f64;
    let integral = h / 3.
        * (f(2.)
            + f(4.)
            + (1..n)
                .map(|i| (if i % 2 == 0 { 2. } else { 4. }) * f(2. + h * i as f64))
                .sum::<f64>());
    check(&result, "native", haar * integral);
    check(&result, "hard_native", haar * 4. * PI / 3. * (64. - 8.));
    assert!(result["hard_rejected"].as_u64().unwrap() > 1000);
    assert!(result["raw_cloud_points"].as_u64().unwrap() > 0);
    let mut bad = fixture(&root.join("bad-anchor"), metric([0.; 3], 4.), 4., true, 0.)?;
    bad.model_anchor_index = 2;
    assert!(native_region::run(bad).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
