use super::*;
use crate::geometry::Atom;

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn chart() -> Chart {
    let diagonal = [2., 1.5, 0.7, 1.1, 0.9, 0.8];
    Chart {
        lower: std::array::from_fn(|i| {
            std::array::from_fn(|j| if i == j { diagonal[i] } else { 0. })
        }),
        mean: [0.; 6],
        fixed: pose([-1.5, 0., 0.]),
        anchor_position: [1.5, 0., 0.],
        anchor_rotation: IDENTITY,
        ell: 2.,
        log_det: diagonal.iter().map(|x| x.ln()).sum(),
    }
}
fn setup(
    spectator: Option<Vec3>,
    localized: f64,
    components: usize,
) -> (Chart, SphereTree, DockingConfig, ContactDistanceGuide) {
    let chart = chart();
    let tree = SphereTree::new(Shape {
        name: "sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
    .unwrap();
    let mut fixed = vec![pose([-1.5, 0., 0.]), pose([1.5, 0., 0.])];
    if let Some(p) = spectator {
        fixed.push(pose(p));
    }
    let cfg:DockingConfig=serde_json::from_value(json!({"shape":"unused",
        "fixed_poses":fixed,"initial_pose":pose([0.,-2.2,0.]),"capture_center":[0.,0.,0.],
        "capture_radius":64.,"depletant_radius":0.1,"reservoir_density":0.,"poisson_lambda_ratio":64.,
        "translation_steps":[0.1],"rotation_steps_deg":[1.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.5,"seed":1})).unwrap();
    let gaussians:Vec<_>=(0..components).map(|i|{
        let diagonal=[1.,0.3+0.7*i as f64,0.5,0.7,0.8,0.9];
        json!({"weight":1.,"mean":[0.,-0.8,0.3,0.,0.,0.],
            "covariance":std::array::from_fn::<_,6,_>(|j|std::array::from_fn::<_,6,_>(|k|if j==k {diagonal[j]*diagonal[j]} else {0.}))})
    }).collect();
    let pair = json!([{"neighbor_index":0,"moving_atom":0,"fixed_atom":0},
        {"neighbor_index":1,"moving_atom":0,"fixed_atom":0}]);
    let raw = json!({"schema":"defensive-contact-distance-guide-v1","region_sha256":"toy",
        "defensive_uniform_shell_probability":0.5,"conditional_probability":0.5,
        "gaussian_components":gaussians,"component_contact_pairs":vec![pair;components],
        "contact_widths_A":[0.3,0.5],"contact_neighbor_indices":[0,1],
        "minimum_center_distance":1e-8,"minimum_polygon_area":1e-16,
        "azimuth":{"localized_probability":localized,"radius_floor":1e-8,"projection_floor":1e-10,
            "gamma_min":0.01,"gamma_max":PI}});
    let guide = ContactDistanceGuide::from_bytes(
        &serde_json::to_vec(&raw).unwrap(),
        "toy",
        &chart,
        0.,
        &cfg,
        &tree,
    )
    .unwrap();
    (chart, tree, cfg, guide)
}
fn point(phi: f64) -> Pose {
    let frame = ContactFrame::new([-1.5, 0., 0.], [1.5, 0., 0.]).unwrap();
    pose(frame.circle([2.2, 2.2]).unwrap().point(phi).unwrap())
}
fn score(
    guide: &ContactDistanceGuide,
    chart: &Chart,
    tree: &SphereTree,
    cfg: &DockingConfig,
    p: Pose,
    floor: f64,
) -> Value {
    let u = chart.encode(p).unwrap();
    let decoded = chart.decode(u).0;
    guide
        .arc_density_details(
            u,
            true,
            10.,
            chart,
            floor,
            &mut ArcQueryCache::new(decoded, tree, &cfg.fixed_poses),
        )
        .unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a:.17e} != {b:.17e}");
}

#[test]
fn two_selected_spheres_recover_old_complete_density() {
    let (chart, tree, cfg, guide) = setup(None, 0.9, 2);
    for phi in [0.1, 1., 2.5, 4.7, 6.] {
        let p = point(phi);
        let u = chart.encode(p).unwrap();
        let expected = guide.density_details(u, true, 10., &chart, 4.).unwrap().0;
        let actual = score(&guide, &chart, &tree, &cfg, p, 1e-12);
        close(
            actual["log_proposal_density"].as_f64().unwrap(),
            expected,
            2e-13,
        );
        close(
            actual["old_distance_log_density"].as_f64().unwrap(),
            expected,
            2e-13,
        );
        for c in actual["components"].as_array().unwrap() {
            for w in c["widths"].as_array().unwrap() {
                close(w["arc_mass"].as_f64().unwrap(), 1., 1e-14);
            }
        }
    }
}

#[test]
fn spectator_uniform_law_matches_analytic_arc_and_nonunit_chart_density() {
    let (chart, tree, cfg, guide) = setup(Some([0., 2., 0.]), 0., 1);
    let p = point(PI / 2.);
    let u = chart.encode(p).unwrap();
    let actual = score(&guide, &chart, &tree, &cfg, p, 1e-12);
    let rho = (2.2_f64.powi(2) - 1.5_f64.powi(2)).sqrt();
    let excluded = 2. * ((rho * rho + 4. - 4.) / (4. * rho)).acos();
    let allowed = 2. * PI - excluded;
    let angle_scales = [1.1 * 0.7, 0.9 * 0.8, 0.8 * 0.9];
    let angular = (2. * PI).powf(-1.5) / angle_scales.iter().product::<f64>();
    let f_mean = [0.3_f64, 0.5]
        .iter()
        .map(|w| 3. / (w * w * 2.2 * 2.2 * allowed))
        .sum::<f64>()
        / 2.;
    let h = chart.log_det.exp() * angular * f_mean;
    let sigmas = [1., 0.3, 0.5, 0.7, 0.8, 0.9];
    let mean = [0., -0.8, 0.3, 0., 0., 0.];
    let gaussian = (2. * PI).powi(-3) / sigmas.iter().product::<f64>()
        * (-0.5
            * (0..6)
                .map(|i| ((u[i] - mean[i]) / sigmas[i]).powi(2))
                .sum::<f64>())
        .exp();
    let expected = 0.5 * (-10_f64).exp() + 0.25 * gaussian + 0.25 * h;
    close(
        actual["log_proposal_density"].as_f64().unwrap(),
        expected.ln(),
        2e-12,
    );
    for w in actual["components"][0]["widths"].as_array().unwrap() {
        close(w["arc_mass"].as_f64().unwrap(), allowed / (2. * PI), 2e-14);
        assert_eq!(w["query_phi_allowed"], true);
    }
}

#[test]
fn hard_invalid_arc_is_zero_but_empty_and_declared_floor_keep_original_law() {
    let (chart, tree, cfg, guide) = setup(Some([0., 2., 0.]), 0.9, 1);
    let p = point(3. * PI / 2.);
    let a = score(&guide, &chart, &tree, &cfg, p, 1e-12);
    for w in a["components"][0]["widths"].as_array().unwrap() {
        assert_eq!(w["query_phi_allowed"], false);
        assert_eq!(w["arc_fallback"], false);
        assert!(w["translation_log_density"].is_null());
        assert!(w["old_translation_log_density"].is_number());
    }
    let floor = score(&guide, &chart, &tree, &cfg, p, 1.);
    close(
        floor["log_proposal_density"].as_f64().unwrap(),
        floor["old_distance_log_density"].as_f64().unwrap(),
        1e-14,
    );
    let (chart, tree, cfg, guide) = setup(Some([0.; 3]), 0.9, 1);
    let empty = score(&guide, &chart, &tree, &cfg, point(1.), 1e-12);
    for w in empty["components"][0]["widths"].as_array().unwrap() {
        assert_eq!(w["arc_mass"], 0.);
        assert_eq!(w["arc_fallback"], true);
        assert_eq!(
            w["translation_log_density"],
            w["old_translation_log_density"]
        );
    }
    close(
        empty["log_proposal_density"].as_f64().unwrap(),
        empty["old_distance_log_density"].as_f64().unwrap(),
        1e-14,
    );
}

#[test]
fn shared_geometry_keeps_component_specific_probabilities_and_rejects_stale_pose() {
    let (chart, tree, cfg, guide) = setup(Some([0., 2., 0.]), 0.9, 2);
    let u = chart.encode(point(PI / 2.)).unwrap();
    let p = chart.decode(u).0;
    let mut cache = ArcQueryCache::new(p, &tree, &cfg.fixed_poses);
    let a = guide
        .arc_density_details(u, true, 10., &chart, 1e-12, &mut cache)
        .unwrap();
    assert_eq!(cache.entries.len(), 1);
    assert_eq!(cache.requests, 4);
    let z0 = a["components"][0]["widths"][0]["arc_mass"]
        .as_f64()
        .unwrap();
    let z1 = a["components"][1]["widths"][0]["arc_mass"]
        .as_f64()
        .unwrap();
    assert!(
        (z0 - z1).abs() > 1e-3,
        "Shared geometry must not share component mass"
    );
    let b = guide
        .arc_density_details(u, true, 10., &chart, 1e-12, &mut cache)
        .unwrap();
    assert_eq!(cache.entries.len(), 1);
    assert_eq!(a, b);
    let other = chart.encode(point(1.)).unwrap();
    assert!(
        guide
            .arc_density_details(other, true, 10., &chart, 1e-12, &mut cache)
            .is_err()
    );
}

#[test]
fn disabled_laws_and_outer_fallback_do_not_request_geometry() {
    let (chart, tree, cfg, mut guide) = setup(Some([0., 2., 0.]), 0.9, 2);
    let u = chart.encode(point(1.)).unwrap();
    let p = chart.decode(u).0;
    for mode in 0..4 {
        guide.base.alpha = if mode == 0 { 1. } else { 0.5 };
        guide.beta = if mode == 1 { 0. } else { 0.5 };
        guide.minimum_distance = if mode == 2 { 10. } else { 1e-8 };
        guide.minimum_area = if mode == 3 { 1. } else { 1e-16 };
        let mut cache = ArcQueryCache::new(p, &tree, &cfg.fixed_poses);
        let actual = guide
            .arc_density_details(u, true, 10., &chart, 1e-12, &mut cache)
            .unwrap();
        let old = guide.density_details(u, true, 10., &chart, 4.).unwrap().0;
        close(actual["log_proposal_density"].as_f64().unwrap(), old, 1e-14);
        assert_eq!(cache.requests, 0);
        assert!(cache.entries.is_empty());
    }
}

#[test]
fn conditional_azimuth_integrates_to_one_with_component_and_width_mixture() {
    let (chart, tree, cfg, guide) = setup(Some([0., 2., 0.]), 0.9, 2);
    let frame = ContactFrame::new([-1.5, 0., 0.], [1.5, 0., 0.]).unwrap();
    let circle = frame.circle([2.2, 2.2]).unwrap();
    let geom =
        circle_forbidden_arcs(&tree, &circle, [1., 0., 0., 0.], &tree, &cfg.fixed_poses).unwrap();
    let mut integral = [[0.; 2]; 2];
    // Deterministic composite-midpoint quadrature within each analytic allowed
    // interval, checking the actual scored translation density and Jacobian.
    for interval in geom.allowed.intervals() {
        let n = 2048;
        let step = interval.length() / n as f64;
        for i in 0..n {
            let phi = interval.lower + (i as f64 + 0.5) * step;
            let a = score(&guide, &chart, &tree, &cfg, point(phi), 1e-12);
            for (ci, c) in a["components"].as_array().unwrap().iter().enumerate() {
                for (wi, w) in c["widths"].as_array().unwrap().iter().enumerate() {
                    let density = w["translation_log_density"].as_f64().unwrap().exp();
                    // f_t dr1dr2dphi converts by r1*r2/D; polygon area
                    // cancels the fixed-radius uniform factor for this check.
                    integral[ci][wi] +=
                        step * density * w["polygon_area"].as_f64().unwrap() * 2.2 * 2.2 / 3.;
                }
            }
        }
    }
    for component in integral {
        for value in component {
            close(value, 1., 2e-6);
        }
    }
}
