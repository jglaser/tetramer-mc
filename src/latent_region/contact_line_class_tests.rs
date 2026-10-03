//! Toy-only complete-class laws. No protein geometry or physical draws.
use super::*;
use crate::geometry::Atom;

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}
fn identity() -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }))
}
fn chart() -> Chart {
    let mut lower = identity();
    lower[1][0] = 0.2;
    lower[3][0] = 0.4;
    lower[4][1] = -0.3;
    Chart {
        lower,
        mean: [0.; 6],
        anchor_position: [0.; 3],
        anchor_rotation: IDENTITY,
        fixed: pose([0.; 3]),
        ell: 2.,
        log_det: 0.,
    }
}
fn cfg() -> DockingConfig {
    serde_json::from_value(
        json!({"shape":"unused-toy.json", "fixed_poses":[pose([-2.1,0.,0.]),pose([2.1,0.,0.])],
        "initial_pose":pose([0.;3]),"capture_center":[0.,0.,0.],"capture_radius":64.,
        "depletant_radius":0.1,"reservoir_density":0.,"poisson_lambda_ratio":64.,
        "translation_steps":[0.1],"rotation_steps_deg":[1.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.5,"seed":1}),
    )
    .unwrap()
}
fn native_data() -> Value {
    json!({"schema":"native-entry-compiled-v1","source_definition_sha256":"0".repeat(64),
        "source_input_sha256":{"tetramer-shape.json":"1".repeat(64)},
        "criteria":{"body_member_position_entry_A":2.,"body_orientation_entry_deg":15.,"monomer_position_entry_A":3.,"monomer_orientation_entry_deg":20.,"contact_entry_A":2.,"native_reference_patch_gap_A":1.,"minimum_shared_native_residue_pairs":1,"hard_overlap_tolerance_A":1e-8,"catalogue_cycle_position_tolerance_A":1e-6,"catalogue_cycle_angle_tolerance_deg":1e-6},
        "fixed_poses":cfg().fixed_poses,"members":[{"position":[0.,0.,0.],"rotation":IDENTITY}],
        "monomer_atoms":[{"center":[0.,0.,0.],"radius":1.,"residue":0}],"residue_count":1,
        "references":[{"label":"A","family":"A","position":[2.,0.,0.],"rotation":IDENTITY,"native_residue_pairs":[0]}],
        "motifs":[{"id":7,"position":[2.,0.,0.],"rotation":IDENTITY,"member_contacts":[{"member_i":0,"member_j":0,"directed_class":"A"}]}]})
}
fn base_data(alpha: f64, beta: f64) -> Value {
    json!({"schema":"defensive-hard-free-line-guide-v1","region_sha256":"toy",
        "defensive_uniform_shell_probability":alpha,
        "gaussian_components":[{"weight":0.6,"mean":([0.;6]),"covariance":identity()},
        {"weight":0.4,"mean":[0.3,-0.2,0.,0.1,0.,0.],"covariance":identity()}],
        "raw_translation_axes":[0,1,2],"conditional_probability":beta,"minimum_conditional_mass":1e-10})
}
fn guide(alpha: f64, beta: f64, channels: Value) -> ContactLineGuide {
    let tree = SphereTree::new(Shape {
        name: "toy".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
    .unwrap();
    let mut g = ContactLineGuide::from_bytes(
        &serde_json::to_vec(&base_data(alpha, beta)).unwrap(),
        "toy",
        &chart(),
        0.,
        &cfg(),
        &tree,
    )
    .unwrap();
    g.hard_free_only = false;
    g.classes = Some(ClassConditioning {
        native: CompleteNativeEntry::from_bytes(&serde_json::to_vec(&native_data()).unwrap())
            .unwrap(),
        channels: serde_json::from_value(channels).unwrap(),
        depletant_radius: 0.1,
        shape_compatibility: Value::Null,
    });
    g
}
fn channels() -> Value {
    json!([{"class":"hard_free","probability":0.2},
        {"class":"native","probability":0.3},
        {"class":"contact_without_native","probability":0.4},
        {"class":"native","probability":0.1,"orthant":55}])
}
fn volume() -> f64 {
    (PI.powi(3) * 4_f64.powi(6) / 6.).ln()
}

#[test]
fn class_line_orthants_use_all_whitened_coordinates_and_inclusive_zero() {
    let a = [0.; 6];
    let d = [1., -1., 0., 0., 0., 0.];
    let zero = orthant_intervals(a, d, 63, [-2., 2.]).unwrap();
    assert_eq!(zero.intervals(), &[Interval::closed(0., 0.).unwrap()]);
    assert!(orthant_intervals(a, d, 60, [-2., 2.]).unwrap().is_empty());
    let positive = orthant_intervals(a, d, 61, [-2., 2.]).unwrap();
    assert!(!positive.contains(0.) && positive.contains(1.) && !positive.contains(-1.));
    let c = chart();
    let mut x = [0.3, -0.2, 0.5, 0.1, -0.4, 0.2];
    x[0] = 0.;
    let u0 = raw_to_latent(&c, x);
    let mut one = x;
    one[0] = 1.;
    let u1 = raw_to_latent(&c, one);
    let du = std::array::from_fn(|i| u1[i] - u0[i]);
    assert!(du[3].abs() > 0.1 && du[1].abs() > 0.1);
    for mask in 0..64 {
        let set = orthant_intervals(u0, du, mask, [-3., 3.]).unwrap();
        for j in -100..=100 {
            let s = j as f64 * 0.03;
            let expected = (0..6).all(|i| ((u0[i] + du[i] * s) >= 0.) == (mask & (1 << i) != 0));
            assert_eq!(set.contains(s), expected, "mask={mask} s={s}");
        }
    }
}

#[test]
fn class_line_fallback_and_single_class_density_dominance() {
    let mut g = guide(0.5, 1., channels());
    let h = IntervalSet::segment([-2., 2.]).unwrap();
    let c = IntervalSet::segment([-0.2, 0.2]).unwrap();
    let empty = IntervalSet::empty();
    for mean in [0., 1.5, 30.] {
        let narrow = g.class_law(&h, &c, mean, 1.).unwrap();
        let broad = g.class_law(&h, &h, mean, 1.).unwrap();
        for x in [-0.1, 0., 0.1] {
            assert!(narrow.multiplier(x) >= broad.multiplier(x));
        }
        let fallback = g.class_law(&h, &empty, mean, 1.).unwrap();
        assert_eq!(
            fallback.target,
            if mean == 30. {
                "unconditional"
            } else {
                "hard_free"
            }
        );
        for x in [-3., 0., 3.] {
            assert_eq!(fallback.multiplier(x), broad.multiplier(x));
        }
    }
    let mass = ContactLineGuide::masses(&c, 0., 1.).unwrap().1;
    g.minimum_mass = mass; // Equality falls back, strictly greater conditions.
    assert_eq!(g.class_law(&h, &c, 0., 1.).unwrap().target, "hard_free");
    g.minimum_mass = mass * 0.999;
    assert_eq!(g.class_law(&h, &c, 0., 1.).unwrap().target, "class");
    // A two-class mixture can LOWER density on the union: a large first class
    // carries only 10% of channel mass while the other class excludes this x.
    g.minimum_mass = 1e-10;
    let first = IntervalSet::segment([-2., 1.]).unwrap();
    let second = IntervalSet::segment([1., 2.]).unwrap();
    let a = g.class_law(&h, &first, 0., 1.).unwrap();
    let b = g.class_law(&h, &second, 0., 1.).unwrap();
    let old = g.class_law(&h, &h, 0., 1.).unwrap();
    assert!(0.1 * a.multiplier(0.) + 0.9 * b.multiplier(0.) < old.multiplier(0.));
}

#[test]
fn class_line_conditional_integrals_are_one_including_fallbacks() {
    let g = guide(0.5, 1., channels());
    let h = IntervalSet::from_intervals(vec![
        Interval::closed(-3., -1.).unwrap(),
        Interval::closed(0.2, 2.).unwrap(),
    ])
    .unwrap();
    let c = IntervalSet::segment([0.2, 0.5]).unwrap();
    for mean in [0., 5., 40.] {
        for class in [&h, &c, &IntervalSet::empty()] {
            let law = g.class_law(&h, class, mean, 1.).unwrap();
            let integral = law.intervals.map_or(1., |v| {
                ContactLineGuide::masses(v, mean, 1.).unwrap().1 / law.effective_mass
            });
            assert!((integral - 1.).abs() < 2e-15);
        }
    }
}

#[test]
fn class_line_disabled_and_pure_uniform_preserve_rng_exactly() {
    let c = chart();
    for (alpha, beta) in [(1., 1.), (0.5, 0.)] {
        let g = guide(alpha, beta, channels());
        let mut a = StdRng::seed_from_u64(6100401001);
        let mut b = StdRng::seed_from_u64(6100401001);
        for _ in 0..128 {
            let (u, r, k) = g.base.draw(&mut a, 4., 0., 1.).unwrap();
            let (v, s, l, flag) = g.draw(&mut b, &c, 4., 0., 1.).unwrap();
            assert_eq!((u, r, k), (v, s, l));
            assert_eq!(flag, None);
            assert_eq!(
                g.base.log_density(u, r <= 4., volume()),
                g.density_details(v, s <= 4., volume(), &c, 4.).unwrap().0
            );
        }
        assert_eq!(a.random::<u64>(), b.random::<u64>());
    }
}

#[test]
fn class_line_full_density_is_all_axis_component_channel_mixture() {
    let c = chart();
    let g = guide(0.5, 0.8, channels());
    let mut rng = StdRng::seed_from_u64(6100401002);
    for _ in 0..96 {
        let (u, r, _) = g.base.draw(&mut rng, 4., 0., 1.).unwrap();
        let actual = g
            .density_details(u, r <= 4., volume(), &c, 4.)
            .unwrap()
            .0
            .exp();
        let mut expected = 0.;
        for axis in 0..3 {
            for component in 0..2 {
                for channel in channels().as_array().unwrap() {
                    let mut only = guide(
                        0.5,
                        0.8,
                        json!([{"class":channel["class"],"probability":1.,"orthant":channel.get("orthant").cloned().unwrap_or(Value::Null)}]),
                    );
                    let w = only.base.components[component].weight;
                    let selected = only.base.components.remove(component);
                    only.base.components = vec![selected];
                    only.base.components[0].weight = 1.;
                    only.axes = vec![axis];
                    only.conditionals = vec![vec![
                        ConditionalNormal::new(&only.base.components[0], &c, axis).unwrap(),
                    ]];
                    expected += channel["probability"].as_f64().unwrap() * w / 3.
                        * only
                            .density_details(u, r <= 4., volume(), &c, 4.)
                            .unwrap()
                            .0
                            .exp();
                }
            }
        }
        assert!((actual - expected).abs() < 3e-12 * actual.max(expected));
    }
}

#[test]
fn class_line_draw_changes_one_raw_coordinate_and_restarts_exactly() {
    let c = chart();
    let g = guide(0.1, 1., channels());
    let mut a = StdRng::seed_from_u64(6100401003);
    let mut b = StdRng::seed_from_u64(6100401003);
    let mut conditioned = 0;
    for _ in 0..256 {
        let first = g.draw(&mut a, &c, 4., 0., 1.).unwrap();
        let trace = g.last_draw.borrow().clone();
        let second = g.draw(&mut b, &c, 4., 0., 1.).unwrap();
        assert_eq!(first, second);
        assert_eq!(trace, *g.last_draw.borrow());
        if trace["conditional"] != true {
            continue;
        }
        let old: [f64; 6] = serde_json::from_value(trace["original_latent"].clone()).unwrap();
        let raw_old = c.coordinates(old);
        let raw_new = c.coordinates(first.0);
        let axis = trace["axis"].as_u64().unwrap() as usize;
        for i in 0..6 {
            if i != axis {
                assert!((raw_new[i] - raw_old[i]).abs() < 2e-14);
            }
        }
        if trace["fallback_target"] != "unconditional" {
            conditioned += 1;
            assert!(trace["inverse_probability_error"].as_f64().unwrap() < 2e-11);
        }
    }
    assert!(conditioned > 50);
    assert_eq!(a.random::<u64>(), b.random::<u64>());
}

#[test]
fn class_line_schema_binds_shape_scaffold_and_compiled_definition() {
    let path = std::env::temp_dir().join(format!("class-line-bindings-{}", std::process::id()));
    assert!(!path.exists());
    std::fs::create_dir(&path).unwrap();
    let shape = Shape {
        name: "toy".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    };
    let shape_raw = serde_json::to_vec(&shape).unwrap();
    std::fs::write(path.join("shape.json"), &shape_raw).unwrap();
    let mut cfg = cfg();
    cfg.shape = path.join("shape.json");
    let mut native = native_data();
    native["source_input_sha256"]["tetramer-shape.json"] = json!(hash_bytes(&shape_raw));
    let native_raw = serde_json::to_vec(&native).unwrap();
    std::fs::write(path.join("native.json"), &native_raw).unwrap();
    let mut data = base_data(0.5, 1.);
    data["schema"] = json!("defensive-native-class-line-guide-v1");
    data["class_channels"] = channels();
    data["compiled_native"] =
        json!({"path":path.join("native.json"),"sha256":hash_bytes(&native_raw)});
    data["shape_sha256"] = json!(hash_bytes(&shape_raw));
    data["fixed_poses"] = json!(cfg.fixed_poses);
    data["capture_center"] = json!(cfg.capture_center);
    data["capture_radius"] = json!(cfg.capture_radius);
    data["depletant_radius"] = json!(cfg.depletant_radius);
    let tree = SphereTree::new(shape).unwrap();
    let parse = |v: &Value| {
        ContactLineGuide::from_bytes(
            &serde_json::to_vec(v).unwrap(),
            "toy",
            &chart(),
            0.,
            &cfg,
            &tree,
        )
    };
    assert!(parse(&data).is_ok());
    for (field, bad) in [
        ("shape_sha256", json!("0".repeat(64))),
        ("fixed_poses", json!([pose([0.; 3])])),
        ("capture_radius", json!(99.)),
        ("depletant_radius", json!(0.2)),
        ("raw_translation_axes", json!([3])),
        (
            "class_channels",
            json!([{"class":"native_remainder","probability":1.}]),
        ),
        (
            "class_channels",
            json!([{"class":"native","probability":0.9}]),
        ),
        (
            "class_channels",
            json!([{"class":"native","probability":1.,"orthant":64}]),
        ),
    ] {
        let mut bad_data = data.clone();
        bad_data[field] = bad;
        assert!(parse(&bad_data).is_err(), "{field}");
    }
    let mut wrong = data.clone();
    wrong["compiled_native"]["sha256"] = json!("0".repeat(64));
    assert!(parse(&wrong).is_err());
    wrong = data.clone();
    wrong["compiled_native"]["path"] = json!("relative.json");
    assert!(parse(&wrong).is_err());
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn class_line_importance_normalization_preserves_five_coordinate_marginals() {
    // Fixed allocation and seeds, no tuning or extension after outcomes.
    let c = chart();
    let mut g = guide(0.2, 0.7, channels());
    g.axes = vec![0];
    g.conditionals = vec![g.conditionals.remove(0)];
    let mut rng = StdRng::seed_from_u64(6100401004);
    let mut sum = 0.;
    let mut sum2 = 0.;
    let mut m = [0.; 5];
    let mut v = [0.; 5];
    const N: usize = 8192;
    for _ in 0..N {
        let (u, r, _, _) = g.draw(&mut rng, &c, 4., 0., 1.).unwrap();
        let q = g.density_details(u, r <= 4., volume(), &c, 4.).unwrap().0;
        let old = g.base.log_density(u, r <= 4., volume());
        let w = (old - q).exp();
        sum += w;
        sum2 += w * w;
        let raw = c.coordinates(u);
        for i in 0..5 {
            m[i] += raw[i + 1];
            v[i] += raw[i + 1] * raw[i + 1];
        }
    }
    let mean = sum / N as f64;
    let se = ((sum2 / N as f64 - mean * mean) / (N - 1) as f64).sqrt();
    assert!(
        (mean - 1.).abs() < 5. * se + 1e-12,
        "normalization {mean} +/- {se}"
    );
    for i in 1..6 {
        let expected = (1. - g.base.alpha)
            * g.base
                .components
                .iter()
                .map(|component| {
                    component.weight
                        * (0..6)
                            .map(|j| c.lower[i][j] * component.mean[j])
                            .sum::<f64>()
                })
                .sum::<f64>();
        let mean = m[i - 1] / N as f64;
        let se = ((v[i - 1] / N as f64 - mean * mean) / (N - 1) as f64).sqrt();
        assert!(
            (mean - expected).abs() < 5. * se + 1e-12,
            "unchanged raw marginal {i}: {mean} expected {expected} SE {se}"
        );
    }
}

#[test]
fn class_line_chord_overflow_is_fatal_and_tangent_is_retained() {
    assert!(checked_ball_chord([1e308, 0., 0.], [1., 0., 0.], 4.).is_err());
    assert!(checked_ball_chord([0.; 3], [1e308, 0., 0.], 4.).is_err());
    assert!(checked_ball_chord([0.; 3], [0.; 3], 4.).is_err());
    assert_eq!(
        checked_ball_chord([0., 4., 0.], [1., 0., 0.], 4.).unwrap(),
        Some([0., 0.])
    );
    assert_eq!(
        checked_ball_chord([0., 5., 0.], [1., 0., 0.], 4.).unwrap(),
        None
    );
}

#[test]
fn class_line_sphere_hard_volume_and_analytic_depletion_reference() {
    let c = chart();
    let g = guide(0.5, 1., channels());
    let mut guided_rng = StdRng::seed_from_u64(6100401005);
    let mut uniform_rng = StdRng::seed_from_u64(6100401006);
    // Disjoint exclusion spheres around the two fixed centers make AO additive.
    let physical = |u: [f64; 6]| -> [f64; 2] {
        if u.iter().map(|v| v * v).sum::<f64>() > 16. {
            return [0.; 2];
        }
        let raw = c.coordinates(u);
        let p = [raw[0], raw[1], raw[2]];
        let mut overlap = 0.;
        for center in [[-2.1, 0., 0.], [2.1, 0., 0.]] {
            let d = norm(sub(p, center));
            if d < 2. {
                return [0.; 2];
            }
            let r = 1.1;
            if d < 2. * r {
                overlap += PI * (4. * r + d) * (2. * r - d).powi(2) / 12.;
            }
        }
        let c2 = (3..6).map(|i| (raw[i] / 2.).powi(2)).sum::<f64>();
        let j = 1. / (8. * PI.powi(2) * (1. + c2).powi(2));
        [j, j * (30. * overlap).exp()]
    };
    let mut sums = [[0.; 2]; 2];
    let mut squares = [[0.; 2]; 2];
    let counts = [8192, 65536];
    for arm in 0..2 {
        for _ in 0..counts[arm] {
            let (u, scale) = if arm == 0 {
                let (u, r, _, _) = g.draw(&mut guided_rng, &c, 4., 0., 1.).unwrap();
                let logq = g.density_details(u, r <= 4., volume(), &c, 4.).unwrap().0;
                (u, (-logq).exp())
            } else {
                // Independent radial-angular construction, including all draws.
                let z: [f64; 6] = std::array::from_fn(|_| StandardNormal.sample(&mut uniform_rng));
                let length = z.iter().map(|v| v * v).sum::<f64>().sqrt();
                let r = 4. * uniform_rng.random::<f64>().powf(1. / 6.);
                (z.map(|v| v * r / length), volume().exp())
            };
            for (k, v) in physical(u).into_iter().enumerate() {
                let value = v * scale;
                sums[arm][k] += value;
                squares[arm][k] += value * value;
            }
        }
    }
    for k in 0..2 {
        let means: [f64; 2] = std::array::from_fn(|a| sums[a][k] / counts[a] as f64);
        let variances: [f64; 2] = std::array::from_fn(|a| {
            (squares[a][k] / counts[a] as f64 - means[a] * means[a]) / (counts[a] - 1) as f64
        });
        let se = (variances[0] + variances[1]).sqrt();
        assert!(
            (means[0] - means[1]).abs() <= 5. * se + 1e-10,
            "class sphere physical {k}: guided={} uniform={} combinedSE={se}",
            means[0],
            means[1]
        );
    }
    assert!(sums[0][1] > sums[0][0] && sums[1][1] > sums[1][0]);
}

#[test]
fn class_line_hard_free_only_channel_matches_legacy_density() {
    let c = chart();
    let new = guide(0.5, 1., json!([{"class":"hard_free","probability":1.}]));
    let old = ContactLineGuide::from_bytes(
        &serde_json::to_vec(&base_data(0.5, 1.)).unwrap(),
        "toy",
        &c,
        0.,
        &cfg(),
        &new.tree,
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(6100401007);
    for _ in 0..256 {
        let (u, r, _) = old.base.draw(&mut rng, 4., 0., 1.).unwrap();
        let a = old.density_details(u, r <= 4., volume(), &c, 4.).unwrap().0;
        let b = new.density_details(u, r <= 4., volume(), &c, 4.).unwrap().0;
        assert!((a - b).abs() < 2e-12 || a == b, "legacy H={a} class H={b}");
    }
}
