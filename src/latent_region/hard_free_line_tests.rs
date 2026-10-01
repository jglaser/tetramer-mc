use super::*;

fn hard_data(alpha: f64, beta: f64) -> Value {
    let mut raw = data(alpha, beta);
    raw["schema"] = json!("defensive-hard-free-line-guide-v1");
    raw.as_object_mut().unwrap().remove("contact_widths_A");
    raw.as_object_mut()
        .unwrap()
        .remove("contact_neighbor_indices");
    raw
}

#[test]
fn hard_free_schema_rejects_ambiguous_contact_controls() {
    let c = chart(false);
    let cfg = configuration();
    let parse = |r: &Value| {
        ContactLineGuide::from_bytes(
            &serde_json::to_vec(r).unwrap(),
            "toy-region",
            &c,
            0.,
            &cfg,
            &sphere(),
        )
    };
    let mut raw = hard_data(0.5, 1.);
    assert!(parse(&raw).unwrap().hard_free_only);
    raw["contact_neighbor_indices"] = json!([]);
    assert!(parse(&raw).is_err());
    raw = data(0.5, 1.);
    raw["contact_neighbor_indices"] = json!([]);
    assert!(parse(&raw).is_err());
}

#[test]
fn hard_free_two_disconnected_intervals_normalize_actual_scored_density() {
    let c = chart(false);
    let mut cfg = configuration();
    cfg.fixed_poses = vec![pose([-1., 0., 0.]), pose([1., 0., 0.])];
    let mut raw = hard_data(0.5, 1.);
    raw["raw_translation_axes"] = json!([0]);
    raw["gaussian_components"] = json!([component([0.; 6], identity(), 1.)]);
    let g = guide(&raw, &c, &cfg);
    let (sets, _) = g.intervals([0.; 6], &c, OUTER, 0).unwrap();
    assert_eq!(sets[0].intervals().len(), 2);
    assert_eq!(sets[0].intervals()[0].upper, -3.);
    assert_eq!(sets[0].intervals()[1].lower, 3.);
    let uniform = 0.5 * (-log_volume()).exp();
    let outer_marginal = (2. * PI).powf(-2.5);
    let mut integral = 0.;
    for interval in sets[0].intervals() {
        let n = 256;
        let step = (interval.upper - interval.lower) / n as f64;
        for i in 0..=n {
            let mut u = [0.; 6];
            u[0] = interval.lower + i as f64 * step;
            let q = g
                .density_details(u, true, log_volume(), &c, OUTER)
                .unwrap()
                .0
                .exp();
            let conditional = (q - uniform) / (0.5 * outer_marginal);
            let weight = if i == 0 || i == n {
                1.
            } else if i % 2 == 0 {
                2.
            } else {
                4.
            };
            integral += weight * step * conditional / 3.;
        }
    }
    assert!((integral - 1.).abs() < 2e-10, "{integral}");
}

#[test]
fn hard_free_full_density_dominates_gaussian_on_all_valid_test_poses() {
    let c = chart(true);
    let cfg = configuration();
    let raw = hard_data(0.5, 1.);
    let g = guide(&raw, &c, &cfg);
    let mut checked = 0;
    for t in [
        [0., 0., 0.],
        [0., 0., 2.8],
        [0., 2.5, 0.],
        [-0.2, 1., 0.],
        [0.1, 0.4, -0.2],
    ] {
        for angles in [[0.; 3], [0.2, -0.1, 0.3]] {
            let x = [t[0], t[1], t[2], angles[0], angles[1], angles[2]];
            let u = raw_to_latent(&c, x);
            let p = c.decode(u).0;
            if u.iter().map(|v| v * v).sum::<f64>() > OUTER * OUTER
                || cfg
                    .fixed_poses
                    .iter()
                    .any(|q| sphere().overlaps(&Placed::new(p), &Placed::new(*q)))
            {
                continue;
            }
            let (q, detail) = g.density_details(u, true, log_volume(), &c, OUTER).unwrap();
            let baseline = g.base.log_density(u, true, log_volume());
            assert!(q >= baseline - 1e-13);
            let mut axes = f64::NEG_INFINITY;
            for axis in detail["axes"].as_array().unwrap() {
                let qa = axis["axis_log_proposal_density"].as_f64().unwrap();
                assert!(qa >= baseline - 1e-13);
                axes = log_add(axes, qa - 3_f64.ln());
                for k in axis["components"].as_array().unwrap() {
                    assert!(k["fallback"] == true || k["query_coordinate_allowed"] == true);
                }
            }
            assert!((q - axes).abs() < 1e-12);
            checked += 1;
        }
    }
    assert!(checked >= 6);
}

#[test]
fn hard_free_empty_and_small_mass_fallback_keep_original_gaussian() {
    let c = chart(false);
    let mut cfg = configuration();
    cfg.fixed_poses = vec![pose([0.; 3])];
    cfg.capture_radius = 0.5;
    let mut raw = hard_data(0.5, 1.);
    raw["raw_translation_axes"] = json!([0]);
    let g = guide(&raw, &c, &cfg);
    let u = [0.; 6];
    let (q, d) = g.density_details(u, true, log_volume(), &c, OUTER).unwrap();
    assert_eq!(q, g.base.log_density(u, true, log_volume()));
    assert_eq!(d["axes"][0]["empty_reason"], "no_positive_hard_free_length");
    cfg = configuration();
    raw["gaussian_components"] = json!([component([12., 0., 0., 0., 0., 0.], identity(), 1.)]);
    let g = guide(&raw, &c, &cfg);
    let (q, d) = g.density_details(u, true, log_volume(), &c, OUTER).unwrap();
    assert_eq!(q, g.base.log_density(u, true, log_volume()));
    let k = &d["axes"][0]["components"][0];
    assert!(k["conditional_mass"].as_f64().unwrap() > 0.);
    assert_eq!(k["fallback"], true);
}

#[test]
fn hard_free_conditional_draw_retains_five_coordinates_and_inverts_cdf() {
    let c = chart(true);
    let cfg = configuration();
    let g = guide(&hard_data(0.5, 1.), &c, &cfg);
    let mut successful = 0;
    for seed in 0..256 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (u, _, _, _) = g.draw(&mut rng, &c, OUTER, 0., 1.).unwrap();
        let trace = g.last_draw.borrow();
        if trace["conditional"] != true {
            continue;
        }
        let original: [f64; 6] = serde_json::from_value(trace["original_latent"].clone()).unwrap();
        if trace["fallback"] == true {
            assert_eq!(original, u);
            continue;
        }
        let axis = trace["axis"].as_u64().unwrap() as usize;
        let before = c.coordinates(original);
        let after = c.coordinates(u);
        for i in 0..6 {
            if i != axis {
                assert!((before[i] - after[i]).abs() < 2e-13);
            }
        }
        assert!(trace["inverse_probability_error"].as_f64().unwrap() < 1e-11);
        let p = c.decode(u).0;
        assert!(cfg.contains(p));
        assert!(u.iter().map(|v| v * v).sum::<f64>() <= OUTER * OUTER + 1e-12);
        assert!(
            cfg.fixed_poses
                .iter()
                .all(|q| !sphere().overlaps(&Placed::new(p), &Placed::new(*q)))
        );
        successful += 1;
    }
    assert!(successful > 40);
}

#[test]
fn hard_free_disabled_limits_and_zero_density_outside_domain() {
    let c = chart(false);
    let cfg = configuration();
    for (alpha, beta) in [(0.5, 0.), (1., 1.)] {
        let g = guide(&hard_data(alpha, beta), &c, &cfg);
        let u = [0.; 6];
        let (q, d) = g.density_details(u, true, log_volume(), &c, OUTER).unwrap();
        assert_eq!(q, g.base.log_density(u, true, log_volume()));
        assert_eq!(d["conditioning_disabled"], true);
    }
    let g = guide(&hard_data(0.5, 1.), &c, &cfg);
    let u = [2.8, 2.8, 1., 0., 0., 0.];
    assert!(u.iter().map(|v| v * v).sum::<f64>() > OUTER * OUTER);
    let (q, _) = g
        .density_details(u, false, log_volume(), &c, OUTER)
        .unwrap();
    assert_eq!(q, f64::NEG_INFINITY);
}

#[test]
fn hard_free_query_journal_retains_failure_identity() {
    let path = std::env::temp_dir().join(format!(
        "hard-free-query-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&path).unwrap();
    let mut journal = File::create(path.join("attempts.jsonl")).unwrap();
    let result: Result<()> =
        audited_query(&path, &mut journal, 3, "probe", &json!("retained"), || {
            anyhow::bail!("deliberate reference failure")
        });
    assert!(result.is_err());
    let attempt: Value =
        serde_json::from_str(&fs::read_to_string(path.join("attempts.jsonl")).unwrap()).unwrap();
    let failure: Value =
        serde_json::from_slice(&fs::read(path.join("failure.json")).unwrap()).unwrap();
    assert_eq!(attempt["ordinal"], 3);
    assert_eq!(attempt["id"], "retained");
    assert_eq!(failure["id"], attempt["id"]);
    assert_eq!(failure["complete"], false);
    fs::remove_dir_all(path).unwrap();
}
