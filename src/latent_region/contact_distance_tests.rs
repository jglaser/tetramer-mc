//! Fixed-allocation proposal-law tests: no Poisson clouds or protein sampling.
use super::*;
use crate::geometry::Atom;

const OUTER: f64 = 4.;
const DRAWS: usize = 16_384;

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}

fn chart() -> Chart {
    let mut lower = identity();
    lower[0][0] = 0.8;
    lower[1][0] = 0.25;
    lower[1][1] = 1.1;
    lower[2][1] = -0.15;
    lower[2][2] = 0.9;
    lower[3][0] = 0.7;
    lower[3][1] = -0.2;
    lower[3][3] = 0.6;
    lower[4][1] = -0.45;
    lower[4][4] = 0.7;
    lower[5][0] = -0.3;
    lower[5][2] = 0.4;
    lower[5][5] = 0.8;
    let mut fixed = pose([-1.5, 0., 0.]);
    fixed.orientation = [0.2_f64.cos(), 0., 0.2_f64.sin(), 0.];
    Chart {
        lower,
        mean: [0.1, -0.2, 0.3, 0.04, -0.05, 0.06],
        anchor_position: [1.5, 0., 0.],
        anchor_rotation: IDENTITY,
        fixed,
        ell: 2.,
        log_det: (0..6).map(|i| lower[i][i].ln()).sum(),
    }
}

fn configuration() -> DockingConfig {
    serde_json::from_value(json!({"shape":"unused.json",
        "fixed_poses":[pose([-1.5,0.,0.]),pose([1.5,0.,0.])],
        "initial_pose":pose([0.,1.5,0.]),"capture_center":[0.,0.,0.],"capture_radius":64.,
        "depletant_radius":0.1,"reservoir_density":30.,"poisson_lambda_ratio":64.,
        "translation_steps":[0.1],"rotation_steps_deg":[1.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.5,"seed":1}))
    .unwrap()
}

fn sphere() -> SphereTree {
    SphereTree::new(Shape {
        name: "unit-sphere".into(),
        volume: 4. * PI / 3.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 1.,
        }],
    })
    .unwrap()
}

fn identity() -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }))
}

fn component(mean: [f64; 6], lower: [[f64; 6]; 6], weight: f64) -> Value {
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    json!({"weight":weight,"mean":mean,"covariance":covariance})
}

fn data(alpha: f64, beta: f64) -> Value {
    let mut lower = identity();
    lower[1][0] = 0.2;
    lower[3][0] = -0.15;
    lower[4][2] = 0.1;
    lower[5][5] = 0.8;
    let pair = json!([{"neighbor_index":0,"moving_atom":0,"fixed_atom":0},
                      {"neighbor_index":1,"moving_atom":0,"fixed_atom":0}]);
    json!({"schema":"defensive-contact-distance-guide-v1","region_sha256":"toy-region",
        "defensive_uniform_shell_probability":alpha,
        "gaussian_components":[component([0.;6],identity(),0.6),
            component([0.4,-0.2,0.1,0.3,0.,-0.1],lower,0.4)],
        "component_contact_pairs":[pair,pair], "contact_widths_A":[0.1,0.3],
        "contact_neighbor_indices":[0,1],"conditional_probability":beta,
        "minimum_center_distance":1e-8,"minimum_polygon_area":1e-16,
        "azimuth":{"localized_probability":0.9,"radius_floor":1e-8,"projection_floor":1e-10,
                   "gamma_min":0.01,"gamma_max":PI}})
}

fn guide(
    raw: &Value,
    chart: &Chart,
    cfg: &DockingConfig,
    tree: &SphereTree,
) -> ContactDistanceGuide {
    ContactDistanceGuide::from_bytes(
        &serde_json::to_vec(raw).unwrap(),
        "toy-region",
        chart,
        0.,
        cfg,
        tree,
    )
    .unwrap()
}

fn log_volume() -> f64 {
    (PI.powi(3) * OUTER.powi(6) / 6.).ln()
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() < tolerance,
        "{a:.17e} != {b:.17e}, tolerance={tolerance}"
    );
}

#[derive(Default)]
struct Stats {
    n: usize,
    sum: f64,
    sum2: f64,
}
impl Stats {
    fn push(&mut self, value: f64) {
        self.n += 1;
        self.sum += value;
        self.sum2 += value * value;
    }
    fn mean(&self) -> f64 {
        self.sum / self.n as f64
    }
    fn se(&self) -> f64 {
        ((self.sum2 / self.n as f64 - self.mean().powi(2)).max(0.) / (self.n - 1) as f64).sqrt()
    }
    fn check(&self, expected: f64, label: &str) {
        eprintln!(
            "{label}: n={} mean={} expected={expected} SE={}",
            self.n,
            self.mean(),
            self.se()
        );
        assert!(
            (self.mean() - expected).abs() <= 5. * self.se() + 2e-10,
            "{label} failed fixed five-SE criterion"
        );
    }
}

#[test]
fn distance_zero_beta_and_uniform_preserve_exact_old_stream() {
    let chart = chart();
    for (alpha, beta) in [(0.5, 0.), (1., 1.)] {
        let guide = guide(&data(alpha, beta), &chart, &configuration(), &sphere());
        let mut old = StdRng::seed_from_u64(610062001);
        let mut new = StdRng::seed_from_u64(610062001);
        for _ in 0..256 {
            let expected = guide.base.draw(&mut old, OUTER, 0., 1.).unwrap();
            let actual = guide.draw(&mut new, &chart, OUTER, 0., 1.).unwrap();
            assert_eq!(expected, (actual.0, actual.1, actual.2));
            assert_eq!(actual.3, None);
            assert_eq!(
                guide
                    .base
                    .log_density(actual.0, actual.1 <= OUTER, log_volume()),
                guide
                    .density_details(actual.0, actual.1 <= OUTER, log_volume(), &chart, OUTER)
                    .unwrap()
                    .0
            );
        }
        assert_eq!(old.random::<u64>(), new.random::<u64>());
    }
}

#[test]
fn distance_orientation_and_all_fallbacks_never_redraw() {
    let chart = chart();
    let tree = sphere();
    for reason in ["center_distance", "polygon_area"] {
        let mut cfg = configuration();
        if reason == "center_distance" {
            cfg.fixed_poses[1] = cfg.fixed_poses[0];
        } else {
            cfg.fixed_poses[1].position = [30., 0., 0.];
        }
        let guide = guide(&data(0.5, 1.), &chart, &cfg, &tree);
        let mut old = StdRng::seed_from_u64(610062002);
        let mut new = StdRng::seed_from_u64(610062002);
        for _ in 0..256 {
            let expected = guide.base.draw(&mut old, OUTER, 0., 1.).unwrap();
            if expected.2.is_some() {
                let _ = old.random::<f64>();
                let _ = old.random_range(0..guide.widths.len());
            }
            let actual = guide.draw(&mut new, &chart, OUTER, 0., 1.).unwrap();
            assert_eq!(expected, (actual.0, actual.1, actual.2));
            assert_eq!(actual.3, expected.2.map(|_| true));
            if actual.2.is_some() {
                assert_eq!(guide.last_draw.borrow()["fallback_reason"], reason);
            }
            close(
                guide
                    .density_details(actual.0, actual.1 <= OUTER, log_volume(), &chart, OUTER)
                    .unwrap()
                    .0,
                guide
                    .base
                    .log_density(actual.0, actual.1 <= OUTER, log_volume()),
                2e-14,
            );
        }
        assert_eq!(old.random::<u64>(), new.random::<u64>());
    }
}

#[test]
fn distance_component_and_width_fallbacks_enter_the_full_mixture() {
    let chart = chart();
    let cfg = configuration();
    let tree = SphereTree::new(Shape {
        name: "two-atoms".into(),
        volume: 8. * PI / 3.,
        atoms: vec![
            Atom {
                center: [0.; 3],
                radius: 1.,
            },
            Atom {
                center: [20., 0., 0.],
                radius: 1.,
            },
        ],
    })
    .unwrap();
    let mut raw = data(0.5, 0.8);
    raw["minimum_polygon_area"] = json!(0.02); // width .1 fails; .3 has area .09.
    raw["component_contact_pairs"][1][1]["fixed_atom"] = json!(1); // second component: D=23, always empty.
    let combined = guide(&raw, &chart, &cfg, &tree);
    let mut singles = vec![];
    for ci in 0..2 {
        for width in [0.1, 0.3] {
            let mut one = raw.clone();
            one["gaussian_components"] = json!([raw["gaussian_components"][ci]]);
            one["gaussian_components"][0]["weight"] = json!(1.);
            one["component_contact_pairs"] = json!([raw["component_contact_pairs"][ci]]);
            one["contact_widths_A"] = json!([width]);
            singles.push((
                combined.base.components[ci].weight / 2.,
                guide(&one, &chart, &cfg, &tree),
            ));
        }
    }
    let mut rng = StdRng::seed_from_u64(610062003);
    for _ in 0..128 {
        let (u, radius, _) = combined.base.draw(&mut rng, OUTER, 0., 1.).unwrap();
        let (q, details) = combined
            .density_details(u, radius <= OUTER, log_volume(), &chart, OUTER)
            .unwrap();
        assert_eq!(details["fallback_component_branches"], 3);
        let expected: f64 = singles
            .iter()
            .map(|(weight, single)| {
                weight
                    * single
                        .density_details(u, radius <= OUTER, log_volume(), &chart, OUTER)
                        .unwrap()
                        .0
                        .exp()
            })
            .sum();
        close(q.exp(), expected, 1e-13 * expected.max(1e-15));
    }
}

fn inverse3(a: Mat3) -> Mat3 {
    let cofactors: Mat3 = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let rows: Vec<_> = (0..3).filter(|k| *k != i).collect();
            let cols: Vec<_> = (0..3).filter(|k| *k != j).collect();
            let sign = if (i + j) % 2 == 0 { 1. } else { -1. };
            sign * (a[rows[0]][cols[0]] * a[rows[1]][cols[1]]
                - a[rows[0]][cols[1]] * a[rows[1]][cols[0]])
        })
    });
    let determinant = dot(a[0], cofactors[0]);
    transpose(cofactors).map(|row| row.map(|x| x / determinant))
}

#[test]
fn distance_correlated_chart_conditional_matches_independent_schur_complement() {
    let chart = chart();
    let guide = guide(&data(0.5, 0.7), &chart, &configuration(), &sphere());
    let raw = [0.2, -0.1, 0.4, 0.7, -0.3, 0.8];
    for (component, conditional) in guide.base.components.iter().zip(&guide.conditionals) {
        let product: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6)
                    .map(|k| chart.lower[i][k] * component.lower[k][j])
                    .sum()
            })
        });
        let cov: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..6).map(|k| product[i][k] * product[j][k]).sum())
        });
        let angular: Mat3 = std::array::from_fn(|i| std::array::from_fn(|j| cov[i + 3][j + 3]));
        let cross_cov: Mat3 = std::array::from_fn(|i| std::array::from_fn(|j| cov[i][j + 3]));
        let regression = matmul(cross_cov, inverse3(angular));
        let mean = chart.coordinates(component.mean);
        let residual = std::array::from_fn(|i| raw[i + 3] - mean[i + 3]);
        let expected_mean = add([mean[0], mean[1], mean[2]], matvec(regression, residual));
        let correction = matmul(regression, transpose(cross_cov));
        let schur: Mat3 =
            std::array::from_fn(|i| std::array::from_fn(|j| cov[i][j] - correction[i][j]));
        let r = rotation(chart.fixed.orientation);
        let world = matmul(matmul(r, schur), transpose(r));
        let actual_mean = conditional.conditional(raw, &chart).1;
        let expected_mean = chart.fixed.apply(add(chart.anchor_position, expected_mean));
        for i in 0..3 {
            close(actual_mean[i], expected_mean[i], 2e-15);
            for j in 0..3 {
                close(conditional.world_covariance[i][j], world[i][j], 2e-15);
            }
        }
        // Independent factorization: Gaussian in latent u equals the angular
        // density times conditional world translation density times det(L0).
        let u = latent(&chart, raw);
        let (pose, _) = chart.decode(u);
        let delta = sub(pose.position, actual_mean);
        let inverse = inverse3(world);
        let determinant = dot(
            world[0],
            [
                world[1][1] * world[2][2] - world[1][2] * world[2][1],
                world[1][2] * world[2][0] - world[1][0] * world[2][2],
                world[1][0] * world[2][1] - world[1][1] * world[2][0],
            ],
        );
        let translation_log = -1.5 * (2. * PI).ln()
            - 0.5 * determinant.ln()
            - 0.5 * dot(delta, matvec(inverse, delta));
        close(
            gaussian_log(component, u),
            chart.log_det + conditional.conditional(raw, &chart).0 + translation_log,
            4e-15,
        );
    }
}

fn lens(a: f64, b: f64, d: f64) -> f64 {
    PI * (a + b - d).powi(2) * (d * d + 2. * d * (a + b) - 3. * (a - b).powi(2)) / (12. * d)
}

#[test]
fn distance_full_mixture_normalization_angular_marginal_and_shell_volume() {
    let chart = chart();
    let cfg = configuration();
    let guide = guide(&data(0.5, 0.7), &chart, &cfg, &sphere());
    let mut rng = StdRng::seed_from_u64(610062004);
    let mut normalization = Stats::default();
    let mut first: [Stats; 6] = std::array::from_fn(|_| Stats::default());
    let mut second: [Stats; 6] = std::array::from_fn(|_| Stats::default());
    let mut angular_first: [Stats; 3] = std::array::from_fn(|_| Stats::default());
    let mut angular_second: [Stats; 3] = std::array::from_fn(|_| Stats::default());
    let mut shell_volume = Stats::default();
    let mut conditioned = 0;
    for _ in 0..DRAWS {
        let (u, radius, _, flag) = guide.draw(&mut rng, &chart, OUTER, 0., 1.).unwrap();
        let raw = chart.coordinates(u);
        if flag == Some(false) {
            conditioned += 1;
            let trace = guide.last_draw.borrow();
            let original: [f64; 6] =
                serde_json::from_value(trace["original_latent"].clone()).unwrap();
            let old_raw = chart.coordinates(original);
            for j in 3..6 {
                close(raw[j], old_raw[j], 3e-15);
            }
            let before = rotation(chart.decode(original).0.orientation);
            let after = rotation(chart.decode(u).0.orientation);
            for i in 0..3 {
                for j in 0..3 {
                    close(before[i][j], after[i][j], 4e-15);
                }
            }
        }
        let old = guide.base.log_density(u, radius <= OUTER, log_volume());
        let q = guide
            .density_details(u, radius <= OUTER, log_volume(), &chart, OUTER)
            .unwrap()
            .0;
        assert!(q.is_finite() && q + 1e-12 >= old + (1. - guide.beta).ln());
        let w = (old - q).exp();
        normalization.push(w);
        for j in 0..6 {
            first[j].push(w * u[j]);
            second[j].push(w * u[j] * u[j]);
        }
        for j in 0..3 {
            angular_first[j].push(raw[j + 3]);
            angular_second[j].push(raw[j + 3] * raw[j + 3]);
        }
        let point = chart.decode(u).0.position;
        let in_shell = cfg.fixed_poses.iter().all(|p| {
            let r = norm(sub(point, p.position));
            (2. ..=2.3).contains(&r)
        });
        let angular_log = guide
            .conditionals
            .iter()
            .zip(&guide.base.components)
            .fold(f64::NEG_INFINITY, |sum, (c, g)| {
                log_add(sum, g.weight.ln() + c.conditional(raw, &chart).0)
            });
        shell_volume.push(if in_shell {
            (chart.log_det + angular_log - q).exp()
        } else {
            0.
        });
    }
    assert!(conditioned > 3000);
    normalization.check(1., "distance E_new[q_old/q_new]");
    for j in 0..6 {
        let mean = (1. - guide.base.alpha)
            * guide
                .base
                .components
                .iter()
                .map(|c| c.weight * c.mean[j])
                .sum::<f64>();
        let square = guide.base.alpha * OUTER.powi(2) / 8.
            + (1. - guide.base.alpha)
                * guide
                    .base
                    .components
                    .iter()
                    .map(|c| {
                        c.weight
                            * (c.mean[j] * c.mean[j]
                                + c.lower[j].iter().map(|x| x * x).sum::<f64>())
                    })
                    .sum::<f64>();
        first[j].check(mean, &format!("distance old first u{j}"));
        second[j].check(square, &format!("distance old second u{j}"));
    }
    for j in 3..6 {
        let mut mean = guide.base.alpha * chart.mean[j];
        let mut second = guide.base.alpha
            * (chart.mean[j] * chart.mean[j]
                + OUTER.powi(2) / 8. * chart.lower[j].iter().map(|x| x * x).sum::<f64>());
        for c in &guide.base.components {
            let m = chart.coordinates(c.mean)[j];
            let row: [f64; 6] =
                std::array::from_fn(|k| (0..6).map(|l| chart.lower[j][l] * c.lower[l][k]).sum());
            mean += (1. - guide.base.alpha) * c.weight * m;
            second += (1. - guide.base.alpha)
                * c.weight
                * (m * m + row.iter().map(|x| x * x).sum::<f64>());
        }
        angular_first[j - 3].check(mean, &format!("distance new angular first {j}"));
        angular_second[j - 3].check(second, &format!("distance new angular second {j}"));
    }
    let exact = lens(2.3, 2.3, 3.) - 2. * lens(2., 2.3, 3.) + lens(2., 2., 3.);
    let radius_integral = 2. * PI / 3. * ((2.3_f64.powi(2) - 4.) / 2.).powi(2);
    close(exact, radius_integral, 3e-15);
    shell_volume.check(exact, "distance analytic two-shell intersection volume");
    eprintln!("distance conditioned={conditioned}/{DRAWS}");
}

#[test]
fn distance_unselected_blocker_retains_invalid_draws_and_target_zeros() {
    let chart = chart();
    let cfg = configuration();
    let mut blocked = cfg.clone();
    blocked.fixed_poses.push(pose([0.; 3]));
    let tree = sphere();
    let raw = data(0.5, 1.);
    let first = guide(&raw, &chart, &cfg, &tree);
    let second = guide(&raw, &chart, &blocked, &tree);
    let mut a = StdRng::seed_from_u64(610062005);
    let mut b = StdRng::seed_from_u64(610062005);
    let mut invalid = 0;
    for _ in 0..256 {
        let x = first.draw(&mut a, &chart, OUTER, 0., 1.).unwrap();
        let y = second.draw(&mut b, &chart, OUTER, 0., 1.).unwrap();
        assert_eq!(x, y);
        assert_eq!(
            first
                .density_details(x.0, x.1 <= OUTER, log_volume(), &chart, OUTER)
                .unwrap()
                .0,
            second
                .density_details(y.0, y.1 <= OUTER, log_volume(), &chart, OUTER)
                .unwrap()
                .0
        );
        if x.3 == Some(false) {
            let point = chart.decode(x.0).0.position;
            assert!(
                cfg.fixed_poses
                    .iter()
                    .all(|p| norm(sub(point, p.position)) >= 2. - 1e-14)
            );
            assert!(norm(point) < 2.); // Every geometric draw overlaps the unselected third sphere.
            let hard_valid = blocked
                .fixed_poses
                .iter()
                .all(|p| norm(sub(point, p.position)) >= 2.);
            assert!(!hard_valid);
            let unconditional_contribution = if hard_valid { 1. } else { 0. };
            assert_eq!(unconditional_contribution, 0.);
            invalid += 1;
        }
    }
    assert!(invalid > 90);
    assert_eq!(a.random::<u64>(), b.random::<u64>());
}

fn independent_ball(rng: &mut StdRng) -> [f64; 6] {
    let direction: [f64; 6] = std::array::from_fn(|_| StandardNormal.sample(rng));
    let length = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    let radius = OUTER * rng.random::<f64>().powf(1. / 6.);
    direction.map(|x| radius * x / length)
}

fn independent_physical_integrands(chart: &Chart, cfg: &DockingConfig, u: [f64; 6]) -> [f64; 2] {
    if u.iter().map(|x| x * x).sum::<f64>() > OUTER.powi(2) {
        return [0.; 2];
    }
    let raw: [f64; 6] = std::array::from_fn(|i| {
        chart.mean[i] + (0..6).map(|j| chart.lower[i][j] * u[j]).sum::<f64>()
    });
    let position = chart
        .fixed
        .apply(add(chart.anchor_position, [raw[0], raw[1], raw[2]]));
    if norm(sub(position, cfg.capture_center)) > cfg.capture_radius {
        return [0.; 2];
    }
    let mut overlap = 0.;
    let radius = 1. + cfg.depletant_radius;
    for fixed in &cfg.fixed_poses {
        let d = norm(sub(position, fixed.position));
        if d < 2. {
            return [0.; 2];
        }
        if d < 2. * radius {
            overlap += PI * (4. * radius + d) * (2. * radius - d).powi(2) / 12.;
        }
    }
    // The fixed exclusion balls are disjoint, so this sum has no omitted
    // three-body intersection term. These are analytic AO weights, not clouds.
    let determinant = (0..6).map(|i| chart.lower[i][i]).product::<f64>();
    let c2 = (3..6).map(|i| (raw[i] / chart.ell).powi(2)).sum::<f64>();
    let jacobian = determinant / (chart.ell.powi(3) * PI.powi(2) * (1. + c2).powi(2));
    [jacobian, jacobian * (cfg.reservoir_density * overlap).exp()]
}

#[test]
fn distance_full_conditioning_preserves_physical_hard_and_depletion_integrals() {
    const UNIFORM_DRAWS: usize = 65_536;
    let chart = chart();
    let mut cfg = configuration();
    cfg.fixed_poses = vec![pose([-2.1, 0., 0.]), pose([2.1, 0., 0.])];
    assert!(
        norm(sub(
            cfg.fixed_poses[0].position,
            cfg.fixed_poses[1].position
        )) > 2. * (1. + cfg.depletant_radius)
    );
    let guide = guide(&data(0.5, 1.), &chart, &cfg, &sphere());
    let mut uniform_rng = StdRng::seed_from_u64(610062006);
    let mut guided_rng = StdRng::seed_from_u64(610062007);
    let mut reference: [Stats; 2] = std::array::from_fn(|_| Stats::default());
    let mut measured: [Stats; 2] = std::array::from_fn(|_| Stats::default());
    let mut uniform_mass = Stats::default();
    let mut conditioned = 0;
    for _ in 0..UNIFORM_DRAWS {
        let u = independent_ball(&mut uniform_rng);
        for (stat, value) in reference
            .iter_mut()
            .zip(independent_physical_integrands(&chart, &cfg, u))
        {
            stat.push(value * log_volume().exp());
        }
    }
    for _ in 0..DRAWS {
        let (u, radius, _, flag) = guide.draw(&mut guided_rng, &chart, OUTER, 0., 1.).unwrap();
        conditioned += usize::from(flag == Some(false));
        let q = guide
            .density_details(u, radius <= OUTER, log_volume(), &chart, OUTER)
            .unwrap()
            .0;
        uniform_mass.push(if radius <= OUTER {
            (-log_volume() - q).exp()
        } else {
            0.
        });
        for (stat, value) in measured
            .iter_mut()
            .zip(independent_physical_integrands(&chart, &cfg, u))
        {
            stat.push(value * (-q).exp());
        }
    }
    assert!(conditioned > 1000);
    uniform_mass.check(1., "distance beta=1 uniform-R4 normalization");
    for i in 0..2 {
        let se = measured[i].se().hypot(reference[i].se());
        eprintln!(
            "distance physical {}: guided={} SE={}, uniform={} SE={}, combinedSE={se}",
            if i == 0 { "hard" } else { "AO" },
            measured[i].mean(),
            measured[i].se(),
            reference[i].mean(),
            reference[i].se()
        );
        assert!((measured[i].mean() - reference[i].mean()).abs() <= 5. * se + 2e-10);
    }
    assert!(measured[1].mean() > measured[0].mean());
    assert!(reference[1].mean() > reference[0].mean());
}
