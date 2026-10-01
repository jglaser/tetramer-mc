//! Independent proposal-law checks. Allocations, seeds and five-SE tolerances
//! are fixed here before execution; failures must not trigger sample extension.
//! No Poisson clouds, protein production sampling or fitted test thresholds.
use super::*;
#[path = "hard_free_line_tests.rs"]
mod hard_free;
use crate::geometry::{Atom, Shape};
use rand::SeedableRng;

const OUTER: f64 = 4.;
const NORMALIZATION_DRAWS: usize = 16_384;
const GUIDED_PHYSICAL_DRAWS: usize = 16_384;
const UNIFORM_REFERENCE_DRAWS: usize = 65_536;
const INVERSE_DRAWS: usize = 2_048;

fn pose(position: Vec3) -> Pose {
    Pose {
        position,
        orientation: [1., 0., 0., 0.],
    }
}

fn chart(correlated: bool) -> Chart {
    let mut lower = [[0.; 6]; 6];
    for (i, row) in lower.iter_mut().enumerate() {
        row[i] = 1.;
    }
    if correlated {
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
    }
    Chart {
        lower,
        mean: [0.; 6],
        anchor_position: [2.1, 0., 0.],
        anchor_rotation: IDENTITY,
        fixed: pose([-2.1, 0., 0.]),
        ell: 2.,
        log_det: (0..6).map(|i| lower[i][i].ln()).sum(),
    }
}

fn configuration() -> DockingConfig {
    serde_json::from_value(json!({
        "shape":"unused-sphere.json", "fixed_poses":[pose([-2.1,0.,0.]),pose([2.1,0.,0.])],
        "initial_pose":pose([0.;3]),"capture_center":[0.,0.,0.],"capture_radius":64.,
        "depletant_radius":0.1,"reservoir_density":30.,"poisson_lambda_ratio":64.,
        "translation_steps":[0.1],"rotation_steps_deg":[1.],"rotation_probability":0.5,
        "local_attempts_per_cycle":1,"uniform_probability":0.5,"seed":1
    }))
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

fn component(mean: [f64; 6], lower: [[f64; 6]; 6], weight: f64) -> Value {
    let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
    });
    json!({"weight":weight,"mean":mean,"covariance":covariance})
}

fn identity() -> [[f64; 6]; 6] {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }))
}

fn data(alpha: f64, beta: f64) -> Value {
    let mut l = identity();
    l[1][0] = 0.2;
    l[3][0] = -0.15;
    l[4][2] = 0.1;
    l[5][5] = 0.8;
    json!({"schema":"defensive-contact-line-guide-v1","region_sha256":"toy-region",
        "defensive_uniform_shell_probability":alpha,
        "gaussian_components":[component([0.;6],identity(),0.6),
            component([0.4,-0.2,0.1,0.3,0.,-0.1],l,0.4)],
        "raw_translation_axes":[0,1,2],"contact_widths_A":[0.1,0.3],
        "contact_neighbor_indices":[0,1],"conditional_probability":beta,
        "minimum_conditional_mass":1e-10})
}

fn guide(raw: &Value, chart: &Chart, cfg: &DockingConfig) -> ContactLineGuide {
    ContactLineGuide::from_bytes(
        &serde_json::to_vec(raw).unwrap(),
        "toy-region",
        chart,
        0.,
        cfg,
        &sphere(),
    )
    .unwrap()
}

fn log_volume() -> f64 {
    (PI.powi(3) * OUTER.powi(6) / 6.).ln()
}

#[derive(Default)]
struct Stats {
    n: usize,
    sum: f64,
    sum2: f64,
}
impl Stats {
    fn push(&mut self, x: f64) {
        self.n += 1;
        self.sum += x;
        self.sum2 += x * x;
    }
    fn mean(&self) -> f64 {
        self.sum / self.n as f64
    }
    fn se(&self) -> f64 {
        ((self.sum2 / self.n as f64 - self.mean().powi(2)).max(0.) / (self.n - 1) as f64).sqrt()
    }
    fn check(&self, expected: f64, name: &str) {
        eprintln!(
            "{name}: n={} mean={} reference={} SE={}",
            self.n,
            self.mean(),
            expected,
            self.se()
        );
        assert!(
            (self.mean() - expected).abs() <= 5. * self.se() + 2e-10,
            "{name}: mean={}, expected={expected}, SE={}",
            self.mean(),
            self.se()
        );
    }
}

#[test]
fn contact_line_zero_beta_and_pure_uniform_preserve_exact_stream() {
    let chart = chart(true);
    let cfg = configuration();
    for (alpha, beta) in [(0.5, 0.), (1., 1.)] {
        let g = guide(&data(alpha, beta), &chart, &cfg);
        let mut old = StdRng::seed_from_u64(202610012101);
        let mut new = StdRng::seed_from_u64(202610012101);
        for _ in 0..256 {
            let (u, radius, c) = g.base.draw(&mut old, OUTER, 0., 1.).unwrap();
            let (v, other, d, conditional) = g.draw(&mut new, &chart, OUTER, 0., 1.).unwrap();
            assert_eq!(u, v);
            assert_eq!(radius, other);
            assert_eq!(c, d);
            assert_eq!(conditional, None);
            let inside = radius <= OUTER;
            assert_eq!(
                g.base.log_density(u, inside, log_volume()),
                g.density_details(v, inside, log_volume(), &chart, OUTER)
                    .unwrap()
                    .0
            );
        }
        assert_eq!(old.random::<u64>(), new.random::<u64>());
        if alpha == 1. {
            let mut a = StdRng::seed_from_u64(202610012102);
            let mut b = StdRng::seed_from_u64(202610012102);
            for _ in 0..64 {
                let expected = draw_uniform(&mut a, OUTER, 0., 1.).unwrap();
                let actual = g.draw(&mut b, &chart, OUTER, 0., 1.).unwrap();
                assert_eq!(expected, (actual.0, actual.1));
            }
            assert_eq!(a.random::<u64>(), b.random::<u64>());
        }
    }
}

#[test]
fn contact_line_raw_axis_preserves_rotation_for_correlated_chart() {
    let mut chart = chart(true);
    chart.fixed.orientation = quaternion(cayley([0.2, -0.3, 0.1]));
    chart.anchor_rotation = cayley([-0.1, 0.15, 0.2]);
    let u = [0.6, -0.2, 0.1, 0.4, -0.1, 0.3];
    let raw = chart.coordinates(u);
    let (old, old_logj) = chart.decode(u);
    for axis in 0..3 {
        for displacement in [-1.3, 0.7] {
            let mut x = raw;
            x[axis] += displacement;
            let v = raw_to_latent(&chart, x);
            let (new, new_logj) = chart.decode(v);
            let direction = matvec(
                rotation(chart.fixed.orientation),
                std::array::from_fn(|k| if k == axis { 1. } else { 0. }),
            );
            for k in 0..3 {
                assert!(
                    (new.position[k] - old.position[k] - displacement * direction[k]).abs() < 2e-14
                );
            }
            for k in 0..4 {
                assert!((new.orientation[k] - old.orientation[k]).abs() < 2e-14);
            }
            assert!((new_logj - old_logj).abs() < 2e-14);
            let changed_angular_latents = (3..6).any(|k| (u[k] - v[k]).abs() > 1e-3);
            assert!(
                changed_angular_latents,
                "test must exercise coupled latent angular coordinates"
            );
        }
    }
}

#[test]
fn contact_line_component_specific_floor_and_empty_fallback() {
    let chart = chart(false);
    let cfg = configuration();
    let mut raw = data(0.5, 1.);
    raw["raw_translation_axes"] = json!([0]);
    raw["contact_widths_A"] = json!([0.3]);
    raw["minimum_conditional_mass"] = json!(0.02);
    raw["gaussian_components"] = json!([
        component([0.; 6], identity(), 1.),
        component([2., 0., 0., 0., 0., 0.], identity(), 1.)
    ]);
    let g = guide(&raw, &chart, &cfg);
    let u = [0.; 6];
    let x = chart.coordinates(u);
    let (sets, _) = g.intervals(x, &chart, OUTER, 0).unwrap();
    assert_eq!(sets[0].intervals().len(), 1);
    assert!((sets[0].intervals()[0].lower + 0.1).abs() < 1e-14);
    assert!((sets[0].intervals()[0].upper - 0.1).abs() < 1e-14);
    let (m0, s0) = g.conditionals[0][0].conditional(x);
    let (m1, s1) = g.conditionals[0][1].conditional(x);
    let p0 = ContactLineGuide::masses(&sets[0], m0, s0).unwrap().1;
    let p1 = ContactLineGuide::masses(&sets[0], m1, s1).unwrap().1;
    assert!(p0 > 0.02 && p1 < 0.02);
    let q0 = (-3. * (2. * PI).ln()).exp();
    let q1 = q0 * (-2_f64).exp();
    let expected = 0.5 * (-log_volume()).exp() + 0.25 * (q0 / p0 + q1);
    let (actual, details) = g
        .density_details(u, true, log_volume(), &chart, OUTER)
        .unwrap();
    assert!((actual.exp() - expected).abs() < 1e-13);
    assert_eq!(details["fallback_component_branches"], 1);
    let mut far = cfg.clone();
    far.fixed_poses = [pose([-100., 0., 0.]), pose([100., 0., 0.])].to_vec();
    let empty = guide(&raw, &chart, &far);
    for seed in 202610012110..202610012174 {
        let mut old = StdRng::seed_from_u64(seed);
        let mut new = StdRng::seed_from_u64(seed);
        let (u, r, c) = empty.base.draw(&mut old, OUTER, 0., 1.).unwrap();
        let (v, r2, c2, fallback) = empty.draw(&mut new, &chart, OUTER, 0., 1.).unwrap();
        assert_eq!((u, r, c), (v, r2, c2));
        assert_eq!(fallback, c.map(|_| true));
        assert!(
            (empty
                .density_details(u, r <= OUTER, log_volume(), &chart, OUTER)
                .unwrap()
                .0
                - empty.base.log_density(u, r <= OUTER, log_volume()))
            .abs()
                < 1e-12
        );
    }
}

#[test]
fn contact_line_all_axis_width_density_is_complete_arithmetic_mixture() {
    let chart = chart(true);
    let cfg = configuration();
    let raw = data(0.5, 0.8);
    let g = guide(&raw, &chart, &cfg);
    let mut components = vec![];
    for axis in 0..3 {
        for width in [0.1, 0.3] {
            let mut one = raw.clone();
            one["raw_translation_axes"] = json!([axis]);
            one["contact_widths_A"] = json!([width]);
            components.push(guide(&one, &chart, &cfg));
        }
    }
    let mut rng = StdRng::seed_from_u64(202610012201);
    for _ in 0..64 {
        let (u, r, _) = g.base.draw(&mut rng, OUTER, 0., 1.).unwrap();
        let inside = r <= OUTER;
        let q = g
            .density_details(u, inside, log_volume(), &chart, OUTER)
            .unwrap()
            .0
            .exp();
        let average = components
            .iter()
            .map(|one| {
                one.density_details(u, inside, log_volume(), &chart, OUTER)
                    .unwrap()
                    .0
                    .exp()
            })
            .sum::<f64>()
            / 6.;
        assert!((q - average).abs() < 2e-12 * q.max(average));
    }
}

#[test]
fn contact_line_unselected_spectator_blocks_and_capture_is_center_only() {
    let chart = chart(false);
    let mut raw = data(0.5, 1.);
    raw["raw_translation_axes"] = json!([0]);
    raw["contact_widths_A"] = json!([0.3]);
    let cfg = configuration();
    let base = guide(&raw, &chart, &cfg);
    let (ordinary, _) = base.intervals([0.; 6], &chart, OUTER, 0).unwrap();
    assert!(!ordinary[0].is_empty());
    let mut blocked = cfg.clone();
    blocked.fixed_poses.push(pose([0.; 3]));
    // Only anchors 0/1 are selected for contact, but spectator 2 still excludes
    // every central line pose. Ignoring unselected hard bodies would fail this.
    let g = guide(&raw, &chart, &blocked);
    let (sets, _) = g.intervals([0.; 6], &chart, OUTER, 0).unwrap();
    assert!(sets[0].is_empty());
    let mut clipped = cfg;
    clipped.capture_radius = 0.05;
    let g = guide(&raw, &chart, &clipped);
    let (sets, _) = g.intervals([0.; 6], &chart, OUTER, 0).unwrap();
    assert_eq!(sets[0].intervals().len(), 1);
    assert!((sets[0].intervals()[0].lower + 0.05).abs() < 1e-14);
    assert!((sets[0].intervals()[0].upper - 0.05).abs() < 1e-14);
    // The capture condition constrains the center. It is not an atomic wall:
    // a radius-one sphere cannot fit inside a physical radius-0.05 vessel.
    let atomic_wall = translation_intervals(
        &sphere(),
        pose([0.; 3]),
        [1., 0., 0.],
        &sphere(),
        &[],
        [-1., 1.],
        None,
        Some(crate::line_geometry::SphericalWall {
            center: [0.; 3],
            radius: 0.05,
        }),
    )
    .unwrap();
    assert!(atomic_wall.wall_valid.is_empty());
}

#[test]
fn contact_line_new_draw_importance_recovers_old_normalization_and_moments() {
    // beta<1 retains every Gaussian tail. beta=1 is tested on the physical R4
    // domain separately, because its defensive uniform has bounded support.
    let chart = chart(true);
    let cfg = configuration();
    let g = guide(&data(0.5, 0.7), &chart, &cfg);
    let mut rng = StdRng::seed_from_u64(202610012301);
    let mut normalization = Stats::default();
    let mut first: [Stats; 6] = std::array::from_fn(|_| Stats::default());
    let mut second: [Stats; 6] = std::array::from_fn(|_| Stats::default());
    let mut cross = Stats::default();
    let mut conditioned = 0;
    for _ in 0..NORMALIZATION_DRAWS {
        let (u, r, _, flag) = g.draw(&mut rng, &chart, OUTER, 0., 1.).unwrap();
        conditioned += usize::from(flag == Some(false));
        let old = g.base.log_density(u, r <= OUTER, log_volume());
        let new = g
            .density_details(u, r <= OUTER, log_volume(), &chart, OUTER)
            .unwrap()
            .0;
        let w = (old - new).exp();
        normalization.push(w);
        for i in 0..6 {
            first[i].push(w * u[i]);
            second[i].push(w * u[i] * u[i]);
        }
        cross.push(w * u[0] * u[3]);
    }
    assert!(
        conditioned > 100,
        "test did not exercise enough nonfallback conditioning: {conditioned}"
    );
    normalization.check(1., "E_new[q_old/q_new]");
    for i in 0..6 {
        let mean = (1. - g.base.alpha)
            * g.base
                .components
                .iter()
                .map(|c| c.weight * c.mean[i])
                .sum::<f64>();
        let square = g.base.alpha * OUTER.powi(2) / 8.
            + (1. - g.base.alpha)
                * g.base
                    .components
                    .iter()
                    .map(|c| {
                        c.weight
                            * (c.mean[i] * c.mean[i]
                                + c.lower[i].iter().map(|a| a * a).sum::<f64>())
                    })
                    .sum::<f64>();
        first[i].check(mean, &format!("old moment u[{i}]"));
        second[i].check(square, &format!("old moment u[{i}]^2"));
    }
    let expected = (1. - g.base.alpha)
        * g.base
            .components
            .iter()
            .map(|c| {
                c.weight
                    * (c.mean[0] * c.mean[3]
                        + (0..6).map(|j| c.lower[0][j] * c.lower[3][j]).sum::<f64>())
            })
            .sum::<f64>();
    cross.check(expected, "old cross moment u0*u3");
    eprintln!("nonfallback conditioned draws: {conditioned}/{NORMALIZATION_DRAWS}");
}

fn independent_uniform_ball(rng: &mut StdRng) -> [f64; 6] {
    let v: [f64; 6] = std::array::from_fn(|_| StandardNormal.sample(rng));
    let length = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let radius = OUTER * rng.random::<f64>().powf(1. / 6.);
    v.map(|x| radius * x / length)
}

/// Equal-radius analytical sphere lens. Fixed exclusion spheres are disjoint
/// (separation 4.2 > 2*1.1), so their mobile overlap contributions are additive.
fn physical_integrands(chart: &Chart, cfg: &DockingConfig, u: [f64; 6]) -> [f64; 2] {
    if u.iter().map(|x| x * x).sum::<f64>() > OUTER * OUTER {
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
    let r = 1. + cfg.depletant_radius;
    for fixed in &cfg.fixed_poses {
        let d = norm(sub(position, fixed.position));
        if d < 2. {
            return [0.; 2];
        }
        if d < 2. * r {
            overlap += PI * (4. * r + d) * (2. * r - d).powi(2) / 12.;
        }
    }
    let det = (0..6).map(|i| chart.lower[i][i]).product::<f64>();
    let c2 = (3..6).map(|i| (raw[i] / chart.ell).powi(2)).sum::<f64>();
    let jacobian = det / (chart.ell.powi(3) * PI.powi(2) * (1. + c2).powi(2));
    [jacobian, jacobian * (cfg.reservoir_density * overlap).exp()]
}

#[test]
fn contact_line_full_conditioning_recovers_R4_volume_hard_and_AO_weights() {
    let chart = chart(false);
    let cfg = configuration();
    let g = guide(&data(0.5, 1.), &chart, &cfg);
    let mut old_rng = StdRng::seed_from_u64(202610012401);
    let mut new_rng = StdRng::seed_from_u64(202610012402);
    let mut reference: [Stats; 2] = std::array::from_fn(|_| Stats::default());
    let mut estimated: [Stats; 2] = std::array::from_fn(|_| Stats::default());
    let mut uniform_mass = Stats::default();
    let mut conditioned = 0;
    for _ in 0..UNIFORM_REFERENCE_DRAWS {
        let u = independent_uniform_ball(&mut old_rng);
        for (a, v) in reference
            .iter_mut()
            .zip(physical_integrands(&chart, &cfg, u))
        {
            a.push(v * log_volume().exp());
        }
    }
    for _ in 0..GUIDED_PHYSICAL_DRAWS {
        let (u, r, _, flag) = g.draw(&mut new_rng, &chart, OUTER, 0., 1.).unwrap();
        conditioned += usize::from(flag == Some(false));
        let logq = g
            .density_details(u, r <= OUTER, log_volume(), &chart, OUTER)
            .unwrap()
            .0;
        uniform_mass.push(if r <= OUTER {
            (-log_volume() - logq).exp()
        } else {
            0.
        });
        for (a, v) in estimated
            .iter_mut()
            .zip(physical_integrands(&chart, &cfg, u))
        {
            a.push(v * (-logq).exp());
        }
    }
    assert!(conditioned > 100);
    uniform_mass.check(1., "beta=1 E_new[uniform_R4/q_new]");
    for i in 0..2 {
        let se = (estimated[i].se().powi(2) + reference[i].se().powi(2)).sqrt();
        eprintln!(
            "physical {}: guided={} +/- {}, independent uniform={} +/- {}, combined SE={se}",
            if i == 0 { "hard" } else { "AO" },
            estimated[i].mean(),
            estimated[i].se(),
            reference[i].mean(),
            reference[i].se()
        );
        assert!((estimated[i].mean() - reference[i].mean()).abs() <= 5. * se + 2e-10);
    }
    assert!(reference[1].mean() > reference[0].mean());
    assert!(estimated[1].mean() > estimated[0].mean());
}

#[test]
fn contact_line_inverse_CDF_union_probability_integral_transform() {
    let chart = chart(false);
    let mut cfg = configuration();
    cfg.fixed_poses = vec![pose([0.; 3])];
    let mut raw = data(0.1, 1.);
    raw["raw_translation_axes"] = json!([0]);
    raw["contact_neighbor_indices"] = json!([0]);
    raw["contact_widths_A"] = json!([0.4]);
    let mut l = identity();
    for i in 1..6 {
        l[i][i] = 0.03;
    }
    raw["gaussian_components"] = json!([component([0.6, 0., 0., 0., 0., 0.], l, 1.)]);
    let g = guide(&raw, &chart, &cfg);
    let mut rng = StdRng::seed_from_u64(202610012501);
    let mut pit = Stats::default();
    let mut sign_residual = Stats::default();
    let mut bins = [0usize; 10];
    for _ in 0..INVERSE_DRAWS {
        let (u, _, component, flag) = g.draw(&mut rng, &chart, OUTER, 0., 1.).unwrap();
        if component.is_none() {
            continue;
        }
        assert_eq!(flag, Some(false));
        let x = chart.coordinates(u);
        let (sets, _) = g.intervals(x, &chart, OUTER, 0).unwrap();
        assert_eq!(sets[0].intervals().len(), 2);
        let (mean, sigma) = g.conditionals[0][0].conditional(x);
        let (masses, total) = ContactLineGuide::masses(&sets[0], mean, sigma).unwrap();
        let mut below = 0.;
        for (iv, mass) in sets[0].intervals().iter().zip(&masses) {
            if x[0] >= iv.upper {
                below += mass;
            } else if x[0] > iv.lower {
                below += normal_mass((iv.lower - mean) / sigma, (x[0] - mean) / sigma).unwrap();
            }
        }
        let quantile = below / total;
        assert!((0. ..=1.).contains(&quantile));
        pit.push(quantile);
        bins[((quantile * 10.) as usize).min(9)] += 1;
        sign_residual.push((if x[0] < 0. { 1. } else { 0. }) - masses[0] / total);
    }
    assert!(pit.n > 1500);
    pit.check(0.5, "conditional inverse-CDF PIT mean");
    sign_residual.check(0., "conditional interval selection residual");
    let n = pit.n as f64;
    for count in bins {
        assert!((count as f64 - n / 10.).abs() <= 5. * (0.09 * n).sqrt());
    }
}

#[test]
fn contact_line_normal_interval_reference_fixture() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../tests/data/contact_line_normal_reference.json"
    ))
    .unwrap();
    // The independently generated fixture's documented fields are checked
    // below; central, reflected-tail and close-endpoint cases are all included.
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 8);
    for case in cases {
        let lo = case["lo"].as_f64().unwrap();
        let hi = case["hi"].as_f64().unwrap();
        let expected = case["mass"].as_f64().unwrap();
        let actual = normal_mass(lo, hi).unwrap();
        assert!(
            (actual - expected).abs() <= 3e-12 * expected.abs() + 1e-300,
            "normal mass [{lo},{hi}]: {actual} != {expected}"
        );
    }
}
