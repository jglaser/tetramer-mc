use super::*;
use crate::math::{IDENTITY, matmul, rotation, transpose};
use rand::{
    SeedableRng,
    distr::{Distribution, Open01},
    rngs::StdRng,
};

fn uniforms<const N: usize>(rng: &mut StdRng) -> [f64; N] {
    std::array::from_fn(|_| Open01.sample(rng))
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance,
        "{a:.17e} != {b:.17e}, tolerance {tolerance:e}"
    );
}

fn parameters() -> AzimuthParameters {
    AzimuthParameters {
        rho_floor: 1e-8,
        projection_floor: 1e-10,
        gamma_min: 0.01,
        gamma_max: PI,
        localized_probability: 0.9,
    }
}

#[test]
fn analytic_polygon_areas_and_zero_cases() {
    for (distance, expected) in [
        (1., 1.),
        (3., 0.5),
        (4., 0.),
        (4.1, 0.),
        (0., 0.),
        (0.25, 0.4375),
    ] {
        let polygon = RadiusPolygon::new(distance, [[1., 2.], [1., 2.]]).unwrap();
        close(polygon.area, expected, 1e-15);
        for vertex in &polygon.vertices {
            assert!(polygon.contains(*vertex));
        }
        if expected == 0. {
            assert!(polygon.sample_radii([0.4, 0.5, 0.6]).is_err());
        }
    }
    assert_eq!(
        RadiusPolygon::new(2., [[1., 1.], [1., 2.]]).unwrap().area,
        0.
    );
    assert!(RadiusPolygon::new(-1., [[1., 2.], [1., 2.]]).is_err());
    assert!(RadiusPolygon::new(1., [[-1., 2.], [1., 2.]]).is_err());
    assert!(RadiusPolygon::new(1., [[2., 1.], [1., 2.]]).is_err());
    assert!(RadiusPolygon::new(f64::NAN, [[1., 2.], [1., 2.]]).is_err());
}

#[test]
fn thin_polygon_areas_do_not_subtract_large_absolute_products() {
    let lo = 1e8;
    let hi = lo + 1e-4;
    let p = RadiusPolygon::new(1., [[lo, hi], [lo, hi]]).unwrap();
    close(p.area, (hi - lo).powi(2), 1e-22);
    let p = RadiusPolygon::new(1e-150, [[0., 1e-150], [0., 1e-150]]).unwrap();
    close(p.area / 1e-300, 0.5, 1e-15);
    assert!(p.area > 0.);
}

#[test]
fn polygon_area_matches_independent_decimal_floor_fixtures() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/data/contact_distance_polygon_reference.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let bounds: [[f64; 2]; 2] = serde_json::from_value(case["bounds"].clone()).unwrap();
        let polygon = RadiusPolygon::new(case["distance"].as_f64().unwrap(), bounds).unwrap();
        let expected = case["area"].as_f64().unwrap();
        if expected > 0. {
            close(polygon.area / expected, 1., 1e-12);
        } else {
            assert_eq!(polygon.area, 0.);
        }
        assert_eq!(
            polygon.area <= 1e-16,
            case["fallback_at_1e_16"].as_bool().unwrap()
        );
    }
}

#[test]
fn uniform_polygon_draws_match_analytic_triangle_moments() {
    // Fixed allocation and seed: no adaptive extension on a failed assertion.
    const N: usize = 32768;
    let mut rng = StdRng::seed_from_u64(610061001);
    let p = RadiusPolygon::new(3., [[1., 2.], [1., 2.]]).unwrap();
    let mut sums = [0.; 2];
    let mut centered = [0.; 3];
    for _ in 0..N {
        let r = p.sample_radii(uniforms(&mut rng)).unwrap();
        assert!(p.contains(r));
        for j in 0..2 {
            sums[j] += r[j];
            centered[j] += (r[j] - 5. / 3.).powi(2);
        }
        centered[2] += (r[0] - 5. / 3.) * (r[1] - 5. / 3.);
    }
    for sum in sums {
        close(sum / N as f64, 5. / 3., 5. * (1. / (18. * N as f64)).sqrt());
    }
    for value in &centered[..2] {
        close(value / N as f64, 1. / 18., 0.002);
    }
    close(centered[2] / N as f64, -1. / 36., 0.002);
    assert!(p.sample_radii([0., 0.5, 0.5]).is_err());
    assert!(p.sample_radii([0.5, 1., 0.5]).is_err());
}

#[test]
fn deterministic_basis_and_forward_inverse() {
    let frame = ContactFrame::new([0.2, -0.7, 0.9], [1.1, 0.8, -0.4]).unwrap();
    close(norm(frame.axis), 1., 1e-15);
    for b in frame.basis {
        close(norm(b), 1., 1e-15);
        close(dot(b, frame.axis), 0., 2e-16);
    }
    close(dot(frame.basis[0], frame.basis[1]), 0., 2e-16);
    close(
        dot(cross(frame.basis[0], frame.basis[1]), frame.axis),
        1.,
        1e-15,
    );
    let again = ContactFrame::new(frame.centers[0], frame.centers[1]).unwrap();
    assert_eq!(frame.basis, again.basis);
    let polygon = RadiusPolygon::new(frame.distance, [[1.2, 2.2], [1.4, 2.4]]).unwrap();
    let mut rng = StdRng::seed_from_u64(610061002);
    for _ in 0..2048 {
        let radii = polygon.sample_radii(uniforms(&mut rng)).unwrap();
        let circle = frame.circle(radii).unwrap();
        let phi = TAU * uniforms::<1>(&mut rng)[0];
        let point = circle.point(phi).unwrap();
        let recovered = frame.encode(point).unwrap().unwrap();
        for j in 0..2 {
            close(recovered.radii[j], radii[j], 3e-15);
        }
        close((recovered.phi - phi).sin(), 0., 4e-15);
        close(recovered.transverse_radius, circle.radius, 3e-15);
    }
}

#[test]
fn translation_jacobian_and_density_normalization() {
    let frame = ContactFrame::new([-0.3, 0.2, 0.8], [1.9, 0.6, -0.1]).unwrap();
    let radii = [2.1, 1.7];
    let phi = 1.2;
    let epsilon = 1e-5;
    let mut derivatives = [[0.; 3]; 3];
    for axis in 0..3 {
        let mut minus = [radii[0], radii[1], phi];
        let mut plus = minus;
        minus[axis] -= epsilon;
        plus[axis] += epsilon;
        let a = frame
            .circle([minus[0], minus[1]])
            .unwrap()
            .point(minus[2])
            .unwrap();
        let b = frame
            .circle([plus[0], plus[1]])
            .unwrap()
            .point(plus[2])
            .unwrap();
        derivatives[axis] = scale(sub(b, a), 1. / (2. * epsilon));
    }
    let determinant = dot(derivatives[0], cross(derivatives[1], derivatives[2])).abs();
    close(determinant, radii[0] * radii[1] / frame.distance, 2e-9);
    // Independent Cartesian quadrature of the uniform-azimuth density is hard
    // at shell boundaries. Instead integrate its full f*Jacobian over an exact
    // rectangle-radius support and a deterministic angular grid.
    let polygon = RadiusPolygon::new(frame.distance, [[2., 2.2], [1.6, 1.8]]).unwrap();
    close(polygon.area, 0.04, 1e-15);
    let law = AzimuthLaw::new(0.7, 0.18, 0.9).unwrap();
    let n = 8192;
    let mut integral = 0.;
    for i in 0..n {
        let phi = TAU * (i as f64 + 0.5) / n as f64;
        let point = frame.circle(radii).unwrap().point(phi).unwrap();
        integral += frame
            .cartesian_log_density(point, &polygon, &law)
            .unwrap()
            .exp()
            * radii[0]
            * radii[1]
            / frame.distance
            * polygon.area
            * TAU
            / n as f64;
    }
    close(integral, 1., 3e-14);
    assert_eq!(
        frame
            .cartesian_log_density([20., 20., 20.], &polygon, &law)
            .unwrap(),
        f64::NEG_INFINITY
    );
}

#[test]
fn wrapped_cauchy_inverse_density_and_fourier_moments() {
    for gamma in [1e-7, 0.01, 0.2, 1., PI, 50.] {
        let law = AzimuthLaw::new(0.7, gamma, 0.9).unwrap();
        for u in [1e-7, 0.01, 0.2, 0.5, 0.8, 0.99, 1. - 1e-7] {
            let phi = law.wrapped_inverse(u).unwrap();
            close(
                law.wrapped_cdf(phi).unwrap(),
                u,
                if gamma < 1e-6 { 2e-9 } else { 3e-13 },
            );
        }
        let peak = ((1. - law.localized_probability)
            + law.localized_probability / (gamma / 2.).tanh())
            / TAU;
        close(law.density(law.mode).unwrap() / peak, 1., 3e-15);
        assert!(law.log_density(law.mode + PI).unwrap().is_finite());
    }
    // Extremely narrow peaks remain finite in log space.
    let narrow = AzimuthLaw::new(0., 1e-200, 0.9).unwrap();
    assert!(narrow.log_density(0.).unwrap().is_finite());
    const N: usize = 65536;
    let law = AzimuthLaw::new(1.1, 0.2, 0.9).unwrap();
    let mut rng = StdRng::seed_from_u64(610061003);
    let mut sums = [[0.; 2]; 3];
    for _ in 0..N {
        let phi = law.sample(uniforms(&mut rng)).unwrap();
        for (i, sum) in sums.iter_mut().enumerate() {
            let (sin, cos) = ((i + 1) as f64 * (phi - law.mode)).sin_cos();
            sum[0] += cos;
            sum[1] += sin;
        }
    }
    for (i, sum) in sums.iter().enumerate() {
        let expected = law.localized_probability * (-(i as f64 + 1.) * law.gamma).exp();
        close(sum[0] / N as f64, expected, 5. / (N as f64).sqrt());
        close(sum[1] / N as f64, 0., 5. / (N as f64).sqrt());
    }
}

#[test]
fn uniform_limit_and_declared_projection_fallback() {
    let law = AzimuthLaw::new(2.7, 0.01, 0.).unwrap();
    for u in [0.01, 0.3, 0.99] {
        assert_eq!(law.sample([0.2, u]).unwrap(), TAU * u);
        assert_eq!(law.log_density(u).unwrap(), -TAU.ln());
    }
    let frame = ContactFrame::new([0., 0., 0.], [2., 0., 0.]).unwrap();
    let circle = frame.circle([2., 2.]).unwrap();
    assert_eq!(
        circle
            .azimuth_from_gaussian(circle.center, IDENTITY, parameters())
            .unwrap()
            .localized_probability,
        0.
    );
    let mean = circle.point(0.7).unwrap();
    let covariance = [[0.0001, 0., 0.], [0., 0.0001, 0.], [0., 0., 0.0001]];
    let law = circle
        .azimuth_from_gaussian(mean, covariance, parameters())
        .unwrap();
    close(law.mode, 0.7, 1e-15);
    assert_eq!(law.gamma, parameters().gamma_min);
    let broad = circle
        .azimuth_from_gaussian(mean, [[100.; 3]; 3], parameters())
        .unwrap();
    assert_eq!(broad.gamma, parameters().gamma_max);
    assert!(
        circle
            .azimuth_from_gaussian(mean, [[0.; 3]; 3], parameters())
            .is_err()
    );
    let mut invalid = parameters();
    invalid.localized_probability = 1.;
    assert!(invalid.validate().is_err());
}

#[test]
fn label_exchange_preserves_cartesian_density_and_localization() {
    let first = ContactFrame::new([0.3, -0.6, 0.7], [2.1, 0.3, -0.4]).unwrap();
    let second = ContactFrame::new(first.centers[1], first.centers[0]).unwrap();
    let p1 = RadiusPolygon::new(first.distance, [[1.4, 2.2], [1.7, 2.4]]).unwrap();
    let p2 = RadiusPolygon::new(second.distance, [[1.7, 2.4], [1.4, 2.2]]).unwrap();
    close(p1.area, p2.area, 1e-15);
    let r = rotation([0.5, 0.5, 0.5, 0.5]);
    let covariance = matmul(
        matmul(r, [[0.03, 0., 0.], [0., 0.01, 0.], [0., 0., 0.02]]),
        transpose(r),
    );
    let mean = [1., 1.3, -0.2];
    let mut rng = StdRng::seed_from_u64(610061004);
    for _ in 0..4096 {
        let radii = p1.sample_radii(uniforms(&mut rng)).unwrap();
        let circle = first.circle(radii).unwrap();
        let law = circle
            .azimuth_from_gaussian(mean, covariance, parameters())
            .unwrap();
        let point = circle
            .point(law.sample(uniforms(&mut rng)).unwrap())
            .unwrap();
        let inverse = second.encode(point).unwrap().unwrap();
        let reverse_circle = second.circle(inverse.radii).unwrap();
        let reverse_law = reverse_circle
            .azimuth_from_gaussian(mean, covariance, parameters())
            .unwrap();
        let a = first.cartesian_log_density(point, &p1, &law).unwrap();
        let b = second
            .cartesian_log_density(point, &p2, &reverse_law)
            .unwrap();
        close(a, b, 2e-12);
    }
}

#[test]
fn tangent_and_near_degenerate_cases_are_explicit() {
    assert!(ContactFrame::new([0.; 3], [0.; 3]).is_err());
    let frame = ContactFrame::new([0.; 3], [2., 0., 0.]).unwrap();
    assert_eq!(frame.circle([1., 1.]).unwrap().radius, 0.);
    assert!(frame.encode([1., 0., 0.]).unwrap().is_none());
    assert!(frame.circle([0.99, 0.99]).is_err());
    assert!(frame.encode([f64::NAN, 0., 0.]).is_err());
    let epsilon = 1e-12;
    let circle = frame.circle([1. + epsilon, 1. + epsilon]).unwrap();
    let represented_epsilon = (1. + epsilon) - 1.;
    close(
        circle.radius / (represented_epsilon * (2. + represented_epsilon)).sqrt(),
        1.,
        5e-16,
    );
    let inverse = frame.encode(circle.point(1.3).unwrap()).unwrap().unwrap();
    close(inverse.phi, 1.3, 1e-15);
    let almost_coincident = ContactFrame::new([0.; 3], [1e-10, 0., 0.]).unwrap();
    let circle = almost_coincident.circle([2., 2.]).unwrap();
    close(circle.radius, 2., 1e-15);
    close(circle.center[0], 5e-11, 1e-25);
}
