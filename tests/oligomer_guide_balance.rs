//! Exhaustive finite-state reference for a fixed-length reversible guide proposal.
//!
//! This independently checks the algebra used by the continuous oligomer guide.
//! It does not certify its geometry, chart density, auxiliary Poisson law, or RNG.
//! No sampling error: every state pair is enumerated using ordinary f64 arithmetic.

type Matrix = Vec<Vec<f64>>;
const EPS: f64 = 3e-13;

fn fixture() -> (Vec<f64>, Vec<f64>, Matrix) {
    (
        vec![1., 7., 2., 11.],
        vec![13., 0.2, 4., 1.4],
        vec![
            vec![0.05, 0.15, 0.65, 0.15],
            vec![0.40, 0.10, 0.10, 0.40],
            vec![0.10, 0.20, 0.30, 0.40],
            vec![0.65, 0.05, 0.25, 0.05],
        ],
    )
}

fn complete(off_diagonal: &Matrix) -> Matrix {
    let mut k = off_diagonal.clone();
    for (i, row) in k.iter_mut().enumerate() {
        row[i] = 0.;
        let accepted: f64 = row.iter().sum();
        assert!(
            accepted <= 1. + EPS,
            "accepted mass exceeds one: {accepted}"
        );
        row[i] = 1. - accepted;
    }
    k
}

/// Standard MH accepts/rejects an asymmetric raw proposal. The diagonal retains
/// both raw self proposals and all rejected transitions.
fn mh(raw: &Matrix, target: &[f64]) -> Matrix {
    let n = target.len();
    let mut a = vec![vec![0.; n]; n];
    for i in 0..n {
        assert!(target[i] > 0.);
        for j in 0..n {
            if i != j && raw[i][j] > 0. {
                a[i][j] = raw[i][j] * (target[j] * raw[j][i] / (target[i] * raw[i][j])).min(1.);
            }
        }
    }
    complete(&a)
}

fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| (0..n).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}

fn power(k: &Matrix, count: usize) -> Matrix {
    let n = k.len();
    let mut p = vec![vec![0.; n]; n];
    for (i, row) in p.iter_mut().enumerate() {
        row[i] = 1.;
    }
    for _ in 0..count {
        p = matmul(&p, k);
    }
    p
}

/// Endpoint correction using guide detailed balance, without evaluating K^m's
/// transition density or the normalizing constant of H.
fn outer(proposal: &Matrix, pi: &[f64], h: &[f64]) -> Matrix {
    let n = pi.len();
    let mut a = vec![vec![0.; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                a[i][j] = proposal[i][j] * (pi[j] * h[i] / (pi[i] * h[j])).min(1.);
            }
        }
    }
    complete(&a)
}

fn invariant_error(k: &Matrix, weights: &[f64]) -> f64 {
    let z: f64 = weights.iter().sum();
    (0..weights.len())
        .map(|j| {
            ((0..weights.len())
                .map(|i| weights[i] * k[i][j])
                .sum::<f64>()
                - weights[j])
                .abs()
                / z
        })
        .fold(0., f64::max)
}

fn balance_error(k: &Matrix, weights: &[f64]) -> f64 {
    let z: f64 = weights.iter().sum();
    (0..weights.len())
        .flat_map(|i| {
            (0..weights.len()).map(move |j| (weights[i] * k[i][j] - weights[j] * k[j][i]).abs() / z)
        })
        .fold(0., f64::max)
}

fn assert_kernel(k: &Matrix, weights: &[f64]) {
    for row in k {
        assert!(
            row.iter()
                .all(|x| x.is_finite() && *x >= -EPS && *x <= 1. + EPS)
        );
        assert!((row.iter().sum::<f64>() - 1.).abs() < EPS);
    }
    assert!(
        balance_error(k, weights) < EPS,
        "balance error {}",
        balance_error(k, weights)
    );
    assert!(
        invariant_error(k, weights) < EPS,
        "invariance error {}",
        invariant_error(k, weights)
    );
}

fn assert_same(a: &Matrix, b: &Matrix) {
    for (ra, rb) in a.iter().zip(b) {
        for (&x, &y) in ra.iter().zip(rb) {
            assert!((x - y).abs() < EPS, "{x} != {y}");
        }
    }
}

#[test]
fn fixed_length_guide_then_endpoint_correction_preserves_physical_target() {
    let (pi, h, raw) = fixture();
    // The raw proposal is neither symmetric nor reversible for either target.
    assert!(balance_error(&raw, &h) > 0.01);
    assert!(balance_error(&raw, &pi) > 0.01);
    let inner = mh(&raw, &h);
    assert_kernel(&inner, &h);
    for m in [0, 1, 4, 16] {
        let proposal = power(&inner, m);
        assert_kernel(&proposal, &h);
        let corrected = outer(&proposal, &pi, &h);
        assert_kernel(&corrected, &pi);
        // Independently construct standard endpoint MH using exact matrix K^m.
        // It must agree with the H(x)/H(y) shortcut, including diagonal holds.
        assert_same(&corrected, &mh(&proposal, &pi));
        println!(
            "m={m}: guide balance={:.3e}, corrected balance={:.3e}, corrected invariant={:.3e}",
            balance_error(&proposal, &h),
            balance_error(&corrected, &pi),
            invariant_error(&corrected, &pi)
        );
    }
}

#[test]
fn arbitrary_guide_and_physical_normalizations_cancel() {
    let (pi, h, raw) = fixture();
    let scaled_h: Vec<_> = h.iter().map(|v| 17.3 * v).collect();
    let scaled_pi: Vec<_> = pi.iter().map(|v| 0.031 * v).collect();
    let inner = mh(&raw, &h);
    let scaled_inner = mh(&raw, &scaled_h);
    assert_same(&inner, &scaled_inner);
    for m in [1, 4, 16] {
        let a = outer(&power(&inner, m), &pi, &h);
        let b = outer(&power(&scaled_inner, m), &scaled_pi, &scaled_h);
        assert_same(&a, &b);
    }
}

#[test]
fn omitting_outer_correction_preserves_the_wrong_distribution() {
    let (pi, h, raw) = fixture();
    let inner = mh(&raw, &h);
    for m in [1, 4, 16] {
        let wrong = power(&inner, m);
        assert_kernel(&wrong, &h);
        assert!(invariant_error(&wrong, &pi) > 0.01);
        println!(
            "m={m}: omitted-correction physical error={:.6}",
            invariant_error(&wrong, &pi)
        );
    }
}

#[test]
fn one_inner_step_is_delayed_acceptance_and_cannot_raise_off_diagonal_flux() {
    let (pi, h, raw) = fixture();
    let delayed = outer(&mh(&raw, &h), &pi, &h);
    let direct = mh(&raw, &pi);
    let mut strict_pairs = 0;
    for i in 0..pi.len() {
        for j in 0..pi.len() {
            if i == j {
                continue;
            }
            let a = h[j] * raw[j][i] / (h[i] * raw[i][j]);
            let b = pi[j] * h[i] / (pi[i] * h[j]);
            // min(1,a) min(1,b) <= min(1,ab), for positive a,b.
            assert!(a.min(1.) * b.min(1.) <= (a * b).min(1.) + EPS);
            assert!((delayed[i][j] - raw[i][j] * a.min(1.) * b.min(1.)).abs() < EPS);
            assert!(delayed[i][j] <= direct[i][j] + EPS);
            strict_pairs += usize::from(delayed[i][j] < direct[i][j] - 1e-7);
        }
    }
    assert!(
        strict_pairs > 0,
        "fixture must expose an actual delayed-acceptance penalty"
    );
    assert_kernel(&delayed, &pi);
}

#[test]
fn rejection_self_loops_must_count_as_inner_steps() {
    let (pi, h, raw) = fixture();
    let inner = mh(&raw, &h);
    assert!((0..h.len()).any(|i| inner[i][i] > raw[i][i] + 0.05));
    // Incorrectly retrying until an accepted off-diagonal move gives the jump
    // chain. It is reversible for H(x)*escape(x), not for H(x).
    let escape: Vec<f64> = (0..h.len()).map(|i| 1. - inner[i][i]).collect();
    let mut jumped = inner.clone();
    for i in 0..h.len() {
        for j in 0..h.len() {
            jumped[i][j] = if i == j { 0. } else { inner[i][j] / escape[i] };
        }
    }
    let jumped_target: Vec<_> = h.iter().zip(&escape).map(|(x, e)| x * e).collect();
    assert_kernel(&jumped, &jumped_target);
    assert!(invariant_error(&jumped, &h) > 0.01);
    // The H-based outer correction is no longer justified after dropping holds.
    let wrong = outer(&jumped, &pi, &h);
    println!(
        "wrong state-dependent/jump correction error={:.6}",
        invariant_error(&wrong, &pi)
    );
    assert!(invariant_error(&wrong, &pi) > 1e-5);
}

#[test]
fn count_must_be_fixed_or_drawn_independently_of_the_state() {
    let (pi, h, raw) = fixture();
    let inner = mh(&raw, &h);
    let one = power(&inner, 1);
    let four = power(&inner, 4);
    let mut mixed = one.clone();
    let mut state_dependent = one.clone();
    for i in 0..h.len() {
        for j in 0..h.len() {
            mixed[i][j] = 0.25 * one[i][j] + 0.75 * four[i][j];
            state_dependent[i][j] = if i % 2 == 0 { one[i][j] } else { four[i][j] };
        }
    }
    assert_kernel(&mixed, &h);
    assert_kernel(&outer(&mixed, &pi, &h), &pi);
    assert!(invariant_error(&state_dependent, &h) > 0.005);
    let wrong = outer(&state_dependent, &pi, &h);
    println!(
        "wrong state-dependent/jump correction error={:.6}",
        invariant_error(&wrong, &pi)
    );
    assert!(invariant_error(&wrong, &pi) > 1e-5);
}
