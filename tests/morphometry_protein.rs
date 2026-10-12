//! Exact union measures of the lysozyme monomer accessible union (atom radii + 1.4 A)
//! and of the native m6 crystal pair, against an independent reference: the Python
//! implementation (dft-ref/union_measures.py) with chi from the weighted alpha complex.
use std::f64::consts::PI;
use tetramer_mc::{geometry::Shape, math::Pose, morphometry::union_measures};

fn monomer() -> (Vec<[f64; 3]>, Vec<f64>) {
    let shape: Shape =
        serde_json::from_str(&std::fs::read_to_string("examples/monomer-shape.json").unwrap())
            .unwrap();
    (
        shape.atoms.iter().map(|a| a.center).collect(),
        shape.atoms.iter().map(|a| a.radius + 1.4).collect(),
    )
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1. + b.abs())
}

#[test]
fn monomer_and_native_pair_match_reference() {
    let (x, r) = monomer();
    let a = union_measures(&x, &r).unwrap();
    assert!(close(a.volume, 23884.59081812342, 1e-9), "{}", a.volume);
    assert!(close(a.area, 6562.865094358206, 1e-9), "{}", a.area);
    assert!(close(a.mean, 294.95810964727644, 1e-9), "{}", a.mean);
    assert!(
        close(a.mean_steiner, -198.2336506527945, 1e-9),
        "{}",
        a.mean_steiner
    );
    assert_eq!(a.euler(), 10);
    assert!((a.gauss / (4. * PI) - 10.).abs() < 1e-6);
    let pose = Pose {
        position: [21.124896616345055, -21.124896616345048, -0.5015618770058232],
        orientation: [0.0, 0.7071067811865475, 0.7071067811865477, 0.0],
    };
    let xb: Vec<[f64; 3]> = x.iter().map(|&c| pose.apply(c)).collect();
    let both: Vec<[f64; 3]> = x.iter().chain(&xb).copied().collect();
    let rr: Vec<f64> = r.iter().chain(&r).copied().collect();
    let ab = union_measures(&both, &rr).unwrap();
    assert!(close(ab.volume - 2. * a.volume, -698.5965988859825, 1e-7));
    assert!(close(ab.area - 2. * a.area, -1074.0045426989927, 1e-8));
    assert!(close(ab.mean - 2. * a.mean, -229.45927975248128, 1e-8));
    assert_eq!(ab.euler() - 2 * a.euler(), 2);
    // Per-ball parts sum to the union values.
    let s: f64 = ab.balls.iter().map(|b| b.area).sum();
    assert!(close(s, ab.area, 1e-12));
}
