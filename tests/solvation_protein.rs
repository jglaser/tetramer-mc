//! Harano atom-type morphometric solvation of the native m6 lysozyme contact:
//! association free energy from incremental moves, against the full recomputation
//! and the Python reference score, and the cost of one incremental move.
use std::time::Instant;
use tetramer_mc::{
    geometry::Shape,
    math::Pose,
    solvation::{SolvationModel, SolvationState},
};

#[test]
fn native_m6_association_free_energy() {
    let shape: Shape =
        serde_json::from_str(&std::fs::read_to_string("examples/monomer-shape.json").unwrap())
            .unwrap();
    let types: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("examples/monomer-harano-atom-types.json").unwrap(),
    )
    .unwrap();
    let types: Vec<usize> = types["types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as usize)
        .collect();
    let centers: Vec<[f64; 3]> = shape.atoms.iter().map(|a| a.center).collect();
    let radii: Vec<f64> = shape.atoms.iter().map(|a| a.radius).collect();
    let model = SolvationModel::harano_octanol_water(types, 298.15).unwrap();
    let identity = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    let far = Pose {
        position: [200., 0., 0.],
        orientation: [1., 0., 0., 0.],
    };
    let m6 = Pose {
        position: [21.124896616345055, -21.124896616345048, -0.5015618770058232],
        orientation: [0.0, 0.7071067811865475, 0.7071067811865477, 0.0],
    };
    let t = Instant::now();
    let mut s = SolvationState::new(model, &centers, &radii, &[identity, far]).unwrap();
    let build = t.elapsed();
    let separated = s.energy();
    let t = Instant::now();
    let u = s.propose(&[(1, m6)]).unwrap();
    let step = t.elapsed();
    let df = u.delta;
    s.commit(u);
    assert!((s.energy() - s.recompute()).abs() < 1e-8 * (1. + s.energy().abs()));
    eprintln!("m6 association dF = {df:.3} kT; build {build:?}, one incremental move {step:?}");
    // Independent Python reference with the same local corner split of the Gaussian
    // curvature (dft-ref/check_m6_harano.py): -16.846 kT (area +93.0, mean -79.4,
    // Gaussian -30.4). A global vertex-count split gives +21.3 instead: the per-type
    // Gaussian terms are sensitive to how vertex curvature is shared between atoms.
    assert!((df + 16.846).abs() < 2e-3, "{df}");
}
