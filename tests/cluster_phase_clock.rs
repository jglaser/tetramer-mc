//! Independent finite-state reference exposes event-indexed equilibrium bias.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tetramer_mc::{
    cluster_phase::{ClusterPhaseConfig, ContactGraph, next_wait},
    geometry::{Atom, Shape, SphereTree},
    math::*,
    rigid_subset::carry,
};

#[test]
fn fixed_horizon_matches_exact_two_state_semigroup_while_event_chain_is_biased() -> Result<()> {
    // Physical π=(1/2,1/2), exchange rate1 each way, plus9 null-attempts in Y.
    // Exact Pτ(X,Y)=(1-exp(-2τ))/2; null events must not affect it.
    let mut rng = StdRng::seed_from_u64(872341);
    let n = 40_000;
    let duration = 0.8;
    let mut y_count = 0;
    let mut rejected = 0;
    for _ in 0..n {
        let mut y = false;
        let mut t = 0.;
        loop {
            let rate = if y { 10. } else { 1. };
            let Some(dt) = next_wait(&mut rng, rate, duration - t)? else {
                break;
            };
            t += dt;
            if rng.random::<f64>() < 1. / rate {
                y = !y;
            } else {
                rejected += 1;
            }
        }
        y_count += u64::from(y);
    }
    let measured = y_count as f64 / n as f64;
    let exact = (1. - (-2. * duration).exp()) / 2.;
    assert!(
        (measured - exact).abs() < 0.014,
        "fixed-horizon {measured} != {exact}"
    );
    assert!(rejected > n, "null/rejected events must be exercised");
    // Deliberately wrong event-indexed chain targets π*Λ, i.e.10/11 in Y.
    let mut y = false;
    let mut count = 0;
    for _ in 0..n * 10 {
        if !y || rng.random::<f64>() < 0.1 {
            y = !y;
        }
        count += u64::from(y);
    }
    let event = count as f64 / (10 * n) as f64;
    assert!((event - 10. / 11.).abs() < 0.006);
    assert!((event - 0.5).abs() > 0.35);
    Ok(())
}
#[test]
fn zero_rate_and_zero_horizon_are_identity_without_random_draws() -> Result<()> {
    let mut a = StdRng::seed_from_u64(19);
    let mut b = StdRng::seed_from_u64(19);
    assert!(next_wait(&mut a, 0., 1.)?.is_none());
    assert!(next_wait(&mut a, 3., 0.)?.is_none());
    assert_eq!(a.random::<u64>(), b.random::<u64>());
    assert!(next_wait(&mut a, -1., 1.).is_err());
    Ok(())
}
#[test]
fn subsets_are_unique_connected_internal_and_remain_available_after_attachment() -> Result<()> {
    let cfg = ClusterPhaseConfig::default();
    // A triangle plus tail: all four physical vertices are connected, yet all
    // internal pairs and connected triplets are eligible independently.
    let g = ContactGraph {
        adjacency: vec![
            vec![false, true, true, false],
            vec![true, false, true, false],
            vec![true, true, false, true],
            vec![false, false, true, false],
        ],
    };
    let channels = g.channels(&cfg);
    let pairs: Vec<_> = channels.iter().filter(|c| c.members.len() == 2).collect();
    let triples: Vec<_> = channels
        .iter()
        .filter(|c| c.members.len() == 3)
        .map(|c| c.members.clone())
        .collect();
    assert_eq!(pairs.len(), 4);
    assert_eq!(triples, vec![vec![0, 1, 2], vec![0, 2, 3], vec![1, 2, 3]]);
    let mut detached = g.clone();
    detached.adjacency[2][3] = false;
    detached.adjacency[3][2] = false;
    assert!(
        detached
            .channels(&cfg)
            .iter()
            .any(|c| c.members == vec![0, 1, 2])
    );
    assert!(g.internal_equal(&detached, &[0, 1, 2]));
    assert_eq!(g.boundary_edges(&[0, 1, 2]).len(), 1);
    Ok(())
}
#[test]
fn incremental_graph_matches_full_geometry_after_rigid_dimer_move() -> Result<()> {
    let tree = SphereTree::new(Shape {
        name: "exclusion spheres".into(),
        volume: 0.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 0.7,
        }],
    })?;
    let p = |x| Pose {
        position: x,
        orientation: [1., 0., 0., 0.],
    };
    let old = vec![
        p([0., 0., 0.]),
        p([1., 0., 0.]),
        p([4., 0., 0.]),
        p([4., 2., 0.]),
    ];
    let next = carry(
        &old,
        &[0, 1],
        0,
        Pose {
            position: [3., 0., 0.],
            orientation: quaternion(cayley([0.1, 0.3, -0.2])),
        },
    )?;
    let a = ContactGraph::build(&tree, &old);
    let b = a.updated(&tree, &next, &[0, 1]);
    assert_eq!(b.adjacency, ContactGraph::build(&tree, &next).adjacency);
    assert!(a.internal_equal(&b, &[0, 1]));
    Ok(())
}
