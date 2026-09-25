//! Passive diagnostic metadata must describe every attempt without changing it.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tetramer_mc::{
    cluster_phase::{ClusterPhase, ClusterPhaseConfig, ClusterPhaseCounts, ContactGraph},
    depletion::GateOptions,
    geometry::{Atom, Shape, SphereTree},
    math::Pose,
    spherical::Container,
};

fn graph(n: usize, edges: &[(usize, usize)]) -> ContactGraph {
    let mut adjacency = vec![vec![false; n]; n];
    for &(i, j) in edges {
        adjacency[i][j] = true;
        adjacency[j][i] = true;
    }
    ContactGraph { adjacency }
}

#[test]
fn distinguishes_whole_oligomer_from_embedded_subset_and_shared_neighbors() -> Result<()> {
    let detached = graph(6, &[(0, 1), (1, 2), (0, 2), (3, 4)]);
    let attached = graph(6, &[(0, 1), (1, 2), (0, 2), (3, 4), (1, 3), (2, 3)]);
    let a = detached.subset_context(&[2, 0, 1])?;
    let b = attached.subset_context(&[0, 1, 2])?;
    assert!(a.whole_component);
    assert_eq!(a.parent_component_members, [0, 1, 2]);
    assert_eq!(
        (
            a.internal_contacts,
            a.external_contacts,
            a.external_neighbors
        ),
        (3, 0, 0)
    );
    assert!(!b.whole_component);
    assert_eq!(b.parent_component_members, [0, 1, 2, 3, 4]);
    assert_eq!(
        (
            b.internal_contacts,
            b.external_contacts,
            b.external_neighbors
        ),
        (3, 2, 1)
    );
    // The two cut edges lead to the same outsider; remote body4 is in the parent,
    // but neither it nor disconnected body5 is an immediate external neighbor.
    let pair = detached.subset_context(&[3, 4])?;
    assert!(pair.whole_component);
    assert_eq!(pair.internal_contacts, 1);
    assert!(detached.internal_equal(&attached, &[0, 1, 2]));
    Ok(())
}

#[test]
fn malformed_or_internally_disconnected_subsets_are_rejected() {
    let g = graph(4, &[(0, 1), (1, 2)]);
    for members in [vec![], vec![0, 0], vec![0, 4], vec![0, 2], vec![0, 3]] {
        assert!(g.subset_context(&members).is_err(), "{members:?}");
    }
}

#[test]
fn logging_preserves_trajectory_counts_and_every_random_stream() -> Result<()> {
    let tree = SphereTree::new(Shape {
        name: "diagnostic sphere".into(),
        volume: 0.,
        atoms: vec![Atom {
            center: [0.; 3],
            radius: 0.35,
        }],
    })?;
    let wall = Container::new(5., &tree)?;
    let phase = ClusterPhase::new(
        &tree,
        &wall,
        0.5,
        0.3,
        2.,
        GateOptions::default(),
        ClusterPhaseConfig {
            duration: 1.,
            dimer_rate: 2.,
            trimer_rate: 1.,
            transport_probability: 0.,
            local_translation_std_a: 0.6,
            local_small_angle_std_degrees: 10.,
            ..Default::default()
        },
        None,
    )?;
    let run = |record| -> Result<_> {
        let mut poses: Vec<_> = [-0.9, 0., 0.9, 2.]
            .into_iter()
            .map(|x| Pose {
                position: [x, 0., 0.],
                orientation: [1., 0., 0., 0.],
            })
            .collect();
        let mut streams: [StdRng; 5] =
            std::array::from_fn(|i| StdRng::seed_from_u64(50900 + i as u64));
        let [clock, proposal, gate, accept, bias] = &mut streams;
        let mut counts = ClusterPhaseCounts::default();
        let mut rows = vec![];
        let mut bias_state = None;
        for _ in 0..25 {
            let (c, r) = phase.run(
                &mut poses,
                clock,
                proposal,
                gate,
                accept,
                bias,
                None,
                &mut bias_state,
                record,
            )?;
            counts.add(&c);
            rows.extend(r);
        }
        let next: Vec<_> = streams.iter_mut().map(|s| s.random::<u64>()).collect();
        Ok((poses, counts, next, rows))
    };
    let (a, ca, ra, rows) = run(true)?;
    let (b, cb, rb, empty) = run(false)?;
    assert_eq!(a, b);
    assert_eq!(ca, cb);
    assert_eq!(ra, rb);
    assert!(empty.is_empty());
    assert!(ca.events > 20 && ca.hard_rejected > 0 && ca.accepted > 0);
    for event in rows.iter().filter(|r| r["kind"] == "cluster_event") {
        let c = &event["subset_context"];
        assert_eq!(
            c["subset_size"].as_u64().unwrap() as usize,
            event["members"].as_array().unwrap().len()
        );
        assert_eq!(c["whole_component"] == true, c["external_contacts"] == 0);
        assert!(c["internal_contacts"].as_u64().unwrap() > 0);
    }
    Ok(())
}
