//! Independent enumeration and retained-auxiliary support checks.
use anyhow::Result;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tetramer_mc::cluster_phase::{
    AnchorPoolSelection, ClusterPhaseConfig, ContactGraph, TransportCharts, contact_anchor_pool,
};

fn graph(n: usize, edges: &[[usize; 2]]) -> ContactGraph {
    let mut adjacency = vec![vec![false; n]; n];
    for &[i, j] in edges {
        adjacency[i][j] = true;
        adjacency[j][i] = true;
    }
    ContactGraph { adjacency }
}

fn enumerate(prefix: &mut Vec<usize>, labels: &[usize], k: usize, out: &mut Vec<Vec<usize>>) {
    if prefix.len() == k {
        out.push(prefix.clone());
    } else {
        for &a in labels {
            if !prefix.contains(&a) {
                prefix.push(a);
                enumerate(prefix, labels, k, out);
                prefix.pop();
            }
        }
    }
}

#[test]
fn contact_pool_ordered_law_normalizes_and_handles_exhausted_contacts() -> Result<()> {
    // Includes a two-member subset: contact weights count edges, not neighbors.
    for g in [graph(6, &[]), graph(6, &[[0, 2], [1, 2], [0, 3], [2, 4]])] {
        for eps in [0.01, 0.2, 1.] {
            for k in 1..=4 {
                let mut pools = vec![];
                enumerate(&mut vec![], &[2, 3, 4, 5], k, &mut pools);
                let total: f64 = pools
                    .iter()
                    .map(|p| {
                        g.anchor_pool_log_probability(&[0, 1], p, eps)
                            .unwrap()
                            .exp()
                    })
                    .sum();
                assert!(
                    (total - 1.).abs() < 1e-12,
                    "k={k}, eps={eps}, total={total}"
                );
            }
        }
    }
    let g = graph(5, &[[0, 1], [0, 2]]);
    let p = g.remaining_anchor_probabilities(&[0], &[1, 2], 0.1)?;
    assert_eq!(p, vec![0., 0., 0., 0.5, 0.5]);
    assert!(g.anchor_pool_log_probability(&[0], &[1, 1], 0.1).is_err());
    assert!(g.anchor_pool_log_probability(&[0], &[1, 0], 0.1).is_err());
    assert!(g.anchor_pool_log_probability(&[0], &[], 0.1).is_err());
    Ok(())
}

#[test]
fn contact_pool_secondary_sampling_and_reverse_support_after_contact_exchange() -> Result<()> {
    let x = graph(5, &[[0, 1], [0, 2]]);
    let y = graph(5, &[[0, 3]]);
    let pool = [1, 2];
    let fx = x.anchor_pool_log_probability(&[0], &pool, 0.2)?;
    let fy = y.anchor_pool_log_probability(&[0], &pool, 0.2)?;
    assert!((fx.exp() - 0.45 * (0.8 + 0.2 / 3.)).abs() < 1e-14);
    assert!((fy.exp() - 0.05 * 0.2 / 3.).abs() < 1e-14);
    // The retained tuple remains possible after both selected contacts vanish.
    assert!(fx.is_finite() && fy.is_finite());
    let mut rng = StdRng::seed_from_u64(71038);
    let mut hits = 0;
    for _ in 0..20_000 {
        let p = contact_anchor_pool(&mut rng, &x, &[0], 1, 2, 0.2)?;
        assert_eq!(p[0], 1);
        assert_ne!(p[1], 1);
        hits += usize::from(p[1] == 2);
    }
    assert!((hits as f64 / 20_000. - (0.8 + 0.2 / 3.)).abs() < 0.01);
    Ok(())
}

#[test]
fn contact_pool_one_anchor_preserves_rng_and_default_config_is_unchanged() -> Result<()> {
    let g = graph(3, &[[0, 1]]);
    let mut a = StdRng::seed_from_u64(81);
    let mut b = StdRng::seed_from_u64(81);
    assert_eq!(contact_anchor_pool(&mut a, &g, &[0], 1, 1, 0.1)?, vec![1]);
    assert_eq!(a.random::<u64>(), b.random::<u64>());
    assert_eq!(
        g.anchor_pool_log_probability(&[0], &[1], 0.1)?,
        g.anchor_probabilities(&[0], 0.1)?[1].ln()
    );
    let mut cfg = ClusterPhaseConfig::default();
    assert!(
        serde_json::to_value(&cfg)?
            .get("anchor_pool_selection")
            .is_none()
    );
    cfg.anchor_pool_selection = AnchorPoolSelection::ContactWithoutReplacement;
    assert!(cfg.validate().is_err());
    cfg.transport_charts = TransportCharts::Members;
    assert!(cfg.validate().is_err());
    cfg.anchor_contact_uniform_probability = Some(0.1);
    cfg.validate()?;
    assert_eq!(
        serde_json::to_value(&cfg)?["anchor_pool_selection"],
        "contact_without_replacement"
    );
    Ok(())
}
