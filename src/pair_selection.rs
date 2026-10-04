//! Standalone defensive selection of an ordered pair in ordinary space.
//!
//! The root is uniform. Conditional on that retained root, a nonempty center
//! neighborhood supplies `(1-eta)/degree` and every other label supplies
//! `eta/(N-1)`; an empty neighborhood uses the uniform law alone. Neighborhoods
//! include the cutoff boundary. No periodic minimum image or hard-core test is
//! implied. This module is not wired into any simulation or acceptance gate.
//!
//! Only the ordered (root, partner) is the selection label. The drawn branch is
//! diagnostic; densities always marginalize it. Add the returned reverse-minus-
//! forward log probability to the SAME final physical decision, before state
//! commitment. This factor alone is not a complete physical acceptance rule.
//!
//! Every endpoint evaluation scans its actual poses, including a moved root
//! and partner. No source-neighborhood cache is accepted: unbounded proposals
//! can enter new neighborhoods. Work is O(N) per root, not an N² graph rebuild;
//! this exhaustive implementation does not claim intensive runtime.
use crate::math::{Pose, norm, sub};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DefensivePairSelector {
    cutoff: f64,
    defensive_probability: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PairSelectionBranch {
    Neighbor,
    Defensive,
    EmptyNeighborhood,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PairSelectionDensity {
    pub bodies: usize,
    pub root: usize,
    pub partner: usize,
    pub degree: usize,
    pub partner_is_neighbor: bool,
    pub log_partner_probability: f64,
    /// Includes the uniform root probability 1/N.
    pub log_ordered_probability: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SelectedPair {
    pub selector: DefensivePairSelector,
    pub branch: PairSelectionBranch,
    pub density: PairSelectionDensity,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PairSelectionCorrection {
    pub forward: PairSelectionDensity,
    pub reverse: PairSelectionDensity,
    pub log_reverse_minus_forward: f64,
}

impl DefensivePairSelector {
    /// A zero cutoff is allowed and includes exactly coincident centers.
    pub fn new(cutoff: f64, defensive_probability: f64) -> Result<Self> {
        ensure!(
            cutoff.is_finite() && cutoff >= 0.,
            "invalid pair-selection cutoff"
        );
        ensure!(
            defensive_probability.is_finite()
                && defensive_probability > 0.
                && defensive_probability < 1.,
            "pair-selection defensive probability must be in (0, 1)"
        );
        Ok(Self {
            cutoff,
            defensive_probability,
        })
    }

    fn neighbors(&self, state: &[Pose], root: usize) -> Result<Vec<usize>> {
        ensure!(
            state.len() >= 2 && root < state.len(),
            "invalid pair-selection population/root"
        );
        for pose in state {
            pose.validate()?;
        }
        Ok((0..state.len())
            .filter(|&j| {
                j != root && norm(sub(state[j].position, state[root].position)) <= self.cutoff
            })
            .collect())
    }

    fn density(
        &self,
        bodies: usize,
        root: usize,
        partner: usize,
        degree: usize,
        neighbor: bool,
    ) -> PairSelectionDensity {
        let log_uniform = -((bodies - 1) as f64).ln();
        let log_partner_probability = if degree == 0 {
            log_uniform
        } else {
            let defensive = self.defensive_probability.ln() + log_uniform;
            if neighbor {
                let local = (-self.defensive_probability).ln_1p() - (degree as f64).ln();
                let hi = local.max(defensive);
                hi + (local.min(defensive) - hi).exp().ln_1p()
            } else {
                defensive
            }
        };
        PairSelectionDensity {
            bodies,
            root,
            partner,
            degree,
            partner_is_neighbor: neighbor,
            log_partner_probability,
            log_ordered_probability: log_partner_probability - (bodies as f64).ln(),
        }
    }

    /// Complete positive selection law at the supplied CURRENT endpoint.
    /// Log-space evaluation keeps tiny defensive tails from underflowing to a
    /// false zero. Self-pairs/out-of-range labels are invalid API inputs, not
    /// eligible selections with a minus-infinity density.
    pub fn evaluate(
        &self,
        state: &[Pose],
        root: usize,
        partner: usize,
    ) -> Result<PairSelectionDensity> {
        ensure!(
            partner < state.len() && root != partner,
            "invalid pair-selection partner"
        );
        let neighbors = self.neighbors(state, root)?;
        Ok(self.density(
            state.len(),
            root,
            partner,
            neighbors.len(),
            neighbors.contains(&partner),
        ))
    }

    pub fn draw(&self, state: &[Pose], rng: &mut StdRng) -> Result<SelectedPair> {
        ensure!(state.len() >= 2, "pair selection needs at least two bodies");
        // Validate before consuming selection randomness, including invalid poses.
        for pose in state {
            pose.validate()?;
        }
        let root = rng.random_range(0..state.len());
        let neighbors = self.neighbors(state, root)?;
        let branch = if neighbors.is_empty() {
            PairSelectionBranch::EmptyNeighborhood
        } else if rng.random::<f64>() < self.defensive_probability {
            PairSelectionBranch::Defensive
        } else {
            PairSelectionBranch::Neighbor
        };
        let partner = if branch == PairSelectionBranch::Neighbor {
            neighbors[rng.random_range(0..neighbors.len())]
        } else {
            let rank = rng.random_range(0..state.len() - 1);
            rank + usize::from(rank >= root)
        };
        Ok(SelectedPair {
            selector: *self,
            branch,
            density: self.density(
                state.len(),
                root,
                partner,
                neighbors.len(),
                neighbors.contains(&partner),
            ),
        })
    }

    /// Recompute BOTH endpoints with the same retained ordered labels and
    /// selector parameters. Supply all actual trial poses, not a stale list.
    /// N must stay fixed, so the two uniform-root terms cancel. Since eta>0,
    /// changing eligibility never creates zero reverse support.
    pub fn log_reverse_minus_forward(
        &self,
        old: &[Pose],
        proposed: &[Pose],
        root: usize,
        partner: usize,
    ) -> Result<PairSelectionCorrection> {
        ensure!(
            old.len() == proposed.len(),
            "pair-selection population changed"
        );
        let forward = self.evaluate(old, root, partner)?;
        let reverse = self.evaluate(proposed, root, partner)?;
        let log_reverse_minus_forward =
            reverse.log_partner_probability - forward.log_partner_probability;
        ensure!(
            log_reverse_minus_forward.is_finite(),
            "nonfinite pair-selection correction"
        );
        Ok(PairSelectionCorrection {
            forward,
            reverse,
            log_reverse_minus_forward,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn pose(x: f64) -> Pose {
        Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        }
    }
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 2e-14, "{a} != {b}");
    }

    #[test]
    fn enumerated_law_normalizes_including_empty_and_boundary_neighborhoods() {
        let selector = DefensivePairSelector::new(1., 0.2).unwrap();
        for xs in [vec![0., 1.], vec![0., 0.5, 1., 4.], vec![0., 3., 6., 9.]] {
            let state: Vec<_> = xs.into_iter().map(pose).collect();
            let mut total = 0.;
            for root in 0..state.len() {
                let mut conditional = 0.;
                for partner in 0..state.len() {
                    if root == partner {
                        continue;
                    }
                    let law = selector.evaluate(&state, root, partner).unwrap();
                    assert!(law.log_partner_probability.is_finite());
                    conditional += law.log_partner_probability.exp();
                    total += law.log_ordered_probability.exp();
                    if law.degree == 0 {
                        close(
                            law.log_partner_probability.exp(),
                            1. / (state.len() - 1) as f64,
                        );
                    }
                }
                close(conditional, 1.);
            }
            close(total, 1.);
        }
        let boundary = selector
            .evaluate(&[pose(0.), pose(1.), pose(3.)], 0, 1)
            .unwrap();
        assert_eq!(boundary.degree, 1);
        assert!(boundary.partner_is_neighbor);
    }

    #[test]
    fn evaluates_full_mixture_not_the_generation_branch() {
        let selector = DefensivePairSelector::new(1., 0.2).unwrap();
        let state = [pose(0.), pose(0.5), pose(4.), pose(8.)];
        let local = selector.evaluate(&state, 0, 1).unwrap();
        close(local.log_partner_probability.exp(), 0.8 + 0.2 / 3.);
        assert!((local.log_partner_probability.exp() - 0.8).abs() > 0.01);
        close(
            selector
                .evaluate(&state, 0, 2)
                .unwrap()
                .log_partner_probability
                .exp(),
            0.2 / 3.,
        );
        let empty = selector.evaluate(&state, 3, 0).unwrap();
        assert_eq!(empty.degree, 0);
        close(empty.log_partner_probability.exp(), 1. / 3.);
        // Log density remains finite even when eta/(N-1) is not representable.
        let tiny = DefensivePairSelector::new(1., f64::from_bits(1)).unwrap();
        let law = tiny.evaluate(&state, 0, 2).unwrap();
        assert!(law.log_partner_probability.is_finite());
        assert!(law.log_partner_probability < f64::from_bits(1).ln());
    }

    #[test]
    fn actual_endpoint_recomputes_degree_when_both_labels_move() {
        let selector = DefensivePairSelector::new(1., 0.2).unwrap();
        let old = [pose(0.), pose(0.5), pose(2.), pose(3.), pose(5.)];
        let mut proposed = old;
        proposed[0] = pose(2.5);
        proposed[1] = pose(10.);
        let change = selector
            .log_reverse_minus_forward(&old, &proposed, 0, 1)
            .unwrap();
        assert_eq!((change.forward.degree, change.reverse.degree), (1, 2));
        assert!(change.forward.partner_is_neighbor && !change.reverse.partner_is_neighbor);
        close(change.log_reverse_minus_forward, (0.05_f64 / 0.85).ln());
        let back = selector
            .log_reverse_minus_forward(&proposed, &old, 0, 1)
            .unwrap();
        close(
            change.log_reverse_minus_forward,
            -back.log_reverse_minus_forward,
        );
        proposed[0] = pose(20.);
        let empty = selector
            .log_reverse_minus_forward(&old, &proposed, 0, 1)
            .unwrap();
        assert_eq!(empty.reverse.degree, 0);
        close(empty.log_reverse_minus_forward, (0.25_f64 / 0.85).ln());
    }

    #[test]
    fn sampling_retains_root_full_law_and_replays_seeded_prefix() {
        let selector = DefensivePairSelector::new(1., 0.4).unwrap();
        let state = [pose(0.), pose(0.5), pose(4.), pose(8.)];
        let mut first = StdRng::seed_from_u64(9041);
        for _ in 0..7 {
            selector.draw(&state, &mut first).unwrap();
        }
        // StdRng in the pinned rand release is not Clone. Reconstruct the
        // retained RNG point by replaying its seed and consumed prefix.
        let mut resumed = StdRng::seed_from_u64(9041);
        for _ in 0..7 {
            selector.draw(&state, &mut resumed).unwrap();
        }
        let mut roots = [false; 4];
        let mut branches = [false; 3];
        for _ in 0..256 {
            let sample = selector.draw(&state, &mut first).unwrap();
            assert_eq!(sample, selector.draw(&state, &mut resumed).unwrap());
            assert_eq!(
                sample.density,
                selector
                    .evaluate(&state, sample.density.root, sample.density.partner)
                    .unwrap()
            );
            roots[sample.density.root] = true;
            branches[match sample.branch {
                PairSelectionBranch::Neighbor => 0,
                PairSelectionBranch::Defensive => 1,
                PairSelectionBranch::EmptyNeighborhood => 2,
            }] = true;
        }
        assert!(roots.into_iter().all(|v| v));
        assert!(branches.into_iter().all(|v| v));
    }

    #[test]
    fn invalid_configuration_or_population_is_an_error() {
        for eta in [0., 1., -0.1, f64::NAN, f64::INFINITY] {
            assert!(DefensivePairSelector::new(1., eta).is_err());
        }
        for cutoff in [-1., f64::NAN, f64::INFINITY] {
            assert!(DefensivePairSelector::new(cutoff, 0.2).is_err());
        }
        let selector = DefensivePairSelector::new(0., 0.2).unwrap();
        assert_eq!(
            selector
                .evaluate(&[pose(0.), pose(0.)], 0, 1)
                .unwrap()
                .degree,
            1
        );
        assert!(selector.evaluate(&[pose(0.)], 0, 0).is_err());
        assert!(selector.evaluate(&[pose(0.), pose(1.)], 2, 0).is_err());
        assert!(selector.evaluate(&[pose(0.), pose(1.)], 0, 0).is_err());
        assert!(
            selector
                .evaluate(&[pose(0.), pose(f64::NAN)], 0, 1)
                .is_err()
        );
        assert!(
            selector
                .log_reverse_minus_forward(&[pose(0.), pose(1.)], &[pose(0.)], 0, 1)
                .is_err()
        );
    }

    #[test]
    fn enumerated_toy_balance_exposes_omitted_and_wrong_sign_corrections() {
        let selector = DefensivePairSelector::new(1., 0.2).unwrap();
        let states: Vec<Vec<Pose>> = (0..8)
            .map(|s| {
                (0..3)
                    .map(|i| pose(i as f64 * 0.7 + ((s >> i) & 1) as f64 * 2.8))
                    .collect()
            })
            .collect();
        let pi: [f64; 8] = [1., 2., 4., 3., 5., 8., 7., 6.].map(|w| w / 36.);
        let mut residuals = Vec::new();
        for sign in [1., 0., -1.] {
            let mut transition = [[0.; 8]; 8];
            for x in 0..8 {
                for root in 0..3 {
                    for partner in 0..3 {
                        if root == partner {
                            continue;
                        }
                        // Label-dependent symmetric involution: toggle only this
                        // partner's binary location. Rejection stays at x.
                        let y = x ^ (1 << partner);
                        let correction = selector
                            .log_reverse_minus_forward(&states[x], &states[y], root, partner)
                            .unwrap();
                        let selected = correction.forward.log_ordered_probability.exp();
                        let acceptance = ((pi[y] / pi[x]).ln()
                            + sign * correction.log_reverse_minus_forward)
                            .min(0.)
                            .exp();
                        transition[x][y] += selected * acceptance;
                        transition[x][x] += selected * (1. - acceptance);
                    }
                }
                close(transition[x].iter().sum(), 1.);
            }
            let mut residual = 0_f64;
            for x in 0..8 {
                for y in 0..8 {
                    residual =
                        residual.max((pi[x] * transition[x][y] - pi[y] * transition[y][x]).abs());
                }
                if sign == 1. {
                    close((0..8).map(|y| pi[y] * transition[y][x]).sum(), pi[x]);
                }
            }
            residuals.push(residual);
        }
        assert!(residuals[0] < 2e-14, "corrected total flow: {residuals:?}");
        assert!(
            residuals[1] > 1e-4,
            "omitted correction escaped control: {residuals:?}"
        );
        assert!(
            residuals[2] > 1e-4,
            "wrong sign escaped control: {residuals:?}"
        );
    }
}
