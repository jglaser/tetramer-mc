//! Composable fixed-label, nonperiodic updates for an evolving configuration.
//!
//! Each elementary update is reversible for the hard-shape/ideal-bath target.
//! Labels, cap law, atlas and root-body guidance cloud are state independent.
//! Local moves are NOT conditioned on preserving the selected pair's contact.
//! The factorized move is the identity outside that contact domain. Composing
//! these updates preserves pi; a deterministic ordered sweep need not itself
//! be reversible. Changing spectators requires a fresh context (built here on
//! every attempt). This module makes no claim about equilibration or assembly.
use crate::{
    auxiliary_overlap_threshold::AuxiliaryOverlapThreshold,
    bounded_singleton_path::{Budget, bounded_path, bounded_singleton},
    capped_dimer::FixedDimerContext,
    depletion::GateOptions,
    factorized_dimer::FactorizedDimerProposal,
    geometry::SphereTree,
    math::Pose,
    rigid_subset::RigidSubset,
    simulation::spherical_local_pose,
    singleton_path::SingletonPath,
    spherical::Container,
};
use anyhow::{Result, ensure};
use rand::{RngExt, distr::Open01, rngs::StdRng};
use serde_json::{Value, json};

/// All paths, including fatal failures, leave `state` unchanged until acceptance.
/// A caller must durably drain the supplied record before propagating an error.
pub struct FixedLabelUpdates<'a> {
    pub core: &'a SphereTree,
    pub exclusion: &'a SphereTree,
    pub wall: &'a Container,
    pub radius: f64,
    pub members: [usize; 2],
    pub anchor: usize,
    pub rd: f64,
    pub activity: f64,
    pub lambda: f64,
    pub envelope: GateOptions,
}

fn log_value(value: f64) -> Value {
    if value == f64::NEG_INFINITY {
        json!("-inf")
    } else {
        json!(value)
    }
}

impl FixedLabelUpdates<'_> {
    pub fn selected(&self, state: &[Pose]) -> [Pose; 2] {
        [state[self.members[0]], state[self.members[1]]]
    }

    pub fn context<'a>(&'a self, state: &[Pose]) -> Result<FixedDimerContext<'a>> {
        FixedDimerContext::new(
            self.core,
            self.exclusion,
            state,
            self.members,
            self.anchor,
            Some(self.radius),
            [0.; 3],
        )
    }

    /// The unchanged production Gaussian translation/Cayley rotation proposal.
    /// The angle parameter has exactly the production convention (Cayley-vector
    /// component standard deviation equals half the supplied angle in radians).
    pub fn local(
        &self,
        state: &mut [Pose],
        member_slot: usize,
        translation_std: f64,
        angle_std_degrees: f64,
        proposal_rng: &mut StdRng,
        bath_rng: &mut StdRng,
        accept_rng: &mut StdRng,
        budget: &mut Budget,
        record: &mut Value,
    ) -> Result<()> {
        ensure!(member_slot < 2, "invalid local member slot");
        ensure!(
            translation_std.is_finite()
                && translation_std >= 0.
                && angle_std_degrees.is_finite()
                && angle_std_degrees >= 0.,
            "invalid local step"
        );
        budget.check_cpu()?;
        let member = self.members[member_slot];
        let old = state[member];
        let new = spherical_local_pose(
            proposal_rng,
            old,
            translation_std,
            angle_std_degrees.to_radians() / 2.,
        );
        *record = json!({"kind":"local", "member":member, "old":old,
            "proposed":new, "accepted":false, "status":"in_progress"});
        let gate = RigidSubset::new(self.core, state, &[member], member, new, self.rd)?;
        if !gate.hard_valid(Some(self.wall), [0.; 3]) {
            record["status"] = json!("hard_rejected");
            return Ok(());
        }
        let bath = match bounded_singleton(
            &gate,
            bath_rng,
            self.lambda,
            self.activity,
            self.envelope,
            budget,
        ) {
            Ok(result) => result,
            Err(error) => {
                record["bath_failure"] = json!(error);
                anyhow::bail!("bounded local bath failure");
            }
        };
        let log_u = accept_rng.sample::<f64, _>(Open01).ln();
        let accepted = log_u < bath.log_weight.min(0.);
        record["bath"] = json!(bath);
        record["log_u"] = json!(log_u);
        record["log_acceptance_ratio"] = json!(bath.log_weight);
        record["accepted"] = json!(accepted);
        record["status"] = json!("completed");
        if accepted {
            state[member] = new;
        }
        Ok(())
    }

    pub fn dimer(
        &self,
        state: &mut [Pose],
        proposal: &FactorizedDimerProposal<'_>,
        guide: Option<&AuxiliaryOverlapThreshold<'_>>,
        proposal_rng: &mut StdRng,
        threshold_rng: &mut StdRng,
        bath_rng: &mut StdRng,
        accept_rng: &mut StdRng,
        budget: &mut Budget,
        record: &mut Value,
    ) -> Result<()> {
        budget.check_cpu()?;
        let old = self.selected(state);
        *record = json!({"kind":"factorized_dimer", "members":self.members,
            "anchor":self.anchor, "old":old, "accepted":false, "status":"in_progress"});
        let context = self.context(state)?;
        let result = if let Some(guide) = guide {
            proposal.propose_guided(proposal_rng, threshold_rng, &context, old, guide)
        } else {
            proposal.propose(proposal_rng, &context, old)
        };
        let outcome = match result {
            Ok(value) => value,
            Err(error) => {
                record["proposal_failure"] = json!(error);
                anyhow::bail!("factorized proposal failure");
            }
        };
        record["proposal"] = json!(outcome);
        let Some(candidate) = &outcome.candidate else {
            record["status"] = json!("proposal_self_loop");
            return Ok(());
        };
        // One accessor, one summed correction, one physical acceptance decision.
        let correction = outcome.complete_log_correction()?;
        let new = [candidate.root, candidate.child];
        record["complete_log_correction"] = log_value(correction);
        record["proposed"] = json!(new);
        let path = SingletonPath::new(self.core, state, &self.members, &new, self.rd)?;
        ensure!(
            path.hard_valid(Some(self.wall), [0.; 3]),
            "final hard predicates disagree"
        );
        let bath = match bounded_path(
            self.core,
            state,
            self.members,
            new,
            &path,
            bath_rng,
            self.rd,
            self.lambda,
            self.activity,
            self.envelope,
            budget,
            None,
        ) {
            Ok(value) => value,
            Err(error) => {
                record["bath_failure"] = json!(error);
                anyhow::bail!("bounded dimer bath failure");
            }
        };
        let log_ratio = correction + bath.aggregate.log_weight;
        ensure!(
            log_ratio.is_finite() || log_ratio == f64::NEG_INFINITY,
            "invalid MH ratio"
        );
        let log_u = accept_rng.sample::<f64, _>(Open01).ln();
        let accepted = log_u < log_ratio.min(0.);
        record["bath"] = json!(bath);
        record["log_u"] = json!(log_u);
        record["log_acceptance_ratio"] = log_value(log_ratio);
        record["accepted"] = json!(accepted);
        record["status"] = json!("completed");
        if accepted {
            state[self.members[0]] = new[0];
            state[self.members[1]] = new[1];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bounded_singleton_path::Limits,
        defensive_dimer_proposal::DefensiveDimerProposal,
        docking::{DockingMethod, DockingProposal},
        factorized_dimer::{FactorizedDimerCaps, FactorizedDimerOrder},
        geometry::{Atom, Placed, Shape},
        math::rotation,
        proposal::FrozenRelativePoseProposal,
    };
    use rand::SeedableRng;
    const ID: Pose = Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    };
    fn sphere(r: f64) -> SphereTree {
        SphereTree::new(Shape {
            name: "test sphere".into(),
            volume: 4. * std::f64::consts::PI * r.powi(3) / 3.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: r,
            }],
        })
        .unwrap()
    }
    fn budget() -> Budget {
        Budget::new(Limits {
            raw_per_leg: 1_000_000,
            raw_per_outer: 2_000_000,
            raw_campaign: 20_000_000,
            retained_per_leg: 1_000_000,
            retained_per_outer: 2_000_000,
            retained_campaign: 20_000_000,
            cpu_seconds: 60.,
        })
        .unwrap()
    }
    fn atlas() -> DockingProposal {
        let shape_sha = "0".repeat(64);
        let covariance: [[f64; 6]; 6] =
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }));
        let model=json!({"coordinate_convention":"anchor-body-relative","shape_sha256":shape_sha,
            "angular_length":1.,"weights":[1.],"anchors":[{"position":[0.,0.,0.],"rotation":rotation(ID.orientation)}],
            "means":[[0.,0.,0.,0.,0.,0.]],"covariances":[covariance]}).to_string();
        DockingProposal::new(
            FrozenRelativePoseProposal::from_json_str_open(&model, [100.; 3], 0.1, &shape_sha)
                .unwrap(),
            DockingMethod::PosteriorInvolution,
            0.,
            [0.; 3],
        )
        .unwrap()
    }
    fn state() -> Vec<Pose> {
        vec![
            ID,
            Pose {
                position: [1.05, 0., 0.],
                ..ID
            },
            Pose {
                position: [5., 0., 0.],
                ..ID
            },
        ]
    }
    fn engine<'a>(
        core: &'a SphereTree,
        excluded: &'a SphereTree,
        wall: &'a Container,
    ) -> FixedLabelUpdates<'a> {
        FixedLabelUpdates {
            core,
            exclusion: excluded,
            wall,
            radius: 50.,
            members: [0, 1],
            anchor: 2,
            rd: 0.15,
            activity: 0.05,
            lambda: 3.2,
            envelope: GateOptions {
                max_cells: 63,
                max_depth: 6,
                min_width: 0.,
            },
        }
    }
    #[test]
    fn local_can_detach_and_preserves_spectators() -> Result<()> {
        let core = sphere(0.45);
        let ex = sphere(0.6);
        let wall = Container::new(50., &core)?;
        let mut e = engine(&core, &ex, &wall);
        e.activity = 0.;
        let mut s = state();
        let fixed = s[2];
        let mut record = Value::Null;
        let mut p = StdRng::seed_from_u64(112);
        let mut b = StdRng::seed_from_u64(113);
        let mut a = StdRng::seed_from_u64(114);
        e.local(
            &mut s,
            1,
            5.,
            1.,
            &mut p,
            &mut b,
            &mut a,
            &mut budget(),
            &mut record,
        )?;
        assert_eq!(record["accepted"], true);
        assert_eq!(s[2], fixed);
        assert!(!ex.overlaps(&Placed::new(s[0]), &Placed::new(s[1])));
        Ok(())
    }
    #[test]
    fn disconnected_dimer_is_identity_without_spending_proposal_or_bath_rng() -> Result<()> {
        let core = sphere(0.45);
        let ex = sphere(0.6);
        let wall = Container::new(50., &core)?;
        let e = engine(&core, &ex, &wall);
        let atlas = atlas();
        let proposal = FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(&atlas, 4., 0.5)?,
            FactorizedDimerCaps {
                root: 16,
                internal: 16,
                joint: 1,
            },
            FactorizedDimerOrder::RootFirst,
        );
        let guide = AuxiliaryOverlapThreshold::new(&ex, vec![[0.; 3]], 4)?;
        let mut s = state();
        s[1].position = [10., 0., 0.];
        let old = s.clone();
        let mut record = Value::Null;
        let mut p = StdRng::seed_from_u64(1);
        let mut t = StdRng::seed_from_u64(2);
        let mut b = StdRng::seed_from_u64(3);
        let mut a = StdRng::seed_from_u64(4);
        e.dimer(
            &mut s,
            &proposal,
            Some(&guide),
            &mut p,
            &mut t,
            &mut b,
            &mut a,
            &mut budget(),
            &mut record,
        )?;
        assert_eq!(s, old);
        assert_eq!(record["status"], "proposal_self_loop");
        for (i, r) in [&mut p, &mut t, &mut b, &mut a].into_iter().enumerate() {
            assert_eq!(
                r.random::<u64>(),
                StdRng::seed_from_u64(i as u64 + 1).random::<u64>()
            );
        }
        Ok(())
    }
    #[test]
    fn fatal_partial_local_cloud_does_not_mutate_the_configuration() -> Result<()> {
        let core = sphere(0.45);
        let ex = sphere(0.6);
        let wall = Container::new(50., &core)?;
        let mut e = engine(&core, &ex, &wall);
        e.lambda = 1000.;
        e.activity = 1.;
        let mut s = state();
        let old = s.clone();
        let mut record = Value::Null;
        let mut budget = budget();
        budget.limits.raw_per_leg = 0;
        let mut p = StdRng::seed_from_u64(31);
        let mut b = StdRng::seed_from_u64(32);
        let mut a = StdRng::seed_from_u64(33);
        assert!(
            e.local(
                &mut s,
                1,
                0.01,
                1.,
                &mut p,
                &mut b,
                &mut a,
                &mut budget,
                &mut record
            )
            .is_err()
        );
        assert_eq!(s, old);
        assert_eq!(record["accepted"], false);
        assert!(
            record["bath_failure"]["failed_progress"]["gate"]["raw_points"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert_eq!(budget.raw, 0);
        assert_eq!(a.random::<u64>(), StdRng::seed_from_u64(33).random::<u64>());
        Ok(())
    }
    #[test]
    fn split_trajectory_has_identical_poses_counts_and_complete_attempt_traces() -> Result<()> {
        let core = sphere(0.45);
        let ex = sphere(0.6);
        let wall = Container::new(50., &core)?;
        let e = engine(&core, &ex, &wall);
        let atlas = atlas();
        let proposal = FactorizedDimerProposal::new(
            DefensiveDimerProposal::new(&atlas, 4., 0.5)?,
            FactorizedDimerCaps {
                root: 16,
                internal: 16,
                joint: 1,
            },
            FactorizedDimerOrder::RootFirst,
        );
        let guide =
            AuxiliaryOverlapThreshold::new(&ex, vec![[0.; 3], [0.3, 0., 0.], [-0.3, 0., 0.]], 4)?;
        let evolve =
            |s: &mut Vec<Pose>, budget: &mut Budget, start: u64, end: u64| -> Result<Vec<Value>> {
                let mut rows = vec![];
                for block in start..=end {
                    for slot in 0..2 {
                        let base = block * 100 + slot as u64 * 10;
                        let mut row = Value::Null;
                        e.local(
                            s,
                            slot,
                            0.08,
                            2.,
                            &mut StdRng::seed_from_u64(base),
                            &mut StdRng::seed_from_u64(base + 1),
                            &mut StdRng::seed_from_u64(base + 2),
                            budget,
                            &mut row,
                        )?;
                        rows.push(row);
                    }
                    let base = block * 100 + 30;
                    let mut row = Value::Null;
                    e.dimer(
                        s,
                        &proposal,
                        Some(&guide),
                        &mut StdRng::seed_from_u64(base),
                        &mut StdRng::seed_from_u64(base + 1),
                        &mut StdRng::seed_from_u64(base + 2),
                        &mut StdRng::seed_from_u64(base + 3),
                        budget,
                        &mut row,
                    )?;
                    rows.push(row);
                }
                Ok(rows)
            };
        let mut full = state();
        let mut full_budget = budget();
        let full_rows = evolve(&mut full, &mut full_budget, 1, 12)?;
        let mut split = state();
        let mut split_budget = budget();
        let mut split_rows = evolve(&mut split, &mut split_budget, 1, 5)?;
        let saved = serde_json::to_vec(&split)?;
        split = serde_json::from_slice(&saved)?;
        let mut restored = budget();
        restored.raw = split_budget.raw;
        restored.retained = split_budget.retained;
        split_rows.extend(evolve(&mut split, &mut restored, 6, 12)?);
        assert_eq!(full, split);
        assert_eq!(full_rows, split_rows);
        assert_eq!(
            (full_budget.raw, full_budget.retained),
            (restored.raw, restored.retained)
        );
        Ok(())
    }
}
