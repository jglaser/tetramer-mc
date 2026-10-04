//! A fixed-length random-scan surrogate chain for two fixed, flexible labels.
//!
//! Each inner attempt chooses one member with state-independent probability 1/2
//! and applies the same symmetric translation/proper-rotation proposal. Both
//! coordinate MH kernels are reversible under the common hard × exp(S) target;
//! their fixed mixture and its fixed m-th power are therefore reversible. Hard
//! failures and MH rejections consume a step. The cloud, spectators, labels and
//! settings stay fixed throughout the call; neither internal contacts nor the
//! relative pose are constrained to remain as they were at the outer source.
//!
//! One fair-order two-singleton physical bath path receives S(old)-S(new), once.
//! Its internal exclusion-union change is included even without spectators.
//! Only physical endpoints are hard-checked: the copied bath intermediate may
//! violate cores or the protein wall and must never be independently filtered.
//!
//! This conditional building block does not select pairs or supply ergodicity.
//! Callers must journal a begun outer attempt, then durably drain `record` on
//! BOTH success and error. State changes only after the final acceptance;
//! resource/numerical errors are fatal, never rejection or permission to retry.
use crate::{
    bounded_singleton_path::{Budget, bounded_path},
    depletion::GateOptions,
    depletion_surrogate::DimerDepletionSurrogate,
    flexible_subset::FlexibleSubset,
    geometry::SphereTree,
    math::{Pose, Vec3},
    simulation::spherical_local_pose,
    singleton_path::SingletonPath,
    spherical::Container,
};
use anyhow::{Result, ensure};
use rand::{RngExt, distr::Open01, rngs::StdRng};
use serde_json::{Value, json};

/// The same validated horizon/scales/strength settings as the rigid chain.
/// This alias shares settings only; the flexible kernel has no rigid handle.
pub use crate::rigid_surrogate_chain::RigidSurrogateConfig as FlexibleSurrogateConfig;

pub struct FlexibleSurrogateKernel<'a> {
    pub core: &'a SphereTree,
    /// The same core atoms, each inflated exactly once by `rd`.
    pub exclusion: &'a SphereTree,
    pub wall: Option<&'a Container>,
    pub wall_center: Vec3,
    pub members: [usize; 2],
    pub rd: f64,
    pub activity: f64,
    pub lambda: f64,
    pub envelope: GateOptions,
    pub body_points: &'a [Vec3],
    /// Raw quadrature box volume divided by its original raw count.
    pub point_volume: f64,
    pub config: FlexibleSurrogateConfig,
}

impl FlexibleSurrogateKernel<'_> {
    fn validate(&self, state: &[Pose]) -> Result<()> {
        self.config.validate()?;
        self.envelope.validate()?;
        ensure!(
            self.members[0] != self.members[1] && self.members.iter().all(|&i| i < state.len()),
            "invalid fixed flexible labels"
        );
        ensure!(
            self.wall_center.iter().all(|x| x.is_finite()),
            "invalid wall center"
        );
        ensure!(
            self.rd.is_finite() && self.rd >= 0.,
            "invalid exclusion radius"
        );
        ensure!(
            self.activity.is_finite()
                && self.activity >= 0.
                && self.lambda.is_finite()
                && self.lambda > 0.
                && (self.lambda + self.activity).is_finite(),
            "invalid physical bath"
        );
        ensure!(
            self.core.shape.atoms.len() == self.exclusion.shape.atoms.len()
                && self
                    .core
                    .shape
                    .atoms
                    .iter()
                    .zip(&self.exclusion.shape.atoms)
                    .all(|(a, b)| a.center == b.center && a.radius + self.rd == b.radius),
            "surrogate exclusion differs from physical inflated shape"
        );
        for &p in state {
            p.validate()?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        state: &mut [Pose],
        proposal_rng: &mut StdRng,
        inner_accept_rng: &mut StdRng,
        bath_rng: &mut StdRng,
        outer_accept_rng: &mut StdRng,
        budget: &mut Budget,
        record: &mut Value,
    ) -> Result<()> {
        *record = json!({"kind":"flexible_surrogate_chain","status":"in_progress",
            "accepted":false,"members":self.members,"config":self.config,
            "selection_probabilities":[0.5,0.5],"steps":[],"physical_decisions":0,
            "inner_counts":{"attempted":0,"accepted":0,"hard_rejected":0,"mh_rejected":0},
            "budget_before":{"raw":budget.raw,"retained":budget.retained}});
        let result = self.step_inner(
            state,
            proposal_rng,
            inner_accept_rng,
            bath_rng,
            outer_accept_rng,
            budget,
            record,
        );
        record["budget_after"] = json!({"raw":budget.raw,"retained":budget.retained});
        if let Err(error) = &result {
            record["status"] = json!("fatal");
            record["error"] = json!(format!("{error:#}"));
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn step_inner(
        &self,
        state: &mut [Pose],
        proposal_rng: &mut StdRng,
        inner_accept_rng: &mut StdRng,
        bath_rng: &mut StdRng,
        outer_accept_rng: &mut StdRng,
        budget: &mut Budget,
        record: &mut Value,
    ) -> Result<()> {
        self.validate(state)?;
        budget.limits.validate()?;
        ensure!(
            budget.raw <= budget.limits.raw_campaign
                && budget.retained <= budget.limits.retained_campaign,
            "invalid completed campaign counters"
        );
        budget.check_cpu()?;
        let old = self.members.map(|i| state[i]);
        record["old"] = json!(old);
        let identity = FlexibleSubset::new(self.core, state, &self.members, &old, self.rd)?;
        ensure!(
            identity.hard_valid(self.wall, self.wall_center),
            "hard-invalid surrogate source"
        );
        let spectators: Vec<_> = state
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| (!self.members.contains(&i)).then_some(p))
            .collect();
        let surrogate = DimerDepletionSurrogate::new(
            self.exclusion,
            self.body_points.to_vec(),
            &spectators,
            self.point_volume,
            self.activity * self.config.guidance_strength,
        )?;
        let old_score = surrogate.score(old)?;
        let mut score = old_score.clone();
        let mut current = old;
        let mut accepted_count = 0;
        let mut hard_rejected = 0;
        let mut mh_rejected = 0;
        record["old_score"] = json!(old_score);
        for index in 0..self.config.inner_steps {
            budget.check_cpu()?;
            record["steps"].as_array_mut().unwrap().push(json!({
                "index":index,"old":current,"old_score":score.log_surrogate,
                "accepted":false,"status":"begun"}));
            record["inner_counts"]["attempted"] = json!(index + 1);
            let trace = &mut record["steps"][index];
            // Fresh fixed fair scan at EVERY attempt, including hard failures.
            // Deterministic alternating sweeps need not be reversible.
            let slot = usize::from(proposal_rng.random::<bool>());
            trace["selected_slot"] = json!(slot);
            trace["selected_label"] = json!(self.members[slot]);
            let mut target = current;
            target[slot] = spherical_local_pose(
                proposal_rng,
                current[slot],
                self.config.translation_std,
                self.config.rotation_std_degrees.to_radians() / 2.,
            );
            trace["proposed"] = json!(target);
            let endpoint = FlexibleSubset::new(self.core, state, &self.members, &target, self.rd)?;
            if !endpoint.hard_valid(self.wall, self.wall_center) {
                trace["status"] = json!("hard_rejected");
                trace["retained"] = json!(current);
                trace["retained_score"] = json!(score.log_surrogate);
                hard_rejected += 1;
                record["inner_counts"]["hard_rejected"] = json!(hard_rejected);
                continue;
            }
            budget.check_cpu()?;
            let candidate_score = surrogate.score(target)?;
            let delta = candidate_score.log_surrogate - score.log_surrogate;
            ensure!(delta.is_finite(), "nonfinite inner score difference");
            let log_u = inner_accept_rng.sample::<f64, _>(Open01).ln();
            let accepted = log_u < delta.min(0.);
            trace["proposed_score"] = json!(candidate_score);
            trace["log_acceptance_ratio"] = json!(delta);
            trace["log_u"] = json!(log_u);
            trace["accepted"] = json!(accepted);
            trace["status"] = json!("completed");
            if accepted {
                current = target;
                score = candidate_score;
                accepted_count += 1;
            } else {
                mh_rejected += 1;
            }
            trace["retained"] = json!(current);
            trace["retained_score"] = json!(score.log_surrogate);
            record["inner_counts"]["accepted"] = json!(accepted_count);
            record["inner_counts"]["mh_rejected"] = json!(mh_rejected);
        }
        record["proposed"] = json!(current);
        record["proposed_score"] = json!(score);
        let correction = old_score.log_surrogate - score.log_surrogate;
        ensure!(correction.is_finite(), "nonfinite outer score difference");
        record["complete_log_correction"] = json!(correction);
        if current == old {
            record["status"] = json!("identity_self_loop");
            return Ok(());
        }
        budget.check_cpu()?;
        let path = SingletonPath::new(self.core, state, &self.members, &current, self.rd)?;
        ensure!(
            path.hard_valid(self.wall, self.wall_center),
            "final surrogate/physical endpoint disagreement"
        );
        // The fair path-order coin and both clouds belong to one physical gate.
        // Do not apply hard support to the copied auxiliary intermediate.
        let bath = match bounded_path(
            self.core,
            state,
            self.members,
            current,
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
                anyhow::bail!("bounded flexible surrogate bath failure");
            }
        };
        record["bath"] = json!(bath);
        let log_ratio = bath.aggregate.log_weight + correction;
        ensure!(log_ratio.is_finite(), "nonfinite final acceptance ratio");
        let log_u = outer_accept_rng.sample::<f64, _>(Open01).ln();
        let accepted = log_u < log_ratio.min(0.);
        record["log_acceptance_ratio"] = json!(log_ratio);
        record["log_u"] = json!(log_u);
        record["physical_decisions"] = json!(1);
        record["accepted"] = json!(accepted);
        record["status"] = json!("completed");
        if accepted {
            for (i, &label) in self.members.iter().enumerate() {
                state[label] = current[i];
            }
        }
        Ok(())
    }
}
