//! A fixed-length reversible surrogate chain on a fixed rigid dimer fiber.
//!
//! The identical symmetric handle proposal is used at every inner step. Hard
//! failures and MH rejections consume a step. The invariant surrogate density
//! is 1_hard exp(S); its m-step kernel is reversible, without equilibrating it.
//! One outer exact physical bath gate receives S(old)-S(new), once. The cloud,
//! spectators, labels, handle and step settings stay fixed throughout this call.
//!
//! This conditional building block does not select clusters, supply ergodicity,
//! update an atlas, or replace the physical local moves. Its caller must journal
//! a begun outer attempt, then durably drain `record` on BOTH success and error.
//! The physical state is unchanged until the final acceptance. A resource or
//! numerical error is fatal, never an ordinary rejection or a retried draw.
use crate::{
    bounded_singleton_path::{Budget, bounded_singleton},
    depletion::GateOptions,
    depletion_surrogate::DimerDepletionSurrogate,
    geometry::SphereTree,
    math::{Pose, Vec3},
    rigid_subset::RigidSubset,
    simulation::spherical_local_pose,
    spherical::Container,
};
use anyhow::{Result, ensure};
use rand::{RngExt, distr::Open01, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RigidSurrogateConfig {
    pub inner_steps: usize,
    pub translation_std: f64,
    pub rotation_std_degrees: f64,
    pub guidance_strength: f64,
}

impl RigidSurrogateConfig {
    pub fn validate(self) -> Result<()> {
        ensure!(self.inner_steps <= 1024, "surrogate horizon exceeds 1024");
        for value in [
            self.translation_std,
            self.rotation_std_degrees,
            self.guidance_strength,
        ] {
            ensure!(
                value.is_finite() && value >= 0.,
                "invalid surrogate setting"
            );
        }
        ensure!(
            (self.rotation_std_degrees.to_radians() / 2.).is_finite(),
            "invalid rotation scale"
        );
        Ok(())
    }
}

pub struct RigidSurrogateKernel<'a> {
    pub core: &'a SphereTree,
    /// The same core atoms, each inflated exactly once by `rd`.
    pub exclusion: &'a SphereTree,
    pub wall: Option<&'a Container>,
    pub wall_center: Vec3,
    pub members: [usize; 2],
    pub handle: usize,
    pub rd: f64,
    pub activity: f64,
    pub lambda: f64,
    pub envelope: GateOptions,
    pub body_points: &'a [Vec3],
    /// Raw quadrature box volume divided by its original raw count.
    pub point_volume: f64,
    pub config: RigidSurrogateConfig,
}

impl RigidSurrogateKernel<'_> {
    fn validate(&self, state: &[Pose]) -> Result<()> {
        self.config.validate()?;
        self.envelope.validate()?;
        ensure!(
            self.members[0] != self.members[1]
                && self.members.iter().all(|&i| i < state.len())
                && self.members.contains(&self.handle),
            "invalid fixed dimer labels"
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
        *record = json!({"kind":"rigid_surrogate_chain","status":"in_progress",
            "accepted":false,"members":self.members,"handle":self.handle,"config":self.config,
            "steps":[],"physical_decisions":0});
        let result = self.step_inner(
            state,
            proposal_rng,
            inner_accept_rng,
            bath_rng,
            outer_accept_rng,
            budget,
            record,
        );
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
        budget.check_cpu()?;
        let old = self.members.map(|i| state[i]);
        record["old"] = json!(old);
        let identity = RigidSubset::new(
            self.core,
            state,
            &self.members,
            self.handle,
            state[self.handle],
            self.rd,
        )?;
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
        let mut handle_pose = state[self.handle];
        record["old_score"] = json!(old_score);
        for index in 0..self.config.inner_steps {
            budget.check_cpu()?;
            record["steps"]
                .as_array_mut()
                .unwrap()
                .push(json!({"index":index,
                "old_handle":handle_pose,"old_score":score.log_surrogate,
                "accepted":false,"status":"begun"}));
            let trace = &mut record["steps"][index];
            let proposed = spherical_local_pose(
                proposal_rng,
                handle_pose,
                self.config.translation_std,
                self.config.rotation_std_degrees.to_radians() / 2.,
            );
            trace["proposed_handle"] = json!(proposed);
            // Always reconstruct from the OUTER source. Repeated inner rigid
            // updates must not successively distort the reference geometry.
            let gate = RigidSubset::new(
                self.core,
                state,
                &self.members,
                self.handle,
                proposed,
                self.rd,
            )?;
            let target: [Pose; 2] = gate.proposed_poses().try_into().unwrap();
            trace["proposed"] = json!(target);
            if !gate.hard_valid(self.wall, self.wall_center) {
                trace["status"] = json!("hard_rejected");
                trace["retained"] = json!(current);
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
                handle_pose = proposed;
            }
            trace["retained"] = json!(current);
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
        let gate = RigidSubset::new(
            self.core,
            state,
            &self.members,
            self.handle,
            handle_pose,
            self.rd,
        )?;
        ensure!(
            gate.proposed_poses() == current && gate.hard_valid(self.wall, self.wall_center),
            "final surrogate/physical endpoint disagreement"
        );
        let bath = match bounded_singleton(
            &gate,
            bath_rng,
            self.lambda,
            self.activity,
            self.envelope,
            budget,
        ) {
            Ok(value) => value,
            Err(error) => {
                record["bath_failure"] = json!(error);
                anyhow::bail!("bounded surrogate bath failure");
            }
        };
        record["bath"] = json!(bath);
        let log_ratio = bath.log_weight + correction;
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
