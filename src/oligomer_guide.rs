//! Fixed-length reversible search of a joint oligomer guide before depletion.
//! The guide depends on invariant internal poses and an external-only anchor
//! pool. Its unknown normalizer cancels in the endpoint physical correction.
use crate::{
    docking::DockingProposal,
    geometry::{Placed, SphereTree},
    math::{Pose, dot, sub},
    rigid_subset::RigidSubset,
    simulation::cpu_seconds,
    spherical::Container,
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, rngs::StdRng};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct OligomerGuideConfig {
    /// Fixed number of inner attempts, including rejections/nulls. Zero disables.
    pub steps: usize,
    /// Primary anchor plus its nearest external neighbors; fixed during a move.
    pub anchor_count: usize,
    /// Temper the complete joint log guide; zero is the uniform-guide control.
    pub score_power: f64,
}
impl Default for OligomerGuideConfig {
    fn default() -> Self {
        Self {
            steps: 4,
            anchor_count: 4,
            score_power: 1.,
        }
    }
}
impl OligomerGuideConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.anchor_count > 0,
            "oligomer guide needs a positive anchor count"
        );
        ensure!(
            self.score_power.is_finite() && self.score_power >= 0.,
            "invalid oligomer guide score power"
        );
        Ok(())
    }
    pub fn enabled(&self) -> bool {
        self.steps > 0
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GuideCounts {
    pub inner_attempts: u64,
    pub inner_hard_valid: u64,
    pub inner_hard_rejected: u64,
    pub inner_internal_nulls: u64,
    pub inner_proposal_nulls: u64,
    pub inner_accepted: u64,
    pub inner_pose_changes: u64,
    pub density_evaluations: u64,
    pub endpoint_changed: bool,
}

/// Selection depends only on spectator identities/positions. The same primary
/// label produces exactly the same pool at either endpoint of a rigid-S move.
pub fn anchor_pool(
    state: &[Pose],
    members: &[usize],
    primary: usize,
    count: usize,
) -> Result<Vec<usize>> {
    ensure!(
        count > 0 && primary < state.len() && !members.contains(&primary),
        "invalid guide primary anchor"
    );
    let mut labels: Vec<_> = (0..state.len()).filter(|i| !members.contains(i)).collect();
    labels.sort_by(|&a, &b| {
        let da = sub(state[a].position, state[primary].position);
        let db = sub(state[b].position, state[primary].position);
        dot(da, da).total_cmp(&dot(db, db)).then(a.cmp(&b))
    });
    labels.truncate(count.min(labels.len()));
    Ok(labels)
}

fn score(
    proposal: &DockingProposal,
    member_poses: &[Pose],
    anchors: &[Pose],
    power: f64,
    counts: &mut GuideCounts,
) -> Result<f64> {
    if power == 0. {
        return Ok(0.);
    }
    let mut sum = 0.;
    for &pose in member_poses {
        sum += proposal.guide_log_density(pose, anchors)?;
        counts.density_evaluations += anchors.len() as u64;
    }
    let result = power * sum;
    ensure!(result.is_finite(), "nonfinite joint oligomer guide density");
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub fn propose(
    tree: &SphereTree,
    exclusion: &SphereTree,
    wall: &Container,
    rd: f64,
    state: &[Pose],
    members: &[usize],
    handle: usize,
    proposal: &DockingProposal,
    config: &OligomerGuideConfig,
    rng: &mut StdRng,
    record: bool,
) -> Result<(Option<Pose>, Value, GuideCounts)> {
    config.validate()?;
    ensure!(config.enabled(), "disabled guide invoked");
    ensure!(members.contains(&handle), "guide handle outside subset");
    ensure!(
        proposal.uniform_weight() > 0.,
        "joint guide requires a positive defensive uniform component"
    );
    let started = cpu_seconds();
    let external: Vec<_> = (0..state.len()).filter(|i| !members.contains(i)).collect();
    if external.is_empty() {
        return Ok((
            None,
            json!({"branch":"oligomer_guide","null_reason":"no_external_anchor"}),
            GuideCounts::default(),
        ));
    }
    let primary = external[rng.random_range(0..external.len())];
    let pool = anchor_pool(state, members, primary, config.anchor_count)?;
    let anchors: Vec<_> = pool.iter().map(|&i| state[i]).collect();
    let internal: Vec<_> = members
        .iter()
        .enumerate()
        .flat_map(|(k, &i)| {
            members[k + 1..].iter().map(move |&j| {
                (
                    i,
                    j,
                    exclusion.overlaps(&Placed::new(state[i]), &Placed::new(state[j])),
                )
            })
        })
        .collect();
    let mut counts = GuideCounts::default();
    let mut current = state[handle];
    let original: Vec<_> = members.iter().map(|&i| state[i]).collect();
    let initial_density_start = cpu_seconds();
    let initial_score = score(
        proposal,
        &original,
        &anchors,
        config.score_power,
        &mut counts,
    )?;
    let mut current_score = initial_score;
    let mut traces = Vec::new();
    let mut geometry_cpu = 0.;
    let mut density_cpu = cpu_seconds() - initial_density_start;
    for index in 0..config.steps {
        counts.inner_attempts += 1;
        let old = current;
        let old_score = current_score;
        let (candidate, mut base) = proposal.propose(rng, current, &anchors)?;
        if let Some(j) = base["anchor_index"].as_u64() {
            base["anchor_index"] = json!(pool[j as usize]);
        }
        let mut hard_valid = false;
        let mut internal_equal = true;
        let mut proposed_score = None;
        let mut log_acceptance = None;
        let mut accepted = false;
        if let Some(next) = candidate {
            let geometry_start = cpu_seconds();
            // Always reconstruct from the immutable original internal geometry,
            // rather than accumulating roundoff by carrying the previous subset.
            let trial = RigidSubset::new(tree, state, members, handle, next, rd)?;
            hard_valid = trial.hard_valid(Some(wall), [0.; 3]);
            if hard_valid {
                counts.inner_hard_valid += 1;
                internal_equal = internal.iter().all(|&(i, j, v)| {
                    let ai = members.iter().position(|&k| k == i).unwrap();
                    let aj = members.iter().position(|&k| k == j).unwrap();
                    exclusion.overlaps(
                        &Placed::new(trial.proposed_poses()[ai]),
                        &Placed::new(trial.proposed_poses()[aj]),
                    ) == v
                });
            } else {
                counts.inner_hard_rejected += 1;
            }
            geometry_cpu += cpu_seconds() - geometry_start;
            if hard_valid && internal_equal {
                let density_start = cpu_seconds();
                let next_score = score(
                    proposal,
                    trial.proposed_poses(),
                    &anchors,
                    config.score_power,
                    &mut counts,
                )?;
                density_cpu += cpu_seconds() - density_start;
                let correction = base["log_reverse_forward"]
                    .as_f64()
                    .context("missing inner guide map correction")?;
                let a = (next_score - current_score + correction).min(0.);
                ensure!(a.is_finite(), "nonfinite inner guide acceptance");
                accepted = rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < a;
                proposed_score = Some(next_score);
                log_acceptance = Some(a);
                if accepted {
                    counts.inner_accepted += 1;
                    counts.inner_pose_changes += u64::from(next != current);
                    current = next;
                    current_score = next_score;
                }
            } else if hard_valid {
                counts.inner_internal_nulls += 1;
            }
        } else {
            counts.inner_proposal_nulls += 1;
        }
        if record {
            traces.push(json!({"step":index,"old_handle":old,"proposed_handle":candidate,"retained_handle":current,
                "proposal":base,"hard_valid":hard_valid,"internal_contact_graph_preserved":internal_equal,
                "old_log_guide":old_score,"proposed_log_guide":proposed_score,
                "log_acceptance":log_acceptance,"accepted":accepted}));
        }
    }
    counts.endpoint_changed = current != state[handle];
    let correction = initial_score - current_score;
    let info = json!({"branch":"oligomer_guide","anchor_index":primary,"guide_anchor_indices":pool,
        "source_law":"fixed_length_reversible_joint_guide","guide_config":config,"guide_counts":counts,
        "old_log_guide":initial_score,"new_log_guide":current_score,"log_reverse_forward":correction,
        "guide_geometry_cpu_seconds":geometry_cpu,"guide_density_cpu_seconds":density_cpu,
        "guide_cpu_seconds":cpu_seconds()-started,"inner_steps":traces,
        "endpoint_changed":counts.endpoint_changed,
        "null_reason":if counts.endpoint_changed {None} else {Some("unchanged_guide_endpoint")}});
    Ok((counts.endpoint_changed.then_some(current), info, counts))
}
