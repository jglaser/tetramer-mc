//! Fixed-duration continuous-time mixture of reversible rigid-subset kernels.
//! Eligibility depends only on internal contacts. State-dependent total rate is
//! handled by exponential holding times, never by an extra acceptance factor.
use crate::{
    assembly_bias::{self, AssemblyBias, AssemblyBiasState},
    depletion::GateOptions,
    docking::DockingProposal,
    geometry::{Placed, SphereTree},
    math::*,
    oligomer_guide::{self, OligomerGuideConfig},
    rigid_subset::RigidSubset,
    spherical::Container,
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ClusterPhaseConfig {
    pub duration: f64,
    pub dimer_rate: f64,
    pub trimer_rate: f64,
    pub transport_probability: f64,
    pub correlation: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guide: Option<OligomerGuideConfig>,
    #[serde(rename = "local_translation_std_A")]
    pub local_translation_std_a: f64,
    pub local_small_angle_std_degrees: f64,
}
impl Default for ClusterPhaseConfig {
    fn default() -> Self {
        Self {
            duration: 0.01,
            dimer_rate: 1.,
            trimer_rate: 0.25,
            transport_probability: 0.5,
            correlation: 0.9,
            guide: None,
            local_translation_std_a: 0.2,
            local_small_angle_std_degrees: 1.,
        }
    }
}
impl ClusterPhaseConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(guide) = &self.guide {
            guide.validate()?;
        }
        for v in [
            self.duration,
            self.dimer_rate,
            self.trimer_rate,
            self.local_translation_std_a,
            self.local_small_angle_std_degrees,
        ] {
            ensure!(
                v.is_finite() && v >= 0.,
                "invalid cluster phase rate, duration or local scale"
            );
        }
        ensure!(
            self.transport_probability.is_finite()
                && (0. ..=1.).contains(&self.transport_probability),
            "invalid cluster transport probability"
        );
        ensure!(
            self.correlation.is_finite() && (-1. ..=1.).contains(&self.correlation),
            "invalid cluster map correlation"
        );
        Ok(())
    }
    pub fn enabled(&self) -> bool {
        self.duration > 0. && (self.dimer_rate > 0. || self.trimer_rate > 0.)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ClusterPhaseCounts {
    pub phases: u64,
    pub events: u64,
    pub dimer_events: u64,
    pub trimer_events: u64,
    pub local_events: u64,
    pub transport_events: u64,
    pub proposal_nulls: u64,
    pub hard_valid: u64,
    pub hard_rejected: u64,
    pub internal_geometry_nulls: u64,
    pub physical_accepted: u64,
    pub bias_rejected: u64,
    pub accepted: u64,
    pub transformed_bodies: u64,
    pub accepted_attachments: u64,
    pub accepted_detachments: u64,
    pub completed_exchanges: u64,
    pub gained_contacts: u64,
    pub lost_contacts: u64,
    pub gate_raw_points: u64,
    pub zero_rate_stops: u64,
    pub horizon_stops: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub guide_events: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub guide_inner_attempts: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub guide_inner_hard_valid: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub guide_inner_accepted: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub guide_endpoint_nulls: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub guide_accepted: u64,
}
fn is_zero(value: &u64) -> bool {
    *value == 0
}
macro_rules! fields {
    ($m:ident, $a:ident, $b:ident) => {
        $m!(
            $a,
            $b,
            phases,
            events,
            dimer_events,
            trimer_events,
            local_events,
            transport_events,
            proposal_nulls,
            hard_valid,
            hard_rejected,
            internal_geometry_nulls,
            physical_accepted,
            bias_rejected,
            accepted,
            transformed_bodies,
            accepted_attachments,
            accepted_detachments,
            completed_exchanges,
            gained_contacts,
            lost_contacts,
            gate_raw_points,
            zero_rate_stops,
            horizon_stops,
            guide_events,
            guide_inner_attempts,
            guide_inner_hard_valid,
            guide_inner_accepted,
            guide_endpoint_nulls,
            guide_accepted
        )
    };
}
impl ClusterPhaseCounts {
    pub fn add(&mut self, b: &Self) {
        macro_rules! add { ($a:ident,$b:ident,$($f:ident),*) => { $( $a.$f += $b.$f; )* }; }
        fields!(add, self, b);
    }
    pub fn since(&self, b: &Self) -> Self {
        macro_rules! sub { ($a:ident,$b:ident,$($f:ident),*) => { Self { $($f:$a.$f-$b.$f,)* } }; }
        fields!(sub, self, b)
    }
}

/// Contact graph has no native labels and is independent of the proposal atlas.
#[derive(Clone, Debug)]
pub struct ContactGraph {
    pub adjacency: Vec<Vec<bool>>,
}
/// Passive pre-move context. These values never enter selection or acceptance.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SubsetContext {
    pub subset_size: usize,
    pub parent_component_members: Vec<usize>,
    pub whole_component: bool,
    pub internal_contacts: usize,
    pub external_contacts: usize,
    pub external_neighbors: usize,
}
impl ContactGraph {
    /// Classify a connected selected subset against the full exclusion graph.
    /// Connectivity here is purely geometric; no native labels are consulted.
    pub fn subset_context(&self, members: &[usize]) -> Result<SubsetContext> {
        let n = self.adjacency.len();
        ensure!(!members.is_empty(), "empty diagnostic subset");
        let mut selected = vec![false; n];
        for &i in members {
            ensure!(i < n && !selected[i], "invalid diagnostic subset index");
            selected[i] = true;
        }
        let component = |restricted: bool| {
            let mut reached = vec![false; n];
            let mut stack = vec![members[0]];
            reached[members[0]] = true;
            while let Some(i) = stack.pop() {
                for j in 0..n {
                    if self.adjacency[i][j] && !reached[j] && (!restricted || selected[j]) {
                        reached[j] = true;
                        stack.push(j);
                    }
                }
            }
            reached
        };
        let internal = component(true);
        ensure!(
            members.iter().all(|&i| internal[i]),
            "disconnected diagnostic subset"
        );
        let parent_component_members: Vec<_> = component(false)
            .iter()
            .enumerate()
            .filter_map(|(i, &reached)| reached.then_some(i))
            .collect();
        let mut internal_contacts = 0;
        let mut external_contacts = 0;
        let mut neighbors = vec![false; n];
        for &i in members {
            for j in 0..n {
                if self.adjacency[i][j] {
                    if selected[j] {
                        internal_contacts += usize::from(i < j);
                    } else {
                        external_contacts += 1;
                        neighbors[j] = true;
                    }
                }
            }
        }
        Ok(SubsetContext {
            subset_size: members.len(),
            whole_component: parent_component_members.len() == members.len(),
            parent_component_members,
            internal_contacts,
            external_contacts,
            external_neighbors: neighbors.iter().filter(|&&v| v).count(),
        })
    }
    pub fn build(exclusion: &SphereTree, poses: &[Pose]) -> Self {
        let mut g = Self {
            adjacency: vec![vec![false; poses.len()]; poses.len()],
        };
        for i in 0..poses.len() {
            for j in i + 1..poses.len() {
                let v = exclusion.overlaps(&Placed::new(poses[i]), &Placed::new(poses[j]));
                g.adjacency[i][j] = v;
                g.adjacency[j][i] = v;
            }
        }
        g
    }
    pub fn updated(&self, exclusion: &SphereTree, poses: &[Pose], members: &[usize]) -> Self {
        let mut g = self.clone();
        for &i in members {
            for j in 0..poses.len() {
                if i == j || (j < i && members.contains(&j)) {
                    continue;
                }
                let v = exclusion.overlaps(&Placed::new(poses[i]), &Placed::new(poses[j]));
                g.adjacency[i][j] = v;
                g.adjacency[j][i] = v;
            }
        }
        g
    }
    /// Unique labeled connected triples; triangles are counted once.
    pub fn channels(&self, cfg: &ClusterPhaseConfig) -> Vec<Channel> {
        let mut out = Vec::new();
        let n = self.adjacency.len();
        if cfg.dimer_rate > 0. {
            for i in 0..n {
                for j in i + 1..n {
                    if self.adjacency[i][j] {
                        out.push(Channel {
                            members: vec![i, j],
                            rate: cfg.dimer_rate,
                        });
                    }
                }
            }
        }
        if cfg.trimer_rate > 0. {
            let mut triples = BTreeSet::new();
            for i in 0..n {
                let neighbors: Vec<_> = (0..n).filter(|&j| self.adjacency[i][j]).collect();
                for a in 0..neighbors.len() {
                    for b in a + 1..neighbors.len() {
                        let mut t = [i, neighbors[a], neighbors[b]];
                        t.sort_unstable();
                        triples.insert(t);
                    }
                }
            }
            out.extend(triples.into_iter().map(|t| Channel {
                members: t.to_vec(),
                rate: cfg.trimer_rate,
            }));
        }
        out
    }
    pub fn boundary_edges(&self, members: &[usize]) -> BTreeSet<[usize; 2]> {
        let mut out = BTreeSet::new();
        for &i in members {
            for j in 0..self.adjacency.len() {
                if !members.contains(&j) && self.adjacency[i][j] {
                    out.insert([i.min(j), i.max(j)]);
                }
            }
        }
        out
    }
    pub fn internal_equal(&self, other: &Self, members: &[usize]) -> bool {
        members.iter().all(|&i| {
            members
                .iter()
                .all(|&j| self.adjacency[i][j] == other.adjacency[i][j])
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Channel {
    pub members: Vec<usize>,
    pub rate: f64,
}

/// None means no event inside the remaining fixed horizon. Rejecting events
/// still consume this waiting time. Zero rate consumes no RNG.
pub fn next_wait(rng: &mut StdRng, rate: f64, remaining: f64) -> Result<Option<f64>> {
    ensure!(
        rate.is_finite() && rate >= 0. && remaining.is_finite() && remaining >= 0.,
        "invalid event clock"
    );
    if rate == 0. || remaining == 0. {
        return Ok(None);
    }
    let u: f64 = rng.sample(Open01);
    let dt = -u.ln() / rate;
    ensure!(dt > 0., "unrepresentable cluster waiting time");
    Ok((dt < remaining).then_some(dt))
}
fn choose_channel<'a>(rng: &mut StdRng, channels: &'a [Channel], rate: f64) -> &'a Channel {
    let mut r = rng.random::<f64>() * rate;
    for c in channels {
        if r < c.rate {
            return c;
        }
        r -= c.rate;
    }
    channels.last().expect("positive rate without channels")
}

pub struct ClusterPhase<'a> {
    tree: &'a SphereTree,
    wall: &'a Container,
    exclusion: SphereTree,
    rd: f64,
    z: f64,
    lambda: f64,
    gate: GateOptions,
    pub config: ClusterPhaseConfig,
    proposal: Option<DockingProposal>,
}
impl<'a> ClusterPhase<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tree: &'a SphereTree,
        wall: &'a Container,
        rd: f64,
        z: f64,
        lambda: f64,
        gate: GateOptions,
        config: ClusterPhaseConfig,
        proposal: Option<DockingProposal>,
    ) -> Result<Self> {
        config.validate()?;
        gate.validate()?;
        ensure!(
            (lambda + z).is_finite(),
            "invalid total cluster bath intensity"
        );
        ensure!(
            rd.is_finite()
                && rd >= 0.
                && z.is_finite()
                && z >= 0.
                && lambda.is_finite()
                && lambda > 0.,
            "invalid cluster bath"
        );
        ensure!(
            !config.enabled() || config.transport_probability == 0. || proposal.is_some(),
            "cluster transport requires frozen learned proposal"
        );
        if config.guide.as_ref().is_some_and(|g| g.enabled()) && config.transport_probability > 0. {
            ensure!(
                proposal.as_ref().is_some_and(|p| p.uniform_weight() > 0.),
                "oligomer guide requires a frozen proposal with positive uniform support"
            );
        }
        let mut shape = tree.shape.clone();
        for atom in &mut shape.atoms {
            atom.radius += rd;
        }
        Ok(Self {
            tree,
            wall,
            exclusion: SphereTree::new(shape)?,
            rd,
            z,
            lambda,
            gate,
            config,
            proposal,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        poses: &mut Vec<Pose>,
        clock: &mut StdRng,
        proposal_rng: &mut StdRng,
        gate_rng: &mut StdRng,
        accept_rng: &mut StdRng,
        bias_rng: &mut StdRng,
        bias: Option<&AssemblyBias>,
        bias_state: &mut Option<AssemblyBiasState>,
        record: bool,
    ) -> Result<(ClusterPhaseCounts, Vec<Value>)> {
        let mut counts = ClusterPhaseCounts::default();
        let mut rows = Vec::new();
        if !self.config.enabled() {
            return Ok((counts, rows));
        }
        counts.phases = 1;
        let mut graph = ContactGraph::build(&self.exclusion, poses);
        let mut channels = graph.channels(&self.config);
        let mut t = 0.;
        loop {
            let rate: f64 = channels.iter().map(|c| c.rate).sum();
            ensure!(rate.is_finite(), "unrepresentable total cluster rate");
            let wait = next_wait(clock, rate, self.config.duration - t)?;
            let Some(dt) = wait else {
                if rate == 0. {
                    counts.zero_rate_stops += 1;
                } else {
                    counts.horizon_stops += 1;
                }
                if record {
                    rows.push(json!({"kind":"cluster_phase_end","event_time":t,"duration":self.config.duration,
                    "remaining_time":self.config.duration-t,"total_rate":rate,"eligible_channels":channels.len(),
                    "reason":if rate==0.{"zero_rate"}else{"fixed_horizon"}}));
                }
                break;
            };
            ensure!(t + dt > t, "cluster clock failed to advance");
            t += dt;
            let selected = choose_channel(clock, &channels, rate).clone();
            let members = &selected.members;
            // Logging-only graph traversal: no random draws, geometry queries,
            // or changes to the phase clock, selection, or acceptance rule.
            let subset_context = if record {
                Some(graph.subset_context(members)?)
            } else {
                None
            };
            let handle = members[proposal_rng.random_range(0..members.len())];
            let old_member_poses: Vec<_> = members.iter().map(|&i| poses[i]).collect();
            let transport = proposal_rng.random::<f64>() < self.config.transport_probability;
            counts.events += 1;
            if members.len() == 2 {
                counts.dimer_events += 1;
            } else {
                counts.trimer_events += 1;
            }
            if transport {
                counts.transport_events += 1;
            } else {
                counts.local_events += 1;
            }
            let mut info = json!({"branch":"local_rigid","log_reverse_forward":0.});
            let guided = transport && self.config.guide.as_ref().is_some_and(|g| g.enabled());
            let candidate = if guided {
                let (candidate, trace, c) = oligomer_guide::propose(
                    self.tree,
                    &self.exclusion,
                    self.wall,
                    self.rd,
                    poses,
                    members,
                    handle,
                    self.proposal.as_ref().unwrap(),
                    self.config.guide.as_ref().unwrap(),
                    proposal_rng,
                    record,
                )?;
                info = trace;
                counts.guide_events += 1;
                counts.guide_inner_attempts += c.inner_attempts;
                counts.guide_inner_hard_valid += c.inner_hard_valid;
                counts.guide_inner_accepted += c.inner_accepted;
                counts.guide_endpoint_nulls += u64::from(!c.endpoint_changed);
                candidate
            } else if transport {
                let spectator_indices: Vec<_> =
                    (0..poses.len()).filter(|i| !members.contains(i)).collect();
                if spectator_indices.is_empty() {
                    info = json!({"branch":"transport","null_reason":"no_external_anchor"});
                    None
                } else {
                    let spectators: Vec<_> = spectator_indices.iter().map(|&i| poses[i]).collect();
                    let (p, mut trace) = self.proposal.as_ref().unwrap().propose(
                        proposal_rng,
                        poses[handle],
                        &spectators,
                    )?;
                    if let Some(i) = trace["anchor_index"].as_u64() {
                        trace["anchor_index"] = json!(spectator_indices[i as usize]);
                    }
                    info = trace;
                    p
                }
            } else {
                let displacement = std::array::from_fn(|_| {
                    let v: f64 = StandardNormal.sample(proposal_rng);
                    v * self.config.local_translation_std_a
                });
                let c = std::array::from_fn(|_| {
                    let v: f64 = StandardNormal.sample(proposal_rng);
                    v * self.config.local_small_angle_std_degrees.to_radians() / 2.
                });
                Some(Pose {
                    position: add(poses[handle].position, displacement),
                    orientation: quaternion(matmul(cayley(c), rotation(poses[handle].orientation))),
                })
            };
            let mut hard_valid = false;
            let mut accepted = false;
            let mut physical_accepted = false;
            let mut sampled = None;
            let mut alpha = None;
            let mut bias_decision = None;
            let mut proposed_member_poses = None;
            let mut gained = Vec::new();
            let mut lost = Vec::new();
            let mut internal_equal = true;
            if let Some(new_handle) = candidate {
                let correction = info["log_reverse_forward"]
                    .as_f64()
                    .context("cluster proposal correction missing")?;
                ensure!(correction.is_finite(), "nonfinite cluster map correction");
                let trial =
                    RigidSubset::new(self.tree, poses, members, handle, new_handle, self.rd)?;
                proposed_member_poses = Some(trial.proposed_poses().to_vec());
                hard_valid = trial.hard_valid(Some(self.wall), [0.; 3]);
                if hard_valid {
                    counts.hard_valid += 1;
                    let mut next = poses.clone();
                    for (&i, &p) in members.iter().zip(trial.proposed_poses()) {
                        next[i] = p;
                    }
                    let next_graph = graph.updated(&self.exclusion, &next, members);
                    internal_equal = graph.internal_equal(&next_graph, members);
                    if internal_equal {
                        let old_edges = graph.boundary_edges(members);
                        let new_edges = next_graph.boundary_edges(members);
                        gained = new_edges.difference(&old_edges).copied().collect();
                        lost = old_edges.difference(&new_edges).copied().collect();
                        let gate = trial.sample(gate_rng, self.lambda, self.z, self.gate)?;
                        counts.gate_raw_points += gate.raw_points;
                        let a = (gate.log_weight + correction).min(0.);
                        ensure!(!a.is_nan(), "NaN cluster acceptance");
                        physical_accepted =
                            accept_rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < a;
                        accepted = physical_accepted;
                        alpha = Some(a);
                        sampled = Some(gate);
                        if accepted {
                            counts.physical_accepted += 1;
                            if let Some(engine) = bias {
                                let decision = assembly_bias::decide(
                                    bias_rng,
                                    bias_state.context("missing cluster bias state")?,
                                    engine.score(&next)?,
                                );
                                accepted = decision.accepted;
                                if accepted {
                                    *bias_state = Some(decision.proposed);
                                } else {
                                    counts.bias_rejected += 1;
                                }
                                bias_decision = Some(decision);
                            }
                            if accepted {
                                *poses = next;
                                graph = next_graph;
                                channels = graph.channels(&self.config);
                                counts.accepted += 1;
                                counts.guide_accepted += u64::from(guided);
                                counts.transformed_bodies += members.len() as u64;
                                counts.gained_contacts += gained.len() as u64;
                                counts.lost_contacts += lost.len() as u64;
                                if !gained.is_empty() && lost.is_empty() {
                                    counts.accepted_attachments += 1;
                                }
                                if gained.is_empty() && !lost.is_empty() {
                                    counts.accepted_detachments += 1;
                                }
                                if !gained.is_empty() && !lost.is_empty() {
                                    counts.completed_exchanges += 1;
                                }
                            }
                        }
                    } else {
                        counts.internal_geometry_nulls += 1;
                    }
                } else {
                    counts.hard_rejected += 1;
                }
            } else {
                counts.proposal_nulls += 1;
            }
            if record {
                rows.push(json!({"kind":"cluster_event","event":counts.events,"event_time":t,"waiting_time":dt,
                "duration":self.config.duration,"total_rate":rate,"selected_rate":selected.rate,"members":members,"handle":handle,
                "old_poses":old_member_poses,"proposed_poses":proposed_member_poses,
                "retained_poses":members.iter().map(|&i|poses[i]).collect::<Vec<_>>(),"proposal":info,
                "hard_valid":hard_valid,"internal_contact_graph_preserved":internal_equal,
                "physical_accepted":physical_accepted,"accepted":accepted,"gate":sampled,"log_acceptance":alpha,
                "assembly_bias_decision":bias_decision,"proposed_gained_contacts":gained,"proposed_lost_contacts":lost,
                "selection_log_correction":0.,"subset_context":subset_context}));
            }
        }
        Ok((counts, rows))
    }
}
