//! Fixed-duration continuous-time mixture of reversible rigid-subset kernels.
//! Eligibility depends only on internal contacts. State-dependent total rate is
//! handled by exponential holding times, never by an extra acceptance factor.
use crate::{
    assembly_bias::{self, AssemblyBias, AssemblyBiasState},
    depletion::GateOptions,
    docking::{DockingMethod, DockingProposal},
    geometry::{Placed, SphereTree},
    math::*,
    oligomer_proposal::{OligomerConfig, OligomerMixture, internal_key},
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
    /// Constant channel rate per body, independent of external contacts. Zero
    /// preserves the historical dimer/trimer channel order and random stream.
    #[serde(skip_serializing_if = "is_zero")]
    pub singleton_rate: f64,
    /// Optional target-only independent singleton redraw, conditioned on hard
    /// validity for at most this many complete mixture draws. None preserves
    /// the existing correlated transport and its random stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub singleton_independent_max_trials: Option<usize>,
    pub dimer_rate: f64,
    pub trimer_rate: f64,
    pub transport_probability: f64,
    pub correlation: f64,
    #[serde(rename = "local_translation_std_A")]
    pub local_translation_std_a: f64,
    pub local_small_angle_std_degrees: f64,
    /// Transport chart family: the single-body handle atlas (default) or the
    /// joint member/anchor mixture of `DockingProposal::propose_members`.
    #[serde(skip_serializing_if = "TransportCharts::is_handle")]
    pub transport_charts: TransportCharts,
    /// Member charts only: number of retained spectator anchors. The default
    /// nearest pool is fixed by spectator poses and its primary label. Optional
    /// contact-weighted pools correct the complete ordered selection law.
    #[serde(skip_serializing_if = "is_one")]
    pub anchor_count: usize,
    /// Retain a contact-weighted ordered pool and correct its joint selection
    /// probability, or use the historical nearest-spectator pool.
    #[serde(skip_serializing_if = "AnchorPoolSelection::is_nearest")]
    pub anchor_pool_selection: AnchorPoolSelection,
    /// Members only: defensive uniform fraction of primary-anchor selection.
    /// The remaining probability is proportional to boundary contact edges.
    /// None preserves the original uniformly anchored mixture and RNG order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor_contact_uniform_probability: Option<f64>,
    /// Members only: add deterministic two-contact oligomer components to the
    /// learned member-chart mixture. None keeps the member mixture.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oligomer: Option<OligomerConfig>,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransportCharts {
    #[default]
    Handle,
    Members,
}
impl TransportCharts {
    fn is_handle(&self) -> bool {
        *self == Self::Handle
    }
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnchorPoolSelection {
    #[default]
    Nearest,
    ContactWithoutReplacement,
}
impl AnchorPoolSelection {
    fn is_nearest(&self) -> bool {
        *self == Self::Nearest
    }
}
fn is_zero(v: &f64) -> bool {
    *v == 0.
}
fn is_zero_count(v: &u64) -> bool {
    *v == 0
}
fn is_one(v: &usize) -> bool {
    *v == 1
}
impl Default for ClusterPhaseConfig {
    fn default() -> Self {
        Self {
            duration: 0.01,
            singleton_rate: 0.,
            singleton_independent_max_trials: None,
            dimer_rate: 1.,
            trimer_rate: 0.25,
            transport_probability: 0.5,
            correlation: 0.9,
            local_translation_std_a: 0.2,
            local_small_angle_std_degrees: 1.,
            transport_charts: TransportCharts::Handle,
            anchor_count: 1,
            anchor_pool_selection: AnchorPoolSelection::Nearest,
            anchor_contact_uniform_probability: None,
            oligomer: None,
        }
    }
}
impl ClusterPhaseConfig {
    pub fn validate(&self) -> Result<()> {
        for v in [
            self.duration,
            self.singleton_rate,
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
        ensure!(
            self.anchor_count > 0,
            "cluster anchor count must be positive"
        );
        if let Some(epsilon) = self.anchor_contact_uniform_probability {
            ensure!(
                self.transport_charts == TransportCharts::Members,
                "contact-aware anchors require member transport charts"
            );
            ensure!(
                epsilon.is_finite() && epsilon > 0. && epsilon <= 1.,
                "anchor contact uniform probability must be in (0, 1]"
            );
        }
        if self.anchor_pool_selection == AnchorPoolSelection::ContactWithoutReplacement {
            ensure!(
                self.transport_charts == TransportCharts::Members
                    && self.anchor_contact_uniform_probability.is_some(),
                "contact pool selection requires member charts and a defensive anchor probability"
            );
        }
        if let Some(oligomer) = &self.oligomer {
            ensure!(
                self.transport_charts == TransportCharts::Members,
                "oligomer components require member transport charts"
            );
            oligomer.validate()?;
        }
        if let Some(cap) = self.singleton_independent_max_trials {
            ensure!(cap > 0, "independent singleton trial cap must be positive");
            ensure!(
                self.singleton_rate > 0. && self.dimer_rate == 0. && self.trimer_rate == 0.,
                "independent redraw control requires singleton-only channels"
            );
            ensure!(
                self.transport_charts == TransportCharts::Members
                    && self.anchor_contact_uniform_probability.is_some(),
                "independent singleton redraw requires member charts and defensive contact anchors"
            );
        }
        Ok(())
    }
    pub fn enabled(&self) -> bool {
        self.duration > 0.
            && (self.singleton_rate > 0. || self.dimer_rate > 0. || self.trimer_rate > 0.)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ClusterPhaseCounts {
    pub phases: u64,
    pub events: u64,
    #[serde(default, skip_serializing_if = "is_zero_count")]
    pub singleton_events: u64,
    #[serde(default, skip_serializing_if = "is_zero_count")]
    pub independent_raw_trials: u64,
    #[serde(default, skip_serializing_if = "is_zero_count")]
    pub independent_cap_exhaustions: u64,
    #[serde(default, skip_serializing_if = "is_zero_count")]
    pub independent_valid_candidates: u64,
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
}
macro_rules! fields {
    ($m:ident, $a:ident, $b:ident) => {
        $m!(
            $a,
            $b,
            phases,
            events,
            singleton_events,
            independent_raw_trials,
            independent_cap_exhaustions,
            independent_valid_candidates,
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
            horizon_stops
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
    /// Primary-anchor probabilities for a fixed labeled moving subset.
    /// If c_a is the number of subset-to-spectator contact edges at a, then
    /// q_a = epsilon/M + (1-epsilon)c_a/sum(c). With no boundary contacts the
    /// law is uniform. Every spectator has positive probability, including a
    /// departing contact partner in the reverse move. No native labels enter.
    pub fn anchor_probabilities(&self, members: &[usize], epsilon: f64) -> Result<Vec<f64>> {
        ensure!(
            epsilon.is_finite() && epsilon > 0. && epsilon <= 1.,
            "anchor contact uniform probability must be in (0, 1]"
        );
        let n = self.adjacency.len();
        ensure!(
            self.adjacency.iter().all(|row| row.len() == n),
            "invalid anchor contact graph dimensions"
        );
        ensure!(!members.is_empty(), "empty anchor-selection subset");
        let mut selected = vec![false; n];
        for &i in members {
            ensure!(
                i < n && !selected[i],
                "invalid anchor-selection subset index"
            );
            selected[i] = true;
        }
        let spectators = n - members.len();
        ensure!(spectators > 0, "no external anchor");
        let contacts: Vec<usize> = (0..n)
            .map(|a| {
                if selected[a] {
                    0
                } else {
                    members.iter().filter(|&&i| self.adjacency[i][a]).count()
                }
            })
            .collect();
        let total: usize = contacts.iter().sum();
        let uniform = if total == 0 { 1. } else { epsilon } / spectators as f64;
        ensure!(uniform > 0., "unrepresentable defensive anchor probability");
        Ok((0..n)
            .map(|a| {
                if selected[a] {
                    0.
                } else if total == 0 {
                    uniform
                } else {
                    uniform + (1. - epsilon) * (contacts[a] as f64 / total as f64)
                }
            })
            .collect())
    }
    /// Next ordered anchor, excluding already retained spectators. The contact
    /// term is renormalized over the remainder; if no remaining contacts exist,
    /// every remaining spectator has equal probability. An empty retained list
    /// follows the historical primary law bit for bit.
    pub fn remaining_anchor_probabilities(
        &self,
        members: &[usize],
        retained: &[usize],
        epsilon: f64,
    ) -> Result<Vec<f64>> {
        let primary = self.anchor_probabilities(members, epsilon)?;
        if retained.is_empty() {
            return Ok(primary);
        }
        let n = self.adjacency.len();
        let mut excluded = vec![false; n];
        for &i in members {
            excluded[i] = true;
        }
        for &i in retained {
            ensure!(i < n && !excluded[i], "invalid retained anchor pool");
            excluded[i] = true;
        }
        let remaining = n - members.len() - retained.len();
        ensure!(remaining > 0, "no remaining external anchor");
        let contacts: Vec<usize> = (0..n)
            .map(|a| {
                if excluded[a] {
                    0
                } else {
                    members.iter().filter(|&&i| self.adjacency[i][a]).count()
                }
            })
            .collect();
        let total: usize = contacts.iter().sum();
        let uniform = if total == 0 { 1. } else { epsilon } / remaining as f64;
        ensure!(uniform > 0., "unrepresentable defensive pool probability");
        Ok((0..n)
            .map(|a| {
                if excluded[a] {
                    0.
                } else if total == 0 {
                    uniform
                } else {
                    uniform + (1. - epsilon) * (contacts[a] as f64 / total as f64)
                }
            })
            .collect())
    }
    /// Joint probability of the SAME ordered tuple at either endpoint. Do not
    /// redraw, sort, or substitute a new pool when evaluating the reverse law.
    pub fn anchor_pool_log_probability(
        &self,
        members: &[usize],
        pool: &[usize],
        epsilon: f64,
    ) -> Result<f64> {
        ensure!(!pool.is_empty(), "empty retained anchor pool");
        let mut log_probability = 0.;
        for (j, &a) in pool.iter().enumerate() {
            let probabilities =
                self.remaining_anchor_probabilities(members, &pool[..j], epsilon)?;
            ensure!(
                a < probabilities.len() && probabilities[a] > 0.,
                "invalid retained anchor label"
            );
            log_probability += probabilities[a].ln();
        }
        ensure!(
            log_probability.is_finite(),
            "nonfinite retained-pool probability"
        );
        Ok(log_probability)
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
    /// Singletons have constant per-label rates regardless of contacts; pairs
    /// and unique connected triples depend only on carried internal geometry.
    /// No singleton filtering by isolation or number of neighbors is allowed.
    pub fn channels(&self, cfg: &ClusterPhaseConfig) -> Vec<Channel> {
        let mut out = Vec::new();
        let n = self.adjacency.len();
        if cfg.singleton_rate > 0. {
            out.extend((0..n).map(|i| Channel {
                members: vec![i],
                rate: cfg.singleton_rate,
            }));
        }
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

/// Primary spectator plus its nearest spectators by center distance, ties
/// by label; at most `count` labels. Only spectator poses enter, so a rigid
/// move of `members` leaves the pool unchanged.
pub fn anchor_pool(
    state: &[Pose],
    members: &[usize],
    primary: usize,
    count: usize,
) -> Result<Vec<usize>> {
    ensure!(
        count > 0 && primary < state.len() && !members.contains(&primary),
        "invalid cluster primary anchor"
    );
    let mut rest: Vec<_> = (0..state.len())
        .filter(|&i| i != primary && !members.contains(&i))
        .map(|i| {
            let d = sub(state[i].position, state[primary].position);
            (dot(d, d), i)
        })
        .collect();
    rest.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    Ok(std::iter::once(primary)
        .chain(rest.into_iter().map(|(_, i)| i))
        .take(count)
        .collect())
}

/// Continue a separately sampled historical primary anchor. Every new label is
/// drawn once, without replacement, and retained as proposal auxiliary state.
pub fn contact_anchor_pool(
    rng: &mut StdRng,
    graph: &ContactGraph,
    members: &[usize],
    primary: usize,
    count: usize,
    epsilon: f64,
) -> Result<Vec<usize>> {
    ensure!(count > 0, "anchor pool count must be positive");
    graph.anchor_pool_log_probability(members, &[primary], epsilon)?;
    let count = count.min(graph.adjacency.len() - members.len());
    let mut pool = vec![primary];
    while pool.len() < count {
        let probabilities = graph.remaining_anchor_probabilities(members, &pool, epsilon)?;
        let mut draw = rng.random::<f64>() * probabilities.iter().sum::<f64>();
        let mut chosen = None;
        for (a, &p) in probabilities.iter().enumerate().filter(|(_, p)| **p > 0.) {
            chosen = Some(a);
            if draw < p {
                break;
            }
            draw -= p;
        }
        pool.push(chosen.context("empty remaining anchor law")?);
    }
    Ok(pool)
}

struct AnchorSelection {
    primary: usize,
    primary_forward: f64,
    epsilon: f64,
    /// None keeps legacy trace fields and arithmetic unchanged.
    retained_pool: Option<(Vec<usize>, f64)>,
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
        if let (TransportCharts::Members, Some(p)) = (config.transport_charts, &proposal) {
            ensure!(
                p.method() == DockingMethod::PosteriorInvolution && !p.is_periodic(),
                "member transport charts need a nonperiodic posterior-involution model"
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
    /// Learned member-chart branch: the plain member mixture, or the member
    /// mixture with fused oligomer components. Neither draws a branch coin.
    fn learned_member_proposal(
        &self,
        rng: &mut StdRng,
        poses: &[Pose],
        members: &[usize],
        handle_position: usize,
        anchors: &[Pose],
        record: bool,
    ) -> Result<(Option<Pose>, Value)> {
        let proposal = self.proposal.as_ref().unwrap();
        let member_poses: Vec<_> = members.iter().map(|&i| poses[i]).collect();
        if let Some(cap) = self.config.singleton_independent_max_trials {
            return self.conditioned_singleton_proposal(rng, poses, members, anchors, cap, record);
        }
        let Some(settings) = &self.config.oligomer else {
            return proposal.propose_members_learned(rng, &member_poses, handle_position, anchors);
        };
        let spectators: Vec<_> = (0..poses.len())
            .filter(|i| !members.contains(i))
            .map(|i| poses[i])
            .collect();
        let mixture = OligomerMixture::build(
            proposal,
            self.tree,
            self.wall,
            &member_poses,
            &spectators,
            anchors,
            settings,
        )?;
        mixture.propose(rng, &member_poses, handle_position)
    }

    /// Independent complete-mixture rejection sampling, with one retained
    /// spectator context and finite-cap self-loop. The common success
    /// normalizer cancels; no source chart or correlated map is retried.
    fn conditioned_singleton_proposal(
        &self,
        rng: &mut StdRng,
        poses: &[Pose],
        members: &[usize],
        anchors: &[Pose],
        cap: usize,
        record: bool,
    ) -> Result<(Option<Pose>, Value)> {
        ensure!(
            members.len() == 1 && cap > 0,
            "invalid conditioned singleton context"
        );
        let proposal = self.proposal.as_ref().unwrap();
        let old = [poses[members[0]]];
        let spectators: Vec<_> = poses
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != members[0])
            .map(|(_, p)| *p)
            .collect();
        let placed: Vec<_> = spectators.iter().map(|&p| Placed::new(p)).collect();
        let mixture = self
            .config
            .oligomer
            .as_ref()
            .map(|settings| {
                OligomerMixture::build(
                    proposal,
                    self.tree,
                    self.wall,
                    &old,
                    &spectators,
                    anchors,
                    settings,
                )
            })
            .transpose()?;
        let density = |p: Pose| -> Result<f64> {
            match &mixture {
                Some(m) => Ok(m.log_density(&[p])),
                None => proposal.members_log_density(&[p], anchors),
            }
        };
        let old_log = density(old[0])?;
        let mut info = json!({"branch":"independent_conditioned",
            "charts":if mixture.is_some(){"oligomer"}else{"members"},
            "max_trials":cap,"raw_trial_count":0,"cap_exhausted":false,
            "full_old_log_density":old_log});
        if let Some(m) = &mixture {
            info["oligomer"] = json!({"fused_components":m.fused.len(),
                "fit_candidates":m.fit_candidates(),"hard_checks":m.hard_checks(),
                "build_seconds":m.build_seconds});
        }
        if record {
            info["raw_trials"] = json!([]);
        }
        if !old_log.is_finite() {
            info["null_reason"] = json!("Nonfinite independent source density; no retries");
            return Ok((None, info));
        }
        for trial in 1..=cap {
            let (candidate, mut raw) = match &mixture {
                Some(m) => m.draw_singleton_independent(rng)?,
                None => proposal.draw_singleton_independent(rng, anchors)?,
            };
            let hard = candidate.is_some_and(|p| {
                self.wall.contains(p) && {
                    let body = Placed::new(p);
                    !placed.iter().any(|s| self.tree.overlaps(&body, s))
                }
            });
            raw["trial"] = json!(trial);
            raw["candidate"] = json!(candidate);
            raw["hard_valid"] = json!(hard);
            info["raw_trial_count"] = json!(trial);
            if record {
                info["raw_trials"].as_array_mut().unwrap().push(raw.clone());
            }
            if !hard {
                continue;
            }
            let p = candidate.unwrap();
            let new_log = density(p)?;
            info["full_new_log_density"] = json!(new_log);
            info["target_label"] = raw["target_label"].clone();
            info["target_latent"] = raw["target_latent"].clone();
            if !new_log.is_finite() || !(old_log - new_log).is_finite() {
                info["null_reason"] = json!("Nonfinite independent endpoint density; no retries");
                return Ok((None, info));
            }
            info["map_log_reverse_forward"] = json!(old_log - new_log);
            info["log_reverse_forward"] = json!(old_log - new_log);
            return Ok((Some(p), info));
        }
        info["cap_exhausted"] = json!(true);
        info["null_reason"] = json!("Independent hard-conditioning trial cap exhausted");
        Ok((None, info))
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
            match members.len() {
                1 => counts.singleton_events += 1,
                2 => counts.dimer_events += 1,
                3 => counts.trimer_events += 1,
                _ => unreachable!("unsupported cluster channel size"),
            }
            if transport {
                counts.transport_events += 1;
            } else {
                counts.local_events += 1;
            }
            let mut info = json!({"branch":"local_rigid","log_reverse_forward":0.});
            // Retain the sampled primary label for its exact reverse law.
            // This selection is separate from the invariant internal-subset clock.
            let mut anchor_selection: Option<AnchorSelection> = None;
            let candidate = if transport {
                let spectator_indices: Vec<_> =
                    (0..poses.len()).filter(|i| !members.contains(i)).collect();
                if spectator_indices.is_empty() {
                    info = json!({"branch":"transport","null_reason":"no_external_anchor"});
                    None
                } else if let Some(epsilon) = self.config.anchor_contact_uniform_probability {
                    let proposal = self.proposal.as_ref().unwrap();
                    // The defensive branch is an independent symmetric kernel.
                    // It must not inherit a state-dependent anchor label that it
                    // never uses to generate its candidate.
                    if proposal_rng.random::<f64>() < proposal.member_uniform_weight() {
                        let (p, trace) = proposal.draw_member_uniform(proposal_rng)?;
                        info = trace;
                        p
                    } else {
                        let probabilities = graph.anchor_probabilities(members, epsilon)?;
                        let mut draw =
                            proposal_rng.random::<f64>() * probabilities.iter().sum::<f64>();
                        let mut primary = *spectator_indices.last().unwrap();
                        for &a in &spectator_indices {
                            if draw < probabilities[a] {
                                primary = a;
                                break;
                            }
                            draw -= probabilities[a];
                        }
                        let forward = probabilities[primary];
                        let pool = match self.config.anchor_pool_selection {
                            AnchorPoolSelection::Nearest => {
                                anchor_pool(poses, members, primary, self.config.anchor_count)?
                            }
                            AnchorPoolSelection::ContactWithoutReplacement => contact_anchor_pool(
                                proposal_rng,
                                &graph,
                                members,
                                primary,
                                self.config.anchor_count,
                                epsilon,
                            )?,
                        };
                        let retained_pool = if self.config.anchor_pool_selection
                            == AnchorPoolSelection::ContactWithoutReplacement
                        {
                            Some((
                                pool.clone(),
                                graph.anchor_pool_log_probability(members, &pool, epsilon)?,
                            ))
                        } else {
                            None
                        };
                        let anchors: Vec<_> = pool.iter().map(|&i| poses[i]).collect();
                        let position = members.iter().position(|&i| i == handle).unwrap();
                        let (p, mut trace) = self.learned_member_proposal(
                            proposal_rng,
                            poses,
                            members,
                            position,
                            &anchors,
                            record,
                        )?;
                        trace["primary_anchor"] = json!(primary);
                        trace["anchor_pool"] = json!(pool);
                        trace["anchor_forward_probability"] = json!(forward);
                        trace["anchor_reverse_probability"] = Value::Null;
                        trace["anchor_log_reverse_forward"] = Value::Null;
                        if let Some((_, log_forward)) = &retained_pool {
                            trace["anchor_pool_selection"] =
                                json!(self.config.anchor_pool_selection);
                            trace["pool_forward_log_probability"] = json!(log_forward);
                            trace["pool_reverse_log_probability"] = Value::Null;
                        }
                        trace["map_log_reverse_forward"] = trace["log_reverse_forward"].clone();
                        // The complete correction exists only after evaluating
                        // the reverse primary probability at a valid endpoint.
                        trace["log_reverse_forward"] = Value::Null;
                        anchor_selection = Some(AnchorSelection {
                            primary,
                            primary_forward: forward,
                            epsilon,
                            retained_pool,
                        });
                        info = trace;
                        p
                    }
                } else if self.config.transport_charts == TransportCharts::Members {
                    let primary =
                        spectator_indices[proposal_rng.random_range(0..spectator_indices.len())];
                    let pool = anchor_pool(poses, members, primary, self.config.anchor_count)?;
                    let member_poses: Vec<_> = members.iter().map(|&i| poses[i]).collect();
                    let anchors: Vec<_> = pool.iter().map(|&i| poses[i]).collect();
                    let position = members.iter().position(|&i| i == handle).unwrap();
                    let proposal = self.proposal.as_ref().unwrap();
                    let (p, mut trace) = if self.config.oligomer.is_none() {
                        proposal.propose_members(proposal_rng, &member_poses, position, &anchors)?
                    } else if proposal_rng.random::<f64>() < proposal.member_uniform_weight() {
                        proposal.draw_member_uniform(proposal_rng)?
                    } else {
                        self.learned_member_proposal(
                            proposal_rng,
                            poses,
                            members,
                            position,
                            &anchors,
                            record,
                        )?
                    };
                    trace["anchor_pool"] = json!(pool);
                    info = trace;
                    p
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
            if info["branch"] == "independent_conditioned" {
                counts.independent_raw_trials += info["raw_trial_count"]
                    .as_u64()
                    .context("missing independent raw-trial count")?;
                counts.independent_cap_exhaustions += u64::from(info["cap_exhausted"] == true);
                counts.independent_valid_candidates += u64::from(candidate.is_some());
            }
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
            let mut offsets_preserved = true;
            if let Some(new_handle) = candidate {
                let mut correction = if anchor_selection.is_some() {
                    &info["map_log_reverse_forward"]
                } else {
                    &info["log_reverse_forward"]
                }
                .as_f64()
                .context("cluster proposal correction missing")?;
                ensure!(correction.is_finite(), "nonfinite cluster map correction");
                let trial =
                    RigidSubset::new(self.tree, poses, members, handle, new_handle, self.rd)?;
                proposed_member_poses = Some(trial.proposed_poses().to_vec());
                hard_valid = trial.hard_valid(Some(self.wall), [0.; 3]);
                ensure!(
                    info["branch"] != "independent_conditioned" || hard_valid,
                    "conditioned candidate disagrees with production hard predicate"
                );
                if hard_valid {
                    counts.hard_valid += 1;
                    let mut next = poses.clone();
                    for (&i, &p) in members.iter().zip(trial.proposed_poses()) {
                        next[i] = p;
                    }
                    let next_graph = graph.updated(&self.exclusion, &next, members);
                    internal_equal = graph.internal_equal(&next_graph, members);
                    // The oligomer catalogue is a function of the rounded
                    // internal offsets. Rejecting a trial that changes them is
                    // symmetric in the endpoints and makes both catalogues
                    // bitwise identical whenever the trial proceeds.
                    if info["charts"] == "oligomer" {
                        offsets_preserved = internal_key(&old_member_poses).0
                            == internal_key(trial.proposed_poses()).0;
                    }
                    if internal_equal && offsets_preserved {
                        if let Some(selection) = &anchor_selection {
                            let AnchorSelection {
                                primary,
                                primary_forward: forward,
                                epsilon,
                                retained_pool,
                            } = selection;
                            let reverse =
                                next_graph.anchor_probabilities(members, *epsilon)?[*primary];
                            ensure!(
                                forward.is_finite()
                                    && *forward > 0.
                                    && reverse.is_finite()
                                    && reverse > 0.,
                                "invalid forward/reverse primary-anchor probability"
                            );
                            let anchor_correction = if let Some((pool, log_forward)) = retained_pool
                            {
                                let log_reverse = next_graph
                                    .anchor_pool_log_probability(members, pool, *epsilon)?;
                                info["pool_reverse_log_probability"] = json!(log_reverse);
                                log_reverse - log_forward
                            } else {
                                reverse.ln() - forward.ln()
                            };
                            correction += anchor_correction;
                            ensure!(correction.is_finite(), "nonfinite total cluster correction");
                            info["anchor_reverse_probability"] = json!(reverse);
                            info["anchor_log_reverse_forward"] = json!(anchor_correction);
                            info["log_reverse_forward"] = json!(correction);
                        }
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
                "hard_valid":hard_valid,"internal_contact_graph_preserved":internal_equal,"internal_offsets_preserved":offsets_preserved,
                "physical_accepted":physical_accepted,"accepted":accepted,"gate":sampled,"log_acceptance":alpha,
                "assembly_bias_decision":bias_decision,"proposed_gained_contacts":gained,"proposed_lost_contacts":lost,
                "selection_log_correction":0.,"subset_context":subset_context}));
            }
        }
        Ok((counts, rows))
    }
}
