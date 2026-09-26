//! Normalized two-contact proposal charts for one rigid oligomer pose.
//!
//! All original member/anchor charts (including reciprocal branches) survive.
//! Additional Gaussian charts are local products of two distinct members'
//! preferences pulled back to the canonical member-zero pose. Products define
//! a proposal, never a pair approximation to the physical depletion energy.
//! The catalogue depends only on internal geometry and stationary spectators.
use crate::{
    basin_involution::{BasinPair, BasinTrace, FixedBasinInvolution},
    docking::DockingProposal,
    math::*,
    proposal::GaussianComponentParameters,
};
use anyhow::{Context, Result, ensure};
use rand::{distr::Open01, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::f64::consts::PI;

pub type Vec6 = [f64; 6];
pub type Mat6 = [[f64; 6]; 6];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct FusionConfig {
    /// Fraction of learned destination mass given to fused charts. The
    /// defensive uniform branch remains separate and unchanged.
    pub fused_weight: f64,
    pub max_fused: usize,
    pub candidates_per_member_anchor: usize,
    pub max_pair_candidates: usize,
}
impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            fused_weight: 0.5,
            max_fused: 32,
            candidates_per_member_anchor: 8,
            max_pair_candidates: 128,
        }
    }
}
impl FusionConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.fused_weight.is_finite() && (0. ..1.).contains(&self.fused_weight),
            "Fused mass must lie in [0,1), retaining every original chart"
        );
        ensure!(
            self.max_fused > 0
                && self.candidates_per_member_anchor > 0
                && self.max_pair_candidates > 0,
            "Fusion budgets must be positive"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionDiagnostics {
    pub original_charts: usize,
    pub short_list_candidates: usize,
    pub possible_distinct_member_pairs: usize,
    pub screened_pairs: usize,
    pub exact_pair_candidates: usize,
    pub compatible_products: usize,
    pub numerical_candidate_skips: usize,
    pub numerical_pair_skips: usize,
    pub fused_charts: usize,
    pub effective_fused_weight: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContactLabel {
    pub member: usize,
    pub anchor: usize,
    pub branch: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct GaussianProduct {
    pub mean: Vec6,
    pub covariance: Mat6,
    /// log integral N(x; mean1,cov1) N(x; mean2,cov2) dx.
    pub log_compatibility: f64,
}

/// Product of two normalized full-covariance Euclidean Gaussians. One solve
/// against cov1+cov2 supplies both compatibility and the normalized product.
pub fn gaussian_product(
    mean1: Vec6,
    cov1: Mat6,
    mean2: Vec6,
    cov2: Mat6,
) -> Result<GaussianProduct> {
    cholesky(cov1)?;
    cholesky(cov2)?;
    let sum = std::array::from_fn(|i| std::array::from_fn(|j| cov1[i][j] + cov2[i][j]));
    let (lower, logdet) = cholesky(sum)?;
    let difference = std::array::from_fn(|i| mean2[i] - mean1[i]);
    let solved = solve_spd(lower, difference);
    let mean = std::array::from_fn(|i| mean1[i] + dot6(cov1[i], solved));
    let columns: Mat6 = std::array::from_fn(|j| solve_spd(lower, cov1[j]));
    let covariance: Mat6 =
        std::array::from_fn(|i| std::array::from_fn(|j| cov1[i][j] - dot6(cov1[i], columns[j])));
    let covariance = std::array::from_fn(|i| {
        std::array::from_fn(|j| 0.5 * (covariance[i][j] + covariance[j][i]))
    });
    cholesky(covariance)?;
    let log_compatibility = -3. * (2. * PI).ln() - logdet - 0.5 * dot6(difference, solved);
    ensure!(
        mean.iter().all(|x| x.is_finite()) && log_compatibility.is_finite(),
        "Nonfinite Gaussian product"
    );
    Ok(GaussianProduct {
        mean,
        covariance,
        log_compatibility,
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct PulledContact {
    pub mean_pose: Pose,
    /// Covariance in [p-p_mean, ell*cayley(R R_mean^-1)]. Its complete
    /// linearization includes reciprocal inversion and the member lever arm.
    pub covariance: Mat6,
    pub jacobian: Mat6,
}

/// Analytic first derivative of one member's Gaussian coordinate map into the
/// common member-zero pose. No finite-difference geometry or current contacts
/// enter the construction. `internal` is member i relative to member zero.
pub fn pullback_contact(
    parameters: &GaussianComponentParameters,
    inverted: bool,
    anchor: Pose,
    internal: Pose,
    angular_length: f64,
) -> Result<PulledContact> {
    ensure!(
        angular_length.is_finite() && angular_length > 0.,
        "Invalid angular length"
    );
    let x = parameters.mean;
    let u = [
        x[3] / angular_length,
        x[4] / angular_length,
        x[5] / angular_length,
    ];
    let raw_rotation = matmul(cayley(u), parameters.anchor_rotation);
    let raw_position = std::array::from_fn(|i| parameters.anchor_position[i] + x[i]);
    let relative = Pose {
        position: raw_position,
        orientation: quaternion(raw_rotation),
    };
    let relative = if inverted {
        invert_relative_pose(relative)
    } else {
        relative
    };
    let member = compose(anchor, relative);
    let mean_pose = compose(member, invert_relative_pose(internal));
    let mean_rotation = rotation(mean_pose.orientation);
    let anchor_rotation = rotation(anchor.orientation);
    let internal_inverse_rotation = transpose(rotation(internal.orientation));
    let mut jacobian = [[0.; 6]; 6];
    for k in 0..6 {
        let raw_dp = std::array::from_fn(|i| if i == k { 1. } else { 0. });
        let raw_dr = if k < 3 {
            [[0.; 3]; 3]
        } else {
            matmul(
                cayley_derivative(u, k - 3, angular_length),
                parameters.anchor_rotation,
            )
        };
        let (dp, dr) = if inverted {
            let rt = transpose(raw_rotation);
            let drt = transpose(raw_dr);
            (
                scale(add(matvec(drt, raw_position), matvec(rt, raw_dp)), -1.),
                drt,
            )
        } else {
            (raw_dp, raw_dr)
        };
        let dh_rotation = matmul(matmul(anchor_rotation, dr), internal_inverse_rotation);
        let dh_position = sub(
            matvec(anchor_rotation, dp),
            matvec(dh_rotation, internal.position),
        );
        let drelative = matmul(dh_rotation, transpose(mean_rotation));
        // At identity, d cayley(R) = vex(dR-dR^T)/4.
        let du = scale(vex_skew(drelative), 0.25 * angular_length);
        for i in 0..3 {
            jacobian[i][k] = dh_position[i];
            jacobian[i + 3][k] = du[i];
        }
    }
    let covariance = congruence(jacobian, parameters.covariance);
    cholesky(covariance)?;
    Ok(PulledContact {
        mean_pose,
        covariance,
        jacobian,
    })
}

#[derive(Clone, Debug)]
struct Candidate {
    label: ContactLabel,
    pulled: PulledContact,
    log_weight: f64,
}
#[derive(Clone, Debug)]
struct FusedChart {
    parameters: GaussianComponentParameters,
    logdet: f64,
    contacts: [ContactLabel; 2],
    log_compatibility: f64,
}

/// A frozen normalized catalogue over the pose of member zero. Catalogue
/// reconstruction is unnecessary for an elementary map: reuse this very object
/// at both endpoints. Rebuilding from rigidly moved members gives the same
/// mathematical catalogue (floating-point geometry remains an obligation).
pub struct FusedCatalogue {
    internals: Vec<Pose>,
    pool: Vec<Pose>,
    branches: usize,
    branch_metadata: Vec<(usize, bool)>,
    log_weights: Vec<f64>,
    fused: Vec<FusedChart>,
    fused_map: Option<FixedBasinInvolution>,
    diagnostics: FusionDiagnostics,
    angular_length: f64,
    correlation: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FusionStep {
    pub pose: Pose,
    pub inverse_trace: BasinTrace,
    pub source_latent: Vec6,
    pub target_latent: Vec6,
    pub log_extended_jacobian: f64,
    pub log_auxiliary_ratio: f64,
    pub component_log_reverse_forward: f64,
    pub label_log_reverse_forward: f64,
    pub expanded_log_reverse_forward: f64,
    pub full_old_member_log_density: f64,
    pub full_new_member_log_density: f64,
    pub log_reverse_forward: f64,
    pub identity: bool,
}

impl FusedCatalogue {
    pub fn build(
        proposal: &DockingProposal,
        members: &[Pose],
        pool: &[Pose],
        config: &FusionConfig,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            proposal.fusion_supported(),
            "Fused charts require open posterior member maps"
        );
        ensure!(
            !members.is_empty() && !pool.is_empty(),
            "Empty member or anchor pool"
        );
        for p in members.iter().chain(pool) {
            p.validate()?;
        }
        let internals: Vec<_> = members.iter().map(|&m| relative(members[0], m)).collect();
        let branches = proposal.fusion_log_weights().len();
        let original_charts = members
            .len()
            .checked_mul(pool.len())
            .and_then(|v| v.checked_mul(branches))
            .context("Too many member charts")?;
        let ell = proposal.fusion_angular_length();
        let mut diagnostics = FusionDiagnostics {
            original_charts,
            short_list_candidates: 0,
            possible_distinct_member_pairs: 0,
            screened_pairs: 0,
            exact_pair_candidates: 0,
            compatible_products: 0,
            numerical_candidate_skips: 0,
            numerical_pair_skips: 0,
            fused_charts: 0,
            effective_fused_weight: 0.,
        };
        let mut fused = Vec::new();
        let mut product_scores = Vec::new();
        if config.fused_weight > 0. && members.len() > 1 {
            let parameters = proposal.fusion_parameters();
            // Density-height ranking is metadata only; broad fallback charts
            // remain in the exact original catalogue with their full mass.
            let mut ranked: Vec<_> = parameters
                .iter()
                .enumerate()
                .map(|(b, p)| {
                    let (_, logdet) = cholesky(p.covariance)?;
                    Ok((b, proposal.fusion_log_weights()[b] - logdet))
                })
                .collect::<Result<_>>()?;
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            ranked.truncate(config.candidates_per_member_anchor.min(ranked.len()));
            let mut candidates = Vec::new();
            let label_offset = ((members.len() * pool.len()) as f64).ln();
            for (i, &internal) in internals.iter().enumerate() {
                for (a, &anchor) in pool.iter().enumerate() {
                    for &(b, _) in &ranked {
                        match pullback_contact(
                            &parameters[b],
                            proposal.fusion_inverted(b),
                            anchor,
                            internal,
                            ell,
                        ) {
                            Ok(pulled) => candidates.push(Candidate {
                                label: ContactLabel {
                                    member: i,
                                    anchor: a,
                                    branch: b,
                                },
                                pulled,
                                log_weight: proposal.fusion_log_weights()[b] - label_offset,
                            }),
                            Err(_) => diagnostics.numerical_candidate_skips += 1,
                        }
                    }
                }
            }
            diagnostics.short_list_candidates = candidates.len();
            // A cheap isotropic compatibility screen avoids matrix solves for
            // every tuple. It ranks proposal candidates only, never truncates
            // physical support or modifies the original component densities.
            let mut pairs = Vec::new();
            for i in 0..candidates.len() {
                for j in i + 1..candidates.len() {
                    if candidates[i].label.member == candidates[j].label.member {
                        continue;
                    }
                    diagnostics.possible_distinct_member_pairs += 1;
                    if let Ok(score) = cheap_pair_score(&candidates[i], &candidates[j], ell) {
                        pairs.push((i, j, score));
                    } else {
                        diagnostics.numerical_pair_skips += 1;
                    }
                }
            }
            diagnostics.screened_pairs = pairs.len();
            pairs.sort_by(|a, b| {
                ranking_score(b.2)
                    .total_cmp(&ranking_score(a.2))
                    .then(candidates[a.0].label.cmp(&candidates[b.0].label))
                    .then(candidates[a.1].label.cmp(&candidates[b.1].label))
            });
            pairs.truncate(config.max_pair_candidates.min(pairs.len()));
            diagnostics.exact_pair_candidates = pairs.len();
            let mut products = Vec::new();
            for (i, j, _) in pairs {
                match fuse_pair(&candidates[i], &candidates[j], ell) {
                    Ok((chart, score)) => products.push((chart, score, i, j)),
                    Err(_) => diagnostics.numerical_pair_skips += 1,
                }
            }
            diagnostics.compatible_products = products.len();
            products.sort_by(|a, b| {
                ranking_score(b.1)
                    .total_cmp(&ranking_score(a.1))
                    .then(a.0.contacts.cmp(&b.0.contacts))
            });
            products.truncate(config.max_fused.min(products.len()));
            // Chart labels are semantic contact tuples, not near-tied score
            // ranks. Spectator rotations can give identical compatibility at
            // different locations; floating roundoff must not swap those maps.
            products.sort_by(|a, b| a.0.contacts.cmp(&b.0.contacts));
            for (chart, score, _, _) in products {
                fused.push(chart);
                product_scores.push(score);
            }
        }
        diagnostics.fused_charts = fused.len();
        let eta = if fused.is_empty() {
            0.
        } else {
            config.fused_weight
        };
        diagnostics.effective_fused_weight = eta;
        let offset = ((members.len() * pool.len()) as f64).ln();
        let mut log_weights = Vec::with_capacity(original_charts + fused.len());
        for _ in 0..members.len() * pool.len() {
            log_weights.extend(
                proposal
                    .fusion_log_weights()
                    .iter()
                    .map(|w| w - offset + (-eta).ln_1p()),
            );
        }
        let fused_map = if fused.is_empty() {
            None
        } else {
            let normalizer = log_sum(&product_scores);
            log_weights.extend(product_scores.iter().map(|s| eta.ln() + s - normalizer));
            Some(FixedBasinInvolution::new(
                fused.iter().map(|f| f.parameters.clone()).collect(),
                ell,
                proposal.fusion_correlation(),
                (0..fused.len())
                    .map(|i| BasinPair {
                        first: i,
                        second: i,
                        weight: 1.,
                    })
                    .collect(),
            )?)
        };
        ensure!(
            (log_sum(&log_weights)).abs() < 1e-10,
            "Catalogue weights are not normalized"
        );
        Ok(Self {
            internals,
            pool: pool.to_vec(),
            branches,
            branch_metadata: (0..branches)
                .map(|b| {
                    (
                        proposal.fusion_base_component(b),
                        proposal.fusion_inverted(b),
                    )
                })
                .collect(),
            log_weights,
            fused,
            fused_map,
            diagnostics,
            angular_length: ell,
            correlation: proposal.fusion_correlation(),
        })
    }
    pub fn original_count(&self) -> usize {
        self.diagnostics.original_charts
    }
    pub fn chart_count(&self) -> usize {
        self.log_weights.len()
    }
    pub fn diagnostics(&self) -> &FusionDiagnostics {
        &self.diagnostics
    }
    pub fn log_weights(&self) -> &[f64] {
        &self.log_weights
    }
    pub fn internal_poses(&self) -> &[Pose] {
        &self.internals
    }
    pub fn member_pose(&self, collective: Pose, member: usize) -> Result<Pose> {
        Ok(compose(
            collective,
            *self.internals.get(member).context("Invalid member index")?,
        ))
    }
    pub fn label(&self, index: usize) -> Result<Value> {
        ensure!(index < self.chart_count(), "Invalid fused catalogue label");
        if index < self.original_count() {
            let label = self.original_label(index);
            Ok(
                json!({"kind":"original","member":label.member,"anchor":label.anchor,"branch":label.branch,
                    "base_component_index":self.branch_metadata[label.branch].0, "inverted":self.branch_metadata[label.branch].1}),
            )
        } else {
            let f = &self.fused[index - self.original_count()];
            Ok(
                json!({"kind":"fused","contacts":f.contacts,"log_compatibility":f.log_compatibility}),
            )
        }
    }
    fn original_label(&self, index: usize) -> ContactLabel {
        ContactLabel {
            member: index / (self.pool.len() * self.branches),
            anchor: index / self.branches % self.pool.len(),
            branch: index % self.branches,
        }
    }
    pub fn encode(
        &self,
        proposal: &DockingProposal,
        index: usize,
        collective: Pose,
    ) -> Result<Vec6> {
        ensure!(index < self.chart_count(), "Invalid fused catalogue index");
        if index < self.original_count() {
            let l = self.original_label(index);
            proposal.fusion_encode(
                l.branch,
                relative(self.pool[l.anchor], self.member_pose(collective, l.member)?),
            )
        } else {
            self.fused_map
                .as_ref()
                .unwrap()
                .encode(index - self.original_count(), collective)
        }
    }
    pub fn decode(&self, proposal: &DockingProposal, index: usize, z: Vec6) -> Result<Pose> {
        ensure!(index < self.chart_count(), "Invalid fused catalogue index");
        if index < self.original_count() {
            let l = self.original_label(index);
            let m = compose(self.pool[l.anchor], proposal.fusion_decode(l.branch, z)?);
            Ok(compose(m, invert_relative_pose(self.internals[l.member])))
        } else {
            self.fused_map
                .as_ref()
                .unwrap()
                .decode(index - self.original_count(), z)
        }
    }
    pub fn component_log_density(
        &self,
        proposal: &DockingProposal,
        index: usize,
        collective: Pose,
    ) -> Result<f64> {
        ensure!(index < self.chart_count(), "Invalid fused catalogue index");
        if index < self.original_count() {
            let l = self.original_label(index);
            proposal.fusion_component_log_density(
                l.branch,
                relative(self.pool[l.anchor], self.member_pose(collective, l.member)?),
            )
        } else {
            let f = &self.fused[index - self.original_count()];
            let q = quaternion(matmul(
                rotation(collective.orientation),
                transpose(f.parameters.anchor_rotation),
            ));
            if q[0] == 0. {
                return Ok(f64::NEG_INFINITY);
            }
            let z = self.encode(proposal, index, collective)?;
            let norm_u = q[1..].iter().fold(1_f64, |n, v| n.hypot(*v / q[0]));
            Ok(log_standard_normal(z) - f.logdet
                + 3. * self.angular_length.ln()
                + 2. * PI.ln()
                + 4. * norm_u.ln())
        }
    }
    pub fn label_log_densities(
        &self,
        proposal: &DockingProposal,
        collective: Pose,
    ) -> Result<Vec<f64>> {
        (0..self.chart_count())
            .map(|i| Ok(self.log_weights[i] + self.component_log_density(proposal, i, collective)?))
            .collect()
    }
    pub fn log_density(&self, proposal: &DockingProposal, collective: Pose) -> Result<f64> {
        Ok(log_sum(&self.label_log_densities(proposal, collective)?))
    }
    pub fn apply(
        &self,
        proposal: &DockingProposal,
        old: Pose,
        trace: &BasinTrace,
    ) -> Result<FusionStep> {
        ensure!(
            trace.source < self.chart_count() && trace.target < self.chart_count(),
            "Invalid catalogue trace"
        );
        ensure!(
            trace.noise.iter().all(|v| v.is_finite()),
            "Nonfinite trace noise"
        );
        let z = self.encode(proposal, trace.source, old)?;
        let sine = ((1. - self.correlation) * (1. + self.correlation)).sqrt();
        let zp = std::array::from_fn(|i| self.correlation * z[i] + sine * trace.noise[i]);
        let inverse_noise =
            std::array::from_fn(|i| sine * z[i] - self.correlation * trace.noise[i]);
        let identity = self.correlation == 1. && trace.source == trace.target;
        let pose = if identity {
            old
        } else {
            self.decode(proposal, trace.target, zp)?
        };
        let old_logs = self.label_log_densities(proposal, old)?;
        let new_logs = if identity {
            old_logs.clone()
        } else {
            self.label_log_densities(proposal, pose)?
        };
        let full_old = log_sum(&old_logs);
        let full_new = log_sum(&new_logs);
        ensure!(
            full_old.is_finite() && full_new.is_finite(),
            "Nonfinite fused mixture density"
        );
        let source_q = old_logs[trace.source] - self.log_weights[trace.source];
        let target_q = new_logs[trace.target] - self.log_weights[trace.target];
        // Each wrapper is Haar/translation-volume preserving. Recover chart
        // volume from q(pose) = phi(z)/J; this handles original reciprocal
        // wrappers and fused Cayley charts in exactly the same expression.
        let old_volume = log_standard_normal(z) - source_q;
        let new_volume = log_standard_normal(zp) - target_q;
        let log_extended_jacobian = new_volume - old_volume;
        let log_auxiliary_ratio =
            log_standard_normal(inverse_noise) - log_standard_normal(trace.noise);
        let component = log_extended_jacobian + log_auxiliary_ratio;
        let labels = new_logs[trace.target] - full_new + self.log_weights[trace.source]
            - old_logs[trace.source]
            + full_old
            - self.log_weights[trace.target];
        let correction = if identity { 0. } else { full_old - full_new };
        ensure!(
            [
                log_extended_jacobian,
                log_auxiliary_ratio,
                component,
                labels,
                correction
            ]
            .iter()
            .all(|v| v.is_finite()),
            "Nonfinite fused map correction"
        );
        Ok(FusionStep {
            pose,
            inverse_trace: BasinTrace {
                source: trace.target,
                target: trace.source,
                noise: inverse_noise,
            },
            source_latent: z,
            target_latent: zp,
            log_extended_jacobian,
            log_auxiliary_ratio,
            component_log_reverse_forward: component,
            label_log_reverse_forward: labels,
            expanded_log_reverse_forward: component + labels,
            full_old_member_log_density: full_old,
            full_new_member_log_density: full_new,
            log_reverse_forward: correction,
            identity,
        })
    }
    /// Posterior source, independent normalized destination, and an orthogonal
    /// Gaussian noise map. Numerical errors are recorded nulls, never retries.
    pub fn propose(
        &self,
        proposal: &DockingProposal,
        rng: &mut StdRng,
        old: Pose,
    ) -> Result<(Option<FusionStep>, Value)> {
        let source_logs = self.label_log_densities(proposal, old)?;
        let Some(source) = draw_log_category(rng, &source_logs)? else {
            return Ok((
                None,
                json!({"branch":"involution","charts":"fused_members",
                "catalogue":self.diagnostics,"null_reason":"No finite source density"}),
            ));
        };
        let target =
            draw_log_category(rng, &self.log_weights)?.context("No fused catalogue target")?;
        let trace = BasinTrace {
            source,
            target,
            noise: std::array::from_fn(|_| StandardNormal.sample(rng)),
        };
        let labels = json!({"source":self.label(source)?,"target":self.label(target)?});
        match self.apply(proposal, old, &trace) {
            Ok(step) => {
                let mut info = serde_json::to_value(&step)?;
                info["branch"] = json!("involution");
                info["charts"] = json!("fused_members");
                info["source_law"] = json!("posterior");
                info["labels"] = labels;
                info["source_kind"] = self.label(source)?["kind"].clone();
                info["target_kind"] = self.label(target)?["kind"].clone();
                info["trace"] = json!(trace);
                info["catalogue"] = json!(self.diagnostics);
                Ok((Some(step), info))
            }
            Err(error) => Ok((
                None,
                json!({"branch":"involution","charts":"fused_members",
                "catalogue":self.diagnostics,"labels":labels,"trace":trace,"null_reason":error.to_string()}),
            )),
        }
    }
}

/// Geometry-dependent shortlist rankings use transitive fixed bins rather than
/// a nontransitive fuzzy comparator. The unrounded scores still determine all
/// normalized proposal weights. This makes symmetry ties deterministic through
/// both budget cutoffs; exact bin-boundary decisions remain a floating-point
/// implementation obligation. The scale is in dimensionless log-score units.
fn ranking_score(score: f64) -> f64 {
    const QUANTUM: f64 = 1e-9;
    let rounded = (score / QUANTUM).round() * QUANTUM;
    if rounded.is_finite() { rounded } else { score }
}

fn cheap_pair_score(a: &Candidate, b: &Candidate, ell: f64) -> Result<f64> {
    let r = matmul(
        rotation(b.pulled.mean_pose.orientation),
        transpose(rotation(a.pulled.mean_pose.orientation)),
    );
    let denominator = 1. + r[0][0] + r[1][1] + r[2][2];
    ensure!(
        denominator > 1e-10,
        "Pair chart crosses Cayley half-turn seam"
    );
    let numerator = vex_skew(r);
    let m: Vec6 = std::array::from_fn(|i| {
        if i < 3 {
            b.pulled.mean_pose.position[i] - a.pulled.mean_pose.position[i]
        } else {
            ell * numerator[i - 3] / denominator
        }
    });
    let variance = (0..6)
        .map(|i| a.pulled.covariance[i][i] + b.pulled.covariance[i][i])
        .sum::<f64>()
        / 6.;
    ensure!(
        variance > 0. && variance.is_finite(),
        "Invalid pair screen variance"
    );
    let score = a.log_weight + b.log_weight - 3. * variance.ln() - 0.5 * dot6(m, m) / variance;
    ensure!(score.is_finite(), "Nonfinite pair screen");
    Ok(score)
}
fn fuse_pair(a: &Candidate, b: &Candidate, ell: f64) -> Result<(FusedChart, f64)> {
    let (mean_b, cov_b) = rechart(&b.pulled, a.pulled.mean_pose, ell)?;
    let product = gaussian_product([0.; 6], a.pulled.covariance, mean_b, cov_b)?;
    let (_, logdet) = cholesky(product.covariance)?;
    let score = a.log_weight + b.log_weight + product.log_compatibility;
    let parameters = GaussianComponentParameters {
        weight: 1.,
        anchor_position: a.pulled.mean_pose.position,
        anchor_rotation: rotation(a.pulled.mean_pose.orientation),
        mean: product.mean,
        covariance: product.covariance,
    };
    Ok((
        FusedChart {
            parameters,
            logdet,
            contacts: [a.label, b.label],
            log_compatibility: product.log_compatibility,
        },
        score,
    ))
}
/// Transfer a local collective-pose Gaussian to another orientation chart.
/// Translation stays in the common world frame. Cayley rotation derivatives
/// are analytic, including the nonlinear offset between chart centers.
fn rechart(contact: &PulledContact, reference: Pose, ell: f64) -> Result<(Vec6, Mat6)> {
    let r = matmul(
        rotation(contact.mean_pose.orientation),
        transpose(rotation(reference.orientation)),
    );
    let denominator = 1. + r[0][0] + r[1][1] + r[2][2];
    ensure!(
        denominator > 1e-10,
        "Pair chart crosses Cayley half-turn seam"
    );
    let numerator = vex_skew(r);
    let mean = std::array::from_fn(|i| {
        if i < 3 {
            contact.mean_pose.position[i] - reference.position[i]
        } else {
            ell * numerator[i - 3] / denominator
        }
    });
    let mut j = [[0.; 6]; 6];
    for k in 0..3 {
        j[k][k] = 1.;
    }
    for k in 0..3 {
        let dr = matmul(cayley_derivative([0.; 3], k, ell), r);
        let dn = vex_skew(dr);
        let dd = dr[0][0] + dr[1][1] + dr[2][2];
        for i in 0..3 {
            j[i + 3][k + 3] =
                ell * (dn[i] * denominator - numerator[i] * dd) / (denominator * denominator);
        }
    }
    Ok((mean, congruence(j, contact.covariance)))
}
fn cayley_derivative(u: Vec3, k: usize, ell: f64) -> Mat3 {
    let cross = skew(u);
    let mut e = [0.; 3];
    e[k] = 1.;
    let dk = skew(e);
    let cross2 = matmul(cross, cross);
    let left = matmul(dk, cross);
    let right = matmul(cross, dk);
    let d = 1. + dot(u, u);
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (2. * (dk[i][j] + left[i][j] + right[i][j]) / d
                - 4. * u[k] * (cross[i][j] + cross2[i][j]) / (d * d))
                / ell
        })
    })
}
fn skew(u: Vec3) -> Mat3 {
    [[0., -u[2], u[1]], [u[2], 0., -u[0]], [-u[1], u[0], 0.]]
}
fn vex_skew(r: Mat3) -> Vec3 {
    [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]]
}
fn compose(a: Pose, b: Pose) -> Pose {
    Pose {
        position: a.apply(b.position),
        orientation: quaternion(matmul(rotation(a.orientation), rotation(b.orientation))),
    }
}
fn relative(a: Pose, b: Pose) -> Pose {
    compose(invert_relative_pose(a), b)
}
fn dot6(a: Vec6, b: Vec6) -> f64 {
    (0..6).map(|i| a[i] * b[i]).sum()
}
fn congruence(j: Mat6, cov: Mat6) -> Mat6 {
    let left: Mat6 =
        std::array::from_fn(|i| std::array::from_fn(|k| (0..6).map(|m| j[i][m] * cov[m][k]).sum()));
    let out: Mat6 = std::array::from_fn(|i| std::array::from_fn(|k| dot6(left[i], j[k])));
    std::array::from_fn(|i| std::array::from_fn(|k| 0.5 * (out[i][k] + out[k][i])))
}
fn cholesky(cov: Mat6) -> Result<(Mat6, f64)> {
    ensure!(
        cov.iter().flatten().all(|v| v.is_finite()),
        "Nonfinite covariance"
    );
    let magnitude = cov.iter().flatten().fold(0_f64, |a, b| a.max(b.abs()));
    ensure!(magnitude > 0., "Degenerate covariance");
    for i in 0..6 {
        for j in 0..6 {
            ensure!(
                (cov[i][j] - cov[j][i]).abs() <= 1e-10 * magnitude,
                "Asymmetric covariance"
            );
        }
    }
    let mut l = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let residual =
                0.5 * (cov[i][j] + cov[j][i]) - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            l[i][j] = if i == j {
                ensure!(
                    residual > 0. && residual.is_finite(),
                    "Non-positive covariance"
                );
                residual.sqrt()
            } else {
                residual / l[j][j]
            };
        }
    }
    let logdet = (0..6).map(|i| l[i][i].ln()).sum();
    Ok((l, logdet))
}
fn solve_spd(l: Mat6, rhs: Vec6) -> Vec6 {
    let mut x = [0.; 6];
    for i in 0..6 {
        x[i] = (rhs[i] - (0..i).map(|j| l[i][j] * x[j]).sum::<f64>()) / l[i][i];
    }
    for i in (0..6).rev() {
        x[i] = (x[i] - (i + 1..6).map(|j| l[j][i] * x[j]).sum::<f64>()) / l[i][i];
    }
    x
}
fn log_standard_normal(z: Vec6) -> f64 {
    -3. * (2. * PI).ln() - 0.5 * dot6(z, z)
}
fn log_sum(values: &[f64]) -> f64 {
    let m = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        m
    } else {
        m + values.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
    }
}
fn draw_log_category(rng: &mut StdRng, weights: &[f64]) -> Result<Option<usize>> {
    let mut best = f64::NEG_INFINITY;
    let mut index = None;
    for (i, &w) in weights.iter().enumerate() {
        ensure!(
            w.is_finite() || w == f64::NEG_INFINITY,
            "Invalid category weight"
        );
        if w == f64::NEG_INFINITY {
            continue;
        }
        let u: f64 = Open01.sample(rng);
        let score = w - (-u.ln()).ln();
        if score > best {
            best = score;
            index = Some(i);
        }
    }
    Ok(index)
}
