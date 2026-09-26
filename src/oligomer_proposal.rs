//! Oligomer destination charts for rigid-subset transport.
//!
//! The member/anchor mixture places one carried member in one atlas basin
//! against one pool anchor. An embedded subset usually has several external
//! contacts, and a one-contact destination gives most of them up. This module
//! adds deterministic two-contact components: pairs of single member/anchor
//! labels whose implied subset poses agree are fused by a Gauss-Newton fit of
//! both latent residuals, and the fused Gaussian (inverse summed information)
//! becomes one more chart of the pose of the first listed member, g0.
//!
//! Construction reads only invariant context: the internal offsets
//! u_i = g0^{-1} g_i, the fixed pool anchors, the fixed spectators and the
//! wall. It never reads g0 itself, so the same normalized mixture is rebuilt
//! at both endpoints of a rigid move, up to floating-point roundoff in the
//! recomputed offsets. The posterior-source map and its correction
//! log G_O(old) - log G_O(new) then follow the member-chart argument exactly.
use crate::{
    basin_involution::{BasinPair, BasinStep, FixedBasinInvolution, cross_chart_step},
    docking::{DockingProposal, draw_log_category},
    geometry::{Placed, SphereTree},
    math::*,
    proposal::GaussianComponentParameters,
    spherical::Container,
};
use anyhow::{Context, Result, ensure};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

type Vec6 = [f64; 6];
type Mat6 = [[f64; 6]; 6];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct OligomerConfig {
    /// Proposal mass of the fused two-contact components when any exist.
    pub multi_contact_mass: f64,
    /// Largest summed squared whitened residual of both fused labels.
    pub max_mismatch: f64,
    /// Prefilter on the implied subset positions of two labels.
    #[serde(rename = "pair_distance_A")]
    pub pair_distance_a: f64,
    /// Prefilter on the implied subset orientations of two labels.
    pub pair_angle_degrees: f64,
    /// Gauss-Newton fits per attempt, in prefilter order.
    pub max_candidates: usize,
    /// Hard-core checks of fused centers per attempt, in mismatch order.
    pub max_hard_checks: usize,
    /// Fused components retained per attempt.
    pub max_components: usize,
}
impl Default for OligomerConfig {
    fn default() -> Self {
        Self {
            multi_contact_mass: 0.8,
            max_mismatch: 12.,
            pair_distance_a: 8.,
            pair_angle_degrees: 60.,
            max_candidates: 4096,
            max_hard_checks: 256,
            max_components: 32,
        }
    }
}
impl OligomerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.multi_contact_mass.is_finite() && (0. ..1.).contains(&self.multi_contact_mass),
            "oligomer multi-contact mass must lie in [0,1)"
        );
        for v in [
            self.max_mismatch,
            self.pair_distance_a,
            self.pair_angle_degrees,
        ] {
            ensure!(v.is_finite() && v > 0., "invalid oligomer fusion threshold");
        }
        ensure!(
            self.max_candidates > 0 && self.max_hard_checks > 0 && self.max_components > 0,
            "oligomer component limits must be positive"
        );
        Ok(())
    }
}

/// Single labels are (member * anchors + anchor) * branches + branch; fused
/// labels follow them in construction order.
#[derive(Clone, Debug, Serialize)]
pub struct FusedComponent {
    pub first: usize,
    pub second: usize,
    pub mismatch: f64,
    pub center: Pose,
}

#[derive(Clone, Debug, Serialize)]
pub struct OligomerStep {
    pub handle: Pose,
    pub source: usize,
    pub target: usize,
    pub step: BasinStep,
    pub identity: bool,
    pub full_old_log_density: f64,
    pub full_new_log_density: f64,
    pub label_log_reverse_forward: f64,
    pub expanded_log_reverse_forward: f64,
    pub log_reverse_forward: f64,
}

pub struct OligomerMixture<'p> {
    map: &'p FixedBasinInvolution,
    inverted: Vec<bool>,
    offsets: Vec<Pose>,
    anchors: Vec<Pose>,
    fused_map: Option<FixedBasinInvolution>,
    pub fused: Vec<FusedComponent>,
    log_weights: Vec<f64>,
    singles: usize,
    fit_candidates: usize,
    hard_checks: usize,
    /// Wall-clock seconds: label means and pair search, fits, hard checks.
    pub build_seconds: [f64; 3],
}

fn compose(a: Pose, b: Pose) -> Pose {
    let ra = rotation(a.orientation);
    Pose {
        position: add(a.position, matvec(ra, b.position)),
        orientation: quaternion(matmul(ra, rotation(b.orientation))),
    }
}
fn perturb(center: Pose, delta: Vec6) -> Pose {
    Pose {
        position: std::array::from_fn(|k| center.position[k] + delta[k]),
        orientation: quaternion(matmul(
            cayley([delta[3], delta[4], delta[5]]),
            rotation(center.orientation),
        )),
    }
}
fn rotation_angle(a: Pose, b: Pose) -> f64 {
    let d: f64 = a
        .orientation
        .iter()
        .zip(&b.orientation)
        .map(|(x, y)| x * y)
        .sum();
    2. * d.abs().min(1.).acos()
}
fn cholesky6(a: &Mat6) -> Option<Mat6> {
    let mut l = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let r = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !(r > 0. && r.is_finite()) {
                    return None;
                }
                l[i][i] = r.sqrt();
            } else {
                l[i][j] = r / l[j][j];
            }
        }
    }
    Some(l)
}
fn cholesky_solve(l: &Mat6, b: Vec6) -> Vec6 {
    let mut y = [0.; 6];
    for i in 0..6 {
        y[i] = (b[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
    }
    let mut x = [0.; 6];
    for i in (0..6).rev() {
        x[i] = (y[i] - (i + 1..6).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
    }
    x
}

impl<'p> OligomerMixture<'p> {
    /// `members` in fixed label order (the first is g0), `pool` the fixed
    /// anchors, `spectators` every non-member body (hard-core screen only).
    pub fn build(
        proposal: &'p DockingProposal,
        tree: &SphereTree,
        wall: &Container,
        members: &[Pose],
        spectators: &[Pose],
        pool: &[Pose],
        config: &OligomerConfig,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            !members.is_empty() && !pool.is_empty(),
            "Oligomer charts need members and anchors"
        );
        let (map, inverted, log_branch) = proposal.member_chart_parts();
        let nb = log_branch.len();
        let na = pool.len();
        let g0_inverse = invert_relative_pose(members[0]);
        let offsets: Vec<_> = members.iter().map(|&g| compose(g0_inverse, g)).collect();
        let mut mixture = Self {
            map,
            inverted,
            offsets,
            anchors: pool.to_vec(),
            fused_map: None,
            fused: Vec::new(),
            log_weights: Vec::new(),
            singles: members.len() * na * nb,
            fit_candidates: 0,
            hard_checks: 0,
            build_seconds: [0.; 3],
        };
        let clock = std::time::Instant::now();
        // Implied g0 at every single label's chart mean.
        let means: Vec<Option<Pose>> = (0..mixture.singles)
            .map(|l| {
                let y = map.decode(l % nb, [0.; 6]).ok()?;
                Some(mixture.single_to_g0(l, y))
            })
            .collect();
        let mut order: Vec<_> = (0..mixture.singles)
            .filter(|&l| means[l].is_some())
            .collect();
        order.sort_by(|&a, &b| {
            means[a].unwrap().position[0]
                .total_cmp(&means[b].unwrap().position[0])
                .then(a.cmp(&b))
        });
        let angle = config.pair_angle_degrees.to_radians();
        let mut candidates = Vec::new();
        for (k, &l1) in order.iter().enumerate() {
            let m1 = means[l1].unwrap();
            for &l2 in &order[k + 1..] {
                let m2 = means[l2].unwrap();
                if m2.position[0] - m1.position[0] > config.pair_distance_a {
                    break;
                }
                // Two contacts need distinct (member, anchor) interfaces.
                if l1 / nb == l2 / nb {
                    continue;
                }
                let d = norm(sub(m1.position, m2.position));
                if d < config.pair_distance_a && rotation_angle(m1, m2) < angle {
                    candidates.push((d, l1.min(l2), l1.max(l2)));
                }
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        candidates.truncate(config.max_candidates);
        mixture.fit_candidates = candidates.len();
        mixture.build_seconds[0] = clock.elapsed().as_secs_f64();
        let mut fits = Vec::new();
        for &(_, l1, l2) in &candidates {
            if let Some((center, information, mismatch)) =
                mixture.fuse(l1, l2, means[l1].unwrap(), config.max_mismatch)
                && mismatch <= config.max_mismatch
            {
                fits.push((mismatch, l1, l2, center, information));
            }
        }
        fits.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
        mixture.build_seconds[1] = clock.elapsed().as_secs_f64() - mixture.build_seconds[0];
        let fixed: Vec<_> = spectators.iter().map(|&p| Placed::new(p)).collect();
        let scale = proposal.angular_length();
        let mut parameters = Vec::new();
        let mut fused_logs = Vec::new();
        for (mismatch, l1, l2, center, information) in fits {
            if mixture.fused.len() >= config.max_components
                || mixture.hard_checks >= config.max_hard_checks
            {
                break;
            }
            mixture.hard_checks += 1;
            let bodies: Vec<_> = mixture
                .offsets
                .iter()
                .map(|&u| compose(center, u))
                .collect();
            let valid = bodies.iter().all(|&p| {
                p.validate().is_ok() && wall.contains(p) && {
                    let placed = Placed::new(p);
                    !fixed.iter().any(|s| tree.overlaps(&placed, s))
                }
            });
            if !valid {
                continue;
            }
            // Chart coordinates are (t - c, L * Cayley), so scale rotations.
            let Some(lower) = cholesky6(&information) else {
                continue;
            };
            let mut covariance = [[0.; 6]; 6];
            for j in 0..6 {
                let mut e = [0.; 6];
                e[j] = 1.;
                let column = cholesky_solve(&lower, e);
                let sj = if j < 3 { 1. } else { scale };
                for (i, row) in covariance.iter_mut().enumerate() {
                    let si = if i < 3 { 1. } else { scale };
                    row[j] = si * column[i] * sj;
                }
            }
            let covariance: Mat6 = std::array::from_fn(|i| {
                std::array::from_fn(|j| 0.5 * (covariance[i][j] + covariance[j][i]))
            });
            if cholesky6(&covariance).is_none() {
                continue;
            }
            parameters.push(GaussianComponentParameters {
                anchor_position: center.position,
                anchor_rotation: rotation(center.orientation),
                mean: [0.; 6],
                covariance,
                weight: 1.,
            });
            fused_logs.push(log_branch[l1 % nb] + log_branch[l2 % nb] - 0.5 * mismatch);
            mixture.fused.push(FusedComponent {
                first: l1,
                second: l2,
                mismatch,
                center,
            });
        }
        mixture.build_seconds[2] =
            clock.elapsed().as_secs_f64() - mixture.build_seconds[0] - mixture.build_seconds[1];
        // Mismatch order only spends the check budget. Index components by
        // label pair so near-tied mismatches cannot permute chart indices
        // between the two endpoints' reconstructions.
        let mut order: Vec<_> = (0..mixture.fused.len()).collect();
        order.sort_by_key(|&k| (mixture.fused[k].first, mixture.fused[k].second));
        let parameters: Vec<_> = order.iter().map(|&k| parameters[k].clone()).collect();
        let fused_logs: Vec<_> = order.iter().map(|&k| fused_logs[k]).collect();
        mixture.fused = order.iter().map(|&k| mixture.fused[k].clone()).collect();
        let mass = if parameters.is_empty() {
            0.
        } else {
            mixture.fused_map = Some(FixedBasinInvolution::new(
                parameters,
                scale,
                proposal.correlation(),
                vec![BasinPair {
                    first: 0,
                    second: 0,
                    weight: 1.,
                }],
            )?);
            config.multi_contact_mass
        };
        let single_offset = (1. - mass).ln() - ((members.len() * na) as f64).ln();
        mixture.log_weights = (0..mixture.singles)
            .map(|l| single_offset + log_branch[l % nb])
            .collect();
        let fused_total = log_sum(&fused_logs);
        mixture
            .log_weights
            .extend(fused_logs.iter().map(|w| mass.ln() + w - fused_total));
        Ok(mixture)
    }

    pub fn fit_candidates(&self) -> usize {
        self.fit_candidates
    }
    pub fn hard_checks(&self) -> usize {
        self.hard_checks
    }
    pub fn label_count(&self) -> usize {
        self.log_weights.len()
    }
    pub fn log_weights(&self) -> &[f64] {
        &self.log_weights
    }

    fn nb(&self) -> usize {
        self.inverted.len()
    }
    fn to_single(&self, label: usize, g0: Pose) -> Pose {
        let nb = self.nb();
        let (member, anchor, branch) = (
            label / (self.anchors.len() * nb),
            label / nb % self.anchors.len(),
            label % nb,
        );
        let relative = compose(
            invert_relative_pose(self.anchors[anchor]),
            compose(g0, self.offsets[member]),
        );
        if self.inverted[branch] {
            invert_relative_pose(relative)
        } else {
            relative
        }
    }
    fn single_to_g0(&self, label: usize, y: Pose) -> Pose {
        let nb = self.nb();
        let (member, anchor, branch) = (
            label / (self.anchors.len() * nb),
            label / nb % self.anchors.len(),
            label % nb,
        );
        let relative = if self.inverted[branch] {
            invert_relative_pose(y)
        } else {
            y
        };
        compose(
            compose(self.anchors[anchor], relative),
            invert_relative_pose(self.offsets[member]),
        )
    }
    fn chart(&self, label: usize) -> (&FixedBasinInvolution, usize) {
        if label < self.singles {
            (self.map, label % self.nb())
        } else {
            (
                self.fused_map.as_ref().expect("fused label without charts"),
                label - self.singles,
            )
        }
    }
    fn to_chart(&self, label: usize, g0: Pose) -> Pose {
        if label < self.singles {
            self.to_single(label, g0)
        } else {
            g0
        }
    }
    fn chart_to_g0(&self, label: usize, y: Pose) -> Pose {
        if label < self.singles {
            self.single_to_g0(label, y)
        } else {
            y
        }
    }

    /// Minimize both labels' squared whitened residuals over g0, starting at
    /// the first label's implied mean. Returns the center, the information
    /// matrix in (translation, Cayley) coordinates and the residual.
    /// Fits still above ten times `limit` after two steps are abandoned.
    fn fuse(&self, l1: usize, l2: usize, start: Pose, limit: f64) -> Option<(Pose, Mat6, f64)> {
        let nb = self.nb();
        let residual = |g: Pose| -> Option<[f64; 12]> {
            let z1 = self.map.encode(l1 % nb, self.to_single(l1, g)).ok()?;
            let z2 = self.map.encode(l2 % nb, self.to_single(l2, g)).ok()?;
            Some(std::array::from_fn(
                |k| if k < 6 { z1[k] } else { z2[k - 6] },
            ))
        };
        let chi2 = |r: &[f64; 12]| r.iter().map(|x| x * x).sum::<f64>();
        let jacobian = |c: Pose| -> Option<[[f64; 6]; 12]> {
            let mut j = [[0.; 6]; 12];
            for k in 0..6 {
                let h = if k < 3 { 1e-5 } else { 1e-6 };
                let mut d = [0.; 6];
                d[k] = h;
                let plus = residual(perturb(c, d))?;
                d[k] = -h;
                let minus = residual(perturb(c, d))?;
                for r in 0..12 {
                    j[r][k] = (plus[r] - minus[r]) / (2. * h);
                }
            }
            Some(j)
        };
        let normal = |j: &[[f64; 6]; 12]| -> Mat6 {
            std::array::from_fn(|a| {
                std::array::from_fn(|b| (0..12).map(|r| j[r][a] * j[r][b]).sum())
            })
        };
        let mut center = start;
        let mut r = residual(center)?;
        let mut value = chi2(&r);
        for iteration in 0..16 {
            if iteration >= 2 && value > 10. * limit {
                return None;
            }
            let j = jacobian(center)?;
            let lower = cholesky6(&normal(&j))?;
            let gradient: Vec6 =
                std::array::from_fn(|a| -(0..12).map(|k| j[k][a] * r[k]).sum::<f64>());
            let mut delta = cholesky_solve(&lower, gradient);
            let mut improved = false;
            for _ in 0..12 {
                let trial = perturb(center, delta);
                if let Some(tr) = residual(trial)
                    && chi2(&tr) <= value
                {
                    center = trial;
                    r = tr;
                    value = chi2(&tr);
                    improved = true;
                    break;
                }
                delta = delta.map(|x| 0.5 * x);
            }
            if !improved || delta.iter().map(|x| x * x).sum::<f64>() < 1e-24 {
                break;
            }
        }
        let j = jacobian(center)?;
        Some((center, normal(&j), value))
    }

    /// Per-label log terms w_l q_l(g0); their log-sum is log G_O(g0).
    pub fn label_logs(&self, g0: Pose) -> Vec<f64> {
        self.log_weights
            .iter()
            .enumerate()
            .map(|(l, &w)| {
                if w == f64::NEG_INFINITY {
                    return w;
                }
                let (map, chart) = self.chart(l);
                w + map.log_density(chart, self.to_chart(l, g0))
            })
            .collect()
    }
    pub fn log_density(&self, members: &[Pose]) -> f64 {
        log_sum(&self.label_logs(members[0]))
    }

    /// Posterior source label, independent destination label and noise.
    pub fn propose(
        &self,
        rng: &mut StdRng,
        members: &[Pose],
        handle: usize,
    ) -> Result<(Option<Pose>, Value)> {
        let old = self.label_logs(members[0]);
        let Some(source) = draw_log_category(rng, &old)? else {
            return Ok((
                None,
                json!({"branch":"involution","charts":"oligomer",
                    "null_reason":"No finite Gaussian source density"}),
            ));
        };
        let target =
            draw_log_category(rng, &self.log_weights)?.context("No destination component")?;
        let noise: Vec6 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let context = json!({"fused_components":self.fused.len(),
            "fit_candidates":self.fit_candidates,"hard_checks":self.hard_checks,
            "build_seconds":self.build_seconds,
            "source_label":self.describe(source),"target_label":self.describe(target)});
        match self.apply(members, handle, source, target, noise) {
            Ok(step) => {
                let mut info = serde_json::to_value(&step)?;
                info["branch"] = json!("involution");
                info["charts"] = json!("oligomer");
                info["source_law"] = json!("posterior");
                info["oligomer"] = context;
                Ok((Some(step.handle), info))
            }
            Err(error) => Ok((
                None,
                json!({"branch":"involution","charts":"oligomer","oligomer":context,
                    "noise":noise,"null_reason":error.to_string()}),
            )),
        }
    }

    pub fn describe(&self, label: usize) -> Value {
        if label < self.singles {
            let nb = self.nb();
            json!({"kind":"single","member":label / (self.anchors.len() * nb),
                "anchor":label / nb % self.anchors.len(),"branch":label % nb})
        } else {
            let f = &self.fused[label - self.singles];
            json!({"kind":"fused","index":label - self.singles,"first":self.describe(f.first),
                "second":self.describe(f.second),"mismatch":f.mismatch})
        }
    }

    /// Deterministic map for given labels and noise. The inverse applies the
    /// swapped labels and `step.inverse_trace.noise` at the carried endpoint.
    pub fn apply(
        &self,
        members: &[Pose],
        handle: usize,
        source: usize,
        target: usize,
        noise: Vec6,
    ) -> Result<OligomerStep> {
        ensure!(
            handle < members.len()
                && members.len() == self.offsets.len()
                && source < self.label_count()
                && target < self.label_count(),
            "Invalid oligomer label or members"
        );
        let old = self.label_logs(members[0]);
        let full_old = log_sum(&old);
        ensure!(full_old.is_finite(), "Nonfinite oligomer density");
        let step = cross_chart_step(
            self.chart(source),
            self.chart(target),
            self.to_chart(source, members[0]),
            noise,
        )?;
        if self.map.correlation() == 1. && source == target {
            return Ok(OligomerStep {
                handle: members[handle],
                source,
                target,
                step,
                identity: true,
                full_old_log_density: full_old,
                full_new_log_density: full_old,
                label_log_reverse_forward: 0.,
                expanded_log_reverse_forward: 0.,
                log_reverse_forward: 0.,
            });
        }
        let moved = self.chart_to_g0(target, step.pose);
        moved.validate()?;
        let reference = members[0];
        let delta = matmul(
            rotation(moved.orientation),
            transpose(rotation(reference.orientation)),
        );
        let carry = |p: Pose| Pose {
            position: add(
                moved.position,
                matvec(delta, sub(p.position, reference.position)),
            ),
            orientation: quaternion(matmul(delta, rotation(p.orientation))),
        };
        let proposed = carry(members[handle]);
        proposed.validate()?;
        let new = self.label_logs(carry(members[0]));
        let full_new = log_sum(&new);
        ensure!(full_new.is_finite(), "Nonfinite oligomer density");
        let labels = (new[target] - full_new) - (old[source] - full_old) + self.log_weights[source]
            - self.log_weights[target];
        Ok(OligomerStep {
            handle: proposed,
            source,
            target,
            identity: false,
            full_old_log_density: full_old,
            full_new_log_density: full_new,
            label_log_reverse_forward: labels,
            expanded_log_reverse_forward: step.log_correction + labels,
            log_reverse_forward: full_old - full_new,
            step,
        })
    }
}

fn log_sum(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return maximum;
    }
    maximum + values.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln()
}
