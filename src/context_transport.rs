//! Transport between two context-defined charts, with an ORIGINAL-atlas selector.
//!
//! This utility does not optimize charts, select labels, query geometry or run MC.
//! Each supplied map contains one ordinary physical-pose chart. Its parameters
//! must depend only on the unchanged outside context and its global label, never
//! on the moving pose, auxiliary noise, or source/destination role. In particular,
//! an inverse-origin child is an ordinary chart, not another reciprocal wrapper.
//! Exact reciprocal branches belong in the ORIGINAL weighted selector scores.
//!
//! If a_i(x)=p_i g_i^0(x)/G^0(x), forward labels have law a_i(x)p_j.
//! The map correction must therefore be augmented by
//! log[a_j(y)p_i/(a_i(x)p_j)]. The old posterior shortcut log G^0(x)/G^0(y)
//! is only valid when the transport charts generate the selector densities.

use anyhow::{Result, ensure};
use serde::Serialize;

use crate::{
    basin_involution::{BasinStep, BasinTrace, FixedBasinInvolution, cross_chart_step},
    math::Pose,
};

/// A global original-atlas label paired with its single ordinary adjusted chart.
/// The integer is NOT a local chart index: every supplied map uses chart zero.
pub type SelectedChart<'a> = (&'a FixedBasinInvolution, usize);

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReverseZeroReason {
    EmptyOriginalSupport,
    SelectedOriginalBranchZero,
}

/// Zero reverse support is an ordinary rejection, not a numerical failure.
/// Serialization retains that distinction without encoding infinity as null.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum OriginalLabelCorrection {
    Finite {
        log_forward_label_probability: f64,
        log_reverse_label_probability: f64,
        log_selection_reverse_forward: f64,
        log_map_correction: f64,
        log_reverse_forward: f64,
    },
    ReverseZero {
        reason: ReverseZeroReason,
        log_forward_label_probability: f64,
        log_map_correction: f64,
    },
}

impl OriginalLabelCorrection {
    /// The caller must handle `ReverseZero` as a rejection before any downstream
    /// finite-only physical-gate API. This accessor returns -infinity for it.
    pub fn log_reverse_forward(&self) -> f64 {
        match self {
            Self::Finite {
                log_reverse_forward,
                ..
            } => *log_reverse_forward,
            Self::ReverseZero { .. } => f64::NEG_INFINITY,
        }
    }
}

/// The complete map result, with GLOBAL labels in both auxiliary traces.
/// Private fields prevent replacing the stored map correction independently.
#[derive(Clone, Debug, Serialize)]
pub struct SelectedChartStep {
    forward_trace: BasinTrace,
    map_step: BasinStep,
}

impl SelectedChartStep {
    pub fn forward_trace(&self) -> &BasinTrace {
        &self.forward_trace
    }

    pub fn map_step(&self) -> &BasinStep {
        &self.map_step
    }

    /// Complete selection plus map correction. Inputs cover the SAME complete
    /// ordered original label inventory at both poses; no successful-fit filter.
    /// `weighted_*[b]` is log(p_b g_b^0(pose)), not an unweighted chart density.
    /// Priors are strictly positive: all log priors must be finite. Common log
    /// offsets are allowed because priors and responsibilities are normalized.
    /// Weighted scores may be -infinity only for genuine zero chart support.
    /// NaN, +infinity, invalid lengths and impossible forward selections error.
    /// The caller is responsible for binding these scores to old/proposed poses
    /// and to the exact ORIGINAL prepared factors, including reciprocal branches.
    pub fn correction(
        &self,
        log_priors: &[f64],
        weighted_old: &[f64],
        weighted_new: &[f64],
    ) -> Result<OriginalLabelCorrection> {
        let i = self.forward_trace.source;
        let j = self.forward_trace.target;
        ensure!(
            !log_priors.is_empty()
                && weighted_old.len() == log_priors.len()
                && weighted_new.len() == log_priors.len()
                && i < log_priors.len()
                && j < log_priors.len(),
            "Changed original label inventory or invalid global label"
        );
        ensure!(
            log_priors.iter().all(|v| v.is_finite()),
            "Original branch priors must have finite logs (strictly positive support)"
        );
        let prior_total = log_normalizer(log_priors)?.expect("Nonempty finite priors");
        let old_total = log_normalizer(weighted_old)?;
        let new_total = log_normalizer(weighted_new)?;
        ensure!(
            old_total.is_some() && weighted_old[i].is_finite(),
            "Selected forward source has zero original probability"
        );
        let forward = old_total.unwrap().log_probability(weighted_old[i])?
            + prior_total.log_probability(log_priors[j])?;
        ensure!(
            forward.is_finite(),
            "Unrepresentable forward label probability"
        );
        let zero_reason = if new_total.is_none() {
            Some(ReverseZeroReason::EmptyOriginalSupport)
        } else if weighted_new[j] == f64::NEG_INFINITY {
            Some(ReverseZeroReason::SelectedOriginalBranchZero)
        } else {
            None
        };
        if let Some(reason) = zero_reason {
            return Ok(OriginalLabelCorrection::ReverseZero {
                reason,
                log_forward_label_probability: forward,
                log_map_correction: self.map_step.log_correction,
            });
        }
        let reverse = new_total.unwrap().log_probability(weighted_new[j])?
            + prior_total.log_probability(log_priors[i])?;
        let labels = reverse - forward;
        let correction = self.map_step.log_correction + labels;
        ensure!(
            reverse.is_finite() && labels.is_finite() && correction.is_finite(),
            "Unrepresentable reverse label probability or correction"
        );
        Ok(OriginalLabelCorrection::Finite {
            log_forward_label_probability: forward,
            log_reverse_label_probability: reverse,
            log_selection_reverse_forward: labels,
            log_map_correction: self.map_step.log_correction,
            log_reverse_forward: correction,
        })
    }
}

/// Apply the existing augmented Gaussian involution using only two prepared
/// charts. The reverse call swaps these global-label/chart pairs and supplies
/// `map_step().inverse_trace.noise`. At equal global labels, reuse the same
/// borrowed chart object, guarding against accidental role-dependent refits.
/// Different labels may refer to the same chart. No optimizer Jacobian appears:
/// chart construction conditions on the unchanged outside context.
pub fn apply_selected_charts(
    source: SelectedChart<'_>,
    target: SelectedChart<'_>,
    old: Pose,
    noise: [f64; 6],
) -> Result<SelectedChartStep> {
    ensure!(
        source.0.chart_count() == 1 && target.0.chart_count() == 1,
        "Selected context maps must each contain exactly one ordinary chart"
    );
    ensure!(
        source.1 != target.1 || std::ptr::eq(source.0, target.0),
        "The same global label must reuse the same context chart object"
    );
    let mut map_step = cross_chart_step((source.0, 0), (target.0, 0), old, noise)?;
    map_step.inverse_trace.source = target.1;
    map_step.inverse_trace.target = source.1;
    Ok(SelectedChartStep {
        forward_trace: BasinTrace {
            source: source.1,
            target: target.1,
            noise,
        },
        map_step,
    })
}

#[derive(Clone, Copy)]
struct RelativeLogNormalizer {
    maximum: f64,
    log_scaled_sum: f64,
}

impl RelativeLogNormalizer {
    fn log_probability(self, value: f64) -> Result<f64> {
        // Form the difference first: a rounded maximum + log_scaled_sum loses
        // normalization at common offsets of magnitude 1e300, even for equal logs.
        let result = (value - self.maximum) - self.log_scaled_sum;
        ensure!(
            result.is_finite(),
            "Unrepresentable selected log probability"
        );
        Ok(result)
    }
}

fn log_normalizer(values: &[f64]) -> Result<Option<RelativeLogNormalizer>> {
    ensure!(
        values
            .iter()
            .all(|v| v.is_finite() || *v == f64::NEG_INFINITY),
        "Original selector scores contain NaN or positive infinity"
    );
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return Ok(None);
    }
    let log_scaled_sum = values.iter().map(|v| (v - maximum).exp()).sum::<f64>().ln();
    ensure!(
        log_scaled_sum.is_finite(),
        "Unrepresentable original selector normalizer"
    );
    Ok(Some(RelativeLogNormalizer {
        maximum,
        log_scaled_sum,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        basin_involution::BasinPair,
        math::{cayley, invert_relative_pose, quaternion, rotation},
        proposal::GaussianComponentParameters,
    };

    fn parameters(shift: f64) -> GaussianComponentParameters {
        // Noncommuting rotations and translation/rotation covariance coupling.
        let lower: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i == j {
                    0.7 + 0.09 * i as f64 + shift.abs() * 0.04
                } else if i > j {
                    0.025 * (i + j + 1) as f64
                } else {
                    0.
                }
            })
        });
        GaussianComponentParameters {
            anchor_position: [shift, 0.2 * shift, -0.3],
            anchor_rotation: cayley([0.13, -0.07 * shift, 0.19]),
            mean: [0.03, -0.04, 0.01, 0.02 * shift, -0.03, 0.01],
            covariance: std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..6).map(|k| lower[i][k] * lower[j][k]).sum())
            }),
            weight: 1.,
        }
    }

    fn chart(shift: f64, rho: f64) -> FixedBasinInvolution {
        FixedBasinInvolution::new(
            vec![parameters(shift)],
            1.8,
            rho,
            vec![BasinPair {
                first: 0,
                second: 0,
                weight: 1.,
            }],
        )
        .unwrap()
    }

    fn pose() -> Pose {
        Pose {
            position: [0.8, -0.2, 0.5],
            orientation: quaternion(cayley([0.1, -0.2, 0.3])),
        }
    }

    const NOISE: [f64; 6] = [0.2, -0.7, 0.3, 0.8, -0.1, 0.4];
    const PRIORS: [f64; 3] = [-1.7, -0.9, -1.2];

    fn original_logs(pose: Pose) -> Vec<f64> {
        [-0.7, 0.1, 1.4]
            .into_iter()
            .enumerate()
            .map(|(i, shift)| {
                let p = if i == 0 {
                    invert_relative_pose(pose)
                } else {
                    pose
                };
                PRIORS[i] + chart(shift, 0.65).checked_log_density(0, p).unwrap()
            })
            .collect()
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 3e-11, "{a} != {b}");
    }

    fn log_total(values: &[f64]) -> Result<Option<f64>> {
        Ok(log_normalizer(values)?.map(|v| v.maximum + v.log_scaled_sum))
    }

    fn pose_close(a: Pose, b: Pose) {
        for (x, y) in a.position.into_iter().zip(b.position) {
            close(x, y);
        }
        for (x, y) in rotation(a.orientation)
            .into_iter()
            .flatten()
            .zip(rotation(b.orientation).into_iter().flatten())
        {
            close(x, y);
        }
    }

    #[test]
    fn reverse_recovers_pose_noise_and_global_labels() {
        for rho in [-1., 0., 0.65, 1.] {
            let a = chart(-0.2, rho);
            let b = chart(2.1, rho);
            let step = apply_selected_charts((&a, 1), (&b, 2), pose(), NOISE).unwrap();
            let inverse = &step.map_step().inverse_trace;
            assert_eq!((inverse.source, inverse.target), (2, 1));
            let back = apply_selected_charts(
                (&b, inverse.source),
                (&a, inverse.target),
                step.map_step().pose,
                inverse.noise,
            )
            .unwrap();
            pose_close(back.map_step().pose, pose());
            for (x, y) in back.map_step().inverse_trace.noise.into_iter().zip(NOISE) {
                close(x, y);
            }
            close(
                step.map_step().log_correction,
                -back.map_step().log_correction,
            );
            let old = original_logs(pose());
            let new = original_logs(step.map_step().pose);
            let f = step.correction(&PRIORS, &old, &new).unwrap();
            let r = back.correction(&PRIORS, &new, &old).unwrap();
            close(f.log_reverse_forward(), -r.log_reverse_forward());
        }
    }

    #[test]
    fn unchanged_charts_reduce_to_original_posterior_shortcut() {
        let a = chart(0.1, 0.65);
        let b = chart(1.4, 0.65);
        let step = apply_selected_charts((&a, 1), (&b, 2), pose(), NOISE).unwrap();
        let old = original_logs(pose());
        let new = original_logs(step.map_step().pose);
        close(
            step.correction(&PRIORS, &old, &new)
                .unwrap()
                .log_reverse_forward(),
            log_total(&old).unwrap().unwrap() - log_total(&new).unwrap().unwrap(),
        );
    }

    #[test]
    fn unchanged_self_chart_at_unit_correlation_is_identity() {
        let a = chart(0.1, 1.);
        let step = apply_selected_charts((&a, 2), (&a, 2), pose(), NOISE).unwrap();
        pose_close(step.map_step().pose, pose());
        let old = original_logs(pose());
        close(
            step.correction(&PRIORS, &old, &original_logs(step.map_step().pose))
                .unwrap()
                .log_reverse_forward(),
            0.,
        );
        for (x, y) in step.map_step().inverse_trace.noise.into_iter().zip(NOISE) {
            close(x, -y);
        }
    }

    #[test]
    fn reciprocal_original_selection_does_not_invert_ordinary_child_again() {
        let a = chart(-0.4, 0.65);
        let b = chart(1.7, 0.65);
        let step = apply_selected_charts((&a, 0), (&b, 2), pose(), NOISE).unwrap();
        let expected = cross_chart_step((&a, 0), (&b, 0), pose(), NOISE).unwrap();
        pose_close(step.map_step().pose, expected.pose);
        let back = apply_selected_charts(
            (&b, 2),
            (&a, 0),
            step.map_step().pose,
            step.map_step().inverse_trace.noise,
        )
        .unwrap();
        let old = original_logs(pose());
        let new = original_logs(step.map_step().pose);
        close(
            step.correction(&PRIORS, &old, &new)
                .unwrap()
                .log_reverse_forward(),
            -back
                .correction(&PRIORS, &new, &old)
                .unwrap()
                .log_reverse_forward(),
        );
    }

    #[test]
    fn full_correction_satisfies_pointwise_flow_and_old_shortcut_fails() {
        let a = chart(-0.2, 0.65);
        let b = chart(2.1, 0.65);
        let step = apply_selected_charts((&a, 1), (&b, 2), pose(), NOISE).unwrap();
        let old = original_logs(pose());
        let new = original_logs(step.map_step().pose);
        let OriginalLabelCorrection::Finite {
            log_forward_label_probability: f,
            log_reverse_label_probability: r,
            log_reverse_forward: c,
            ..
        } = step.correction(&PRIORS, &old, &new).unwrap()
        else {
            panic!("Finite fixture");
        };
        // Physical target is the normalized ORIGINAL mixture itself.
        let gx = log_total(&old).unwrap().unwrap();
        let gy = log_total(&new).unwrap().unwrap();
        let log_phi = |v: [f64; 6]| -0.5 * v.iter().map(|x| x * x).sum::<f64>();
        let forward = gx + f + log_phi(NOISE);
        let reverse = gy
            + r
            + log_phi(step.map_step().inverse_trace.noise)
            + step.map_step().log_extended_jacobian;
        let ratio = gy - gx + c;
        close(reverse - forward, ratio);
        close(forward + ratio.min(0.), reverse + (-ratio).min(0.));
        // Wrong old shortcut cancels the physical density ratio and accepts all.
        let wrong = gy - gx + (gx - gy);
        close(wrong, 0.);
        assert!((forward + wrong.min(0.) - reverse - (-wrong).min(0.)).abs() > 1e-3);
    }

    #[test]
    fn zero_reverse_support_is_explicit_rejection_without_nan() {
        let a = chart(0., 0.65);
        let b = chart(1., 0.65);
        let step = apply_selected_charts((&a, 1), (&b, 2), pose(), NOISE).unwrap();
        let old = original_logs(pose());
        for (new, reason) in [
            (
                vec![f64::NEG_INFINITY; 3],
                ReverseZeroReason::EmptyOriginalSupport,
            ),
            (
                vec![-1., -2., f64::NEG_INFINITY],
                ReverseZeroReason::SelectedOriginalBranchZero,
            ),
        ] {
            let result = step.correction(&PRIORS, &old, &new).unwrap();
            assert_eq!(result.log_reverse_forward(), f64::NEG_INFINITY);
            assert!(
                matches!(result,OriginalLabelCorrection::ReverseZero {reason:r,..} if r==reason)
            );
            let value = serde_json::to_value(&result).unwrap();
            assert_eq!(value["status"], "reverse_zero");
            assert!(!value.as_object().unwrap().values().any(|v| v.is_null()));
        }
    }

    #[test]
    fn invalid_priors_scores_inventory_and_forward_support_error() {
        let a = chart(0., 0.65);
        let b = chart(1., 0.65);
        let step = apply_selected_charts((&a, 1), (&b, 2), pose(), NOISE).unwrap();
        let old = original_logs(pose());
        let new = original_logs(step.map_step().pose);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut priors = PRIORS;
            priors[0] = invalid;
            assert!(step.correction(&priors, &old, &new).is_err());
        }
        for invalid in [f64::NAN, f64::INFINITY] {
            let mut bad = old.clone();
            bad[0] = invalid;
            assert!(step.correction(&PRIORS, &bad, &new).is_err());
            assert!(step.correction(&PRIORS, &old, &bad).is_err());
        }
        let mut bad = old.clone();
        bad[1] = f64::NEG_INFINITY;
        assert!(step.correction(&PRIORS, &bad, &new).is_err());
        assert!(
            step.correction(&PRIORS, &[f64::NEG_INFINITY; 3], &new)
                .is_err()
        );
        assert!(step.correction(&PRIORS, &old[..2], &new).is_err());
        assert!(step.correction(&[], &[], &[]).is_err());
        let invalid_label = apply_selected_charts((&a, 5), (&b, 2), pose(), NOISE).unwrap();
        assert!(invalid_label.correction(&PRIORS, &old, &new).is_err());
    }

    #[test]
    fn maps_require_one_chart_consistent_roles_and_valid_cross_map_inputs() {
        let a = chart(0., 0.65);
        let b = chart(1., 0.65);
        assert!(apply_selected_charts((&a, 1), (&b, 1), pose(), NOISE).is_err());
        let multi = FixedBasinInvolution::new(
            vec![parameters(0.), parameters(1.)],
            1.8,
            0.65,
            vec![BasinPair {
                first: 0,
                second: 1,
                weight: 1.,
            }],
        )
        .unwrap();
        assert!(apply_selected_charts((&multi, 1), (&b, 2), pose(), NOISE).is_err());
        assert!(apply_selected_charts((&a, 1), (&chart(1., 0.3), 2), pose(), NOISE).is_err());
        assert!(apply_selected_charts((&a, 1), (&b, 2), pose(), [f64::NAN; 6]).is_err());
        let mut bad = pose();
        bad.orientation = [0.; 4];
        assert!(apply_selected_charts((&a, 1), (&b, 2), bad, NOISE).is_err());
    }

    #[test]
    fn log_normalization_offsets_leave_the_actual_label_law_unchanged() {
        let a = chart(0., 0.65);
        let b = chart(1., 0.65);
        let step = apply_selected_charts((&a, 1), (&b, 2), pose(), NOISE).unwrap();
        let old = original_logs(pose());
        let new = original_logs(step.map_step().pose);
        let expected = step
            .correction(&PRIORS, &old, &new)
            .unwrap()
            .log_reverse_forward();
        let p = PRIORS.map(|v| v + 43.);
        let x: Vec<_> = old.iter().map(|v| v - 17.).collect();
        let y: Vec<_> = new.iter().map(|v| v + 93.).collect();
        close(
            step.correction(&p, &x, &y).unwrap().log_reverse_forward(),
            expected,
        );
    }

    #[test]
    fn extreme_common_offsets_keep_equal_label_probabilities_normalized() {
        let a = chart(0., 0.65);
        let b = chart(1., 0.65);
        let step = apply_selected_charts((&a, 1), (&b, 2), pose(), NOISE).unwrap();
        for priors in [[0.; 3], [1e300; 3], [-1e300; 3]] {
            let result = step.correction(&priors, &[-1e300; 3], &[0.; 3]).unwrap();
            let OriginalLabelCorrection::Finite {
                log_forward_label_probability,
                log_reverse_label_probability,
                log_selection_reverse_forward,
                log_reverse_forward,
                ..
            } = result
            else {
                panic!("Equal positive labels must retain support");
            };
            close(log_forward_label_probability, -2. * 3_f64.ln());
            close(log_reverse_label_probability, -2. * 3_f64.ln());
            close(log_selection_reverse_forward, 0.);
            close(log_reverse_forward, step.map_step().log_correction);
        }
    }
}
