//! Bounded, trace-preserving singleton bath gates and two-singleton paths.
//!
//! Extracted from the validated immutable replay example. The low-level leg,
//! path ordering, Poisson draws, point loop and aggregation retain that exact
//! arithmetic and RNG order. Limits only produce fatal errors; never interpret
//! a partial cloud as an ordinary rejection or retry it with a new cloud.
//!
//! Physical callers must validate old/final hard geometry before drawing the
//! bath and compose the returned log weight with their full proposal/auxiliary
//! correction in ONE MH decision. Path intermediates may overlap or cross the
//! protein wall. The ideal bath remains unbounded and wall permeable.
use crate::{
    depletion::{GateOptions, GateResult},
    geometry::SphereTree,
    math::Pose,
    rigid_subset::RigidSubset,
    simulation::cpu_seconds,
    singleton_path::{SingletonOrder, SingletonPath, SingletonPathResult},
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub raw_per_leg: u64,
    pub raw_per_outer: u64,
    pub raw_campaign: u64,
    pub retained_per_leg: u64,
    pub retained_per_outer: u64,
    pub retained_campaign: u64,
    pub cpu_seconds: f64,
}
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct LegProgress {
    pub gate: GateResult,
    /// Planned Poisson count is in gate.raw_points, even when its cap is exceeded.
    pub processed_points: u64,
    pub complete: bool,
}
#[derive(Debug, Serialize)]
pub struct PathFailure {
    pub reason: String,
    pub order: SingletonOrder,
    pub ordered_members: [usize; 2],
    pub intermediate_selected: [Pose; 2],
    pub completed_legs: Vec<GateResult>,
    pub failed_leg: usize,
    pub failed_progress: LegProgress,
}
pub struct Budget {
    pub limits: Limits,
    pub started: f64,
    pub raw: u64,
    pub retained: u64,
}
impl Budget {
    pub fn check_cpu(&self) -> Result<()> {
        ensure!(
            cpu_seconds() - self.started <= self.limits.cpu_seconds,
            "fatal campaign CPU budget exceeded"
        );
        Ok(())
    }
}

/// Same draw order and arithmetic as RigidSubset::sample_envelope. Only fatal
/// resource guards are added. A partial cloud never supplies an MH factor.
///
/// Low-level API: supply a fresh default progress record, validated finite
/// lambda>0, z>=0, valid gate options, and caps already including the remaining
/// outer/campaign budgets. This function does not update Budget counters;
/// prefer bounded_singleton for a complete single-body outer call.
pub fn bounded_leg(
    gate: &RigidSubset<'_>,
    rng: &mut StdRng,
    lambda: f64,
    z: f64,
    opts: GateOptions,
    raw_cap: u64,
    retained_cap: u64,
    budget: &Budget,
    progress: &mut LegProgress,
) -> Result<GateResult> {
    budget.check_cpu()?;
    if z == 0. {
        progress.complete = true;
        return Ok(GateResult::default());
    }
    let envelope = gate.envelope(opts)?;
    progress.gate = GateResult {
        envelope_volume: envelope.volume,
        retained_cells: envelope.cells.len(),
        created_cells: envelope.created,
        ..Default::default()
    };
    budget.check_cpu()?;
    if envelope.volume == 0. {
        progress.complete = true;
        return Ok(progress.gate);
    }
    let mean = (lambda + z) * envelope.volume;
    ensure!(mean.is_finite() && mean < 9e15, "unsupported Poisson mean");
    let number = Poisson::<f64>::new(mean)?.sample(rng) as u64;
    progress.gate.raw_points = number;
    ensure!(number <= raw_cap, "fatal planned raw-point budget exceeded");
    for index in 0..number {
        if index % 1024 == 0 {
            budget.check_cpu()?;
        }
        let target = rng.random::<f64>() * envelope.volume;
        let k = envelope
            .cumulative
            .partition_point(|&v| v <= target)
            .min(envelope.cells.len() - 1);
        let cell = envelope.cells[k];
        let point =
            std::array::from_fn(|j| cell.lo[j] + rng.random::<f64>() * (cell.hi[j] - cell.lo[j]));
        let (old, new) = gate.overlap_indicators(point);
        if old && !new {
            progress.gate.lost += 1;
        }
        if new && !old && rng.random::<f64>() < lambda / (lambda + z) {
            progress.gate.gained += 1;
        }
        progress.processed_points = index + 1;
        progress.gate.retained_points = progress.gate.gained + progress.gate.lost;
        ensure!(
            progress.gate.retained_points <= retained_cap,
            "fatal retained-point budget exceeded"
        );
    }
    let ratio = z / lambda;
    let coefficient = if ratio.is_finite() {
        ratio.ln_1p()
    } else {
        (lambda + z).ln() - lambda.ln()
    };
    progress.gate.log_weight =
        coefficient * (progress.gate.gained as f64 - progress.gate.lost as f64);
    progress.complete = true;
    Ok(progress.gate)
}

pub fn aggregate(a: GateResult, b: GateResult) -> Result<GateResult> {
    let result = GateResult {
        gained: a.gained.checked_add(b.gained).context("gained overflow")?,
        lost: a.lost.checked_add(b.lost).context("lost overflow")?,
        raw_points: a
            .raw_points
            .checked_add(b.raw_points)
            .context("raw overflow")?,
        retained_points: a
            .retained_points
            .checked_add(b.retained_points)
            .context("retained overflow")?,
        retained_cells: a
            .retained_cells
            .checked_add(b.retained_cells)
            .context("cell overflow")?,
        created_cells: a
            .created_cells
            .checked_add(b.created_cells)
            .context("created overflow")?,
        envelope_volume: a.envelope_volume + b.envelope_volume,
        log_weight: a.log_weight + b.log_weight,
    };
    ensure!(
        result.envelope_volume.is_finite() && result.log_weight.is_finite(),
        "nonfinite path aggregate"
    );
    Ok(result)
}

/// Production call uses None: exactly one fair order coin. Explicit order is
/// only used by tiny reference tests and is unavailable through the config.
pub fn bounded_path(
    tree: &SphereTree,
    state: &[Pose],
    members: [usize; 2],
    proposed: [Pose; 2],
    path: &SingletonPath<'_>,
    rng: &mut StdRng,
    rd: f64,
    lambda: f64,
    z: f64,
    opts: GateOptions,
    budget: &mut Budget,
    explicit_order: Option<SingletonOrder>,
) -> std::result::Result<SingletonPathResult, PathFailure> {
    let order = explicit_order.unwrap_or_else(|| {
        if rng.random::<bool>() {
            SingletonOrder::SecondThenFirst
        } else {
            SingletonOrder::FirstThenSecond
        }
    });
    let ordered_members = path.ordered_members(order);
    let intermediate_selected = path.intermediate_selected(order);
    let indices = if order == SingletonOrder::FirstThenSecond {
        [0, 1]
    } else {
        [1, 0]
    };
    let mut intermediate = state.to_vec();
    intermediate[members[indices[0]]] = proposed[indices[0]];
    let mut completed = Vec::new();
    let mut outer_raw = 0u64;
    let mut outer_retained = 0u64;
    for (leg, &i) in indices.iter().enumerate() {
        let mut progress = LegProgress::default();
        let result = (|| -> Result<GateResult> {
            let source = if leg == 0 { state } else { &intermediate };
            let gate = RigidSubset::new(tree, source, &[members[i]], members[i], proposed[i], rd)?;
            let raw_cap = budget
                .limits
                .raw_per_leg
                .min(budget.limits.raw_per_outer - outer_raw)
                .min(budget.limits.raw_campaign - budget.raw);
            let retained_cap = budget
                .limits
                .retained_per_leg
                .min(budget.limits.retained_per_outer - outer_retained)
                .min(budget.limits.retained_campaign - budget.retained);
            bounded_leg(
                &gate,
                rng,
                lambda,
                z,
                opts,
                raw_cap,
                retained_cap,
                budget,
                &mut progress,
            )
        })();
        match result {
            Ok(gate) => {
                outer_raw += gate.raw_points;
                outer_retained += gate.retained_points;
                budget.raw += gate.raw_points;
                budget.retained += gate.retained_points;
                completed.push(gate);
            }
            Err(error) => {
                return Err(PathFailure {
                    reason: format!("{error:#}"),
                    order,
                    ordered_members,
                    intermediate_selected,
                    completed_legs: completed,
                    failed_leg: leg,
                    failed_progress: progress,
                });
            }
        }
    }
    let legs = [completed[0], completed[1]];
    let aggregate = aggregate(legs[0], legs[1]).map_err(|error| PathFailure {
        reason: format!("{error:#}"),
        order,
        ordered_members,
        intermediate_selected,
        completed_legs: completed,
        failed_leg: 2,
        failed_progress: LegProgress::default(),
    })?;
    Ok(SingletonPathResult {
        order,
        ordered_members,
        intermediate_selected,
        legs,
        aggregate,
    })
}

impl Limits {
    /// Zero point caps are legal reference/failure controls. Every actual cap
    /// is the minimum of the leg, outer and remaining campaign allocations.
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.cpu_seconds.is_finite() && self.cpu_seconds > 0.,
            "invalid campaign CPU limit"
        );
        Ok(())
    }
}

impl Budget {
    /// Start a new bounded invocation. Completed counters are updated only
    /// after a complete leg; a fatal partial leg remains in its failure trace.
    pub fn new(limits: Limits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            limits,
            started: cpu_seconds(),
            raw: 0,
            retained: 0,
        })
    }
}

/// A failed single-body cloud, including the planned count and every processed
/// point counter. Save this record and terminate the allocation; do not retry.
#[derive(Debug, Serialize)]
pub struct LegFailure {
    pub reason: String,
    pub failed_progress: LegProgress,
}
impl fmt::Display for LegFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bounded singleton bath failed: {}", self.reason)
    }
}
impl std::error::Error for LegFailure {}
impl fmt::Display for PathFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bounded singleton path failed at leg {}: {}",
            self.failed_leg, self.reason
        )
    }
}
impl std::error::Error for PathFailure {}

/// Safe complete single-body outer call for a local or other singleton move.
/// The caller constructs one RigidSubset and checks its physical hard endpoint.
/// This wrapper validates sampling inputs before consuming RNG, computes all
/// resource caps and accounts one completed cloud exactly once. It does NOT
/// make an MH decision or mutate particle state. All failure paths expose the
/// partial leg record and leave completed budget counters unchanged.
#[allow(clippy::too_many_arguments)]
pub fn bounded_singleton(
    gate: &RigidSubset<'_>,
    rng: &mut StdRng,
    lambda: f64,
    z: f64,
    opts: GateOptions,
    budget: &mut Budget,
) -> std::result::Result<GateResult, LegFailure> {
    let mut progress = LegProgress::default();
    let result = (|| -> Result<GateResult> {
        budget.limits.validate()?;
        ensure!(
            lambda.is_finite()
                && lambda > 0.
                && z.is_finite()
                && z >= 0.
                && (lambda + z).is_finite(),
            "invalid bath intensity"
        );
        opts.validate()?;
        let raw_cap = budget
            .limits
            .raw_per_leg
            .min(budget.limits.raw_per_outer)
            .min(
                budget
                    .limits
                    .raw_campaign
                    .checked_sub(budget.raw)
                    .context("completed raw count exceeds campaign budget")?,
            );
        let retained_cap = budget
            .limits
            .retained_per_leg
            .min(budget.limits.retained_per_outer)
            .min(
                budget
                    .limits
                    .retained_campaign
                    .checked_sub(budget.retained)
                    .context("completed retained count exceeds campaign budget")?,
            );
        let result = bounded_leg(
            gate,
            rng,
            lambda,
            z,
            opts,
            raw_cap,
            retained_cap,
            budget,
            &mut progress,
        )?;
        // Successful raw and retained counts are bounded by the checked
        // remainders above, so these additions cannot overflow.
        budget.raw += result.raw_points;
        budget.retained += result.retained_points;
        Ok(result)
    })();
    result.map_err(|error| LegFailure {
        reason: format!("{error:#}"),
        failed_progress: progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Atom, Shape};
    use rand::SeedableRng;
    fn pose(x: f64) -> Pose {
        Pose {
            position: [x, 0., 0.],
            orientation: [1., 0., 0., 0.],
        }
    }
    fn tree() -> SphereTree {
        SphereTree::new(Shape {
            name: "bounded path sphere".into(),
            volume: 1.,
            atoms: vec![Atom {
                center: [0.; 3],
                radius: 0.2,
            }],
        })
        .unwrap()
    }
    fn budget() -> Budget {
        Budget {
            limits: Limits {
                raw_per_leg: 1_000_000,
                raw_per_outer: 2_000_000,
                raw_campaign: 10_000_000,
                retained_per_leg: 1_000_000,
                retained_per_outer: 2_000_000,
                retained_campaign: 10_000_000,
                cpu_seconds: 60.,
            },
            started: cpu_seconds(),
            raw: 0,
            retained: 0,
        }
    }
    fn opts() -> GateOptions {
        GateOptions {
            max_cells: 31,
            max_depth: 4,
            min_width: 0.,
        }
    }

    #[test]
    fn bounded_singleton_matches_reference_rng_and_accounts_one_cloud() -> Result<()> {
        let tree = tree();
        let state = [pose(0.), pose(1.), pose(2.)];
        for (proposed, z) in [(pose(-0.3), 1.5), (pose(-0.3), 0.), (state[0], 1.5)] {
            let gate = RigidSubset::new(&tree, &state, &[0], 0, proposed, 0.8)?;
            for seed in [11, 29, 67] {
                let mut a = StdRng::seed_from_u64(seed);
                let mut b = StdRng::seed_from_u64(seed);
                let mut budget = budget();
                budget.raw = 17;
                budget.retained = 9;
                let expected = gate.sample(&mut a, 2., z, opts())?;
                let actual = bounded_singleton(&gate, &mut b, 2., z, opts(), &mut budget)?;
                assert_eq!(
                    serde_json::to_value(expected)?,
                    serde_json::to_value(actual)?
                );
                assert_eq!(a.random::<u64>(), b.random::<u64>());
                assert_eq!(budget.raw, 17 + actual.raw_points);
                assert_eq!(budget.retained, 9 + actual.retained_points);
            }
        }
        Ok(())
    }

    #[test]
    fn singleton_failure_preserves_partial_progress_and_completed_budget() -> Result<()> {
        let tree = tree();
        let state = [pose(0.), pose(1.), pose(2.)];
        let gate = RigidSubset::new(&tree, &state, &[0], 0, pose(-0.3), 0.8)?;
        for cap in [
            "leg_raw",
            "outer_raw",
            "campaign_raw",
            "leg_retained",
            "outer_retained",
            "campaign_retained",
        ] {
            let mut budget = budget();
            budget.raw = 17;
            budget.retained = 9;
            match cap {
                "leg_raw" => budget.limits.raw_per_leg = 0,
                "outer_raw" => budget.limits.raw_per_outer = 0,
                "campaign_raw" => budget.limits.raw_campaign = 17,
                "leg_retained" => budget.limits.retained_per_leg = 0,
                "outer_retained" => budget.limits.retained_per_outer = 0,
                _ => budget.limits.retained_campaign = 9,
            }
            let error = bounded_singleton(
                &gate,
                &mut StdRng::seed_from_u64(11),
                200.,
                1.5,
                opts(),
                &mut budget,
            )
            .unwrap_err();
            assert_eq!((budget.raw, budget.retained), (17, 9), "{cap}");
            assert!(!error.failed_progress.complete);
            assert!(error.failed_progress.gate.raw_points > 0);
            if cap.ends_with("raw") {
                assert_eq!(error.failed_progress.processed_points, 0);
                assert!(error.reason.contains("planned raw-point"));
            } else {
                assert!(error.failed_progress.processed_points > 0);
                assert_eq!(error.failed_progress.gate.retained_points, 1);
                assert!(error.reason.contains("retained-point"));
            }
            assert!(serde_json::to_string(&error)?.contains("failed_progress"));
        }
        Ok(())
    }

    #[test]
    fn invalid_singleton_inputs_or_exhausted_cpu_consume_no_rng() -> Result<()> {
        let tree = tree();
        let state = [pose(0.), pose(1.), pose(2.)];
        let gate = RigidSubset::new(&tree, &state, &[0], 0, pose(-0.3), 0.8)?;
        for defect in [
            "lambda",
            "z",
            "options",
            "raw_budget",
            "retained_budget",
            "cpu",
            "cpu_limit",
        ] {
            let mut budget = budget();
            let mut lambda = 2.;
            let mut z = 1.5;
            let mut options = opts();
            match defect {
                "lambda" => lambda = 0.,
                "z" => z = -1.,
                "options" => options.max_cells = 0,
                "raw_budget" => budget.raw = budget.limits.raw_campaign + 1,
                "retained_budget" => budget.retained = budget.limits.retained_campaign + 1,
                "cpu" => budget.started = cpu_seconds() - 120.,
                _ => budget.limits.cpu_seconds = f64::NAN,
            }
            let before = (budget.raw, budget.retained);
            let mut a = StdRng::seed_from_u64(11);
            let mut b = StdRng::seed_from_u64(11);
            let error =
                bounded_singleton(&gate, &mut a, lambda, z, options, &mut budget).unwrap_err();
            assert_eq!(a.random::<u64>(), b.random::<u64>(), "{defect}");
            assert_eq!((budget.raw, budget.retained), before);
            assert_eq!(error.failed_progress.gate.raw_points, 0);
            assert_eq!(error.failed_progress.processed_points, 0);
            assert!(!error.failed_progress.complete);
        }
        let mut limits = budget().limits;
        limits.cpu_seconds = -1.;
        assert!(Budget::new(limits).is_err());
        limits.cpu_seconds = 60.;
        let fresh = Budget::new(limits)?;
        assert_eq!((fresh.raw, fresh.retained), (0, 0));
        Ok(())
    }
}
