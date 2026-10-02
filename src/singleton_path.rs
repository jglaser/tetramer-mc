//! Two-singleton auxiliary bath path for an independently proposed endpoint.
//!
//! A fair order coin chooses which selected body is replaced first. The reverse
//! move uses the reverse order and exactly the same copied intermediate. Each
//! leg uses the unchanged body-frame `RigidSubset` gate. Positive bath weights
//! telescope through an intermediate that may violate hard cores or the wall.
//! Callers check the final endpoint and make ONE MH decision using the summed
//! bath factor plus the complete endpoint proposal/selection/bias correction.
//! This is not a proposal, production kernel, or established efficiency gain.
use crate::{
    depletion::{GateOptions, GateResult},
    flexible_subset::FlexibleSubset,
    geometry::SphereTree,
    math::{Pose, Vec3},
    rigid_subset::RigidSubset,
    spherical::Container,
};
use anyhow::{Context, Result, ensure};
use rand::{RngExt, rngs::StdRng};
use serde::Serialize;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SingletonOrder {
    FirstThenSecond,
    SecondThenFirst,
}
impl SingletonOrder {
    fn index(self) -> usize {
        match self {
            Self::FirstThenSecond => 0,
            Self::SecondThenFirst => 1,
        }
    }
    fn indices(self) -> [usize; 2] {
        match self {
            Self::FirstThenSecond => [0, 1],
            Self::SecondThenFirst => [1, 0],
        }
    }
    pub fn reversed(self) -> Self {
        match self {
            Self::FirstThenSecond => Self::SecondThenFirst,
            Self::SecondThenFirst => Self::FirstThenSecond,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SingletonPathResult {
    pub order: SingletonOrder,
    pub ordered_members: [usize; 2],
    /// In the constructor's members order, not the sampled execution order.
    pub intermediate_selected: [Pose; 2],
    /// In execution order. Clouds are independent conditional on this path.
    pub legs: [GateResult; 2],
    /// Counts, work diagnostics and log weights are summed over both legs.
    pub aggregate: GateResult,
}

/// An unexpected leg failure retains the chosen path and all completed draws.
/// It is an error, not permission to redraw a cloud, change order, or retry.
#[derive(Clone, Debug, Serialize)]
pub struct SingletonPathError {
    pub order: SingletonOrder,
    pub ordered_members: [usize; 2],
    pub intermediate_selected: [Pose; 2],
    pub completed_legs: Vec<GateResult>,
    /// None means aggregation failed after both legs completed.
    pub failed_leg: Option<usize>,
    pub reason: String,
}
impl fmt::Display for SingletonPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "singleton path failed: ")?;
        match serde_json::to_string(self) {
            Ok(value) => f.write_str(&value),
            Err(_) => write!(f, "{self:?}"),
        }
    }
}
impl std::error::Error for SingletonPathError {}

struct Path<'a> {
    intermediate_selected: [Pose; 2],
    legs: [RigidSubset<'a>; 2],
}

pub struct SingletonPath<'a> {
    endpoint: FlexibleSubset<'a>,
    members: [usize; 2],
    paths: [Path<'a>; 2],
}
impl<'a> SingletonPath<'a> {
    /// Exactly two selected labels and one explicit proposed pose per label.
    /// Numeric/label validation precedes construction of either auxiliary path.
    /// Old/intermediate hard validity is not required by the positive bath law.
    pub fn new(
        tree: &'a SphereTree,
        state: &[Pose],
        members: &[usize],
        proposed: &[Pose],
        rd: f64,
    ) -> Result<Self> {
        ensure!(
            members.len() == 2 && proposed.len() == 2,
            "singleton path needs two endpoints"
        );
        let endpoint = FlexibleSubset::new(tree, state, members, proposed, rd)?;
        let members = [members[0], members[1]];
        let build = |order: SingletonOrder| -> Result<Path<'a>> {
            let [first, second] = order.indices();
            let mut intermediate = state.to_vec();
            intermediate[members[first]] = proposed[first];
            let legs = [
                RigidSubset::new(
                    tree,
                    state,
                    &[members[first]],
                    members[first],
                    proposed[first],
                    rd,
                )?,
                RigidSubset::new(
                    tree,
                    &intermediate,
                    &[members[second]],
                    members[second],
                    proposed[second],
                    rd,
                )?,
            ];
            Ok(Path {
                intermediate_selected: members.map(|i| intermediate[i]),
                legs,
            })
        };
        let paths = [
            build(SingletonOrder::FirstThenSecond)?,
            build(SingletonOrder::SecondThenFirst)?,
        ];
        Ok(Self {
            endpoint,
            members,
            paths,
        })
    }

    /// Full selected-pair/spectator/atomic-wall endpoint check. Fixed spectators'
    /// mutual validity remains the caller's state invariant; the bath has no wall.
    pub fn hard_valid(&self, wall: Option<&Container>, wall_center: Vec3) -> bool {
        self.endpoint.hard_valid(wall, wall_center)
    }
    pub fn members(&self) -> &[usize; 2] {
        &self.members
    }
    pub fn intermediate_selected(&self, order: SingletonOrder) -> [Pose; 2] {
        self.paths[order.index()].intermediate_selected
    }
    pub fn ordered_members(&self, order: SingletonOrder) -> [usize; 2] {
        order.indices().map(|i| self.members[i])
    }

    fn validate_sampling(lambda: f64, z: f64, opts: GateOptions) -> Result<()> {
        ensure!(
            lambda.is_finite()
                && lambda > 0.
                && z.is_finite()
                && z >= 0.
                && (lambda + z).is_finite(),
            "invalid bath intensity"
        );
        opts.validate()
    }

    /// One endpoint-independent fair order coin, then two fresh leg clouds.
    /// Invalid numeric inputs/options consume no RNG. Identity and z=0 still
    /// retain the fair coin; the unchanged underlying gates consume no cloud RNG.
    pub fn sample(
        &self,
        rng: &mut StdRng,
        lambda: f64,
        z: f64,
        opts: GateOptions,
    ) -> Result<SingletonPathResult> {
        Self::validate_sampling(lambda, z, opts)?;
        let order = if rng.random::<bool>() {
            SingletonOrder::SecondThenFirst
        } else {
            SingletonOrder::FirstThenSecond
        };
        self.sample_with_order(rng, lambda, z, opts, order)
    }

    /// Explicit order for diagnostics and paired reverse-path tests. This method
    /// supplies NO order-selection correction. An asymmetric/adaptive production
    /// order law must include its reverse/forward probability ratio; choosing the
    /// cheaper order alone is not licensed by this API. Reverse this order when
    /// reversing the endpoint. Never accept/filter a leg or its intermediate.
    pub fn sample_with_order(
        &self,
        rng: &mut StdRng,
        lambda: f64,
        z: f64,
        opts: GateOptions,
        order: SingletonOrder,
    ) -> Result<SingletonPathResult> {
        Self::validate_sampling(lambda, z, opts)?;
        let path = &self.paths[order.index()];
        let failure = |completed_legs: Vec<GateResult>, failed_leg, error: anyhow::Error| {
            SingletonPathError {
                order,
                ordered_members: self.ordered_members(order),
                intermediate_selected: path.intermediate_selected,
                completed_legs,
                failed_leg,
                reason: format!("{error:#}"),
            }
        };
        let first = path.legs[0]
            .sample(rng, lambda, z, opts)
            .map_err(|e| failure(vec![], Some(0), e))?;
        let second = path.legs[1]
            .sample(rng, lambda, z, opts)
            .map_err(|e| failure(vec![first], Some(1), e))?;
        let aggregate =
            aggregate(first, second).map_err(|e| failure(vec![first, second], None, e))?;
        Ok(SingletonPathResult {
            order,
            ordered_members: self.ordered_members(order),
            intermediate_selected: path.intermediate_selected,
            legs: [first, second],
            aggregate,
        })
    }
}

fn aggregate(a: GateResult, b: GateResult) -> Result<GateResult> {
    let result = GateResult {
        gained: a
            .gained
            .checked_add(b.gained)
            .context("gained count overflow")?,
        lost: a.lost.checked_add(b.lost).context("lost count overflow")?,
        raw_points: a
            .raw_points
            .checked_add(b.raw_points)
            .context("raw point count overflow")?,
        retained_points: a
            .retained_points
            .checked_add(b.retained_points)
            .context("retained point count overflow")?,
        retained_cells: a
            .retained_cells
            .checked_add(b.retained_cells)
            .context("retained cell count overflow")?,
        created_cells: a
            .created_cells
            .checked_add(b.created_cells)
            .context("created cell count overflow")?,
        envelope_volume: a.envelope_volume + b.envelope_volume,
        log_weight: a.log_weight + b.log_weight,
    };
    ensure!(
        result.envelope_volume.is_finite() && result.log_weight.is_finite(),
        "nonfinite path aggregate"
    );
    Ok(result)
}
