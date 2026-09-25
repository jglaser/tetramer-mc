//! Exact conditional Poisson acceptance after independent endpoint selection.
//! Unknown cells remain in a conservative, disjoint XOR envelope at any depth.
use crate::{
    geometry::{Cell, Coverage, Environment, Placed, SphereTree},
    math::{Pose, Vec3},
};
use anyhow::{Result, ensure};
use rand::{RngExt, rngs::StdRng};
use rand_distr::{Distribution, Poisson};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GateOptions {
    pub max_cells: usize,
    pub max_depth: u32,
    pub min_width: f64,
}
impl Default for GateOptions {
    fn default() -> Self {
        Self {
            max_cells: 2047,
            max_depth: 14,
            min_width: 0.5,
        }
    }
}
impl GateOptions {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.max_cells > 0 && self.min_width.is_finite() && self.min_width >= 0.,
            "invalid envelope options"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Envelope {
    pub cells: Vec<Cell>,
    pub cumulative: Vec<f64>,
    pub volume: f64,
    pub created: usize,
}
impl Envelope {
    pub fn build(env: &Environment, old: Pose, new: Pose, opts: GateOptions) -> Result<Self> {
        opts.validate()?;
        old.validate()?;
        new.validate()?;
        if env.fixed.is_empty() || old == new {
            return Ok(Self {
                cells: Vec::new(),
                cumulative: Vec::new(),
                volume: 0.,
                created: 0,
            });
        }
        let root = env.tree.bounds(env.rd);
        ensure!(
            root.volume().is_finite() && root.volume() > 0.,
            "invalid root AABB"
        );
        let old = Placed::new(old);
        let new = Placed::new(new);
        let mut queue = VecDeque::from([(root, 0)]);
        let mut cells = Vec::new();
        let mut created = 1;
        while let Some((cell, depth)) = queue.pop_front() {
            let c = cell.center();
            let r = cell.radius();
            let body = env.tree.classify_ball(c, r, env.rd);
            if body == Coverage::Outside {
                continue;
            }
            let a = env.classify_ball(old.apply(c), r);
            let b = env.classify_ball(new.apply(c), r);
            if a == b && a != Coverage::Unknown {
                continue;
            }
            let known =
                body == Coverage::Inside && a != Coverage::Unknown && b != Coverage::Unknown;
            if known
                || depth >= opts.max_depth
                || created + 2 > opts.max_cells
                || cell.hi[cell.longest()] - cell.lo[cell.longest()] <= opts.min_width
            {
                cells.push(cell);
            } else if let Some((left, right)) = cell.split() {
                queue.push_back((left, depth + 1));
                queue.push_back((right, depth + 1));
                created += 2;
            } else {
                cells.push(cell);
            }
        }
        let mut cumulative = Vec::new();
        let mut volume = 0.;
        for cell in &cells {
            let next = volume + cell.volume();
            if !next.is_finite() || next <= volume {
                return Ok(Self {
                    cells: vec![root],
                    cumulative: vec![root.volume()],
                    volume: root.volume(),
                    created,
                });
            }
            volume = next;
            cumulative.push(volume);
        }
        Ok(Self {
            cells,
            cumulative,
            volume,
            created,
        })
    }
}

/// Immutable body-frame data for one cell of the canonical longest-axis tree.
/// Neighborhood classifications are deliberately not cached: they depend on poses.
#[derive(Clone, Copy, Debug)]
struct BodyCell {
    cell: Cell,
    center: Vec3,
    radius: f64,
    coverage: Coverage,
    width: f64,
    children: Option<(usize, usize)>,
}
impl BodyCell {
    fn new(tree: &SphereTree, rd: f64, cell: Cell) -> Self {
        let center = cell.center();
        let radius = cell.radius();
        let axis = cell.longest();
        Self {
            cell,
            center,
            radius,
            coverage: tree.classify_ball(center, radius, rd),
            width: cell.hi[axis] - cell.lo[axis],
            children: None,
        }
    }
}

#[derive(Clone, Copy)]
enum BodyCellRef {
    Cached(usize),
    Uncached(Cell),
}

/// Lazy, bounded cache of geometry shared by all depletion endpoint envelopes.
///
/// The borrowed tree and fixed depletant radius bind this cache to one rigid
/// shape. Dyadic cells depend only on that shape, not on the current endpoints,
/// environment, or traversal budget. We retain the reference BFS order and do
/// the same neighborhood predicates on every query. Consequently the resulting
/// envelope, Poisson gate, and RNG continuation match the uncached algorithm.
/// Cache exhaustion falls back to uncached traversal; it never truncates a query.
/// This is performance state only and need not be stored in a checkpoint.
#[derive(Debug)]
pub struct BodyEnvelopeCache<'a> {
    tree: &'a SphereTree,
    rd: f64,
    nodes: Vec<BodyCell>,
    max_nodes: usize,
}
impl<'a> BodyEnvelopeCache<'a> {
    pub fn new(tree: &'a SphereTree, rd: f64) -> Result<Self> {
        Self::with_node_limit(tree, rd, 65_535)
    }

    pub fn with_node_limit(tree: &'a SphereTree, rd: f64, max_nodes: usize) -> Result<Self> {
        ensure!(
            rd.is_finite() && rd >= 0.,
            "invalid cached depletant radius"
        );
        ensure!(max_nodes > 0, "body-envelope cache needs at least one node");
        // Use exactly the reference arithmetic, including its outward guard.
        let root = tree.bounds(rd);
        ensure!(
            root.volume().is_finite() && root.volume() > 0.,
            "invalid root AABB"
        );
        Ok(Self {
            tree,
            rd,
            nodes: vec![BodyCell::new(tree, rd, root)],
            max_nodes,
        })
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn validate_environment(&self, env: &Environment) -> Result<()> {
        ensure!(
            std::ptr::eq(self.tree, env.tree),
            "cached shape differs from environment"
        );
        ensure!(
            self.rd.to_bits() == env.rd.to_bits(),
            "cached depletant radius differs from environment"
        );
        Ok(())
    }

    pub fn build(
        &mut self,
        env: &Environment,
        old: Pose,
        new: Pose,
        opts: GateOptions,
    ) -> Result<Envelope> {
        Ok(self.build_for_points::<false>(env, old, new, opts)?.0)
    }

    // Certificates belong to this endpoint pair and environment only. Keep
    // them private, alongside the exact envelope, until its point loop ends.
    fn build_for_points<const CERTIFY: bool>(
        &mut self,
        env: &Environment,
        old: Pose,
        new: Pose,
        opts: GateOptions,
    ) -> Result<(Envelope, Vec<CellCoverage>)> {
        self.validate_environment(env)?;
        opts.validate()?;
        old.validate()?;
        new.validate()?;
        if env.fixed.is_empty() || old == new {
            return Ok((
                Envelope {
                    cells: Vec::new(),
                    cumulative: Vec::new(),
                    volume: 0.,
                    created: 0,
                },
                Vec::new(),
            ));
        }
        let root = self.nodes[0].cell;
        let old = Placed::new(old);
        let new = Placed::new(new);
        let mut queue = VecDeque::from([(BodyCellRef::Cached(0), 0)]);
        let mut cells = Vec::new();
        let mut coverage = Vec::new();
        let mut created = 1;
        while let Some((reference, depth)) = queue.pop_front() {
            let node = match reference {
                BodyCellRef::Cached(index) => self.nodes[index],
                BodyCellRef::Uncached(cell) => BodyCell::new(self.tree, self.rd, cell),
            };
            if node.coverage == Coverage::Outside {
                continue;
            }
            let a = env.classify_ball(old.apply(node.center), node.radius);
            let b = env.classify_ball(new.apply(node.center), node.radius);
            if a == b && a != Coverage::Unknown {
                continue;
            }
            let known = node.coverage == Coverage::Inside
                && a != Coverage::Unknown
                && b != Coverage::Unknown;
            if known
                || depth >= opts.max_depth
                || created + 2 > opts.max_cells
                || node.width <= opts.min_width
            {
                cells.push(node.cell);
                if CERTIFY {
                    coverage.push(CellCoverage {
                        body: node.coverage,
                        old: a,
                        new: b,
                    });
                }
                continue;
            }
            let children = if let Some((left, right)) = node.children {
                Some((BodyCellRef::Cached(left), BodyCellRef::Cached(right)))
            } else if let Some((left, right)) = node.cell.split() {
                if let BodyCellRef::Cached(index) = reference {
                    if self.max_nodes - self.nodes.len() >= 2 {
                        let left_index = self.nodes.len();
                        self.nodes.push(BodyCell::new(self.tree, self.rd, left));
                        self.nodes.push(BodyCell::new(self.tree, self.rd, right));
                        self.nodes[index].children = Some((left_index, left_index + 1));
                        Some((
                            BodyCellRef::Cached(left_index),
                            BodyCellRef::Cached(left_index + 1),
                        ))
                    } else {
                        Some((BodyCellRef::Uncached(left), BodyCellRef::Uncached(right)))
                    }
                } else {
                    Some((BodyCellRef::Uncached(left), BodyCellRef::Uncached(right)))
                }
            } else {
                None
            };
            if let Some((left, right)) = children {
                queue.push_back((left, depth + 1));
                queue.push_back((right, depth + 1));
                created += 2;
            } else {
                cells.push(node.cell);
                if CERTIFY {
                    coverage.push(CellCoverage {
                        body: node.coverage,
                        old: a,
                        new: b,
                    });
                }
            }
        }
        let mut cumulative = Vec::new();
        let mut volume = 0.;
        for cell in &cells {
            let next = volume + cell.volume();
            if !next.is_finite() || next <= volume {
                // Coarsening invalidates child certificates. Unknown forces
                // the original point predicates for the root fallback.
                return Ok((
                    Envelope {
                        cells: vec![root],
                        cumulative: vec![root.volume()],
                        volume: root.volume(),
                        created,
                    },
                    if CERTIFY {
                        vec![CellCoverage::UNKNOWN]
                    } else {
                        Vec::new()
                    },
                ));
            }
            volume = next;
            cumulative.push(volume);
        }
        Ok((
            Envelope {
                cells,
                cumulative,
                volume,
                created,
            },
            coverage,
        ))
    }

    /// Reference cached gate without point certificates, retained for audits.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_unclassified(
        &mut self,
        rng: &mut StdRng,
        env: &Environment,
        old: Pose,
        new: Pose,
        lambda: f64,
        z: f64,
        opts: GateOptions,
    ) -> Result<GateResult> {
        self.validate_environment(env)?;
        ensure!(
            lambda.is_finite()
                && lambda > 0.
                && z.is_finite()
                && z >= 0.
                && (lambda + z).is_finite(),
            "invalid bath intensity"
        );
        old.validate()?;
        new.validate()?;
        opts.validate()?;
        if z == 0. {
            return Ok(GateResult::default());
        }
        let envelope = self.build(env, old, new, opts)?;
        sample_with_envelope(rng, env, old, new, lambda, z, &envelope)
    }

    /// Reuse cell certificates while preserving all point and thinning draws.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &mut self,
        rng: &mut StdRng,
        env: &Environment,
        old: Pose,
        new: Pose,
        lambda: f64,
        z: f64,
        opts: GateOptions,
    ) -> Result<GateResult> {
        Ok(self
            .sample_certified::<false>(rng, env, old, new, lambda, z, opts)?
            .0)
    }

    /// Same gate with query counters for diagnostics; production omits counting.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_profiled(
        &mut self,
        rng: &mut StdRng,
        env: &Environment,
        old: Pose,
        new: Pose,
        lambda: f64,
        z: f64,
        opts: GateOptions,
    ) -> Result<(GateResult, ContainmentQueries)> {
        self.sample_certified::<true>(rng, env, old, new, lambda, z, opts)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_certified<const PROFILE: bool>(
        &mut self,
        rng: &mut StdRng,
        env: &Environment,
        old: Pose,
        new: Pose,
        lambda: f64,
        z: f64,
        opts: GateOptions,
    ) -> Result<(GateResult, ContainmentQueries)> {
        self.validate_environment(env)?;
        ensure!(
            lambda.is_finite()
                && lambda > 0.
                && z.is_finite()
                && z >= 0.
                && (lambda + z).is_finite(),
            "invalid bath intensity"
        );
        old.validate()?;
        new.validate()?;
        opts.validate()?;
        if z == 0. {
            return Ok((GateResult::default(), ContainmentQueries::default()));
        }
        let (envelope, coverage) = self.build_for_points::<true>(env, old, new, opts)?;
        sample_with_certificates::<PROFILE>(rng, env, old, new, lambda, z, &envelope, &coverage)
    }
}

/// Per-point containment calls performed or avoided by conservative cell labels.
/// These count full body/environment queries, not internal BVH nodes visited.
#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize)]
pub struct ContainmentQueries {
    pub body_queries: u64,
    pub body_skipped: u64,
    pub old_queries: u64,
    pub old_skipped: u64,
    pub new_queries: u64,
    pub new_skipped: u64,
}

#[derive(Clone, Copy, Debug)]
struct CellCoverage {
    body: Coverage,
    old: Coverage,
    new: Coverage,
}
impl CellCoverage {
    const UNKNOWN: Self = Self {
        body: Coverage::Unknown,
        old: Coverage::Unknown,
        new: Coverage::Unknown,
    };
}

#[inline]
fn certified_contains<const PROFILE: bool>(
    coverage: Coverage,
    queries: &mut u64,
    skipped: &mut u64,
    predicate: impl FnOnce() -> bool,
) -> bool {
    match coverage {
        Coverage::Unknown => {
            if PROFILE {
                *queries += 1;
            }
            predicate()
        }
        known => {
            if PROFILE {
                *skipped += 1;
            }
            known == Coverage::Inside
        }
    }
}

#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize)]
pub struct GateResult {
    pub gained: u64,
    pub lost: u64,
    pub raw_points: u64,
    pub retained_points: u64,
    pub log_weight: f64,
    pub envelope_volume: f64,
    pub retained_cells: usize,
    pub created_cells: usize,
}

pub fn sample_with_envelope(
    rng: &mut StdRng,
    env: &Environment,
    old: Pose,
    new: Pose,
    lambda: f64,
    z: f64,
    envelope: &Envelope,
) -> Result<GateResult> {
    ensure!(
        lambda.is_finite() && lambda > 0. && z.is_finite() && z >= 0. && (lambda + z).is_finite(),
        "invalid bath intensity"
    );
    old.validate()?;
    new.validate()?;
    let mut result = GateResult {
        envelope_volume: envelope.volume,
        retained_cells: envelope.cells.len(),
        created_cells: envelope.created,
        ..Default::default()
    };
    if z == 0. || envelope.volume == 0. {
        return Ok(result);
    }
    let mean = (lambda + z) * envelope.volume;
    ensure!(mean.is_finite() && mean < 9e15, "unsupported Poisson mean");
    let number = Poisson::<f64>::new(mean)?.sample(rng) as u64;
    result.raw_points = number;
    let old = Placed::new(old);
    let new = Placed::new(new);
    for _ in 0..number {
        let target = rng.random::<f64>() * envelope.volume;
        let k = envelope
            .cumulative
            .partition_point(|&v| v <= target)
            .min(envelope.cells.len() - 1);
        let cell = envelope.cells[k];
        let p =
            std::array::from_fn(|j| cell.lo[j] + rng.random::<f64>() * (cell.hi[j] - cell.lo[j]));
        if !env.tree.contains(p, env.rd) {
            continue;
        }
        let a = env.contains(old.apply(p));
        let b = env.contains(new.apply(p));
        if a && !b {
            result.lost += 1;
        }
        if b && !a && rng.random::<f64>() < lambda / (lambda + z) {
            result.gained += 1;
        }
    }
    result.retained_points = result.gained + result.lost;
    let ratio = z / lambda;
    let coefficient = if ratio.is_finite() {
        ratio.ln_1p()
    } else {
        (lambda + z).ln() - lambda.ln()
    };
    result.log_weight = coefficient * (result.gained as f64 - result.lost as f64);
    Ok(result)
}

// The envelope and certificates are constructed together by the private cache
// path. Certificates cannot be supplied by a caller or reused with other poses.
// Known labels cover every point of the circumscribed cell ball; the geometry
// predicates include their existing outward floating-point guards. Unknown
// labels always take the original exact point-containment path.
#[allow(clippy::too_many_arguments)]
fn sample_with_certificates<const PROFILE: bool>(
    rng: &mut StdRng,
    env: &Environment,
    old: Pose,
    new: Pose,
    lambda: f64,
    z: f64,
    envelope: &Envelope,
    coverage: &[CellCoverage],
) -> Result<(GateResult, ContainmentQueries)> {
    debug_assert_eq!(envelope.cells.len(), coverage.len());
    let mut result = GateResult {
        envelope_volume: envelope.volume,
        retained_cells: envelope.cells.len(),
        created_cells: envelope.created,
        ..Default::default()
    };
    let mut queries = ContainmentQueries::default();
    if z == 0. || envelope.volume == 0. {
        return Ok((result, queries));
    }
    let mean = (lambda + z) * envelope.volume;
    ensure!(mean.is_finite() && mean < 9e15, "unsupported Poisson mean");
    let number = Poisson::<f64>::new(mean)?.sample(rng) as u64;
    result.raw_points = number;
    let old = Placed::new(old);
    let new = Placed::new(new);
    for _ in 0..number {
        let target = rng.random::<f64>() * envelope.volume;
        let k = envelope
            .cumulative
            .partition_point(|&v| v <= target)
            .min(envelope.cells.len() - 1);
        let cell = envelope.cells[k];
        // Even fully certified cells consume exactly the same position draws.
        let p =
            std::array::from_fn(|j| cell.lo[j] + rng.random::<f64>() * (cell.hi[j] - cell.lo[j]));
        let certificate = coverage[k];
        if !certified_contains::<PROFILE>(
            certificate.body,
            &mut queries.body_queries,
            &mut queries.body_skipped,
            || env.tree.contains(p, env.rd),
        ) {
            continue;
        }
        let a = certified_contains::<PROFILE>(
            certificate.old,
            &mut queries.old_queries,
            &mut queries.old_skipped,
            || env.contains(old.apply(p)),
        );
        let b = certified_contains::<PROFILE>(
            certificate.new,
            &mut queries.new_queries,
            &mut queries.new_skipped,
            || env.contains(new.apply(p)),
        );
        if a && !b {
            result.lost += 1;
        }
        if b && !a && rng.random::<f64>() < lambda / (lambda + z) {
            result.gained += 1;
        }
    }
    result.retained_points = result.gained + result.lost;
    let ratio = z / lambda;
    let coefficient = if ratio.is_finite() {
        ratio.ln_1p()
    } else {
        (lambda + z).ln() - lambda.ln()
    };
    result.log_weight = coefficient * (result.gained as f64 - result.lost as f64);
    Ok((result, queries))
}

pub fn sample(
    rng: &mut StdRng,
    env: &Environment,
    old: Pose,
    new: Pose,
    lambda: f64,
    z: f64,
    opts: GateOptions,
) -> Result<GateResult> {
    ensure!(
        lambda.is_finite() && lambda > 0. && z.is_finite() && z >= 0. && (lambda + z).is_finite(),
        "invalid bath intensity"
    );
    old.validate()?;
    new.validate()?;
    opts.validate()?;
    if z == 0. {
        return Ok(GateResult::default());
    }
    let envelope = Envelope::build(env, old, new, opts)?;
    sample_with_envelope(rng, env, old, new, lambda, z, &envelope)
}
