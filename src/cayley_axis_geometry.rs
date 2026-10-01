//! Analytic hard-overlap intervals on one raw Cayley orientation coordinate.
//!
//! The center and the other five raw pose coordinates stay fixed.  A moving
//! atom has world position `world_center + R_chart*Cayley(c)*R_anchor*atom`,
//! with `c[axis]=raw/length_scale`.  Sphere predicates are quadratic, not
//! quartic.  Bounding spheres can only prune; atomic leaves decide overlap.
//!
//! "Analytic" is not a floating-point certificate. Compensated coefficients
//! and stable roots reduce cancellation, but transforms, predicates and roots
//! remain FP64 obligations. Unrepresentable arithmetic returns an error.
use crate::{
    geometry::{Placed, SphereTree},
    line_geometry::{Interval, IntervalSet},
    math::*,
};
use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CayleyAxis {
    pub world_center: Vec3,
    pub chart_orientation: [f64; 4],
    pub anchor_rotation: Mat3,
    /// Dimensionless Cayley vector. The selected entry is ignored.
    pub fixed_cayley: Vec3,
    pub axis: usize,
    pub length_scale: f64,
}

impl CayleyAxis {
    pub fn validate(&self) -> Result<()> {
        Pose {
            position: self.world_center,
            orientation: self.chart_orientation,
        }
        .validate()?;
        ensure!(
            self.axis < 3
                && self.length_scale.is_finite()
                && self.length_scale > 0.
                && self.fixed_cayley.iter().all(|v| v.is_finite())
                && self.anchor_rotation.iter().flatten().all(|v| v.is_finite()),
            "Invalid Cayley axis"
        );
        let gram = matmul(self.anchor_rotation, transpose(self.anchor_rotation));
        ensure!(
            (0..3).all(|i| (0..3).all(|j| (gram[i][j] - IDENTITY[i][j]).abs() <= 1e-10))
                && dot(
                    self.anchor_rotation[0],
                    cross(self.anchor_rotation[1], self.anchor_rotation[2])
                ) > 0.,
            "Anchor rotation must be proper orthogonal"
        );
        Ok(())
    }

    /// Direct pose for independent predicates/tests; no interval evaluation.
    pub fn pose(&self, raw: f64) -> Result<Pose> {
        self.validate()?;
        ensure!(raw.is_finite(), "Nonfinite raw Cayley coordinate");
        let mut c = self.fixed_cayley;
        c[self.axis] = raw / self.length_scale;
        ensure!(
            c.iter().all(|v| v.is_finite()) && norm(c).is_finite(),
            "Unrepresentable Cayley vector"
        );
        Ok(Pose {
            position: self.world_center,
            orientation: quaternion(matmul(
                rotation(self.chart_orientation),
                matmul(cayley(c), self.anchor_rotation),
            )),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CayleyAxisCounts {
    pub node_pairs_visited: u64,
    pub node_pairs_pruned: u64,
    pub leaf_pairs_tested: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CayleyAxisGeometry {
    pub hard_overlap: IntervalSet,
    pub hard_free: IntervalSet,
    pub counts: CayleyAxisCounts,
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}

/// Two-component arithmetic, used for coefficient and discriminant signs.
/// This is compensated floating point, not outward-rounded interval arithmetic.
#[derive(Clone, Copy, Debug)]
struct Wide(f64, f64);
impl Wide {
    fn scalar(v: f64) -> Self {
        Self(v, 0.)
    }
    fn normalized(a: f64, b: f64) -> Self {
        let sum = a + b;
        let virtual_b = sum - a;
        Self(sum, (a - (sum - virtual_b)) + (b - virtual_b))
    }
    fn add(self, b: Self) -> Self {
        let s = Self::normalized(self.0, b.0);
        Self::normalized(s.0, s.1 + self.1 + b.1)
    }
    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }
    fn sub(self, b: Self) -> Self {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Self {
        let p = self.0 * b.0;
        Self::normalized(
            p,
            self.0.mul_add(b.0, -p) + self.0 * b.1 + self.1 * b.0 + self.1 * b.1,
        )
    }
    fn div_scalar(self, b: f64) -> Self {
        let q = self.0 / b;
        Self::normalized(q, (-q).mul_add(b, self.0) / b + self.1 / b)
    }
    fn value(self) -> f64 {
        self.0 + self.1
    }
    fn sign(self) -> i8 {
        let x = if self.0 != 0. { self.0 } else { self.1 };
        if x < 0. {
            -1
        } else if x > 0. {
            1
        } else {
            0
        }
    }
    fn sqrt(self) -> Result<Self> {
        ensure!(
            self.0.is_finite() && self.1.is_finite() && self.sign() >= 0,
            "Invalid quadratic discriminant"
        );
        if self.sign() == 0 {
            return Ok(Self::scalar(0.));
        }
        let r = self.value().sqrt();
        ensure!(
            r > 0. && r.is_finite(),
            "Unrepresentable quadratic discriminant root"
        );
        Ok(Self::normalized(
            r,
            self.sub(Self::scalar(r).mul(Self::scalar(r))).value() / (2. * r),
        ))
    }
}
fn wide_dot(a: Vec3, b: Vec3) -> Wide {
    (0..3).fold(Wide::scalar(0.), |sum, i| {
        sum.add(Wide::scalar(a[i]).mul(Wide::scalar(b[i])))
    })
}

fn clipped_open(lower: f64, upper: f64, segment: [f64; 2]) -> Option<Interval> {
    let lo = lower.max(segment[0]);
    let hi = upper.min(segment[1]);
    let lc = lo != lower;
    let hc = hi != upper;
    (lo < hi || (lo == hi && lc && hc)).then_some(Interval {
        lower: lo,
        upper: hi,
        lower_closed: lc,
        upper_closed: hc,
    })
}

/// Solve A*t²+B*t+C<0, mapping t to raw=t*scale without squaring the
/// potentially enormous finite chord. A zero discriminant keeps missing
/// tangent points; a nonzero leading coefficient is never tolerance-trimmed.
fn quadratic_intervals(
    a: Wide,
    b: Wide,
    c: Wide,
    scale: f64,
    segment: [f64; 2],
) -> Result<IntervalSet> {
    let whole = IntervalSet::segment(segment)?;
    ensure!(
        [a.0, a.1, b.0, b.1, c.0, c.1, scale]
            .iter()
            .all(|x| x.is_finite())
            && scale > 0.,
        "Unrepresentable Cayley quadratic"
    );
    let magnitude = [
        a.0.abs(),
        a.1.abs(),
        b.0.abs(),
        b.1.abs(),
        c.0.abs(),
        c.1.abs(),
    ]
    .into_iter()
    .fold(0., f64::max);
    if magnitude == 0. {
        return Ok(IntervalSet::empty());
    }
    let normalize = |v: Wide| v.div_scalar(magnitude);
    let (a, b, c) = (normalize(a), normalize(b), normalize(c));
    let mut pieces = Vec::new();
    let mut add_piece = |lo, hi| {
        if let Some(i) = clipped_open(lo, hi, segment) {
            pieces.push(i);
        }
    };
    let root = |numerator: f64, denominator: f64| -> Result<f64> {
        // Try both operation orders: an intermediate quotient/product can
        // overflow although the final raw root is finite.
        let first = (numerator / denominator) * scale;
        let second = (numerator * scale) / denominator;
        let value = if first.is_finite() && (first != 0. || second == 0. || !second.is_finite()) {
            first
        } else if second.is_finite() {
            second
        } else {
            first
        };
        ensure!(!value.is_nan(), "Unrepresentable raw Cayley root");
        Ok(value)
    };
    if a.sign() == 0 {
        if b.sign() == 0 {
            return Ok(if c.sign() < 0 {
                whole
            } else {
                IntervalSet::empty()
            });
        }
        let r = root(-c.value(), b.value())?;
        if b.sign() > 0 {
            add_piece(f64::NEG_INFINITY, r);
        } else {
            add_piece(r, f64::INFINITY);
        }
    } else {
        let disc = b.mul(b).sub(a.mul(c).mul(Wide::scalar(4.)));
        if disc.sign() < 0 {
            return Ok(if a.sign() < 0 {
                whole
            } else {
                IntervalSet::empty()
            });
        }
        if disc.sign() == 0 {
            if a.sign() > 0 {
                return Ok(IntervalSet::empty());
            }
            let r = root(-0.5 * b.value(), a.value())?;
            add_piece(f64::NEG_INFINITY, r);
            add_piece(r, f64::INFINITY);
        } else {
            let square = disc.sqrt()?.value();
            let q = -0.5 * (b.value() + square.copysign(b.value()));
            ensure!(
                q != 0. && q.is_finite(),
                "Unrepresentable stable quadratic root"
            );
            let first = root(q, a.value())?;
            let second = root(c.value(), q)?;
            let lo = first.min(second);
            let hi = first.max(second);
            ensure!(
                lo != hi || !lo.is_finite() || lo < segment[0] || lo > segment[1],
                "Distinct Cayley roots collapsed inside finite chord"
            );
            if a.sign() > 0 {
                add_piece(lo, hi);
            } else {
                add_piece(f64::NEG_INFINITY, lo);
                add_piece(hi, f64::INFINITY);
            }
        }
    }
    IntervalSet::from_intervals(pieces)
}

#[derive(Clone, Copy)]
struct Orbit {
    rotation_zero: Mat3,
    axis: Vec3,
    raw_scale: f64,
}
impl Orbit {
    fn new(mut v: Vec3, axis: usize, ell: f64) -> Result<Self> {
        ensure!(
            axis < 3 && v.iter().all(|x| x.is_finite()) && ell.is_finite() && ell > 0.,
            "Invalid Cayley orbit"
        );
        v[axis] = 0.;
        let h = 1_f64.hypot(norm(v));
        let mut e = [0.; 3];
        e[axis] = 1.;
        let k = add(e, cross(v, e)).map(|x| x / h);
        let raw_scale = ell * h;
        ensure!(
            h.is_finite()
                && raw_scale.is_finite()
                && raw_scale > 0.
                && k.iter().all(|x| x.is_finite()),
            "Unrepresentable Cayley orbit scale"
        );
        Ok(Self {
            rotation_zero: cayley(v),
            axis: k,
            raw_scale,
        })
    }
    fn sphere(self, a: Vec3, b: Vec3, radius: f64, segment: [f64; 2]) -> Result<IntervalSet> {
        ensure!(
            a.iter().chain(b.iter()).all(|x| x.is_finite()) && radius.is_finite() && radius >= 0.,
            "Invalid Cayley sphere pair"
        );
        let spatial = norm(a).max(norm(b)).max(radius);
        ensure!(spatial.is_finite(), "Unrepresentable Cayley sphere scale");
        if radius == 0. {
            return Ok(IntervalSet::empty());
        }
        let a = a.map(|x| x / spatial);
        let b = b.map(|x| x / spatial);
        let r = radius / spatial;
        let r2 = Wide::scalar(r).mul(Wide::scalar(r));
        ensure!(
            r2.sign() > 0,
            "Unrepresentable scaled sphere radius squared"
        );
        let p = matvec(self.rotation_zero, a);
        let axial = dot(self.axis, p);
        let opposite = std::array::from_fn(|i| (2. * self.axis[i]).mul_add(axial, -p[i]));
        let minus = sub(p, b);
        let plus = sub(opposite, b);
        let c = wide_dot(minus, minus).sub(r2);
        let aa = wide_dot(plus, plus).sub(r2);
        let bb = wide_dot(b, cross(self.axis, p)).mul(Wide::scalar(-4.));
        quadratic_intervals(aa, bb, c, self.raw_scale, segment)
    }
}

/// Leaf sphere predicate in the chart frame. `a` has already received the
/// anchor rotation; `b=R_chartᵀ*(fixed_world-world_center)`. Endpoints use the
/// raw coordinate ell*c_j. The selected entry of fixed_cayley is ignored.
#[allow(clippy::too_many_arguments)]
pub fn sphere_overlap_intervals(
    a: Vec3,
    b: Vec3,
    radius: f64,
    fixed_cayley: Vec3,
    axis: usize,
    length_scale: f64,
    segment: [f64; 2],
) -> Result<IntervalSet> {
    IntervalSet::segment(segment)?;
    Orbit::new(fixed_cayley, axis, length_scale)?.sphere(a, b, radius, segment)
}

/// Full sphere-union hard intervals at fixed center. Capture and any physical
/// wall are caller constraints. Callers supply all required periodic images.
pub fn cayley_axis_intervals(
    moving_tree: &SphereTree,
    axis: &CayleyAxis,
    fixed_tree: &SphereTree,
    fixed_poses: &[Pose],
    segment: [f64; 2],
) -> Result<CayleyAxisGeometry> {
    axis.validate()?;
    let whole = IntervalSet::segment(segment)?;
    let orbit = Orbit::new(axis.fixed_cayley, axis.axis, axis.length_scale)?;
    let inverse = transpose(rotation(axis.chart_orientation));
    let mut counts = CayleyAxisCounts::default();
    let mut forbidden = IntervalSet::empty();
    for pose in fixed_poses {
        pose.validate()?;
        let fixed = Placed::new(*pose);
        let mut pending = vec![(0, 0)];
        while let Some((i, j)) = pending.pop() {
            counts.node_pairs_visited += 1;
            let a = moving_tree.line_node(i);
            let b = fixed_tree.line_node(j);
            let ac = matvec(axis.anchor_rotation, a.center);
            let world_b = fixed.apply(b.center);
            let bc = matvec(inverse, sub(world_b, axis.world_center));
            let guard = 2048.
                * f64::EPSILON
                * (1.
                    + norm(ac)
                    + norm(bc)
                    + norm(world_b)
                    + norm(axis.world_center)
                    + a.radius
                    + b.radius);
            ensure!(guard.is_finite(), "Unrepresentable Cayley BVH guard");
            if orbit
                .sphere(ac, bc, a.radius + b.radius + guard, segment)?
                .is_empty()
            {
                counts.node_pairs_pruned += 1;
                continue;
            }
            if let (Some(ai), Some(bi)) = (a.atom, b.atom) {
                counts.leaf_pairs_tested += 1;
                let aa = &moving_tree.shape.atoms[ai];
                let bb = &fixed_tree.shape.atoms[bi];
                let ac = matvec(axis.anchor_rotation, aa.center);
                let bc = matvec(inverse, sub(fixed.apply(bb.center), axis.world_center));
                forbidden =
                    forbidden.union(&orbit.sphere(ac, bc, aa.radius + bb.radius, segment)?);
            } else if a.children.is_some() && (b.children.is_none() || a.radius >= b.radius) {
                let (l, r) = a.children.unwrap();
                pending.push((r, j));
                pending.push((l, j));
            } else {
                let (l, r) = b.children.unwrap();
                pending.push((i, r));
                pending.push((i, l));
            }
        }
    }
    Ok(CayleyAxisGeometry {
        hard_free: whole.difference(&forbidden),
        hard_overlap: forbidden,
        counts,
    })
}

#[cfg(test)]
mod tests;
