//! Atomic hard-overlap arcs on a fixed translation circle.
//!
//! This is geometry and normalized azimuth-mass evaluation only. It samples
//! nothing, does not change a proposal, and adds no physical-domain constraint.
//! Angles use [0,2π); overlap inequalities are strict, including tangent
//! singleton exceptions. Bounding spheres only prune separated node pairs.
//! Analytic exactness is relative to the sphere-union model; FP64 predicates,
//! rotations and arc endpoints remain explicit numerical obligations.
use crate::{
    contact_distances::{AzimuthLaw, ContactCircle},
    geometry::{Placed, SphereTree},
    line_geometry::{Interval, IntervalSet},
    math::{Pose, Vec3, dot, norm, sub},
};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct CircleCounts {
    pub node_pairs_visited: u64,
    pub node_pairs_pruned: u64,
    pub leaf_pairs_tested: u64,
    pub full_cover_early_exit: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CircleGeometry {
    pub forbidden: IntervalSet,
    pub allowed: IntervalSet,
    pub counts: CircleCounts,
}

fn whole() -> Result<IntervalSet> {
    IntervalSet::from_intervals(vec![Interval {
        lower: 0.,
        upper: TAU,
        lower_closed: true,
        upper_closed: false,
    }])
}

fn validate_circle(circle: &ContactCircle) -> Result<()> {
    ensure!(
        circle
            .center
            .iter()
            .chain(circle.axis.iter())
            .chain(circle.basis.iter().flatten())
            .all(|v| v.is_finite())
            && circle.radius.is_finite()
            && circle.radius >= 0.,
        "Invalid translation circle"
    );
    let vectors = [circle.axis, circle.basis[0], circle.basis[1]];
    for i in 0..3 {
        ensure!(
            (norm(vectors[i]) - 1.).abs() <= 1e-12,
            "Circle basis must be unit length"
        );
        for j in 0..i {
            ensure!(
                dot(vectors[i], vectors[j]).abs() <= 1e-12,
                "Circle basis must be orthogonal"
            );
        }
    }
    Ok(())
}

fn projected_distances(offset: Vec3, circle: &ContactCircle) -> Result<(f64, f64, [f64; 2])> {
    let xy = circle.basis.map(|v| dot(offset, v));
    let perpendicular = xy[0].hypot(xy[1]);
    let axial = dot(offset, circle.axis);
    let minimum = axial.hypot(perpendicular - circle.radius);
    let maximum = axial.hypot(perpendicular + circle.radius);
    ensure!(
        [xy[0], xy[1], perpendicular, axial, minimum, maximum]
            .iter()
            .all(|x| x.is_finite()),
        "Nonfinite circle distance"
    );
    Ok((minimum, maximum, xy))
}

fn open_arc(center: f64, half_width: f64) -> Result<IntervalSet> {
    ensure!(
        center.is_finite() && half_width.is_finite() && half_width > 0. && half_width <= PI,
        "Invalid open arc"
    );
    let center = center.rem_euclid(TAU);
    if half_width == PI {
        let point = (center + PI).rem_euclid(TAU);
        return Ok(
            whole()?.difference(&IntervalSet::from_intervals(vec![Interval::closed(
                point, point,
            )?])?),
        );
    }
    let lower = center - half_width;
    let upper = center + half_width;
    let intervals = if lower < 0. {
        vec![
            Interval {
                lower: 0.,
                upper,
                lower_closed: true,
                upper_closed: false,
            },
            Interval {
                lower: lower + TAU,
                upper: TAU,
                lower_closed: false,
                upper_closed: false,
            },
        ]
    } else if upper > TAU {
        vec![
            Interval {
                lower,
                upper: TAU,
                lower_closed: false,
                upper_closed: false,
            },
            Interval {
                lower: 0.,
                upper: upper - TAU,
                lower_closed: true,
                upper_closed: false,
            },
        ]
    } else {
        vec![Interval {
            lower,
            upper,
            lower_closed: false,
            upper_closed: false,
        }]
    };
    IntervalSet::from_intervals(intervals)
}

/// Two-component arithmetic is used only near an extremum, where rounding a
/// hypot result before comparing it to R can change a one-ulp topology. FMA
/// retains product residuals; this improves FP64 predicates but is not a formal
/// interval-arithmetic certificate for arbitrary magnitudes or input rotations.
#[derive(Clone, Copy)]
struct Wide(f64, f64);
impl Wide {
    fn scalar(x: f64) -> Self {
        Self(x, 0.)
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
    fn sub(self, b: Self) -> Self {
        self.add(Self(-b.0, -b.1))
    }
    fn mul(self, b: Self) -> Self {
        let p = self.0 * b.0;
        Self::normalized(
            p,
            self.0.mul_add(b.0, -p) + self.0 * b.1 + self.1 * b.0 + self.1 * b.1,
        )
    }
    fn sqrt(self) -> Result<Self> {
        ensure!(
            self.0.is_finite() && self.1.is_finite() && self.value() >= 0.,
            "Invalid compensated square root"
        );
        if self.value() == 0. {
            return Ok(Self::scalar(0.));
        }
        let root = self.0.sqrt();
        let residual = self.sub(Self::scalar(root).mul(Self::scalar(root)));
        let result = Self::normalized(root, residual.value() / (2. * root));
        ensure!(
            result.0.is_finite() && result.1.is_finite(),
            "Unrepresentable compensated root"
        );
        Ok(result)
    }
    fn value(self) -> f64 {
        self.0 + self.1
    }
}
fn wide_dot(a: Vec3, b: Vec3) -> Wide {
    (0..3).fold(Wide::scalar(0.), |sum, i| {
        sum.add(Wide::scalar(a[i]).mul(Wide::scalar(b[i])))
    })
}
fn near_tangent_arcs(offset: Vec3, radius: f64, circle: &ContactCircle) -> Result<IntervalSet> {
    let x = wide_dot(offset, circle.basis[0]);
    let y = wide_dot(offset, circle.basis[1]);
    let perpendicular = x.mul(x).add(y.mul(y)).sqrt()?;
    let rho = Wide::scalar(circle.radius);
    let base = wide_dot(offset, offset).add(rho.mul(rho));
    let amplitude = rho.mul(perpendicular).mul(Wide::scalar(2.));
    let target = Wide::scalar(radius).mul(Wide::scalar(radius));
    ensure!(
        [base.0, base.1, amplitude.0, amplitude.1, target.0, target.1]
            .iter()
            .all(|v| v.is_finite()),
        "Nonfinite compensated circle predicate"
    );
    if amplitude.value() == 0. {
        return if target.sub(base).value() > 0. {
            whole()
        } else {
            Ok(IntervalSet::empty())
        };
    }
    let near_gap = target.sub(base.sub(amplitude)).value();
    let far_gap = base.add(amplitude).sub(target).value();
    if near_gap <= 0. {
        return Ok(IntervalSet::empty());
    }
    if far_gap < 0. {
        return whole();
    }
    let center = (-y.value()).atan2(-x.value());
    if far_gap == 0. {
        return open_arc(center, PI);
    }
    let near = near_gap <= far_gap;
    let sine = ((if near { near_gap } else { far_gap }) / (2. * amplitude.value())).sqrt();
    ensure!(
        sine.is_finite() && sine > 0. && sine <= 1.,
        "Unrepresentable compensated circle root"
    );
    open_arc(
        center,
        if near {
            2. * sine.asin()
        } else {
            PI - 2. * sine.asin()
        },
    )
}

/// Leaf arcs for |offset + rho(e1 cos(phi)+e2 sin(phi))| < radius.
fn ball_forbidden_arcs(offset: Vec3, radius: f64, circle: &ContactCircle) -> Result<IntervalSet> {
    ensure!(
        offset.iter().all(|v| v.is_finite()) && radius.is_finite() && radius >= 0.,
        "Invalid circle-ball predicate"
    );
    let (minimum, maximum, xy) = projected_distances(offset, circle)?;
    let extrema_guard = 64. * f64::EPSILON * (1. + radius + norm(offset) + circle.radius);
    ensure!(extrema_guard.is_finite(), "Nonfinite circle extrema guard");
    if (radius - minimum).abs() <= extrema_guard || (radius - maximum).abs() <= extrema_guard {
        return near_tangent_arcs(offset, radius, circle);
    }
    let perpendicular = xy[0].hypot(xy[1]);
    if circle.radius == 0. || perpendicular == 0. {
        let distance = if circle.radius == 0. {
            norm(offset)
        } else {
            minimum
        };
        return if distance < radius {
            whole()
        } else {
            Ok(IntervalSet::empty())
        };
    }
    if radius <= minimum {
        return Ok(IntervalSet::empty());
    }
    if radius > maximum {
        return whole();
    }
    let center = (-xy[1]).atan2(-xy[0]);
    if radius == maximum {
        return open_arc(center, PI);
    }
    // d² = d_min² + 4*rho*|v_perp|*sin²((phi-center)/2).
    // Factored square roots avoid subtracting two nearly equal squared lengths.
    // Near the farthest-point tangency, solve for the complementary small
    // allowed arc instead. Forming asin(1-epsilon) would lose one-ulp gaps.
    let near_minimum = radius - minimum <= maximum - radius;
    let numerator = if near_minimum {
        (radius - minimum).sqrt() * (radius + minimum).sqrt()
    } else {
        (maximum - radius).sqrt() * (maximum + radius).sqrt()
    };
    let sine = numerator / (2. * circle.radius.sqrt() * perpendicular.sqrt());
    ensure!(
        sine.is_finite() && sine > 0. && sine <= 1.,
        "Unrepresentable circle intersection root"
    );
    let half_width = if near_minimum {
        2. * sine.asin()
    } else {
        PI - 2. * sine.asin()
    };
    open_arc(center, half_width)
}

/// Exact leaf arcs for all fixed bodies. Periodic images, if wanted, must be
/// supplied explicitly by the caller. Inputs and tree topology are immutable.
pub fn circle_forbidden_arcs(
    moving_tree: &SphereTree,
    circle: &ContactCircle,
    orientation: [f64; 4],
    fixed_tree: &SphereTree,
    fixed_poses: &[Pose],
) -> Result<CircleGeometry> {
    validate_circle(circle)?;
    let origin = Pose {
        position: circle.center,
        orientation,
    };
    origin.validate()?;
    let moving = Placed::new(origin);
    let mut counts = CircleCounts::default();
    let mut intervals = Vec::new();
    let whole = whole()?;
    for pose in fixed_poses {
        pose.validate()?;
        let fixed = Placed::new(*pose);
        let mut pending = vec![(0, 0)];
        while let Some((i, j)) = pending.pop() {
            counts.node_pairs_visited += 1;
            let a = moving_tree.line_node(i);
            let b = fixed_tree.line_node(j);
            let ac = moving.apply(a.center);
            let bc = fixed.apply(b.center);
            let offset = sub(ac, bc);
            let guard = 1024.
                * f64::EPSILON
                * (1. + norm(ac) + norm(bc) + a.radius + b.radius + circle.radius);
            ensure!(guard.is_finite(), "Nonfinite circle BVH guard");
            let minimum = projected_distances(offset, circle)?.0;
            if minimum > a.radius + b.radius + guard {
                counts.node_pairs_pruned += 1;
                continue;
            }
            if let (Some(ai), Some(bi)) = (a.atom, b.atom) {
                counts.leaf_pairs_tested += 1;
                let aa = &moving_tree.shape.atoms[ai];
                let bb = &fixed_tree.shape.atoms[bi];
                let offset = sub(moving.apply(aa.center), fixed.apply(bb.center));
                let arcs = ball_forbidden_arcs(offset, aa.radius + bb.radius, circle)?;
                // Only an exact atomic full-circle result certifies this exit.
                if arcs == whole {
                    counts.full_cover_early_exit = true;
                    return Ok(CircleGeometry {
                        forbidden: whole,
                        allowed: IntervalSet::empty(),
                        counts,
                    });
                }
                intervals.extend_from_slice(arcs.intervals());
            } else if a.children.is_some() && (b.children.is_none() || a.radius >= b.radius) {
                let (left, right) = a.children.unwrap();
                pending.push((right, j));
                pending.push((left, j));
            } else {
                let (left, right) = b.children.unwrap();
                pending.push((i, right));
                pending.push((i, left));
            }
        }
    }
    let forbidden = IntervalSet::from_intervals(intervals)?;
    let allowed = whole.difference(&forbidden);
    Ok(CircleGeometry {
        forbidden,
        allowed,
        counts,
    })
}

/// A stable wrapped-Cauchy CDF difference over one centered, non-cut interval.
/// The atan2 difference uses sin(width/2), retaining small interval masses even
/// when subtracting two nearly equal CDF values would lose precision.
fn centered_wrapped_mass(lower: f64, width: f64, t: f64) -> Result<f64> {
    if width == 0. {
        return Ok(0.);
    }
    let (sa, ca) = (lower / 2.).sin_cos();
    let (sb, cb) = ((lower + width) / 2.).sin_cos();
    let scale = t.max(sa.abs()).max(sb.abs());
    let cross = (t / scale) * ((width / 2.).sin() / scale);
    let inner = (sa / scale) * (sb / scale) + (t / scale).powi(2) * ca * cb;
    let mass = cross.atan2(inner) / PI;
    ensure!(
        mass.is_finite() && mass >= 0. && mass <= 1.,
        "Invalid wrapped-Cauchy interval mass"
    );
    Ok(mass)
}

/// Complete (uniform + wrapped-Cauchy) mass of an interval in [0,2π].
/// Open/closed endpoints have the same continuous mass; singletons have zero.
pub fn azimuth_interval_mass(law: &AzimuthLaw, interval: &Interval) -> Result<f64> {
    ensure!(
        interval.lower.is_finite()
            && interval.upper.is_finite()
            && 0. <= interval.lower
            && interval.lower <= interval.upper
            && interval.upper <= TAU,
        "Azimuth interval is outside the canonical circle"
    );
    let checked = AzimuthLaw::new(law.mode, law.gamma, law.localized_probability)?;
    let width = interval.upper - interval.lower;
    if width == 0. {
        return Ok(0.);
    }
    if width == TAU {
        return Ok(1.);
    }
    if checked.localized_probability == 0. {
        return Ok(width / TAU);
    }
    let t = (checked.gamma / 2.).tanh();
    let relative = (interval.lower - checked.mode).rem_euclid(TAU);
    let lower = if relative >= PI {
        relative - TAU
    } else {
        relative
    };
    let before_cut = PI - lower;
    let wrapped = if width <= before_cut {
        centered_wrapped_mass(lower, width, t)?
    } else {
        centered_wrapped_mass(lower, before_cut, t)?
            + centered_wrapped_mass(-PI, width - before_cut, t)?
    };
    let mass = (1. - checked.localized_probability) * width / TAU
        + checked.localized_probability * wrapped;
    ensure!(
        mass.is_finite() && mass >= 0. && mass <= 1. + 8. * f64::EPSILON,
        "Invalid mixed azimuth interval mass"
    );
    Ok(mass)
}

/// Sum every disjoint interval's complete law; no truncation or mass floor.
pub fn azimuth_allowed_mass(law: &AzimuthLaw, allowed: &IntervalSet) -> Result<f64> {
    let mut sum = 0.;
    let mut correction = 0.;
    for interval in allowed.intervals() {
        let mass = azimuth_interval_mass(law, interval)?;
        let adjusted = mass - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    ensure!(
        sum.is_finite() && sum >= 0. && sum <= 1. + 32. * f64::EPSILON,
        "Invalid allowed azimuth mass"
    );
    Ok(sum)
}

#[cfg(test)]
mod tests;
