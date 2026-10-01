//! Normalized translation proposals in two distances and a circle azimuth.
//!
//! At fixed contact centers, the radius polygon is sampled uniformly in
//! `(r1,r2)`. Every azimuth law integrates to one conditionally on those radii,
//! including the Gaussian-projection fallback to uniform. The Cartesian
//! Jacobian is `r1*r2/D`, so the density is `D*p_phi/(area*r1*r2)`.
//!
//! This module does not reject whole-particle overlaps or condition on a
//! physical domain. Orientation-dependent distance/area floors and their
//! normalized fallback laws belong to the caller. No hidden retry or geometry
//! clamp is performed. Coincident centers and invalid arithmetic are errors;
//! axis points have an explicit measure-zero inverse convention.

use crate::math::{Mat3, Vec3, add, dot, matvec, norm, scale, sub};
use anyhow::{Result, ensure};
use serde::Serialize;
use std::f64::consts::{PI, TAU};

fn finite(values: &[f64]) -> bool {
    values.iter().all(|x| x.is_finite())
}

fn open_uniform(u: f64) -> Result<()> {
    ensure!(
        u.is_finite() && u > 0. && u < 1.,
        "Expected an open-interval uniform"
    );
    Ok(())
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0].mul_add(b[1], -a[1] * b[0])
}

/// A convex radius rectangle clipped by all three triangle inequalities.
#[derive(Clone, Debug, Serialize)]
pub struct RadiusPolygon {
    pub distance: f64,
    pub bounds: [[f64; 2]; 2],
    pub vertices: Vec<[f64; 2]>,
    pub area: f64,
    offset_vertices: Vec<[f64; 2]>,
    triangle_areas: Vec<f64>,
}

impl RadiusPolygon {
    pub fn new(distance: f64, bounds: [[f64; 2]; 2]) -> Result<Self> {
        ensure!(
            distance.is_finite() && distance >= 0.,
            "Invalid center distance"
        );
        for b in bounds {
            ensure!(
                finite(&b) && b[0] >= 0. && b[1] >= b[0],
                "Invalid radius bounds"
            );
        }
        let [lo1, lo2] = [bounds[0][0], bounds[1][0]];
        let [w1, w2] = [bounds[0][1] - lo1, bounds[1][1] - lo2];
        let mut vertices = vec![[0., 0.], [w1, 0.], [w1, w2], [0., w2]];
        // Work relative to the rectangle corner, avoiding shoelace subtraction
        // between large absolute radii when the widths are small.
        for (a, b) in [
            ([1., 1.], lo1 + lo2 - distance),
            ([-1., 1.], distance - lo1 + lo2),
            ([1., -1.], distance + lo1 - lo2),
        ] {
            ensure!(b.is_finite(), "Overflow in triangle half-plane");
            vertices = clip(vertices, a, b)?;
        }
        let mut triangle_areas = Vec::new();
        if vertices.len() >= 3 {
            for i in 1..vertices.len() - 1 {
                let a = std::array::from_fn(|j| vertices[i][j] - vertices[0][j]);
                let b = std::array::from_fn(|j| vertices[i + 1][j] - vertices[0][j]);
                let area = cross2(a, b) / 2.;
                ensure!(
                    area.is_finite() && area >= 0.,
                    "Invalid oriented polygon area"
                );
                triangle_areas.push(area);
            }
        }
        let area: f64 = triangle_areas.iter().sum();
        ensure!(area.is_finite(), "Nonfinite polygon area");
        Ok(Self {
            distance,
            bounds,
            area,
            vertices: vertices.iter().map(|p| [lo1 + p[0], lo2 + p[1]]).collect(),
            offset_vertices: vertices,
            triangle_areas,
        })
    }

    pub fn contains(&self, radii: [f64; 2]) -> bool {
        finite(&radii)
            && radii
                .iter()
                .zip(self.bounds)
                .all(|(r, b)| *r >= b[0] && *r <= b[1])
            && radii[0] + radii[1] >= self.distance
            && radii[0] - radii[1] <= self.distance
            && radii[1] - radii[0] <= self.distance
    }

    /// Uniform polygon sample. Uniforms select a triangle and its barycentric
    /// coordinates; no RNG, endpoint rejection, or retry is hidden here.
    pub fn sample_radii(&self, uniforms: [f64; 3]) -> Result<[f64; 2]> {
        for u in uniforms {
            open_uniform(u)?;
        }
        ensure!(self.area > 0., "Cannot sample a zero-area radius polygon");
        let target = uniforms[0] * self.area;
        let mut cumulative = 0.;
        let mut selected = None;
        for (index, area) in self.triangle_areas.iter().enumerate() {
            cumulative += area;
            if *area > 0. && (target < cumulative || index + 1 == self.triangle_areas.len()) {
                selected = Some(index + 1);
                break;
            }
        }
        let i = selected.ok_or_else(|| anyhow::anyhow!("Triangle selection failed"))?;
        let s = uniforms[1].sqrt();
        let weights = [1. - s, s * (1. - uniforms[2]), s * uniforms[2]];
        let corners = [
            self.offset_vertices[0],
            self.offset_vertices[i],
            self.offset_vertices[i + 1],
        ];
        let result = std::array::from_fn(|j| {
            self.bounds[j][0] + (0..3).map(|k| weights[k] * corners[k][j]).sum::<f64>()
        });
        ensure!(finite(&result), "Nonfinite sampled radii");
        Ok(result)
    }
}

/// Sutherland-Hodgman clipping against a.x+b >= 0, in local coordinates.
fn clip(vertices: Vec<[f64; 2]>, a: [f64; 2], b: f64) -> Result<Vec<[f64; 2]>> {
    if vertices.is_empty() {
        return Ok(vertices);
    }
    let signed = |p: [f64; 2]| a[0].mul_add(p[0], a[1] * p[1] + b);
    let mut output = Vec::new();
    let mut previous = *vertices.last().unwrap();
    let mut prev_value = signed(previous);
    for current in vertices {
        let value = signed(current);
        ensure!(
            value.is_finite() && prev_value.is_finite(),
            "Nonfinite clip predicate"
        );
        if (prev_value >= 0.) != (value >= 0.) {
            let fraction = prev_value / (prev_value - value);
            ensure!(
                fraction.is_finite() && (0. ..=1.).contains(&fraction),
                "Invalid clip intersection"
            );
            output.push(std::array::from_fn(|j| {
                (1. - fraction) * previous[j] + fraction * current[j]
            }));
        }
        if value >= 0. {
            output.push(current);
        }
        previous = current;
        prev_value = value;
    }
    output.dedup();
    if output.len() > 1 && output.first() == output.last() {
        output.pop();
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ContactFrame {
    pub centers: [Vec3; 2],
    pub distance: f64,
    pub axis: Vec3,
    pub basis: [Vec3; 2],
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ContactCircle {
    pub radii: [f64; 2],
    pub center: Vec3,
    pub radius: f64,
    pub axis: Vec3,
    pub basis: [Vec3; 2],
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DistanceCoordinates {
    pub radii: [f64; 2],
    pub phi: f64,
    pub transverse_radius: f64,
}

impl ContactFrame {
    /// The reference axis is the Cartesian axis with smallest absolute axial
    /// component (first index on ties). e1 = e × reference, e2 = e × e1.
    pub fn new(c1: Vec3, c2: Vec3) -> Result<Self> {
        ensure!(finite(&c1) && finite(&c2), "Nonfinite contact center");
        let difference = sub(c2, c1);
        let distance = norm(difference);
        ensure!(
            distance.is_finite() && distance > 0.,
            "Coincident or nonfinite contact centers"
        );
        let axis = difference.map(|v| v / distance);
        let mut index = 0;
        for i in 1..3 {
            if axis[i].abs() < axis[index].abs() {
                index = i;
            }
        }
        let mut reference = [0.; 3];
        reference[index] = 1.;
        let first = cross(axis, reference);
        let first = scale(first, 1. / norm(first));
        let second = cross(axis, first);
        Ok(Self {
            centers: [c1, c2],
            distance,
            axis,
            basis: [first, second],
        })
    }

    /// Stable sorted-Heron radius; negative triangle factors are errors, never
    /// silently clamped. Exact tangencies return radius zero.
    pub fn circle(&self, radii: [f64; 2]) -> Result<ContactCircle> {
        ensure!(
            finite(&radii) && radii.iter().all(|r| *r >= 0.),
            "Invalid contact radii"
        );
        let mut sides = [radii[0], radii[1], self.distance];
        sides.sort_by(|a, b| b.total_cmp(a));
        let [a, b, c] = sides;
        let factors = [a + (b + c), c - (a - b), c + (a - b), a + (b - c)];
        ensure!(
            finite(&factors) && factors.iter().all(|v| *v >= 0.),
            "Radii do not form a finite triangle"
        );
        // Pair square roots before multiplication: no squared-length
        // subtraction and much less underflow near a tangent circle.
        let radius = ((factors[0].sqrt() * factors[1].sqrt()) / (2. * self.distance))
            * (factors[2].sqrt() * factors[3].sqrt());
        let height =
            (self.distance + (radii[0] - radii[1]) * ((radii[0] + radii[1]) / self.distance)) / 2.;
        let center = add(self.centers[0], scale(self.axis, height));
        ensure!(
            radius.is_finite() && height.is_finite() && finite(&center),
            "Nonfinite contact circle"
        );
        Ok(ContactCircle {
            radii,
            center,
            radius,
            axis: self.axis,
            basis: self.basis,
        })
    }

    /// Axis points have no unique azimuth and return `None`. Other invalid
    /// arithmetic returns `Err`, rather than masquerading as a geometric zero.
    pub fn encode(&self, point: Vec3) -> Result<Option<DistanceCoordinates>> {
        ensure!(finite(&point), "Nonfinite Cartesian point");
        let relative = sub(point, self.centers[0]);
        let radii = [norm(relative), norm(sub(point, self.centers[1]))];
        let xy = self.basis.map(|v| dot(relative, v));
        let transverse_radius = xy[0].hypot(xy[1]);
        ensure!(
            finite(&radii) && finite(&xy) && transverse_radius.is_finite(),
            "Nonfinite inverse distances"
        );
        if transverse_radius == 0. {
            return Ok(None);
        }
        let phi = xy[1].atan2(xy[0]).rem_euclid(TAU);
        Ok(Some(DistanceCoordinates {
            radii,
            phi,
            transverse_radius,
        }))
    }

    /// `law` must be the same radius-dependent azimuth law used for generation.
    /// Density is zero outside the polygon and on the null set of axis points.
    pub fn cartesian_log_density(
        &self,
        point: Vec3,
        polygon: &RadiusPolygon,
        law: impl std::borrow::Borrow<AzimuthLaw>,
    ) -> Result<f64> {
        ensure!(
            polygon.distance == self.distance && polygon.area > 0.,
            "Mismatched or zero-area polygon"
        );
        let Some(coordinates) = self.encode(point)? else {
            return Ok(f64::NEG_INFINITY);
        };
        if !polygon.contains(coordinates.radii) {
            return Ok(f64::NEG_INFINITY);
        }
        ensure!(
            coordinates.radii.iter().all(|r| *r > 0.),
            "Singular positive-density radii"
        );
        Ok(self.distance.ln()
            - polygon.area.ln()
            - coordinates.radii[0].ln()
            - coordinates.radii[1].ln()
            + law.borrow().log_density(coordinates.phi)?)
    }
}

impl ContactCircle {
    pub fn point(&self, phi: f64) -> Result<Vec3> {
        ensure!(phi.is_finite(), "Nonfinite azimuth");
        let (sin, cos) = phi.sin_cos();
        let result = add(
            self.center,
            scale(
                add(scale(self.basis[0], cos), scale(self.basis[1], sin)),
                self.radius,
            ),
        );
        ensure!(finite(&result), "Nonfinite sampled translation");
        Ok(result)
    }

    pub fn azimuth_from_gaussian(
        &self,
        mean: Vec3,
        covariance: Mat3,
        parameters: AzimuthParameters,
    ) -> Result<AzimuthLaw> {
        parameters.validate()?;
        ensure!(
            finite(&mean) && covariance.iter().all(|row| finite(row)),
            "Nonfinite Gaussian parameters"
        );
        if parameters.localized_probability == 0. {
            return Ok(AzimuthLaw::uniform());
        }
        let delta = sub(mean, self.center);
        let projected = self.basis.map(|v| dot(delta, v));
        let length = projected[0].hypot(projected[1]);
        ensure!(length.is_finite(), "Nonfinite Gaussian projection");
        if length <= parameters.projection_floor {
            return Ok(AzimuthLaw::uniform());
        }
        let mode = projected[1].atan2(projected[0]);
        let tangent = add(
            scale(self.basis[0], -mode.sin()),
            scale(self.basis[1], mode.cos()),
        );
        let variance = dot(tangent, matvec(covariance, tangent));
        ensure!(
            variance.is_finite() && variance > 0.,
            "Nonpositive tangent Gaussian variance"
        );
        let gamma = (variance.sqrt() / self.radius.max(parameters.rho_floor))
            .clamp(parameters.gamma_min, parameters.gamma_max);
        AzimuthLaw::new(mode, gamma, parameters.localized_probability)
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct AzimuthParameters {
    pub rho_floor: f64,
    pub projection_floor: f64,
    pub gamma_min: f64,
    pub gamma_max: f64,
    pub localized_probability: f64,
}

impl AzimuthParameters {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            finite(&[
                self.rho_floor,
                self.projection_floor,
                self.gamma_min,
                self.gamma_max,
                self.localized_probability
            ]) && self.rho_floor > 0.
                && self.projection_floor >= 0.
                && self.gamma_min > 0.
                && self.gamma_max >= self.gamma_min
                && (0. ..1.).contains(&self.localized_probability),
            "Invalid frozen azimuth parameters"
        );
        Ok(())
    }
}

/// Wrapped Cauchy plus an explicit uniform defensive azimuth component.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct AzimuthLaw {
    pub mode: f64,
    pub gamma: f64,
    pub localized_probability: f64,
}

impl AzimuthLaw {
    pub fn new(mode: f64, gamma: f64, localized_probability: f64) -> Result<Self> {
        ensure!(
            finite(&[mode, gamma, localized_probability])
                && gamma > 0.
                && (gamma / 2.).tanh() > 0.
                && (0. ..1.).contains(&localized_probability),
            "Invalid wrapped-Cauchy parameters"
        );
        Ok(Self {
            mode: mode.rem_euclid(TAU),
            gamma,
            localized_probability,
        })
    }

    pub fn uniform() -> Self {
        Self {
            mode: 0.,
            gamma: 1.,
            localized_probability: 0.,
        }
    }

    pub fn wrapped_inverse(&self, u: f64) -> Result<f64> {
        open_uniform(u)?;
        let (sin, cos) = (PI * (u - 0.5)).sin_cos();
        let delta = 2. * ((self.gamma / 2.).tanh() * sin).atan2(cos);
        Ok((self.mode + delta).rem_euclid(TAU))
    }

    /// CDF with its cut at `mode-pi`, rather than at Cartesian azimuth zero.
    pub fn wrapped_cdf(&self, phi: f64) -> Result<f64> {
        ensure!(phi.is_finite(), "Nonfinite azimuth");
        let delta = (phi - self.mode + PI).rem_euclid(TAU) - PI;
        let (sin, cos) = (delta / 2.).sin_cos();
        Ok(0.5 + sin.atan2((self.gamma / 2.).tanh() * cos) / PI)
    }

    pub fn sample(&self, uniforms: [f64; 2]) -> Result<f64> {
        for u in uniforms {
            open_uniform(u)?;
        }
        if uniforms[0] < self.localized_probability {
            self.wrapped_inverse(uniforms[1])
        } else {
            Ok(TAU * uniforms[1])
        }
    }

    pub fn log_density(&self, phi: f64) -> Result<f64> {
        ensure!(phi.is_finite(), "Nonfinite azimuth");
        if self.localized_probability == 0. {
            return Ok(-TAU.ln());
        }
        let delta = (phi - self.mode).rem_euclid(TAU);
        let (sin, cos) = (delta / 2.).sin_cos();
        let t = (self.gamma / 2.).tanh();
        let wrapped_log = t.ln() - TAU.ln() - 2. * sin.hypot(t * cos).ln();
        let a = self.localized_probability.ln() + wrapped_log;
        let b = (-self.localized_probability).ln_1p() - TAU.ln();
        let high = a.max(b);
        Ok(high + ((a - high).exp() + (b - high).exp()).ln())
    }

    pub fn density(&self, phi: f64) -> Result<f64> {
        Ok(self.log_density(phi)?.exp())
    }
}

#[cfg(test)]
mod tests;
