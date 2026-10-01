//! Exact one-coordinate conditional Gaussian guides for cooperative contacts.
//!
//! Raw chart coordinates are used: changing a whitened latent coordinate would
//! generally couple translation and rotation. The other five raw coordinates
//! retain their complete Gaussian marginal, including empty-line fallbacks.
use super::*;
use crate::line_geometry::{Interval, IntervalSet, translation_intervals};
use rand::distr::Open01;
use std::cell::RefCell;

#[link(name = "m")]
unsafe extern "C" {
    fn erfc(x: f64) -> f64;
}

/// Positive interval probability, with reflected tails and a close-endpoint
/// quadrature branch. No subtraction of two almost equal central CDFs.
pub(crate) fn normal_mass(lo: f64, hi: f64) -> Result<f64> {
    ensure!(
        lo.is_finite() && hi.is_finite() && lo <= hi,
        "Invalid normal interval"
    );
    if lo == hi {
        return Ok(0.);
    }
    let width = hi - lo;
    let middle = lo + 0.5 * width;
    let mass = if width * (1. + middle.abs()) < 0.1 {
        let nodes = [
            0.1834346424956498,
            0.5255324099163290,
            0.7966664774136267,
            0.9602898564975363,
        ];
        let weights = [
            0.3626837833783620,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763,
        ];
        let half = 0.5 * width;
        half / (2. * PI).sqrt()
            * nodes
                .into_iter()
                .zip(weights)
                .map(|(n, w)| {
                    let a = middle - half * n;
                    let b = middle + half * n;
                    w * ((-0.5 * a * a).exp() + (-0.5 * b * b).exp())
                })
                .sum::<f64>()
    } else {
        let sf = |z: f64| unsafe { 0.5 * erfc(z / 2_f64.sqrt()) };
        if lo >= 0. {
            sf(lo) - sf(hi)
        } else if hi <= 0. {
            sf(-hi) - sf(-lo)
        } else {
            1. - sf(hi) - sf(-lo)
        }
    };
    ensure!(
        mass.is_finite() && (0. ..=1.).contains(&mass),
        "Invalid conditional Gaussian mass; never silently fall back"
    );
    Ok(mass)
}

struct ConditionalNormal {
    mean: [f64; 6],
    lower: [[f64; 6]; 6],
    order: [usize; 6],
}
impl ConditionalNormal {
    fn new(component: &ImportanceComponent, chart: &Chart, axis: usize) -> Result<Self> {
        let mean = chart.coordinates(component.mean);
        let product: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6)
                    .map(|k| chart.lower[i][k] * component.lower[k][j])
                    .sum()
            })
        });
        let indices: Vec<_> = (0..6).filter(|&i| i != axis).chain([axis]).collect();
        let order: [usize; 6] = indices.try_into().unwrap();
        let covariance: [[f64; 6]; 6] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..6)
                    .map(|k| product[order[i]][k] * product[order[j]][k])
                    .sum()
            })
        });
        let mut lower = [[0.; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let v = covariance[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
                lower[i][j] = if i == j {
                    ensure!(
                        v.is_finite() && v > 0.,
                        "Unresolved conditional covariance pivot"
                    );
                    v.sqrt()
                } else {
                    v / lower[j][j]
                };
            }
        }
        Ok(Self { mean, lower, order })
    }
    fn conditional(&self, x: [f64; 6]) -> (f64, f64) {
        let mut z = [0.; 5];
        for i in 0..5 {
            z[i] = (x[self.order[i]]
                - self.mean[self.order[i]]
                - (0..i).map(|j| self.lower[i][j] * z[j]).sum::<f64>())
                / self.lower[i][i];
        }
        (
            self.mean[self.order[5]] + (0..5).map(|i| self.lower[5][i] * z[i]).sum::<f64>(),
            self.lower[5][5],
        )
    }
}

pub(super) struct ContactLineGuide {
    pub(super) base: ImportanceGuide,
    hard_free_only: bool,
    pose_coordinates: bool,
    axes: Vec<usize>,
    widths: Vec<f64>,
    beta: f64,
    minimum_mass: f64,
    contacts: Vec<usize>,
    conditionals: Vec<Vec<ConditionalNormal>>,
    tree: SphereTree,
    fixed: Vec<Pose>,
    capture_center: Vec3,
    capture_radius: f64,
    pub(super) last_draw: RefCell<Value>,
}

fn raw_to_latent(chart: &Chart, x: [f64; 6]) -> [f64; 6] {
    let mut u = [0.; 6];
    for i in 0..6 {
        u[i] = (x[i] - chart.mean[i] - (0..i).map(|j| chart.lower[i][j] * u[j]).sum::<f64>())
            / chart.lower[i][i];
    }
    u
}

/// Finite line/ball chord; tangent sets have zero one-dimensional measure.
fn ball_chord<const N: usize>(
    origin: [f64; N],
    direction: [f64; N],
    radius: f64,
) -> Option<[f64; 2]> {
    let aa = direction.iter().map(|d| d * d).sum::<f64>();
    let center = -origin
        .iter()
        .zip(direction)
        .map(|(o, d)| o * d)
        .sum::<f64>()
        / aa;
    let minimum = (0..N)
        .map(|i| (origin[i] + center * direction[i]).powi(2))
        .sum::<f64>();
    let residual = radius * radius - minimum;
    if !(aa > 0. && aa.is_finite() && residual > 0.) {
        return None;
    }
    let half = (residual / aa).sqrt();
    let range = [center - half, center + half];
    (range[0].is_finite() && range[1].is_finite() && range[0] < range[1]).then_some(range)
}

impl ContactLineGuide {
    pub(super) fn from_bytes(
        raw: &[u8],
        region_hash: &str,
        chart: &Chart,
        inner: f64,
        cfg: &DockingConfig,
        tree: &SphereTree,
    ) -> Result<Self> {
        ensure!(
            inner == 0.,
            "Contact-line guide currently requires a complete latent ball"
        );
        let mut data: Value = serde_json::from_slice(raw)?;
        let pose_coordinates = data["schema"] == "defensive-hard-free-pose-line-guide-v1";
        let hard_free_only = pose_coordinates || data["schema"] == "defensive-hard-free-line-guide-v1";
        ensure!(
            hard_free_only || data["schema"] == "defensive-contact-line-guide-v1",
            "Wrong contact-line schema"
        );
        let axis_key = if pose_coordinates { "raw_pose_axes" } else { "raw_translation_axes" };
        let other_key = if pose_coordinates { "raw_translation_axes" } else { "raw_pose_axes" };
        ensure!(data.get(other_key).is_none(), "Ambiguous coordinate-axis fields");
        let axes: Vec<usize> = serde_json::from_value(data[axis_key].clone())?;
        let (widths, contacts): (Vec<f64>, Vec<usize>) = if hard_free_only {
            ensure!(
                data.get("contact_widths_A").is_none()
                    && data.get("contact_neighbor_indices").is_none(),
                "Hard-free guide must not specify contact widths or neighbor labels"
            );
            // One set per axis. Zero is an internal sentinel only: no contact
            // query or additional contact predicate is applied in this mode.
            (vec![0.], Vec::new())
        } else {
            (
                serde_json::from_value(data["contact_widths_A"].clone())?,
                serde_json::from_value(data["contact_neighbor_indices"].clone())?,
            )
        };
        let beta = data["conditional_probability"]
            .as_f64()
            .context("Missing conditional probability")?;
        let minimum_mass = data["minimum_conditional_mass"]
            .as_f64()
            .context("Missing conditional mass floor")?;
        ensure!(
            !axes.is_empty()
                && axes.iter().all(|&a| a < if pose_coordinates { 6 } else { 3 })
                && axes.iter().enumerate().all(|(i, a)| !axes[..i].contains(a)),
            "Invalid/duplicate raw chart axes"
        );
        ensure!(
            hard_free_only
                || (!widths.is_empty() && widths.iter().all(|w| w.is_finite() && *w > 0.)),
            "Invalid contact widths"
        );
        ensure!(
            hard_free_only
                || (!contacts.is_empty()
                    && contacts.iter().all(|&i| i < cfg.fixed_poses.len())
                    && contacts
                        .iter()
                        .enumerate()
                        .all(|(i, a)| !contacts[..i].contains(a))),
            "Invalid contact-neighbor labels"
        );
        ensure!(
            beta.is_finite()
                && (0. ..=1.).contains(&beta)
                && minimum_mass.is_finite()
                && minimum_mass > 0.
                && minimum_mass < 1.,
            "Invalid conditional controls"
        );
        data["schema"] = json!("defensive-latent-shell-guide-v1");
        for key in [
            "raw_translation_axes",
            "raw_pose_axes",
            "contact_widths_A",
            "contact_neighbor_indices",
            "conditional_probability",
            "minimum_conditional_mass",
        ] {
            data.as_object_mut().unwrap().remove(key);
        }
        let base = ImportanceGuide::from_bytes(&serde_json::to_vec(&data)?, region_hash)?;
        let conditionals = axes
            .iter()
            .map(|&axis| {
                base.components
                    .iter()
                    .map(|c| ConditionalNormal::new(c, chart, axis))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            base,
            hard_free_only,
            pose_coordinates,
            axes,
            widths,
            beta,
            minimum_mass,
            contacts,
            conditionals,
            tree: tree.clone(),
            fixed: cfg.fixed_poses.clone(),
            capture_center: cfg.capture_center,
            capture_radius: cfg.capture_radius,
            last_draw: RefCell::new(Value::Null),
        })
    }

    /// A single geometry set per axis/width, shared by ALL mixture components.
    fn intervals(
        &self,
        x: [f64; 6],
        chart: &Chart,
        outer: f64,
        axis: usize,
    ) -> Result<(Vec<IntervalSet>, Value)> {
        let mut raw = x;
        raw[axis] = 0.;
        let u0 = raw_to_latent(chart, raw);
        let mut du = [0.; 6];
        for i in 0..6 {
            du[i] = ((if i == axis { 1. } else { 0. })
                - (0..i).map(|j| chart.lower[i][j] * du[j]).sum::<f64>())
                / chart.lower[i][i];
        }
        let (origin, _) = chart.decode(u0);
        let direction = matvec(
            rotation(chart.fixed.orientation),
            std::array::from_fn(|i| if i == axis { 1. } else { 0. }),
        );
        let empty = || vec![IntervalSet::default(); self.widths.len()];
        let Some(mut segment) = ball_chord(u0, du, outer) else {
            return Ok((empty(), json!({"axis":axis,"empty_reason":"no_R4_chord"})));
        };
        if axis >= 3 {
            // A raw Cayley coordinate rotates the body at a fixed center.
            // Capture is therefore one Boolean, never a zero-direction chord.
            let center = chart.fixed.apply(add(chart.anchor_position, [raw[0], raw[1], raw[2]]));
            if norm(sub(center, self.capture_center)) > self.capture_radius {
                return Ok((empty(), json!({"axis":axis,"empty_reason":"no_capture_at_fixed_center"})));
            }
            let family = crate::cayley_axis_geometry::CayleyAxis {
                world_center: center,
                chart_orientation: chart.fixed.orientation,
                anchor_rotation: chart.anchor_rotation,
                fixed_cayley: [raw[3] / chart.ell, raw[4] / chart.ell, raw[5] / chart.ell],
                axis: axis - 3,
                length_scale: chart.ell,
            };
            let core = crate::cayley_axis_geometry::cayley_axis_intervals(
                &self.tree, &family, &self.tree, &self.fixed, segment,
            )?;
            let detail = json!({"axis":axis,"coordinate_kind":"raw-scaled-Cayley",
                "segment":segment,"world_center":center,"fixed_cayley":family.fixed_cayley,
                "length_scale":chart.ell,"core_counts":core.counts,
                "hard_free_intervals":core.hard_free.intervals(),
                "empty_reason":if core.hard_free.length() == 0. { Some("no_positive_hard_free_length") } else { None }});
            return Ok((vec![core.hard_free], detail));
        }
        let Some(capture) = ball_chord(
            sub(origin.position, self.capture_center),
            direction,
            self.capture_radius,
        ) else {
            return Ok((
                empty(),
                json!({"axis":axis,"empty_reason":"no_capture_chord"}),
            ));
        };
        segment[0] = segment[0].max(capture[0]);
        segment[1] = segment[1].min(capture[1]);
        if segment[0] >= segment[1] {
            return Ok((
                empty(),
                json!({"axis":axis,"empty_reason":"disjoint_chords"}),
            ));
        }
        let core = translation_intervals(
            &self.tree,
            origin,
            direction,
            &self.tree,
            &self.fixed,
            segment,
            None,
            None,
        )?;
        let hard_free =
            IntervalSet::from_intervals(vec![Interval::closed(segment[0], segment[1])?])?
                .difference(&core.hard_overlap);
        if hard_free.length() == 0. {
            return Ok((
                empty(),
                json!({"axis":axis,"empty_reason":"no_positive_hard_free_length",
                "segment":segment,"core_counts":core.counts,"origin":origin,"direction":direction}),
            ));
        }
        if self.hard_free_only {
            let detail = json!({"axis":axis,"segment":segment,"core_counts":core.counts,
                "origin":origin,"direction":direction,"hard_free_intervals":hard_free.intervals()});
            return Ok((vec![hard_free], detail));
        }
        let mut sets = Vec::new();
        let mut details = Vec::new();
        for &width in &self.widths {
            let mut joint = hard_free.clone();
            let mut counts = Vec::new();
            for &index in &self.contacts {
                let query = translation_intervals(
                    &self.tree,
                    origin,
                    direction,
                    &self.tree,
                    &[self.fixed[index]],
                    segment,
                    Some(width),
                    None,
                )?;
                joint = joint.intersection(query.contact_overlap.as_ref().unwrap());
                counts.push(json!(query.counts));
            }
            details.push(
                json!({"width_A":width,"intervals":joint.intervals(),"contact_queries":counts}),
            );
            sets.push(joint);
        }
        Ok((
            sets,
            json!({"axis":axis,"segment":segment,"core_counts":core.counts,
            "origin":origin,"direction":direction,"widths":details}),
        ))
    }

    fn masses(intervals: &IntervalSet, mean: f64, sigma: f64) -> Result<(Vec<f64>, f64)> {
        ensure!(
            mean.is_finite() && sigma.is_finite() && sigma > 0.,
            "Invalid conditional Normal parameters"
        );
        let masses = intervals
            .intervals()
            .iter()
            .map(|r| normal_mass((r.lower - mean) / sigma, (r.upper - mean) / sigma))
            .collect::<Result<Vec<_>>>()?;
        let total = masses.iter().sum::<f64>();
        ensure!(
            total.is_finite() && total >= 0. && total <= 1. + 1e-12,
            "Invalid union Gaussian mass"
        );
        Ok((masses, total))
    }

    pub(super) fn density_details(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
    ) -> Result<(f64, Value)> {
        self.density_trace(u, inside, volume, chart, outer, true)
    }

    /// Production audit trace: retain every axis's exact interval union, but
    /// omit per-component diagnostics and nondeterministic timing fields.
    pub(super) fn density_compact(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
    ) -> Result<(f64, Value)> {
        self.density_trace(u, inside, volume, chart, outer, false)
    }

    /// World-pose queries can have genuine zero conditional support. Establish
    /// that support from intervals and fallback predicates, independently of
    /// floating-point Gaussian values. Geometry is traversed only once; this
    /// check reuses the saved intervals and repeats only cheap Normal algebra.
    /// The existing regional/diagnostic methods and their arithmetic are intact.
    pub(super) fn density_compact_checked(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
    ) -> Result<(f64, Value, bool)> {
        ensure!(
            self.hard_free_only,
            "Checked vessel scorer requires hard-free law"
        );
        let (density, detail) = self.density_compact(u, inside, volume, chart, outer)?;
        let mut active = vec![self.beta < 1.; self.base.components.len()];
        if self.base.alpha < 1. && self.beta == 1. {
            let x = chart.coordinates(u);
            let axes = detail["axes"]
                .as_array()
                .context("Missing hard-free axes")?;
            ensure!(axes.len() == self.axes.len(), "Incomplete hard-free axes");
            for (ai, &axis) in self.axes.iter().enumerate() {
                let intervals = IntervalSet::from_intervals(serde_json::from_value(
                    axes[ai]["hard_free_intervals"].clone(),
                )?)?;
                let allowed = intervals.contains(x[axis]);
                for (ci, normal) in self.conditionals[ai].iter().enumerate() {
                    let (mean, sigma) = normal.conditional(x);
                    let (_, mass) = Self::masses(&intervals, mean, sigma)?;
                    active[ci] |= mass <= self.minimum_mass || allowed;
                }
            }
        }
        if self.base.alpha < 1. {
            for (component, &present) in self.base.components.iter().zip(&active) {
                if !present {
                    continue;
                }
                let mut z = [0.; 6];
                for i in 0..6 {
                    z[i] = (u[i]
                        - component.mean[i]
                        - (0..i).map(|j| component.lower[i][j] * z[j]).sum::<f64>())
                        / component.lower[i][i];
                }
                let log_g = component.log_normalizer - 0.5 * z.iter().map(|v| v * v).sum::<f64>();
                ensure!(
                    log_g.is_finite(),
                    "Unrepresentable active Gaussian log density, not a support zero"
                );
            }
        }
        let positive =
            (inside && self.base.alpha > 0.) || (self.base.alpha < 1. && active.iter().any(|v| *v));
        ensure!(
            (positive && density.is_finite()) || (!positive && density == f64::NEG_INFINITY),
            "Conditional density disagrees with analytic support"
        );
        Ok((density, detail, !positive))
    }

    fn density_trace(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
        full_trace: bool,
    ) -> Result<(f64, Value)> {
        let base = self.base.log_density(u, inside, volume);
        if self.beta == 0. || self.base.alpha == 1. {
            return Ok((base, json!({"conditioning_disabled":true})));
        }
        let x = chart.coordinates(u);
        let gaussian_logs: Vec<_> = self
            .base
            .components
            .iter()
            .map(|c| {
                let mut z = [0.; 6];
                for k in 0..6 {
                    z[k] = (u[k] - c.mean[k] - (0..k).map(|j| c.lower[k][j] * z[j]).sum::<f64>())
                        / c.lower[k][k];
                }
                c.log_normalizer - 0.5 * z.iter().map(|a| a * a).sum::<f64>()
            })
            .collect();
        let compose = |factors: &[f64], branches: f64| {
            let mut result = if inside {
                self.base.alpha.ln() - volume
            } else {
                f64::NEG_INFINITY
            };
            for (i, c) in self.base.components.iter().enumerate() {
                let correction = (1. - self.beta) + self.beta * factors[i] / branches;
                if correction > 0. {
                    result = log_add(
                        result,
                        (1. - self.base.alpha).ln()
                            + c.weight.ln()
                            + gaussian_logs[i]
                            + correction.ln(),
                    );
                }
            }
            result
        };
        let mut factor = vec![0.; self.base.components.len()];
        let mut geometry = Vec::new();
        let mut fallback = 0;
        for (ai, &axis) in self.axes.iter().enumerate() {
            let geometry_start = full_trace.then(cpu_seconds);
            let (sets, mut detail) = self.intervals(x, chart, outer, axis)?;
            let geometry_cpu = geometry_start.map(|start| cpu_seconds() - start);
            let mut axis_factors = vec![0.; self.base.components.len()];
            let mut components = Vec::new();
            for (ci, normal) in self.conditionals[ai].iter().enumerate() {
                let (mean, sigma) = normal.conditional(x);
                for intervals in &sets {
                    let (_, mass) = Self::masses(intervals, mean, sigma)?;
                    let is_fallback = mass <= self.minimum_mass;
                    let allowed = intervals.contains(x[axis]);
                    let multiplier = if is_fallback {
                        fallback += 1;
                        1.
                    } else if allowed {
                        1. / mass
                    } else {
                        0.
                    };
                    factor[ci] += multiplier;
                    axis_factors[ci] += multiplier;
                    if self.hard_free_only && full_trace {
                        components.push(json!({"component":ci,"gaussian_log_density":gaussian_logs[ci],
                            "conditional_mean":mean,"conditional_sigma":sigma,"conditional_mass":mass,
                            "fallback":is_fallback,"query_coordinate_allowed":allowed}));
                    }
                }
            }
            if self.hard_free_only {
                detail["hard_free_intervals"] = json!(sets[0].intervals());
                detail["axis_log_proposal_density"] = json!(compose(&axis_factors, 1.));
                if full_trace {
                    detail["components"] = json!(components);
                    detail["geometry_cpu_seconds"] = json!(geometry_cpu.unwrap());
                }
            }
            geometry.push(detail);
        }
        let branches = (self.axes.len() * self.widths.len()) as f64;
        let result = compose(&factor, branches);
        Ok((
            result,
            json!({"raw_coordinates":x,"axes":geometry,
            "fallback_component_branches":fallback,"component_branches":branches as usize*self.base.components.len(),
            "baseline_log_density":base}),
        ))
    }

    pub(super) fn draw(
        &self,
        rng: &mut StdRng,
        chart: &Chart,
        outer: f64,
        inner: f64,
        fraction: f64,
    ) -> Result<([f64; 6], f64, Option<usize>, Option<bool>)> {
        let (original, radius, component) = self.base.draw(rng, outer, inner, fraction)?;
        *self.last_draw.borrow_mut() = json!({"conditional":false});
        if self.hard_free_only {
            self.last_draw.borrow_mut()["original_latent"] = json!(original);
        }
        let Some(component) = component else {
            return Ok((original, radius, None, None));
        };
        if self.beta == 0. || rng.random::<f64>() >= self.beta {
            return Ok((original, radius, Some(component), None));
        }
        let ai = rng.random_range(0..self.axes.len());
        let axis = self.axes[ai];
        let wi = rng.random_range(0..self.widths.len());
        let mut x = chart.coordinates(original);
        let (sets, mut geometry) = self.intervals(x, chart, outer, axis)?;
        let intervals = &sets[wi];
        let (mean, sigma) = self.conditionals[ai][component].conditional(x);
        let (masses, total) = Self::masses(intervals, mean, sigma)?;
        let fallback = total <= self.minimum_mass;
        *self.last_draw.borrow_mut() = json!({"conditional":true,"axis":axis,"width_index":wi,
            "width_A":self.widths[wi],"component":component,"conditional_mean":mean,"conditional_sigma":sigma,
            "conditional_mass":total,"fallback":fallback,"geometry":geometry});
        if self.hard_free_only {
            geometry["hard_free_intervals"] = json!(intervals.intervals());
            let mut trace = self.last_draw.borrow_mut();
            trace.as_object_mut().unwrap().remove("width_index");
            trace.as_object_mut().unwrap().remove("width_A");
            trace["original_latent"] = json!(original);
            trace["geometry"] = geometry;
        }
        if fallback {
            return Ok((original, radius, Some(component), Some(true)));
        }
        let draw: f64 = rng.sample(Open01);
        let mut probability = draw * total;
        let mut chosen = None;
        for (interval, mass) in intervals.intervals().iter().zip(masses) {
            if mass <= 0. {
                continue;
            }
            chosen = Some((interval, mass));
            if probability < mass {
                break;
            }
            probability -= mass;
        }
        let (interval, mass) = chosen.context("Positive mass without conditional interval")?;
        let lo = (interval.lower - mean) / sigma;
        let within: f64 = rng.sample(Open01);
        let target = within * mass;
        ensure!(
            target > 0. && target < mass,
            "Unresolved interior conditional probability"
        );
        // Bisect directly in raw translation to avoid rounding a rescaled
        // standardized root outside the geometric interval. Never clip.
        let (mut left, mut right) = (interval.lower, interval.upper);
        for _ in 0..80 {
            let mid = left + 0.5 * (right - left);
            if mid == left || mid == right {
                break;
            }
            if normal_mass(lo, (mid - mean) / sigma)? < target {
                left = mid;
            } else {
                right = mid;
            }
        }
        x[axis] = left + 0.5 * (right - left);
        ensure!(
            intervals.contains(x[axis]),
            "Conditional inverse landed outside the retained interval"
        );
        {
            let mut trace = self.last_draw.borrow_mut();
            if self.hard_free_only {
                trace["uniform_interval_selection"] = json!(draw);
            }
            trace["selected_interval"] = json!(interval);
            trace["selected_interval_mass"] = json!(mass);
            trace["uniform_within_interval"] = json!(within);
            trace["returned_raw_coordinate"] = json!(x[axis]);
            trace["inverse_probability_error"] =
                json!((normal_mass(lo, (x[axis] - mean) / sigma)? / mass - within).abs());
        }
        let u = raw_to_latent(chart, x);
        ensure!(
            u.iter().all(|v| v.is_finite()),
            "Nonfinite conditioned coordinates"
        );
        let radius = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok((u, radius, Some(component), Some(false)))
    }
}

/// Diagnostic only: no Poisson clouds, physical weights or native classifier.
pub struct AuditOptions {
    pub config: PathBuf,
    pub region: PathBuf,
    pub importance_guide: PathBuf,
    pub out: PathBuf,
    pub samples: u64,
    pub seed: u64,
    pub probes: Option<PathBuf>,
}

fn audited_query<T>(
    out: &Path,
    journal: &mut File,
    ordinal: u64,
    kind: &str,
    id: &Value,
    work: impl FnOnce() -> Result<T>,
) -> Result<T> {
    writeln!(
        journal,
        "{}",
        json!({"ordinal":ordinal,"kind":kind,"id":id,"state":"begin"})
    )?;
    journal.flush()?;
    match work() {
        Ok(value) => Ok(value),
        Err(error) => {
            save(
                &out.join("failure.json"),
                &json!({"complete":false,"ordinal":ordinal,
                "kind":kind,"id":id,"error":format!("{error:#}")}),
            )?;
            Err(error)
        }
    }
}

/// Share the diagnostic bookkeeping without changing either sampling law.
enum DiagnosticGuide {
    Line(ContactLineGuide),
    Distances(super::contact_distance::ContactDistanceGuide),
}
impl DiagnosticGuide {
    fn from_bytes(
        raw: &[u8],
        hash: &str,
        chart: &Chart,
        inner: f64,
        cfg: &DockingConfig,
        tree: &SphereTree,
    ) -> Result<Self> {
        let value: Value = serde_json::from_slice(raw)?;
        match value["schema"].as_str() {
            Some("defensive-contact-line-guide-v1" | "defensive-hard-free-line-guide-v1" | "defensive-hard-free-pose-line-guide-v1") => {
                Ok(Self::Line(ContactLineGuide::from_bytes(
                    raw, hash, chart, inner, cfg, tree,
                )?))
            }
            Some("defensive-contact-distance-guide-v1") => Ok(Self::Distances(
                super::contact_distance::ContactDistanceGuide::from_bytes(
                    raw, hash, chart, inner, cfg, tree,
                )?,
            )),
            _ => anyhow::bail!("Unsupported contact audit guide"),
        }
    }
    fn base(&self) -> &ImportanceGuide {
        match self {
            Self::Line(g) => &g.base,
            Self::Distances(g) => &g.base,
        }
    }
    fn widths(&self) -> &[f64] {
        match self {
            Self::Line(g) => &g.widths,
            Self::Distances(g) => &g.widths,
        }
    }
    fn contacts(&self) -> &[usize] {
        match self {
            Self::Line(g) => &g.contacts,
            Self::Distances(g) => &g.contacts,
        }
    }
    fn last_draw(&self) -> &RefCell<Value> {
        match self {
            Self::Line(g) => &g.last_draw,
            Self::Distances(g) => &g.last_draw,
        }
    }
    fn schema(&self) -> &'static str {
        match self {
            Self::Line(g) if g.pose_coordinates => "hard-free-pose-line-guide-audit-v1",
            Self::Line(g) if g.hard_free_only => "hard-free-line-guide-audit-v1",
            Self::Line(_) => "contact-line-guide-audit-v1",
            Self::Distances(_) => "contact-distance-guide-audit-v1",
        }
    }
    fn role(&self) -> &'static str {
        match self {
            Self::Line(g) if g.pose_coordinates => "hard-free-pose-line-proposal-audit",
            Self::Line(g) if g.hard_free_only => "hard-free-line-proposal-audit",
            Self::Line(_) => "contact-line-proposal-audit",
            Self::Distances(_) => "contact-distance-proposal-audit",
        }
    }
    fn draw(
        &self,
        rng: &mut StdRng,
        chart: &Chart,
        outer: f64,
        inner: f64,
        fraction: f64,
    ) -> Result<([f64; 6], f64, Option<usize>, Option<bool>)> {
        match self {
            Self::Line(g) => g.draw(rng, chart, outer, inner, fraction),
            Self::Distances(g) => g.draw(rng, chart, outer, inner, fraction),
        }
    }
    fn density_details(
        &self,
        u: [f64; 6],
        inside: bool,
        volume: f64,
        chart: &Chart,
        outer: f64,
    ) -> Result<(f64, Value)> {
        match self {
            Self::Line(g) => g.density_details(u, inside, volume, chart, outer),
            Self::Distances(g) => g.density_details(u, inside, volume, chart, outer),
        }
    }
}

pub fn run_audit(options: AuditOptions) -> Result<Value> {
    ensure!(!options.out.exists(), "Audit output must be new");
    let config_raw = fs::read(&options.config)?;
    let mut cfg: DockingConfig = serde_json::from_slice(&config_raw)?;
    cfg.validate()?;
    ensure!(
        cfg.target_region.is_none(),
        "Unsupported docking-only target restriction"
    );
    if cfg.shape.is_relative() {
        cfg.shape = options.config.parent().unwrap().join(&cfg.shape);
    }
    let region_raw = fs::read(&options.region)?;
    let region: Value = serde_json::from_slice(&region_raw)?;
    let guide_raw = fs::read(&options.importance_guide)?;
    let shape_raw = fs::read(&cfg.shape)?;
    let shape_hash = hash_bytes(&shape_raw);
    let fixed: Pose = serde_json::from_value(region["fixed_neighbor"].clone())?;
    let physical: Vec<Pose> = if let Some(v) = region.get("physical_fixed_neighbors") {
        serde_json::from_value(v.clone())?
    } else {
        vec![fixed]
    };
    ensure!(
        region["shape_sha256"] == shape_hash
            && cfg.fixed_poses == physical
            && physical.contains(&fixed),
        "Audit shape or physical scaffold mismatch"
    );
    ensure!(
        region["capture_center"] == json!(cfg.capture_center)
            && region["capture_radius"] == cfg.capture_radius
            && region["activity"] == cfg.reservoir_density
            && region["depletant_radius"] == cfg.depletant_radius
            && region["physical_metric"] == cfg.metadata,
        "Audit physical definition mismatch"
    );
    let radius = region["mahalanobis_radius"]
        .as_f64()
        .context("Missing region radius")?;
    let inner = region
        .get("minimum_mahalanobis_radius")
        .and_then(Value::as_f64)
        .unwrap_or(0.);
    ensure!(
        radius.is_finite() && radius > 0. && inner == 0.,
        "Invalid audit region"
    );
    let chart = Chart::new(&region, &shape_hash, cfg.capture_radius, fixed)?;
    let tree = SphereTree::new(serde_json::from_slice(&shape_raw)?)?;
    let guide = DiagnosticGuide::from_bytes(
        &guide_raw,
        &hash_bytes(&region_raw),
        &chart,
        inner,
        &cfg,
        &tree,
    )?;
    let log_volume = 3. * PI.ln() + 6. * radius.ln() - 6_f64.ln();
    let placed: Vec<_> = physical.iter().copied().map(Placed::new).collect();
    let inflated = guide
        .widths()
        .iter()
        .map(|&width| {
            let mut shape = tree.shape.clone();
            for atom in &mut shape.atoms {
                atom.radius += width / 2.;
            }
            SphereTree::new(shape)
        })
        .collect::<Result<Vec<_>>>()?;
    fs::create_dir_all(options.out.join("provenance"))?;
    for (name, bytes) in [
        ("config.json", config_raw.as_slice()),
        ("region.json", region_raw.as_slice()),
        ("importance-guide.json", guide_raw.as_slice()),
        ("shape.json", shape_raw.as_slice()),
    ] {
        fs::write(options.out.join("provenance").join(name), bytes)?;
    }
    let bundle = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(options.out.join("provenance/source-bundle.json"), bundle)?;
    let probe_raw = options.probes.as_ref().map(fs::read).transpose()?;
    if let Some(bytes) = &probe_raw {
        fs::write(options.out.join("provenance/probes.jsonl"), bytes)?;
    }
    let mut probe_items = Vec::<Value>::new();
    let mut probe_ids = std::collections::BTreeSet::new();
    if let Some(bytes) = &probe_raw {
        for line in std::str::from_utf8(bytes)?.lines() {
            let item: Value = serde_json::from_str(line)?;
            ensure!(
                !item["id"].is_null() && probe_ids.insert(item["id"].to_string()),
                "Missing or duplicated saved query ID"
            );
            probe_items.push(item);
        }
    }
    let manifest = json!({"schema":guide.schema(),"samples":options.samples,"seed":options.seed,
        "scope":"Proposal and saved-pose geometry/density only; no Poisson sampling, physical mass or native classification",
        "config_sha256":hash_bytes(&config_raw),"region_sha256":hash_bytes(&region_raw),
        "guide_sha256":hash_bytes(&guide_raw),"shape_sha256":shape_hash,
        "probes_sha256":probe_raw.as_ref().map(|b|hash_bytes(b)),
        "executable_sha256":hash_file(&std::env::current_exe()?)?,"source_bundle_sha256":hash_bytes(bundle.as_bytes()),
        "log_latent_ball_volume":log_volume,"latent_radius":radius,"physical_jobs":0});
    save(&options.out.join("manifest.json"), &manifest)?;
    let mut writer = BufWriter::new(File::create(options.out.join("samples.jsonl"))?);
    let mut journal = File::create(options.out.join("attempts.jsonl"))?;
    let started = cpu_seconds();
    let mut draw_cpu = 0.;
    let mut density_cpu = 0.;
    let mut nonzero = 0;
    let mut conditional = 0;
    let mut fallback = 0;
    let mut maximum_backmap = 0_f64;
    let evaluate = |u: [f64; 6],
                    kind: &str,
                    id: Value,
                    draw: Value,
                    draw_elapsed: f64|
     -> Result<(Value, f64)> {
        let radial = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inside = radial <= radius;
        let tick = cpu_seconds();
        let (log_density, details) =
            guide.density_details(u, inside, log_volume, &chart, radius)?;
        let elapsed = cpu_seconds() - tick;
        ensure!(
            !log_density.is_nan() && log_density != f64::INFINITY,
            "Invalid scored proposal density"
        );
        let (pose, jacobian) = chart.decode(u);
        pose.validate()?;
        let backmap = chart.encode(pose)?;
        let error = u
            .iter()
            .zip(backmap)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f64, f64::max);
        ensure!(
            error < 2e-7 * (1. + radial.max(radius)),
            "Audit chart inverse failed"
        );
        let body = Placed::new(pose);
        let hard_valid = !placed.iter().any(|p| tree.overlaps(&body, p));
        let contacts: Vec<_> = inflated
            .iter()
            .map(|shape| {
                guide
                    .contacts()
                    .iter()
                    .map(|&i| shape.overlaps(&body, &placed[i]))
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok((
            json!({"kind":kind,"id":id,"latent":u,"latent_radius":radial,"raw_coordinates":chart.coordinates(u),
            "pose":pose,"shell_valid":inside,"capture_valid":cfg.contains(pose),"hard_valid":hard_valid,
            "log_proposal_density":log_density,"baseline_log_density":guide.base().log_density(u,inside,log_volume),
            "log_physical_jacobian":jacobian,"backmap_error":error,"width_contacts":contacts,
            "draw":draw,"density_details":details,"draw_cpu_seconds":draw_elapsed,"density_cpu_seconds":elapsed}),
            elapsed,
        ))
    };
    for index in 0..options.samples {
        let (row, cpu, elapsed) = audited_query(
            &options.out,
            &mut journal,
            index,
            "fresh",
            &json!(index),
            || {
                let mut rng = stream(options.seed, index, 0, guide.role());
                let tick = cpu_seconds();
                let (u, _, component, _) = guide.draw(&mut rng, &chart, radius, 0., 1.)?;
                let elapsed = cpu_seconds() - tick;
                let mut info = guide.last_draw().borrow().clone();
                info["component"] = json!(component);
                let (row, cpu) = evaluate(u, "fresh", json!(index), info, elapsed)?;
                ensure!(
                    row["log_proposal_density"]
                        .as_f64()
                        .is_some_and(f64::is_finite),
                    "Generated draw has zero or invalid proposal density"
                );
                Ok((row, cpu, elapsed))
            },
        )?;
        draw_cpu += elapsed;
        conditional += u64::from(row["draw"]["conditional"] == true);
        fallback += u64::from(row["draw"]["fallback"] == true);
        density_cpu += cpu;
        nonzero += u64::from(
            row["hard_valid"] == true && row["shell_valid"] == true && row["capture_valid"] == true,
        );
        maximum_backmap = maximum_backmap.max(row["backmap_error"].as_f64().unwrap());
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    writer.flush()?;
    let fresh_cpu = cpu_seconds() - started;
    let mut probe_writer = BufWriter::new(File::create(options.out.join("probes.jsonl"))?);
    let mut probes = 0;
    let probe_start = cpu_seconds();
    for (index, item) in probe_items.iter().enumerate() {
        let ordinal = options
            .samples
            .checked_add(index as u64)
            .context("Audit ordinal overflow")?;
        let (row, _) = audited_query(
            &options.out,
            &mut journal,
            ordinal,
            "probe",
            &item["id"],
            || {
                let u: [f64; 6] = serde_json::from_value(item["latent"].clone())?;
                ensure!(u.iter().all(|v| v.is_finite()), "Nonfinite saved query");
                evaluate(u, "probe", item["id"].clone(), Value::Null, 0.)
            },
        )?;
        serde_json::to_writer(&mut probe_writer, &row)?;
        probe_writer.write_all(b"\n")?;
        probe_writer.flush()?;
        probes += 1;
    }
    probe_writer.flush()?;
    let summary = json!({"complete":true,"samples":options.samples,"probes":probes,"hard_capture_shell_valid":nonzero,
        "conditioned_draws":conditional,"fallback_draws":fallback,"maximum_backmap_error":maximum_backmap,
        "draw_cpu_seconds":draw_cpu,"density_cpu_seconds":density_cpu,"fresh_total_cpu_seconds":fresh_cpu,
        "probe_total_cpu_seconds":cpu_seconds()-probe_start,"manifest":manifest,
        "samples_sha256":hash_file(&options.out.join("samples.jsonl"))?,"probes_sha256":hash_file(&options.out.join("probes.jsonl"))?,
        "attempts_sha256":hash_file(&options.out.join("attempts.jsonl"))?});
    save(&options.out.join("summary.json"), &summary)?;
    Ok(summary)
}

#[cfg(test)]
#[path = "contact_line_tests.rs"]
mod tests;
