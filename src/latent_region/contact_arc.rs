//! Score a normalized candidate law without exposing an unimplemented draw.
//!
//! Every original radius/orientation branch keeps its probability. At fixed
//! radii, p(phi) is conditioned on the hard-free arcs if their mass exceeds a
//! declared floor; otherwise p(phi) is unchanged on the same circle. This is
//! a conditional normalization, never rejection/redrawing of outer variables.
//! Production importance and assembly kernels do not dispatch to this module.
use super::*;
use crate::{
    circle_geometry::{CircleGeometry, azimuth_allowed_mass, circle_forbidden_arcs},
    contact_distances::ContactCircle,
};
use std::collections::BTreeMap;

struct CachedGeometry {
    id: usize,
    circle: ContactCircle,
    geometry: CircleGeometry,
    cpu_seconds: f64,
}

/// Scoped to one exact pose and immutable physical environment. Effective
/// centers encode the ordered contact labels' frame; duplicate frames have
/// identical radii and basis at this query. No component's azimuth mass is
/// cached: means/covariances and hence Z can differ on the same circle.
struct ArcQueryCache<'a> {
    pose: Pose,
    tree: &'a SphereTree,
    fixed: &'a [Pose],
    entries: BTreeMap<[u64; 6], CachedGeometry>,
    requests: usize,
}

impl<'a> ArcQueryCache<'a> {
    fn new(pose: Pose, tree: &'a SphereTree, fixed: &'a [Pose]) -> Self {
        Self {
            pose,
            tree,
            fixed,
            entries: BTreeMap::new(),
            requests: 0,
        }
    }

    fn get(&mut self, frame: &ContactFrame, circle: ContactCircle) -> Result<&CachedGeometry> {
        self.requests += 1;
        let key = std::array::from_fn(|i| frame.centers[i / 3][i % 3].to_bits());
        if !self.entries.contains_key(&key) {
            let started = cpu_seconds();
            let geometry = circle_forbidden_arcs(
                self.tree,
                &circle,
                self.pose.orientation,
                self.tree,
                self.fixed,
            )?;
            self.entries.insert(
                key,
                CachedGeometry {
                    id: self.entries.len(),
                    circle,
                    geometry,
                    cpu_seconds: cpu_seconds() - started,
                },
            );
        }
        Ok(&self.entries[&key])
    }

    fn details(&self) -> Value {
        let mut entries: Vec<_> = self.entries.values().collect();
        entries.sort_by_key(|v| v.id);
        json!({"requests":self.requests,"distinct_circles":entries.len(),
            "geometry_cpu_seconds":entries.iter().map(|v|v.cpu_seconds).sum::<f64>(),
            "circles":entries.iter().map(|v|json!({"id":v.id,"circle":v.circle,
                "geometry":v.geometry,"cpu_seconds":v.cpu_seconds})).collect::<Vec<_>>()})
    }
}

impl ContactDistanceGuide {
    fn arc_density_details(
        &self,
        u: [f64; 6],
        inside: bool,
        log_volume: f64,
        chart: &Chart,
        minimum_arc_mass: f64,
        cache: &mut ArcQueryCache<'_>,
    ) -> Result<Value> {
        ensure!(
            minimum_arc_mass.is_finite() && minimum_arc_mass > 0. && minimum_arc_mass <= 1.,
            "Invalid fixed arc mass floor"
        );
        let baseline = self.base.log_density(u, inside, log_volume);
        if self.beta == 0. || self.base.alpha == 1. {
            return Ok(
                json!({"log_proposal_density":baseline,"old_distance_log_density":baseline,
                "baseline_log_density":baseline,"conditioning_disabled":true,"components":[]}),
            );
        }
        let raw = chart.coordinates(u);
        let (pose, _) = chart.decode(u);
        ensure!(pose == cache.pose, "Arc cache cannot cross query poses");
        let mut result = if inside {
            self.base.alpha.ln() - log_volume
        } else {
            f64::NEG_INFINITY
        };
        let mut old_result = result;
        let mut details = Vec::new();
        for (ci, component) in self.base.components.iter().enumerate() {
            let g = gaussian_log(component, u);
            let (angular, world_mean) = self.conditionals[ci].conditional(raw, chart);
            let frame = self.frame(ci, pose)?;
            let (mut h_old, mut h_new) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            let mut widths = Vec::new();
            for wi in 0..self.widths.len() {
                let mut d = json!({"width_index":wi,"width_A":self.widths[wi]});
                let (old_term, new_term) = if let Some(frame) = &frame {
                    let polygon = self.polygon(ci, wi, frame)?;
                    d["distance"] = json!(frame.distance);
                    d["polygon_area"] = json!(polygon.area);
                    if polygon.area <= self.minimum_area {
                        d["fallback"] = json!(true);
                        d["fallback_reason"] = json!("polygon_area");
                        (g, g)
                    } else {
                        d["fallback"] = json!(false);
                        let (mut old_translation, mut new_translation) =
                            (f64::NEG_INFINITY, f64::NEG_INFINITY);
                        if let Some(coordinates) = frame.encode(pose.position)? {
                            if polygon.contains(coordinates.radii) {
                                let circle = frame.circle(coordinates.radii)?;
                                let law = circle.azimuth_from_gaussian(
                                    world_mean,
                                    self.conditionals[ci].world_covariance,
                                    self.azimuth,
                                )?;
                                old_translation =
                                    frame.cartesian_log_density(pose.position, &polygon, law)?;
                                let geometry = cache.get(frame, circle)?;
                                let z = azimuth_allowed_mass(&law, &geometry.geometry.allowed)?;
                                let allowed = geometry.geometry.allowed.contains(coordinates.phi);
                                let fallback = z <= minimum_arc_mass;
                                new_translation = if fallback {
                                    old_translation
                                } else if allowed {
                                    old_translation - z.ln()
                                } else {
                                    f64::NEG_INFINITY
                                };
                                d["geometry_id"] = json!(geometry.id);
                                d["arc_mass"] = json!(z);
                                d["arc_fallback"] = json!(fallback);
                                d["query_phi_allowed"] = json!(allowed);
                                d["query_phi"] = json!(coordinates.phi);
                                d["azimuth_law"] = json!(law);
                            }
                        }
                        ensure!(
                            !old_translation.is_nan()
                                && old_translation != f64::INFINITY
                                && !new_translation.is_nan()
                                && new_translation != f64::INFINITY,
                            "Invalid arc translation density"
                        );
                        // Null represents zero density in the machine-readable trace.
                        d["old_translation_log_density"] =
                            json!(old_translation.is_finite().then_some(old_translation));
                        d["translation_log_density"] =
                            json!(new_translation.is_finite().then_some(new_translation));
                        (
                            chart.log_det + angular + old_translation,
                            chart.log_det + angular + new_translation,
                        )
                    }
                } else {
                    d["fallback"] = json!(true);
                    d["fallback_reason"] = json!("center_distance");
                    (g, g)
                };
                let width_logp = -(self.widths.len() as f64).ln();
                h_old = log_add(h_old, old_term + width_logp);
                h_new = log_add(h_new, new_term + width_logp);
                widths.push(d);
            }
            let outside = (1. - self.base.alpha).ln() + component.weight.ln();
            old_result = log_add(
                old_result,
                outside + log_add((1. - self.beta).ln() + g, self.beta.ln() + h_old),
            );
            result = log_add(
                result,
                outside + log_add((1. - self.beta).ln() + g, self.beta.ln() + h_new),
            );
            details.push(json!({"component":ci,"angular_log_density":angular,
                "conditional_world_mean":world_mean,"widths":widths}));
        }
        ensure!(
            result.is_finite() && old_result.is_finite(),
            "Nonfinite complete arc score"
        );
        Ok(
            json!({"log_proposal_density":result,"old_distance_log_density":old_result,
            "baseline_log_density":baseline,"components":details}),
        )
    }
}

pub struct ContactArcAuditOptions {
    pub config: PathBuf,
    pub region: PathBuf,
    pub importance_guides: Vec<PathBuf>,
    pub probes: PathBuf,
    pub minimum_arc_mass: f64,
    pub out: PathBuf,
}

/// Deterministic score-only audit. It cannot draw from, or register, the
/// candidate law with production sampling kernels.
pub fn run_contact_arc_audit(options: ContactArcAuditOptions) -> Result<Value> {
    ensure!(!options.out.exists(), "Fresh score output required");
    ensure!(
        !options.importance_guides.is_empty(),
        "At least one frozen distance guide required"
    );
    let config_raw = fs::read(&options.config)?;
    let mut cfg: DockingConfig = serde_json::from_slice(&config_raw)?;
    cfg.validate()?;
    ensure!(
        cfg.target_region.is_none(),
        "Unsupported extra docking target"
    );
    if cfg.shape.is_relative() {
        cfg.shape = options.config.parent().unwrap().join(&cfg.shape);
    }
    let shape_raw = fs::read(&cfg.shape)?;
    let shape_hash = hash_bytes(&shape_raw);
    let region_raw = fs::read(&options.region)?;
    let region: Value = serde_json::from_slice(&region_raw)?;
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
        "Score shape or scaffold mismatch"
    );
    ensure!(
        region["capture_center"] == json!(cfg.capture_center)
            && region["capture_radius"] == cfg.capture_radius
            && region["activity"] == cfg.reservoir_density
            && region["depletant_radius"] == cfg.depletant_radius
            && region["physical_metric"] == cfg.metadata,
        "Score physical definition changed"
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
        "Invalid score region"
    );
    ensure!(
        options.minimum_arc_mass.is_finite()
            && options.minimum_arc_mass > 0.
            && options.minimum_arc_mass <= 1.,
        "Invalid arc mass floor"
    );
    let chart = Chart::new(&region, &shape_hash, cfg.capture_radius, fixed)?;
    let tree = SphereTree::new(serde_json::from_slice(&shape_raw)?)?;
    let guide_raw = options
        .importance_guides
        .iter()
        .map(fs::read)
        .collect::<std::io::Result<Vec<_>>>()?;
    let guides = guide_raw
        .iter()
        .map(|data| {
            ContactDistanceGuide::from_bytes(
                data,
                &hash_bytes(&region_raw),
                &chart,
                inner,
                &cfg,
                &tree,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let probe_raw = fs::read(&options.probes)?;
    let probes = std::str::from_utf8(&probe_raw)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<serde_json::Result<Vec<_>>>()?;
    let mut ids = std::collections::BTreeSet::new();
    ensure!(
        probes
            .iter()
            .all(|v| !v["id"].is_null() && ids.insert(v["id"].to_string())),
        "Missing/duplicate score identities"
    );
    fs::create_dir_all(options.out.join("provenance"))?;
    for (name, data) in [
        ("config.json", config_raw.as_slice()),
        ("region.json", region_raw.as_slice()),
        ("shape.json", shape_raw.as_slice()),
        ("probes.jsonl", probe_raw.as_slice()),
    ] {
        fs::write(options.out.join("provenance").join(name), data)?;
    }
    for (i, data) in guide_raw.iter().enumerate() {
        fs::write(
            options
                .out
                .join("provenance")
                .join(format!("guide-{i}.json")),
            data,
        )?;
    }
    let bundle = include_bytes!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    fs::write(options.out.join("provenance/source-bundle.json"), bundle)?;
    let log_volume = 3. * PI.ln() + 6. * radius.ln() - 6_f64.ln();
    let manifest = json!({"schema":"contact-arc-density-score-v1","queries":probes.len(),"arms":guides.len(),
        "minimum_arc_mass":options.minimum_arc_mass,"config_sha256":hash_bytes(&config_raw),
        "region_sha256":hash_bytes(&region_raw),"shape_sha256":shape_hash,"probes_sha256":hash_bytes(&probe_raw),
        "guide_sha256":guide_raw.iter().map(|v|hash_bytes(v)).collect::<Vec<_>>(),
        "source_bundle_sha256":hash_bytes(bundle),"executable_sha256":hash_file(&std::env::current_exe()?)?,
        "new_pose_draws":0,"new_Poisson_clouds":0,"log_latent_ball_volume":log_volume,
        "scope":"Complete candidate density only; existing guides and production sampling are unchanged"});
    save(&options.out.join("manifest.json"), &manifest)?;
    let mut writer = BufWriter::new(File::create(options.out.join("scores.jsonl"))?);
    let mut journal = File::create(options.out.join("attempts.jsonl"))?;
    let placed: Vec<_> = physical.iter().copied().map(Placed::new).collect();
    let started = cpu_seconds();
    let mut total_geometry = 0.;
    let mut total_distinct = 0;
    let mut total_score = 0.;
    for (ordinal, probe) in probes.iter().enumerate() {
        writeln!(
            journal,
            "{}",
            json!({"ordinal":ordinal,"id":probe["id"],"state":"begin"})
        )?;
        journal.flush()?;
        let result: Result<Value> = (|| {
            let u: [f64; 6] = serde_json::from_value(probe["latent"].clone())?;
            ensure!(u.iter().all(|v| v.is_finite()), "Nonfinite probe");
            let radial = u.iter().map(|v| v * v).sum::<f64>().sqrt();
            let inside = radial <= radius;
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
                "Score chart inverse failed"
            );
            let mut cache = ArcQueryCache::new(pose, &tree, &physical);
            let mut arms = Vec::new();
            for (i, guide) in guides.iter().enumerate() {
                let tick = cpu_seconds();
                let mut score = guide.arc_density_details(
                    u,
                    inside,
                    log_volume,
                    &chart,
                    options.minimum_arc_mass,
                    &mut cache,
                )?;
                let elapsed = cpu_seconds() - tick;
                score["arm_index"] = json!(i);
                score["score_cpu_seconds"] = json!(elapsed);
                total_score += elapsed;
                arms.push(score);
            }
            let cache_details = cache.details();
            total_geometry += cache_details["geometry_cpu_seconds"].as_f64().unwrap();
            total_distinct += cache.entries.len();
            let hard_valid = !placed.iter().any(|p| tree.overlaps(&Placed::new(pose), p));
            Ok(
                json!({"ordinal":ordinal,"id":probe["id"],"latent":u,"raw_coordinates":chart.coordinates(u),
                "pose":pose,"log_physical_jacobian":jacobian,"backmap_error":error,
                "hard_valid":hard_valid,"shell_valid":inside,"capture_valid":cfg.contains(pose),
                "arms":arms,"geometry_cache":cache_details}),
            )
        })();
        match result {
            Ok(row) => {
                writeln!(writer, "{row}")?;
                writer.flush()?;
            }
            Err(error) => {
                let failure = json!({"complete":false,"ordinal":ordinal,"id":probe["id"],
                    "error":format!("{error:#}"),"manifest":manifest});
                let _ = save(&options.out.join("failure.json"), &failure);
                eprintln!("{failure}");
                return Err(error);
            }
        }
    }
    let summary = json!({"complete":true,"manifest":manifest,"queries":probes.len(),"arms":guides.len(),
        "distinct_circles":total_distinct,"geometry_cpu_seconds":total_geometry,"score_cpu_seconds":total_score,
        "cpu_seconds":cpu_seconds()-started,"scores_sha256":hash_file(&options.out.join("scores.jsonl"))?,
        "attempts_sha256":hash_file(&options.out.join("attempts.jsonl"))?});
    save(&options.out.join("summary.json"), &summary)?;
    Ok(summary)
}

#[cfg(test)]
#[path = "contact_arc_tests.rs"]
mod tests;
