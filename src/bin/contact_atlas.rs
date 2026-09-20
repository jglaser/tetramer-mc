//! Generate a native-blind contact atlas using only the supplied rigid shape.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::Serialize;
use serde_json::json;
use std::{fs, io::Write, path::PathBuf, time::Instant};
use tetramer_mc::{
    geometry::{Placed, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
    simulation::hash_bytes,
};

#[derive(Parser, Debug, Serialize)]
#[command(
    about = "Build Gaussian relative-pose charts at geometry-only radial contacts; no native inputs"
)]
struct Args {
    #[arg(long)]
    shape: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 96)]
    count: usize,
    #[arg(long, default_value_t = 2026091907)]
    seed: u64,
    /// Outward displacement along the separation ray, in shape length units.
    #[arg(long, default_value_t = 0.25)]
    gap: f64,
    /// Independent Cartesian translation standard deviation, in length units.
    #[arg(long, default_value_t = 1.0)]
    translation_width: f64,
    /// Per-axis small-angle standard deviation, in degrees.
    #[arg(long, default_value_t = 3.0)]
    angle_width_degrees: f64,
    /// Cayley scale; set this to a native atlas's value before blending models.
    #[arg(long)]
    angular_length: Option<f64>,
    /// Disable linearized rolling covariance about the witness surface contact.
    #[arg(long)]
    uncoupled: bool,
}

fn normal_direction(rng: &mut StdRng) -> Vec3 {
    loop {
        let v: Vec3 = std::array::from_fn(|_| StandardNormal.sample(rng));
        let n = norm(v);
        if n.is_finite() && n > 0. {
            return v.map(|x| x / n);
        }
    }
}

fn haar_orientation(rng: &mut StdRng) -> [f64; 4] {
    loop {
        let q: [f64; 4] = std::array::from_fn(|_| StandardNormal.sample(rng));
        let n = q.iter().fold(0_f64, |a, v| a.hypot(*v));
        if n.is_finite() && n > 0. {
            return q.map(|x| x / n);
        }
    }
}

fn covariance(
    translation: f64,
    angle: f64,
    angular_length: f64,
    lever: Vec3,
    rolling: bool,
) -> [[f64; 6]; 6] {
    let angular_std = angular_length * angle / 2.;
    let cross = [
        [0., -lever[2], lever[1]],
        [lever[2], 0., -lever[0]],
        [-lever[1], lever[0], 0.],
    ];
    // ξ and ζ are independent unit normals. Latent displacement is
    // [σ_t ξ + (lever × ω), ell*ω/2], ω = angle*ζ.
    // This full-rank linear map guarantees a positive definite covariance.
    let mut factor = [[0.; 6]; 6];
    for i in 0..3 {
        factor[i][i] = translation;
        factor[i + 3][i + 3] = angular_std;
        if rolling {
            for j in 0..3 {
                factor[i][j + 3] = angle * cross[i][j];
            }
        }
    }
    std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..6).map(|k| factor[i][k] * factor[j][k]).sum())
    })
}

fn write_new(path: &std::path::Path, value: &impl Serialize) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "Create {} without overwriting existing data",
                path.display()
            )
        })?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.count > 0 && args.count <= 100_000,
        "count must lie in 1..=100000"
    );
    ensure!(
        args.gap.is_finite() && args.gap >= 0.,
        "gap must be finite and nonnegative"
    );
    ensure!(
        [args.translation_width, args.angle_width_degrees]
            .iter()
            .all(|x| x.is_finite() && *x > 0.),
        "Gaussian widths must be positive and finite"
    );
    ensure!(!args.out.exists(), "atlas output already exists");
    let provenance_path = args.out.with_extension("provenance.json");
    ensure!(
        provenance_path != args.out && !provenance_path.exists(),
        "provenance output already exists or aliases the atlas"
    );
    let started = Instant::now();
    let raw_shape = fs::read(&args.shape)?;
    let shape_sha = hash_bytes(&raw_shape);
    let tree = SphereTree::new(serde_json::from_slice::<Shape>(&raw_shape)?)?;
    let mean: Vec3 = std::array::from_fn(|d| {
        tree.shape.atoms.iter().map(|a| a.center[d]).sum::<f64>() / tree.shape.atoms.len() as f64
    });
    let rms = (tree
        .shape
        .atoms
        .iter()
        .map(|a| dot(sub(a.center, mean), sub(a.center, mean)))
        .sum::<f64>()
        / tree.shape.atoms.len() as f64)
        .sqrt();
    let maximum_atom = tree.shape.atoms.iter().map(|a| a.radius).fold(0., f64::max);
    let angular_length = args.angular_length.unwrap_or(2. * rms.max(maximum_atom));
    ensure!(
        angular_length.is_finite() && angular_length > 0.,
        "angular length must be positive and finite"
    );
    let angle = args.angle_width_degrees.to_radians();
    let numerical_gap = 4096. * f64::EPSILON * (1. + tree.bound);
    let effective_gap = args.gap + numerical_gap;
    let fixed = Placed::new(Pose {
        position: [0.; 3],
        orientation: [1., 0., 0., 0.],
    });
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut anchors = Vec::with_capacity(args.count);
    let mut covariances = Vec::with_capacity(args.count);
    let mut witnesses = Vec::with_capacity(args.count);
    let mut attempts = 0_usize;
    while anchors.len() < args.count {
        attempts += 1;
        ensure!(
            attempts <= 1000 * args.count,
            "too many rays miss the hard core; generated {} contacts",
            anchors.len()
        );
        let orientation = haar_orientation(&mut rng);
        let direction = normal_direction(&mut rng);
        let Some(contact) = tree.outermost_radial_contact(rotation(orientation), direction)? else {
            continue;
        };
        let r = rotation(contact.orientation);
        let contact_position = scale(contact.direction, contact.distance);
        let position = scale(contact.direction, contact.distance + effective_gap);
        let moving = Placed::new(Pose {
            position,
            orientation: contact.orientation,
        });
        ensure!(
            !tree.overlaps(&moving, &fixed),
            "generated contact gap is not hard-free; no atlas written"
        );
        let a = &tree.shape.atoms[contact.moving_atom];
        let b = &tree.shape.atoms[contact.fixed_atom];
        let delta = sub(add(contact_position, matvec(r, a.center)), b.center);
        let length = norm(delta);
        ensure!(
            length.is_finite() && length > 0.,
            "invalid radial witness normal"
        );
        let normal = delta.map(|x| x / length);
        let contact_point = add(b.center, scale(normal, b.radius));
        let lever = sub(contact_point, contact_position);
        covariances.push(covariance(
            args.translation_width,
            angle,
            angular_length,
            lever,
            !args.uncoupled,
        ));
        anchors.push(json!({"position":position,"rotation":r}));
        witnesses.push(json!({"radial_contact":contact,"surface_point":contact_point,"mobile_surface_lever":lever}));
    }
    let source_bundle_text = include_str!(concat!(env!("OUT_DIR"), "/source-bundle.json"));
    let source_bundle: serde_json::Value = serde_json::from_str(source_bundle_text)?;
    let construction = json!({
        "kind":"geometry-only-outermost-ray-contact-v1","native_information":false,
        "native_information_scope":"No inter-tetramer native geometry or templates are used. The supplied rigid shape may itself contain a native tetramer.",
        "shape_path":fs::canonicalize(&args.shape)?,"shape_sha256":shape_sha,
        "seed":args.seed,"rng":"StdRng; rand version pinned by archived Cargo.lock",
        "count":args.count,"attempted_rays":attempts,"missed_rays":attempts-args.count,
        "requested_gap":args.gap,"numerical_outward_gap":numerical_gap,"effective_gap":effective_gap,
        "translation_width":args.translation_width,"small_angle_width_degrees":args.angle_width_degrees,
        "angular_length":angular_length,"covariance":if args.uncoupled {"independent translation and Cayley coordinates"} else {"linearized contact-pivot rolling plus independent translation"},
        "prior_weights":"equal; no native registration or depletion energy used",
        "source_bundle_sha256":hash_bytes(source_bundle_text.as_bytes()),
        "provenance_path":provenance_path.file_name().unwrap(),
        "scope":"Finite proposal atlas, generated independently of production. Conditional on a ray hitting the core, its last atomic exit supplies a contact; no inverse-CDF normalization is used in MC. This does not enumerate internal interlocking pockets, estimate basin masses, or establish native accessibility.",
    });
    let model = json!({"coordinate_convention":"anchor-body-relative","shape_sha256":shape_sha,
        "angular_length":angular_length,"anchors":anchors,"means":vec![[0_f64;6];args.count],
        "covariances":covariances,"weights":vec![1./args.count as f64;args.count],
        "dfs":vec![Option::<f64>::None;args.count],"construction":construction,
        "contact_witnesses":witnesses});
    // Validate through the same normalized Gaussian parser used by production,
    // including every full-covariance Cholesky factor and proper chart rotation.
    FrozenRelativePoseProposal::from_json_str_open(
        &serde_json::to_string(&model)?,
        [4. * tree.bound.max(1.); 3],
        0.1,
        &shape_sha,
    )?;
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    write_new(
        &provenance_path,
        &json!({"schema":1,"arguments":args,"construction":construction,
        "source_bundle":source_bundle,"physical_shape":serde_json::from_slice::<serde_json::Value>(&raw_shape)?,
        "executable_sha256":hash_bytes(&fs::read(std::env::current_exe()?)?)}),
    )?;
    write_new(&args.out, &model)?;
    println!(
        "{}",
        json!({"atlas":args.out,"provenance":provenance_path,"components":args.count,
        "attempted_rays":attempts,"wall_seconds":started.elapsed().as_secs_f64(),
        "atomic_pair_tests":witnesses.iter().map(|w|w["radial_contact"]["atomic_pairs_tested"].as_u64().unwrap()).sum::<u64>()})
    );
    Ok(())
}
