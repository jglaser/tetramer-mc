//! Atomic all-pairs references for the two-node radial BVH search and atlas CLI.
use anyhow::Result;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde_json::{Value, json};
use std::{fs, process::Command};
use tetramer_mc::{
    geometry::{Atom, Placed, Shape, SphereTree},
    math::*,
    proposal::FrozenRelativePoseProposal,
    simulation::hash_file,
};

fn tree(atoms: Vec<Atom>) -> SphereTree {
    SphereTree::new(Shape {
        name: "radial reference".into(),
        volume: 0.,
        atoms,
    })
    .unwrap()
}
fn pose(position: Vec3, orientation: [f64; 4]) -> Pose {
    Pose {
        position,
        orientation,
    }
}

/// Independent reference: axial projection and closest line approach, without
/// tree bounds, traversal, stable-product roots, or candidate pruning.
fn brute(tree: &SphereTree, r: Mat3, direction: Vec3) -> Option<f64> {
    let n = norm(direction);
    let direction = direction.map(|v| v / n);
    let a = dot(direction, direction);
    let mut best = None;
    for moving in &tree.shape.atoms {
        for fixed in &tree.shape.atoms {
            let d = sub(matvec(r, moving.center), fixed.center);
            let axial = dot(d, direction) / a;
            let perpendicular = sub(d, scale(direction, axial));
            let gap = (moving.radius + fixed.radius).powi(2) - dot(perpendicular, perpendicular);
            if gap > 0. {
                let exit = -axial + (gap / a).sqrt();
                if exit > 0. {
                    best = Some(best.map_or(exit, |x: f64| x.max(exit)));
                }
            }
        }
    }
    best
}

fn check(tree: &SphereTree, r: Mat3, direction: Vec3) {
    let actual = tree.outermost_radial_contact(r, direction).unwrap();
    let canonical = rotation(quaternion(r));
    let expected = brute(tree, canonical, direction);
    assert_eq!(actual.is_some(), expected.is_some());
    if let Some(c) = actual {
        let tolerance = 2e-11 * (1. + tree.bound);
        assert!(
            (c.distance - expected.unwrap()).abs() < tolerance,
            "{:?} versus {:?}",
            c,
            expected
        );
        assert!(c.atomic_pairs_tested > 0);
        assert!(c.atomic_pairs_tested <= (tree.shape.atoms.len() * tree.shape.atoms.len()) as u64);
        let margin = 1e-7 * (1. + tree.bound);
        let fixed = Placed::new(pose([0.; 3], [1., 0., 0., 0.]));
        let outside = Placed::new(pose(scale(c.direction, c.distance + margin), c.orientation));
        assert!(!tree.overlaps(&outside, &fixed));
        let width = c.distance - c.witness_interval[0].max(0.);
        let inward = margin.min(width * 0.2);
        assert!(inward > 0.);
        let inside = Placed::new(pose(scale(c.direction, c.distance - inward), c.orientation));
        assert!(
            tree.overlaps(&inside, &fixed),
            "small inward displacement must cross the witnessed core"
        );
        let a = &tree.shape.atoms[c.moving_atom];
        let b = &tree.shape.atoms[c.fixed_atom];
        let d = sub(
            add(
                scale(c.direction, c.distance),
                matvec(rotation(c.orientation), a.center),
            ),
            b.center,
        );
        assert!((norm(d) - a.radius - b.radius).abs() < tolerance);
    }
}

#[test]
fn analytic_sphere_dumbbell_and_missed_ray_limits() {
    let sphere = tree(vec![Atom {
        center: [0.; 3],
        radius: 1.7,
    }]);
    let contact = sphere
        .outermost_radial_contact(IDENTITY, [2., -1., 0.3])
        .unwrap()
        .unwrap();
    assert!((contact.distance - 3.4).abs() < 1e-14);
    check(&sphere, cayley([0.1, 0.5, -0.3]), [2., -1., 0.3]);
    let dumbbell = tree(vec![
        Atom {
            center: [-2., 0., 0.],
            radius: 0.4,
        },
        Atom {
            center: [2., 0., 0.],
            radius: 0.4,
        },
    ]);
    let contact = dumbbell
        .outermost_radial_contact(IDENTITY, [1., 0., 0.])
        .unwrap()
        .unwrap();
    assert!(
        (contact.distance - 4.8).abs() < 1e-14,
        "must find the last disconnected collision interval"
    );
    assert!((contact.witness_interval[0] - 3.2).abs() < 1e-14);
    check(&dumbbell, IDENTITY, [1., 0., 0.]);
    check(&dumbbell, IDENTITY, [0., 1., 0.]);
    let offset = tree(vec![Atom {
        center: [2., 0., 0.],
        radius: 0.2,
    }]);
    assert!(
        offset
            .outermost_radial_contact(rotation([0., 0., 0., 1.]), [0., 0., 1.])
            .unwrap()
            .is_none()
    );
    check(&offset, rotation([0., 0., 0., 1.]), [1., 0., 0.]);
    assert!(offset.outermost_radial_contact(IDENTITY, [0.; 3]).is_err());
    assert!(
        offset
            .outermost_radial_contact([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [1., 0., 0.])
            .is_err()
    );
}

#[test]
fn rotated_concave_union_matches_brute_atomic_intervals() {
    let atoms = (-3..=3)
        .flat_map(|i| {
            let x = i as f64 * 0.75;
            [
                Atom {
                    center: [x, -2., 0.],
                    radius: 0.53,
                },
                Atom {
                    center: [x, 2., 0.1 * x],
                    radius: 0.49,
                },
            ]
        })
        .chain((-2..=2).map(|i| Atom {
            center: [-2.25, i as f64 * 0.75, 0.],
            radius: 0.57,
        }))
        .collect();
    let shape = tree(atoms);
    let mut rng = StdRng::seed_from_u64(19950802);
    let mut pruning_seen = false;
    for _ in 0..250 {
        let p = uniform_pose(&mut rng, [1.; 3]);
        let d = std::array::from_fn(|_| StandardNormal.sample(&mut rng));
        check(&shape, rotation(p.orientation), d);
        if let Some(c) = shape
            .outermost_radial_contact(rotation(p.orientation), d)
            .unwrap()
        {
            pruning_seen |= c.node_pairs_pruned > 0
                && c.atomic_pairs_tested
                    < (shape.shape.atoms.len() * shape.shape.atoms.len() / 2) as u64;
        }
    }
    assert!(
        pruning_seen,
        "test must exercise meaningful geometric pruning"
    );
}

#[test]
fn geometry_only_cli_emits_replayable_normalized_full_covariance_atlas() -> Result<()> {
    let root = std::env::temp_dir().join(format!("tetramer-contact-atlas-{}", std::process::id()));
    if root.exists() {
        fs::remove_dir_all(&root)?;
    }
    fs::create_dir_all(&root)?;
    let shape = json!({"name":"unequal dumbbell","atoms":[{"center":[-0.6,0.,0.],"radius":0.8},{"center":[0.7,0.1,0.],"radius":0.6}]});
    fs::write(root.join("shape.json"), shape.to_string())?;
    for output in ["one.json", "two.json"] {
        let result = Command::new(env!("CARGO_BIN_EXE_contact_atlas"))
            .args([
                "--shape",
                root.join("shape.json").to_str().unwrap(),
                "--out",
                root.join(output).to_str().unwrap(),
                "--count",
                "12",
                "--seed",
                "42",
                "--angular-length",
                "3.0",
            ])
            .output()?;
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let first: Value = serde_json::from_slice(&fs::read(root.join("one.json"))?)?;
    let second: Value = serde_json::from_slice(&fs::read(root.join("two.json"))?)?;
    for field in [
        "anchors",
        "means",
        "covariances",
        "weights",
        "contact_witnesses",
    ] {
        assert_eq!(first[field], second[field]);
    }
    assert_eq!(first["construction"]["native_information"], false);
    assert_eq!(first["angular_length"], 3.);
    let hash = hash_file(&root.join("shape.json"))?;
    let model =
        FrozenRelativePoseProposal::from_json_str_open(&first.to_string(), [12.; 3], 0.1, &hash)?;
    assert_eq!(model.component_count(), 12);
    assert!(root.join("one.provenance.json").is_file());
    let provenance: Value = serde_json::from_slice(&fs::read(root.join("one.provenance.json"))?)?;
    assert!(provenance["source_bundle"]["files"]["src/bin/contact_atlas.rs"]["sha256"].is_string());
    assert!(
        first["covariances"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| (0..3).any(|i| (3..6).any(|j| c[i][j].as_f64().unwrap().abs() > 1e-10))),
        "rolling cross covariance must be present"
    );
    let tree = SphereTree::new(serde_json::from_value(shape)?)?;
    let fixed = Placed::new(pose([0.; 3], [1., 0., 0., 0.]));
    for a in first["anchors"].as_array().unwrap() {
        let r: Mat3 = serde_json::from_value(a["rotation"].clone())?;
        let t: Vec3 = serde_json::from_value(a["position"].clone())?;
        assert!(!tree.overlaps(&Placed::new(pose(t, quaternion(r))), &fixed));
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
