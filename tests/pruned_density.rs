//! The pruned member-chart density equals the full one on the blind monomer contact map
//! at poses sampled from the map itself (near contacts) and at random poses.
use rand::{SeedableRng, rngs::StdRng};
use tetramer_mc::{
    docking::{DockingMethod, DockingProposal},
    math::{Pose, uniform_pose},
    proposal::FrozenRelativePoseProposal,
    simulation::hash_bytes,
};

#[test]
fn pruned_density_matches_full_on_blind_map() {
    let shape = std::fs::read("examples/monomer-shape.json").unwrap();
    let model =
        std::fs::read_to_string("examples/frozen-blind-monomer-contact-mixture-512.json").unwrap();
    let frozen =
        FrozenRelativePoseProposal::from_json_str_open(&model, [400.; 3], 0.1, &hash_bytes(&shape))
            .unwrap();
    let p = DockingProposal::new(frozen, DockingMethod::PosteriorInvolution, 0.9, [0.; 3]).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let anchors: Vec<Pose> = (0..3).map(|_| uniform_pose(&mut rng, [60.; 3])).collect();
    let mut checked = 0;
    for k in 0..200 {
        let x = if k % 4 == 0 {
            uniform_pose(&mut rng, [60.; 3])
        } else {
            match p.draw_singleton_independent(&mut rng, &anchors).unwrap().0 {
                Some(x) => x,
                None => continue,
            }
        };
        let full = p.members_log_density(&[x], &anchors).unwrap();
        let pruned = p.members_log_density_pruned(&[x], &anchors).unwrap();
        if full > -50. {
            assert!(
                (full - pruned).abs() < 1e-9 * (1. + full.abs()),
                "pose {k}: full {full}, pruned {pruned}"
            );
            checked += 1;
        } else {
            assert!(pruned <= full + 1e-9 || pruned == f64::NEG_INFINITY);
        }
    }
    assert!(checked > 100, "{checked}");
}
