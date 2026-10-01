//! Normalized angular conditioning, separate from physical production sampling.
use super::*;

fn pose_data(alpha: f64, beta: f64, axes: &[usize]) -> Value {
    let mut raw = data(alpha, beta);
    raw["schema"] = json!("defensive-hard-free-pose-line-guide-v1");
    for key in ["raw_translation_axes", "contact_widths_A", "contact_neighbor_indices"] {
        raw.as_object_mut().unwrap().remove(key);
    }
    raw["raw_pose_axes"] = json!(axes);
    raw
}

#[test]
fn pose_line_schema_is_separate_and_legacy_axes_keep_their_meaning() {
    let c = chart(false);
    let cfg = configuration();
    let parse = |r: &Value| ContactLineGuide::from_bytes(
        &serde_json::to_vec(r).unwrap(), "toy-region", &c, 0., &cfg, &sphere());
    let mut raw = pose_data(0.5, 1., &[0,1,2,3,4,5]);
    assert!(parse(&raw).unwrap().pose_coordinates);
    raw["raw_translation_axes"] = json!([0]);
    assert!(parse(&raw).is_err());
    raw = pose_data(0.5, 1., &[3,3]);
    assert!(parse(&raw).is_err());
    raw["raw_pose_axes"] = json!([6]);
    assert!(parse(&raw).is_err());
    raw = pose_data(0.5, 1., &[3]);
    raw["schema"] = json!("defensive-hard-free-line-guide-v1");
    raw.as_object_mut().unwrap().remove("raw_pose_axes");
    raw["raw_translation_axes"] = json!([3]);
    assert!(parse(&raw).is_err());
}

#[test]
fn pose_line_translation_only_is_identical_to_existing_law() {
    let c = chart(true);
    let cfg = configuration();
    let raw = pose_data(0.5, 1., &[0,1,2]);
    let mut old = raw.clone();
    old["schema"] = json!("defensive-hard-free-line-guide-v1");
    old.as_object_mut().unwrap().remove("raw_pose_axes");
    old["raw_translation_axes"] = json!([0,1,2]);
    let g = guide(&raw, &c, &cfg);
    let legacy = guide(&old, &c, &cfg);
    let mut a = StdRng::seed_from_u64(71201);
    let mut b = StdRng::seed_from_u64(71201);
    for _ in 0..128 {
        let d = g.draw(&mut a, &c, OUTER, 0., 1.).unwrap();
        assert_eq!(d, legacy.draw(&mut b, &c, OUTER, 0., 1.).unwrap());
        assert_eq!(*g.last_draw.borrow(), *legacy.last_draw.borrow());
        let inside = d.1 <= OUTER;
        assert_eq!(g.density_details(d.0, inside, log_volume(), &c, OUTER).unwrap().0,
            legacy.density_details(d.0, inside, log_volume(), &c, OUTER).unwrap().0);
    }
}

#[test]
fn pose_line_fixed_center_capture_and_spherical_limit() {
    let c = chart(false);
    let mut cfg = configuration();
    cfg.capture_radius = 0.1;
    let g = guide(&pose_data(0.5, 1., &[3,4,5]), &c, &cfg);
    for axis in 3..6 {
        let (sets, detail) = g.intervals([0.;6], &c, OUTER, axis).unwrap();
        assert!(detail["empty_reason"].is_null());
        assert_eq!(sets[0], IntervalSet::segment([-OUTER,OUTER]).unwrap());
        let (mean, sigma) = g.conditionals[axis-3][0].conditional([0.;6]);
        let (_, mass) = ContactLineGuide::masses(&sets[0], mean, sigma).unwrap();
        assert!(mass > 0. && mass < 1.);
        let mut outside = [0.;6]; outside[0] = 0.2;
        let (empty, detail) = g.intervals(outside, &c, OUTER, axis).unwrap();
        assert_eq!(detail["empty_reason"], "no_capture_at_fixed_center");
        assert!(empty[0].is_empty());
    }
}

#[test]
fn pose_line_conditional_density_normalizes_and_jacobian_varies() {
    let c = chart(false);
    let cfg = configuration();
    let mut raw = pose_data(0.5, 1., &[4]);
    raw["gaussian_components"] = json!([component([0.;6],identity(),1.)]);
    let g = guide(&raw, &c, &cfg);
    let uniform = 0.5 * (-log_volume()).exp();
    let outer_marginal = (2. * PI).powf(-2.5);
    let n = 1024;
    let step = 2.*OUTER/n as f64;
    let mut integral = 0.;
    for i in 0..=n {
        let mut u = [0.;6]; u[4] = -OUTER+i as f64*step;
        let q = g.density_details(u,true,log_volume(),&c,OUTER).unwrap().0.exp();
        let weight = if i == 0 || i == n { 1. } else if i%2 == 0 { 2. } else { 4. };
        integral += weight*step*(q-uniform)/(0.5*outer_marginal)/3.;
    }
    assert!((integral-1.).abs()<2e-10,"{integral}");
    let mut u = [0.;6]; u[4] = 2.;
    assert!((c.decode(u).1-c.decode([0.;6]).1 + 2.*2_f64.ln()).abs() < 1e-13);
}

#[test]
fn pose_line_mixture_scores_every_axis_and_preserves_retained_coordinates() {
    let c = chart(true);
    let cfg = configuration();
    let g = guide(&pose_data(0.5, 1., &[0,1,2,3,4,5]), &c, &cfg);
    let mut rng = StdRng::seed_from_u64(71202);
    let mut seen = [0;6];
    let mut fallbacks = 0;
    for _ in 0..512 {
        let (u,r,_,_) = g.draw(&mut rng,&c,OUTER,0.,1.).unwrap();
        let trace = g.last_draw.borrow();
        let (q,details) = g.density_details(u,r<=OUTER,log_volume(),&c,OUTER).unwrap();
        let mixture = details["axes"].as_array().unwrap().iter().fold(f64::NEG_INFINITY,
            |acc,a| log_add(acc,a["axis_log_proposal_density"].as_f64().unwrap_or(f64::NEG_INFINITY)-6_f64.ln()));
        assert!((q-mixture).abs()<1e-11);
        if trace["conditional"] == true {
            let axis = trace["axis"].as_u64().unwrap() as usize; seen[axis] += 1;
            let original: [f64;6] = serde_json::from_value(trace["original_latent"].clone()).unwrap();
            let x0 = c.coordinates(original); let x1 = c.coordinates(u);
            for j in 0..6 { if j != axis { assert!((x0[j]-x1[j]).abs()<1e-12); } }
            if trace["fallback"] == true { fallbacks += 1; assert_eq!(u,original); }
            else {
                assert!(r<=OUTER+1e-12);
                assert!(cfg.contains(c.decode(u).0));
                assert!(!cfg.fixed_poses.iter().any(|p| sphere().overlaps(&Placed::new(c.decode(u).0),&Placed::new(*p))));
            }
        }
    }
    assert!(seen.iter().all(|&n|n>10));
    assert!(fallbacks>0);
}
