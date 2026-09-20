/* Pure display geometry: node web/viewer.test.cjs. No graphics context needed. */
const assert = require('assert');
const { displayedCenters, nativeBondSegments } = require('./viewer.js');
const pose = p => ({ position: p, orientation: [1, 0, 0, 0] });
const show = (frame, origin) => displayedCenters(frame, [20,20,20], 'box', 1, 0, 'spherical', [0,0,0], origin);
const before = { poses: [pose([1,2,3]), pose([4,2,3])], coordinate_wall_center: [0,0,0], seed_labels: [] };
const afterShift = { poses: [pose([3,1,7]), pose([6,1,7])], coordinate_wall_center: [-2,1,-4], seed_labels: [] };
assert.deepStrictEqual(show(before, 'coordinate').centers, show(afterShift, 'coordinate').centers);
assert.deepStrictEqual(show(afterShift, 'coordinate').boundaryCenter, [-2,1,-4]);
assert.deepStrictEqual(show(afterShift, 'sphere').centers, afterShift.poses.map(p=>p.position));
assert.deepStrictEqual(show(afterShift, 'sphere').boundaryCenter, [0,0,0]);
// A proper half-turn about z acts about C in lab coordinates, leaving C fixed.
const afterGca = { ...afterShift, poses: afterShift.poses.map(p=>pose([-p.position[0],-p.position[1],p.position[2]])) };
const labBefore = show(afterShift, 'coordinate'), labAfter = show(afterGca, 'coordinate');
for (let i=0;i<2;i++) {
  const c = labBefore.boundaryCenter, p = labBefore.centers[i];
  assert.deepStrictEqual(labAfter.centers[i], [2*c[0]-p[0],2*c[1]-p[1],p[2]]);
}
assert.deepStrictEqual(labAfter.boundaryCenter, labBefore.boundaryCenter);
const data = { boundary: 'spherical', box_lengths: [20,20,20], native_bonds: {available:true,member_positions:[[.2,0,0]]} };
afterShift.native_bonds = [{bodies:[0,1],members:[0,0]}];
const a = nativeBondSegments(data, afterShift, show(afterShift,'sphere').centers)[0];
const b = nativeBondSegments(data, afterShift, show(afterShift,'coordinate').centers)[0];
for(let d=0;d<3;d++) {
  assert(Math.abs((b.start[d]-a.start[d])-afterShift.coordinate_wall_center[d])<1e-12);
  assert(Math.abs((b.end[d]-a.end[d])-afterShift.coordinate_wall_center[d])<1e-12);
}
// Existing periodic image behavior remains independent of the new convention.
const periodic = {poses:[pose([4.5,0,0]),pose([-4.5,0,0])],seed_labels:[0,1]};
const p = displayedCenters(periodic,[10,10,10],'seed',1,0,'periodic');
assert.strictEqual(Math.abs(p.centers[0][0]-p.centers[1][0]),1);
console.log('Accumulated-shift wall motion, GCA about the moving wall, bond offsets, and periodic control pass.');
