# Offline trajectory viewer

`tools/export_viewer.py --run RUN_DIRECTORY --out viewer.html` bundles the actual sphere union, saved rigid-body poses and this directory's JavaScript/CSS into one offline HTML file. Open the HTML in a browser; no server, package installation or network access is needed.

The viewer offers frame stepping and playback, drag rotation, shift-drag panning, wheel zoom, body/initial-seed colors, optional body-center labels and a periodic box. Focus on the seed, an individual body, the whole box, or the largest nearby group. Periodic image choices bring neighboring bodies together. When recorded contact edges are unavailable, nearby groups use a bounding-sphere proximity graph strictly for camera framing; they are not an association or crystallinity metric.

Spherical runs offer **Sphere center** (default) and **Coordinate origin** frames. Both use the original unwrapped coordinates with a fixed offset; neither follows the seed, a body, or a changing group. The initial/reset scale fits the container, and manual rotation, zoom, and panning remain available. The exporter reads `spherical_wall_center` from the config or its metadata, or `center` from the boundary object; absent metadata means `[0,0,0]`. The choices therefore coincide for the current simulations. The outline shows the hard protein wall (the ideal bath permeates it). The exporter recognizes `boundary: {"kind":"spherical","radius":R}` or the Python spherical-boundary metadata.

When native geometry is available, **Native bonds** toggles a purple overlay joining the centers of monomers in distinct tetramers. The unchanged research `TetramerOrder` classifier requires its strict per-frame entry criteria: proper relative monomer pose, atomic gap, and a shared native residue patch. Intrinsic contacts inside a tetramer are excluded. These bonds are distinct from proximity-based camera groups and do not certify whole-tetramer registration, crystal stability, or new attachment events. Classification is independent at each saved frame; no retention hysteresis is inferred. The overlay is drawn above the atoms for visibility, with no depth occlusion. Periodic bonds whose endpoints use different displayed images are drawn as two dashed half-bonds anchored to the displayed monomers, avoiding false box-spanning lines.

Native extraction is optional and occurs only during export. The default `--native-bonds auto` hides the control when geometry or research dependencies are unavailable; `--native-bonds off` keeps the exporter entirely standard-library-only. `--native-bonds on` requires the research NumPy/SciPy environment, rigid-member poses (config or archived shape), `monomer_shape`, native templates, and any configured native motif file. Override the research checkout with `--reference PATH`. For example:

```bash
/home/xvg/protein-nucleation/.venv/bin/python tools/export_viewer.py \
  --run runs/spherical-pilot/python/seeded-transport-gca-shift \
  --out runs/spherical-pilot/seeded-python-viewer.html --native-bonds on
```

The exported HTML remains standalone and needs none of these Python dependencies. Embedded native metadata records the classifier, criteria and input/source hashes. For spherical data, an analysis-only box large enough to leave every pair displacement unwrapped allows reuse of the classifier without modifying it.

WebGL2 renders individual sphere surfaces with per-fragment depth. A Canvas2D fallback draws sorted projected atom circles. Both use the supplied atom radii; the fallback has approximate occlusion. GPU display buffers are float32; authoritative simulation coordinates remain in the source files and embedded JSON.

This is a lightweight browser viewer, **not hoomd-bevy**. A future optional adapter could read the same saved body poses and shape into hoomd-bevy, or consume the GSD output. The browser does not run the sampler or classifier; it displays geometry and any native bonds computed during export.

`viewer.js` exports its pure geometry helpers under Node for numerical checks. JavaScript syntax and geometry checks do not substitute for an actual browser rendering test.
