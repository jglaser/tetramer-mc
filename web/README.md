# Offline trajectory viewer

`tools/export_viewer.py --run RUN_DIRECTORY --out viewer.html` bundles the actual sphere union, saved rigid-body poses and this directory's JavaScript/CSS into one offline HTML file. Open the HTML in a browser; no server, package installation or network access is needed.

The viewer offers frame stepping and playback, drag rotation, shift-drag panning, wheel zoom, body/initial-seed colors, optional body-center labels and a periodic box. Focus on the seed, an individual body, the whole box, or the largest nearby group. Periodic image choices bring neighboring bodies together. When recorded contact edges are unavailable, nearby groups use a bounding-sphere proximity graph strictly for camera framing; they are not an association or crystallinity metric.

Spherical runs use their original unwrapped coordinates in the fixed container frame. The wall center stays at the origin throughout playback; the camera does not follow the seed, a body, or a changing group. The initial/reset scale fits the sphere radius, and manual rotation, zoom, and panning remain available. The outline shows the hard protein wall (the ideal bath permeates it). The exporter recognizes `boundary: {"kind":"spherical","radius":R}` in the run configuration.

WebGL2 renders individual sphere surfaces with per-fragment depth. A Canvas2D fallback draws sorted projected atom circles. Both use the supplied atom radii; the fallback has approximate occlusion. GPU display buffers are float32; authoritative simulation coordinates remain in the source files and embedded JSON.

This is a lightweight browser viewer, **not hoomd-bevy**. A future optional adapter could read the same saved body poses and shape into hoomd-bevy, or consume the GSD output. The viewer itself does not run the sampler or identify native arrangements.

`viewer.js` exports its pure geometry helpers under Node for numerical checks. JavaScript syntax and geometry checks do not substitute for an actual browser rendering test.
