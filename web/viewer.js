/* Standalone viewer: display geometry only, no sampling or native classifier. */
(function (global) {
  "use strict";
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const norm = a => Math.hypot(...a);
  const minimumImage = (a, box) => a.map((x, i) => x - box[i] * Math.floor(x / box[i] + 0.5));
  const rotate = (r, a) => r.map(row => row.reduce((s, x, i) => s + x * a[i], 0));
  function quaternionMatrix(q) {
    const [w, x, y, z] = q;
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
  }
  function cameraMatrix(yaw, pitch) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return [[cy, 0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]];
  }
  function nearbyGroups(poses, box, cutoff, recordedEdges, periodic = true) {
    const adjacency = poses.map(() => []);
    if (Array.isArray(recordedEdges)) {
      recordedEdges.forEach(([i, j]) => { if (i < poses.length && j < poses.length) { adjacency[i].push(j); adjacency[j].push(i); } });
    } else {
      for (let i = 0; i < poses.length; i++) for (let j = i + 1; j < poses.length; j++) {
        const delta = sub(poses[j].position, poses[i].position);
        if (norm(periodic ? minimumImage(delta, box) : delta) <= cutoff) {
          adjacency[i].push(j); adjacency[j].push(i);
        }
      }
    }
    const seen = new Set(), groups = [];
    for (let i = 0; i < poses.length; i++) if (!seen.has(i)) {
      const group = [i]; seen.add(i);
      for (let k = 0; k < group.length; k++) adjacency[group[k]].forEach(j => { if (!seen.has(j)) { seen.add(j); group.push(j); } });
      groups.push(group);
    }
    groups.sort((a, b) => b.length - a.length || a[0] - b[0]);
    return { groups, adjacency };
  }
  function displayedCenters(frame, box, focus, bodyBound, radius, boundary = "periodic", wallCenter = [0, 0, 0], origin = "sphere") {
    const poses = frame.poses;
    if (boundary === "spherical") {
      // The simulation stores sphere-centered poses. C is the accumulated
      // coordinate origin, not a center of mass or a fit to the current frame.
      const center = frame.coordinate_wall_center || wallCenter;
      const offset = origin === "coordinate" ? center : [0, 0, 0];
      const originSweep = frame.coordinate_origin_sweep || 0;
      return { centers: poses.map(p => add(p.position, offset)), group: poses.map((_, i) => i),
        boundaryCenter: [...offset],
        label: origin === "coordinate" ? "Coordinate origin · accumulated shift convention" + (originSweep ? `\nOrigin established at sweep ${originSweep}` : "") : "Sphere-centered frame · fixed wall" };
    }
    if (focus === "box") return { centers: poses.map(p => minimumImage(p.position, box)), group: poses.map((_, i) => i), label: "Primary cell · body centers wrapped" };
    const proximity = nearbyGroups(poses, box, 2 * bodyBound + 2 * radius, frame.contact_edges);
    let group;
    if (focus === "seed" && frame.seed_labels.length) group = frame.seed_labels.filter(i => i < poses.length);
    else if (focus.startsWith("body:")) {
      const index = Number(focus.slice(5)); group = [index, ...proximity.adjacency[index]];
    } else group = proximity.groups[0];
    const anchor = group[0];
    // Follow a spanning tree when possible so neighbors across a cell boundary
    // are displayed next to one another. Every resulting shift is a box image.
    const centers = poses.map(p => minimumImage(sub(p.position, poses[anchor].position), box));
    const visited = new Set([anchor]), queue = [anchor];
    for (let k = 0; k < queue.length; k++) proximity.adjacency[queue[k]].forEach(j => {
      if (!visited.has(j)) {
        centers[j] = add(centers[queue[k]], minimumImage(sub(poses[j].position, poses[queue[k]].position), box));
        visited.add(j); queue.push(j);
      }
    });
    const mean = group.reduce((s, i) => add(s, centers[i]), [0, 0, 0]).map(x => x / group.length);
    const label = focus === "seed" && frame.seed_labels.length ? `Initial seed · ${group.length} bodies` : focus.startsWith("body:") ? `Body ${anchor} and nearby bodies` : `Largest nearby group · ${group.length} bodies`;
    return { centers: centers.map(p => sub(p, mean)), group, label };
  }
  function bodyGeometry(data, frame, centers) {
    const n = data.atoms.length * frame.poses.length;
    const result = new Float32Array(n * 4);
    let offset = 0;
    frame.poses.forEach((pose, body) => {
      const r = quaternionMatrix(pose.orientation), center = centers[body];
      data.atoms.forEach(atom => {
        const p = add(rotate(r, atom.slice(0, 3)), center);
        result[offset++] = p[0]; result[offset++] = p[1]; result[offset++] = p[2]; result[offset++] = atom[3];
      });
    });
    return result;
  }
  function nativeBondSegments(data, frame, centers) {
    if (!data.native_bonds || !data.native_bonds.available) return [];
    const members = data.native_bonds.member_positions, segments = [];
    (frame.native_bonds || []).forEach(bond => {
      const [i, j] = bond.bodies, [a, b] = bond.members;
      const start = add(centers[i], rotate(quaternionMatrix(frame.poses[i].orientation), members[a]));
      const end = add(centers[j], rotate(quaternionMatrix(frame.poses[j].orientation), members[b]));
      const raw = sub(end, start);
      const delta = data.boundary === "spherical" ? raw : minimumImage(raw, data.box_lengths);
      if (norm(sub(raw, delta)) < 1e-8 * Math.max(1, norm(data.box_lengths))) {
        segments.push({ start, end, periodicSplit: false });
      } else {
        // Two halves represent the same torus bond, each attached to its
        // actually displayed monomer. Never draw a spurious box-spanning line.
        segments.push({ start, end: add(start, delta.map(x => x / 2)), periodicSplit: true });
        segments.push({ start: sub(end, delta.map(x => x / 2)), end, periodicSplit: true });
      }
    });
    return segments;
  }
  // Export pure geometry helpers for Node checks; no DOM or graphics context is
  // required to test periodic images and rigid-body coordinates independently.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { quaternionMatrix, minimumImage, nearbyGroups, displayedCenters, bodyGeometry, nativeBondSegments, cameraMatrix };
    return;
  }

  const $ = id => document.getElementById(id);
  const data = JSON.parse($("trajectory-data").textContent);
  const canvas = $("scene"), overlay = $("overlay"), overlayContext = overlay.getContext("2d");
  const viewport = $("viewport");
  const state = { index: 0, yaw: -0.45, pitch: 0.35, scale: 1, maximumScale: 60, pan: [0, 0], playing: false,
    lastTick: 0, geometry: null, centers: null, group: [], colors: null, dirty: true };
  let gl = null, fallback = null, program = null, geometryBuffer = null, colorBuffer = null;
  let rendererName = "Canvas2D";
  const seedColor = [0.22, 0.62, 0.78], otherColor = [0.89, 0.63, 0.29];
  const palette = [[.22,.61,.78],[.9,.55,.27],[.39,.67,.43],[.7,.42,.68],[.82,.39,.37],[.43,.48,.78],[.7,.64,.29],[.23,.69,.65],[.82,.55,.65],[.53,.62,.72],[.63,.49,.31],[.42,.7,.77]];

  function shader(type, source) {
    const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function startWebGL() {
    gl = canvas.getContext("webgl2", { antialias: true, alpha: true, preserveDrawingBuffer: true });
    if (!gl) return false;
    const vertex = shader(gl.VERTEX_SHADER, `#version 300 es
      precision highp float;
      in vec4 a_sphere; in vec3 a_color;
      uniform mat3 u_rotation; uniform vec2 u_viewport; uniform vec2 u_pan;
      uniform float u_scale; uniform float u_depth; uniform float u_max_point;
      out vec3 v_color; out float v_z; out float v_radius;
      void main(){vec3 p=u_rotation*a_sphere.xyz;
        gl_Position=vec4((p.xy*u_scale+u_pan)*2.0/u_viewport,-p.z/u_depth,1.0);
        gl_PointSize=min(u_max_point,max(1.0,2.0*a_sphere.w*u_scale));
        v_color=a_color;v_z=p.z;v_radius=a_sphere.w;}`);
    const fragment = shader(gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float;
      in vec3 v_color; in float v_z; in float v_radius; uniform float u_depth;
      out vec4 out_color;
      void main(){vec2 uv=2.0*gl_PointCoord-1.0;uv.y=-uv.y;
        float rr=dot(uv,uv);if(rr>1.0)discard;
        vec3 n=vec3(uv,sqrt(max(0.0,1.0-rr)));
        float lighting=0.35+0.65*max(0.0,dot(n,normalize(vec3(-0.35,0.55,1.0))));
        float shine=0.13*pow(max(0.0,dot(n,normalize(vec3(-0.2,0.4,1.0)))),30.0);
        out_color=vec4(v_color*lighting+shine,1.0);
        gl_FragDepth=0.5*(1.0-(v_z+v_radius*n.z)/u_depth);}`);
    program = gl.createProgram(); gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    geometryBuffer = gl.createBuffer(); colorBuffer = gl.createBuffer();
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.clearColor(0, 0, 0, 0);
    state.maximumScale = Math.min(60, gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1] / (2 * Math.max(...data.atoms.map(a => a[3]))));
    rendererName = "WebGL2"; return true;
  }
  try {
    if (!startWebGL()) fallback = canvas.getContext("2d");
  } catch (error) {
    // A failed WebGL context cannot be changed to 2D on the same canvas.
    const replacement = document.createElement("canvas"); replacement.id = "fallback-scene";
    viewport.insertBefore(replacement, canvas); canvas.style.display = "none";
    fallback = replacement.getContext("2d"); gl = null;
    console.warn("Using Canvas2D fallback", error);
  }
  if (!gl && !fallback) { $("error").hidden = false; $("error").textContent = "This browser could not initialize a graphics canvas."; return; }

  function setColors() {
    const seeds = new Set(data.frames[state.index].seed_labels), values = new Float32Array(data.atoms.length * data.body_count * 3);
    for (let body = 0; body < data.body_count; body++) {
      const color = $("color").value === "body" ? palette[body % palette.length] : seeds.has(body) ? seedColor : otherColor;
      for (let atom = 0; atom < data.atoms.length; atom++) values.set(color, (body * data.atoms.length + atom) * 3);
    }
    state.colors = values;
    $("legend").style.display = $("color").value === "body" ? "none" : "flex";
    if (gl) { gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer); gl.bufferData(gl.ARRAY_BUFFER, values, gl.STATIC_DRAW); }
  }
  function updateGeometry(reset) {
    const frame = data.frames[state.index];
    const displayed = displayedCenters(frame, data.box_lengths, $("focus").value, data.body_bound, data.depletant_radius || 0, data.boundary || "periodic", [0, 0, 0], $("frame-origin").value);
    state.centers = displayed.centers; state.group = displayed.group;
    state.boundaryCenter = displayed.boundaryCenter || [0, 0, 0];
    state.bonds = nativeBondSegments(data, frame, displayed.centers);
    state.geometry = bodyGeometry(data, frame, displayed.centers); setColors();
    if (gl) { gl.bindBuffer(gl.ARRAY_BUFFER, geometryBuffer); gl.bufferData(gl.ARRAY_BUFFER, state.geometry, gl.DYNAMIC_DRAW); }
    $("view-caption").textContent = displayed.label + (data.boundary === "spherical" ? "\nHard atom wall · permeable depletant bath" : "\nPeriodic images brought together around the focus");
    if ($("focus").value === "box" && data.boundary !== "spherical") $("view-caption").textContent = displayed.label;
    $("frame-label").textContent = `Frame ${state.index + 1}/${data.frames.length} · sweep ${frame.sweep}`;
    $("frame").value = state.index;
    $("status").textContent = `${data.body_count} rigid bodies · actual sphere radii in Å · saved sweep ${frame.sweep}`;
    if (data.native_bonds && data.native_bonds.available) $("status").textContent += ` · ${(frame.native_bonds || []).length} external native bonds`;
    $("native-bond-note").hidden = !$("native-bonds").checked || !(data.native_bonds && data.native_bonds.available);
    if (reset) resetCamera();
    state.dirty = true;
  }
  function resize() {
    const dpr = Math.min(global.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(viewport.clientWidth * dpr)), height = Math.max(1, Math.round(viewport.clientHeight * dpr));
    const target = fallback ? fallback.canvas : canvas;
    if (target.width !== width || target.height !== height) {
      target.width = width; target.height = height; overlay.width = width; overlay.height = height;
      if (gl) gl.viewport(0, 0, width, height);
      state.dirty = true;
    }
    return [width, height, dpr];
  }
  function resetCamera() {
    state.pan = [0, 0];
    const [, height] = resize();
    const extent = data.boundary === "spherical" ? ($("frame-origin").value === "coordinate" ? data.coordinate_frame.fixed_view_extent : data.spherical_wall_radius) : $("focus").value === "box" ? norm(data.box_lengths) / 2 + data.body_bound : Math.max(...state.group.map(i => norm(state.centers[i]))) + data.body_bound;
    state.scale = Math.min(viewport.clientWidth, viewport.clientHeight) * (global.devicePixelRatio || 1) * .43 / Math.max(extent, 1);
    // Match the pixel-ratio cap used by resize.
    if ((global.devicePixelRatio || 1) > 2) state.scale *= 2 / global.devicePixelRatio;
    if (!Number.isFinite(state.scale)) state.scale = height / Math.max(4 * data.body_bound, 1);
    state.scale = Math.min(state.scale, state.maximumScale);
    state.dirty = true;
  }
  function project(point, rotation, width, height) {
    const p = rotate(rotation, point);
    return [width / 2 + p[0] * state.scale + state.pan[0], height / 2 - p[1] * state.scale - state.pan[1], p[2]];
  }
  function drawOverlay(r, width, height, dpr) {
    const ctx = overlayContext; ctx.clearRect(0, 0, width, height); ctx.lineWidth = dpr;
    if ($("native-bonds").checked) {
      (state.bonds || []).forEach(bond => {
        const a = project(bond.start, r, width, height), b = project(bond.end, r, width, height);
        ctx.setLineDash(bond.periodicSplit ? [4 * dpr, 2 * dpr] : []);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
        ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 4 * dpr; ctx.stroke();
        ctx.strokeStyle = "#61318e"; ctx.lineWidth = 2 * dpr; ctx.stroke();
      });
      ctx.setLineDash([]); ctx.lineWidth = dpr;
    }
    if ($("box").checked && data.boundary === "spherical") {
      const center = state.boundaryCenter, radius = data.spherical_wall_radius;
      ctx.strokeStyle = "rgba(89,116,137,0.42)";
      const p = project(center, r, width, height);
      ctx.beginPath(); ctx.arc(p[0], p[1], radius * state.scale, 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = "rgba(89,116,137,0.17)";
      for (let axis = 0; axis < 3; axis++) {
        ctx.beginPath();
        for (let k = 0; k <= 128; k++) {
          const angle = 2 * Math.PI * k / 128, q = [...center];
          q[(axis + 1) % 3] += radius * Math.cos(angle); q[(axis + 2) % 3] += radius * Math.sin(angle);
          const v = project(q, r, width, height); if (k === 0) ctx.moveTo(v[0], v[1]); else ctx.lineTo(v[0], v[1]);
        }
        ctx.stroke();
      }
    } else if ($("box").checked) {
      const corners = [];
      for (let x = -1; x <= 1; x += 2) for (let y = -1; y <= 1; y += 2) for (let z = -1; z <= 1; z += 2) corners.push([x * data.box_lengths[0] / 2, y * data.box_lengths[1] / 2, z * data.box_lengths[2] / 2]);
      ctx.strokeStyle = "rgba(89,116,137,0.35)"; ctx.beginPath();
      corners.forEach((a, i) => corners.forEach((b, j) => {
        if (j > i && a.reduce((s, value, k) => s + Number(value !== b[k]), 0) === 1) {
          const u = project(a, r, width, height), v = project(b, r, width, height); ctx.moveTo(u[0], u[1]); ctx.lineTo(v[0], v[1]);
        }
      })); ctx.stroke();
    }
    if ($("centers").checked) {
      ctx.font = `${11 * dpr}px system-ui`; ctx.textAlign = "center";
      state.centers.forEach((center, body) => {
        const p = project(center, r, width, height); ctx.beginPath(); ctx.arc(p[0], p[1], 4 * dpr, 0, 2 * Math.PI);
        ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#273f53"; ctx.stroke();
        ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "#ffffffdd"; ctx.strokeText(String(body), p[0], p[1] - 8 * dpr);
        ctx.fillStyle = "#17364e"; ctx.fillText(String(body), p[0], p[1] - 8 * dpr); ctx.lineWidth = dpr;
      });
    }
    const barWorld = Math.pow(10, Math.floor(Math.log10(100 * dpr / state.scale)));
    const bar = barWorld * state.scale; ctx.strokeStyle = "#45657d"; ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.moveTo(width - 24 * dpr - bar, height - 27 * dpr); ctx.lineTo(width - 24 * dpr, height - 27 * dpr); ctx.stroke();
    ctx.fillStyle = "#45657d"; ctx.font = `${10 * dpr}px system-ui`; ctx.textAlign = "center";
    ctx.fillText(`${barWorld} Å`, width - 24 * dpr - bar / 2, height - 35 * dpr);
  }
  function render() {
    const [width, height, dpr] = resize(), r = cameraMatrix(state.yaw, state.pitch);
    if (gl) {
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.useProgram(program);
      const vertex = gl.getAttribLocation(program, "a_sphere"), color = gl.getAttribLocation(program, "a_color");
      gl.bindBuffer(gl.ARRAY_BUFFER, geometryBuffer); gl.enableVertexAttribArray(vertex); gl.vertexAttribPointer(vertex, 4, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer); gl.enableVertexAttribArray(color); gl.vertexAttribPointer(color, 3, gl.FLOAT, false, 0, 0);
      gl.uniformMatrix3fv(gl.getUniformLocation(program, "u_rotation"), false, new Float32Array([r[0][0], r[1][0], r[2][0], r[0][1], r[1][1], r[2][1], r[0][2], r[1][2], r[2][2]]));
      gl.uniform2f(gl.getUniformLocation(program, "u_viewport"), width, height);
      gl.uniform2f(gl.getUniformLocation(program, "u_pan"), state.pan[0], state.pan[1]);
      gl.uniform1f(gl.getUniformLocation(program, "u_scale"), state.scale);
      const wholeHistoryExtent = data.coordinate_frame && data.coordinate_frame.fixed_view_extent || 0;
      gl.uniform1f(gl.getUniformLocation(program, "u_depth"), 2 * Math.max(norm(data.box_lengths) + 2 * data.body_bound, wholeHistoryExtent + data.body_bound));
      gl.uniform1f(gl.getUniformLocation(program, "u_max_point"), gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1]);
      gl.drawArrays(gl.POINTS, 0, state.geometry.length / 4);
    } else {
      const ctx = fallback; ctx.clearRect(0, 0, width, height);
      const points = [];
      for (let i = 0; i < state.geometry.length / 4; i++) {
        const p = project(Array.from(state.geometry.subarray(i * 4, i * 4 + 3)), r, width, height), radius = Math.max(.5, state.geometry[i * 4 + 3] * state.scale);
        if (p[0] + radius >= 0 && p[0] - radius <= width && p[1] + radius >= 0 && p[1] - radius <= height) points.push({ p, radius, i });
      }
      points.sort((a, b) => a.p[2] - b.p[2]);
      points.forEach(({ p, radius, i }) => {
        const color = Array.from(state.colors.subarray(i * 3, i * 3 + 3)).map(x => Math.round(x * 255));
        ctx.fillStyle = `rgb(${color.join(",")})`; ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, 2 * Math.PI); ctx.fill();
        if (radius > 2) { ctx.strokeStyle = "rgba(20,40,50,.18)"; ctx.lineWidth = .5 * dpr; ctx.stroke(); }
      });
    }
    drawOverlay(r, width, height, dpr); state.dirty = false;
  }
  function setFrame(index) { state.index = Math.max(0, Math.min(data.frames.length - 1, index)); updateGeometry(false); }
  function play() { state.playing = !state.playing; $("play").textContent = state.playing ? "Pause" : "Play"; state.lastTick = 0; }
  $("play").onclick = play;
  $("previous").onclick = () => setFrame(state.index - 1);
  $("next").onclick = () => setFrame(state.index + 1);
  $("frame").oninput = event => setFrame(Number(event.target.value));
  $("focus").onchange = () => updateGeometry(true);
  $("frame-origin").onchange = () => updateGeometry(true);
  $("native-bonds").onchange = () => { $("native-bond-note").hidden = !$("native-bonds").checked; state.dirty = true; };
  $("color").onchange = () => { setColors(); state.dirty = true; };
  $("centers").onchange = $("box").onchange = () => { state.dirty = true; };
  $("reset").onclick = () => { state.yaw = -.45; state.pitch = .35; resetCamera(); };
  let drag = null;
  viewport.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY }; viewport.setPointerCapture(e.pointerId); });
  viewport.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag = { x: e.clientX, y: e.clientY };
    if (e.shiftKey) { const dpr = Math.min(global.devicePixelRatio || 1, 2); state.pan[0] += dx * dpr; state.pan[1] -= dy * dpr; }
    else { state.yaw += dx * .007; state.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, state.pitch + dy * .007)); }
    state.dirty = true;
  });
  viewport.addEventListener("pointerup", () => { drag = null; });
  viewport.addEventListener("pointercancel", () => { drag = null; });
  viewport.addEventListener("wheel", e => { e.preventDefault(); state.scale = Math.max(.01, Math.min(state.maximumScale, state.scale * Math.exp(-e.deltaY * .0015))); state.dirty = true; }, { passive: false });
  global.addEventListener("keydown", e => {
    if (["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); play(); }
    if (e.code === "ArrowLeft") { e.preventDefault(); setFrame(state.index - 1); }
    if (e.code === "ArrowRight") { e.preventDefault(); setFrame(state.index + 1); }
    if (e.key.toLowerCase() === "r") resetCamera();
  });
  global.addEventListener("resize", () => { resize(); state.dirty = true; });
  $("run-name").textContent = data.run_name;
  if (data.boundary === "spherical") {
    $("boundary-label").textContent = "Wall";
    $("focus-control").hidden = true;
    $("frame-origin-control").hidden = false;
    $("frame-origin").value = "sphere";
    const coordinate = Array.from($("frame-origin").options).find(option => option.value === "coordinate");
    if (!data.coordinate_frame || !data.coordinate_frame.available) {
      coordinate.disabled = true;
      coordinate.textContent = "Coordinate origin unavailable";
      $("frame-origin-control").title = data.coordinate_frame && data.coordinate_frame.reason || "Saved shift history unavailable";
    } else {
      $("frame-origin-control").title = data.coordinate_frame.convention;
    }
    $("focus").value = "box";
  }
  const nativeAvailable = Boolean(data.native_bonds && data.native_bonds.available);
  $("native-bonds").disabled = !nativeAvailable;
  $("native-bonds").checked = false;
  $("native-bonds-control").hidden = !nativeAvailable;
  $("native-bonds-control").title = nativeAvailable ? data.native_bonds.criterion : "Native geometry unavailable";
  $("body-count").textContent = `${data.body_count} tetramers`;
  $("sphere-count").textContent = `${(data.body_count * data.atoms.length).toLocaleString()} atom spheres`;
  $("frame-count").textContent = `${data.frames.length} saved frames`;
  $("frame").max = data.frames.length - 1;
  if (data.boundary !== "spherical") {
    for (let body = 0; body < data.body_count; body++) { const option = document.createElement("option"); option.value = `body:${body}`; option.textContent = `Body ${body}`; $("focus").appendChild(option); }
    if (data.frames[0].seed_labels.length) $("focus").value = "seed";
  }
  if (!data.frames[0].seed_labels.length) $("color").value = "body";
  updateGeometry(true);
  function tick(time) {
    if (state.playing && (!state.lastTick || time - state.lastTick >= 1000 / Number($("speed").value))) {
      setFrame((state.index + 1) % data.frames.length); state.lastTick = time;
    }
    if (state.dirty) render();
    global.requestAnimationFrame(tick);
  }
  global.requestAnimationFrame(tick);
  global.TetramerViewer = { setFrame, getFrame: () => state.index, renderer: rendererName, data, render };
})(globalThis);
