// 3-D view of the VAE latent space with three.js: one dot per minute, placed by the VAE's 6 latent numbers
// reduced to 3 with PCA. Drag to turn, scroll or pinch to zoom, hover a dot to read it.

import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { cssVar, el, fmt } from "./app.js";
import { motionAllowed } from "./motion.js";

const VIEWS = {
  split: { label: "Learning vs test", legend: [["learn", "February (learned as normal)"], ["test", "Test months"]] },
  alert: { label: "Alerts", legend: [["normal", "Below the VAE's alert line"], ["alert", "VAE alert"]] },
  event: { label: "Leaks", legend: [["other", "Any other minute"], ["event", "Inside a reported window"]] },
};

// Each view highlights one group of minutes (drawn on top) against the rest.
const HIGHLIGHT = {
  split: { pick: (p, i) => p.split[i] !== 2, on: "--brand", off: "--muted" },
  alert: { pick: (p, i) => p.alert[i] === 1, on: "--alert", off: "--line" },
  event: { pick: (p, i) => p.event[i] > 0, on: "--fail", off: "--line" },
};

/** Mount the explorer into `host`. `leakWord` is "leak" for MetroGuard and "reported problem" for uploads. */
export function latentExplorer(host, latent, { leakWord = "leak" } = {}) {
  const { points } = latent;
  const n = points.x.length;
  // Scale on the learned (February) cloud, not on outliers, so the normal minutes fill the view. Far minutes are
  // pulled in smoothly (asinh: linear near the centre, logarithmic far out), so their direction is kept.
  const learned = [];
  for (let i = 0; i < n; i++) if (points.split[i] === 0) learned.push(Math.abs(points.x[i]), Math.abs(points.y[i]), Math.abs(points.z[i]));
  learned.sort((a, b) => a - b);
  const span = learned[Math.floor(learned.length * 0.98)] || 1;
  const place = (v) => Math.asinh(v / span);
  let view = "split";

  const canvasHost = el("div", { class: "latent-canvas" });
  const tip = el("div", { class: "tip", role: "status" });
  const legend = el("div", { class: "legend" });
  const buttons = Object.entries(VIEWS).map(([key, v]) => el("button", {
    type: "button", "aria-pressed": String(key === view),
    onclick: () => { view = key; buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(Object.keys(VIEWS)[i] === view))); paint(); },
  }, v.label));
  host.replaceChildren(
    el("div", { class: "controls" }, el("span", { class: "control-label" }, "Colour by"), el("div", { class: "seg seg-plain", role: "group", "aria-label": "Colour the dots by" }, buttons)),
    el("div", { class: "latent-wrap" }, canvasHost, tip),
    legend,
  );

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  canvasHost.append(renderer.domElement);
  renderer.domElement.setAttribute("aria-label", `3-D cloud of ${fmt.int(n)} minutes placed by the VAE. Drag to rotate.`);
  renderer.domElement.setAttribute("role", "img");

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(2.6, 1.8, 3.1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = motionAllowed();
  controls.autoRotateSpeed = 0.6;
  controls.minDistance = 0.8;
  controls.maxDistance = 14;
  controls.addEventListener("start", () => { controls.autoRotate = false; });

  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    positions[i * 3] = place(points.x[i]);
    positions[i * 3 + 1] = place(points.y[i]);
    positions[i * 3 + 2] = place(points.z[i]);
  }
  const material = (size, opacity) => new THREE.PointsMaterial({ size, transparent: true, opacity, depthWrite: false });
  const base = new THREE.Points(new THREE.BufferGeometry(), material(0.035, 0.5));
  const top = new THREE.Points(new THREE.BufferGeometry(), material(0.05, 0.95));
  top.renderOrder = 1;
  scene.add(base, top);
  const axes = new THREE.AxesHelper(0.6);
  scene.add(axes);

  const groups = { top: [], base: [] };

  function paint() {
    const rule = HIGHLIGHT[view];
    groups.top = [];
    groups.base = [];
    for (let i = 0; i < n; i++) groups[rule.pick(points, i) ? "top" : "base"].push(i);
    const take = (rows) => Float32Array.from(rows.flatMap((i) => [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]]));
    base.geometry.setAttribute("position", new THREE.BufferAttribute(take(groups.base), 3));
    top.geometry.setAttribute("position", new THREE.BufferAttribute(take(groups.top), 3));
    base.geometry.computeBoundingSphere();
    top.geometry.computeBoundingSphere();
    base.material.color.set(cssVar(rule.off));
    top.material.color.set(cssVar(rule.on));
    const swatch = { learn: "--brand", test: "--muted", normal: "--line", alert: "--alert", other: "--line", event: "--fail" };
    legend.replaceChildren(...VIEWS[view].legend.map(([key, text]) => el("span", {}, el("i", { class: "sw-dot", style: `background:var(${swatch[key]})` }), text.replace("reported window", leakWord))));
  }

  function resize() {
    const width = canvasHost.clientWidth;
    const height = Math.min(520, Math.max(320, width * 0.62));
    renderer.setSize(width, height, false);
    renderer.domElement.style.width = `${width}px`;
    renderer.domElement.style.height = `${height}px`;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  const raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 0.04;
  const pointer = new THREE.Vector2();
  renderer.domElement.addEventListener("pointermove", (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects([top, base])[0];
    if (!hit) { tip.style.display = "none"; return; }
    const i = (hit.object === top ? groups.top : groups.base)[hit.index];
    const where = ["February, learned as normal", "End of February, sets the alert line", "Test months"][points.split[i]];
    tip.replaceChildren(
      el("b", {}, fmt.dateTime(points.minute[i])),
      el("div", {}, where),
      el("div", {}, points.alert[i] ? "VAE alert" : "Below the VAE's alert line"),
      points.event[i] ? el("div", { class: "over" }, `Inside ${leakWord} ${points.event[i]}`) : null,
    );
    tip.style.display = "block";
    tip.style.left = `${Math.min(event.clientX - rect.left + 14, rect.width - 190)}px`;
    tip.style.top = `${event.clientY - rect.top + 10}px`;
  });
  renderer.domElement.addEventListener("pointerleave", () => { tip.style.display = "none"; });

  let visible = true;
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(canvasHost);
  renderer.setAnimationLoop(() => {
    if (!visible) return;
    controls.update();
    renderer.render(scene, camera);
  });
  new ResizeObserver(resize).observe(canvasHost);
  window.addEventListener("themechange", paint);
  resize();
  paint();
}
