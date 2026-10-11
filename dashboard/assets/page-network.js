import { el, fmt, footer, initShell, loadJSON, plainTable, showError, svg } from "./app.js";

const WIDTH = 760;
const HEIGHT = 540;
const PAD = 70;
const RADIUS = 24;

const short = (id) => id.replace(/_mean$/, "").replace(/_active$/, "");
const isAnalog = (id) => id.endsWith("_mean");

function blanket(edges, id) {
  const parents = edges.filter((e) => e.child === id).map((e) => e.parent);
  const children = edges.filter((e) => e.parent === id).map((e) => e.child);
  const coParents = edges.filter((e) => children.includes(e.child) && e.parent !== id).map((e) => e.parent);
  return { parents, children, blanket: new Set([...parents, ...children, ...coParents]) };
}

function heatCell(text) {
  const value = Number(text);
  if (Number.isNaN(value)) return text;
  return el("span", { class: "heat-cell num", style: `background:color-mix(in srgb, var(--brand) ${Math.round(value * 70)}%, transparent)` }, value.toFixed(3));
}

function conditionalTable(heading, table) {
  return plainTable({
    caption: heading,
    columns: table.columns.map((c, i) => ({ label: i === 0 ? "" : c, align: i === 0 ? "left" : "right", render: (r) => (i === 0 ? el("b", {}, r[i]) : heatCell(r[i])) })),
    rows: table.rows,
    className: "cond",
  });
}

try {
  const [context, network] = await Promise.all([loadJSON("context"), loadJSON("network")]);
  const { wrap } = initShell(
    "network.html",
    "Bayesian network",
    "A network of sensor states learned on the training month only. Arrows are statistical dependencies, not causes, and some directions are arbitrary.",
  );

  const { nodes, edges } = network;
  const position = Object.fromEntries(nodes.map((n) => [n.id, [PAD + n.x * (WIDTH - 2 * PAD), PAD * 0.6 + n.y * (HEIGHT - PAD * 1.4)]]));
  let selected = null;

  const root = svg("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, role: "group", "aria-label": "Learned Bayesian network. Select a node to highlight its Markov blanket." });
  root.append(svg("defs", {}, svg("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, svg("path", { d: "M0 0 L10 5 L0 10 z", fill: "context-stroke" }))));
  const edgeEls = edges.map((e) => {
    const [x1, y1] = position[e.parent];
    const [x2, y2] = position[e.child];
    const startX = x1 + RADIUS;
    const endX = x2 - RADIUS - 5;
    const bend = Math.max(40, (endX - startX) * 0.5);
    const path = svg("path", { class: "edge", d: `M${startX} ${y1} C${startX + bend} ${y1} ${endX - bend} ${y2} ${endX} ${y2}`, "marker-end": "url(#arrow)" });
    root.append(path);
    return { e, path };
  });
  const nodeEls = nodes.map((n) => {
    const [x, y] = position[n.id];
    const g = svg("g", { class: `node ${isAnalog(n.id) ? "analog" : "digital"}`, tabindex: "0", role: "button", "aria-label": `${short(n.id)}: select to see its neighbours` },
      svg("circle", { cx: x, cy: y, r: RADIUS }),
      svg("text", { x, y: y + RADIUS + 15, "text-anchor": "middle" }, short(n.id)),
    );
    const pick = () => { selected = selected === n.id ? null : n.id; refresh(); };
    g.addEventListener("click", pick);
    g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } });
    root.append(g);
    return { n, g };
  });

  const side = el("div");
  function refresh() {
    const info = selected ? blanket(edges, selected) : null;
    nodeEls.forEach(({ n, g }) => {
      g.classList.toggle("sel", n.id === selected);
      g.classList.toggle("hl", !!info && info.blanket.has(n.id));
      g.classList.toggle("dim", !!info && n.id !== selected && !info.blanket.has(n.id));
    });
    edgeEls.forEach(({ e, path }) => {
      const touches = info && (e.parent === selected || e.child === selected);
      path.classList.toggle("hl", !!touches);
      path.classList.toggle("dim", !!info && !touches);
    });
    side.replaceChildren(
      info
        ? el("div", { class: "note accent" },
            el("p", {}, el("strong", {}, short(selected)), isAnalog(selected) ? " (analog window mean)" : " (digital, active for at least half the window)"),
            el("p", {}, `Parents: ${info.parents.map(short).join(", ") || "none"}.`),
            el("p", {}, `Children: ${info.children.map(short).join(", ") || "none"}.`),
            el("p", { class: "muted small" }, `Markov blanket (${info.blanket.size} nodes): the variables that, once known, make this one independent of the rest.`))
        : el("div", { class: "note" }, el("p", {}, "Select a node to see its parents, children and Markov blanket. Teal outlines are analog sensors; grey outlines are digital signals.")),
    );
  }

  const demo = network.demonstration;
  const alert = network.alert_context;
  wrap.append(
    el("div", { class: "net" },
      el("div", { class: "net-canvas" }, root),
      el("div", {}, side,
        plainTable({
          caption: `Learned edges (${edges.length})`,
          note: "Hill-climb search with a BIC score on the training windows.",
          columns: [{ label: "Parent", render: (r) => short(r.parent) }, { label: "Child", render: (r) => short(r.child) }],
          rows: edges,
        })),
    ),
    el("div", { class: "note warn" },
      el("p", {}, el("strong", {}, "These arrows are not causes. "), "Several edges belong to a group of directions the data cannot tell apart: in two otherwise identical runs, some edges came back pointing the other way. The structure shows which sensor states vary together, nothing more.")),

    el("h2", {}, "What the network predicts"),
    el("p", {}, `The strongest learned dependency (mutual information ${fmt.num(demo.mutual_information, 3)} nats) is between ${short(demo.parent)} and ${short(demo.child)}. They are two pressure measurements on the same circuit, so each state of one almost fixes the state of the other, in either direction.`),
    el("div", { class: "grid-2" },
      el("div", {}, conditionalTable(demo.causal_heading, demo.causal_table)),
      el("div", {}, conditionalTable(demo.evidential_heading, demo.evidential_table)),
    ),

    el("h2", {}, "Independence checks"),
    el("p", {}, "The three local-independence cases and the three d-separation patterns, evaluated on the learned graph."),
    el("div", { class: "grid-2" },
      el("div", {}, el("h3", {}, "Local independence"), el("ul", { class: "list" }, demo.local_independence.map((t) => el("li", {}, t.replace(/\*\*/g, ""))))),
      el("div", {}, el("h3", {}, "Chain, fork and collider"), el("ul", { class: "list" }, demo.d_separation.map((t) => el("li", {}, t.replace(/connected/g, "d-connected"))))),
    ),

    el("h2", {}, "What the network says about two alerts"),
    el("p", {}, `For the alert events the ${context.model_labels[alert.model]} raised inside reported windows 1 and 2, each variable's observed state was scored by how unlikely it is given the variables around it. The column "share above calibration p95" is the fraction of event windows more surprising than the 95th percentile of the training month's calibration windows; a variable is marked unusual when that share is more than twice what chance would give (5%).`),
    ...alert.events.flatMap((event) => [
      el("h3", {}, event.heading.replace(/^Reported window (\d+): alert event (.+) to (.+) \((\d+) usable windows\)$/, (m, n, a, b, w) => `Reported window ${n}: alert event ${a} to ${b} (${w} usable windows)`)),
      plainTable({
        columns: [
          { label: "Variable", render: (r) => el("b", {}, short(r.variable)) },
          { label: "Event mean", align: "right", render: (r) => fmt.num(r.event_mean, 4) },
          { label: "Calibration mean", align: "right", render: (r) => fmt.num(r.calib_mean, 4) },
          { label: "Calibration p95", align: "right", render: (r) => fmt.num(r.calib_p95, 4) },
          { label: "Share above calibration p95", align: "right", render: (r) => fmt.pct(r.share_above_calib_p95) },
          { label: "Unusual", render: (r) => (r.unusual === "yes" ? el("span", { class: "tag warn" }, "unusual") : el("span", { class: "muted" }, "no")) },
        ],
        rows: event.table,
      }),
    ]),
    el("p", { class: "muted small" }, "Values are −log P(state | Markov blanket) in nats. The flagged variables are digital compressor signals and the effects are small in absolute terms. This describes association with the alert, not a cause."),
    footer(context),
  );
  refresh();
} catch (error) {
  initShell("network.html", "Bayesian network");
  showError(error);
}
