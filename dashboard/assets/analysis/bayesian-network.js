// Discrete Bayesian network in the browser (mirrors models/bayesian_network.py, which uses pgmpy): hill-climb
// structure search with a BIC score, BDeu parameters, exact inference by enumeration, d-separation, and
// Markov-blanket surprisal for alert context. Edges are learned dependencies, never causes.

import { createRng } from "./numeric.js";

const EPSILON = 1e-4;
const TIE_TOLERANCE = 1e-9;
const RESTARTS = 10;
const TABU_LENGTH = 100;
const MAX_JOINT_STATES = 2_000_000;
const LAYOUT_SWEEPS = 4;

function graph(nodes) {
  const parents = new Map(nodes.map((n) => [n, new Set()]));
  const children = new Map(nodes.map((n) => [n, new Set()]));
  return {
    parents, children,
    add(x, y) { parents.get(y).add(x); children.get(x).add(y); },
    remove(x, y) { parents.get(y).delete(x); children.get(x).delete(y); },
    has(x, y) { return parents.get(y).has(x); },
    reachable(from, to, skipDirect = false) {
      const stack = [...children.get(from)].filter((c) => !(skipDirect && c === to));
      const seen = new Set();
      while (stack.length) {
        const n = stack.pop();
        if (n === to) return true;
        if (seen.has(n)) continue;
        seen.add(n);
        stack.push(...children.get(n));
      }
      return false;
    },
    edges() { const out = []; for (const [y, ps] of parents) for (const x of ps) out.push([x, y]); return out.sort((a, b) => (a[0] + "\u0000" + a[1]).localeCompare(b[0] + "\u0000" + b[1])); },
  };
}

/**
 * Hill-climb search with the BIC score (pgmpy "bic-d"). A single edge X->Y and Y->X score exactly the same, so the
 * search is path-dependent; it is run from several tie-break orders and the structure with the best total BIC
 * is kept. Returns { edges, score, scores } with edges sorted as [parent, child].
 */
export function learnStructure(data, nodes, cards, maxIndegree, seed = 42) {
  const local = bicScorer(data, cards);
  const rng = createRng(seed);
  const runs = [];
  for (let r = 0; r < RESTARTS; r++) {
    const order = r === 0 ? nodes : Array.from(rng.permutation(nodes.length), (i) => nodes[i]);
    const edges = hillClimb(order, local, maxIndegree);
    const score = nodes.reduce((sum, node) => sum + local(node, new Set(edges.filter(([, c]) => c === node).map(([p]) => p))), 0);
    runs.push({ edges, score });
  }
  const best = runs.reduce((a, b) => (b.score > a.score + 1e-9 ? b : a));
  return { edges: best.edges, score: best.score, scores: runs.map((x) => x.score) };
}

function bicScorer(data, cards) {
  const n = data[Object.keys(data)[0]].length;
  const logN = Math.log(n);
  const cache = new Map();
  const local = (node, parentSet) => {
    const ps = [...parentSet].sort();
    const key = `${node}|${ps.join(",")}`;
    if (cache.has(key)) return cache.get(key);
    const r = cards[node];
    const q = ps.reduce((m, p) => m * cards[p], 1);
    const counts = new Float64Array(q * r);
    const col = data[node];
    const pcols = ps.map((p) => data[p]);
    const mult = ps.map((_, i) => ps.slice(i + 1).reduce((m, p) => m * cards[p], 1));
    for (let row = 0; row < n; row++) {
      let cfg = 0;
      for (let k = 0; k < ps.length; k++) cfg += pcols[k][row] * mult[k];
      counts[cfg * r + col[row]]++;
    }
    let ll = 0;
    for (let c = 0; c < q; c++) {
      let total = 0;
      for (let s = 0; s < r; s++) total += counts[c * r + s];
      for (let s = 0; s < r; s++) { const v = counts[c * r + s]; if (v > 0) ll += v * Math.log(v / total); }
    }
    const score = ll - 0.5 * logN * q * (r - 1);
    cache.set(key, score);
    return score;
  };
  return local;
}

function hillClimb(nodes, local, maxIndegree) {
  const g = graph(nodes);
  const tabu = [];
  const without = (set, x) => { const s = new Set(set); s.delete(x); return s; };
  const withX = (set, x) => new Set([...set, x]);
  for (let iteration = 0; iteration < 1_000_000; iteration++) {
    let best = null;
    let bestDelta = -Infinity;
    // Score-equivalent moves (X->Y versus Y->X) differ only by rounding, which varies between JavaScript engines;
    // treating near-equal deltas as ties keeps the first move in iteration order, so every browser learns the same graph.
    const consider = (op, delta) => {
      if (best === null || delta > bestDelta + TIE_TOLERANCE * Math.max(1, Math.abs(bestDelta))) { bestDelta = delta; best = op; }
    };
    for (const x of nodes) for (const y of nodes) {
      if (x === y || g.has(x, y) || g.has(y, x)) continue;
      const op = `+|${x}|${y}`;
      if (tabu.includes(op) || g.reachable(y, x) || g.parents.get(y).size + 1 > maxIndegree) continue;
      const ps = g.parents.get(y);
      consider({ kind: "+", x, y }, local(y, withX(ps, x)) - local(y, ps));
    }
    for (const [x, y] of g.edges()) {
      if (tabu.includes(`-|${x}|${y}`)) continue;
      const ps = g.parents.get(y);
      consider({ kind: "-", x, y }, local(y, without(ps, x)) - local(y, ps));
    }
    for (const [x, y] of g.edges()) {
      if (tabu.includes(`flip|${x}|${y}`) || tabu.includes(`flip|${y}|${x}`)) continue;
      if (g.reachable(x, y, true) || g.parents.get(x).size + 1 > maxIndegree) continue;
      const px = g.parents.get(x);
      const py = g.parents.get(y);
      consider({ kind: "flip", x, y }, local(x, withX(px, y)) - local(x, px) + local(y, without(py, x)) - local(y, py));
    }
    if (!best || bestDelta < EPSILON) break;
    if (best.kind === "+") { g.add(best.x, best.y); tabu.push(`-|${best.x}|${best.y}`); }
    else if (best.kind === "-") { g.remove(best.x, best.y); tabu.push(`+|${best.x}|${best.y}`); }
    else { g.remove(best.x, best.y); g.add(best.y, best.x); tabu.push(`flip|${best.x}|${best.y}`); }
    if (tabu.length > TABU_LENGTH) tabu.shift();
  }
  return g.edges();
}

/** BDeu conditional probability tables; node order follows first appearance in the edges, then the rest. */
export function fitNetwork(edges, data, nodeColumns, cards, ess) {
  const order = [];
  for (const [a, b] of edges) for (const n of [a, b]) if (!order.includes(n)) order.push(n);
  for (const n of nodeColumns) if (!order.includes(n)) order.push(n);
  const g = graph(order);
  edges.forEach(([a, b]) => g.add(a, b));
  const n = data[order[0]].length;
  const cpds = {};
  for (const node of order) {
    const ps = order.filter((p) => g.parents.get(node).has(p));
    const r = cards[node];
    const q = ps.reduce((m, p) => m * cards[p], 1);
    const mult = ps.map((_, i) => ps.slice(i + 1).reduce((m, p) => m * cards[p], 1));
    const counts = new Float64Array(q * r);
    for (let row = 0; row < n; row++) {
      let cfg = 0;
      for (let k = 0; k < ps.length; k++) cfg += data[ps[k]][row] * mult[k];
      counts[cfg * r + data[node][row]]++;
    }
    const alpha = ess / (r * q);
    const table = new Float64Array(q * r);
    for (let c = 0; c < q; c++) {
      let total = 0;
      for (let s = 0; s < r; s++) total += counts[c * r + s];
      for (let s = 0; s < r; s++) table[c * r + s] = (counts[c * r + s] + alpha) / (total + ess / q);
    }
    cpds[node] = { parents: ps, mult, r, table };
  }
  return { nodes: order, edges, cards, cpds, graph: g };
}

const cpdValue = (cpd, state, assignment) => {
  let cfg = 0;
  for (let k = 0; k < cpd.parents.length; k++) cfg += assignment[cpd.parents[k]] * cpd.mult[k];
  return cpd.table[cfg * cpd.r + state];
};

/** P(target | evidenceVariable = s) for every s, by enumerating the joint distribution. */
export function conditionalTable(model, target, evidenceVariable) {
  const { nodes, cards, cpds } = model;
  const total = nodes.reduce((m, n) => m * cards[n], 1);
  if (total > MAX_JOINT_STATES) return null;
  const joint = new Float64Array(cards[target] * cards[evidenceVariable]);
  const assignment = Object.fromEntries(nodes.map((n) => [n, 0]));
  for (let index = 0; index < total; index++) {
    let rest = index;
    for (let k = nodes.length - 1; k >= 0; k--) { assignment[nodes[k]] = rest % cards[nodes[k]]; rest = Math.floor(rest / cards[nodes[k]]); }
    let p = 1;
    for (const n of nodes) p *= cpdValue(cpds[n], assignment[n], assignment);
    joint[assignment[target] * cards[evidenceVariable] + assignment[evidenceVariable]] += p;
  }
  const columns = ["", ...Array.from({ length: cards[evidenceVariable] }, (_, e) => `${evidenceVariable}=${e}`)];
  const rows = Array.from({ length: cards[target] }, (_, t) => [`${target}=${t}`, ...Array.from({ length: cards[evidenceVariable] }, (_, e) => {
    let col = 0;
    for (let tt = 0; tt < cards[target]; tt++) col += joint[tt * cards[evidenceVariable] + e];
    return String(Number((joint[t * cards[evidenceVariable] + e] / col).toFixed(4)));
  })]);
  return { columns, rows };
}

/** The learned edge whose endpoints have the highest empirical mutual information (nats). */
export function strongestEdge(model, data) {
  let best = null;
  for (const [p, c] of model.edges) {
    const rp = model.cards[p];
    const rc = model.cards[c];
    const joint = new Float64Array(rp * rc);
    const n = data[p].length;
    for (let i = 0; i < n; i++) joint[data[p][i] * rc + data[c][i]]++;
    const rowSum = new Float64Array(rp);
    const colSum = new Float64Array(rc);
    for (let a = 0; a < rp; a++) for (let b = 0; b < rc; b++) { rowSum[a] += joint[a * rc + b] / n; colSum[b] += joint[a * rc + b] / n; }
    let mi = 0;
    for (let a = 0; a < rp; a++) for (let b = 0; b < rc; b++) { const pj = joint[a * rc + b] / n; if (pj > 0) mi += pj * Math.log(pj / (rowSum[a] * colSum[b])); }
    if (!best || mi > best.mi) best = { parent: p, child: c, mi };
  }
  return best;
}

/** Nodes reachable from `start` by an active trail given `observed` (Koller & Friedman, Algorithm 3.1). */
function activeTrailNodes(model, start, observed) {
  const { parents, children } = model.graph;
  const obs = new Set(observed);
  const ancestors = new Set();
  const stack = [...observed];
  while (stack.length) { const y = stack.pop(); if (!ancestors.has(y)) { ancestors.add(y); stack.push(...parents.get(y)); } }
  const visit = [[start, "up"]];
  const visited = new Set();
  const reachable = new Set();
  while (visit.length) {
    const [y, d] = visit.pop();
    const key = `${y}|${d}`;
    if (visited.has(key)) continue;
    visited.add(key);
    if (!obs.has(y)) reachable.add(y);
    if (d === "up" && !obs.has(y)) {
      for (const p of parents.get(y)) visit.push([p, "up"]);
      for (const c of children.get(y)) visit.push([c, "down"]);
    } else if (d === "down") {
      if (!obs.has(y)) for (const c of children.get(y)) visit.push([c, "down"]);
      if (ancestors.has(y)) for (const p of parents.get(y)) visit.push([p, "up"]);
    }
  }
  return reachable;
}

const dConnected = (model, a, b, observed) => activeTrailNodes(model, a, observed).has(b);
const pyBool = (v) => (v ? "True" : "False");

/** Node vs a parent, vs a non-parent non-descendant (given parents), vs a descendant: bullet sentences. */
export function localIndependenceCases(model) {
  const { parents, children } = model.graph;
  const descendants = (node) => { const found = new Set(); const st = [...children.get(node)]; while (st.length) { const c = st.pop(); if (!found.has(c)) { found.add(c); st.push(...children.get(c)); } } return found; };
  for (const node of model.nodes) {
    const ps = [...parents.get(node)].sort();
    const cs = [...children.get(node)].sort();
    const desc = descendants(node);
    const others = model.nodes.filter((n) => n !== node && !desc.has(n) && !ps.includes(n)).sort();
    if (!ps.length || !cs.length || !others.length) continue;
    const relation = (connected) => (connected ? "dependent (d-connected)" : "independent (d-separated)");
    return [
      `node and a parent: ${node} and ${ps[0]}, given nothing: **${relation(dConnected(model, node, ps[0], []))}**`,
      `node and a non-parent non-descendant, given its parents: ${node} and ${others[0]}, given ${ps.join(", ")}: **${relation(dConnected(model, node, others[0], ps))}**`,
      `node and a descendant: ${node} and ${cs[0]}, given nothing: **${relation(dConnected(model, node, cs[0], []))}**`,
    ];
  }
  return [];
}

/** A chain, fork and collider, preferring those where observing the middle changes d-connection. */
export function structurePatterns(model) {
  const { parents, children } = model.graph;
  const adjacent = (a, b) => model.graph.has(a, b) || model.graph.has(b, a);
  const out = [];
  for (const kind of ["chain", "fork", "collider"]) {
    const candidates = [];
    for (const middle of [...model.nodes].sort()) {
      const ps = [...parents.get(middle)].sort();
      const cs = [...children.get(middle)].sort();
      const pairs = kind === "chain" ? ps.flatMap((p) => cs.map((c) => [p, c]))
        : kind === "fork" ? cs.flatMap((a, i) => cs.slice(i + 1).map((b) => [a, b]))
        : ps.flatMap((a, i) => ps.slice(i + 1).map((b) => [a, b]));
      for (const [a, b] of pairs) {
        if (adjacent(a, b)) continue;
        candidates.push({ kind, a, b, middle, without: dConnected(model, a, b, []), given: dConnected(model, a, b, [middle]) });
      }
    }
    const chosen = candidates.find((c) => c.without !== c.given) ?? candidates[0];
    if (chosen) out.push(`${kind} ${chosen.a} / ${chosen.middle} / ${chosen.b}: connected without evidence = ${pyBool(chosen.without)}; connected given ${chosen.middle} = ${pyBool(chosen.given)}`);
  }
  return out;
}

/** -log P(observed state | Markov blanket) per row, from the CPD factors. */
export function blanketSurprisal(model, rows, variable) {
  const { cpds, cards, graph: g } = model;
  const factors = [variable, ...g.children.get(variable)].map((n) => ({ n, cpd: cpds[n] }));
  return rows.map((assignment) => {
    let observed = 0;
    let total = 0;
    for (let s = 0; s < cards[variable]; s++) {
      const a = { ...assignment, [variable]: s };
      let p = 1;
      for (const f of factors) p *= cpdValue(f.cpd, a[f.n], a);
      total += p;
      if (s === assignment[variable]) observed = p;
    }
    return -Math.log(observed / total);
  });
}

/** Deterministic layered layout in [0, 1]: parents left of children (same rule as dashboard_export.py). */
export function layeredLayout(nodes, edges) {
  const g = graph(nodes);
  edges.forEach(([a, b]) => g.add(a, b));
  const indegree = new Map(nodes.map((n) => [n, g.parents.get(n).size]));
  const layers = [];
  let frontier = nodes.filter((n) => indegree.get(n) === 0).sort();
  while (frontier.length) {
    layers.push(frontier);
    const next = [];
    for (const n of frontier) for (const c of g.children.get(n)) { indegree.set(c, indegree.get(c) - 1); if (indegree.get(c) === 0) next.push(c); }
    frontier = next.sort();
  }
  const barycentre = (neighbours, order) => { const pos = [...neighbours].filter((n) => order.has(n)).map((n) => order.get(n)); return pos.length ? pos.reduce((s, v) => s + v, 0) / pos.length : 0; };
  for (let sweep = 0; sweep < LAYOUT_SWEEPS; sweep++) {
    for (let i = 1; i < layers.length; i++) { const order = new Map(layers[i - 1].map((n, j) => [n, j])); layers[i].sort((a, b) => barycentre(g.parents.get(a), order) - barycentre(g.parents.get(b), order)); }
    for (let i = layers.length - 2; i >= 0; i--) { const order = new Map(layers[i + 1].map((n, j) => [n, j])); layers[i].sort((a, b) => barycentre(g.children.get(a), order) - barycentre(g.children.get(b), order)); }
  }
  const last = Math.max(layers.length - 1, 1);
  return layers.flatMap((layer, col) => layer.map((id, j) => ({ id, x: Number((col / last).toFixed(4)), y: Number(((j + 0.5) / layer.length).toFixed(4)) })));
}
