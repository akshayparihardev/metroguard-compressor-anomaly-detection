// Numeric helpers shared by the in-browser pipeline: a seeded RNG and NumPy-compatible statistics.

/** Deterministic 32-bit PRNG (mulberry32) so a given file and seed always give the same result. */
export function createRng(seed) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    while (u === 0) u = uniform();
    const v = uniform();
    const radius = Math.sqrt(-2 * Math.log(u));
    spare = radius * Math.sin(2 * Math.PI * v);
    return radius * Math.cos(2 * Math.PI * v);
  };
  const permutation = (n) => {
    const order = new Uint32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(uniform() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    return order;
  };
  return { uniform, normal, permutation };
}

/** Sorted copy of the finite values (Float64). */
export function sortedFinite(values) {
  let count = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) count++;
  const out = new Float64Array(count);
  let j = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) out[j++] = values[i];
  return out.sort();
}

/** Quantile with linear interpolation, matching numpy.quantile's default method. */
export function quantileSorted(sorted, q) {
  if (!sorted.length) return NaN;
  const position = q * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(lower + 1, sorted.length - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export const percentile = (values, p) => quantileSorted(sortedFinite(values), p / 100);
export const median = (values) => quantileSorted(sortedFinite(values), 0.5);

/** Mean and population standard deviation (ddof = 0) of the finite values. */
export function meanStd(values) {
  let n = 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) { sum += values[i]; n++; }
  const mean = n ? sum / n : NaN;
  let sq = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) sq += (values[i] - mean) ** 2;
  return { mean, std: n ? Math.sqrt(sq / n) : NaN, n };
}

/** numpy.digitize(x, edges) with right=False: the number of edges that are <= x. */
export function digitize(x, edges) {
  let i = 0;
  while (i < edges.length && edges[i] <= x) i++;
  return i;
}

export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/** Numerically stable softplus, log(1 + exp(x)). */
export const softplus = (x) => (x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x)));

/** "Nice" step (1, 2 or 5 times a power of ten) close to range / target. */
export function niceStep(range, target) {
  const raw = range / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / magnitude;
  return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * magnitude;
}

export const round = (value, digits) => (Number.isFinite(value) ? Number(value.toFixed(digits)) : null);
