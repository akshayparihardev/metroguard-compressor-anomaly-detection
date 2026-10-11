// Bernoulli-Bernoulli RBM trained with CD-k, written from scratch (mirrors models/restricted_boltzmann_machine.py).
// Energy E(v,h) = -a.v - b.h - v.W.h; anomaly score = free energy F(v) (higher = less typical).

import { sigmoid, softplus } from "./numeric.js";

const INIT_STD = 0.01;

export function createRbm(nVisible, nHidden, rng) {
  const W = new Float64Array(nVisible * nHidden);
  for (let i = 0; i < W.length; i++) W[i] = rng.normal() * INIT_STD;
  return { nVisible, nHidden, W, a: new Float64Array(nVisible), b: new Float64Array(nHidden), vW: new Float64Array(W.length), va: new Float64Array(nVisible), vb: new Float64Array(nHidden) };
}

/** p(h=1|v) for a batch of rows (row-major), written into out. */
function hiddenProbabilities(rbm, v, rows, out) {
  const { nVisible: nv, nHidden: nh, W, b } = rbm;
  for (let s = 0; s < rows; s++) {
    for (let j = 0; j < nh; j++) out[s * nh + j] = b[j];
    for (let i = 0; i < nv; i++) {
      const x = v[s * nv + i];
      if (x === 0) continue;
      const row = i * nh;
      for (let j = 0; j < nh; j++) out[s * nh + j] += x * W[row + j];
    }
    for (let j = 0; j < nh; j++) out[s * nh + j] = sigmoid(out[s * nh + j]);
  }
}

/** p(v=1|h) for a batch of rows. */
function visibleProbabilities(rbm, h, rows, out) {
  const { nVisible: nv, nHidden: nh, W, a } = rbm;
  for (let s = 0; s < rows; s++) {
    for (let i = 0; i < nv; i++) {
      let sum = a[i];
      const row = i * nh;
      for (let j = 0; j < nh; j++) sum += h[s * nh + j] * W[row + j];
      out[s * nv + i] = sigmoid(sum);
    }
  }
}

const sample = (probs, rng, out) => { for (let i = 0; i < probs.length; i++) out[i] = rng.uniform() < probs[i] ? 1 : 0; };

/** One CD-k mini-batch update with momentum and L2 weight decay on W. */
function cdStep(rbm, v0, rows, k, lr, momentum, decay, rng, work) {
  const { nVisible: nv, nHidden: nh, W, a, b } = rbm;
  const { ph0, hs, pv, vk, phk } = work;
  hiddenProbabilities(rbm, v0, rows, ph0);
  let current = v0;
  for (let step = 0; step < k; step++) {
    hiddenProbabilities(rbm, current, rows, phk);
    sample(phk.subarray(0, rows * nh), rng, hs);
    visibleProbabilities(rbm, hs, rows, pv);
    if (step === k - 1) vk.set(pv.subarray(0, rows * nv));
    else sample(pv.subarray(0, rows * nv), rng, vk);
    current = vk;
  }
  hiddenProbabilities(rbm, vk, rows, phk);
  for (let i = 0; i < nv; i++) {
    for (let j = 0; j < nh; j++) {
      let g = 0;
      for (let s = 0; s < rows; s++) g += v0[s * nv + i] * ph0[s * nh + j] - vk[s * nv + i] * phk[s * nh + j];
      const idx = i * nh + j;
      rbm.vW[idx] = momentum * rbm.vW[idx] + lr * (g / rows - decay * W[idx]);
      W[idx] += rbm.vW[idx];
    }
  }
  for (let i = 0; i < nv; i++) {
    let g = 0;
    for (let s = 0; s < rows; s++) g += v0[s * nv + i] - vk[s * nv + i];
    rbm.va[i] = momentum * rbm.va[i] + lr * (g / rows);
    a[i] += rbm.va[i];
  }
  for (let j = 0; j < nh; j++) {
    let g = 0;
    for (let s = 0; s < rows; s++) g += ph0[s * nh + j] - phk[s * nh + j];
    rbm.vb[j] = momentum * rbm.vb[j] + lr * (g / rows);
    b[j] += rbm.vb[j];
  }
}

/** F(v) = -a.v - sum_j softplus(b_j + (v.W)_j) for each row of a row-major matrix. */
export function freeEnergy(rbm, visible, rows) {
  const { nVisible: nv, nHidden: nh, W, a, b } = rbm;
  const out = new Float64Array(rows);
  const hidden = new Float64Array(nh);
  for (let s = 0; s < rows; s++) {
    let av = 0;
    hidden.set(b);
    for (let i = 0; i < nv; i++) {
      const x = visible[s * nv + i];
      if (x === 0) continue;
      av += a[i] * x;
      const row = i * nh;
      for (let j = 0; j < nh; j++) hidden[j] += x * W[row + j];
    }
    let sp = 0;
    for (let j = 0; j < nh; j++) sp += softplus(hidden[j]);
    out[s] = -av - sp;
  }
  return out;
}

/** Mean squared error of v versus p(v | p(h | v)), averaged over units and rows. */
export function reconstructionError(rbm, visible, rows) {
  const { nVisible: nv, nHidden: nh } = rbm;
  const chunk = 1024;
  const h = new Float64Array(chunk * nh);
  const p = new Float64Array(chunk * nv);
  let total = 0;
  for (let start = 0; start < rows; start += chunk) {
    const count = Math.min(chunk, rows - start);
    const v = visible.subarray(start * nv, (start + count) * nv);
    hiddenProbabilities(rbm, v, count, h);
    visibleProbabilities(rbm, h, count, p);
    for (let i = 0; i < count * nv; i++) total += (v[i] - p[i]) ** 2;
  }
  return total / (rows * nv);
}

const meanOf = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;

/** Train with CD-k; returns the per-epoch history used by the training curves. */
export function trainRbm(rbm, fitData, fitRows, validationData, validationRows, options, rng, onEpoch = () => {}) {
  const { k, epochs, batchSize, learningRate, momentum, weightDecay } = options;
  const { nVisible: nv, nHidden: nh } = rbm;
  const work = {
    ph0: new Float64Array(batchSize * nh), hs: new Float64Array(batchSize * nh), pv: new Float64Array(batchSize * nv),
    vk: new Float64Array(batchSize * nv), phk: new Float64Array(batchSize * nh),
  };
  const batch = new Float64Array(batchSize * nv);
  const history = { epoch: [], fit_reconstruction_error: [], fit_mean_free_energy: [], validation_reconstruction_error: [], validation_mean_free_energy: [] };
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const order = rng.permutation(fitRows);
    for (let start = 0; start < fitRows; start += batchSize) {
      const rows = Math.min(batchSize, fitRows - start);
      for (let s = 0; s < rows; s++) batch.set(fitData.subarray(order[start + s] * nv, (order[start + s] + 1) * nv), s * nv);
      cdStep(rbm, batch, rows, k, learningRate, momentum, weightDecay, rng, work);
    }
    history.epoch.push(epoch);
    history.fit_reconstruction_error.push(reconstructionError(rbm, fitData, fitRows));
    history.fit_mean_free_energy.push(meanOf(freeEnergy(rbm, fitData, fitRows)));
    history.validation_reconstruction_error.push(reconstructionError(rbm, validationData, validationRows));
    history.validation_mean_free_energy.push(meanOf(freeEnergy(rbm, validationData, validationRows)));
    onEpoch(epoch / epochs);
  }
  return history;
}
