// Small MLP variational autoencoder with hand-written backpropagation and Adam
// (mirrors models/variational_autoencoder.py: in -> 64 -> 32 -> (mu, log var) ; z -> 32 -> 64 -> in).

const BETA1 = 0.9;
const BETA2 = 0.999;
const ADAM_EPS = 1e-8;

/** Dense layer y = xW + b with PyTorch's default init U(-1/sqrt(in), 1/sqrt(in)). */
function dense(nIn, nOut, rng) {
  const bound = 1 / Math.sqrt(nIn);
  const W = Float64Array.from({ length: nIn * nOut }, () => (rng.uniform() * 2 - 1) * bound);
  const b = Float64Array.from({ length: nOut }, () => (rng.uniform() * 2 - 1) * bound);
  return { nIn, nOut, W, b, gW: new Float64Array(W.length), gb: new Float64Array(nOut), m: [new Float64Array(W.length), new Float64Array(nOut)], v: [new Float64Array(W.length), new Float64Array(nOut)] };
}

function forward(layer, x, rows, out) {
  const { nIn, nOut, W, b } = layer;
  for (let s = 0; s < rows; s++) {
    for (let o = 0; o < nOut; o++) out[s * nOut + o] = b[o];
    for (let i = 0; i < nIn; i++) {
      const xi = x[s * nIn + i];
      if (xi === 0) continue;
      const row = i * nOut;
      for (let o = 0; o < nOut; o++) out[s * nOut + o] += xi * W[row + o];
    }
  }
}

/** Accumulates gradients for W and b; writes dL/dx into dx when given. */
function backward(layer, x, dy, rows, dx) {
  const { nIn, nOut, W, gW, gb } = layer;
  for (let s = 0; s < rows; s++) {
    for (let o = 0; o < nOut; o++) gb[o] += dy[s * nOut + o];
    for (let i = 0; i < nIn; i++) {
      const xi = x[s * nIn + i];
      const row = i * nOut;
      let acc = 0;
      for (let o = 0; o < nOut; o++) {
        const g = dy[s * nOut + o];
        gW[row + o] += xi * g;
        acc += W[row + o] * g;
      }
      if (dx) dx[s * nIn + i] = acc;
    }
  }
}

const relu = (arr, n) => { for (let i = 0; i < n; i++) if (arr[i] < 0) arr[i] = 0; };
const reluGrad = (grad, activated, n) => { for (let i = 0; i < n; i++) if (activated[i] <= 0) grad[i] = 0; };

export function createVae(nFeatures, hiddenSizes, latentDim, rng) {
  const [h1, h2] = hiddenSizes;
  return {
    nFeatures, latentDim,
    layers: { enc1: dense(nFeatures, h1, rng), enc2: dense(h1, h2, rng), mu: dense(h2, latentDim, rng), logVar: dense(h2, latentDim, rng), dec1: dense(latentDim, h2, rng), dec2: dense(h2, h1, rng), dec3: dense(h1, nFeatures, rng) },
    step: 0,
  };
}

function buffers(vae, rows) {
  const { nFeatures: f, latentDim: d, layers } = vae;
  const h1 = layers.enc1.nOut;
  const h2 = layers.enc2.nOut;
  const make = (n) => new Float64Array(rows * n);
  return { a1: make(h1), a2: make(h2), mu: make(d), lv: make(d), eps: make(d), z: make(d), d1: make(h2), d2: make(h1), out: make(f), gOut: make(f), gD2: make(h1), gD1: make(h2), gZ: make(d), gMu: make(d), gLv: make(d), gA2: make(h2), gA2b: make(h2), gA1: make(h1) };
}

/** Forward pass for `rows` samples; z is sampled when rng is given, otherwise z = mu. */
function run(vae, x, rows, buf, rng) {
  const { layers, latentDim: d, nFeatures: f } = vae;
  const h1 = layers.enc1.nOut;
  const h2 = layers.enc2.nOut;
  forward(layers.enc1, x, rows, buf.a1); relu(buf.a1, rows * h1);
  forward(layers.enc2, buf.a1, rows, buf.a2); relu(buf.a2, rows * h2);
  forward(layers.mu, buf.a2, rows, buf.mu);
  forward(layers.logVar, buf.a2, rows, buf.lv);
  for (let i = 0; i < rows * d; i++) {
    buf.eps[i] = rng ? rng.normal() : 0;
    buf.z[i] = buf.mu[i] + (rng ? Math.exp(0.5 * buf.lv[i]) * buf.eps[i] : 0);
  }
  forward(layers.dec1, buf.z, rows, buf.d1); relu(buf.d1, rows * h2);
  forward(layers.dec2, buf.d1, rows, buf.d2); relu(buf.d2, rows * h1);
  forward(layers.dec3, buf.d2, rows, buf.out);
  const se = new Float64Array(rows);
  const kl = new Float64Array(rows);
  for (let s = 0; s < rows; s++) {
    for (let j = 0; j < f; j++) se[s] += (x[s * f + j] - buf.out[s * f + j]) ** 2;
    for (let j = 0; j < d; j++) { const m = buf.mu[s * d + j]; const l = buf.lv[s * d + j]; kl[s] += -0.5 * (1 + l - m * m - Math.exp(l)); }
  }
  return { se, kl };
}

function adam(vae, lr) {
  vae.step += 1;
  const c1 = 1 - BETA1 ** vae.step;
  const c2 = 1 - BETA2 ** vae.step;
  for (const layer of Object.values(vae.layers)) {
    [[layer.W, layer.gW], [layer.b, layer.gb]].forEach(([param, grad], p) => {
      const m = layer.m[p];
      const v = layer.v[p];
      for (let i = 0; i < param.length; i++) {
        m[i] = BETA1 * m[i] + (1 - BETA1) * grad[i];
        v[i] = BETA2 * v[i] + (1 - BETA2) * grad[i] * grad[i];
        param[i] -= (lr / c1) * m[i] / (Math.sqrt(v[i]) / Math.sqrt(c2) + ADAM_EPS);
        grad[i] = 0;
      }
    });
  }
}

function trainBatch(vae, x, rows, buf, beta, lr, rng) {
  const { layers, latentDim: d, nFeatures: f } = vae;
  const h1 = layers.enc1.nOut;
  const h2 = layers.enc2.nOut;
  const { se, kl } = run(vae, x, rows, buf, rng);
  for (let i = 0; i < rows * f; i++) buf.gOut[i] = (2 * (buf.out[i] - x[i])) / rows;
  backward(layers.dec3, buf.d2, buf.gOut, rows, buf.gD2); reluGrad(buf.gD2, buf.d2, rows * h1);
  backward(layers.dec2, buf.d1, buf.gD2, rows, buf.gD1); reluGrad(buf.gD1, buf.d1, rows * h2);
  backward(layers.dec1, buf.z, buf.gD1, rows, buf.gZ);
  for (let i = 0; i < rows * d; i++) {
    const sigma = Math.exp(0.5 * buf.lv[i]);
    buf.gMu[i] = buf.gZ[i] + (beta * buf.mu[i]) / rows;
    buf.gLv[i] = buf.gZ[i] * 0.5 * sigma * buf.eps[i] + (beta * -0.5 * (1 - Math.exp(buf.lv[i]))) / rows;
  }
  backward(layers.mu, buf.a2, buf.gMu, rows, buf.gA2);
  backward(layers.logVar, buf.a2, buf.gLv, rows, buf.gA2b);
  for (let i = 0; i < rows * h2; i++) buf.gA2[i] += buf.gA2b[i];
  reluGrad(buf.gA2, buf.a2, rows * h2);
  backward(layers.enc2, buf.a1, buf.gA2, rows, buf.gA1); reluGrad(buf.gA1, buf.a1, rows * h1);
  backward(layers.enc1, x, buf.gA1, rows, null);
  adam(vae, lr);
  let loss = 0, rec = 0, klSum = 0;
  for (let s = 0; s < rows; s++) { loss += se[s] + beta * kl[s]; rec += se[s]; klSum += kl[s]; }
  return { loss, rec, kl: klSum };
}

/** Mean loss, reconstruction and KL per sample with z = mu (deterministic). */
function evaluate(vae, data, rows, beta) {
  const chunk = 2048;
  const buf = buffers(vae, chunk);
  let loss = 0, rec = 0, kl = 0;
  for (let start = 0; start < rows; start += chunk) {
    const count = Math.min(chunk, rows - start);
    const r = run(vae, data.subarray(start * vae.nFeatures, (start + count) * vae.nFeatures), count, buf, null);
    for (let s = 0; s < count; s++) { loss += r.se[s] + beta * r.kl[s]; rec += r.se[s]; kl += r.kl[s]; }
  }
  return { loss: loss / rows, rec: rec / rows, kl: kl / rows };
}

const snapshot = (vae) => Object.fromEntries(Object.entries(vae.layers).map(([k, l]) => [k, { W: l.W.slice(), b: l.b.slice() }]));
const restore = (vae, state) => Object.entries(state).forEach(([k, s]) => { vae.layers[k].W.set(s.W); vae.layers[k].b.set(s.b); });

/** Adam training with early stopping on fit_validation loss; the best weights are restored. */
export function trainVae(vae, fitData, fitRows, validationData, validationRows, options, rng, onEpoch = () => {}) {
  const { epochs, batchSize, learningRate, beta, patience } = options;
  const f = vae.nFeatures;
  const buf = buffers(vae, batchSize);
  const batch = new Float64Array(batchSize * f);
  const history = { epoch: [], train_loss: [], train_reconstruction: [], train_kl: [], validation_loss: [], validation_reconstruction: [], validation_kl: [] };
  let best = Infinity;
  let bestState = snapshot(vae);
  let sinceBest = 0;
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const order = rng.permutation(fitRows);
    let loss = 0, rec = 0, kl = 0;
    for (let start = 0; start < fitRows; start += batchSize) {
      const rows = Math.min(batchSize, fitRows - start);
      for (let s = 0; s < rows; s++) batch.set(fitData.subarray(order[start + s] * f, (order[start + s] + 1) * f), s * f);
      const r = trainBatch(vae, batch, rows, buf, beta, learningRate, rng);
      loss += r.loss; rec += r.rec; kl += r.kl;
    }
    const val = evaluate(vae, validationData, validationRows, beta);
    history.epoch.push(epoch);
    history.train_loss.push(loss / fitRows); history.train_reconstruction.push(rec / fitRows); history.train_kl.push(kl / fitRows);
    history.validation_loss.push(val.loss); history.validation_reconstruction.push(val.rec); history.validation_kl.push(val.kl);
    onEpoch(epoch / epochs);
    if (val.loss < best) { best = val.loss; bestState = snapshot(vae); sinceBest = 0; }
    else if (++sinceBest >= patience) break;
  }
  restore(vae, bestState);
  return history;
}

/** Per row: mean squared reconstruction error (primary), negative ELBO (secondary) and the latent mean. */
export function vaeScores(vae, data, rows) {
  const chunk = 2048;
  const buf = buffers(vae, chunk);
  const d = vae.latentDim;
  const reconstruction = new Float64Array(rows);
  const negativeElbo = new Float64Array(rows);
  const latent = new Float32Array(rows * d);
  for (let start = 0; start < rows; start += chunk) {
    const count = Math.min(chunk, rows - start);
    const r = run(vae, data.subarray(start * vae.nFeatures, (start + count) * vae.nFeatures), count, buf, null);
    for (let s = 0; s < count; s++) {
      reconstruction[start + s] = r.se[s] / vae.nFeatures;
      negativeElbo[start + s] = r.se[s] + r.kl[s];
    }
    latent.set(buf.mu.subarray(0, count * d), start * d);
  }
  return { reconstruction, negativeElbo, latent };
}
