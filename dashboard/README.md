# Dashboard

Live: <https://metroguard-compressor-anomaly-detec.vercel.app>

A static site with no build step: plain HTML, CSS and JavaScript. By default it displays the results the Python
pipeline already saved. The **Analyse your data** page can also run the same pipeline on a visitor's own file,
entirely in their browser.

```
index.html  metrics.html  scores.html  network.html  data.html  claims.html  analyse.html    one page each
assets/
  styles.css        design tokens (light and pure-black dark), layout, components, motion
  app.js            shared shell, theme, data loading (MetroGuard results or the visitor's analysis), tables
  charts.js         canvas time-series stack (hover, zoom, keyboard) and line chart
  linemap.js        the metro-line timeline on the overview
  page-*.js         one script per page
  file-import.js    reads a chosen CSV, Excel or PDF file into a previewable table
  motion.js         GSAP scroll motion for the overview and the Lottie player, both skipped under reduced motion
  latent-explorer.js three.js view of the VAE latent space (The data page)
  animations/       Lottie JSON (the analysis loader)
  saved-analysis.js keeps the visitor's latest analysis in IndexedDB
  analysis/         the pipeline in JavaScript (steps 1 to 8), run in a Web Worker (worker.js)
  vendor/           SheetJS and pdf.js (for Excel and PDF uploads), GSAP, three.js and lottie-web; each loads only
                    where it is used (licences in vendor/THIRD_PARTY_LICENSES.txt)
  fonts/            Schibsted Grotesk and JetBrains Mono (SIL Open Font License)
data/               generated JSON, committed so the site can be served without the pipeline
tests/              node --test unit tests for assets/analysis (not deployed)
vercel.json         clean URLs and cache headers
```

## Data

`data/` is written by `pipeline/step10_build_dashboard_data.py` from `reports/tables`, `outputs/scores` and
`outputs/logs`. Do not edit it by hand. Every number on a page is read from these files; sentences are assembled
in the browser so the site cannot drift from the pipeline.

## Analyse your data

The visitor picks a file, checks the detected column roles and timestamp format in a preview, chooses the training
period, window length, Markov state column, CD-k variants and optional reported windows, and runs the analysis.

- The file never leaves the browser. CSV files are streamed by the worker; spreadsheets and PDF tables are read on
  the page first. PDF tables are reconstructed from text positions, so the preview must be confirmed.
- `assets/analysis/` mirrors the Python modules: `preparation.js` (audit, windows, split, fit-only transforms),
  `rbm.js` (Gibbs sampling, CD-k, free energy), `vae.js` (manual backprop, Adam, early stopping), `markov.js`,
  `bayesian-network.js` (hill climb with BIC, BDeu, d-separation), `alerting.js` and `bundle.js`, which writes the
  same JSON documents as step 10. `pipeline.js` runs them in order and freezes each model's thresholds before it
  scores a test window.
- The result is stored in IndexedDB. While it is shown, every page carries a banner with a way back to the
  MetroGuard results; deleting it on the analyse page removes it from the browser.

Unit tests (from the repository root):

```
node --test --disable-warning=MODULE_TYPELESS_PACKAGE_JSON "dashboard/tests/*.test.mjs"
```

## Preview

```
python -m http.server 8000 --directory dashboard
```

Open http://localhost:8000. Opening the HTML files straight from disk will not work, because the pages fetch
their data over HTTP and the analysis runs in a module worker.

## Deploy on Vercel

Import the repository, set the **Root Directory** to `dashboard`, choose the **Other** framework preset and leave
the build and output settings empty. `.vercelignore` keeps the tests out of the deployment.

## Notes

- Opens in the light theme; the sidebar button switches to the pure-black dark theme and the choice is remembered.
- Alerts are marked by shape and by a text summary as well as colour; charts can be inspected with the keyboard.
- Motion respects `prefers-reduced-motion`.
