# Dashboard

A read-only static site that displays the results the pipeline already saved. It runs no model, reads no raw
data and has no build step: plain HTML, CSS and JavaScript.

```
index.html  metrics.html  scores.html  network.html  data.html  claims.html    one page each
assets/
  styles.css        design tokens (light and pure-black dark), layout, components, motion
  app.js            shared shell, theme, data loading, tables, segmented control
  charts.js         canvas time-series stack (hover, zoom, keyboard) and line chart
  linemap.js        the metro-line timeline on the overview
  page-*.js         one script per page
  fonts/            Schibsted Grotesk and JetBrains Mono (SIL Open Font License)
data/               generated JSON, committed so the site can be served without the pipeline
vercel.json         clean URLs and cache headers
```

## Data

`data/` is written by `pipeline/step10_build_dashboard_data.py` from `reports/tables`, `outputs/scores` and
`outputs/logs`. Do not edit it by hand. Every number on a page is read from these files; sentences are assembled
in the browser so the site cannot drift from the pipeline.

## Preview

```
python -m http.server 8000 --directory dashboard
```

Open http://localhost:8000. Opening the HTML files straight from disk will not work, because the pages fetch
their data over HTTP.

## Deploy on Vercel

Import the repository, set the **Root Directory** to `dashboard`, choose the **Other** framework preset and leave
the build and output settings empty.

## Notes

- Light and dark themes follow the operating system until the visitor picks one.
- Alerts are marked by shape and by a text summary as well as colour; charts can be inspected with the keyboard.
- Motion respects `prefers-reduced-motion`.
