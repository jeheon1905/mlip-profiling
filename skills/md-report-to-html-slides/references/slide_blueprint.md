# HTML slide blueprint

Canonical patterns for `slides.html`. The agent writes `<section>` elements directly inside `<div class="slides">` — there is no markdown source. Every pattern is copy-pastable.

## File skeleton (already created by `create_html_shell.py`)

```html
<div class="reveal">
  <div class="slides">
    <!-- write <section> blocks here -->
  </div>
</div>
```

The shell ships with custom CSS classes used below: `cols`, `cols-1-2`, `cols-2-1`, `fig`, `stacked-tables`, `title-slide`, `callout` (+ `.good`/`.warn`/`.bad`).

## Source rules

- One `<section>` per slide. Speaker notes go inside `<aside class="notes">…</aside>` at the end of the section.
- Sentence-style `<h1>`/`<h2>` titles that already convey the takeaway.
- Default body font size is set in CSS (22px). Override per slide with inline style only if needed.
- Reveal.js renders one slide per `<section>`. Nested `<section>` creates a vertical sub-deck — use sparingly (e.g., backup details).

## 1. Title slide

```html
<section class="title-slide">
  <h1>Deck title in one sentence</h1>
  <div class="subtitle">Subtitle or one-line takeaway</div>
  <div class="meta">Author Name &middot; YYYY-MM-DD &middot; Venue</div>
  <aside class="notes">
    Opening remarks. Keep brief and natural.
  </aside>
</section>
```

## 2. Table of contents (decks ≥ 6 content slides)

```html
<section>
  <h2>Table of contents</h2>
  <ul>
    <li><strong>Setup</strong> — hardware, models, methodology</li>
    <li><strong>Results</strong> — per-model breakdowns and speedups</li>
    <li><strong>Analysis</strong> — root causes and bottlenecks</li>
    <li><strong>Recommendations</strong> — what to deploy next</li>
  </ul>
  <aside class="notes">
    Brief orientation: how many sections, which is most important.
  </aside>
</section>
```

## 3. Bullet slide

```html
<section>
  <h2>Main takeaway as the title</h2>
  <ul>
    <li>Evidence point 1 with quantitative anchor</li>
    <li>Evidence point 2</li>
    <li>Implication or contrast</li>
  </ul>
  <aside class="notes">
    Why this finding matters. Add nuance the bullets omit.
  </aside>
</section>
```

## 4. Image slide (one figure)

```html
<section>
  <h2>Throughput scales sub-linearly past 1372 atoms</h2>
  <figure class="fig">
    <img src="plots/model_comparison_throughput.png" alt="Throughput vs atom count">
    <figcaption>ns/day across 108–2916-atom Cu FCC supercells</figcaption>
  </figure>
  <aside class="notes">
    Walk the audience through the x-axis and highlight the crossover point.
  </aside>
</section>
```

## 5. Image + text (the most common analysis slide)

The CSS class `cols-1-2` puts the image on the left at 1/3 width and bullets on the right at 2/3. Use `cols-2-1` to invert.

```html
<section>
  <h2>cueq backend cuts MACE tensor product cost by 6×</h2>
  <div class="cols-1-2">
    <figure class="fig">
      <img src="plots/mace_cueq_breakdown.png" alt="MACE cueq operation breakdown">
    </figure>
    <ul>
      <li><strong>Tensor product</strong>: 286 ms → 45 ms (6.4×) at 1,372 atoms</li>
      <li><strong>Graph generation</strong> (CPU) becomes the new bottleneck above 500 atoms</li>
      <li>Net inference latency: 286 → 76 ms (3.8× end-to-end)</li>
      <li>Memory overhead negligible (&lt; 4% peak VRAM increase)</li>
    </ul>
  </div>
  <aside class="notes">
    Note that cueq only accelerates the GPU tensor product — CPU graph build is unchanged.
  </aside>
</section>
```

## 6. Side-by-side images (comparison)

```html
<section>
  <h2>Backend acceleration varies by model architecture</h2>
  <div class="cols">
    <figure class="fig">
      <img src="plots/mace_e3nn_breakdown.png" alt="MACE e3nn breakdown">
      <figcaption>MACE (e3nn baseline)</figcaption>
    </figure>
    <figure class="fig">
      <img src="plots/sevenn_e3nn_breakdown.png" alt="SevenNet e3nn breakdown">
      <figcaption>SevenNet (e3nn baseline)</figcaption>
    </figure>
  </div>
  <aside class="notes">
    Both are dominated by tensor products but the relative cost differs.
  </aside>
</section>
```

## 7. Table slide

HTML tables are not constrained by pandoc — no row/column limit, no orphan-table risk. Bullets and tables mix freely.

```html
<section>
  <h2>End-to-end latency (ms) across atom counts</h2>
  <table>
    <thead>
      <tr><th>Config</th><th>108</th><th>500</th><th>1,372</th><th>2,916</th></tr>
    </thead>
    <tbody>
      <tr><td><strong>MACE e3nn</strong></td><td>40.2</td><td>106.5</td><td>286.6</td><td>728.1</td></tr>
      <tr><td><strong>MACE cueq</strong></td><td>45.4</td><td>48.7</td><td>76.0</td><td>187.4</td></tr>
      <tr><td><strong>SevenNet e3nn</strong></td><td>38.1</td><td>98.3</td><td>271.0</td><td>692.7</td></tr>
    </tbody>
  </table>
  <aside class="notes">
    Crossover at ~500 atoms — below that, e3nn and cueq are comparable.
  </aside>
</section>
```

## 8. Stacked tables (two related metrics on one slide)

```html
<section>
  <h2>Latency and throughput both confirm the cueq advantage</h2>
  <div class="stacked-tables">
    <table>
      <thead><tr><th>Config</th><th>108</th><th>500</th><th>1,372</th></tr></thead>
      <tbody>
        <tr><td><strong>MACE e3nn (ms)</strong></td><td>40.2</td><td>106.5</td><td>286.6</td></tr>
        <tr><td><strong>MACE cueq (ms)</strong></td><td>45.4</td><td>48.7</td><td>76.0</td></tr>
      </tbody>
    </table>
    <table>
      <thead><tr><th>Config</th><th>108</th><th>500</th><th>1,372</th></tr></thead>
      <tbody>
        <tr><td><strong>MACE e3nn (ns/day)</strong></td><td>2.15</td><td>0.81</td><td>0.30</td></tr>
        <tr><td><strong>MACE cueq (ns/day)</strong></td><td>1.90</td><td>1.78</td><td>1.14</td></tr>
      </tbody>
    </table>
  </div>
  <aside class="notes">
    Two metrics, same conclusion — show both to anticipate the throughput question.
  </aside>
</section>
```

## 9. Image + table on the same slide (HTML-only capability)

Pandoc cannot do this; HTML can.

```html
<section>
  <h2>Tensor product dominates — cueq removes the bottleneck</h2>
  <div class="cols">
    <figure class="fig">
      <img src="plots/mace_cueq_pie.png" alt="cueq operation share">
    </figure>
    <table class="tight">
      <thead><tr><th>Op</th><th>e3nn</th><th>cueq</th><th>×</th></tr></thead>
      <tbody>
        <tr><td>Tensor product</td><td>198 ms</td><td>32 ms</td><td>6.2</td></tr>
        <tr><td>Graph gen</td><td>22 ms</td><td>22 ms</td><td>1.0</td></tr>
        <tr><td>Other</td><td>67 ms</td><td>22 ms</td><td>3.0</td></tr>
      </tbody>
    </table>
  </div>
  <aside class="notes">
    The pie shows share; the table shows the absolute cost. cueq compresses the dominant slice.
  </aside>
</section>
```

## 10. Process / timeline slide

```html
<section>
  <h2>The benchmark runs in three sequential phases</h2>
  <ol>
    <li><strong>Warmup</strong> — 5 forward passes to amortize compile + cache</li>
    <li><strong>Profile</strong> — 5 active passes captured by torch profiler</li>
    <li><strong>QPS</strong> — timeit.repeat(number=10, repeat=5)</li>
  </ol>
  <aside class="notes">
    The split mirrors fairchem's uma_speed_benchmark methodology.
  </aside>
</section>
```

## 11. Callout / recommendation

```html
<section>
  <h2>Deploy cueq for production MACE workloads above 500 atoms</h2>
  <div class="callout good">
    <strong>Recommendation:</strong> ship MACE-medium with `--backend cueq` as the default
    in the eval pipeline, retain e3nn fallback for environments without
    cuequivariance-ops-torch installed.
  </div>
  <ul>
    <li>3.8× latency reduction at the most common production size (1,372 atoms)</li>
    <li>No accuracy regression observed across 200 test structures</li>
    <li>Install footprint: 220 MB additional CUDA libraries</li>
  </ul>
  <aside class="notes">
    State the ask, owner, and timing here.
  </aside>
</section>
```

## 12. Code snippet slide (with highlight.js)

```html
<section>
  <h2>Reproducing the cueq run</h2>
  <pre><code class="language-bash" data-trim>
python profile_mlip.py --model-type mace --model-path mace.pt \
    --backend cueq --structure-files structures/*.xyz --device cuda
  </code></pre>
  <aside class="notes">
    Note the --backend flag — without it, e3nn is the silent default.
  </aside>
</section>
```

## What to avoid

- Titles like `overview`, `analysis`, `results`, `conclusion` with no takeaway.
- More than ~6 substantive bullets per slide (consider splitting).
- Paragraph blocks copied from the report verbatim.
- Notes that merely repeat the bullets — notes are for *what you'd say*, not *what's on screen*.
- External CDN-loaded images or fonts (breaks offline use; the deck must work without internet).
- Inventing numbers, plots, or speedups that are not in the source report.
- Empty `<section>` shells (validator flags these).

## Why HTML lifts the PPTX restrictions

The pandoc → pptx path imposes "no bullets + tables", "no images + tables", "max 6×4 tables", and "tables on continuation slides need merging." None of those apply to HTML — CSS grid handles layout cleanly. Use the freedom: image + table on one slide, multiple stacked tables, custom column ratios are all first-class.
