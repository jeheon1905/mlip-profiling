# Quality checklist

Review before delivering.

## Structure
- For 6+ content slides: is a Table of Contents slide present as the second slide?
- Does the deck include setup/environment context (hardware, software, methodology)?
- For reports comparing multiple subjects: is a comparison table slide included?
- For technical reports with numeric results: are data tables included alongside charts?
- For benchmark/profiling reports specifically:
  - Is the hardware and environment reproduced as a table slide?
  - Is the model/subject comparison reproduced as a table slide?
  - Is the profiling methodology (schedule, measurement method) a table slide?
  - Is the metric definition (e.g., effective time classification) a table slide?
  - Are **all** per-configuration latency/throughput numbers in a table slide (not just charts)?
  - Is there a per-operation speedup table for each model or comparison pair?
  - Is there a separate analysis slide per model per backend (not merged to reduce count)?

## Narrative
- Does the first content slide surface the report's own key findings early?
- Does every slide have exactly one message?
- Does the sequence feel like a spoken presentation rather than a written report?

## Faithfulness to the source report
- For every slide (titles, bullets, tables, notes), can you point to the specific report section the content came from?
- Does the deck avoid inventing a "Recommendation" or "Next steps" slide if the report has none?
- Are titles factual (section-style or direct paraphrases) rather than editorial ("decisively", "flattens", "the answer")?
- Do speaker notes restate the report's analysis rather than promote the narrative ("lead with the answer", "end with the ask")?
- Are all numbers and claims directly traceable to the report — no extrapolation beyond measured data?

## Slide canvas
- Are most titles one line and sentence-style?
- Are bullet lists short, parallel, and ≤ 6?
- Do tables stay readable at 22px body font (use `class="tight"` for dense data)?
- Do image+text slides include 3–5 substantive bullets alongside the figure?
- Are bullets quantitative and interpretive — not just restating alt text?
- Are stacked tables grouped under one logical takeaway, not arbitrary pairings?
- Do code blocks fit within 65vh (truncate or split if longer)?

## Notes
- Does the title slide include `<aside class="notes">`?
- Does every content slide include `<aside class="notes">`?
- Do notes add context instead of repeating bullets?

## Images and plots
- Are relevant plots from the source report included via `<img src="…">`?
- Do all referenced image files exist at the specified paths (validator confirms)?
- Are paths relative to where `slides.html` lives — no absolute paths or external URLs?
- Is each image slide focused on one visual with a takeaway title?
- Are side-by-side images using `.cols`, not stacked vertically?

## Single-file portability
- Was `pack_html_slides.py --in-place` run after writing the slides?
- Are all images embedded as `data:` URLs (validator reports this)?
- Is reveal.js inlined inside `<script>` blocks (no `<script src="reveal/…">`)?
- Does the page open from `file://` with no network access?
- Can you copy `slides.html` alone (no sibling files) to a different folder and open it?

## Validator
- Does `python scripts/validate_html_slides.py slides.html` exit 0?
- All FAIL items resolved? WARN items reviewed and accepted or fixed?

## Browser sanity check
- Open `slides.html` in a browser, step through every slide with arrow keys.
- Press `S` to verify speaker notes view opens with notes for every slide.
- Press `?` to verify keyboard shortcuts work (sanity-check reveal init).
- Visual: no overflow off slide edges, no broken images, no clipped tables.
