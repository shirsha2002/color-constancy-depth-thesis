# Progress presentation

The presentation reads measured experiment results directly from `../results/summary.json`.
Its editable tables and charts retain the underlying values in the PowerPoint file.
The script records source links and plain-English explanations in speaker notes.

Required evidence images are supplied by the scientific figure-generation stage:

- `../results/figures/cover_depth.png`
- `../results/figures/illumination_examples.png`
- `../results/figures/qualitative_best.png`
- `../results/figures/qualitative_failure.png`

An optional `figure_manifest.json` in the same folder can map the keys `cover`,
`illumination`, `best`, and `failure` to alternative image paths.

Build with the Codex bundled Node runtime and `RUNTIME_NODE_MODULES` set to its
bundled `dependencies/node/node_modules` directory:

```powershell
node presentation/build_deck.mjs
node presentation/export_deck_pdf.mjs
```

`deck-kit.mjs` uses JavaScript `@oai/artifact-tool` for authoring, rendering, and
PPTX export. The immutable presentation finalizer checks the package, text
geometry, font family, editable chart data, and native tables. Private rendering
and validation outputs stay under `.build`. Inspect all slide renders before
distributing the final deck.

The PDF copy embeds the inspected high-resolution slide renders. Use the PPTX
to edit text, tables, charts, and speaker notes. The separate scientific report
contains selectable text and additional methodological detail.
