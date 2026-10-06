# TPP and GP input-to-output diagrams

These six diagrams were authored with the **Mermaid Diagrams** plugin/skill,
using its canonical v2 flowchart format and Mermaid 11.16.1 compatibility profile.
They follow the implemented mark-conditioned timing models:

- **TPP-A:** fixed baseline and an IAT Transformer → timing/mark context.
- **TPP-B:** dynamic snapshots and Transformer histories → full physiology latent → context.
- **TPP-C:** the same feature flow → shared/private split; only current shared
  physiology latents enter the timing/mark context directly.

The GP diagrams show:

- **GP-A:** baseline and causal prefix → query-conditioned multi-output GP →
  exact Gaussian conditioning → latent physiology plus Gaussian noise.
- **GP-B:** the same GP machinery in Gaussian space → invertible observation
  transformation → measured-value forecasts; exact change-of-variables NLL.
- **GP-C:** a sparse variational physiology GP plus independent flow residuals
  → additive measured values; negative ELBO rather than exact marginal NLL.

GP prefix context is fixed; conditioning uses only available measurements. GP-C's
recognition Transformer is a separate instance of the same common encoder class.
All diagram density formulas use standardised values; implementation likelihoods
also include the fixed Jacobian for returning to clinical units. Forecast output
nodes explicitly undo standardisation. Solid arrows describe predictions; dashed
pink arrows identify training objectives.

For the TPPs, each context predicts mark probabilities and one timing flow per mark. Unknown
marks are summed out for forecasting and right censoring. B/C also fit feature
snapshot densities. Their covariate context refreshes at visit origins, rather
than at every asynchronous measurement time.

## Files and editing

For each `tpp_a`, `tpp_b`, `tpp_c`, `gp_a`, `gp_b`, and `gp_c`:

- `.json`: editable canonical diagram definition, including topology and styles.
- `.mmd`: Mermaid source exported and statically linted by the skill.
- `.svg`: portable, script-free structural SVG export.
- `.png`: raster fallback for notebook viewers without SVG support.
- `.dot`: exact Graphviz layout source used to produce the SVG.

`generate_diagrams.py` regenerates all outputs. `render_svg.cjs` contains the SVG
layout adapter. Edit the canonical JSON and regenerate; editing only `.mmd`
will not change the SVG or canonical source. Labels use Unicode mathematical
notation so the SVGs do not require a LaTeX/MathJax runtime.

## Rendering provenance

No local Mermaid CLI (`mmdc`) was installed when these assets were generated.
The SVGs therefore use **Graphviz through the bundled `@viz-js/viz` package** to
lay out the exact nodes, arrows and labels from the canonical Mermaid data.
They are structural exports, not evidence of acceptance by a Mermaid.js renderer.
The skill's static lint and JSON Schema validation both pass.

To obtain Mermaid-native SVGs later, render each `.mmd` with an installed
compatible Mermaid CLI, for example `mmdc -i tpp_a.mmd -o tpp_a.svg`. Such output
may use a different layout.

## Regeneration

From the repository root, with Python (`jsonschema`), Node.js, the Mermaid
Diagrams skill and `@viz-js/viz` already available:

```bash
python diagrams/generate_diagrams.py
```

The script discovers the local Codex skill/runtime by default. For other setups:

```bash
python diagrams/generate_diagrams.py --skill-dir /path/to/mermaid-diagrams --node /path/to/node --node-modules /path/to/node_modules
```

The diagrams are embedded as saved SVG display outputs in
`notebooks/models/01_tpp_models.ipynb`, immediately before the corresponding
model fits, without external links beneath the embedded images.
Each output includes a PNG fallback for viewers without SVG support.
Regeneration also refreshes those notebook outputs automatically.

Use `--models gp` or `--models tpp` to regenerate just one family. By default,
the script regenerates all six diagrams and embeds SVG+PNG display outputs in
the corresponding model notebooks. Existing fitting/prediction outputs are
preserved; diagram-only updates do not rerun model training.
