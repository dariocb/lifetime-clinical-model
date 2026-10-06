# Lifetime clinical models

Tutorial-first research repository for modelling a patient's observed clinical lifetime with temporal point processes (TPPs), Gaussian processes (GPs), Transformers and normalizing flows.

Clinical records are uneven: visits happen at irregular times, measurements are sparse, and a patient's circumstances can change. This repository explores how to predict the next visit and its type while estimating physiology between recorded measurements, with uncertainty. The tutorials build toward a joint model of these two questions using synthetic patient histories; they are research demonstrations rather than a validated clinical prediction tool.

The notebooks are intentionally ordered from textbook models to the clinical problem:

```text
notebooks/
├── introduction/
│   ├── 01_temporal_point_processes.ipynb
│   ├── 02_gaussian_processes.ipynb
│   ├── 03_transformers.ipynb
│   └── 04_normalizing_flows.ipynb
├── data/
│   └── 01_synthetic_clinical_data.ipynb
├── models/
│   ├── 00_baselines.ipynb
│   ├── 01_tpp_models.ipynb
│   ├── 02_gp_models_1_2.ipynb
│   └── 03_gp_model_3.ipynb
└── lifetime_model/
    └── 01_lifetime_model_formulation.ipynb
```

## Canonical data contract

`data/synthetic/` is exactly the synthetic dataset supplied with the project. `data/toy/` is a deterministic 130-patient subset of the richer `data/generated_synthetic/` scenario used to keep model notebooks fast. Both have the same schema. Hidden pre-study events are oracle-only and are never supplied as observed model history.

See the [dataset guide](data/README.md) for variables, static/dynamic covariates, patient timelines, visit marks, missingness and train/validation/test splits.

The baseline notebook first fits five Poisson/Hawkes variants, including patient-specific fits and small cohort MLPs. Its comparisons separate timing from timing-plus-mark likelihoods.

## Six implemented models

- **TPP-A** predicts when the next visit happens and its type, from fixed baseline information and previous waiting times.
- **TPP-B** adds changing patient information and predicts physiological snapshots at future visits.
- **TPP-C** does the same, while separating physiological information used for visit prediction from information retained for generating snapshots.
- **GP-A** estimates continuous physiological trajectories between irregular measurements, with Gaussian measurement uncertainty.
- **GP-B** adds a flexible transformation, allowing observed physiology to have asymmetric distributions and different tail shapes.
- **GP-C** separates an underlying physiological trajectory from flexible residual variation, making their contributions inspectable.

All learned history summaries use the common causal Transformer in `lifetime_clinical/components/history.py`. Input tokens differ by model; separately fitted variants learn their own weights. The lifetime notebook jointly fits a TPP-A-style visit branch and GP-C with the same encoder instance, and explains the prototype's remaining limits.

Clinical age enters neural models through learned continuous absolute-age encodings, rather than a raw scalar feature. Categories use small learned embeddings (2–3 coordinates), with explicit missing/unseen categories. Schema indicators remain in the data adapters for compatibility but are converted to category IDs before neural computation. GP kernels retain age as their continuous time coordinate, and output IDs select cross-feature covariance. The cohort Poisson/Hawkes MLPs use the same input encoding; per-patient classical fits have no neural input encoder.

Fitting uses every eligible patient in the selected **training split**, with separate validation and test patients. A single patient is selected only to illustrate predictions. Full mode uses `data/synthetic` (1,000 training patients before model-specific eligibility checks); fast mode uses the toy cohort (100 training, 20 validation and 10 test patients).

## Running

```bash
python -m pip install -e ".[notebooks,dev]"
jupyter lab
```

The model notebooks default to `FAST_MODE=True`: the toy cohort and 5 epochs. `False` selects the full synthetic cohort and 10 epochs. They use CUDA when available, otherwise CPU. Executed fast-mode outputs are included so the figures and fitted results can be read without rerunning training.

## Generating synthetic data

The single generator implementation is `lifetime_clinical/data/synthetic.py`.
`configs/synthetic_data.json` selects the richer combined scenario.
`configs/synthetic_data_simple.json` preserves the original generator settings.
Existing `data/synthetic/` and `data/toy/` tables are not overwritten; their own
`metadata.json` describes their generation settings.
From the repository root, run:

```bash
python scripts/generate_synthetic_data.py
python scripts/generate_synthetic_data.py --config configs/synthetic_data.json --output data/my_synthetic
```

The default output is `data/generated_synthetic/`. The CLI saves the five CSV
tables, `metadata.json` and a separate oracle trajectory table in rich scenarios; patient splits and the toy subset are separate
tutorial artifacts and are not generated by this command.

For Python usage:

```python
from lifetime_clinical.data.synthetic import SyntheticConfig, generate_dataset

dataset = generate_dataset(SyntheticConfig(n_samples=100, seed=42))
```

## Files to publish

Keep package code, scripts, tests, configs, tutorial notebooks with executed
fast-mode outputs, diagram sources/exports, and the supplied `data/synthetic/`
and `data/toy/` tables with their metadata and splits. These datasets let a clone
run the tutorials and tests immediately.

`.gitignore` excludes temporary benchmark copies in `tmp/`, fitted checkpoints
in `artifacts/`, caches, environments, local credentials and the reproducible
`data/generated_synthetic/` output (including the large oracle trajectory table).
Generate the richer full dataset with `python scripts/generate_synthetic_data.py`.
Do not blanket-ignore `data/` or the exported diagram assets: they support the tutorials.
