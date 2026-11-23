# Model card: pocket transit vetter (milestone 5 final recipe)

## Files

| File | What it is |
|---|---|
| `weights/fold{0..9}.npz` | State dicts of the 10 seeds of the final recipe trained on each of the 10 folds (100 models). One flat float32 vector per seed plus the tensor names, shapes and dtypes; read without pickle by `cazaplanetas.release.load_ensemble(fold)`. |
| `predictions_hito5_veredicto.npz` | Validation and test scores of every seed, for the final recipe (`combo`) and its distilled model (`distilled_combo`), on the 31,045 labelled DR25 signals, with their keys and labels. Input of `scripts/hito5_veredicto.py report`. |
| `demo_signals.npz` | Prepared model inputs of 19 signals and the scores recorded at training time. Input of `scripts/demo.py`. |
| `SHA256SUMS` | Checksums of the files above (`shasum -a 256 -c SHA256SUMS` from this folder). |

All of them are written by `scripts/export_release.py` from a full run of
`scripts/hito5_veredicto.py run`.

## Model

- **Task.** Binary vetting of a Kepler DR25 threshold-crossing event: planet (confirmed planet of the
  cumulative KOI table) against false positive (certified false positive or not transit-like),
  following ExoMiner's labelling rules.
- **Architecture.** `cazaplanetas.models.PocketBranches` as configured by the recipe `combo` in
  `scripts/hito5_receta.py`: one small separable 1-D convolutional branch per diagnostic test, each
  ending in an 8-unit bottleneck, and a scalar branch; **25,745 trainable parameters** per model.
  Build it with `hito5_receta.RECIPES["combo"].make_model()`.
- **Inputs** (names of the `forward()` arguments):
  - 2001-bin global views: `global_view`, `half_global_view` (the global view folded at half the
    period) and `centroid_global_view`;
  - 201-bin local views: `secondary_view`, `sibling_view` (the view of the other signal of the same
    star with the same period, if any) and `centroid_local_view`;
  - `scalars`: 43 values, the 32 DR25 TCE columns allowed by `cazaplanetas.features.SCALAR_COLUMNS`
    plus the sibling, Kepler magnitude and ephemeris-match columns of `cazaplanetas.derived`,
    standardised with the training part of the fold.
  The views are built from the light curves by `scripts/hito3_dr25.py` and normalised by
  `scripts/hito5_receta.py` (`load_dataset`, `device_tensors`); normalisation uses statistics of
  each fold's training part, so the inputs of one fold are not valid for another fold's models.
- **Output.** A logit; `sigmoid` gives the score. The ensemble score is the mean of the 10 seeds'
  scores. Scores are not calibrated probabilities (training used label smoothing of 0.05).
- **Training.** 30 epochs, batch 64, AdamW (learning rate 2·10⁻³ with a cosine schedule, weight
  decay 10⁻⁴), label smoothing 0.05, Gaussian noise augmentation 0.05; the epoch with the best
  validation AUC-PR is kept. Folds: `splits.star_folds(kepid, 10,
  seed=20261122)`; for fold *f*, test = fold *f*, validation = fold *f* + 1, training = the other 8.
  Seeds 0-9.

## Evaluation

Mean over the 10 test folds (each signal is scored by the models that never saw it):

| | Parameters | R@P0.99 exact | R@P0.99 as ExoMiner counts it | AUC-PR |
|---|---|---|---|---|
| One model (mean of 10 seeds) | 25,745 | 0.927 | 0.953 | 0.9934 |
| Ensemble of 10 | 257,450 | 0.950 | 0.963 | 0.9962 |
| ExoMiner, its published scores on the same folds | ≈ 2.5 M | 0.926 | 0.950 | 0.9952 |

Full tables, intervals and the verdict rule: [`docs/hito5_veredicto.md`](../docs/hito5_veredicto.md)
(Spanish) and the [README](../README.md). The signals and labels are the same ones inspected while
designing the recipe; only their grouping into folds is new.

## Intended use and limitations

- Research and benchmarking on Kepler DR25. The models are a reproducible reference for the numbers
  in this repository, not a validated planet-confirmation tool.
- Trained only on Kepler long-cadence data and DR25 events. They have not been tested on TESS, K2,
  short cadence, simulated (injected, inverted or scrambled) DR25 events or signals detected by
  other pipelines.
- There is no single model trained on all the data: each fold's models saw 80 % of the stars. To
  score a new signal, any fold's ensemble can be used with inputs normalised with that fold's
  training statistics; the scripts that compute them need the full DR25 dataset.
- Known weak spots (from `docs/hito5_receta.md`): planets of saturated bright stars and signals
  detected at twice the true period.

## License

MIT, like the rest of the repository (see [`LICENSE`](../LICENSE)). The inputs in `demo_signals.npz`
are derived from public Kepler data obtained from MAST; see the acknowledgements in the README.
