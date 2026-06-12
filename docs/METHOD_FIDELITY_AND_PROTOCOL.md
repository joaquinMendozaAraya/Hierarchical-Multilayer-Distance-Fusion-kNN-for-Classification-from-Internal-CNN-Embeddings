# Raw literature competitor benchmark

Date: 2026-06-05.

## Objective

Rebuild the external multidimensional competitor matrix independently from old
VCHMF/WinMax variants and old result tables. Every method receives the same:

- frozen layer embeddings;
- train/validation/test labels;
- dataset split;
- backbone checkpoint context.

External methods cannot call HMDF-kNN layer ranking, weighting, routing, fusion,
or candidate selection.

## Executable artifacts

- Friendly notebook:
  `notebooks/HMDF_kNN_friendly_walkthrough.ipynb`
- Runner:
  `pipelines/run_raw_literature_multidim_competitors.py`
- Independent method implementations:
  `src/raw_multiview_competitors.py`
- Curated result summaries:
  `results/metrics/`

## Anti-leakage protocol

1. Transformations, fusion parameters, projections, feature selectors, and
   classifiers are fitted only with `train`.
2. Candidate hyperparameters are compared only with `validation`.
3. The selected candidate is frozen.
4. `test` predictions and metrics are computed once for that candidate.
5. HMDF-kNN is independently recomputed in every dataset-backbone context.
6. Every external selected row stores accuracy, macro-F1, and balanced-accuracy
   deltas against that context's HMDF-kNN result.
7. HAM10000 is an external-domain control and is excluded from the main
   brain-MRI aggregate.

## Competitor fidelity

| ID | Role | Fidelity status | Preserved mechanism | Declared adaptation |
|---|---|---|---|---|
| `raw_concat_linear` | mandatory control | control | per-layer L2, all-layer concatenation, linear classifier | none |
| `concat_pca_linear` | mandatory control | control | per-layer L2, concatenation, train-only PCA, linear classifier | none |
| `uniform_layer_softvote` | mandatory control | control | one classifier per fixed layer, uniform probability average | none |
| `uniform_kernel_svm` | mandatory control | control | uniform mean of normalized linear layer kernels | implemented through the equivalent scaled concatenated feature map |
| `fradi_mlcff` | paper competitor | exact algorithm, input-adapted | per-layer L2, concatenate, PCA, LDA, linear one-vs-one SVM | saved GAP stage endpoints replace every convolutional layer; average pooling was already performed during extraction |
| `head2toe` | paper competitor | exact selection algorithm, input-adapted | unit-normalized features, concatenation, row-group L2,1 linear probe, top-feature fraction, retrained unregularized linear head | stage-level GAP vectors replace all block/window-pooled activations; `C=1e6` approximates an unregularized final logistic head |
| `easymkl` | paper competitor | exact EasyMKL algorithm with linear-kernel adapter | one normalized linear kernel per layer, EasyMKL nonnegative kernel weighting, SVM learner | deterministic stratified train cap is used when configured and is recorded in every result |
| `maxvar_gcca` | established algorithm | exact objective with out-of-sample maps | MAXVAR common representation and regularized per-view maps | projected views from the same image are averaged before kNN |
| `gmlda` | paper competitor | exact GMA/GMLDA projection, input-adapted | GMA block eigenproblem, LDA scatter objectives, class means as exemplars | CNN layers are paired views; same-image projected views are averaged |
| `mvda` | paper competitor | exact scatter formulation, input-adapted | MvDA equations 7-12 and generalized eigenproblem | CNN layers are paired views; same-image projected views are averaged |
| `concat_nca_knn` | established composed baseline | exact NCA, composed pipeline | fixed all-layer concatenation, train-only PCA, NCA, kNN | deterministic stratified NCA fit cap is used when configured and recorded |
| `winmax_reference` | proposed method | frozen HMDF-kNN reference; historical internal ID | validation-ranked layers, validation-selected distance weights, distance-weighted kNN | no external method can access its ranking or weights |

## Primary sources

- Fradi, Fradi, and Dugelay, 2021: see the manuscript bibliography.
- Evci et al., 2022: see the manuscript bibliography.
- Head2Toe official code:
  `https://github.com/google-research/head2toe`
- EasyMKL:
  `https://doi.org/10.1016/j.neucom.2014.11.078`
- NCA:
  `https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf`
- GMA/GMLDA:
  `https://doi.org/10.1109/CVPR.2012.6247923`
- MvDA:
  `https://doi.org/10.1109/TPAMI.2015.2435740`

## Methods deliberately excluded

- PatchResNet is not forced onto all nine CNNs because it changes image inputs,
  patch count, and inference cost. It requires a separate ResNet-50 image-level
  experiment.
- AFF is an internal trainable feature-map module, not a raw classifier over
  the saved post-hoc vectors.
- VQT and attentive ViT probing are not applied because the current matrix
  contains CNNs, not Vision Transformers.
- Old exploratory VCHMF/WinMax operators and result-oracle candidates
  are not external competitors.

## Saved evidence

Each method/context directory contains:

- `validation_candidates.csv`;
- `selected_result.json`;
- `predictions.npz`;
- `test_confusion.csv`;
- `test_report.csv`;
- `diagnostics.json`;
- `done.json`.

Global files include:

- `all_selected_results.csv`;
- `delta_vs_winmax_by_context.csv`;
- `method_summary_vs_winmax.csv`;
- `coverage_audit.csv`;
- `input_audit.csv`;
- `error_log.csv`, when applicable.
- `heavy_resume_status.csv`, with one row per isolated heavy-method attempt.

## Resume behavior

- A method/context is skipped only when `selected_result.json`, `done.json`,
  and `predictions.npz` form a valid completed checkpoint.
- Heavy methods run in independent subprocesses, so a native EasyMKL crash
  cannot terminate Head2Toe, NCA, or later contexts.
- Every evaluated validation setting is printed immediately and appended to
  `candidate_progress.jsonl`.
- The default EasyMKL fitting cap is 1000 stratified training samples. This
  uniform budget avoids the native Windows memory crash observed with eight
  `2500 x 2500` kernels.
- `resume_notebook_78_heavy.py --normalize-easymkl-budget` recomputes only
  EasyMKL results created with another fitting cap.

## Known limitation

The imported Colab profiles do not contain persistent sample IDs for every
split. Shape, label alignment, class coverage, and finite-value checks are
performed, but cross-split duplicate-image auditing is marked
`unavailable_no_saved_sample_ids` unless a `paths.npy` or `paths.csv` artifact
is added.
