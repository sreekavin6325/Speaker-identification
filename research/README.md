# IEEE Research Experiment Layer

This directory contains reproducible experiment scripts. It is deliberately
separate from the CustomTkinter application so benchmark changes cannot break
the working desktop interface.

## Current dataset protocol

`LibriSpeech/dev-clean` contains 2,703 utterances from 40 speakers. Its file
layout preserves both speaker and audiobook chapter identifiers.

The requested five-fold benchmark uses deterministic speaker-stratified,
utterance-level folds (`seed=42`). The subset contains only one to four
chapters per speaker, so a true chapter-grouped five-fold split cannot keep
every enrolled speaker in every test fold. This limitation is recorded in the
generated summary and must be disclosed in the paper. A later sensitivity
analysis will evaluate chapter overlap separately.

## Build the chapter-held-out sensitivity protocol

The primary five-fold protocol is utterance stratified because some
`dev-clean` speakers have only one chapter. For a stricter leakage sensitivity
analysis, build a deterministic two-fold subset in which each audiobook
chapter belongs wholly to one fold:

```text
.\.venv\Scripts\python.exe -m research.build_grouped_protocol
```

This protocol keeps only speakers with at least two distinct
`source_group`/chapter values. Single-chapter speakers and their utterances are
excluded with an explicit reason in the summary. For every included speaker,
whole chapters are partitioned into two nonempty folds with the minimum
possible utterance-count imbalance and a deterministic tie-break.

Generated artifacts:

```text
manifests/librispeech_dev_clean_chapter_heldout_2fold.csv
manifests/librispeech_dev_clean_chapter_heldout_2fold_protocol.csv
manifests/librispeech_dev_clean_chapter_heldout_2fold.summary.json
```

The compact protocol CSV can be applied to already-extracted embeddings, so
embedding extraction is not repeated:

```text
.\.venv\Scripts\python.exe -m research.run_baseline_cv --model speechbrain_xvector --protocol-file manifests/librispeech_dev_clean_chapter_heldout_2fold_protocol.csv
```

The runner selects and reorders embeddings by `clip_id`, verifies the stored
speaker label and source chapter, replaces `folds.npy` only for this
evaluation, and hard-fails if any protocol clip is absent. It also requires
zero train/test `source_group` overlap. A silent per-model intersection is
never used because that would make comparisons depend on the selected model.

Protocol results are isolated from the normal baseline:

```text
research_results/baseline_cv/protocols/<safe_protocol_name>__<hash>/<model_id>/<classifier>/<tag>/
```

The default runner behavior and default result path remain unchanged whenever
`--protocol-file` is omitted.

## Build the manifest

From the project directory:

```text
.\.venv\Scripts\python.exe -m research.build_librispeech_manifest
```

Generated artifacts:

```text
manifests/librispeech_dev_clean_manifest.csv
manifests/librispeech_dev_clean_manifest.summary.json
```

## Extract embeddings from the fixed manifest

Validate the complete four-model plan without loading neural models:

```text
.\.venv\Scripts\python.exe -m research.extract_manifest_embeddings --all-models --dry-run
```

Run a five-file SpeechBrain X-Vector smoke test:

```text
.\.venv\Scripts\python.exe -m research.extract_manifest_embeddings --model speechbrain_xvector --limit 5 --tag smoke
```

Run one complete encoder:

```text
.\.venv\Scripts\python.exe -m research.extract_manifest_embeddings --model speechbrain_xvector
```

If a run is interrupted, repeat the same model/tag with `--resume`.
Artifacts are isolated from the desktop application:

```text
research_results/embeddings/librispeech_dev_clean/<model_id>/<tag>/
```

## Run the fixed-fold baseline

After a model's full embedding extraction has completed, run its deterministic
linear-SVM baseline:

```text
.\.venv\Scripts\python.exe -m research.run_baseline_cv --model speechbrain_xvector
```

Evaluate all four completed embedding models with the same baseline:

```text
.\.venv\Scripts\python.exe -m research.run_baseline_cv --all-models
```

Supported classifier baselines are:

- `linear_svm`: linear SVC, `C=1.0`;
- `rbf_svm`: RBF SVC, `C=1.0`, `gamma=scale`;
- `logistic_regression`: multinomial-capable LBFGS, `C=1.0`,
  `max_iter=5000`;
- `cosine_centroid`: nearest L2-normalized class-mean embedding;
- `knn`: five neighbors, uniform weighting, Euclidean distance;
- `random_forest`: 300 deterministic CPU trees, square-root feature sampling;
- `decision_tree`: deterministic Gini tree with unlimited depth;
- `xgboost`: 200 deterministic CPU histogram boosting rounds, maximum depth
  six, learning rate 0.1, multiclass soft probabilities.

Use `--all-classifiers` to run all eight. XGBoost is an optional dependency:
the runner never installs it and raises a clear error if `xgboost` is absent.
All other classifiers use scikit-learn. Fixed defaults can be overridden only
through the documented classifier-specific arguments
`--knn-neighbors`, `--forest-estimators`, `--tree-max-depth`,
`--xgb-estimators`, `--xgb-max-depth`, and `--xgb-learning-rate`.
Every effective estimator parameter is persisted under
`classifier.hyperparameters` in `run_metadata.json`.

Each fold uses the persisted `folds.npy`; its `StandardScaler` is fitted only
on training samples. The same scaling protocol is intentionally applied to
all classifiers so that classifier comparisons differ only in the classifier.
The runner verifies every clip, label, and fold against the hashed source
manifest, rejects exact clip leakage, and records audiobook chapter overlap as
a separate protocol limitation.

Run the complete classifier comparison with the strict chapter-held-out
protocol:

```text
.\.venv\Scripts\python.exe -m research.run_baseline_cv --all-models --all-classifiers --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv --overwrite
```

This command requires the optional XGBoost package for the `xgboost` rows.
Without it, the seven non-XGBoost classifiers still run, the XGBoost runs are
reported as failures, and the process exits with a nonzero status so the
missing comparison cannot be mistaken for a complete experiment.

Results are written without changing the desktop application:

```text
research_results/baseline_cv/<model_id>/<classifier>/<tag>/
```

Each result contains per-fold predictions, metrics, scaler parameters and
confusion matrices, plus aggregate mean, sample standard deviation, and a
two-sided 95% Student-t confidence interval.

## Summarize the four-model baseline

After all four models have completed the same classifier baseline, build the
consolidated publication table and figures:

```text
.\.venv\Scripts\python.exe -m research.summarize_baseline_cv
```

To summarize a different completed classifier:

```text
.\.venv\Scripts\python.exe -m research.summarize_baseline_cv --classifier rbf_svm
```

To summarize a non-default embedding artifact tag, pass the exact tag used by
the CV runner:

```text
.\.venv\Scripts\python.exe -m research.summarize_baseline_cv --tag smoke
```

Before generating any result, the summarizer verifies that all four runs share
the same manifest SHA-256, persisted fold values, sample count and speaker
count. It also verifies the stored fold means and sample standard deviations
against `fold_metrics.csv`.

The default linear-SVM summary is written to:

```text
research_results/baseline_cv/summary/linear_svm/full/
```

Generated artifacts:

```text
model_comparison.csv
model_comparison.json
accuracy_bar.png
accuracy_bar.pdf
macro_f1_bar.png
macro_f1_bar.pdf
warm_extraction_timing_bar.png
warm_extraction_timing_bar.pdf
```

Accuracy and macro-F1 charts show the five-fold mean with fold sample-standard-
deviation error bars. Plot error bounds are clipped to the physical 0–100%
range only for display; the unmodified means, standard deviations and 95%
Student-t intervals remain in the CSV and JSON. Extraction median and mean
times come from each full embedding run's warm-timing metadata.

Summarize results created with an explicit external protocol by supplying the
same protocol file:

```text
.\.venv\Scripts\python.exe -m research.summarize_baseline_cv --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv
```

The protocol file is hashed and converted to the exact same safe protocol ID
used by `run_baseline_cv`. Its file hash, ordered assignment hash, folds,
sample count, speaker count and source-group count are checked against every
model's `run_metadata.json`. The protocol-specific summary remains isolated:

```text
research_results/baseline_cv/protocols/<protocol_id>/summary/<classifier>/<tag>/
```

## Summarize the strict classifier benchmark

After the strict protocol has been evaluated for every embedding model and
classifier, create the complete model-by-classifier comparison:

```text
.\.venv\Scripts\python.exe -m research.summarize_classifier_benchmark --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv
```

The required classifier grid is:

```text
linear_svm
rbf_svm
logistic_regression
knn
random_forest
decision_tree
xgboost
```

`cosine_centroid` is included as an additional baseline when completed for the
model grid. By default the summarizer fails if a required result is missing or
incomplete. During development only, a clearly labelled partial summary can be
generated from the currently completed runs:

```text
.\.venv\Scripts\python.exe -m research.summarize_classifier_benchmark --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv --allow-missing
```

Before combining values, the summarizer re-hashes the supplied protocol and
verifies its ordered assignments, fold counts, sample count, speaker count and
source-group count against every run. For each embedding model, every
classifier must also reference the exact same embedding artifact SHA-256 map.
Aggregate means and fold sample standard deviations are recomputed from
`fold_metrics.csv` and checked against `aggregate_metrics.json`.

The default destination is:

```text
research_results/baseline_cv/protocols/<protocol_id>/summary/classifier_benchmark/<tag>/
```

Generated artifacts include:

```text
classifier_benchmark_long.csv
classifier_benchmark_long.json
accuracy_matrix.csv
macro_f1_matrix.csv
training_time_seconds_matrix.csv
prediction_time_ms_per_sample_matrix.csv
accuracy_heatmap.png / .pdf
macro_f1_heatmap.png / .pdf
training_time_heatmap.png / .pdf
prediction_time_heatmap.png / .pdf
training_time_bar.png / .pdf
prediction_time_bar.png / .pdf
best_classifier_per_model.csv
best_classifier_per_model.json
benchmark_metadata.json
```

All figures are labelled as results from the strict chapter-held-out
closed-set protocol. Missing cells in a partial development summary are shown
as `NA`; a partial summary must not be presented as a complete publication
benchmark.

## Run leakage-free nested SVM tuning

Tune the linear and RBF SVM variants without using the strict outer test
chapters for model selection:

```text
.\.venv\Scripts\python.exe -m research.run_nested_svm_tuning --all-models --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv
```

The default conditional search space is:

```text
linear: C in [0.01, 0.1, 1, 10, 100]
RBF:    C in [0.01, 0.1, 1, 10, 100]
        gamma in [scale, auto, 0.0001, 0.001, 0.01]
```

For each strict outer fold, the corresponding chapters are reserved as an
untouched final test set. `GridSearchCV` receives only the outer-training
partition, fits a `Pipeline(StandardScaler, SVC)`, selects the configuration
with the highest inner macro-F1, refits it on all outer-training utterances,
and evaluates it once on the reserved outer test.

The three inner folds are deterministic, shuffled and utterance-stratified.
They are not chapter-grouped because some speakers have only one chapter left
after the outer chapter is held out; a grouped inner split could not retain
every speaker in every inner fold. This limitation affects hyperparameter
selection only. The outer chapter-held-out evaluation remains group-disjoint,
and the inner search never accesses its test samples.

Custom grids can be supplied explicitly:

```text
.\.venv\Scripts\python.exe -m research.run_nested_svm_tuning --model speechbrain_ecapa --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv --c-grid 0.1 1 10 --gamma-grid scale auto 0.001 0.01 --overwrite
```

Results are isolated from the application and fixed-default baselines:

```text
research_results/hyperparameter_tuning/protocols/<protocol_id>/svm/<model_id>/<tag>/
```

Each result contains per-outer-fold predictions and confusion matrices, every
`GridSearchCV.cv_results_` field in CSV and JSON form, selected parameters,
search/refit/prediction timings, aggregate metrics, hashes, environment
versions, and a leakage audit. Cross-model runs are rejected unless all models
have identical ordered clips, labels, folds and source manifest hashes.

## Summarize leakage-free nested SVM tuning

After `run_nested_svm_tuning` has completed all four embedding models under
the same strict chapter-held-out protocol, validate the result set and compare
the untouched outer-fold scores with the fixed linear- and RBF-SVM baselines:

```text
.\.venv\Scripts\python.exe -m research.summarize_svm_tuning --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv
```

The default input layout is:

```text
research_results/hyperparameter_tuning/protocols/<protocol_id>/svm/<model_id>/<tag>/
research_results/baseline_cv/protocols/<protocol_id>/<model_id>/linear_svm/<tag>/
research_results/baseline_cv/protocols/<protocol_id>/<model_id>/rbf_svm/<tag>/
```

Alternative locations can be supplied with `--tuning-root`,
`--baseline-root`, `--config`, and `--output-root`. The summarizer deliberately
has no partial-results mode: every one of the four tuned runs and all eight
matching fixed-baseline runs must exist and pass validation.

Before combining any scores, it re-hashes the protocol and verifies the exact
ordered assignments, fold/sample counts, speaker/source-group counts, outer
leakage audit, selected-parameter records, aggregate values, embedding
artifact hashes, manifest identity, and per-model tuned/baseline alignment.
The selection metric must be inner-CV macro-F1, and the audit must state that
the outer test fold was never used for tuning.

The default summary destination is:

```text
research_results/hyperparameter_tuning/protocols/<protocol_id>/summary/svm/<tag>/
```

Generated artifacts:

```text
model_level_comparison.csv / .json
fold_level_comparison.csv / .json
selected_parameter_frequency.csv / .json
search_training_time.csv / .json
accuracy_comparison.png / .pdf
macro_f1_comparison.png / .pdf
delta_vs_default.png / .pdf
summary_metadata.json
```

The accuracy, macro-F1, and delta figures compare nested tuned SVM against the
fixed linear and fixed RBF defaults on the same outer folds. Because the
strict protocol contains only two correlated outer folds, all means, standard
deviations, and paired deltas are descriptive. The summary performs no
significance test and its outputs must not be described as evidence of
statistical significance.

## Run short-duration and additive-noise robustness experiments

The robustness study preserves the same strict chapter-held-out protocol. It
extracts transformed embeddings for the exact ordered protocol clips, without
retaining derived audio files:

```text
.\.venv\Scripts\python.exe -m research.extract_robustness_embeddings --all-models --all-conditions --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv
```

The predefined conditions are:

```text
short_0p5s, short_1s, short_2s, short_3s
noise_white_20db, noise_white_10db, noise_white_0db
```

Short clips use a deterministic center crop and zero-padding only when a source
recording is shorter than the target. White noise uses a clip-specific random
seed derived from the global seed, clip ID, and condition name. Mixtures are
scaled only to prevent clipping; measured SNR is recorded per clip. Use
`--checkpoint-every`, `--resume`, or `--overwrite` for long runs. A lightweight
compatibility test can be run first:

```text
.\.venv\Scripts\python.exe -m research.extract_robustness_embeddings --all-models --condition short_0p5s --limit 5 --tag smoke --overwrite
```

Condition embeddings are stored separately from clean and application data:

```text
research_results/robustness_embeddings/<protocol_id>/<model_id>/<condition>/<tag>/
```

After every requested condition has 100% aligned coverage, run the benchmark:

```text
.\.venv\Scripts\python.exe -m research.run_robustness_benchmark --all-models --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv --overwrite
```

For each outer fold, `StandardScaler` and `LogisticRegression(C=1)` are fitted
only on clean, full-duration outer-training embeddings. That frozen pipeline is
evaluated on clean and transformed versions of the exact held-out clips. No
transformed outer-test sample is used for fitting. Clip IDs, labels, fold
assignments, source-group isolation, protocol hashes, and embedding dimensions
are validated before scores are combined.

The summary is written to:

```text
research_results/robustness_benchmark/protocols/<protocol_id>/summary/logistic_regression/<tag>/
```

It includes `model_condition_summary.csv/json`, short-duration accuracy plots,
and noise accuracy plots in PNG and PDF formats. Synthetic white noise is a
controlled diagnostic condition, not a substitute for a real environmental
noise corpus such as MUSAN. With only two correlated strict outer folds, all
robustness means, standard deviations, and degradations are descriptive; no
statistical-significance claim is made.

## Run the preprocessing factorial ablation

The preprocessing study uses a complete `2^3` factorial design for three
binary factors: frame-RMS silence trimming, stationary spectral-gate noise
reduction, and peak normalization. The raw condition reuses the validated
clean embeddings; the seven non-raw combinations are extracted with:

```text
.\.venv\Scripts\python.exe -m research.extract_preprocessing_embeddings --all-models --all-variants --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv
```

The fixed order is trim, denoise, then normalize. Trimming uses a NumPy
frame-RMS gate at 30 dB below the maximum frame RMS, denoising uses the
stationary `noisereduce` spectral gate, and normalization sets peak magnitude
to 0.95. Exact parameters are saved in every `metadata.json`. Temporary WAV
files are removed after embedding extraction; derived audio is not retained.
Long runs support `--checkpoint-every`, `--resume`, and `--overwrite`.

Variant embeddings are isolated under:

```text
research_results/preprocessing_embeddings/<protocol_id>/<model_id>/<variant>/<tag>/
```

Run the strict benchmark only after all 28 model/variant artifacts are
complete:

```text
.\.venv\Scripts\python.exe -m research.run_preprocessing_ablation --all-models --protocol-file manifests\librispeech_dev_clean_chapter_heldout_2fold_protocol.csv --overwrite
```

For each variant and outer fold, `StandardScaler` and
`LogisticRegression(C=1)` are fitted only on that variant's outer-training
embeddings and evaluated on the matching held-out variant. The benchmark
checks clip order, labels, folds, protocol hashes, speaker coverage, and
chapter/source-group isolation before fitting. It reports per-variant scores,
deltas from raw, balanced factorial main effects, all two-way interactions,
and the three-way interaction.

Publication tables and figures are written to:

```text
research_results/preprocessing_benchmark/protocols/<protocol_id>/summary/logistic_regression/<tag>/
```

The two strict outer folds are correlated, so fold means, standard deviations,
and factorial contrasts are descriptive. The pipeline does not perform or
claim a statistical-significance test.
