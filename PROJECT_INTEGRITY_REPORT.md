# Project Integrity Report

Updated: 12 August 2026

## Corrections implemented

- The research UI and IEEE paper now treat the chapter-held-out protocol as
  the primary scientific comparison. The utterance-level five-fold result is
  labelled as an optimistic sensitivity analysis.
- Every embedding model uses its own calibrated open-set detector. A WavLM,
  UniSpeech-SAT or X-Vector prediction is no longer gated by ECAPA-TDNN.
- Unknown calibration, validation and final test identities are split by
  speaker identity with zero identity overlap.
- The six-row probability view is hierarchical: the five closed-set identity
  probabilities are multiplied by `P(known)`, while the sixth row is
  `P(unknown) = 1 - P(known)`. The six displayed rows sum to 100%.
- Related audio files are ranked by cosine similarity in the selected model's
  embedding space. The query file itself is excluded.
- Short clips below 0.25 seconds and exact duplicate audio are removed when a
  path index is available. ECAPA's missing path index was restored only after
  confirming exact label-order equality with the X-Vector extraction.
- SVM and Logistic Regression classifiers were retrained after filtering.
- Cross-dataset results identify the unequal class counts (40 vs 5 speakers),
  include chance-normalized values and are labelled descriptive rather than a
  controlled model ranking.
- PDF export contains a shared overview, one complete page per model, all six
  probabilities, related audio filenames, and a final page containing both the
  waveform and Mel spectrogram.
- Settings that are not valid live-inference controls are disabled or labelled
  accurately. Automatic plot generation remains functional.

## Current application-corpus classifier metrics

These values come from the Kaggle-derived five-speaker application corpus.
They use an utterance-level stratified holdout and are not the primary IEEE
research estimates.

| Embedding | SVM train | SVM test | Logistic train | Logistic test | Quality samples |
|---|---:|---:|---:|---:|---:|
| ECAPA-TDNN | 98.52% | 88.74% | 91.52% | 86.65% | 7,501 -> 7,414 |
| X-Vector | 97.05% | 91.17% | 97.45% | 90.69% | 7,501 -> 7,414 |
| WavLM | 99.44% | 88.47% | 97.36% | 86.83% | 7,329 -> 7,328 |
| UniSpeech-SAT | 98.22% | 89.56% | 96.73% | 87.30% | 7,084 -> 7,083 |

## Current application open-set detector metrics

| Embedding | Test ROC-AUC | Test FAR | Test FRR | Unknown identity overlap |
|---|---:|---:|---:|---:|
| ECAPA-TDNN | 0.997 | 2.47% | 2.52% | 0 |
| X-Vector | 0.943 | 6.11% | 10.33% | 0 |
| WavLM | 0.901 | 10.19% | 21.82% | 0 |
| UniSpeech-SAT | 0.943 | 4.95% | 19.57% | 0 |

FAR means an unknown speaker was incorrectly accepted as enrolled. FRR means
an enrolled speaker was incorrectly rejected as unknown.

## Limitations that must remain disclosed

1. The original five-speaker corpus has no source recording or session IDs.
   Adjacent one-second fragments may therefore appear in both training and
   test partitions. Its holdout scores can be optimistic.
2. The public encoders' full pre-training identity lists cannot be exhaustively
   cross-checked against all evaluation identities. Possible pre-training
   identity overlap remains a threat to validity.
3. Environmental noise conditions are deterministic synthetic proxies, not a
   replacement for a real-noise corpus such as MUSAN plus room impulse
   responses.
4. The strict chapter-held-out protocol has only two correlated outer folds.
   Statistical conclusions are therefore low-power; the five-fold Wilcoxon
   analysis is also limited by `n = 5`.
5. Computational profiling was performed on one Windows CPU host. It is not a
   universal latency or memory benchmark and does not measure GPU performance.
6. Open-set performance should be validated on a larger external cohort before
   production deployment. A low FAR/FRR on the current cohort is not a
   guarantee for every microphone, language, noise condition or population.

These are scientific constraints, not hidden implementation failures. The
project reports them instead of claiming that the available data can support a
stronger conclusion.
