# Multi-Model Speaker Identification

Windows desktop and comparative research project for closed- and open-set
speaker identification. The application supports four frozen pre-trained
speaker embedding models (ECAPA-TDNN, X-Vector, WavLM and UniSpeech-SAT), with
calibrated RBF-SVM and Logistic Regression classifiers.

## Run the desktop application

From the project directory:

```powershell
.\.venv\Scripts\python.exe app.py
```

PowerShell environment activation is optional. Calling the environment's
Python executable directly avoids local execution-policy restrictions.

## Important evaluation protocols

- **Primary:** chapter-held-out two-fold evaluation. Complete source chapters
  are held out to reduce train/test leakage.
- **Secondary:** utterance-level five-fold evaluation. This result is retained
  as an optimistic sensitivity analysis because source chapters overlap.
- **Open set:** every embedding model has its own calibrated unknown-speaker
  detector. Unknown calibration, validation and final test identities are
  speaker-disjoint.

The original five-speaker corpus does not contain source recording/session
identifiers. Its application classifiers therefore use an utterance-level
split and must not be presented as leakage-free research estimates.

## Verify the completed research study

```powershell
.\.venv\Scripts\python.exe research\verify_ieee_completion.py
.\.venv\Scripts\python.exe -m unittest tests.test_research_integrity -v
```

The comparative results are stored in `research_results/ieee_complete`.
Before the integrity fixes, model and detector artifacts were backed up under
`backups/loophole_fix_20260812_100920`.
