# ML Incident Intelligence — Industry-Grade ML Root Cause Analysis System

You are working as a senior ML Engineer / ML Research Engineer on an industry-oriented project called:

**ML Incident Intelligence & Root Cause Analysis (RCA)**

The goal is to build a serious, reproducible ML system for diagnosing incidents in distributed microservice systems using telemetry.

The project is considered complete only when the following criteria are satisfied.

## A. Data & Reproducibility

* [ ] RCAEval RE2-OB is reproducibly downloadable/usable.
* [ ] All 90 RE2-OB cases are accounted for.
* [ ] Case metadata, root-cause service, fault type, and repetition are preserved.
* [ ] Raw telemetry is never overwritten.
* [ ] `Run All` on the main notebook completes without errors.
* [ ] No required variable depends on an accidentally executed previous cell.
* [ ] Random seeds are fixed wherever randomness is used.
* [ ] The preprocessing pipeline can be executed from a clean runtime.

## B. Leakage Prevention

* [ ] No windows from the same case appear in both train and test.
* [ ] Imputation is fitted only on training data.
* [ ] Scaling is fitted only on training data.
* [ ] Feature selection is fitted only on training data where applicable.
* [ ] Test-set information is never used during model development.
* [ ] Root-cause labels are never included as model features.
* [ ] Fault labels are never included as model features.
* [ ] Future telemetry is not used when generating an earlier prediction.
* [ ] Repetition-aware evaluation is performed without repetition leakage.

## C. Classical ML Baseline

A reproducible Random Forest baseline must exist.

It must report:

* [ ] Window-level accuracy
* [ ] Window-level macro F1
* [ ] Weighted F1
* [ ] Confusion matrix
* [ ] Per-class metrics
* [ ] Case-level Top-1 accuracy
* [ ] Case-level Top-3 accuracy

The exact train/test split must be recorded.

The baseline must remain available for comparison with every later model.

## D. Temporal Dataset

The temporal representation must satisfy:

```text
[batch, time, features]
```

and:

* [ ] Each incident is represented as a temporal sequence.
* [ ] Sequence ordering is preserved.
* [ ] Variable-length incidents are supported.
* [ ] Padding does not become artificial telemetry evidence.
* [ ] Sequence lengths are explicitly tracked.
* [ ] The temporal dataset can be reconstructed reproducibly from the processed data.
* [ ] Train/test preprocessing remains leakage-safe.

## E. PyTorch Fundamentals

Before the GRU implementation:

* [ ] Tensor shapes are understood and verified.
* [ ] Dataset implementation works.
* [ ] DataLoader implementation works.
* [ ] A basic `nn.Module` works.
* [ ] Forward pass works.
* [ ] CrossEntropyLoss works.
* [ ] Backpropagation works.
* [ ] Adam optimizer works.
* [ ] A tiny neural-network training example successfully reduces loss.

The project must not jump directly into a complicated deep-learning architecture without verifying these fundamentals.

## F. GRU Model

A PyTorch GRU model must successfully:

* [ ] Accept temporal input in `[batch, time, features]` format.
* [ ] Handle variable sequence lengths correctly.
* [ ] Produce logits for the 5 root-cause services.
* [ ] Train without NaN/Inf loss.
* [ ] Save and reload model weights successfully.
* [ ] Produce deterministic inference when the appropriate seed/settings are used.
* [ ] Have a documented architecture and hyperparameters.

## G. Fair Model Comparison

Random Forest and GRU must be compared using:

* [ ] Same underlying cases.
* [ ] Same ground-truth definitions.
* [ ] Comparable leakage-safe splits.
* [ ] Same primary evaluation metrics.
* [ ] Case-level as well as window-level evaluation.

The project must explicitly answer:

> Does temporal modeling provide useful improvement over the classical window-based baseline?

Do not claim improvement if the measured results do not demonstrate it.

## H. Robustness

The final evaluation must include repetition-aware testing.

At minimum:

```text
Train: repetitions 2 + 3
Test:  repetition 1

Train: repetitions 1 + 3
Test:  repetition 2

Train: repetitions 1 + 2
Test:  repetition 3
```

For each fold report:

* [ ] Window accuracy
* [ ] Window macro F1
* [ ] Case Top-1
* [ ] Case Top-3
* [ ] Confusion matrix
* [ ] Per-service performance

Results must not be averaged in a way that hides a severe failure on one repetition.

## I. Error Analysis

The final system must identify and inspect its failures.

For meaningful errors:

* [ ] Case ID is recorded.
* [ ] True root cause is recorded.
* [ ] Predicted root cause is recorded.
* [ ] Prediction confidence is recorded.
* [ ] Relevant temporal windows are inspected.
* [ ] Important telemetry evidence is inspected.
* [ ] Possible reasons for the error are documented.

Difficult cases must NOT be removed simply because they reduce performance.

## J. Multimodal Expansion

The final system should progress through measurable stages:

```text
Metrics only
      ↓
Metrics + Logs
      ↓
Metrics + Traces
      ↓
Metrics + Logs + Traces
```

For each stage:

* [ ] Define what information from the modality is used.
* [ ] Convert the modality into a reproducible representation.
* [ ] Integrate it into the temporal pipeline.
* [ ] Evaluate it on the same appropriate split.
* [ ] Compare against the previous stage.

The project must not claim "multimodal RCA" unless multiple modalities are actually consumed by the model.

## K. Root-Cause Explanation

For every prediction, the system should be capable of producing structured evidence such as:

```text
Predicted service:
currencyservice

Predicted fault:
socket

Confidence:
...

Supporting evidence:
- ...
- ...
- ...

Relevant time window:
...
```

Evidence must originate from actual telemetry/model analysis.

No explanation may be manually fabricated to match the prediction.

## L. Production Readiness

The final repository must contain reasonably separated components for:

```text
data
preprocessing
features
models
training
evaluation
inference
explainability
application
tests
```

At minimum:

* [ ] Training can be run independently of inference.
* [ ] A saved model can be loaded for inference.
* [ ] Preprocessing required for inference is saved/reproducible.
* [ ] Model predictions can be generated without opening the training notebook.
* [ ] Basic error handling exists.
* [ ] README explains installation and execution.
* [ ] Requirements are documented.

## M. Application

The final application should demonstrate:

* [ ] Incident input/loading.
* [ ] Root-cause prediction.
* [ ] Top-3 candidate services.
* [ ] Confidence/probability information.
* [ ] Fault information where supported.
* [ ] Incident timeline.
* [ ] Supporting telemetry evidence.
* [ ] Clear distinction between model prediction and ground truth.

The UI must represent the actual trained model and preprocessing pipeline.

It must not use hardcoded predictions.

## N. Final Research/Engineering Report

The final project must clearly document:

1. Problem definition
2. Dataset
3. Data preprocessing
4. Feature engineering
5. Random Forest baseline
6. Temporal representation
7. PyTorch architecture
8. GRU experiments
9. Repetition-aware evaluation
10. Error analysis
11. Multimodal experiments
12. Explainability
13. Deployment architecture
14. Limitations
15. Future improvements

Include actual measured results.

Do not manufacture benchmark numbers.

## O. Final Quality Gate

The project should NOT be considered complete if any of the following are true:

* The model only works because of window-level leakage.
* The test set influenced preprocessing.
* The GRU cannot be reproduced from a clean runtime.
* The final reported result comes from a single convenient split.
* Difficult cases were removed without methodological justification.
* Logs/traces are mentioned but not actually used.
* Explanations are hardcoded.
* The application uses a different preprocessing pipeline from training.
* The Random Forest baseline is missing.
* Results are reported without explaining the evaluation protocol.

### Final acceptance statement

The project is accepted only when we can demonstrate, with reproducible experiments, that:

> **Given unseen RCAEval microservice incidents, the system can process temporal telemetry, rank likely root-cause services, evaluate its predictions without case/repetition leakage, provide measurable evidence for its predictions, and be compared fairly against a strong classical ML baseline.**

The final system should prioritize **reproducibility, robustness, temporal reasoning, and engineering correctness over an artificially high benchmark score.**

---

# 1. Dataset
Use **RCAEval** as the primary benchmark dataset.
Repository: `phamquiluan/RCAEval`
Current development subset: **RE2-OB (Online Boutique)**

# 2. Current project progress
A substantial amount of work has already been completed. DO NOT restart the project from zero.
The current project stage is:
**PyTorch fundamentals → temporal sequence model**

# 3. Existing baseline
The current classical ML pipeline is:
Telemetry → 30-second windows → mean / max / baseline delta / trend → Random Forest → root-cause service

# 4. Target architecture
Progressively evolve toward: Baseline Random Forest and PyTorch GRU Temporal Model.

# 5. PyTorch learning requirement
Phase A — tensors
Phase B — Dataset and DataLoader
Phase C — neural network fundamentals
Phase D — training

# 6. GRU model
Build a clean PyTorch GRU model. (batch, time, features) -> GRU -> final hidden -> Linear -> 5 classes (services).
