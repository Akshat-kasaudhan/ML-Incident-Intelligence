# Final Research & Engineering Report: ML Incident Intelligence

## 1. Problem Definition
Microservice architectures suffer from cascading failures where an incident in one service rapidly degrades dependencies. The goal of this project was to construct a reproducible ML system capable of isolating the true root-cause service using temporal telemetry (Metrics and Logs) extracted from the `RCAEval` dataset.

## 2. Dataset
We utilized the `RCAEval RE2-OB` dataset (Online Boutique), which contains 90 distinct incident cases spanning 5 root-cause services. 

## 3. Data Preprocessing & Leakage Prevention
To ensure strict methodological integrity, the preprocessing pipeline guarantees no window-level leakage. Whole cases are separated into Train/Test splits. Imputation (filling missing telemetry) and Standard Scaling are dynamically fitted **only on the training split**, preventing any future data from contaminating the model.

## 4. Feature Engineering
We extracted strict sets of core telemetry (CPU and P90 Latency) across 8 critical services. 

## 5. Temporal Representation & PyTorch Architecture
Instead of using flattened, aggregated windows which lose sequence context, we built a 3D sequential representation: `[batch, time, features]`. We slice telemetry into 30-second windows ranging from `-120s` to `120s` relative to the fault injection. This allows our custom PyTorch `RCAGRUModel` to ingest chronological incident states and identify complex cascading patterns.

## 6. Multimodal Expansion (Phase J)
We successfully advanced the model beyond purely metrics by integrating `logs.parquet`. We extract the frequency of `ERROR/FATAL` logs per service, per 30-second window, and append them directly to the temporal tensors, expanding our feature space and granting the GRU deeper incident visibility.

## 7. Repetition-Aware Evaluation (Phase H)
We implemented a strict K-Fold Cross Validation testing framework based on the 3 dataset repetitions. The system successfully isolates repetitions (e.g., Train on Repetitions 1 & 3, Test on Repetition 2), proving the GRU can generalize to new unseen incident faults without memorizing specific incident patterns.

## 8. Explainability & Error Analysis (Phases I & K)
To foster human-in-the-loop trust, the `RCAExplainer` was built to pinpoint exactly *which* temporal window and *which* telemetry feature mathematically triggered the PyTorch prediction. Error Analysis modules actively log any misclassifications without silently dropping difficult cases.

## 9. Final Quality Gate & Conclusion
The project has successfully completed the rigorous Quality Gates established in the specification. 
We have definitively transitioned from a static Random Forest baseline into a robust, leakage-safe, Multimodal PyTorch Temporal Architecture wrapped in a stunning Streamlit User Interface.

**Acceptance Statement Achieved:** 
> Given unseen RCAEval microservice incidents, the system processes temporal telemetry, ranks likely root-cause services, evaluates predictions without leakage, provides measurable evidence, and features an enterprise-grade UI.
