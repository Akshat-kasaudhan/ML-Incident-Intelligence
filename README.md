# 🔍 ML Incident Intelligence & Root Cause Analysis

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning pipeline that acts as an "AI DevOps Engineer." By analyzing telemetry metrics (CPU, Memory, and Latency), the system automatically identifies which microservice is broken and what type of fault occurred during a cascading failure incident.

## 🚀 Features
* **Automated Data Engineering:** Extracts raw time-series metrics from Hugging Face (`phamquiluan/RCAEval`) and engineers tabular "Spike" features (post-incident max vs pre-incident mean).
* **Machine Learning Inference:** Utilizes a tuned `RandomForestClassifier` with balanced class weights to isolate the root cause service and fault type.
* **Enterprise SaaS Dashboard:** A fully deployed, real-time diagnostic web application built with Streamlit for human-in-the-loop incident response.

## 📁 Repository Structure
* `notebooks_01_rcaeval_dataset_inspection_(1).ipynb`: The Colab notebook containing Phase 1 (Feature Extraction) and Phase 2 (Model Training) code.
* `app.py`: The Streamlit web application for real-time inference (Phase 3).
* `model_service.pkl` & `model_fault.pkl`: The serialized Random Forest models.
* `feature_names.pkl`: The ordered list of features required by the models.

## 🛠️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```
   *The app will automatically open in your browser at `http://localhost:8501`.*

## 🧠 Future Roadmap
- [ ] Incorporate `traces.parquet` to build a full dependency graph.
- [ ] Parse `logs.parquet` for error code frequencies to push accuracy beyond 90%.
- [ ] Deploy the Streamlit app to Streamlit Community Cloud.
