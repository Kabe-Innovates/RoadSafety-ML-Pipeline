# RoadSafety-ML-Pipeline
Predictive analytics pipeline for traffic accident occurrence using a notebook-first, explainable ML workflow.

## Scope and Objectives
- Build a clear, reproducible end-to-end pipeline for accident prediction.
- Prioritize interpretability with foundational models and transparent preprocessing.
- Capture metrics and artifacts for stakeholder review and future iteration.

## Data Asset
- Primary file: datasets/dataset_traffic_accident_prediction1.csv
- Shape: 840 rows, 14 columns
- Target: Accident (binary)
- Missing values: 42 per column (consistent across features)
- Numerical features: Traffic_Density, Speed_Limit, Number_of_Vehicles, Driver_Alcohol, Driver_Age, Driver_Experience
- Categorical features: Weather, Road_Type, Time_of_Day, Accident_Severity, Road_Condition, Vehicle_Type, Road_Light_Condition

## Notebook Workflow
All analysis and modeling are organized in the notebooks directory:

- 01_Preprocessing.ipynb — Data cleaning and missing value handling
- 02_EDA.ipynb — Exploratory data analysis
- 03_Feature_Engineering.ipynb — Encoding, scaling, and feature preparation
- 04_Model_Training.ipynb — Train baseline models
- 05_Model_Evaluation.ipynb — Compare models and metrics
- 07_Final_Report.ipynb — Final summary and documentation

## Modeling Standards
- Models: Logistic Regression, Decision Tree, Random Forest, SVM
- Validation: train-test split (80/20)
- Imputation: median (numerical), mode (categorical)
- Encoding: one-hot for categorical variables
- Scaling: standardization for linear and SVM models

## Results (Latest Stored)
Metrics are stored in results/model_metrics.csv and results/model_metrics.json. Current highlights:

- Decision Tree: accuracy 0.9404, precision 0.8487, recall 0.9619, $F_1$ 0.9018
- Logistic Regression: accuracy 0.6504, precision 0.4277, recall 0.6762, $F_1$ 0.5240

## Project Structure
```
datasets/            Raw and intermediate data files
models/              Trained model artifacts (.joblib)
notebooks/           Primary workflow notebooks
results/             Metrics and evaluation outputs
```

## Outputs
- Trained models: models/<model_name>.joblib
- Metrics: results/model_metrics.csv, results/model_metrics.json

## Governance Notes
- Re-run preprocessing if the source dataset changes.
- Update metrics files after model retraining to keep results current.
- Document key decisions in 07_Final_Report.ipynb for stakeholder traceability.
