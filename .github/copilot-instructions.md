# Copilot Instructions: RoadSafety-ML-Pipeline

## Project goal
Build a clear, beginner-friendly machine learning pipeline to predict traffic accident occurrence.

## Data context
- Dataset: datasets/dataset_traffic_accident_prediction1.csv
- Rows: 840, Columns: 14
- Missing values: 42 per column
- Target: Accident (binary)
- Numerical features: Traffic_Density, Speed_Limit, Number_of_Vehicles, Driver_Alcohol, Driver_Age, Driver_Experience
- Categorical features: Weather, Road_Type, Time_of_Day, Accident_Severity, Road_Condition, Vehicle_Type, Road_Light_Condition

## Modeling approach
- Foundational models only: Logistic Regression, Decision Tree, Random Forest, SVM
- Validation: train-test split only (no cross-validation for now)
- Imputation: numerical median, categorical most frequent
- Encoding: one-hot for categorical features
- Scaling: standardize numeric features for linear and SVM models

## Workflow guidance
- Keep reusable logic in src/ modules
- Use scripts for the main pipeline; create notebooks only for EDA when asked
- Prefer small, testable functions with clear inputs and outputs

## Outputs
- Save metrics to results/model_metrics.csv
- Save models to models/<model_name>.joblib
