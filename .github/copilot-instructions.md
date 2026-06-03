# Copilot Instructions: RoadSafety-ML-Pipeline (Notebook-Only)

## Project goal
Build a clear, beginner-friendly machine learning pipeline to predict traffic accident occurrence using Jupyter notebooks for interactive learning and exploration.

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
- All code lives in Jupyter notebooks (notebooks/ directory)
- Each phase has its own numbered notebook: 01_EDA, 02_Preprocessing, 03_Feature_Engineering, etc.
- Document findings and decisions in markdown cells
- Execute cells sequentially to build the pipeline step-by-step
- Save outputs (models, metrics) to models/ and results/ directories

## Notebook structure
- Each notebook starts with imports and data loading
- Use markdown headers to organize sections clearly
- Add comments explaining what each cell does
- Include visualizations to explore patterns and validate preprocessing

## Outputs
- Save trained models to models/<model_name>.joblib
- Save metrics to results/model_metrics.csv and results/model_metrics.json
- Keep visualizations in notebooks for interactive feedback
