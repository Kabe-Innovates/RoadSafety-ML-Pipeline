# RoadSafety-ML-Pipeline
A predictive analytics project using machine learning to forecast traffic accident risks and collision severity.

## Project goals
- Data Preprocessing & Cleaning
- EDA
- Feature Engineering & Encoding
- Machine Learning Modeling
- Model Evaluation & Tuning

## Dataset
- File: datasets/dataset_traffic_accident_prediction1.csv
- Rows: 840, Columns: 14
- Target: Accident (binary)
- Missing values: 42 per column
- Numerical: Traffic_Density, Speed_Limit, Number_of_Vehicles, Driver_Alcohol, Driver_Age, Driver_Experience
- Categorical: Weather, Road_Type, Time_of_Day, Accident_Severity, Road_Condition, Vehicle_Type, Road_Light_Condition

## Structure
- config/        Configuration and constants
- src/           Reusable pipeline modules
- notebooks/     EDA and exploration (create as needed)
- models/        Trained model artifacts
- results/       Metrics and reports
- cache/         Intermediate outputs

## Quick start
1. Create and activate a virtual environment
2. Install dependencies: pip install -r requirements.txt
3. Train baseline models: python train_pipeline.py

## Modeling approach
- Foundational models: Logistic Regression, Decision Tree, Random Forest, SVM
- Validation: Train-test split (no cross-validation initially)
