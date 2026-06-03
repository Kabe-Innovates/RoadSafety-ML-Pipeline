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

## Notebook-Based Workflow
All work is organized in Jupyter notebooks for interactive learning:

- **01_EDA.ipynb** — Exploratory Data Analysis
- **02_Preprocessing.ipynb** — Data Cleaning & Missing Value Handling
- **03_Feature_Engineering.ipynb** — Feature Encoding & Scaling
- **04_Model_Training.ipynb** — Train foundational models
- **05_Model_Evaluation.ipynb** — Compare models and analyze performance
- **06_Hyperparameter_Tuning.ipynb** (optional) — Improve best model
- **07_Final_Report.ipynb** — Summarize findings

## Quick start
1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter
   ```bash
   jupyter notebook
   ```

4. Open and run notebooks sequentially: 01_EDA → 02_Preprocessing → etc.

## Modeling approach
- Foundational models: Logistic Regression, Decision Tree, Random Forest, SVM
- Validation: Train-test split (80-20)
- Imputation: Median for numerical, mode for categorical features
- Encoding: One-hot for categorical features
- Scaling: StandardScaler for numeric features

## Project structure
```
notebooks/          Jupyter notebooks (main workflow)
datasets/           Raw data
models/             Trained model artifacts (.joblib)
results/            Metrics and evaluation reports
cache/              Intermediate outputs
```
