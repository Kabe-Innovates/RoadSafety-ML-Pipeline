from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "datasets" / "dataset_traffic_accident_prediction1.csv"

TARGET_COLUMN = "Accident"

NUMERIC_FEATURES = [
    "Traffic_Density",
    "Speed_Limit",
    "Number_of_Vehicles",
    "Driver_Alcohol",
    "Driver_Age",
    "Driver_Experience",
]

CATEGORICAL_FEATURES = [
    "Weather",
    "Road_Type",
    "Time_of_Day",
    "Accident_Severity",
    "Road_Condition",
    "Vehicle_Type",
    "Road_Light_Condition",
]

NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"

TEST_SIZE = 0.2
RANDOM_SEED = 42

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "cache"
