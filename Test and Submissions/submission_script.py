import joblib
import pandas as pd
from custom_functions import _encode_dates

loaded_model = joblib.load("CatBoost_model_basic.joblib")

test_data = pd.read_parquet("test_data/final_test.parquet")
test_data["date"] = pd.to_datetime(test_data["date"])
test_data["day_of_week"] = test_data["date"].dt.dayofweek
test_data["month"] = test_data["date"].dt.month
test_data["hour"] = test_data["date"].dt.hour
categorical_cols = [
    "counter_name",
    "site_name",
    "counter_technical_id",
    "day_of_week",
    "month",
    "hour",
]
numerical_cols = ["latitude", "longitude"]

for feature in categorical_cols:
    test_data[feature] = test_data[feature].astype(str)

cols_to_drop = [
    "counter_id",
    "site_id",
    "date",
    "counter_installation_date",
    "coordinates",
]

test_data = test_data.drop(cols_to_drop, axis=1)


predictions = loaded_model.predict(test_data)
predictions_df = pd.DataFrame({"log_bike_count": predictions})
predictions_df.to_csv("submissions.csv", index=True, index_label="Id")
