import joblib
import pandas as pd
from custom_functions import _encode_dates

loaded_model = joblib.load("CatBoost_model_basic.joblib")

data = pd.read_parquet("data/final_test.parquet")
data["date"] = pd.to_datetime(data["date"])
data["day_of_week"] = data["date"].dt.dayofweek
data["month"] = data["date"].dt.month
data["hour"] = data["date"].dt.hour
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
    data[feature] = data[feature].astype(str)

cols_to_drop = [
    "counter_id",
    "site_id",
    "date",
    "counter_installation_date",
    "coordinates",
]

data = data.drop(cols_to_drop, axis=1)


predictions = loaded_model.predict(data)
predictions_df = pd.DataFrame({"log_bike_count": predictions})
predictions_df.to_csv("submissions.csv", index=True, index_label="Id")
