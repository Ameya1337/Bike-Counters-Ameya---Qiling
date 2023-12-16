import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

data = pd.read_parquet("/kaggle/input/mdsb-2023/train.parquet")
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

cols_to_drop_train = [
    "counter_id",
    "site_id",
    "date",
    "counter_installation_date",
    "coordinates",
]

data = data.drop(cols_to_drop_train, axis=1)
data = data.drop("bike_count", axis=1)
X = data.drop("log_bike_count", axis=1)
y = data["log_bike_count"]

# preprocess your data if you want


# best parameters hwich you got after hypertuning
best_params = {
    # your code goes here
}

# create a pipeline if you want

# your model
model = your_model()


# fit your model/pipeline
model.fit(X, y)


test_data = pd.read_parquet("/kaggle/input/mdsb-2023/final_test.parquet")
test_data["date"] = pd.to_datetime(test_data["date"])
test_data["day_of_week"] = test_data["date"].dt.dayofweek
test_data["month"] = test_data["date"].dt.month
test_data["hour"] = test_data["date"].dt.hour
cols_to_drop_test = [
    "counter_id",
    "site_id",
    "date",
    "counter_installation_date",
    "coordinates",
]

test_data = test_data.drop(cols_to_drop_test, axis=1)

predictions = model.predict(test_data)  # predict using your model or pipeline

predictions_df = pd.DataFrame({"log_bike_count": predictions})
predictions_df.to_csv("submissions.csv", index=True, index_label="Id")
