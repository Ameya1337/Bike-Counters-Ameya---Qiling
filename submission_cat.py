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

cols_to_drop_train = ["counter_id", "site_id", "date", "counter_installation_date"]

data = data.drop(cols_to_drop_train, axis=1)
data = data.drop("bike_count", axis=1)
X = data.drop("log_bike_count", axis=1)
y = data["log_bike_count"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for feature in categorical_cols:
    X_train[feature] = X_train[feature].astype(str)
    X_test[feature] = X_test[feature].astype(str)


best_params = {
    "iterations": 1402,
    "depth": 10,
    "learning_rate": 0.03073417636647055,
    "l2_leaf_reg": 9.871350035301967,
}


train_pool = Pool(
    X_train, y_train, cat_features=categorical_cols, feature_names=list(X_train.columns)
)


test_pool = Pool(
    X_test, y_test, cat_features=categorical_cols, feature_names=list(X_test.columns)
)


best_cat_reg = CatBoostRegressor(
    cat_features=categorical_cols,
    verbose=True,
    **best_params,
    early_stopping_rounds=10,
)

best_cat_reg.fit(train_pool, eval_set=test_pool)


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

predictions = best_cat_reg.predict(test_data)
predictions_df = pd.DataFrame({"log_bike_count": predictions})
predictions_df.to_csv("submissions.csv", index=True, index_label="Id")
