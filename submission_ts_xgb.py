import pandas as pd
import xgboost as xgb

data = pd.read_parquet("/kaggle/input/mdsb-2023/train.parquet")
test_data = pd.read_parquet("/kaggle/input/mdsb-2023/final_test.parquet")
data["date"] = pd.to_datetime(data["date"])
data = data.set_index("date")
test_data["date"] = pd.to_datetime(test_data["date"])
test_data = test_data.set_index("date")
selected_counter_name = "Totem 73 boulevard de Sébastopol S-N"
train = data[data["counter_name"] == selected_counter_name]
test = test_data[["counter_name"]]
test = test[test["counter_name"] == selected_counter_name]


def create_features(train):
    """
    Create time series features based on timeseries index
    """
    train["hour"] = train.index.hour
    train["dayofweek"] = train.index.dayofweek
    train["quarter"] = train.index.quarter
    train["month"] = train.index.month
    train["dayofyear"] = train.index.dayofyear
    return train


train = create_features(train)
test = create_features(test)
best_params = {
    "colsample_bytree": 0.9,
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_child_weight": 2,
    "n_estimators": 500,
    "subsample": 0.9,
}
best_reg = xgb.XGBRegressor(
    tree_method="hist",
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    min_child_weight=best_params["min_child_weight"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    enable_categorical=True,
)

features = ["counter_name", "hour", "dayofweek", "quarter", "month", "dayofyear"]
target = ["log_bike_count"]

X_train = train[features]
y_train = train[target]

best_reg.fit(
    X_train,
    y_train,
    verbose=10,
)

X_test = test[features]

predictions = best_reg.predict(X_test)
predictions_df = pd.DataFrame({"log_bike_count": predictions})
predictions_df.to_csv("submissions.csv", index=True, index_label="Id")
