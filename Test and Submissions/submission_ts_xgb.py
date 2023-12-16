import pandas as pd
import xgboost as xgb

# Load the data
data = pd.read_parquet("data/train.parquet")
test_data = pd.read_parquet("data/final_test.parquet")

# Convert 'date' to datetime and set as index
data["date"] = pd.to_datetime(data["date"])
data = data.set_index("date")
test_data["date"] = pd.to_datetime(test_data["date"])
test_data = test_data.set_index("date")

train = data[["counter_name", "log_bike_count"]]
test = test_data[["counter_name"]]


def create_features(df):
    # Ensure the index is a datetime index
    df = df.copy()  # This avoids SettingWithCopyWarning
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)

    # Create time series features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    return df


# Create features for both training and test datasets
train = create_features(train)
test = create_features(test)

# XGBoost parameters
best_params = {
    "colsample_bytree": 0.9,
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_child_weight": 2,
    "n_estimators": 500,
    "subsample": 0.9,
}

# Initialize XGBoost Regressor
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

# Define features and target
features = ["counter_name", "hour", "dayofweek", "quarter", "month", "dayofyear"]
target = ["log_bike_count"]

# Prepare training data
X_train = train[features]
y_train = train[target]

# Train the model
best_reg.fit(
    X_train,
    y_train,
    verbose=10,
)

# Prepare test data
X_test = test[features]

# Make predictions
predictions = best_reg.predict(X_test)

# Create a DataFrame for predictions with the same index as test_data
predictions_df = pd.DataFrame({"log_bike_count": predictions}, index=test.index)
predictions_df.reset_index(drop=True)
predictions_df.to_csv("submissions.csv", index=True, index_label="Id")
