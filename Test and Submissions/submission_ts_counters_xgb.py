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


# Define features and target
features = ["counter_name", "hour", "dayofweek", "quarter", "month", "dayofyear"]
target = ["log_bike_count"]

# Prepare training data
X_train = train[features]
y_train = train[target]

new_features = ["hour", "dayofweek", "quarter", "month", "dayofyear"]

models = {}

# Train the model
for counter in train["counter_name"].unique():
    train_subset = train[train["counter_name"] == counter]

    X_train = train_subset[new_features]
    y_train = train_subset["log_bike_count"]

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    models[counter] = model

predictions_df = pd.DataFrame()

# Make predictions
for counter in test["counter_name"].unique():
    test_subset = test[test["counter_name"] == counter]

    if counter in models:
        X_test = test_subset[new_features]
        pred = models[counter].predict(X_test)
        pred_df = pd.DataFrame(
            {"counter_name": counter, "log_bike_count": pred}, index=test_subset.index
        )
        predictions_df = pd.concat([predictions_df, pred_df])


# Create a DataFrame for predictions with the same index as test_data
predictions_df = predictions_df.drop("counter_name", axis=1)
predictions_df.reset_index(drop=True)
predictions_df.to_csv("submissions.csv", index=True, index_label="Id")
