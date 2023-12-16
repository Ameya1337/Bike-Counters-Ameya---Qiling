# %% [code]
import joblib
import pandas as pd


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


loaded_pipe = joblib.load("/kaggle/input/sgd-init-model/SGD_init_pipeline.joblib")
test_x = pd.read_parquet("/kaggle/input/mdsb-2023/final_test.parquet")


predictions = loaded_pipe.predict(test_x)
predictions_df = pd.DataFrame({"log_bike_count": predictions})
predictions_df.to_csv("submission.csv", index=True, index_label="Id")
