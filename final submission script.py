import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cat
from pandas.tseries.holiday import Holiday, AbstractHolidayCalendar
from dateutil.easter import easter
from datetime import timedelta

# Load the data
data = pd.read_parquet("/kaggle/input/mdsb-2023/train.parquet")
test_data = pd.read_parquet("/kaggle/input/mdsb-2023/final_test.parquet")
data["date"] = pd.to_datetime(data["date"])
data = data.set_index("date")
test_data["date"] = pd.to_datetime(test_data["date"])
test_data = test_data.set_index("date")

train = data[["counter_name", "log_bike_count"]]
test = test_data[["counter_name"]]

weather_data = pd.read_csv(
    "/kaggle/input/external-data-cleaned-updated-2/external_data_cleaned_updated.csv"
)
weather_data["date"] = pd.to_datetime(weather_data["date"])
weather_data.set_index("date", inplace=True)

# Timeseries features


class FrenchHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1),
        Holiday("Labour Day", month=5, day=1),
        Holiday("Victory in Europe Day", month=5, day=8),
        Holiday("Bastille Day", month=7, day=14),
        Holiday("Assumption of Mary", month=8, day=15),
        Holiday("All Saints' Day", month=11, day=1),
        Holiday("Armistice Day", month=11, day=11),
        Holiday("Christmas Day", month=12, day=25),
    ]

    @staticmethod
    def easter_related_holidays(year):
        easter_sunday = easter(year)
        return [
            (easter_sunday + timedelta(days=1), "Easter Monday"),
            (easter_sunday + timedelta(days=39), "Ascension Day"),
        ]


def cyclical_encode(df, column, max_value):
    df[column + "_sin"] = np.sin(2 * np.pi * df[column] / max_value)
    df[column + "_cos"] = np.cos(2 * np.pi * df[column] / max_value)
    return df


def create_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear

    # Boolean for weekends
    df["is_weekend"] = df["dayofweek"].isin([5, 6])

    # cyclical
    df = cyclical_encode(df, "hour", 24)
    df = cyclical_encode(df, "dayofweek", 7)

    # Boolean for holidays
    cal = FrenchHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    easter_holidays = []
    for year in range(df.index.year.min(), df.index.year.max() + 1):
        for date, _ in FrenchHolidayCalendar.easter_related_holidays(year):
            easter_holidays.append(date)
    holidays = holidays.union(pd.to_datetime(easter_holidays))
    df["is_holiday"] = df.index.isin(holidays)

    # Lockdown periods
    lockdowns = {
        "lockdown_1": ("2020-03-17", "2020-05-10"),
        "lockdown_2": ("2020-10-28", "2020-12-01"),
        # with curfew from 7 PM to 6 AM
        "lockdown_3_1": ("2021-04-03 19:00:00", "2021-05-18 06:00:00"),
        # with curfew from 9 PM to 6 AM
        "lockdown_3_2": ("2021-05-19 21:00:00", "2021-06-08 06:00:00"),
        # with curfew from 11 PM to 6 AM
        "lockdown_3_3": ("2021-06-09 23:00:00", "2021-06-29 06:00:00"),
    }
    for lockdown, (start_date, end_date) in lockdowns.items():
        mask = (df.index >= start_date) & (df.index <= end_date)
        df[lockdown] = mask

    return df


train = create_features(train)
test = create_features(test)

# convert booleans to int
boolean_columns = [
    "is_weekend",
    "is_holiday",
    "lockdown_1",
    "lockdown_2",
    "lockdown_3_1",
    "lockdown_3_2",
    "lockdown_3_3",
]

for column in boolean_columns:
    train[column] = train[column].astype(int)


for column in boolean_columns:
    test[column] = test[column].astype(int)

# merge weather data with train and test
combined_train = train.merge(
    weather_data, left_index=True, right_index=True, how="left"
)

test["temp_order"] = range(test.shape[0])

combined_test = test.merge(weather_data, left_index=True, right_index=True, how="left")

combined_test.sort_values(by="temp_order", inplace=True)

combined_test.drop(columns=["temp_order"], inplace=True)

combined_train["temp_hour_interaction"] = combined_train["t"] * combined_train["hour"]
combined_train["humidity_day_interaction"] = (
    combined_train["u"] * combined_train["dayofweek"]
)

combined_test["temp_hour_interaction"] = combined_test["t"] * combined_test["hour"]


combined_test["humidity_day_interaction"] = (
    combined_test["u"] * combined_test["dayofweek"]
)

# Define features and target
features = [
    "hour_cos",
    "temp_hour_interaction",
    "humidity_day_interaction",
    "etat_sol",
    "counter_name",
    "dayofyear",
    "hour_sin",
    "t_lag168h",
    "etat_sol_lag6h",
    "t",
    "t_lag12h",
    "t_lag9h",
    "w1",
    "dayofweek",
    "n",
    "month",
    "t_lag24h",
    "lockdown_2",
    "ww",
    "u",
    "lockdown_3_3",
    "ht_neige",
    "t_lag6h",
    "dayofweek_cos",
    "lockdown_3_1",
    "lockdown_3_2",
    "td_lag12h",
    "hour",
    "rr3",
    "rr24",
    "raf10",
    "dayofweek_sin",
    "n_lag6h",
    "td_lag24h",
    "td_lag168h",
    "etat_sol_lag12h",
    "tend24_lag6h",
    "u_lag9h",
    "etat_sol_lag168h",
    "ww_lag6h",
    "u_lag24h",
    "tend24_lag9h",
    "rr12_lag12h",
    "dd",
    "cl",
    "pres",
    "u_lag12h",
    "u_lag6h",
    "etat_sol_lag9h",
    "rr12_lag168h",
    "tend24",
    "temp_humidity_interaction",
    "rr12_lag24h",
    "td",
    "cm",
]
target = ["log_bike_count"]
cat_feature = ["counter_name"]

# Prepare training & test data
X_train = combined_train[features]
y_train = combined_train[target]
X_test = combined_test[features]

# Train Model

best_params = {
    "n_estimators": 633,
    "max_depth": 11,
    "min_child_weight": 2,
    "gamma": 0.5,
    "learning_rate": 0.01745767642563374,
    "subsample": 0.6852898171340072,
    "colsample_bytree": 0.5752583768824626,
    "reg_alpha": 0.6174748033948815,
    "reg_lambda": 0.37071451261939165,
}

final_model = xgb.XGBRegressor(
    tree_method="hist", **best_params, enable_categorical=True
)
final_model.fit(X_train, y_train)

# Make predictions
predictions = final_model.predict(X_test)

# Create submissions.csv
predictions_df = pd.DataFrame({"log_bike_count": predictions})
predictions_df = predictions_df.reset_index(drop=True)
predictions_df.to_csv("submissions.csv", index=True, index_label="Id")
