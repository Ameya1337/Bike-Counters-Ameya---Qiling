features = [
    "counter_name",
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "hour_sin",
    "hour_cos",
    "is_holiday",
    "lockdown_2",
    "lockdown_3_1",
    "lockdown_3_2",
    "lockdown_3_3",
    "t",
    "rr1",
    "rr3",
    "rr6",
    "ff",
    "raf10",
    "u",
    "n",
    "cm",
]
target = ["log_bike_count"]
cat_feature = ["counter_name"]


X_train = combined_train[features]
y_train = combined_train[target]
X_test = combined_test[features]
y_test = combined_test[target]

reg = xgb.XGBRegressor(tree_method="hist", n_estimators=1000, enable_categorical=True)

reg.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=10,
    verbose=10,
)


y_hat_xgb = reg.predict(X_train)
rmse_xgb_train = mean_squared_error(y_train, y_hat_xgb, squared=False)
print("XGB Train:", rmse_xgb_train)

y_pred_xgb = reg.predict(X_test)
rmse_xgb_test = mean_squared_error(y_test, y_pred_xgb, squared=False)
print("XGB Test:", rmse_xgb_test)

fi = pd.DataFrame(
    data=reg.feature_importances_, index=reg.feature_names_in_, columns=["Importance"]
)

fi.sort_values(by="Importance").plot(kind="barh", title="Feature Importances XGBoost")

fi.sort_values(by="Importance", ascending=False)


cat_reg = cat.CatBoostRegressor(n_estimators=1000, cat_features=cat_feature)
cat_reg.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=10,
    verbose=10,
)
y_hat_cat = cat_reg.predict(X_train)
rmse_cat_train = mean_squared_error(y_train, y_hat_cat, squared=False)
print("CAT Train:", rmse_cat_train)

y_test_cat = cat_reg.predict(X_test)
rmse_cat_test = mean_squared_error(y_test, y_test_cat, squared=False)
print("CAT Test:", rmse_cat_test)

fi = pd.DataFrame(
    data=cat_reg.feature_importances_,
    index=features,
    columns=["Importance"],
)

fi.sort_values(by="Importance").plot(kind="barh", title="Feature Importances CATBoost")

fi.sort_values(by="Importance", ascending=False)


#Train is from 1st Sep 2020 to 9 Aug 2021
#Test is from 10 Aug 2021 to 9 Sep 2021
#Final Test is from 10 Sep 2021 to 18 October 2021

No weather:
1. 
XGB Train: 0.39934484816890625
XGB Test: 0.4288531805244451

2. 
XGB Train: 0.3850702806455573
XGB Test: 0.4218602311746617
CAT Train: 0.42416886423483796
CAT Test: 0.41768004289565547


weather:
1. 
XGB Train: 0.39934484816890625
XGB Test: 0.4288531805244451
CAT Train: 0.45502551308928385
CAT Test: 0.4427993967026589

2.
XGB Train: 0.3879546941579545
XGB Test: 0.4329167584282352
CAT Train: 0.4664098208471778
CAT Test: 0.47189471662101234


3.
XGB Train: 0.38809603490016187
XGB Test: 0.4316781868576021
CAT Train: 0.41169346543866797
CAT Test: 0.43294167313507714


updated weather:
1.
XGB Train: 0.40353772924335907
XGB Test: 0.4513233863066848
CAT Train: 0.45551638095701197
CAT Test: 0.468459811908399

2.
XGB Train: 0.3799935525180389
XGB Test: 0.43704548725301623
CAT Train: 0.45543679117800295
CAT Test: 0.47108370449364
Tuned XGBoost Test: 0.4222693913654174


4. (with lags)
Final Model  Train RMSE: 0.35800959900955176
Final Model Test RMSE: 0.40907155034073844

after rfecv and bootstrapping
XGBoost Tuned

features = [
    "counter_name",
    "hour",
    "dayofweek",
    "month",
    "quarter",
    "dayofyear",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "lockdown_2",
    "lockdown_3_1",
    "lockdown_3_2",
    "lockdown_3_3",
    "t",
    "u",
    "ww",
    "n",
    "etat_sol",
    "rr12",
    "temp_hour_interaction",
    "humidity_day_interaction",
    "t_lag6h",
    "t_lag9h",
    "t_lag24h",
    "td_lag24h",
    "u_lag24h",
    "ww_lag24h",
    "n_lag24h",
    "etat_sol_lag24h",
    "rr12_lag24h",
]
target = ["log_bike_count"]
cat_feature = ["counter_name"]

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
final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=10,
)



y_pred_train = final_model.predict(X_train)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
print("Final Model  Train RMSE:", rmse_train)

y_pred_test = final_model.predict(X_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
print("Final Model Test RMSE:", rmse_test)

fi = pd.DataFrame(
    data=final_model.feature_importances_, index=final_model.feature_names_in_, columns=["Importance"]
)

fi.sort_values(by="Importance").plot(kind="barh", title="Feature Importances XGBoost")

fi.sort_values(by="Importance", ascending=False)

X_train = combined_train[features]
y_train = combined_train[target]
X_test = combined_test[features]
y_test = combined_test[target]
X_pred = combined_prediction[features]