import joblib
import pandas as pd
from custom_functions import _encode_dates

loaded_pipe = joblib.load("SGD_init_pipeline.joblib")
test_x = pd.read_parquet("data/final_test.parquet")


predictions = loaded_pipe.predict(test_x)
predictions_df = pd.DataFrame({"log_bike_count": predictions})
predictions_df.to_csv("predictions0.csv", index=True, index_label="Id")
