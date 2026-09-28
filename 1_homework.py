import numpy as np
import pandas as pd

print("Q1 pandas version:", pd.__version__)

df = pd.read_csv("car_fuel_efficiency_2026.csv")

print("Q2 records:", len(df))

print("Q3 fuel types:", df["fuel_type"].nunique())


print("Q4 columns with missing values:", (df.isnull().sum() > 0).sum())

asia = df[df["origin"] == "Asia"]
print("Q5 max fuel efficiency (Asia):", asia["fuel_efficiency_mpg"].max())

median_before = df["horsepower"].median()
most_frequent = df["horsepower"].mode()[0]
median_after = df["horsepower"].fillna(most_frequent).median()
print("Q6 median before:", median_before, "after:", median_after)

X = asia[["vehicle_weight", "model_year"]].head(7).values
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print("Q7 sum of w:", w.sum())
