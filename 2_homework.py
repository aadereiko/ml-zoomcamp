import numpy as np
import pandas as pd

columns = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
    "fuel_efficiency_mpg",
]
features = columns[:-1]

df = pd.read_csv("car_fuel_efficiency_2026.csv")[columns]

print("Q1 column with missing values:", df.columns[df.isnull().any()].tolist())

print("Q2 horsepower median:", df["horsepower"].median())


def split(df, seed):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]
    return df_train, df_val, df_test


def prepare_X(df, fill_value):
    return df[features].fillna(fill_value).values


def train_linear_regression(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T @ X + r * np.eye(X.shape[1])
    w_full = np.linalg.inv(XTX) @ X.T @ y
    return w_full[0], w_full[1:]


def rmse(y, y_pred):
    return np.sqrt(((y - y_pred) ** 2).mean())


def validation_rmse(df_train, df_val, fill_value, r=0.0):
    w0, w = train_linear_regression(
        prepare_X(df_train, fill_value), df_train["fuel_efficiency_mpg"].values, r
    )
    y_pred = w0 + prepare_X(df_val, fill_value) @ w
    return rmse(df_val["fuel_efficiency_mpg"].values, y_pred)


df_train, df_val, df_test = split(df, 42)

train_mean = df_train["horsepower"].mean()
print("Q3 RMSE with 0:", round(validation_rmse(df_train, df_val, 0), 3))
print("Q3 RMSE with mean:", round(validation_rmse(df_train, df_val, train_mean), 3))

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    print(f"Q4 r={r} RMSE:", round(validation_rmse(df_train, df_val, 0, r), 4))

scores = [validation_rmse(*split(df, seed)[:2], 0) for seed in range(10)]
print("Q5 std of RMSE:", round(np.std(scores), 3))

df_train, df_val, df_test = split(df, 9)
df_full_train = pd.concat([df_train, df_val])
print("Q6 test RMSE:", validation_rmse(df_full_train, df_test, 0, r=0.001))
