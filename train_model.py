"""Train and persist the solar-power LSTM-CNN model for the Streamlit app."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

WINDOW = 365
CSV_PATH = "PV_Elec_Gas3.csv"
MODEL_PATH = Path("models/solar_forecast.keras")


def load_data():
    df = pd.read_csv(
        CSV_PATH,
        header=None,
        skiprows=1,
        names=["date", "cum_power", "Elec_kW", "Gas_mxm"],
        sep=",",
        usecols=[0, 1, 2, 3],
        parse_dates={"dt": ["date"]},
        index_col="dt",
        dayfirst=True,
    )
    df["day_power"] = df["cum_power"] - df["cum_power"].shift(1, freq="D")
    return df.dropna()[["day_power"]]


def split_windows(series, window_in, window_out):
    X, y = [], []
    for step in range(len(series) - window_in + 1):
        if step + window_in + window_out > len(series):
            break
        X.append(series[step:step + window_in])
        y.append(series[step + window_in:step + window_in + window_out])
    return np.array(X), np.array(y)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW, 1)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation="relu"),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(WINDOW),
    ])
    model.compile(optimizer="adam", loss="mae")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    train = load_data()[:"2018-10-28"]
    X, y = split_windows(train.day_power.to_numpy(), WINDOW, WINDOW)
    X = X.reshape((X.shape[0], WINDOW, 1))

    print(f"Training on {X.shape[0]} samples for {args.epochs} epochs...")
    model = build_model()
    model.fit(X, y, epochs=args.epochs, verbose=2)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
