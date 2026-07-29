import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

WINDOW = 365
CSV_PATH = "PV_Elec_Gas3.csv"
MODEL_PATH = Path("models/solar_forecast.keras")

st.set_page_config(page_title="Solar Power Forecast", layout="wide")
st.title("Solar Power Forecasting — LSTM-CNN")


@st.cache_data
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
    shifted = df.shift(periods=1, freq="D", axis=0)
    df["cum_power_shift"] = shifted["cum_power"]
    df["day_power"] = df["cum_power"] - df["cum_power_shift"]
    df = df.dropna()
    return df[["day_power"]]


def split_windows(series, window_in, window_out):
    X, y = [], []
    n_steps = len(series) - window_in + 1
    for step in range(n_steps):
        if step + window_in + window_out > len(series):
            break
        X.append(series[step:step + window_in])
        y.append(series[step + window_in:step + window_in + window_out])
    return np.array(X), np.array(y)


def build_model(window_in, window_out):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(window_in, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation="relu"),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(window_out),
    ])
    model.compile(optimizer="adam", loss="mae")
    return model


def cumulate(series, start=0):
    cum = [start]
    for v in series:
        cum.append(cum[-1] + v)
    return cum


def train_model(epochs, X, y):
    model = build_model(WINDOW, WINDOW)
    model.fit(X, y, epochs=epochs, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    return model


@st.cache_resource(show_spinner=False)
def load_saved_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


data = load_data()
train = data[:"2018-10-28"]
valid = data["2018-10-29":"2019-10-28"]

st.caption(f"Dataset: {data.index.min().date()} to {data.index.max().date()} ({len(data)} days)")

epochs = st.sidebar.slider("Training epochs", min_value=1, max_value=50, value=10)
st.sidebar.caption("Training runs on CPU and is saved locally after it finishes.")

if "model" not in st.session_state:
    saved_model = load_saved_model()
    if saved_model is not None:
        st.session_state["model"] = saved_model
        st.sidebar.success("Loaded the saved local model.")

if st.sidebar.button("Train and save model", type="primary"):
    X, y = split_windows(train.day_power.values, WINDOW, WINDOW)
    X = X.reshape((X.shape[0], WINDOW, 1))
    with st.spinner(f"Training for {epochs} epochs on {X.shape[0]} samples..."):
        model = train_model(epochs, X, y)
    st.session_state["model"] = model
    load_saved_model.clear()
    st.sidebar.success(f"Model saved to {MODEL_PATH}.")

if "model" not in st.session_state:
    st.info("Set epochs and click **Train and save model** to create the local model file.")
    st.stop()

model = st.session_state["model"]

X_input = train[-WINDOW:].day_power.values.reshape(1, WINDOW, 1)
y_hat = model.predict(X_input, verbose=0)[0]
y_true = valid.day_power.values

col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily power: predicted vs true")
    fig, ax = plt.subplots()
    ax.plot(y_hat, label="predicted")
    ax.plot(y_true, label="true")
    ax.legend()
    st.pyplot(fig)

y_true_cum = cumulate(y_true)
y_pred_cum = cumulate(y_hat)

with col2:
    st.subheader("Cumulative power: predicted vs true")
    fig, ax = plt.subplots()
    ax.plot(y_pred_cum, label="predicted")
    ax.plot(y_true_cum, label="true")
    ax.legend()
    st.pyplot(fig)

r2 = r2_score(y_true, y_hat)
mae = mean_absolute_error(y_true, y_hat)
true_final = y_true_cum[-1]
pred_final = y_pred_cum[-1]
acc = (1 - (true_final - pred_final) / true_final) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("R² score", f"{r2:.4f}")
m2.metric("MAE", f"{mae:.2f}")
m3.metric("True cum. power (1yr)", f"{true_final:.0f} kW")
m4.metric("1-year accuracy", f"{acc:.2f}%")
