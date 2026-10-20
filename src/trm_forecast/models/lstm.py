"""Optional LSTM adapter kept behind a small functional API."""
import numpy as np


def forecast_lstm(y, horizon: int = 5, lookback: int = 20, epochs: int = 10):
    """Fit a compact LSTM on a univariate series; requires TensorFlow."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("install the 'models' extra to use LSTM") from exc
    values = np.asarray(y, dtype="float32")
    if len(values) <= lookback:
        raise ValueError("series must contain more observations than lookback")
    scale = max(float(np.std(values)), 1e-8)
    mean = float(np.mean(values))
    normalized = (values - mean) / scale
    x = np.array([normalized[i - lookback:i] for i in range(lookback, len(values))])[:, :, None]
    target = normalized[lookback:]
    model = keras.Sequential([keras.layers.Input((lookback, 1)), keras.layers.LSTM(16), keras.layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x, target, epochs=epochs, verbose=0)
    window = normalized[-lookback:].tolist()
    predictions = []
    for _ in range(horizon):
        prediction = float(model.predict(np.array(window[-lookback:])[None, :, None], verbose=0)[0, 0])
        predictions.append(prediction)
        window.append(prediction)
    return np.asarray(predictions) * scale + mean
