from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.initializers import Orthogonal
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Initialize Flask app
app = Flask(__name__)

# Define the sensors
sensors = ['Battery Voltage Sensor', 'Battery Current Sensor', 'Battery Temperature Sensor',
           'Motor RPM Sensor', 'Motor Temperature Sensor', 'Brake Pressure Sensor']


def prepare_data(df, column, scaler):
    data = df[['Timestamp', column]].set_index('Timestamp')
    scaled_data = scaler.transform(data)
    X = scaled_data[:-1]  # Input sequence
    y = scaled_data[1:]   # Target sequence (predict the next value)
    return X, y


def build_exponential_smoothing(y_train, y_test):
    model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12)
    model_fit = model.fit()
    predictions = model_fit.forecast(len(y_test))
    return predictions


def ensemble_predictions(lstm_preds, gru_preds, es_preds, y_test):
    estimators = [
        ('lstm', LinearRegression()),
        ('gru', LinearRegression()),
        ('es', LinearRegression())
    ]
    stacker = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression())
    X_ensemble = np.column_stack((lstm_preds, gru_preds, es_preds))
    stacker.fit(X_ensemble, y_test)
    ensemble_preds = stacker.predict(X_ensemble)
    return ensemble_preds


def calculate_mse(y_true, y_pred, scaler):
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    return mean_squared_error(y_true_inv, y_pred_inv)


@app.route('/test', methods=['GET'])
def test():
    return "Testing Two Wheeler Backend Server!"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'})

    df = pd.read_csv(file)

    sensor = request.form.get('sensor')
    if sensor not in sensors:
        return jsonify({'error': f'Sensor {sensor} is not recognized'})

    scaler = joblib.load(f'{sensor}_scaler.pkl')
    X, y = prepare_data(df, sensor, scaler)

    X = X.reshape(-1, 1, 1)
    y = y.reshape(-1, 1)

    lstm_model = load_model(f'{sensor}_lstm_model.h5', custom_objects={
                            'Orthogonal': Orthogonal})
    gru_model = load_model(f'{sensor}_gru_model.h5', custom_objects={
                           'Orthogonal': Orthogonal})

    lstm_preds = lstm_model.predict(X)
    gru_preds = gru_model.predict(X)
    es_preds = build_exponential_smoothing(y[:-len(X)], y[-len(X):])

    ensemble_preds = ensemble_predictions(
        lstm_preds, gru_preds, es_preds, y[-len(X):])

    lstm_mse = calculate_mse(y[-len(X):], lstm_preds, scaler)
    gru_mse = calculate_mse(y[-len(X):], gru_preds, scaler)
    es_mse = calculate_mse(y[-len(X):], es_preds, scaler)
    ensemble_mse = calculate_mse(y[-len(X):], ensemble_preds, scaler)

    mse_results = {
        'LSTM': lstm_mse,
        'GRU': gru_mse,
        'Exponential Smoothing': es_mse,
        'Ensemble': ensemble_mse
    }

    predictions = {
        'LSTM': lstm_preds.flatten().tolist(),
        'GRU': gru_preds.flatten().tolist(),
        'Exponential Smoothing': es_preds.flatten().tolist(),
        'Ensemble': ensemble_preds.flatten().tolist()
    }

    return jsonify({'mse_results': mse_results, 'predictions': predictions})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
