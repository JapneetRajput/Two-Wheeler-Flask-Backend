import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
import pickle
from flask import Flask, request, jsonify, render_template
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import date
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from werkzeug.utils import secure_filename

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models
try:
    conv_autoencoder = load_model('models/Behaviour/conv_autoencoder_model.h5')
    with open('models/Behaviour/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Feature extractor model
feature_extractor = tf.keras.models.Model(
    inputs=conv_autoencoder.inputs, outputs=conv_autoencoder.layers[4].output)

columns = [
    'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Roll',
    'Pitch', 'Yaw'
]

class_labels = ['Normal', 'Left Deviation',
                'Right Deviation', 'Sudden Acceleration']

# Fetch Supabase URL and key from environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Supabase client initialization
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Directory to save uploaded files
upload_dir = 'uploads'
os.makedirs(upload_dir, exist_ok=True)


def load_xgboost_model_and_scaler():
    try:
        # Load the pre-trained XGBoost model
        with open('models/Anomaly/xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)

        # Load the scaler used during training
        with open('models/Anomaly/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        return xgb_model, scaler
    except Exception as e:
        print(f"Error loading XGBoost model and scaler: {e}")
        raise


def df_to_X_y(df, window_size=2):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        window_data = df_as_np[i:i + window_size]
        X.append(window_data)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)


def calculate_anomaly_count(data):
    try:
        xgb_model, scaler = load_xgboost_model_and_scaler()
        WINDOW_SIZE = 2
        X, y = df_to_X_y(data, WINDOW_SIZE)
        X_scaled = scaler.transform(
            X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_2d = X_scaled.reshape(X_scaled.shape[0], -1)
        test_predictions = xgb_model.predict(X_2d)

        residual_values = np.zeros((X_2d.shape[0],))

        for i in range(X_2d.shape[0]):
            vector_magnitude_test_predictions = np.linalg.norm(
                test_predictions[i, :])
            vector_magnitude_y_test = np.linalg.norm(y[i, :])
            residual_values[i] = vector_magnitude_test_predictions - \
                vector_magnitude_y_test

        threshold = 0.7
        absolute_residual_values = np.abs(residual_values)
        anomaly_count = np.sum(absolute_residual_values < threshold)

        return anomaly_count
    except Exception as e:
        print(f"Error calculating anomaly count: {e}")
        raise


def process_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Create a copy of the DataFrame to retain all columns
        original_df = df.copy()

        # Calculate total time using the first and last Timestamp value
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        total_time = (df['Timestamp'].iloc[-1] -
                      df['Timestamp'].iloc[0]).total_seconds() / 60  # in minutes

        # Calculate the Average Speed
        average_speed = df['CurrentSpeed'].mean()

        # Calculate the Distance Covered
        total_distance = average_speed * total_time

        # Calculate the Average Acceleration
        accelerations = np.sqrt(
            df['Accel_X']**2 + df['Accel_Y']**2 + df['Accel_Z']**2)
        average_acceleration = accelerations.mean()

        df = df[columns]

        # Convert to numpy array
        data = df.drop(columns=['Timestamp']).values

        # Ensure data is float32
        data = data.astype('float32')

        # Reshape the data to the format expected by the model (batch_size, sequence_length, num_features)
        data = np.expand_dims(data, axis=-1)

        return data, original_df, total_distance, average_speed, average_acceleration, total_time
    except Exception as e:
        print(f"Error processing CSV: {e}")
        raise


def combined_model_predict(cnn_model, rf_model, data):
    try:
        cnn_features = feature_extractor.predict(data)
        cnn_features_flat = cnn_features.reshape(cnn_features.shape[0], -1)
        rf_probabilities = rf_model.predict_proba(cnn_features_flat)
        return rf_probabilities
    except Exception as e:
        print(f"Error in combined model prediction: {e}")
        raise


def get_ride_count(rider_email):
    try:
        response = supabase.table('Riders').select(
            '*').eq('Email', rider_email).execute()
        ride_count = len(response.data)
        return ride_count + 1
    except Exception as e:
        print(f"Error fetching ride count: {e}")
        raise


def anomaly_count_to_rider_score(anomaly_count):
    # Define a mapping of anomaly counts to rider scores
    if anomaly_count <= 2:
        return 100
    elif anomaly_count <= 4:
        return 95
    elif anomaly_count <= 6:
        return 80
    elif anomaly_count <= 10:
        return 60
    elif anomaly_count <= 15:
        return 40
    else:
        return 30


# Define the sensors
sensors = ['Battery Voltage Sensor', 'Battery Current Sensor', 'Battery Temperature Sensor',
           'Motor RPM Sensor', 'Motor Temperature Sensor', 'Brake Pressure Sensor']

# Path to the models
model_dir = './models/Internal'
train_path = 'train.csv'
test_path = 'test.csv'


def load_gru_and_predict(X_new, model_path):
    model = load_model(model_path)
    X_new = X_new.reshape(-1, 1, 1)
    predictions = model.predict(X_new)
    return predictions


def calculate_mse(y_true, y_pred, scaler):
    # Inverse transform the predictions and y_true to the original scale
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    return mean_squared_error(y_true_inv, y_pred_inv)


def prepare_new_data(file_path, column, test_scaler):
    # Load the new data
    new_data = pd.read_csv(file_path)

    # Convert 'Timestamp' to datetime
    new_data['Timestamp'] = pd.to_datetime(new_data['Timestamp'])

    # Prepare the new data
    new_series = new_data[['Timestamp', column]].set_index('Timestamp')
    new_scaled_data = test_scaler.transform(new_series)
    X_new = new_scaled_data[:-1]  # Input sequence
    y_new = new_scaled_data[1:]   # Target sequence (predict the next value)

    return X_new, y_new


def test_on_new_data(file_path, sensor):
    print(f"Processing new data for sensor: {sensor}")

    # Load the scaler from the previous data preparation step
    test_series = pd.read_csv(
        test_path)[['Timestamp', sensor]].set_index('Timestamp')
    test_scaler = MinMaxScaler()
    test_scaled_data = test_scaler.fit_transform(test_series)

    # Prepare the new data
    X_new, y_new = prepare_new_data(file_path, sensor, test_scaler)

    # Load and evaluate GRU model
    gru_model_path = os.path.join(model_dir, f'gru_model_{sensor}.h5')
    gru_preds = load_gru_and_predict(X_new, gru_model_path)
    gru_mse = calculate_mse(y_new, gru_preds, test_scaler)

    return gru_mse


def insert_into_supabase(rider_name, rider_email, total_distance, average_speed, average_acceleration, classification, driving_time, rider_score):
    try:
        ride_no = get_ride_count(rider_email)
        data = {
            "Rider_Name": rider_name,
            "Distance Covered": total_distance,
            "Average Speed": average_speed,
            "Average Acceleration": average_acceleration,
            "Classification": classification,
            "Ride No": ride_no,
            "Date": date.today().isoformat(),
            "Rider Score": rider_score,
            "Email": rider_email,
            "Driving Time": driving_time
        }

        response = supabase.table('Riders').insert(data).execute()
        return response
    except Exception as e:
        print(f"Error inserting into Supabase: {e}")
        raise


@app.route('/test', methods=['GET'])
def test():
    return jsonify("Running")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Get rider name and email from the request form
        rider_name = request.form.get('rider_name')
        rider_email = request.form.get('rider_email')

        if not rider_name or not rider_email:
            return jsonify({"error": "Rider name and email are required"}), 400

        if file:
            # Save the file
            file_path = "uploaded_file.csv"
            file.save(file_path)

            # Process the file
            data, original_df, total_distance, average_speed, average_acceleration, driving_time = process_csv(
                file_path)

            # Make predictions
            predictions = combined_model_predict(
                conv_autoencoder, rf_model, data)

            # Count occurrences of each classification
            class_counts = Counter(np.argmax(predictions, axis=1))

            # Determine classification based on the class with the maximum count
            max_class_index = max(class_counts, key=class_counts.get)
            classification = class_labels[max_class_index]

            # Calculate Rider Score as anomaly count
            anomaly_count = calculate_anomaly_count(original_df[[
                                                    'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Roll', 'Pitch', 'Yaw']])
            rider_score = anomaly_count_to_rider_score(anomaly_count)

            # Insert calculated data into Supabase
            supabase_response = insert_into_supabase(
                rider_name, rider_email, total_distance, average_speed, average_acceleration, classification, driving_time, rider_score)

            return jsonify({"message": "Data processed and inserted into Supabase successfully.", "supabase_response": supabase_response.data})
    except Exception as e:
        print(f"Fatal error: {e}")
        return jsonify({"error": "Fatal error"}), 500


@app.route('/internal', methods=['POST'])
def internal():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        # Process the file and calculate MSE for each sensor
        results = {}
        for sensor in sensors:
            mse = test_on_new_data(file_path, sensor)
            results[sensor] = mse

        # Remove the file after processing
        os.remove(file_path)

        return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
