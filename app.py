import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request, jsonify
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import date

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models
try:
    conv_autoencoder = load_model('models/conv_autoencoder_model.h5')
    with open('models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Feature extractor model
feature_extractor = tf.keras.models.Model(
    inputs=conv_autoencoder.inputs, outputs=conv_autoencoder.layers[4].output)

# Selected columns to be considered from the CSV
selected_columns = [
    'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Roll',
    'Pitch', 'Yaw'
]

# Fetch Supabase URL and key from environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Supabase client initialization
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def process_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Create a copy of the DataFrame to retain all columns
        original_df = df.copy()

        # Calculate total time using the first and last Timestamp value
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        total_time = (df['Timestamp'].iloc[-1] -
                      df['Timestamp'].iloc[0]).total_seconds() / 3600  # in hours

        # Calculate the Average Speed
        average_speed = df['CurrentSpeed'].mean()

        # Calculate the Distance Covered
        total_distance = average_speed * total_time

        # Calculate the Average Acceleration
        accelerations = np.sqrt(
            df['Accel_X']**2 + df['Accel_Y']**2 + df['Accel_Z']**2)
        average_acceleration = accelerations.mean()

        # Ensure only selected columns are used
        df = df[selected_columns]

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


def insert_into_supabase(rider_name, rider_email, total_distance, average_speed, average_acceleration, classification, driving_time):
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
            "Rider Score": 95,
            "Email": rider_email,
            "Driving Time": driving_time
        }

        response = supabase.table('Riders').insert(data).execute()
        return response
    except Exception as e:
        print(f"Error inserting into Supabase: {e}")
        raise


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

            # Assuming rf_probabilities gives a 2D array where the second column is the probability of the positive class
            positive_class_probabilities = predictions[:, 1]

            # Add the predictions to the original DataFrame
            original_df['target'] = positive_class_probabilities

            # Determine classification based on a threshold (e.g., 0.5)
            classification = 'Normal' if positive_class_probabilities.mean() > 0.5 else 'Abnormal'

            # Insert calculated data into Supabase
            supabase_response = insert_into_supabase(
                rider_name, rider_email, total_distance, average_speed, average_acceleration, classification, driving_time)

            return jsonify({"message": "Data processed and inserted into Supabase successfully.", "supabase_response": supabase_response.data})
    except Exception as e:
        print(f"Fatal error: {e}")
        return jsonify({"error": "Fatal error"}), 500


if __name__ == '__main__':
    app.run(debug=True)
