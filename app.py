import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request, jsonify
from supabase import create_client, Client

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

# Supabase client initialization
SUPABASE_URL = 'https://dzvxbisnqbounjiovbng.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR6dnhiaXNucWJvdW5qaW92Ym5nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTA2ODE0ODMsImV4cCI6MjAyNjI1NzQ4M30.KSfNJ34pv3nM7HUmsHtQlxnMmZXal8wnYeLlefuDgh4'
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def process_csv(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Create a copy of the DataFrame to retain all columns
        original_df = df.copy()

        # Ensure only selected columns are used
        df = df[selected_columns]

        # Assuming 'Timestamp' is the index column
        df.set_index('Timestamp', inplace=True)

        # Convert to numpy array
        data = df.values

        # Ensure data is float32
        data = data.astype('float32')

        # Reshape the data to the format expected by the model (batch_size, sequence_length, num_features)
        data = np.expand_dims(data, axis=-1)

        return data, original_df
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


def insert_into_supabase():
    try:
        data = {
            "Rider_Name": "Harsh",
            "Distance Covered": 4.76,
            "Average Speed": 20.7694,
            "Average Acceleration": 9.89604,
            "Classification": "Normal",
            "Ride No": 10,
            "Date": "2024-04-18",
            "Rider Score": 95,
            "Email": "2020.harsh.deshmukh@ves.ac.in",
            "Driving Time": 16
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

        if file:
            # Save the file
            file_path = "uploaded_file.csv"
            file.save(file_path)

            # Process the file
            data, original_df = process_csv(file_path)

            # Make predictions
            predictions = combined_model_predict(
                conv_autoencoder, rf_model, data)

            # Assuming rf_probabilities gives a 2D array where the second column is the probability of the positive class
            positive_class_probabilities = predictions[:, 1]

            # Add the predictions to the original DataFrame
            original_df['target'] = positive_class_probabilities

            # Save the updated DataFrame to a new CSV file (if needed)
            updated_file_path = "updated_file.csv"
            original_df.to_csv(updated_file_path, index=False)

            # Insert hardcoded data into Supabase
            supabase_response = insert_into_supabase()

            return jsonify({"message": "Data processed and inserted into Supabase successfully.", "supabase_response": supabase_response.data})
    except Exception as e:
        print(f"Fatal error: {e}")
        return jsonify({"error": "Fatal error"}), 500


if __name__ == '__main__':
    app.run(debug=True)
