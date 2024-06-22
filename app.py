import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models
conv_autoencoder = load_model('models/conv_autoencoder_model.h5')
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Feature extractor model
feature_extractor = tf.keras.models.Model(
    inputs=conv_autoencoder.inputs, outputs=conv_autoencoder.layers[4].output)

# Selected columns to be considered from the CSV
selected_columns = [
    'Timestamp', 'Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Roll',
    'Pitch', 'Yaw'
]


def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

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

    return data


def combined_model_predict(cnn_model, rf_model, data):
    cnn_features = feature_extractor.predict(data)
    cnn_features_flat = cnn_features.reshape(cnn_features.shape[0], -1)
    rf_probabilities = rf_model.predict_proba(cnn_features_flat)
    return rf_probabilities


@app.route('/predict', methods=['POST'])
def predict():
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
        data = process_csv(file_path)

        # Make predictions
        predictions = combined_model_predict(conv_autoencoder, rf_model, data)

        # Convert predictions to a list
        predictions_list = predictions.tolist()

        return jsonify(predictions_list)


if __name__ == '__main__':
    app.run(debug=True)
