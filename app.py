from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)


@app.route('/test', methods=['GET'])
def test():
    return "Testing Two Wheeler Backend Server!"


@app.route('/behavior', methods=['POST'])
def predict_behavior():
    # Check if a CSV file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is of CSV format
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'})

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file)

    # Return the predictions
    return jsonify({'predictions': "behavior"})


@app.route('/internal', methods=['POST'])
def predict_internal():
    # Check if a CSV file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is of CSV format
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'})

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file)

    # Return the predictions
    return jsonify({'predictions': "internal"})


if __name__ == '__main__':
    app.run(debug=True)
