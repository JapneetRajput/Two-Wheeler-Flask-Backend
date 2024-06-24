import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler

# Paths to the train and test CSV files
train_path = 'train.csv'
test_path = 'test.csv'

# Define the sensors
sensors = ['Battery Voltage Sensor', 'Battery Current Sensor', 'Battery Temperature Sensor',
           'Motor RPM Sensor', 'Motor Temperature Sensor', 'Brake Pressure Sensor']


# def prepare_data(train_path, test_path, column):
#     # Load the train and test data
#     train_data = pd.read_csv(train_path)
#     test_data = pd.read_csv(test_path)

#     # Convert 'Timestamp' to datetime
#     train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])
#     test_data['Timestamp'] = pd.to_datetime(test_data['Timestamp'])

#     # Prepare the train data
#     train_series = train_data[['Timestamp', column]].set_index('Timestamp')
#     train_scaler = MinMaxScaler()
#     train_scaled_data = train_scaler.fit_transform(train_series)
#     X_train = train_scaled_data[:-1]  # Input sequence
#     # Target sequence (predict the next value)
#     y_train = train_scaled_data[1:]

#     # Prepare the test data
#     test_series = test_data[['Timestamp', column]].set_index('Timestamp')
#     test_scaler = MinMaxScaler()
#     test_scaled_data = test_scaler.fit_transform(test_series)
#     X_test = test_scaled_data[:-1]  # Input sequence
#     y_test = test_scaled_data[1:]   # Target sequence (predict the next value)

#     return X_train, X_test, y_train, y_test, train_scaler, test_scaler


# def build_and_save_gru(X_train, y_train, X_test, y_test, sensor, model_path):
#     X_train = X_train.reshape(-1, 1, 1)
#     X_test = X_test.reshape(-1, 1, 1)
#     model = Sequential([
#         GRU(50, input_shape=(1, 1)),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X_train, y_train, epochs=1, batch_size=32,
#               validation_data=(X_test, y_test))

#     # Save the model
#     model.save(model_path)

#     predictions = model.predict(X_test)
#     return predictions


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


# def plot_predictions(y_test, predictions, column, model_name, scaler):
    # Ensure the arrays are 2D
    y_test = y_test.reshape(-1, 1)
    predictions = predictions.reshape(-1, 1)

    # Rescale the test data
    y_test_rescaled = scaler.inverse_transform(y_test)
    predictions_rescaled = scaler.inverse_transform(predictions)

    # Reshape the test data and predictions for plotting
    y_test_copy = y_test_rescaled.reshape(-1)
    pred_copy = predictions_rescaled.reshape(-1)
    X_axis = list(range(len(y_test)))

    # Plot the test data and predictions
    plt.figure(figsize=(10, 6))
    plt.plot(X_axis, y_test_copy, label='Test Data', marker='o', linestyle='-')
    plt.plot(X_axis, pred_copy,
             label=f'{model_name} Predictions', marker='o', linestyle='--', color='red')
    plt.title(f'Test Data and {model_name} Predictions for {column}')
    plt.xlabel('Timestamp')
    plt.ylabel(f'{column}')
    plt.legend()
    plt.grid(True)
    plt.show()


# mse_results = {}

# for sensor in sensors:
#     print(f"Processing sensor: {sensor}")
#     X_train, X_test, y_train, y_test, train_scaler, test_scaler = prepare_data(
#         train_path, test_path, sensor)

#     # Build and evaluate GRU model
#     gru_model_path = f'gru_model_{sensor}.h5'
#     gru_preds = build_and_save_gru(
#         X_train, y_train, X_test, y_test, sensor, gru_model_path)
#     gru_mse = calculate_mse(y_test, gru_preds, test_scaler)
#     plot_predictions(y_test, gru_preds, sensor, 'GRU', test_scaler)

#     # Store MSE values in dictionary
#     mse_results[sensor] = {
#         'GRU': gru_mse
#     }

# print("MSE results for each sensor model:")
# for sensor, results in mse_results.items():
#     print(f"{sensor}: {results}")

# Path to the new unseen data CSV file
new_data_path = 'two_wheeler_diag_unseen.csv'


def prepare_new_data(new_data_path, column, test_scaler):
    # Load the new data
    new_data = pd.read_csv(new_data_path)

    # Convert 'Timestamp' to datetime
    new_data['Timestamp'] = pd.to_datetime(new_data['Timestamp'])

    # Prepare the new data
    new_series = new_data[['Timestamp', column]].set_index('Timestamp')
    new_scaled_data = test_scaler.transform(new_series)
    X_new = new_scaled_data[:-1]  # Input sequence
    y_new = new_scaled_data[1:]   # Target sequence (predict the next value)

    return X_new, y_new


def test_on_new_data(sensor):
    print(f"Processing new data for sensor: {sensor}")

    # Load the scaler from the previous data preparation step
    test_series = pd.read_csv(
        test_path)[['Timestamp', sensor]].set_index('Timestamp')
    test_scaler = MinMaxScaler()
    test_scaled_data = test_scaler.fit_transform(test_series)

    # Prepare the new data
    X_new, y_new = prepare_new_data(new_data_path, sensor, test_scaler)

    # Load and evaluate GRU model
    gru_model_path = f'gru_model_{sensor}.h5'
    gru_preds = load_gru_and_predict(X_new, gru_model_path)
    gru_mse = calculate_mse(y_new, gru_preds, test_scaler)
    # plot_predictions(y_new, gru_preds, sensor, 'GRU', test_scaler)

    # Print MSE values for new data
    print(f"MSE for new data - {sensor}:")
    print(f"GRU: {gru_mse}")


# Loop through all sensors and test on new data
for sensor in sensors:
    test_on_new_data(sensor)
