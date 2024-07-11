from stocks import instrument
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from MonteCarlo import monte_carlo_simulation, plot_training_loss, plot_stock_price_prediction
from LSTM_Model import lstm_model
from ExploratoryAnalysis import extensive_eda
import tensorflow.compat.v2 as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model


print(keras.__version__)

#
# # Min-Max Scaler Object
test_size = 7
day_lag = 60
unit = 50
drop = 0.2
sc = MinMaxScaler(feature_range=(0, 1))
# # Object of class stocks
ticker = 'VTI'
asset = instrument(ticker=ticker, interval='1d')
df = asset.download_price_volume()

# # Perform EDA using extensive_eda class
# # eda = extensive_eda()
# # eda.save_eda_html(data)


df = df[['Close', 'Volume']]
# Ensure the data is sorted by date
df = df.sort_index()
split_index = len(df) - test_size
# Split the DataFrame into training and test sets
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]
features = df.columns
no_indicators = len(features)
training_set = train_df[features].values
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(day_lag, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - day_lag:i, 0:])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array\
                       (X_train), np.array(y_train)
print("\nShape of X_train:", X_train.shape)
print("\nShape of y_train:", y_train.shape)
# Reshaping the dataset to add more indicators
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], no_indicators))
print("Shape of X_train after reshaping:", X_train.shape)


# # Object of class MLClassifier
model = lstm_model(X_train.shape[1], no_indicators)
print(model)

model_trained = model.fit(X_train, y_train, epochs=100, batch_size=32)
# Assuming model_trained is your trained LSTM model and ticker is your stock ticker symbol
plot_training_loss(model_trained, ticker, save_path='training_loss.png')
# # Create a new figure
# plt.figure(figsize=(12, 6))
# plt.plot(model_trained.history['loss'], label='LSTM Training Loss')
# plt.title(f'LSTM Training Loss for- Ticker: {ticker}')
# plt.xlabel('epoch number')
# plt.ylabel('Training Loss')
# plt.legend()
# plt.show(block=False)

# Prediction
real_stock_price = test_df.iloc[:, 0:1].values
dataset_total = pd.concat((train_df[features], test_df[features]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_df) - day_lag:].values
print("Shape of inputs before reshaping:", inputs.shape)
inputs = inputs.reshape(-1, no_indicators)
print("Shape of inputs after reshaping:", inputs.shape)
inputs = sc.transform(inputs)

# Account for the 3D structure required by the LSTM
X_test = []
for i in range(day_lag, day_lag + len(test_df)):
    X_test.append(inputs[i-day_lag:i, 0:2])
X_test = np.array(X_test)
print("Shape of X_test before reshaping:", X_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], no_indicators))
print("\nShape of X_test after reshaping:", X_test.shape)
predicted_stock_price = model.predict(X_test)

# Concatenate each entry with zero individually
# Replace `x` with the desired number of zeros
num_zeros = no_indicators - 1
predicted_stock_price = np.array([np.concatenate((entry, np.zeros(num_zeros))) for entry in predicted_stock_price])

# Apply inverse_transform to concatenated_results
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print("Shape of predicted_stock_price after inverse transform:", predicted_stock_price.shape)
# Extract the first element from each entry (model predictions)
model_predictions = predicted_stock_price[:, 0]
print(model_predictions[0:2])

# Assuming real_stock_price and model_predictions are your data arrays and ticker is your stock ticker symbol
plot_stock_price_prediction(real_stock_price, model_predictions, ticker, save_path='stock_price_prediction.png')
monte_carlo_simulation(model_predictions, real_stock_price, ticker, num_simulations=100, noise_std=0.05, save_path='monte_carlo_simulation.png')

# Assuming model is your trained LSTM model
#model.save('lstm_model.h5')  # Save the entire model as a .h5 file
# To load the model later
# model = load_model('lstm_model.h5')
model.save('my_model.keras')
# Load the model
model = load_model('my_model.keras')
