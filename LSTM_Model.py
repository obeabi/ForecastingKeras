import tensorflow.compat.v2 as tf
from tensorflow import keras
from tensorflow.keras import layers

print(keras.__version__)


# Part 2 - Building and Training the LSTM

def lstm_model(x,y):
    # Initialising the RNN
    regressor = keras.Sequential()

    # Add first layer and dropout to address bias
    #regressor.add(layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], no_indicators)))
    regressor.add(layers.LSTM(units=50, return_sequences=True, input_shape=(x, y)))
    regressor.add(layers.Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(layers.LSTM(units=50, return_sequences=True))
    regressor.add(layers.Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(layers.LSTM(units=50, return_sequences=True))
    regressor.add(layers.Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(layers.LSTM(units=50))
    regressor.add(layers.Dropout(0.2))

    # Adding the output layer
    regressor.add(layers.Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Print Architecture
    print(regressor.summary())

    return regressor
