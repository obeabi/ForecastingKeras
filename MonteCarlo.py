import numpy as np
import matplotlib.pyplot as plt


def plot_training_loss(model_trained, ticker, save_path='training_loss.png'):
    """
    Plot the training loss for the LSTM model and save the plot.

    Parameters:
    - model_trained: Trained LSTM model containing the history of training.
    - ticker: Stock ticker symbol.
    - save_path: Path to save the plot.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(model_trained.history['loss'], label='LSTM Training Loss')
    plt.title(f'LSTM Training Loss for - Ticker: {ticker}')
    plt.xlabel('Epoch Number')
    plt.ylabel('Training Loss')
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()


def monte_carlo_simulation(model_predictions, real_stock_price, ticker, num_simulations=100, noise_std=0.05,
                           save_path='monte_carlo_simulation.png'):
    """
    Perform Monte Carlo simulation for stock price forecasting and save the plot.

    Parameters:
    - model_predictions: Array of model predicted stock prices.
    - real_stock_price: Array of real stock prices.
    - ticker: Stock ticker symbol.
    - num_simulations: Number of simulations for the Monte Carlo simulation (default is 100).
    - noise_std: Standard deviation for the noise to be added (default is 0.05).
    - save_path: Path to save the plot.
    """

    # Create an array to store simulation results
    simulation_results = np.zeros((num_simulations, len(model_predictions)))

    # Perform Monte Carlo simulation
    for i in range(num_simulations):
        # Introduce random noise to the model predictions
        noise = np.random.normal(0, noise_std, len(model_predictions))
        simulated_predictions = model_predictions + noise

        # Store the simulated predictions in the results array
        simulation_results[i, :] = simulated_predictions

    # Plot the Monte Carlo simulation results
    plt.figure(figsize=(10, 6))
    for i in range(num_simulations):
        plt.plot(np.arange(len(model_predictions)), simulation_results[i, :], linestyle='-', marker='', alpha=0.1)

    # Plot the original predictions and real stock price
    plt.plot(real_stock_price, color='blue', label='Real Stock Price')
    plt.plot(np.arange(len(model_predictions)), model_predictions, label='Original Predictions', color='red',
             linewidth=2)

    plt.title(f'Monte Carlo Simulation for Stock Price Forecasting - Ticker: {ticker}')
    plt.xlabel('Days into the Future')
    plt.ylabel('Predicted Stock Price')
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()


def plot_stock_price_prediction(real_stock_price, model_predictions, ticker, save_path='stock_price_prediction.png'):
    """
    Plot the real and predicted stock prices and save the plot.

    Parameters:
    - real_stock_price: Array of real stock prices.
    - model_predictions: Array of model predicted stock prices.
    - ticker: Stock ticker symbol.
    - save_path: Path to save the plot.
    """

    # Create a new figure
    plt.figure(figsize=(12, 6))
    plt.plot(real_stock_price, color='red', label='Real Stock Price')
    plt.plot(model_predictions, color='blue', label='Predicted Stock Price')
    plt.title(f'Stock Price Prediction with VIX - Ticker: {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()

    # Save the plot
    plt.savefig(save_path)
    plt.close()

# Example usage
# Assuming real_stock_price and model_predictions are your data arrays and ticker is your stock ticker symbol
# plot_stock_price_prediction(real_stock_price, model_predictions, ticker, save_path='stock_price_prediction.png')


# Example usage
# model_predictions = np.array([...])  # Your model predictions here
# real_stock_price = np.array([...])   # Your real stock prices here
# ticker = "AAPL"                      # Example ticker
# monte_carlo_simulation(model_predictions, real_stock_price, ticker, num_simulations=100, noise_std=0.05, save_path='monte_carlo_simulation.png')
