import numpy as np
from keras.models import Sequential
from keras.layers import ConvLSTM2D, Dense, Flatten
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Define a function to generate synthetic data
def generate_synthetic_data(num_samples, num_features):
    # Generate random data as a substitute for actual CSV data
    synthetic_data = np.random.rand(num_samples, num_features)
    return synthetic_data

# Define a function to split the data sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Main script execution
if __name__ == '__main__':
    num_samples = 1000  # Total number of samples
    num_features = 1    # Single feature for univariate sequence
    n_steps = 4         # Number of time steps for the LSTM

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(num_samples, num_features)

    # Prepare the data for training
    X, y = split_sequence(synthetic_data.flatten(), n_steps)

    # Reshape parameters for ConvLSTM2D
    n_seq = 2
    n_steps_per_seq = n_steps // n_seq
    n_features = 1  # Because we have a univariate series
    n_filters = 64
    kernel_size = (1, 2)
    epochs = 100

    # Make sure the total elements match for reshaping
    samples = X.shape[0] - (X.shape[0] % n_seq)

    # Reshape X for the ConvLSTM2D input
    X = X[:samples].reshape((samples, n_seq, 1, n_steps_per_seq, n_features))

    # Initialize the Sequential model
    model = Sequential()
    model.add(ConvLSTM2D(filters=n_filters, kernel_size=kernel_size, activation='relu',
                         input_shape=(n_seq, 1, n_steps_per_seq, n_features)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Fit the model with synthetic data (adjust y accordingly)
    model.fit(X, y[:samples], epochs=epochs, verbose=0)
    model.save("SyntheticModel.h5")

    # Prediction with the trained model on synthetic data
    yhat = model.predict(X, verbose=0)
    plt.plot(range(len(yhat)), yhat, color='red', label='Predictions')
    plt.plot(range(samples), y[:samples], color='blue', label='Actual')
    plt.legend()
    plt.show()

    # Output the performance metric
    print("Mean Absolute Error:", mean_absolute_error(y[:samples], yhat))
