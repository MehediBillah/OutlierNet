import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load the previously saved model
model = load_model("SyntheticModel.h5")

# Define a function to generate synthetic data for testing
def generate_synthetic_test_data(num_samples, num_features):
    # Generate new random data for testing
    synthetic_test_data = np.random.rand(num_samples, num_features)
    return synthetic_test_data

# Function to split the sequence (same as before)
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
    num_samples = 1000  # Number of samples for testing
    num_features = 1    # Single feature for univariate sequence
    n_steps = 4         # Number of time steps
    n_seq = 2           # Number of sequences for ConvLSTM2D
    n_steps_per_seq = n_steps // n_seq

    # Generate synthetic test data
    synthetic_test_data = generate_synthetic_test_data(num_samples, num_features)

    # Prepare the data for testing
    X_test, y_test = split_sequence(synthetic_test_data.flatten(), n_steps)

    # Reshape X_test for the ConvLSTM2D input (adjust the number of samples as necessary)
    samples_test = X_test.shape[0] - (X_test.shape[0] % n_seq)
    X_test = X_test[:samples_test].reshape((samples_test, n_seq, 1, n_steps_per_seq, num_features))

    # Prediction with the trained model on new synthetic data
    yhat_test = model.predict(X_test, verbose=0)

    # Plot the predictions against the actual values
    plt.plot(range(len(yhat_test)), yhat_test, color='red', label='Predicted')
    plt.plot(range(samples_test), y_test[:samples_test], color='blue', label='Actual')
    plt.legend()
    plt.show()

    # Output the performance metric
    print("Mean Absolute Error on Test Data:", mean_absolute_error(y_test[:samples_test], yhat_test))
