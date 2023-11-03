# OutlierNet

OutlierNet is an innovative machine learning system specifically designed for the real-time detection of defects in injection molding machines. It leverages advanced Convolutional LSTM neural networks to analyze operational data, identifying anomalies and ensuring the highest standards of manufacturing quality and efficiency.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running OutlierNet, you will need to install the necessary Python libraries. It is recommended to use a virtual environment:

```bash
python -m venv outliernet-env
source outliernet-env/bin/activate  # On Windows use `outliernet-env\Scripts\activate`
```

### Installation

Clone the OutlierNet repository and install the required dependencies.

```bash
git clone https://github.com/MehediBillah/OutlierNet.git
cd OutlierNet
pip install -r requirements.txt
```

## Usage

The system consists of two primary scripts:

- `outliernet_training.py` - This script manages the training process of the OutlierNet model on synthetic data.
- `outliernet_testing.py` - This script evaluates the performance of a trained OutlierNet model against a synthetic test dataset.

### Training

To initiate the training process, execute the following command:

```bash
python outliernet_training.py
```

The script will create synthetic training data, train the OutlierNet model, and save the trained model to the filesystem.

### Testing

To test the trained OutlierNet model's performance, use the following command:

```bash
python outliernet_testing.py
```

The test script will load the saved OutlierNet model and perform evaluation using synthetic test data.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Acknowledgments

A shoutout to the developers of Keras for creating an incredibly powerful platform that made this project possible.

For any inquiries or contributions, please feel free to open an issue or a pull request.
