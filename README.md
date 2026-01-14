# Multi-Class Classification Neural Network

A from-scratch implementation of a multi-class classification neural network using NumPy. This project implements a 3-layer neural network (2 hidden layers + output layer) with backpropagation for multi-class classification tasks.

## Features

- **Custom Neural Network Implementation**: Built from scratch using only NumPy
- **Multi-Class Classification**: Supports classification with multiple classes (default: 10 classes)
- **Batch Training**: Implements mini-batch gradient descent for efficient training
- **Activation Functions**: 
  - Tanh activation for hidden layers
  - Softmax activation for output layer
- **Loss Function**: Cross-entropy loss for multi-class classification
- **Data Preprocessing**: Automatic normalization and one-hot encoding

## Architecture

The neural network consists of:
- **Input Layer**: Variable size (depends on input data)
- **Hidden Layer 1**: 20 neurons with tanh activation
- **Hidden Layer 2**: 15 neurons with tanh activation
- **Output Layer**: 10 neurons with softmax activation

## Requirements

- Python 3.6+
- NumPy
- scikit-learn

See `requirements.txt` for specific package versions.

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The code expects data files in NumPy format:
- `data/X.npy`: Feature matrix (will be normalized by dividing by 255.0)
- `data/y.npy`: Label array (will be one-hot encoded)

Make sure these files exist in the `data/` directory before running the script.

## Usage

Run the main script:

```bash
python "code/Multi-Class Classifier.py"
```

Or from the code directory:

```bash
cd code
python "Multi-Class Classifier.py"
```

## How It Works

1. **Data Loading**: Loads features (X) and labels (y) from `.npy` files
2. **Preprocessing**: 
   - Normalizes features by dividing by 255.0
   - One-hot encodes labels using scikit-learn's OneHotEncoder
3. **Train-Test Split**: Splits data into 80% training and 20% testing sets
4. **Training**: 
   - Initializes a neural network with random weights
   - Trains for 20 epochs with batch size of 32
   - Uses learning rate of 0.01
   - Prints loss and accuracy after each epoch

## Model Parameters

Default training parameters:
- **Learning Rate**: 0.01
- **Epochs**: 20
- **Batch Size**: 32
- **Test Size**: 0.2 (20% of data)
- **Random State**: 42 (for reproducibility)

You can modify these parameters in the `train()` method call or by editing the default values in the code.

## Project Structure

```
multiclass classification tasks/
├── code/
│   └── Multi-Class Classifier.py    # Main implementation
├── data/
│   ├── X.npy                        # Feature data
│   └── y.npy                        # Label data
├── confirmation files/               # Intermediate computation files
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## Key Functions

- `load_data()`: Loads and normalizes data from `.npy` files
- `dtanh()`: Derivative of tanh activation function
- `softmax_batch()`: Softmax activation for batch processing
- `cross_entropy_batch()`: Cross-entropy loss function for batches
- `NeuralNetworkMultiClassifier`: Main neural network class
  - `train()`: Trains the network using backpropagation
  - `test()`: Evaluates the network on test data

## Customization

To customize the network architecture, modify the initialization:

```python
nn = NeuralNetworkMultiClassifier(
    input_dim=X_train.shape[1],  # Input size
    hidden_dim1=20,              # First hidden layer size
    hidden_dim2=15,              # Second hidden layer size
    output_dim=10                # Number of classes
)
```

To adjust training parameters:

```python
nn.train(
    X_train, y_train, X_test, y_test,
    learning_rate=0.01,  # Adjust learning rate
    n_epochs=20,         # Adjust number of epochs
    batch_size=32        # Adjust batch size
)
```

## Notes

- The implementation uses numerical stability techniques (e.g., subtracting max in softmax, adding epsilon in log)
- Weights are initialized randomly using `np.random.randn()`
- Biases are initialized to zeros
- The code assumes input data is in the range [0, 255] and normalizes it to [0, 1]

## License

This project is provided as-is for educational purposes.
