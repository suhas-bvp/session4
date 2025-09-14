This project implements a basic Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch within 25k parameters.

- `mnist_cnn.py`: Contains the CNN model definition, data loading, training, and evaluation logic.
- `data/`: This directory will store the downloaded MNIST dataset.

## Model Architecture

<img width="1265" height="494" alt="image" src="https://github.com/user-attachments/assets/c1d98099-7afa-4b8a-998c-11f26f7d84bb" />

The model consists of:
- **Input Layer**: Expects a single-channel (grayscale) image of size 28x28.
- **Convolutional Block 1**:
    - `nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)`
    - `nn.ReLU()` activation
    - `nn.MaxPool2d(kernel_size=2, stride=2)` (output size: 14x14 with 8 channels)
- **Convolutional Block 2**:
    - `nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)`
    - `nn.ReLU()` activation
    - `nn.MaxPool2d(kernel_size=2, stride=2)` (output size: 7x7 with 16 channels)
- **Fully Connected Layers**:
    - `nn.Linear(16 * 7 * 7, 28)`
    - `nn.ReLU()` activation
    - `nn.Dropout(0.5)` for regularization
    - `nn.Linear(28, 10)` (output for 10 classes: digits 0-9)
      
## Model Parameters
| Layer | Equation | Parameters |
| :--- | :--- | :--- |
| `Conv2d-1` | `(1×3×3+1)×8` | 80 |
| `Conv2d-2` | `(8×3×3+1)×16` | 1,160 |
| `FC-1` | `(16×7×7+1)×28` | 21980 |
| `FC-2` | `(28+1)×10` | 290 |
| **Total** | | **23518** |


**parameters calculations**

Convolutional layers: (input channels×kernel height×kernel width + 1) × output channels 

Fully connected layers: (input features + 1) ) × output features

-> The +1 accounts for the bias term.

## Model Output

<img width="855" height="471" alt="image" src="https://github.com/user-attachments/assets/aee77d31-2c6e-4654-a3b9-b1a8a9a03577" />


## Configuration
The following hyperparameters can be adjusted in `mnist_cnn.py`:
- `BATCH_SIZE`: Number of samples per batch during training (default: 64).
- `LEARNING_RATE`: Learning rate for the Adam optimizer (default: 0.001) --> as EPOCHS is 1, no use in this program
- `EPOCHS`: Number of training epochs (default: 1).

### Prerequisites
- Python 3.12
- `pip` (Python package installer)
-  pip install torch torchvision

### Running the Project
python mnist_cnn.py

The script do:
1. Download the MNIST dataset to the `./data` directory (if not already present).
2. Initialize and print the number of trainable parameters in the model.
3. Train the model for a specified number of epochs (default is 1).
4. Evaluate the trained model on the test set and print the final accuracy.





