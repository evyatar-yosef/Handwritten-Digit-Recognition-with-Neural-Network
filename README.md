# Handwritten Digit Recognition with Neural Network

This project implements a simple neural network to recognize handwritten digits using the MNIST dataset. The neural network is trained on the `train.csv` dataset and tested on the `test.csv` dataset. The network consists of an input layer, one hidden layer, and an output layer. The goal is to predict the correct digit for each input based on the network's learned weights.

## Methods Used in the Neural Network

### 1. Neural Network Architecture
- **Input Layer**: The input layer contains 784 nodes (one for each pixel of the 28x28 image). Each node receives a scaled version of pixel values between 0 and 1.
- **Hidden Layer**: The hidden layer contains 20 neurons. A ReLU (Rectified Linear Unit) activation function is used in this layer, meaning that each neuron output is the maximum of zero and the input, which helps with non-linearity and reduces the risk of vanishing gradients.
- **Output Layer**: The output layer has 10 neurons, one for each digit (0–9). Softmax activation is used to transform the network’s output into a probability distribution over the 10 classes (digits). The neuron with the highest probability represents the predicted digit.

### 2. Forward Propagation
Forward propagation refers to the process of passing input data through the network:
- **Input to Hidden Layer**: The input is multiplied by the weights (`w_i_h`) and added to the bias (`b_i_h`). This result is passed through the ReLU activation function.
- **Hidden to Output Layer**: The activations from the hidden layer are multiplied by the weights (`w_h_o`) and added to the output bias (`b_h_o`). The result is passed through the softmax activation to generate a probability distribution over the 10 possible output digits.

### 3. Activation Functions
- **ReLU (Rectified Linear Unit)**: Used in the hidden layer to introduce non-linearity. It returns the maximum of 0 and the input value, making it suitable for learning complex patterns while helping to prevent vanishing gradients.
- **Softmax**: Used in the output layer to convert the raw output into probabilities. Softmax ensures that the outputs sum to 1, which makes them interpretable as probabilities for classification.

### 4. Cost Function (Log Loss)
The cost function used in this network is **log loss** (or binary cross-entropy for each output):
\[
\text{log\_loss} = -t \cdot \log(a) - (1 - t) \cdot \log(1 - a)
\]
where `a` is the predicted probability and `t` is the target (the actual class). This function penalizes the network more when the predicted probability deviates from the true class.

### 5. Backpropagation and Gradient Descent
Backpropagation is used to update the weights and biases by calculating the gradients of the cost function with respect to each weight. These gradients are then used to adjust the weights using **gradient descent**:
- **Gradients** are computed for the output layer and the hidden layer using the chain rule.
- **Weight Update**: The weights are updated by subtracting the gradient of the cost function multiplied by the learning rate (`learning_rate`).
   
This iterative process continues through the entire dataset for each epoch.

### 6. Training Process
- **Batch Gradient Descent**: The training data is processed in batches, where the network's weights and biases are updated after each batch (size: `batch_size`).
- **Epochs**: The training process runs for a predefined number of epochs (iterations over the entire training dataset). In each epoch, the network makes predictions, computes the loss, and updates the weights.

### 7. Prediction and Evaluation
After training, the network is evaluated on the `test.csv` dataset. The output of the network is compared to the true labels, and the accuracy is calculated. For each incorrect prediction, the predicted digit and the actual digit are displayed along with a visual representation of the input image.

## Hyperparameters

The following hyperparameters are used to control the training process and the model architecture:

### 1. **Epochs**
- **Description**: The number of times the entire training dataset is passed through the network during training.
- **Value**: `3`
- **Explanation**: This means that the network will be trained for 3 complete passes over the training dataset. You can adjust the number of epochs based on your dataset size and training requirements. More epochs typically lead to better accuracy but might cause overfitting if set too high.

### 2. **Batch Size**
- **Description**: The number of samples processed before the model's internal parameters are updated.
- **Value**: `400`
- **Explanation**: The batch size determines how many examples are used in one training step. A smaller batch size might lead to more frequent updates but with higher variance in the updates, while a larger batch size will provide more stable updates but at the cost of slower processing.

### 3. **Learning Rate**
- **Description**: The step size used when updating the weights during training.
- **Value**: `0.4`
- **Explanation**: The learning rate controls how big a step the optimizer takes while moving towards the minimum of the loss function. A higher learning rate may speed up training but might overshoot the minimum, while a lower learning rate ensures a more controlled approach but may slow down training. Typically, a value between 0.001 and 0.1 works well.

### 4. **Input Count**
- **Description**: The number of input features (pixels in the 28x28 image).
- **Value**: `784`
- **Explanation**: Each MNIST image is 28x28 pixels, so there are 784 input nodes in the neural network, one for each pixel.

### 5. **Hidden Layer Count**
- **Description**: The number of neurons in the hidden layer.
- **Value**: `20`
- **Explanation**: The hidden layer captures complex patterns in the data. Increasing the number of neurons might help the network capture more patterns, but it could also lead to overfitting. The number of hidden neurons is typically chosen through experimentation.

### 6. **Output Count**
- **Description**: The number of neurons in the output layer, corresponding to the 10 possible digits (0–9).
- **Value**: `10`
- **Explanation**: This is fixed since the task is digit classification, and there are 10 possible classes (digits 0–9).

## How to Use

1. Clone the repository and install any required dependencies.
2. Prepare the dataset (`train.csv` and `test.csv`) with pixel values and labels.
3. Run the script to train the neural network and evaluate its performance on the test dataset.

```bash
python handwritten.py
