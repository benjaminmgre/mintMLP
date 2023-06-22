import layer
import cost

import numpy as np


def load_data(n_samples):
    """
    Loads practice training data. Relationship: Sum of 10 numbers
    :param n_samples:
    :return: data and labels
    """
    data = []
    labels = []
    for i in range(0, n_samples):
        sample = np.random.uniform(-1, 1, (10, 1))
        sum = sample.sum()
        data.append(sample)
        labels.append(sum)
    return data, labels


class SLP2():
    """
    Single layer perceptron
    """
    def __init__(self, n_hidden: int, n_in: int, n_out: int):
        """
        Build the SLP

        :param n_hidden: The number of hidden nodes
        :param n_in: The number of input nodes
        :param n_out: The number of output nodes
        """

        # Initialize weights from std normal distribution
        self.w1 = np.random.randn(n_hidden, n_in)
        self.w2 = np.random.randn(n_out, n_hidden)
        self.b1 = np.random.randn(n_hidden, 1)
        self.b2 = np.random.randn(n_out, 1)

        # Create test data
        self.test_data, self.test_labels = load_data(200)

        # Define model hyperparameters
        self.epochs = 100
        self.learning_rate = 0.01

    # Activation functions and derivatives
    def ReLU(self, x):
        return np.maximum(x, 0)

    def der_ReLU(self, x):
        return (x > 0) * 1

    def sigmoid(self, x):
        return


    # I will use a linear activation function for the output.
    # The point of writing this out is that it can be replaced with another activation function
    def linear(self, x):
        return x

    def der_linear(self, x):
        return 1

    def forward(self, x, y):
        """
        The forward step. 
        :param x: Input data
        :param y: Data labels
        :return: 
        """
        self.x = x

        self.z1 = np.matmul(self.w1, x) + self.b1
        self.a1 = self.ReLU(self.z1)

        self.z2 = np.matmul(self.w2, self.a1) + self.b2
        self.a2 = self.linear(self.z2)

        self.error =  self.a2 - y

        return

    def backward(self):
        delta2 = self.error * self.der_linear(self.z2)
        w2_grad = np.matmul(delta2, self.a1.T)
        b2_grad = delta2 * self.der_linear(self.a1)

        delta1 = np.matmul(self.w2.T, delta2) * self.der_ReLU(self.z1)
        w1_grad = np.matmul(delta1, self.x.T)
        b1_grad = delta1

        # Update learning rates
        self.w2 -= self.learning_rate * w2_grad
        self.b2 -= self.learning_rate * b2_grad
        self.w1 -= self.learning_rate * w1_grad
        self.b1 -= self.learning_rate * b1_grad

    def train(self):
        for n in range(0, self.epochs):
            loss = 0
            train_data, train_labels = load_data(1000)
            for x in range(0, len(train_data)):
                self.forward(train_data[x], train_labels[x])
                self.backward()
                loss += np.mean(self.error ** 2) * 0.5
            print(f'Loss: {loss/len(train_data)}')


    def test(self):
        for x in range(0, len(self.test_data)):
            self.forward(self.test_data[x], self.test_labels[x])
            print(f'Input: {self.test_data[x]}, Output: {self.a2}, Predicted: {self.test_labels[x]}')

slp = SLP2(20, 10, 1)
slp.train()
slp.test()