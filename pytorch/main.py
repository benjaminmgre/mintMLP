import argparse
import os
import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import slp


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def main():
    # Define constant hyperparameters
    LR = 0.1
    n_epoch = 100
    NUM_CLASSES = 4
    NUM_FEATURES = 2
    RANDOM_SEED = 42

    #Make the dataset
    X, y = make_blobs(n_samples=1000,
                      n_features=NUM_FEATURES,
                      centers=NUM_CLASSES,
                      cluster_std=1.5,
                      random_state=RANDOM_SEED)
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.LongTensor)

    print(X[:5], y[:5])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Plot the data
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    parser = argparse.ArgumentParser(description='Define a Single Layer Perceptron (SLP)')
    parser.add_argument('n_input', metavar='I', type=int, nargs=1, help='The number of input nodes.')
    parser.add_argument('n_hidden', metavar='H', type=int, nargs=1, help='The number of hidden nodes.')
    parser.add_argument('n_output', metavar='O', type=int, nargs=1, help='The number of output nodes.')

    args = parser.parse_args()

    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    model = slp.SingleLayerPerceptron(args.n_input[0], args.n_hidden[0], args.n_output[0]).to(device)

    # Choice of loss and optimizer for multi class classification
    #https://www.learnpytorch.io/02_pytorch_classification/
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    for epoch in range(0, n_epoch):
        model.train()

        # 1. Forward
        y_logits = model(X_train)
        # Find probability of each class with softmax and then find the max
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        # 2. Loss
        loss = loss_function(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # 3. Zero the gradient
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Update model parameters
        optimizer.step()

        # Test
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

            test_loss = loss_function(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%')

if __name__ == '__main__':
    main()
