from src.tensor import Tensor, Scalar
from src import nn

import numpy as np
from scipy.io import loadmat
import random
import matplotlib.pyplot as plt
from statistics import mean, stdev

class MLP(nn.Module):

    def __init__(self):

        # Layers
        self.l1 = nn.Linear(2, 20)
        self.l2 = nn.Linear(20, 1)

    # Forward Pass
    def forward(self, input):
        input.transpose()
        hidden_input = self.l1(input)
        hidden_input = hidden_input.tanh()
        output = self.l2(hidden_input)
        output = output.tanh()
        return output

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()


if __name__=="__main__":

    # Define hyperparameters (and which dataset to use)
    num_epochs = 50
    batch_size = 60
    test_frac = 0.3
    starting_lr = 10 ** -1
    ending_lr = 10 ** -5
    dataset_number = 2
    
    # Model and optimizer
    model = MLP()
    sgd = nn.SGD(model.parameters(), lr=starting_lr)

    # Extract input and output data
    data = loadmat(f"Data/DataSet{dataset_number}_MP1")
    input_data = data.get(f"DataSet{dataset_number}")
    output_data = data.get(f"DataSet{dataset_number}_targets")
    
    # Convert data to lists from numpy arrays
    input_data, output_data = input_data.tolist(), output_data.tolist()
    
    # Shuffle Data
    combined = list(zip(input_data, output_data))
    random.shuffle(combined)
    input_data, output_data = zip(*combined)
    input_data, output_data = list(input_data), list(output_data)
    
    # Separate data for plotting and normalization
    input_data_X, input_data_Y = zip(*input_data)
    input_data_X, input_data_Y = np.array(input_data_X), np.array(input_data_Y)
    
    # Plot data
    plt.figure()
    plt.title("Data")
    plt.scatter(input_data_X, input_data_Y, c=output_data)
    plt.show()
    
    # Normalize data
    input_data_X = (input_data_X - mean(input_data_X)) / stdev(input_data_X)
    input_data_Y = (input_data_Y - mean(input_data_Y)) / stdev(input_data_Y)
    input_data = list(zip(input_data_X, input_data_Y))
    input_data = list(map(list, input_data))
    
    # Plot normalized data
    plt.figure()
    plt.title("Normalized Data")
    plt.scatter(input_data_X, input_data_Y, c=output_data)
    plt.show()
    
    # Get train/test split
    split = int((1 - test_frac) * len(input_data))
    train_input_data = input_data[:split]
    test_input_data = input_data[split:]
    train_output_data = output_data[:split]
    test_output_data = output_data[split:]
    
    # Get the rate of change of the Learning Rate
    num_updates = ((len(train_input_data) / batch_size) * num_epochs)
    lr_rate = (starting_lr - ending_lr) / num_updates
    
    # Instantiate array of errors and accuracies
    train_errors = np.zeros(num_epochs)
    train_accuracies = np.zeros(num_epochs)
    test_errors = np.zeros(num_epochs)
    test_accuracies = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        # Run Training set and update weights
        for i in range(len(train_input_data)):
            
            # Zero gradient vector
            model.zerograd()
            
            # Get model output
            model_output = model(Tensor([train_input_data[i]]))
        
            # Calulate loss, error, and accuracy
            loss = nn.mse(model_output, Tensor([train_output_data[i]]))
            train_errors[epoch] += loss.data
            out = model_output.data[0][0].data/abs(model_output.data[0][0].data)
            if(int(out) == int(train_output_data[i][0])):
                train_accuracies[epoch] += 1

            # Backprop
            loss.backward()

            # Only update weights and lr at the end of a batch
            if((i + 1) % batch_size == 0):
                
                # Step the model weights and zero gradient
                sgd.step()
                model.zerograd()
                
                # Update Learning Rate
                sgd.lr -= lr_rate
        
        # Average the error and calculate accuracy
        train_errors[epoch] /= len(train_input_data)
        train_accuracies[epoch] /= len(train_input_data)
        
        # Run validation dataset
        for i in range(len(test_input_data)):
            
            model_output = model(Tensor([test_input_data[i]]))
        
            test_errors[epoch] += nn.mse(model_output, Tensor([test_output_data[i]])).data
            out = model_output.data[0][0].data/abs(model_output.data[0][0].data)
            if(int(out) == int(test_output_data[i][0])):
                test_accuracies[epoch] += 1
                
        test_errors[epoch] /= len(test_input_data)
        test_accuracies[epoch] /= len(test_input_data)
    
    # Plot error over time
    plt.figure()
    plt.title("Error vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.plot(train_errors)
    plt.plot(test_errors)
    plt.legend(["Training", "Validation"])
    plt.show()
    
    # Plot accuracy over time
    plt.figure()
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(train_accuracies)
    plt.plot(test_accuracies)
    plt.legend(["Training", "Validation"])
    plt.show()
    
    # Calculate a mesh grid of values for plotting the decision boundary
    h = .02
    x_min, x_max = input_data_X.min() - 1, input_data_X.max() + 1
    y_min, y_max = input_data_Y.min() - 1, input_data_Y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    xx_r, yy_r = xx.reshape(xx.shape[0] * xx.shape[1]), yy.reshape(yy.shape[0] * yy.shape[1])
    xx_r, yy_r = list(xx_r), list(yy_r)
    grid_input = list(zip(xx_r, yy_r))
    grid_input = list(map(list, grid_input))
    grid_output = grid_input
    for i in range(len(grid_input)):
        model_output = model(Tensor([grid_input[i]]))
        grid_output[i] = int(model_output.data[0][0].data/abs(model_output.data[0][0].data))
    
    # Plot decision boundary
    plt.figure()
    plt.contourf(xx, yy, np.array(grid_output).reshape(xx.shape), cmap=plt.cm.Paired, alpha=0.8)
    plt.title("Decision Boundary")
    plt.scatter(input_data_X, input_data_Y, c=output_data)
    plt.show()
