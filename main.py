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

    num_epochs = 50
    batch_size = 60
    test_frac = 0.3
    starting_lr = 10 ** -1
    ending_lr = 10 ** -5
    
    # Model and optimizer
    model = MLP()
    sgd = nn.SGD(model.parameters(), lr=starting_lr)

    # Extract input and output data
    data1 = loadmat("Data/DataSet1_MP1")
    input_data1 = data1.get("DataSet1")
    output_data1 = data1.get("DataSet1_targets")
    
    # Convert data to lists from numpy arrays
    input_data1, output_data1 = input_data1.tolist(), output_data1.tolist()
    
    # Shuffle Data
    combined = list(zip(input_data1, output_data1))
    random.shuffle(combined)
    input_data1, output_data1 = zip(*combined)
    input_data1, output_data1 = list(input_data1), list(output_data1)
    
    # Normalize Data
    input_data1_X, input_data_Y = zip(*input_data1)
    input_data1_X, input_data_Y = np.array(input_data1_X), np.array(input_data_Y)
    
    plt.figure()
    plt.title("Data")
    plt.scatter(input_data1_X, input_data_Y, c=output_data1)
    plt.show()
    
    input_data1_X = (input_data1_X - mean(input_data1_X)) / stdev(input_data1_X)
    input_data_Y = (input_data_Y - mean(input_data_Y)) / stdev(input_data_Y)
    input_data1 = list(zip(input_data1_X, input_data_Y))
    input_data1 = list(map(list, input_data1))
    
    plt.figure()
    plt.title("Normalized Data")
    plt.scatter(input_data1_X, input_data_Y, c=output_data1)
    plt.show()
    
    # Get train/test split
    split = int((1 - test_frac) * len(input_data1))
    train_input_data1 = input_data1[:split]
    test_input_data1 = input_data1[split:]
    train_output_data1 = output_data1[:split]
    test_output_data1 = output_data1[split:]
    
    # Get the rate of change of the Learning Rate
    num_updates = ((len(train_input_data1) / batch_size) * num_epochs)
    lr_rate = (starting_lr - ending_lr) / num_updates
    
    # Instantiate array of errors
    train_errors = np.zeros(num_epochs)
    train_accuracies = np.zeros(num_epochs)
    test_errors = np.zeros(num_epochs)
    test_accuracies = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        for i in range(len(train_input_data1)):
            
            model.zerograd()
            
            model_output = model(Tensor([train_input_data1[i]]))
        
            loss = nn.mse(model_output, Tensor([train_output_data1[i]]))
            train_errors[epoch] += loss.data
            out = model_output.data[0][0].data/abs(model_output.data[0][0].data)
            if(int(out) == int(train_output_data1[i][0])):
                train_accuracies[epoch] += 1

            # Backprop
            loss.backward()

            if((i + 1) % batch_size == 0):
                
                # After we have done the backward pass, we can step the model weights
                sgd.step()
                model.zerograd()
                
                # Update Learning Rate
                sgd.lr -= lr_rate
                
        train_errors[epoch] /= len(train_input_data1)
        train_accuracies[epoch] /= len(train_input_data1)
                
        for i in range(len(test_input_data1)):
            
            model_output = model(Tensor([test_input_data1[i]]))
        
            test_errors[epoch] += nn.mse(model_output, Tensor([test_output_data1[i]])).data
            out = model_output.data[0][0].data/abs(model_output.data[0][0].data)
            if(int(out) == int(test_output_data1[i][0])):
                test_accuracies[epoch] += 1
                
        test_errors[epoch] /= len(test_input_data1)
        test_accuracies[epoch] /= len(test_input_data1)
        
    plt.figure()
    plt.title("Error vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.plot(train_errors)
    plt.plot(test_errors)
    plt.legend(["Training", "Validation"])
    plt.show()
    
    plt.figure()
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(train_accuracies)
    plt.plot(test_accuracies)
    plt.legend(["Training", "Validation"])
    plt.show()