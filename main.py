from src.tensor import Tensor, Scalar
from src import nn

from scipy.io import loadmat
import random
import matplotlib.pyplot as plt

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

    num_epochs = 30
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
    train_errors = []
    test_errors = []
    
    for epoch in range(num_epochs):
        
        train_errors.append(0)
        for i in range(len(train_input_data1)):
            
            model.zerograd()
            
            model_output = model(Tensor([train_input_data1[i]]))

            # print(f"Model says {model_output}")
            # print(f"We expected {Tensor([train_output_data1[i]])}")
        
            loss = nn.mse(model_output, Tensor([train_output_data1[i]]))
            train_errors[epoch] += loss.data
            # print(f"Loss: {loss}")

            # Backprop
            loss.backward()

            if((i + 1) % batch_size == 0):
                
                # After we have done the backward pass, we can step the model weights
                sgd.step()
                model.zerograd()
                
                # Update Learning Rate
                sgd.lr -= lr_rate
                
        test_errors.append(0)
        for i in range(len(test_input_data1)):
            
            model_output = model(Tensor([test_input_data1[i]]))

            # print(f"Model says {model_output}")
            # print(f"We expected {Tensor([test_output_data1[i]])}")
        
            test_errors[epoch] += nn.mse(model_output, Tensor([test_output_data1[i]])).data
            # print(f"Loss: {loss}")
        print(test_errors[epoch])
        
    plt.figure()
    plt.title("Training Error vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.plot(train_errors)
    plt.show()
    
    plt.figure()
    plt.title("Validation Error vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.plot(test_errors)
    plt.show()
        
        

        

