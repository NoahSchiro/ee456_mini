from src.tensor import Tensor
from src import nn

class MLP(nn.Module):

    def __init__(self):

        # Example linear layer
        self.l1 = nn.Linear(10, 20)
        self.l2 = nn.Linear(20, 2)

    # Example forward pass
    def forward(self, input):
        input.transpose()
        input = self.l1(input)
        input = self.l2(input)
        return input

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()


if __name__=="__main__":

    # Model and optimizer
    model = MLP()
    sgd = nn.SGD(model.parameters(), lr=0.001)

    # Example tensor
    ex_tensor = Tensor([[1,2,3,4,5,6,7,8,9,10]])

    model_output = model(ex_tensor)

    expected_output = Tensor([[-420], [420]])

    print(f"Model says {model_output}")
    print(f"We expected {expected_output}")
    
    loss = nn.mse(model_output, expected_output)
    print(f"Loss: {loss}")

    # Backprop
    loss.backward()

    # After we have done the backward pass, we can step the model weights
    sgd.step()

