import torch

class NN(torch.nn.Module):
    def __init__(self, layers, distribution=True):
        super().__init__()
        l = []
        for idx in range(len(layers) - 1):
            l.append(torch.nn.Linear(layers[idx], layers[idx+1]))   # add a linear layer
            if idx != len(layers) - 1:
                l.append(torch.nn.ReLU())   # activate
        if distribution:    # if a probability dist output is required
            l.append(torch.nn.Softmax())    # apply softmax to output
            
        self.layers = torch.nn.Sequential(*l)

    def forward(self, x):
        return self.layers(x)