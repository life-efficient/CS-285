import torch

class NN(torch.nn.Module):
    def __init__(self, layers, embedding=False, distribution=False):
        super().__init__()
        l = []
        for idx in range(len(layers) - 1):
            if idx == 0 and embedding:
                l.append(torch.nn.Embedding(layers[idx], layers[idx+1]))
                continue
            l.append(torch.nn.Linear(layers[idx], layers[idx+1]))   # add a linear layer
            if idx != len(layers) - 2: # if this is not the last layer
                l.append(torch.nn.ReLU())   # activate
        if distribution:    # if a probability dist output is required
            l.append(torch.nn.Softmax())    # apply softmax to output
            
        self.layers = torch.nn.Sequential(*l)

    def forward(self, x):
        return self.layers(x)