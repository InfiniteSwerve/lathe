import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.layers.append(nn.Linear(prev_size, output_size))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.softmax(x)

model = [
        torch.tensor("
    ]
