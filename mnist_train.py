import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.layers.append(nn.Linear(prev_size, output_size))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.softmax(x)

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    input_size = 28 * 28
    hidden_sizes = [128, 64]
    output_size = 10
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Create MLP model
    model = MLP(input_size, hidden_sizes, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Testing
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the 10000 test images: {100 * correct / total}%")
    # Save the model weights
    torch.save(model.state_dict(), 'mlp_weights.pth')

    # Load the model weights and extract the matrices
    state_dict = torch.load('mlp_weights.pth')
    matrices = []

    for i in range(0, len(state_dict), 2):
        weight_key = f"layers.{i}.weight"
        bias_key = f"layers.{i}.bias"
        
        weight = state_dict[weight_key].cpu().numpy()
        bias = state_dict[bias_key].cpu().numpy()
        
        # Concatenate the bias to the weight matrix
        matrix = np.concatenate((weight, bias.reshape(-1, 1)), axis=1)
        matrices.append(matrix)

    matrices = []

    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data.cpu().numpy()
            bias = layer.bias.data.cpu().numpy()
            matrix = np.concatenate((weight, bias.reshape(-1, 1)), axis=1)
            matrices.append(matrix)

    # Save the matrices to a file
    np.savez('mlp_matrices.npz', *matrices)
