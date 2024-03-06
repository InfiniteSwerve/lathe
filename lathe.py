import graphviz
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from mnist_train import MLP
from showmethetypes import SMTT
from copy import deepcopy

tt = SMTT()

def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def normalize_weights(matrix):
    """
    Normalize the weights in the matrix to a range suitable for penwidth.
    """
    min_weight = np.min(matrix)
    max_weight = np.max(matrix)
    range_weight = max_weight - min_weight
    if range_weight == 0:
        return np.ones(matrix.shape)
    return 1 + 4 * (matrix - min_weight) / range_weight  # Scale and shift to range 1-5

def visualize_mlp_connections(*matrices):
    """
    Visualize connections between neurons in an MLP using Graphviz.
    Parameters:
    matrices (list): List of weight matrices representing the connections between layers.
    """
    # Validate matrix dimensions
    for i in range(len(matrices) - 1):
        if matrices[i].shape[1] != matrices[i + 1].shape[0]:
            raise ValueError(
                "Matrix dimensions must be compatible (matrix[i]'s columns = matrix[i+1]'s rows)."
            )

    # Create a directed graph
    dot = graphviz.Digraph(format="png")

    # Set graph layout options
    dot.graph_attr["rankdir"] = "LR"  # Left-to-right layout
    dot.graph_attr["nodesep"] = "0.5"  # Increase horizontal spacing between nodes
    dot.graph_attr["ranksep"] = "1.0"  # Increase vertical spacing between ranks

    z = np.array([0])
    # Add nodes for each layer
    for i, matrix in enumerate(matrices):

        # Check if each node has any incoming or outgoing edges
        for j in range(matrix.shape[0]):
            has_connection = False
            for k in range(matrix.shape[1]):
                if not np.isclose(matrix[j, k], z):
                    has_connection = True
                    break
            if has_connection:  # Always include input nodes
                node_label = (
                    f"Input {j}" if i == 0 
                    else f"Output {j}" if i == len(matrices) 
                    else f"Hidden {i-1}-{j}"
                )
                dot.node(f"L{i}N{j}", node_label)

    # Add edges for each connection between layers
    for i in range(len(matrices)):
        matrix = matrices[i]
        matrix_normalized = normalize_weights(matrix)
        for j in range(matrix.shape[0]):
            for k in range(matrix.shape[1]):
                if not np.isclose(matrix[j, k], z):
                    dot.edge(
                        f"L{i}N{j}",
                        f"L{i+1}N{k}",
                        label=str(f"{matrix[j, k]:.2f}"),
                        color=random_color(),
                        penwidth=str(matrix_normalized[j, k]),
                    )

    print("rendering graph")
    # Render and display the graph
    dot.render("mlp_graph")


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))


def lathe(model, inputs, targets, tau):
    print("Running lathe")
    # Set the model to evaluation mode
    model.eval()

    # Perform forward pass with the original model
    with torch.no_grad():
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        initial_loss = criterion(outputs, targets) 
        p = nn.functional.softmax(outputs, dim=1)

    # Get the layers and weights of the model
    layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    weights = [layer.weight for layer in layers]

    # Perform reverse topological sort on the layers
    sorted_layers = list(range(len(layers)))[::-1]
    tt(sorted_layers)

    threshold = tau

    prev_q = p
    initial_edge_count = 0
    final_edge_count = 0
    for layer_idx in sorted_layers:

        layer = layers[layer_idx]
        weight = weights[layer_idx]

        initial_edge_count += torch.count_nonzero(weight)
        
        print()
        print(f"Running on layer: {layer_idx}")
        tt(layer)

        for i in range(weight.shape[0]):
            print(f"\rMilling neuron {i}", end="")
            for j in range(weight.shape[1]):
                # Store the original weight value
                original_weight = weight[i, j].item()
                # Set the weight to zero
                layer.weight.data[i, j] = 0

                # Perform forward pass with the modified model
                with torch.no_grad():
                    outputs_modified = model(inputs)
                    q = nn.functional.softmax(outputs_modified, dim=1)

                # Calculate the KL divergence
                kl_div_old = nn.functional.kl_div(prev_q, p, reduction="batchmean").item()
                kl_div_new = nn.functional.kl_div(q, p, reduction="batchmean").item()

                if kl_div_new - kl_div_old < threshold:
                    # Keep the weight at zero if the KL divergence is below the threshold
                    layer.weight.data[i, j] = 0
                    prev_q = q
                else:
                    # print(f"Old KL, New KL: {kl_div_old:.4f} {kl_div_new:.4f}")
                    # Restore the original weight value if the KL divergence is above the threshold
                    layer.weight.data[i, j] = original_weight
            
        final_edge_count += torch.count_nonzero(weight)


    with torch.no_grad():
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        final_loss = criterion(outputs, targets) 

    print(
        f"\nInitial edges: {initial_edge_count}\nFinal edges: {final_edge_count}\nProportion kept: {(final_edge_count / initial_edge_count):.4f}\n"
    )

    print(f"Initial Loss: {initial_loss}\nFinal Loss: {final_loss}\n")

    return model


if __name__ == "__main__":
    input_size = 28 * 28
    hidden_sizes = [128, 64]
    output_size = 10
    batch_size = 2 ** 17
    tau = 0.0001

    print("loading stuff")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    print("stuff loaded")

    state_dict = torch.load("mlp_weights.pth")
    mlp = MLP(input_size, hidden_sizes, output_size).to(device)


    initial_edge_count = 0
    for i in range(0, len(state_dict), 2):
        weight_key = f"layers.{i}.weight"

        weight = state_dict[weight_key].cpu().numpy()
        initial_edge_count += np.count_nonzero(weight)


    mlp.load_state_dict(state_dict)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.view(-1, input_size).to(device)
        targets = targets.to(device)
        outputs = mlp(inputs)
        criterion = nn.CrossEntropyLoss()
        initial_loss = criterion(outputs, targets) 

        # Apply the lathe function to the model
        model = lathe(mlp, inputs, targets, tau)

        # Break the loop after processing a single batch
        if batch_idx == 0:
            break

    state_dict = mlp.state_dict()

    matrices = []

    for name, param in state_dict.items():
        print(name, param.size())

    final_edge_count = 0
    for i in range(0, len(state_dict), 2):
        weight_key = f"layers.{i}.weight"

        weight = state_dict[weight_key].cpu().numpy()

        matrices.append(weight)

    matrices = [matrix.T for matrix in matrices]

    visualize_mlp_connections(*matrices)

    # Example matrices
    # A = np.random.rand(3, 4)  # Replace with your matrix A
    # B = np.random.rand(4, 6)  # Replace with your matrix A
    # C = np.random.rand(6, 4)  # Replace with your matrix A
    # D = np.random.rand(4, 2)  # Replace with your matrix B

    # visualize_mlp_connections(A, B, C, D)
