import torch


def manual(X_value, neuron_1, neuron_2, neuron_3, neuron_1_bias, neuron_2_bias):
    
    # layer 1
    print(f"input {X_value}")
    X_value_1 = torch.cat((X_value, neuron_1_bias))
    out_1 = torch.dot(neuron_1, X_value_1)
    print(f"output neuron_1 {out_1}")
    
    X_value_2 = torch.cat((X_value, neuron_2_bias))
    out_2 = torch.dot(neuron_2, X_value_2)
    print(f"output neuron_2 {out_2}")
    
    xy_2 = torch.tensor([out_1.item(), out_2.item()])
    print(f"concatinated out_1 and out_1 {xy_2}")

    # layer 2
    out_3 = torch.dot(neuron_3, xy_2)
    print(f"output final neuron_3 {out_3}")
    

def torch_way(X_value, neuron_1, neuron_2, neuron_3):
    layer_1 = torch.nn.Linear(2, 2, bias=False)
    layer_2 = torch.nn.Linear(2, 1, bias=False)

    with torch.no_grad():  # Ensure no gradient tracking during initialization
        combined_weights = torch.stack((neuron_1, neuron_2))
        layer_1.weight.copy_(combined_weights)
        layer_2.weight.copy_(neuron_3)

    print(f"weights layer_1 {layer_1.weight}")
    print(f"weights layer_2 {layer_2.weight}")

    out_1 = layer_1(X_value)
    print(f"layer_1 output {out_1}")
    out_2 = layer_2(out_1)
    print(f"layer_2 output {out_2}")


def main():
    # X_value = torch.tensor([1.0, 2.0], requires_grad=False),
    X_value = torch.tensor([1.0, 2.0])
    
    # layer 1 
    # third item corresponds to the weighting of the
    neuron_1 = torch.tensor([2.0, 2.0, 1.0])
    neuron_1_bias = torch.tensor([1.0])
    neuron_2 = torch.tensor([-2.0, -4.0, -2.0])
    neuron_2_bias = torch.tensor([1.0])
    
    # layer 2
    neuron_3 = torch.tensor([1.0, 1.0])
    
    manual(X_value, neuron_1, neuron_2, neuron_3, neuron_1_bias, neuron_2_bias)
    print("\n", 50 * "=", "\n")
    # torch_way(X_value, neuron_1, neuron_2, neuron_3)


if __name__ == "__main__":
    main()
