import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import load_data_sin_regression

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_torch(model, loss_fn, optimizer, train_loader, epochs=5):
    train_loss = []
    print("start training")

    for epoch in tqdm(range(epochs)):
        train_loss_epoch = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        # Average loss over batches
        train_loss_epoch /= len(train_loader)
        train_loss.append(train_loss_epoch)
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss_epoch:.6f}")

    print("First loss:", train_loss[0])
    print("Final loss:", train_loss[-1])

    return train_loss


def plot_loss(loss):
    plt.clf()
    plt.plot(loss)
    plt.savefig("results/loss_plot.png")


def plot_predictions(model, train_set, test_set):
    plt.clf()

    # Extract x and y values from the list of tuples
    train_x = [item[0] for item in train_set]
    train_y = [item[1] for item in train_set]

    test_x = [item[0] for item in test_set]
    test_y = [item[1] for item in test_set]

    # Generate model predictions for a smooth curve
    x_range = np.linspace(min(train_x + test_x), max(train_x + test_x), 100)
    predictions = [model(x) for x in x_range]

    plt.scatter(train_x, train_y, label="Train Set")
    plt.scatter(test_x, test_y, label="Test Set")
    plt.plot(x_range, predictions, label="Model Predictions", color="red")
    plt.legend()
    plt.savefig("results/predictions_plot.png")

    # Print out the first 5 predictions vs the actual
    first_5_predictions = predictions[:5]
    first_5_actual = train_y[:5]
    print("First 5 Predictions vs Actual:")
    for pred, actual in zip(first_5_predictions, first_5_actual):
        print(f"Prediction: {pred}, Actual: {actual}")


def plot_predictions_scaled(model, train_tensors, test_tensors, scaling_factors=None):
    # Use the scaled inputs, but show original scale outputs if scaling factors are provided
    plt.clf()

    # Unpack the tensors
    train_x_tensor, train_y_tensor = train_tensors
    test_x_tensor, test_y_tensor = test_tensors

    # Convert tensors to numpy for plotting
    train_x = train_x_tensor.numpy().flatten()
    train_y = train_y_tensor.numpy().flatten()
    test_x = test_x_tensor.numpy().flatten()
    test_y = test_y_tensor.numpy().flatten()

    if scaling_factors:
        mean_x = scaling_factors["mean_x"]
        sd_x = scaling_factors["sd_x"]
        mean_y = scaling_factors["mean_y"]
        sd_y = scaling_factors["sd_y"]

        # Unscale the data
        train_x = train_x * sd_x + mean_x
        train_y = train_y * sd_y + mean_y
        test_x = test_x * sd_x + mean_x
        test_y = test_y * sd_y + mean_y

    # Generate model predictions on scaled inputs
    x_range_scaled = np.linspace(
        min(train_x_tensor.min().item(), test_x_tensor.min().item()),
        max(train_x_tensor.max().item(), test_x_tensor.max().item()),
        100,
    )

    # Scale predictions back to original scale if scaling factors are provided
    predictions_scaled = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for x in x_range_scaled:
            try:
                x_tensor = torch.tensor([x], dtype=torch.float32).reshape(1, 1)
                pred = model(x_tensor).item()
                if scaling_factors:
                    pred = pred * sd_y + mean_y
                # Handle NaN/Inf predictions
                if np.isnan(pred) or np.isinf(pred):
                    pred = 0.0
                predictions_scaled.append(pred)
            except Exception as e:
                print(f"Error in prediction at x={x}: {e}")
                predictions_scaled.append(0.0)
    model.train()  # Set model back to training mode

    # Convert x values back to original scale if scaling factors are provided
    if scaling_factors:
        x_range = x_range_scaled * sd_x + mean_x
    else:
        x_range = x_range_scaled

    plt.scatter(train_x, train_y, label="Train Set")
    plt.scatter(test_x, test_y, label="Test Set")
    plt.plot(x_range, predictions_scaled, label="Model Predictions", color="red")
    plt.legend()
    plt.savefig("results/predictions_plot.png")

    # Print out the first 5 predictions vs the actual
    first_5_predictions = predictions_scaled[:5]
    first_5_x = x_range[:5]
    first_5_actual = train_y[:5]
    print("First 5 Predictions vs Actual:")
    for x, pred, actual in zip(first_5_x, first_5_predictions, first_5_actual):
        print(f"Input: {x:.2f}, Prediction: {pred:.2f}, Actual: {actual:.2f}")


def get_data_sin(batch_size=32, n_datapoints=100):
    seed_train_data = 42
    seed_val_data = 41
    # Data choice
    train_set = load_data_sin_regression(n_datapoints, seed_train_data)
    test_set = load_data_sin_regression(20, seed_val_data)

    # Preprocess data to standardize inputs and targets
    train_x = [x for x, _ in train_set]
    train_y = [y for _, y in train_set]

    # Calculate mean and standard deviation
    mean_x = np.mean(train_x)
    sd_x = np.std(train_x)
    mean_y = np.mean(train_y)
    sd_y = np.std(train_y)

    # Standardize the data (z-score normalization)
    train_x_scaled = [(x - mean_x) / sd_x for x in train_x]
    train_y_scaled = [(y - mean_y) / sd_y for y in train_y]

    # Convert to PyTorch tensors
    train_x_tensor = torch.tensor(train_x_scaled, dtype=torch.float32).reshape(-1, 1)
    train_y_tensor = torch.tensor(train_y_scaled, dtype=torch.float32).reshape(-1, 1)

    # Create train dataset and dataloader
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Process test data
    test_x = [x for x, _ in test_set]
    test_y = [y for _, y in test_set]
    test_x_scaled = [(x - mean_x) / sd_x for x in test_x]
    test_y_scaled = [(y - mean_y) / sd_y for y in test_y]

    # Convert test data to tensors
    test_x_tensor = torch.tensor(test_x_scaled, dtype=torch.float32).reshape(-1, 1)
    test_y_tensor = torch.tensor(test_y_scaled, dtype=torch.float32).reshape(-1, 1)

    # Create test dataset and dataloader
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    scaling_factors = {"mean_x": mean_x, "sd_x": sd_x, "mean_y": mean_y, "sd_y": sd_y}

    return train_loader, test_loader, scaling_factors


def main():
    # model
    input_d = 1
    model_d = 50
    output_d = 1
    n_datapoints = 100
    np.random.seed(42)

    # training
    lr = 0.01
    epochs = 200
    batch_size = 1

    train_loader, test_loader, scaling_factors = get_data_sin(batch_size, n_datapoints)

    # Create model with custom layers
    model = nn.Sequential(
        nn.Linear(input_d, model_d),
        nn.ReLU(),
        nn.Linear(model_d, model_d),
        nn.ReLU(),
        nn.Linear(model_d, model_d),
        nn.ReLU(),
        nn.Linear(model_d, output_d),
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Monitor training
    print("training")
    train_loss = train_torch(model, loss_fn, optimizer, train_loader, epochs)
    print(f"Training completed successfully with final loss: {train_loss[-1]}")

    # Plot results
    plot_loss(train_loss)
    plot_predictions_scaled(
        model,
        train_loader.dataset.tensors,
        test_loader.dataset.tensors,
        scaling_factors,
    )


if __name__ == "__main__":
    main()
