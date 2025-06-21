import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules import FeedForwardNeuralNetwork, MSELoss, SGD
from data import load_data_sin_regression


def train_classifier(model, loss_fn, optimiser, train_set, epochs=5):
    train_loss = []
    accuracy = []
    for i in tqdm(
        range(epochs), desc="Training", total=epochs, postfix={"loss": "N/A"}
    ):
        correct_n = 0
        train_loss_epoch = 0.0
        for x, y in train_set:
            y_pred = model(x)
            train_loss_epoch += loss_fn(y_pred, y)

            grad = loss_fn.backward()
            model.backward(grad)
            optimiser.step()

            # calculate correct_n for accuracy
            if np.round(y_pred) == y:
                correct_n += 1

        train_loss.append(train_loss_epoch)
        accuracy.append(correct_n / len(train_set))
        tqdm.get_current().set_postfix({"loss": f"{train_loss_epoch:.4f}"})

    print("first accuracy", accuracy[0])
    print("final accuracy", accuracy[-1])
    print("first loss", train_loss[0])
    print("final loss", train_loss[-1])

    return train_loss, accuracy


def train_regressor(model, loss_fn, optimiser, train_set, epochs=5):
    train_loss = []
    print("start training")

    progress_bar = tqdm(
        range(epochs), desc="Training", total=epochs, postfix={"loss": "N/A"}
    )
    for i in progress_bar:
        train_loss_epoch = 0.0
        # Shuffle training set
        np.random.shuffle(train_set)
        for x, y in train_set:
            optimiser.zero_grad()

            y_pred = model(x)
            train_loss_epoch += loss_fn(y_pred, y)

            grad = loss_fn.backward()
            model.backward(grad)
            optimiser.step()
        train_loss.append(train_loss_epoch)
        progress_bar.set_postfix({"loss": f"{train_loss_epoch:.4f}"})

    print("first loss", train_loss[0])
    print("final loss", train_loss[-1])

    return train_loss


def evaluate(model, loss_fn, eval_set):
    correct_n = 0
    eval_loss = 0.0
    for x, y in eval_set:
        y_pred = model(x)
        eval_loss += loss_fn(y_pred, y)

        if np.round(y_pred) == y:
            correct_n += 1

    accuracy = correct_n / len(eval_set)

    return eval_loss, accuracy


def plot_loss(loss):
    plt.clf()
    plt.plot(loss)
    plt.savefig("results/loss_plot.png")


def plot_accuracy(accuracy):
    plt.clf()
    plt.plot(accuracy)
    plt.savefig("results/accuracy_plot.png")


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


def plot_predictions_scaled(
    model, train_set_scaled, test_set_scaled, mean_x, sd_x, mean_y, sd_y
):
    # Use the scaled inputs, but show original scale outputs
    plt.clf()

    # Extract original values
    train_x = [item[0] * sd_x + mean_x for item in train_set_scaled]
    train_y = [item[1] * sd_y + mean_y for item in train_set_scaled]
    test_x = [item[0] * sd_x + mean_x for item in test_set_scaled]
    test_y = [item[1] * sd_y + mean_y for item in test_set_scaled]

    # Generate model predictions on scaled inputs
    x_range_scaled = np.linspace(
        min(x for x, _ in train_set_scaled + test_set_scaled),
        max(x for x, _ in train_set_scaled + test_set_scaled),
        100,
    )

    # Scale predictions back to original scale
    predictions_scaled = []
    for x in x_range_scaled:
        try:
            pred = model(x)[0] * sd_y + mean_y
            # Handle NaN/Inf predictions
            if np.isnan(pred) or np.isinf(pred):
                pred = 0.0
            predictions_scaled.append(pred)
        except Exception as e:
            print(f"Error in prediction at x={x}: {e}")
            predictions_scaled.append(0.0)

    x_range = [x * sd_x + mean_x for x in x_range_scaled]

    plt.scatter(train_x, train_y, label="Train Set")
    plt.scatter(test_x, test_y, label="Test Set")
    plt.plot(x_range, predictions_scaled, label="Model Predictions", color="red")
    plt.legend()
    plt.savefig("results/predictions_plot.png")

    # Print out the first 5 predictions vs the actual
    first_5_predictions = predictions_scaled[:5]
    first_5_x = [x_range[i] for i in range(5)]
    first_5_actual = train_y[:5]
    print("First 5 Predictions vs Actual:")
    for x, pred, actual in zip(first_5_x, first_5_predictions, first_5_actual):
        print(f"Input: {x:.2f}, Prediction: {pred:.2f}, Actual: {actual:.2f}")


def main():
    # model
    input_d = 1
    model_d = 50
    n_layers = 4  # Reduced from 6 to simplify model
    output_d = 1
    np.random.seed(42)
    seed_train_data = 46
    seed_val_data = 44

    # training
    lr = 0.01
    epochs = 200

    # Data choice
    train_set = load_data_sin_regression(25, seed_train_data)
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
    train_set_scaled = []
    for x, y in train_set:
        train_set_scaled.append(((x - mean_x) / sd_x, (y - mean_y) / sd_y))

    test_set_scaled = [((x - mean_x) / sd_x, (y - mean_y) / sd_y) for x, y in test_set]

    print(
        f"Data standardized with mean_x={mean_x:.2f}, sd_x={sd_x:.2f}, mean_y={mean_y:.2f}, sd_y={sd_y:.2f}"
    )

    # Create model with custom layers
    model = FeedForwardNeuralNetwork(n_layers, model_d, input_d, output_d)
    loss_fn = MSELoss()

    # Add weight decay to prevent extreme weights
    optimiser = SGD(model, lr)

    # Monitor training
    try:
        train_loss = train_regressor(
            model, loss_fn, optimiser, train_set_scaled, epochs
        )
        print(f"Training completed successfully with final loss: {train_loss[-1]}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Try to recover
        train_loss = [float("nan")] if "train_loss" not in locals() else train_loss

    # Plot results
    plot_loss(train_loss)
    plot_predictions_scaled(
        model, train_set_scaled, test_set_scaled, mean_x, sd_x, mean_y, sd_y
    )


if __name__ == "__main__":
    main()
