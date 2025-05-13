import numpy as np
import matplotlib.pyplot as plt

from modules import FeedForwardNeuralNetwork, MSELoss, SGD
from data import load_data_sin_regression


def train(model, loss_fn, optimiser, train_set, epochs=5, calculate_accuracy=False):
    train_loss = []
    accuracy = []
    print("start training")
    for i in range(epochs):
        correct_n = 0
        train_loss_epoch = 0.0
        for x, y in train_set:
            y_pred = model(x)
            train_loss_epoch += loss_fn(y_pred, y)

            grad = loss_fn.backward()
            model.backward(grad)
            optimiser.step()

            # calculate correct_n for accuracy
            if calculate_accuracy and np.round(y_pred) == y:
                correct_n += 1

        train_loss.append(train_loss_epoch)
        if calculate_accuracy:
            accuracy.append(correct_n / len(train_set))
            print(f"accuracy of epoch {i} : {accuracy[i]}")

    if calculate_accuracy:
        print("first accuracy", accuracy[0])
        print("final accuracy", accuracy[-1])
    print("first loss", train_loss[0])
    print("final loss", train_loss[-1])

    return train_loss, accuracy


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


def main():
    # model
    input_d = 1
    model_d = 5
    n_layers = 3
    output_d = 1
    np.random.seed(42)
    seed_train_data = 42
    seed_val_data = 41

    # training
    lr = 0.1
    epochs = 10

    # lets build a postive / negative number classifer
    train_set = load_data_sin_regression(100, seed_train_data)
    test_set = load_data_sin_regression(20, seed_val_data)

    model = FeedForwardNeuralNetwork(n_layers, model_d, input_d, output_d)

    loss_fn = MSELoss()
    optimiser = SGD(lr, model)

    train_loss, accuracy = train(model, loss_fn, optimiser, train_set, epochs)

    eval_loss, accuracy = evaluate(model, loss_fn, test_set)
    print("Evaluation Loss:", eval_loss)
    print("Evaluation Accuracy:", accuracy)

    plot_accuracy(accuracy)
    plot_loss(train_loss)

    plot_predictions(model, train_set, test_set)


if __name__ == "__main__":
    main()
