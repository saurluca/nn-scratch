import numpy as np
import matplotlib.pyplot as plt
from modules import FeedForwardNeuralNetwork, BCELoss, SGD


def load_data(n_samples=10, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-10, 10, n_samples)
    # generates label 0 for negative 1 for positve
    y = np.maximum(0, np.sign(x))
    data = list(zip(x, y))
    return data


def train(model, loss_fn, optimiser, train_set, epochs=5):
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
            if np.round(y_pred) == y:
                correct_n += 1
                
        train_loss.append(train_loss_epoch)
        accuracy.append(correct_n / len(train_set))
        print(f"accuracy of epoch {i} : {accuracy[i]}")
 
    print("first accuracy", accuracy[0])
    print("first loss", train_loss[0])
    print("final accuracy", accuracy[-1])
    print("final loss", train_loss[-1])

    return train_loss, accuracy


def evaluate(model, loss_fn, optimiser, eval_set):
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
    plt.savefig('results/loss_plot.png')


def plot_accuracy(accuracy):
    plt.clf()
    plt.plot(accuracy)
    plt.savefig('results/accuracy_plot.png')


def main():
    # model
    input_d = 1
    model_d = 2
    n_layers = 0
    output_d = 1
    np.random.seed(42)
    seed_train_data = 42
    seed_val_data = 41
    
    # training
    lr = 0.01
    epochs = 10

    # lets build a postive / negative number classifer
    train_set = load_data(100, seed_train_data)
    test_set = load_data(20, seed_val_data)

    model = FeedForwardNeuralNetwork(n_layers, model_d, input_d, output_d)

    loss_fn = BCELoss()
    optimiser = SGD(lr, model)

    train_loss, accuracy = train(model, loss_fn, optimiser, train_set, epochs)

    eval_loss, accuracy = evaluate(model, loss_fn, optimiser, test_set)
    print("Evaluation Loss:", eval_loss)
    print("Evaluation Accuracy:", accuracy)

    plot_accuracy(accuracy)
    plot_loss(train_loss)


if __name__ == "__main__":
    main()
