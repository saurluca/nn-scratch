# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Sigmoid:
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        return self._sigmoid(x)
    
    def backward(self, grad):
        return self._sigmoid(grad) * (1 - self._sigmoid(grad)) * grad
    
    def __call__(self, x):
        return self.forward(x)
    

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, grad):
        return np.where(grad > 0, 1, 0) * grad
    
    def __call__(self, x):
        return self.forward(x)
    
    
class MSE:
    def __init__(self):
        self.pred = None
        self.target = None
    
    def forward(self, pred, target):
        self.pred, self.target = pred, target
        return np.mean((pred - target)**2)
        
    def backward(self):
        return np.mean(0.5 * (self.pred - self.target)).reshape(1)
    
    def __call__(self, pred, target):
        return self.forward(pred, target)


class SGD:
    def __init__(self, params, criterion, lr=0.01, momentum=0.9):
        self.params = params
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.updates = [[np.zeros_like(layer.W), np.zeros_like(layer.b)] for layer in self.params]
    
    def zero_grad(self):
        for layer in self.params:
            layer.dW = 0.0
            layer.db = 0.0
    
    def step(self):
        for i, layer in enumerate(reversed(self.params)):
            # TODO mage less ugly indexing
            # print("self.updates", self.updates[i])
            # print(f"layer {i} W {layer.W} dW {layer.dW} b {layer.b} db {layer.db}")
            self.updates[-i-1][0] = self.lr * layer.dW + self.momentum * self.updates[-i-1][0]
            self.updates[-i-1][1] = self.lr * layer.db + self.momentum * self.updates[-i-1][1]
            layer.W -= self.updates[-i-1][0]
            layer.b -= self.updates[-i-1][1]
            

    def __call__(self):
        return self.step()
    
    
class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params(self):
        """Return a list of layers that have a Weight vectors"""
        params = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                params.append(layer)
        return params
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class LinearLayer:
    def __init__(self, in_dim: int, out_dim: int):
        self.W = 0.1 * np.random.randn(out_dim, in_dim) # TODO xavier init
        self.b = np.zeros(out_dim)
        self.dW = 0.0
        self.db = 0.0
        self.x = None
    
    def forward(self, x):
        # print(f"x: {x}, self.W {self.W}, self.b {self.b}")
        self.x = x
        return self.W @ x + self.b
    
    def backward(self, grad):
        # print(f"shape of incoming grad {grad} \n shape of W {self.W.shape}")
        self.dW = np.outer(grad, self.x)
        self.db = grad 
        grad = self.W.T @ grad
        return grad

    def __call__(self, x):
        return self.forward(x)


def train(train_data, model, criterion, optimiser, n_epochs=10):
    train_losses = []
    outputs = []
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        outputs_epoch = []
        for X, target in train_data:
            # forward pass
            pred = model(X)
            loss = criterion(pred, target)
            train_loss += loss
            outputs_epoch.append(pred)
            
            # backward pass
            # optimiser.zero_grad()
            grad = criterion.backward()
            model.backward(grad)
            optimiser.step()
            
            # print(f"y {target}, pred {pred}, loss {loss}")
        train_losses.append(train_loss)
        outputs.append(outputs_epoch)
    return train_losses, outputs


def plot_loss(losses):
    plt.plot(losses)
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

                    
def plot_predictions(outputs, targets):
    plt.scatter(targets, outputs) 
    plt.title("Predictions vs Targets")
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.show()


def main():
    np.random.seed(42)
    
    # config
    n_epochs = 10
    lr = 0.1
    
    # setup dummy data
    n_samples = 200
    inputs = np.random.uniform(-1, 1, size=(n_samples, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    true_b = -0.1
    targets = Sigmoid()._sigmoid(inputs @ true_w + true_b)
    train_data = list(zip(inputs, targets))
        
    model = Sequential([
        LinearLayer(in_dim=3, out_dim=1), 
        Sigmoid(),
        # LinearLayer(in_dim=4, out_dim=1), 
        # Sigmoid()
    ])
    criterion = MSE()
    optimiser = SGD(model.params(), criterion, lr=lr, momentum=0.9)
    
    train_losses, outputs = train(train_data, model, criterion, optimiser, n_epochs)
    plot_loss(train_losses)
    plot_predictions(outputs[-1], targets)
    
    print(f"final loss {train_losses[-1]}")
    
    # print out final model params
    final_params = model.params()[0]
    print(f"true W {true_w} model w {final_params.W} \n true b {true_b}, model b {final_params.b}")
    
if __name__ == "main":
    main()
    
    
main()