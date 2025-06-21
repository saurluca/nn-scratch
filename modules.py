import numpy as np


class Module:
    def forward(self, input):
        raise NotImplementedError("Not implemented")

    def backward(self, grad):
        raise NotImplementedError("Not implemented")

    def __call__(self, x):
        return self.forward(x)


class Softmax(Module):
    def __init__(self):
        self.input = None

    def forward(self, x):
        exp_values = np.exp(x)
        return exp_values / np.sum(exp_values)

    def backward(self, grad):
        raise NotImplementedError("Not implemented")


class MSELoss(Module):
    def __init__(self):
        self.input_y_pred = None
        self.input_y = None

    def forward(self, y_pred, y):
        self.input_y_pred = y_pred
        self.input_y = y
        return np.mean((y_pred - y) ** 2)

    def backward(self):
        return 2 * (self.input_y_pred - self.input_y) / len(self.input_y_pred)

    def __call__(self, y_pred, y):
        return self.forward(y_pred, y)


class BCELoss(Module):
    def __init__(self):
        self.input_y_pred = None
        self.input_y = None

    def forward(self, y_pred, y):
        self.input_y_pred = y_pred
        self.input_y = y
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def backward(self):
        return -self.input_y / self.input_y_pred + (1 - self.input_y) / (
            1 - self.input_y_pred
        )

    def __call__(self, y_pred, y):
        return self.forward(y_pred, y)


class CrossEntropyLoss(Module):
    def __init__(self):
        self.input = None

    def forward(self, y_pred, y):
        loss = 0

        # TODO make more efficent
        for i in range(len(y_pred)):
            loss += -1 * y[i] * np.log(y_pred[i])

        return loss

    def backward(self, grad):
        raise NotImplementedError("Not implemented")


class Sigmoid(Module):
    def __init__(self):
        self.input = None
        self.grad = None

    def _get_sigmoid_value(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.input = x
        return self._get_sigmoid_value(x)

    def backward(self, grad):
        sigma_input = self._get_sigmoid_value(self.input)
        return grad * sigma_input * (1 - sigma_input)


class ReLU(Module):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad):
        return grad * np.where(self.input > 0, 1, 0)


class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        self.input = None
        self.alpha = alpha  # Slope for negative inputs

    def forward(self, x):
        self.input = x
        return np.maximum(self.alpha * x, x)

    def backward(self, grad):
        return grad * np.where(self.input > 0, 1, self.alpha)


class LinearLayer(Module):
    def __init__(self, input_d, output_d, weights=None, bias=None, use_bias=True):
        self.input = None
        self.grad_weights = None
        self.grad_bias = None
        self.input_d = input_d
        self.output_d = output_d
        self.use_bias = use_bias

        if weights is not None:
            print(f"len(weights): {weights}, input_d: {input_d}")
            self.weights = weights
        else:
            # using He initialization of weights
            self.weights = np.random.normal(
                loc=0, scale=np.sqrt(2.0 / input_d), size=(input_d, output_d)
            )

        if use_bias:
            if bias is not None:
                self.bias = bias
            else:
                self.bias = np.zeros(output_d)

    def forward(self, x):
        if np.isscalar(x):
            x = np.array([x])
        self.input = x
        y = np.matmul(x, self.weights)
        if self.use_bias:
            y += self.bias
        return y

    def backward(self, grad):
        # calculate the jacobian for the weights

        # Ensure input is 2D array (n_samples, input_features)
        if len(self.input.shape) == 1:
            input_reshaped = self.input.reshape(-1, 1)
            grad_reshaped = grad.reshape(1, -1)
            self.grad_weights = input_reshaped @ grad_reshaped
        else:
            self.grad_weights = self.input.T @ grad

        # The bias gradient is just the gradient itself
        if self.use_bias:
            self.grad_bias = grad.copy()

        # calculate the gradient in respect to x
        return grad @ np.transpose(self.weights)


class FeedForwardNeuralNetwork(Module):
    def __init__(
        self,
        n_layers,
        model_d,
        input_d,
        output_d,
        activation_fn="ReLU",
        final_activation_fn=None,
    ):
        self.n_layers = n_layers
        self.model_d = model_d
        self.input_d = input_d
        self.output_d = output_d
        self.activation_fn = activation_fn
        self.l_stack = []

        # initial layer
        self.l_stack.append(LinearLayer(input_d, model_d))
        self.l_stack.append(get_activation_fn(activation_fn))

        # hidden layers
        for i in range(n_layers):
            # linear layer
            self.l_stack.append(LinearLayer(model_d, model_d))
            # add activation function
            self.l_stack.append(get_activation_fn(activation_fn))

        # final layer
        self.l_stack.append(LinearLayer(model_d, output_d))
        if final_activation_fn:
            self.l_stack.append(get_activation_fn(final_activation_fn))

    def forward(self, x):
        # iterate through each layer sequentially
        for layer in self.l_stack:
            x = layer(x)
        return x

    def backward(self, grad):
        # iterate through each layer in reverse order
        for layer in reversed(self.l_stack):
            grad = layer.backward(grad)

    def print_structure(self):
        print("Network structure:")
        for i, layer in enumerate(self.l_stack):
            print(f"Layer {i}: {type(layer).__name__}")


class SGD(Module):
    def __init__(self, model, lr=0.01):
        self.lr = lr
        self.model = model

    def step(self):
        for layer in self.model.l_stack:
            if isinstance(layer, LinearLayer):
                layer.weights = layer.weights - layer.grad_weights * self.lr
                if layer.use_bias:
                    layer.bias = layer.bias - layer.grad_bias * self.lr

    def zero_grad(self):
        for layer in self.model.l_stack:
            if isinstance(layer, LinearLayer):
                layer.grad_weights = np.zeros_like(layer.grad_weights)
                layer.grad_bias = np.zeros_like(layer.grad_bias)
            else:
                layer.grad = 0


def get_activation_fn(name):
    if name == "ReLU":
        return ReLU()
    elif name == "LeakyReLU":
        return LeakyReLU()
    elif name == "Softmax":
        return Softmax()
    elif name == "Sigmoid":
        return Sigmoid()
    else:
        raise ValueError(f"Activation function {name} not recognized.")
