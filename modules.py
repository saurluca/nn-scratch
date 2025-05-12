import numpy as np

VERBOSE = False

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


class LinearLayer(Module):
    def __init__(self, input_d, output_d, weights=None, bias=None, use_bias=True):
        self.input = None
        self.grad_weights = None
        self.grad_bias = None
        self.input_d = input_d
        self.output_d = output_d

        # if bias:
        #     assert len(bias) == input_d
        #     self.bias = bias
        # else:
        #     self.bias = np.zeros(input_d)

        if weights is not None:
            print(f"len(weights): {weights}, input_d: {input_d}")
            self.weights = weights
        else:
            # using He initialization of weights
            self.weights = np.random.normal(
                loc=0, scale=np.sqrt(2.0 / input_d), size=(input_d, output_d)
            )

    def forward(self, x):
        self.input = x
        if np.isscalar(x):
            x = np.array([x])
            if VERBOSE:
                print("x is scalar, converting to 1D array")
        y = np.matmul(x, self.weights)
        # y += self.bias
        return y

    def backward(self, grad):
        # calculate the jacobian
        self.grad_weights = np.outer(grad, self.input)

        # calculate the gradient in respect to x
        return grad @ np.transpose(self.weights)


class FeedForwardNeuralNetwork(Module):
    def __init__(self, n_layers, model_d, input_d, output_d):
        self.n_layers = n_layers
        self.model_d = model_d
        self.input_d = input_d
        self.output_d = output_d
        self.l_stack = []

        self.l_stack.append(LinearLayer(input_d, model_d))
        self.l_stack.append(ReLU())
        self.l_stack.append(LinearLayer(model_d, output_d))
        self.l_stack.append(Sigmoid())
        
    def forward(self, x):
        for layer in self.l_stack:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.l_stack):
            grad = layer.backward(grad)


class SGD(Module):
    def __init__(self, lr, model):
        self.lr = lr
        self.model = model

    def step(self):
        for layer in self.model.l_stack:
            if isinstance(layer, LinearLayer):
                # TODO why do we transpose
                layer.weights = layer.weights - np.transpose(layer.grad_weights) * self.lr 
                
    def zero_grad(self):
        raise NotImplementedError("Not implemented")
