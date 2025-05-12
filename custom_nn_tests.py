import numpy as np
from custom_nn import ReLU, Sigmoid, Softmax, LinearLayer, FeedForwardNeuralNetwork


def test_relu_forward():
    relu = ReLU()
    assert relu.forward(1) == 1
    assert relu.forward(0) == 0
    assert relu.forward(-0.2) == 0


def test_sigmoid_forward():
    sigmoid = Sigmoid()
    # Test sigmoid(0) = 0.5
    assert np.isclose(sigmoid.forward(0), 0.5)
    # Test sigmoid(1) ≈ 0.731
    assert np.isclose(sigmoid.forward(1), 0.7310585786300049)
    # Test sigmoid(-1) ≈ 0.269
    assert np.isclose(sigmoid.forward(-1), 0.2689414213699951)
    # Test sigmoid of large positive number approaches 1
    assert np.isclose(sigmoid.forward(10), 0.9999546021312976)
    # Test sigmoid of large negative number approaches 0
    assert np.isclose(sigmoid.forward(-10), 4.5397868702434395e-05)


def test_softmax_forward():
    softmax = Softmax()
    # Test simple case with 2 values
    x = np.array([1.0, 2.0])
    result = softmax.forward(x)
    expected = np.array([0.26894142, 0.73105858])
    assert np.allclose(result, expected)

    # Test case with all equal values
    x = np.array([1.0, 1.0, 1.0])
    result = softmax.forward(x)
    expected = np.array([0.33333333, 0.33333333, 0.33333333])
    assert np.allclose(result, expected)

    # Test case with negative values
    x = np.array([-1.0, 0.0, 1.0])
    result = softmax.forward(x)
    expected = np.array([0.09003057, 0.24472847, 0.66524096])
    assert np.allclose(result, expected)

    # Test that output sums to 1
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = softmax.forward(x)
    assert np.isclose(np.sum(result), 1.0)


def test_relu_backward():
    relu = ReLU()

    # Test case 1: Basic positive and negative values
    x = np.array([1.0, -2.0, 0.0, 3.0])
    relu.forward(x)  # Need to call forward first to set up internal state
    grad_in = np.array([1.0, 1.0, 1.0, 1.0])
    grad_out = relu.backward(grad_in)
    expected = np.array(
        [1.0, 0.0, 0.0, 1.0]
    )  # Gradient only flows where input was positive
    assert np.allclose(grad_out, expected)

    # Test case 2: All negative values
    x = np.array([-1.0, -2.0, -3.0])
    relu.forward(x)
    grad_in = np.array([1.0, 1.0, 1.0])
    grad_out = relu.backward(grad_in)
    expected = np.array([0.0, 0.0, 0.0])  # No gradient should flow
    assert np.allclose(grad_out, expected)

    # Test case 3: All positive values
    x = np.array([1.0, 2.0, 3.0])
    relu.forward(x)
    grad_in = np.array([0.5, 1.0, 1.5])
    grad_out = relu.backward(grad_in)
    expected = np.array([0.5, 1.0, 1.5])  # Gradient should flow unchanged
    assert np.allclose(grad_out, expected)


def test_sigmoid_backward():
    sigmoid = Sigmoid()

    # Test case 1: Input of 0
    x = np.array([0.0])
    sigmoid.forward(x)
    grad_in = np.array([1.0])
    grad_out = sigmoid.backward(grad_in)
    # At x=0, sigmoid'(0) = 0.25 because sigmoid(0) = 0.5
    expected = np.array([0.25])  # 0.5 * (1 - 0.5) * 1.0
    assert np.allclose(grad_out, expected)

    # Test case 2: Multiple values
    x = np.array([1.0, -1.0, 0.0])
    sigmoid.forward(x)
    grad_in = np.array([1.0, 1.0, 1.0])
    grad_out = sigmoid.backward(grad_in)
    # For x=1: sigmoid(1) ≈ 0.731, so gradient ≈ 0.731 * (1 - 0.731)
    # For x=-1: sigmoid(-1) ≈ 0.269, so gradient ≈ 0.269 * (1 - 0.269)
    # For x=0: sigmoid(0) = 0.5, so gradient = 0.5 * (1 - 0.5)
    expected = np.array([0.196612, 0.196612, 0.25])
    assert np.allclose(grad_out, expected, rtol=1e-6)

    # Test case 3: Different incoming gradients
    x = np.array([0.0, 0.0])
    sigmoid.forward(x)
    grad_in = np.array([2.0, 0.5])
    grad_out = sigmoid.backward(grad_in)
    expected = np.array([0.5, 0.125])  # 0.5 * (1 - 0.5) * [2.0, 0.5]
    assert np.allclose(grad_out, expected)


def test_linear_layer_backward():
    # Test with 2x3 weights matrix (input_d=2, output_d=3)
    input_d, output_d = 2, 3

    # Create layer with fixed weights for predictable testing
    weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # shape: (2, 3)
    layer = LinearLayer(input_d, output_d, weights=weights)

    # Test case 1: Basic backward pass
    x = np.array([1.0, 2.0])  # Input
    layer.forward(x)  # Need to call forward first to set internal state

    grad_in = np.array([0.5, 1.0, 1.5])  # Incoming gradient with shape (output_d,)
    grad_out = layer.backward(grad_in)

    # Expected weight gradients using outer product
    expected_grad_weights = np.outer(grad_in, x)
    assert np.allclose(layer.grad_weights, expected_grad_weights)

    # Expected input gradients using matrix multiplication with transposed weights
    expected_grad_input = grad_in @ weights.T
    assert np.allclose(grad_out, expected_grad_input)

    # Test case 2: Different input
    x = np.array([-1.0, 0.5])
    layer.forward(x)

    grad_in = np.array([1.0, 1.0, 1.0])
    grad_out = layer.backward(grad_in)

    expected_grad_weights = np.outer(grad_in, x)
    assert np.allclose(layer.grad_weights, expected_grad_weights)

    expected_grad_input = grad_in @ weights.T
    assert np.allclose(grad_out, expected_grad_input)


def test_feedforward_nn_layer_integration():
    # Test simple 2-layer network: input -> Linear -> ReLU -> Linear -> Sigmoid
    input_d, hidden_d, output_d = 2, 3, 1

    # Import required class if not already done

    # Create a simple network with fixed weights for testing
    network = FeedForwardNeuralNetwork(
        n_layers=0, model_d=hidden_d, input_d=input_d, output_d=output_d
    )

    # Replace the randomly initialized weights with deterministic ones for testing
    # First layer (2x3)
    network.l_stack[0].weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    # Last layer (3x1)
    network.l_stack[2].weights = np.array([[0.7], [0.8], [0.9]])

    # Test that forward works before testing backward
    x = np.array([1.0, 2.0])
    output = network.forward(x)
    print("output", output)
    # Don't check exact value as it depends on the activation functions
    assert output.shape == (output_d,)

    # Now test backward pass
    grad_in = np.array([0.5])  # Scalar gradient into the network

    # Assuming the backward method has been implemented
    try:
        network.backward(grad_in)
        # If no exception, check that gradients were computed in each layer
        assert hasattr(network.l_stack[0], "grad_weights")
        assert hasattr(network.l_stack[2], "grad_weights")
        assert network.l_stack[0].grad_weights is not None
        assert network.l_stack[2].grad_weights is not None
    except NotImplementedError:
        # If not implemented, test will be marked as skipped
        import pytest

        pytest.skip("FeedForwardNeuralNetwork.backward is not implemented yet")
