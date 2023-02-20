import random
from srcs.engine import Value

# A Multi-Layer Perceptron (MLP) is a type of artificial neural network that is commonly used for classification and regression problems.
# It consists of multiple layers of interconnected nodes, called neurons, which receive inputs, process them, and produce outputs.

# The core idea behind MLP is that each neuron in the network is responsible for computing a nonlinear function of its input.
# This function is usually a weighted sum of the inputs, followed by an activation function that introduces nonlinearity.
# The weights of each neuron are learned during training using an optimization algorithm such as backpropagation.

# MLPs are typically composed of an input layer, one or more hidden layers, and an output layer.
# The input layer receives the input data, the hidden layers perform computations on the input, and the output layer produces the final output of the network.

# In this code, we define a simple MLP consisting of one or more layers of neurons.
# The Layer class represents a single layer of neurons, where each neuron is represented by an instance of the Neuron class.
# The MLP class contains multiple layers of neurons, which are stacked on top of each other to form a deep neural network.
# The parameters of the MLP, which are the weights and biases of each neuron, can be accessed using the parameters() method.

class Module:

	# set the gradient of all parameters to zero
	def zero_grad(self):
		for p in self.parameters():
			p.grad = 0

	# return an empty list of parameters
	def parameters(self):
		return []


class Neuron(Module):

	# create a neuron with nin input weights and one bias term
	def __init__(self, nin):
		# initialize the input weights with random values between -1 and 1
		self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
		# initialize the bias term with a random value between -1 and 1
		self.b = Value(random.uniform(-1,1))

	# compute the output of the neuron for a given input
	def __call__(self, x):
		# calculate the weighted sum of the inputs plus the bias term
		act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
		# apply the hyperbolic tangent activation function to the weighted sum
		out = act.tanh()
		# return the output value
		return out

	# return the list of parameters (input weights and bias term) for the neuron
	def parameters(self):
		return self.w + [self.b]


class Layer(Module):

	# create a layer of nout neurons, each with nin input weights and one bias term
	def __init__(self, nin, nout):
		# create nout neurons, each with nin input weights and one bias term
		self.neurons = [Neuron(nin) for _ in range(nout)]

	# compute the output of the layer for a given input
	def __call__(self, x):
		# compute the output of each neuron in the layer for the input
		outs = [n(x) for n in self.neurons]
		# if there is only one output, return it; otherwise, return a list of outputs
		return outs[0] if len(outs) == 1 else outs

	# return the list of parameters for all the neurons in the layer
	def parameters(self):
		# concatenate the list of parameters for each neuron in the layer
		return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):

	# create a multi-layer perceptron with nin input neurons and a list of nouts, specifying the number of neurons in each layer
	def __init__(self, nin, nouts):
		# create a list of layer objects, each with the appropriate number of input and output neurons
		sz = [nin] + nouts
		self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

	# compute the output of the multi-layer perceptron for a given input
	def __call__(self, x):
		# feed the input through each layer in the MLP and get the output of the final layer
		for layer in self.layers:
			x = layer(x)
		return x

	# return the list of parameters for all the neurons in all the layers of the MLP
	def parameters(self):
		# concatenate the list of parameters for each layer in the MLP
		return [p for layer in self.layers for p in layer.parameters()]
