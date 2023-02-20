import math

class Value:

    # Class initialization method
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        # We save a set of the children, each time we perform an operation we will keep a record of the operands
        self._prev = set(_children)
        # Additionally, we will save a string with the operation performed to reach this value
        self._op = _op
        self.grad = 0.0
        self.label = label
        self._backward= lambda :None

    # Method for data representation
    def __repr__(self):
        return f"Value(data={self.data})"

    # Be careful with magic operators that override default actions when certain actions are performed on an object




    # Addition

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other) , '+')
		# For out backwards functions where we are calculating the gradients(slope) of this node with respect to our end result in each we use the chain rule to calculate
		# If a variable z depends on the variable y, which itself depends on the variable x (that is, y and z are dependent variables), then z depends on x as well, via the intermediate variable y.
		# therefore going backwards through our operations we can deduce the gradient of each node.
		# in the case of adition:
		# We have L = f + g, and we want to calculate dL/df and dL/dg. Using the chain rule, we have dL/df = dout/df, and dL/dg = dout/dg.
		# Therefore, the gradients for the two inputs are simply the same as the output gradient.
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out




    # Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self,other), '*')

		# We have L = f * g, and we want to calculate dL/df and dL/dg. Using the chain rule, we have dL/df = g * dout/df, and dL/dg = f * dout/dg.
		# Therefore, the gradients for the two inputs are the other input multiplied by the output gradient.
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out




    # It is necessary to override reverse multiplication, radd and other methods that can use __add__ and __mul__
    def __rmul__(self, other):
            return self * other

    def __radd__(self, other): # other + self
        return self + other

    # Negation
    def __neg__(self):
        return self * -1

    # Subtraction
    def __sub__(self,other):
        return self + (-other)




    # Power
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only exponents of type int or float, not Value"
        out = Value(self.data**other, (self,), f'**{other}')

		# We have L = (f)**n, and we want to calculate dL/df. Using the chain rule, we have dL/df = n * (f**(n-1)) * dout/df.
		# Therefore, the gradient for the input f is n * (f**(n-1)) * the output gradient, while the gradient for the input n is not directly calculated in this function.
        def _backward():
            self.grad += other * (self.data**(other - 1)) * out.grad
        out._backward = _backward

        return out


	#Same as before with the r functions
    # Division
    def __truediv__(self, other):
        return self * other**-1



    # Exponential
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

		# We have L = e^f, and we want to calculate dL/df. Using the chain rule, we have dL/df = e^f * dout/df.
		# Therefore, the gradient for the input f is e^f * the output gradient. quite straigth forward
        def _backward():
            self.grad += math.exp(x) * out.grad
        out._backward = _backward

        return out



    # Hyperbolic tangent
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = Value( t, (self, ), 'tanh')

		# Doing the local derivative of our tanh exporesion and as usual using the chain rule therefore multipliying with our gradient we have:
        def _backward():
            self.grad +=  (1 - t**2) * out.grad
        out._backward = _backward

        return out



# This method performs backpropagation on the computational graph starting from the current node.
# It first generates a topological order of the graph using a depth-first search algorithm.
# It then traverses the nodes in reverse topological order and computes gradients for each node.
# Finally, it stores the computed gradients in the _grad attribute of each node.
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
