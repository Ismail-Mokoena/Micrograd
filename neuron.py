from value import Value
import random
class Neuron:
    def __init__(self, n_inputs) -> None:
        #create weight thats some rand_num between (-1,1)
        self.weights = [Value(random.uniform(-1,1)) for _ in range(n_inputs)]
        #create bias some rand_num between(-1,1)
        self.bias = Value(random.uniform(-1,1))
    
    def __call__(self, x):
        # we want w*x+b
        fx = sum( (wi*xi for wi, xi in zip(self.weights, x)),  self.bias)
        # pass through non-linearity
        out = fx.tahn()
        return out

class Layer:
    def __init__(self, n_inputs, n_outputs) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.neurons = [Neuron(self.n_inputs) for _ in range(self.n_outputs)]
        
    def __call__(self, inputs):
        out = [n(inputs) for n in self.neurons]
        return out if len(out)==1 else out

class MLP:
    def __init__(self, n_inputs, n_outputs):
        size = [n_inputs] + n_outputs
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outputs))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x