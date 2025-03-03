import random
from backend import Tensor

class Base:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
        
    def parameters(self):
        return []
    
class Neuron(Base):
    def __init__(self, num_input, nonlin = True):
        self.weight = [Tensor(random.uniform(-1,1)) for _ in range(num_input)]
        self.bias = Tensor(0)
        self.nonlin = nonlin

    def __call__(self, input_vec):
        act = sum((wi * xi for wi,xi in zip(self.weight,input_vec)), self.bias)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.weight + [self.bias]
        
    def __repr__(self):
        return f"{'Relu' if self.nonlin else 'Linear'}Neuron({len(self.weight)})"
    
class Layer(Base):
    def __init__(self, num_input, num_output, **kwargs):
        self.neurons = [Neuron(num_input,**kwargs) for _ in range(num_output)]
    
    def __call__(self, vector):
        output = [neuron(vector) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(neuron) for neuron in self.neurons)}]"
    
class MLP(Base): # Multi Layer Perceptron 
    def __init__(self, num_input, num_output):
        sz = [num_input] + num_output
        self.layers = [Layer(sz[i],sz[i+1], nonlin = i != len(num_output)-1) for i in range(len(num_output))]

    def __call__(self, vector):
        for layer in self.layers:
            vector = layer(vector)
        return vector
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
