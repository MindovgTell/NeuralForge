import random
from backend import Tensor

class Base:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
        
    def parameters(self):
        return []
    
class Neuron(Base):
    def __init__(self, inputs_num, nonlin = True):
        self.weight = [Tensor(random.uniform(-1,1)) for _ in range(inputs_num)]
        self.bias = Tensor(0)
        self.nonlin = nonlin

    def __call__(self, input_vec):
        act = sum((wi * xi for wi,xi in zip(self.weight,input_vec)), self.bias)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.weight + [self.bias]
        
    def __repr__(self):
        return f"{'Relu' if self.nonlin else 'Linear'}Neuron({len(self.weight)})"
    
