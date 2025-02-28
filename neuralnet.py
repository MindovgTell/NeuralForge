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
        self.bias = 0
        self.nonlin = nonlin

    def __call__(self, input_vec):
        act = sum()