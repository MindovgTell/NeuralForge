

class Tensor:
    def __init__(self, data, _childs=(), _operation = ''):
        self.data = data
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_childs)        
        self._op = _operation #the operation which produces this node

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data + other.data, (self,other), '+')

        def _backward():
            self.grad  += output.grad
            other.grad += output.grad
        
        output._backward = _backward

        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.grad * output.grad
            other.grad += self.grad  * output.grad

        output._backward = _backward
        return output
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)), "This function support only int/float variables"
        output = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * output.grad
        output._backward = _backward

        return output
    
    def relu(self):
        output = Tensor(0 if self.data < 0 else self.data, (self, ), 'ReLU')
        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward

        return output
    
    def backward(self):
        # For correct implementation of backpropagation through graph
        # were intorduce topological sort for elements in our childrens graph

        graph = []
        visited = set()
        def build_topology(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topology(child)
                graph.append(v)
        build_topology(self)

        self.grad = 1
        for v in reversed(graph):
            v._backward
        
        