from graph import draw_dot
import math

class Value:
    def __init__(self, x, _children=(), _op='', label='') -> None:
        self.x = x
        self.grad = 0.0
        self._backwards = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self) -> str:
        return f"Value(x={self.x})"
    
    def __add__(self, data) -> float:
        equals = Value(self.x + data.x, (self, data), '+')
        #derivative / propogate gradient
        def _backwards():
            self.grad = 1.0*equals.grad
            data.grad = 1.0*equals.grad
        equals._backwards = _backwards
        return equals
  
    def __mul__(self, data) -> float:
        equals = Value(self.x*data.x, (self, data),'*')
        def _backwards():
            self.grad = (data.x)*(equals.grad)
            data.grad = (self.x)*(equals.grad)
        equals._backwards = _backwards
        return equals
    
    #Activation func = hyperbolic tan
    def tahn(self):
        x = self.x
        y = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        t = Value(y, (self, ), 'tanh')
        def _backwards():
            self.grad = (1 - y**2)*(t.grad)
        t._backwards = _backwards
        return t
    
    
#Summations
#test = sigma((lambda x, y: x*y),z) + b
def sigma(func:any, items: list)->float:
    n=0
    for i in items:
        n+=func(i[0],i[1])
    return n
        
        

#inputs      
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
#weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
#bias
b =  Value(6.8813735870195432, label='b') 

#
x1w1 = x1*w1; x1w1.label = 'x1w1' 
x2w2 = x2*w2; x2w2.label = 'x2w2'

fx = x1w1 + x2w2

n = fx + b; n.label='n'

o = n.tahn()
o.grad = 1.0
o._backwards()
n._backwards()
b._backwards()
fx._backwards()
x2w2._backwards()
x1w1._backwards()








print(n)

print(draw_dot(o))


