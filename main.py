from graph import draw_dot

class Value:
    def __init__(self, x, _children=(), _op='') -> None:
        self.x = x
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self) -> str:
        return f"Value(x={self.x})"
    
    def __add__(self, data) -> float:
        equals = Value(self.x + data.x, (self, data), '+')
        return equals
  
    def __mul__(self, data) -> float:
        equals = Value(self.x*data.x, (self, data),'*')
        return equals


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d1 = a*b + c

draw_dot(d1)
# derivative f(x) = lim h->0 (f(x+h) - f (x))/h
#h = 0.001
#f(x+h)
#c.x += h
#d2 = a*b + c

