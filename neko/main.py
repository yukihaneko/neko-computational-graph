import numpy
class var:
 def __init__(self,value):
  self.value = value
  self.grad = None
  self.parents = []

 def backward(self):
  if len(self.parents) != 0:
       for parent in self.parents:
         for inp in parent.inputs:
           if isinstance(self.grad,list):
           	 self.grad = self.grad.pop(0)
           inp.grad = parent.backward(self.grad)
           inp.backward()

class node:
 def __call__(self,*inputs):
  inpus = [inpu.value for inpu in inputs]
  ys = self.forward(*inpus)
  if not isinstance(ys,tuple):
   ys = (ys,)
  outputs = [var(y) for y in ys]
  for output in outputs:
      output.parents.append(self)
  self.inputs = inputs
  self.outputs = outputs
  return outputs if len(outputs)>1 else outputs[0]

 def forward(self,inputs):
  pass

 def backward(self,grad):
  pass

class add(node):
 def forward(self,x,y):
  y = x + y
  return y

 def backward(self,x):
  return [x,x]

class mul(node):
 def forward(self,x,y):
  return x*y

 def backward(self,x):
  return [self.inputs[0].value*x,self.inputs[1].value*x]

class square(node):
 def forward(self,x):
  x = x ** 2
  return x

 def backward(self,xy):
  x = self.inputs[0].value
  y = 2 *x*xy
  return y

class exp(node):
 def forward(self,x):
  y = numpy.exp(x)
  return y

 def backward(self,gy):
  x = self.inputs[0].value
  gx = numpy.exp(x)*gy
  return gx

a = var(numpy.array(2.0))
B = square()
b = B(a)
f = var(numpy.array(6.0))
X = square()
zz = X(f)
S = add()
v = S(b,f)
Y = mul()
y = Y(v,zz)
print(y.value)
y.grad = 1
y.backward()
print(a.grad)
