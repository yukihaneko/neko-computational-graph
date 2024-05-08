import numpy
from main import *
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
