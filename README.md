# Setting

```python
import numpy as np
import torch
import torch.nn as nn
```

# Assignment 1

## **Using NumPy**

data = x1, w1, w2

```python
x1=np.array([1,2,3])

w1=np.arange(0.1, 1.3, 0.1)
w1 = w1.reshape(3, 4)
w2=np.arange(0.2, 1.0, 0.1)
w2 = w2.reshape(4,2)

def NN(x,w):
  x=np.array(x)
  w=np.array(w)
  output=np.dot(x,w)
  return np.array(output)

def relu(x):
  x=x.tolist()
  for i in x:
    idx=x.index(i)
    if i < 0:
      x[idx]=0
  np.array(x)

  return x

def softmax(x):
    t = np.exp(x)
    t = t / t.sum(axis=0, keepdims=False)
    return t

x1=x1.T
linear_1=NN(x1,w1)
relu=relu(linear_1)
linear_2=NN(relu,w2)
softmax = softmax(linear_2)

print('NumPy : x=[1,2,3]----------------------------------------------------')
print(f'\noutput : \n{softmax}')
```

data = x1, w1, w2

```python
x2=np.array([4,5,6])

w1=np.arange(0.1, 1.3, 0.1)
w1 = w1.reshape(3, 4)
w2=np.arange(0.2, 1.0, 0.1)
w2 = w2.reshape(4,2)

def NN(x,w):
  x=np.array(x)
  w=np.array(w)
  output=np.dot(x,w)
  return np.array(output)

def relu(x):
  x=x.tolist()
  for i in x:
    idx=x.index(i)
    if i < 0:
      x[idx]=0
  np.array(x)

  return x

def softmax(x):
    t = np.exp(x)
    t = t / t.sum(axis=0, keepdims=False)
    return t

x2=x2.T
linear_1=NN(x2,w1)
relu=relu(linear_1)
linear_2=NN(relu,w2)
softmax = softmax(linear_2)

print('NumPy : x=[4,5,6]----------------------------------------------------')
print(f'\noutput : \n{softmax}')
```

## Using PyTorch

data = x1, w1, w2

```python
x2=torch.tensor([[1.],[2.],[3.]],dtype=torch.float32)

w1=torch.arange(0.1, 1.3, 0.1)
w1 = w1.view(3, 4)
w2=torch.arange(0.2, 1.0, 0.1)
w2 = w2.view(4,2)

def NN(x,w):
  output=np.dot(x,w)
  return output

relu = nn.ReLU()
softmax = nn.Softmax(dim=1)

x2=x2.t()
x=torch.tensor(NN(x2,w1))
relu(x)
x=torch.tensor(NN(x,w2))
x_ex=x
x = softmax(x)

print('Pytorch : x=[1,2,3]----------------------------------------------------')
print(f'\noutput : {x}\n')
```

data = x2, w1, w2

```python
x2=torch.tensor([[4.],[5.],[6.]],dtype=torch.float32)

w1=torch.arange(0.1, 1.3, 0.1)
w1 = w1.view(3, 4)
w2=torch.arange(0.2, 1.0, 0.1)
w2 = w2.view(4,2)

def NN(x,w):
  output=np.dot(x,w)
  return output

relu = nn.ReLU()
softmax = nn.Softmax(dim=1)

x2=x2.t()
x=torch.tensor(NN(x2,w1))
relu(x)
x=torch.tensor(NN(x,w2))
x_ex=x
x = softmax(x)

print('Pytorch : x=[4,5,6]----------------------------------------------------')
print(f'\noutput : {x}\n')
```

---

# Assignment 2

## **Using NumPy**

data = x1, w1, w2

```python
#class 정의
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / y.shape[0]

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

class linear:
    def __init__(self, W):
        self.W = W
        self.x = None
        self.dW = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W)

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)

        return dx, self.dW

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftmaxWithloss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

x1 = np.array([[1, 2, 3]])

w1=np.arange(0.1, 1.3, 0.1)
w1 = w1.reshape(3, 4)
w2=np.arange(0.2, 1.0, 0.1)
w2 = w2.reshape(4,2)

y1 = np.array([[0, 1]])

linear1 = linear(w1)
linear2 = linear(w2)
relu1 = Relu()
softmaxWithloss = SoftmaxWithloss()

# 순전파
out1 = linear1.forward(x1)
relu_out1 = relu1.forward(out1)
out2 = linear2.forward(relu_out1)
loss = softmaxWithloss.forward(out2, y1)

# 역전파
dout = softmaxWithloss.backward()
dout1, dw2= linear2.backward(dout)
relu_dout1 = relu1.backward(dout1)
dx, dw1 = linear1.backward(relu_dout1)

print('NumPy : x=[1,2,3]----------------------------------------------------')
print(f'\nloss : {loss}\n')
print(f'w1 gradient : \n{dw1}')
```

data = x2, w1, w2

```python
x2 = np.array([[4, 5, 6]])

w1=np.arange(0.1, 1.3, 0.1)
w1 = w1.reshape(3, 4)
w2=np.arange(0.2, 1.0, 0.1)
w2 = w2.reshape(4,2)

y2 = np.array([[1, 0]])

linear1 = linear(w1)
linear2 = linear(w2)
relu1 = Relu()
softmaxWithloss = SoftmaxWithloss()

# 순전파
out1 = linear1.forward(x2)
relu_out1 = relu1.forward(out1)
out2 = linear2.forward(relu_out1)
loss = softmaxWithloss.forward(out2, y2)

# 역전파
dout = softmaxWithloss.backward()
dout1, dw2= linear2.backward(dout)
relu_dout1 = relu1.backward(dout1)
dx, dw1 = linear1.backward(relu_dout1)

print('NumPy : x=[4,5,6]----------------------------------------------------')
print(f'\nloss : {loss}\n')
print(f'w1 gradient : \n{dw1}')
```

## **Using PyTorch**

data = x1, w1, w2

```python
import torch
import torch.nn as nn

def cross_entropy_error(y, t):
    delta = 1e-7
    return -torch.sum(t * torch.log(y + delta)) / y.shape[0]

def softmax(a):
    c = torch.max(a)
    exp_a = torch.exp(a)
    sum_exp_a = torch.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

class linear:
    def __init__(self, W):
        self.W = W
        self.x = None
        self.dW = None

    def forward(self, x):
        self.x = x
        out = torch.matmul(x, self.W)

        return out

    def backward(self, dout):
        dx = torch.matmul(dout, self.W.T)
        self.dW = torch.matmul(self.x.T, dout)

        return dx, self.dW

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.clone()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftmaxWithloss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

x1=torch.tensor([[1.,2.,3.]],dtype=torch.float32)
y1=torch.tensor([[0.,1.]],dtype=torch.float32)

w1=torch.arange(0.1, 1.3, 0.1)
w1 = w1.view(3, 4)
w2=torch.arange(0.2, 1.0, 0.1)
w2 = w2.view(4,2)

x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
y1 = torch.tensor(y1, dtype=torch.float32, requires_grad=True)
w1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)
w2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)

linear1 = linear(w1)
linear2 = linear(w2)
relu1 = Relu()
softmaxWithloss = SoftmaxWithloss()

# 순전파
out1 = linear1.forward(x1)
relu_out1 = relu1.forward(out1)
out2 = linear2.forward(relu_out1)
loss = softmaxWithloss.forward(out2, y1)

# 역전파
dout = softmaxWithloss.backward()
dout1, dw2= linear2.backward(dout)
relu_dout1 = relu1.backward(dout1)
dx, dw1 = linear1.backward(relu_dout1)

print('Pytorch : x=[1,2,3]----------------------------------------------------')
print(f'\nloss : {loss}\n')
print(f'w1 gradient :\n{dw1}')
```

data = x2, w1, w2

```python
x2=torch.tensor([[4.,5.,6.]],dtype=torch.float32)
y2=torch.tensor([[1.,0.]],dtype=torch.float32)

x2 = torch.tensor(x2, dtype=torch.float32, requires_grad=True)
y2 = torch.tensor(y2, dtype=torch.float32, requires_grad=True)

linear1 = linear(w1)
linear2 = linear(w2)
relu1 = Relu()
softmaxWithloss = SoftmaxWithloss()

# 순전파
out1 = linear1.forward(x2)
relu_out1 = relu1.forward(out1)
out2 = linear2.forward(relu_out1)
loss = softmaxWithloss.forward(out2, y2)

# 역전파
dout = softmaxWithloss.backward()
dout1, dw2= linear2.backward(dout)
relu_dout1 = relu1.backward(dout1)
dx, dw1 = linear1.backward(relu_dout1)

print('Pytorch : x=[4,5,6]----------------------------------------------------')
print(f'\nloss : {loss}\n')
print(f'w1 gradient :\n{dw1}')
```

---

# Assignment 3

## **Using NumPy**

data = x1, w1, w2

```python
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / y.shape[0]

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

class linear:
    def __init__(self, W):
        self.W = W
        self.x = None
        self.dW = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W)

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)

        return dx, self.dW

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftmaxWithloss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

class Dropout:
    def __init__(self, p=0.4):
        self.p = p
        self.mask = None

    def forward(self, x, is_training=True):
        if is_training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        else:
            return x

    def backward(self, grad_output):
        return grad_output * self.mask

x1 = np.array([[1, 2, 3]])

w1=np.arange(0.1, 1.3, 0.1)
w1 = w1.reshape(3, 4)
w2=np.arange(0.2, 1.0, 0.1)
w2 = w2.reshape(4,2)

y1 = np.array([[0, 1]])

num_epochs=100

for epoch in range(num_epochs):

  linear1 = linear(w1)
  linear2 = linear(w2)
  relu1 = Relu()
  softmaxWithloss = SoftmaxWithloss()
  dropout=Dropout()

  #Forward
  out1 = linear1.forward(x1)
  relu_out1 = relu1.forward(out1)
  dropout_out=dropout.forward(relu_out1)
  out2 = linear2.forward(dropout_out)
  loss = softmaxWithloss.forward(out2, y1)

  #Backward
  dout = softmaxWithloss.backward()
  dout1, dw2= linear2.backward(dout)
  dropout_dout=dropout.backward(dout1)
  relu_dout1 = relu1.backward(dropout_dout)
  dx, dw1 = linear1.backward(relu_dout1)

  w1 = w1-0.01*dw1
  w2 = w2-0.01*dw2

  # if (epoch+1) % 100 == 0:
  #   print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss}')

print('NumPy : x=[1,2,3]----------------------------------------------------\n')
print(f'w1 weight : \n{w1}')
print(f'w2 weight : \n{w2}')
```

data = x2, w1, w2

```python
x2 = np.array([[4, 5, 6]])

w1=np.arange(0.1, 1.3, 0.1)
w1 = w1.reshape(3, 4)
w2=np.arange(0.2, 1.0, 0.1)
w2 = w2.reshape(4,2)

y2 = np.array([[1, 0]])

num_epochs=100

for epoch in range(num_epochs):

  linear1 = linear(w1)
  linear2 = linear(w2)
  relu1 = Relu()
  softmaxWithloss = SoftmaxWithloss()
  dropout=Dropout()

  #Forward
  out1 = linear1.forward(x2)
  relu_out1 = relu1.forward(out1)
  dropout_out=dropout.forward(relu_out1)
  out2 = linear2.forward(dropout_out)
  loss = softmaxWithloss.forward(out2, y2)

  #Backward
  dout = softmaxWithloss.backward()
  dout1, dw2= linear2.backward(dout)
  dropout_dout=dropout.backward(dout1)
  relu_dout1 = relu1.backward(dropout_dout)
  dx, dw1 = linear1.backward(relu_dout1)

  w1 = w1-0.01*dw1
  w2 = w2-0.01*dw2

  # if (epoch+1) % 100 == 0:
  #   print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss}')

print('NumPy : x=[4,5,6]----------------------------------------------------\n')
print(f'w1 weight : \n{w1}')
print(f'w2 weight : \n{w2}')
```

## Using PyTorch

data = x1, w1, w2

```python
import torch
import torch.nn as nn

def cross_entropy_error(y, t):
    delta = 1e-7
    return -torch.sum(t * torch.log(y + delta)) / y.shape[0]

def softmax(a):
    c = torch.max(a)
    exp_a = torch.exp(a)
    sum_exp_a = torch.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

class linear:
    def __init__(self, W):
        self.W = W
        self.x = None
        self.dW = None

    def forward(self, x):
        self.x = x
        out = torch.matmul(x, self.W)

        return out

    def backward(self, dout):
        dx = torch.matmul(dout, self.W.T)
        self.dW = torch.matmul(self.x.T, dout)

        return dx, self.dW

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.clone()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftmaxWithloss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

class Dropout:
    def __init__(self, p=0.4):
        self.p = p
        self.mask = None

    def forward(self, x, is_training=True):
        if is_training:
            self.mask = (torch.randn(*x.shape) + 0.5 > self.p) / (1 - self.p)
            return x * self.mask
        else:
            return x

    def backward(self, grad_output):
        return grad_output * self.mask

x1=torch.tensor([[1.,2.,3.]],dtype=torch.float32)
y1=torch.tensor([[0.,1.]],dtype=torch.float32)

w1=torch.arange(0.1, 1.3, 0.1)
w1 = w1.view(3, 4)
w2=torch.arange(0.2, 1.0, 0.1)
w2 = w2.view(4,2)

x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True)
y1 = torch.tensor(y1, dtype=torch.float32, requires_grad=True)
w1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)
w2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)

num_epochs=100

for epoch in range(num_epochs):

  linear1 = linear(w1)
  linear2 = linear(w2)
  relu1 = Relu()
  softmaxWithloss = SoftmaxWithloss()
  dropout=Dropout()

  # 순전파
  out1 = linear1.forward(x1)
  relu_out1 = relu1.forward(out1)
  drop_out= dropout.forward(relu_out1)
  out2 = linear2.forward(drop_out)
  loss = softmaxWithloss.forward(out2, y1)

  # 역전파
  dout = softmaxWithloss.backward()
  dout1, dw2= linear2.backward(dout)
  drop_dout= dropout.backward(dout1)
  relu_dout1 = relu1.backward(drop_dout)
  dx, dw1 = linear1.backward(relu_dout1)

  w1 = w1-0.01*dw1
  w2 = w2-0.01*dw2

  # if (epoch+1) % 100 == 0:
  #   print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss}')

print('Pytorch : x=[1,2,3]----------------------------------------------------\n')
print(f'w1 weight : \n{w1}')
print(f'w2 weight : \n{w2}')
```

data = x2, w1, w2

```python
x2=torch.tensor([[4.,5.,6.]],dtype=torch.float32)
y2=torch.tensor([[1.,0.]],dtype=torch.float32)

w1=torch.arange(0.1, 1.3, 0.1)
w1 = w1.view(3, 4)
w2=torch.arange(0.2, 1.0, 0.1)
w2 = w2.view(4,2)

x2 = torch.tensor(x2, dtype=torch.float32, requires_grad=True)
y2 = torch.tensor(y2, dtype=torch.float32, requires_grad=True)
w1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)
w2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)

num_epochs=100

for epoch in range(num_epochs):

  linear1 = linear(w1)
  linear2 = linear(w2)
  relu1 = Relu()
  softmaxWithloss = SoftmaxWithloss()
  
  # 순전파
  out1 = linear1.forward(x1)
  relu_out1 = relu1.forward(out1)
  drop_out= dropout.forward(relu_out1)
  out2 = linear2.forward(drop_out)
  loss = softmaxWithloss.forward(out2, y1)

  # 역전파
  dout = softmaxWithloss.backward()
  dout1, dw2= linear2.backward(dout)
  drop_dout= dropout.backward(dout1)
  relu_dout1 = relu1.backward(drop_dout)
  dx, dw1 = linear1.backward(relu_dout1)

  w1 = w1-0.01*dw1
  w2 = w2-0.01*dw2

  # if (epoch+1) % 100 == 0:
  #   print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss}')

print('Pytorch : x=[4,5,6]----------------------------------------------------\n')
print(f'w1 weight : \n{w1}')
print(f'w2 weight : \n{w2}')
```
