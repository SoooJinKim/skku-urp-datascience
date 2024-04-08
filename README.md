# Assignment 1

- 노드가  3개  있는  입력  레이어
- 4개의  노드와  ReLU  활성화  기능을  갖춘  히든  레이어
- 노드  2개와  소프트맥스  활성화  기능이  있는  출력  레이어
- NumPy와  PyTorch를  모두  사용하여 구현 및 출력값 확인
- do not use bias terms
- y is given

**Code Flow**

1. Building def for each NN
2. Making layers that follows the task
    
      - transpose x matrix to fit the model
      - Layer for linear combination of x and w
      - Non-linearity via activation function
      - Layer for linear combination of output and w
      - Softmax for probability distributions
    

> Numpy with X1, X2
> 

```python
NumPy : x=[1,2,3]----------------------------------------------------

output : 
[0.13238887 0.86761113]

NumPy : x=[4,5,6]----------------------------------------------------

output : 
[0.01448572 0.98551428]
```

> PyTorch with X1, X2
> 

```python
Pytorch : x=[1,2,3]----------------------------------------------------

output : tensor([[0.1324, 0.8676]])

Pytorch : x=[4,5,6]----------------------------------------------------

output : tensor([[0.0145, 0.9855]])
```

---

# Assignment 2

- NN은 assignment 1 사용
- loss function = Cross Entropy(***직접 구현해야 함***)
- w1에 대한 loss gradient 출력
- 학습이나 업데이트 없이 구현
- y is given

**Code Flow**

1. Building def for each NN
2. Making layers that follows the task
    
      - transpose x matrix to fit the model
      - Layer for linear combination of x and w
      - Non-linearity via activation function
      - Layer for linear combination of output and w
      - Softmax for probability distributions
    

> Numpy
> 

```python
NumPy : x=[1,2,3]----------------------------------------------------

loss : 0.14201156044285512

w1 gradient : 
[[-0.01323889 -0.01323889 -0.01323889 -0.01323889]
 [-0.02647777 -0.02647777 -0.02647777 -0.02647777]
 [-0.03971666 -0.03971666 -0.03971666 -0.03971666]]

NumPy : x=[4,5,6]----------------------------------------------------

loss : 4.234584763119424

w1 gradient : 
[[0.39420571 0.39420571 0.39420571 0.39420571]
 [0.49275714 0.49275714 0.49275714 0.49275714]
 [0.59130857 0.59130857 0.59130857 0.59130857]]
```

> PyTorch
> 

```python
Pytorch : x=[1,2,3]----------------------------------------------------

loss : 0.14201155304908752

w1 gradient :
tensor([[-0.0132, -0.0132, -0.0132, -0.0132],
        [-0.0265, -0.0265, -0.0265, -0.0265],
        [-0.0397, -0.0397, -0.0397, -0.0397]], grad_fn=<MmBackward0>)

Pytorch : x=[4,5,6]----------------------------------------------------

loss : 4.234584331512451

w1 gradient :
tensor([[0.3942, 0.3942, 0.3942, 0.3942],
        [0.4928, 0.4928, 0.4928, 0.4928],
        [0.5913, 0.5913, 0.5913, 0.5913]], grad_fn=<MmBackward0>)
```

> Numpy
> 

```python
NumPy : x=[1,2,3]----------------------------------------------------

w1 : 
[[0.10552524 0.20282871 0.303748   0.4042481 ]
 [0.51105049 0.60565741 0.70749601 0.80849619]
 [0.91657573 1.00848612 1.11124401 1.21274429]]
w2 : 
[[0.09421492 0.40578508]
 [0.32560943 0.57439057]
 [0.50190096 0.79809904]
 [0.68486325 1.01513675]]

NumPy : x=[4,5,6]----------------------------------------------------

w1 : 
[[0.10181436 0.2002798  0.29902002 0.39894191]
 [0.50226795 0.60034975 0.69877503 0.79867739]
 [0.90272153 1.0004197  1.09853003 1.19841287]]
w2 : 
[[0.37800336 0.12199664]
 [0.57961347 0.32038653]
 [0.78313149 0.51686851]
 [1.04584536 0.65415464]]
```

---

## Assignment 3

- NN은 assignment 2 사용
- epoch = 100
- drop out = 0.4
- learing_rate = 0.01
- y is given

**Code Flow**

1. Building def for each NN
2. Making layers that follows the task
    
      - transpose x matrix to fit the model
      - Layer for linear combination of x and w
      - Non-linearity via activation function
      - Layer for linear combination of output and w
      - Softmax for probability distributions
    

> PyTorch
> 

```python
Pytorch : x=[1,2,3]----------------------------------------------------

w1 : 
tensor([[0.1031, 0.2033, 0.3035, 0.4037],
        [0.5062, 0.6066, 0.7070, 0.8074],
        [0.9093, 1.0099, 1.1105, 1.2111]], grad_fn=<SubBackward0>)
w2 : 
tensor([[0.1299, 0.3701],
        [0.3188, 0.5812],
        [0.5078, 0.7922],
        [0.6968, 1.0032]], grad_fn=<SubBackward0>)

Pytorch : x=[4,5,6]--------------------------------------------------
w1 : 
tensor([[0.0969, 0.1972, 0.2975, 0.3978],
        [0.4961, 0.5965, 0.6969, 0.7973],
        [0.8953, 0.9958, 1.0962, 1.1967]], grad_fn=<SubBackward0>)
w2 : 
tensor([[0.3016, 0.1984],
        [0.5200, 0.3800],
        [0.7384, 0.5616],
        [0.9568, 0.7432]], grad_fn=<SubBackward0>)
```
