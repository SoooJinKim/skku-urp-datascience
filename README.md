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
