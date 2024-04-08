{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "Y_NnpeCQouV0"
      ],
      "toc_visible": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "##Setting"
      ],
      "metadata": {
        "id": "ah0MOoKPfpvZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import torch\n",
        "import torch.nn as nn"
      ],
      "metadata": {
        "id": "-_QlXWB_voDn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Assignment 1\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "928FXQ7kfdlq"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Using NumPy(DONE)"
      ],
      "metadata": {
        "id": "X4fmES2NyMLN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x1"
      ],
      "metadata": {
        "id": "PSCxuGXKkUAl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x1=np.array([1,2,3])\n",
        "\n",
        "w1=np.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.reshape(3, 4)\n",
        "w2=np.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.reshape(4,2)\n",
        "\n",
        "\n",
        "def NN(x,w):\n",
        "  x=np.array(x)\n",
        "  w=np.array(w)\n",
        "  output=np.dot(x,w)\n",
        "  return np.array(output)\n",
        "\n",
        "def relu(x):\n",
        "  x=x.tolist()\n",
        "  for i in x:\n",
        "    idx=x.index(i)\n",
        "    if i < 0:\n",
        "      x[idx]=0\n",
        "  np.array(x)\n",
        "\n",
        "  return x\n",
        "\n",
        "def softmax(x):\n",
        "    t = np.exp(x)\n",
        "    t = t / t.sum(axis=0, keepdims=False)\n",
        "    return t\n",
        "\n",
        "x1=x1.T\n",
        "linear_1=NN(x1,w1)\n",
        "relu=relu(linear_1)\n",
        "linear_2=NN(relu,w2)\n",
        "softmax = softmax(linear_2)\n",
        "\n",
        "\n",
        "print('NumPy : x=[1,2,3]----------------------------------------------------')\n",
        "print(f'\\noutput : \\n{softmax}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LPm0wG4Gu1Ug",
        "outputId": "3cdec135-1e58-4307-ad07-e00f9ba820a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "NumPy : x=[1,2,3]----------------------------------------------------\n",
            "\n",
            "output : \n",
            "[0.13238887 0.86761113]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x2"
      ],
      "metadata": {
        "id": "wDrLfdIvkVDx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x2=np.array([4,5,6])\n",
        "\n",
        "w1=np.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.reshape(3, 4)\n",
        "w2=np.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.reshape(4,2)\n",
        "\n",
        "\n",
        "def NN(x,w):\n",
        "  x=np.array(x)\n",
        "  w=np.array(w)\n",
        "  output=np.dot(x,w)\n",
        "  return np.array(output)\n",
        "\n",
        "def relu(x):\n",
        "  x=x.tolist()\n",
        "  for i in x:\n",
        "    idx=x.index(i)\n",
        "    if i < 0:\n",
        "      x[idx]=0\n",
        "  np.array(x)\n",
        "\n",
        "  return x\n",
        "\n",
        "def softmax(x):\n",
        "    t = np.exp(x)\n",
        "    t = t / t.sum(axis=0, keepdims=False)\n",
        "    return t\n",
        "\n",
        "x2=x2.T\n",
        "linear_1=NN(x2,w1)\n",
        "relu=relu(linear_1)\n",
        "linear_2=NN(relu,w2)\n",
        "softmax = softmax(linear_2)\n",
        "\n",
        "print('NumPy : x=[4,5,6]----------------------------------------------------')\n",
        "print(f'\\noutput : \\n{softmax}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pxCvvkN-kd5q",
        "outputId": "03ba642e-b21c-4e4d-844a-a4b083afb671"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "NumPy : x=[4,5,6]----------------------------------------------------\n",
            "\n",
            "output : \n",
            "[0.01448572 0.98551428]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Using PyTorch(DONE)"
      ],
      "metadata": {
        "id": "_weTvoNMyQlx"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x1"
      ],
      "metadata": {
        "id": "IAPJULtYlr5p"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x2=torch.tensor([[1.],[2.],[3.]],dtype=torch.float32)\n",
        "\n",
        "w1=torch.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.view(3, 4)\n",
        "w2=torch.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.view(4,2)\n",
        "\n",
        "def NN(x,w):\n",
        "  output=np.dot(x,w)\n",
        "  return output\n",
        "\n",
        "\n",
        "relu = nn.ReLU()\n",
        "softmax = nn.Softmax(dim=1)\n",
        "\n",
        "x2=x2.t()\n",
        "x=torch.tensor(NN(x2,w1))\n",
        "relu(x)\n",
        "x=torch.tensor(NN(x,w2))\n",
        "x_ex=x\n",
        "x = softmax(x)\n",
        "\n",
        "print('Pytorch : x=[1,2,3]----------------------------------------------------')\n",
        "print(f'\\noutput : {x}\\n')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z70sV30Dl139",
        "outputId": "7683d526-004f-4b75-d096-f83a0e148533"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Pytorch : x=[1,2,3]----------------------------------------------------\n",
            "\n",
            "output : tensor([[0.1324, 0.8676]])\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x2"
      ],
      "metadata": {
        "id": "oNaVysjqlz9T"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x2=torch.tensor([[4.],[5.],[6.]],dtype=torch.float32)\n",
        "\n",
        "w1=torch.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.view(3, 4)\n",
        "w2=torch.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.view(4,2)\n",
        "\n",
        "def NN(x,w):\n",
        "  output=np.dot(x,w)\n",
        "  return output\n",
        "\n",
        "\n",
        "relu = nn.ReLU()\n",
        "softmax = nn.Softmax(dim=1)\n",
        "\n",
        "x2=x2.t()\n",
        "x=torch.tensor(NN(x2,w1))\n",
        "relu(x)\n",
        "x=torch.tensor(NN(x,w2))\n",
        "x_ex=x\n",
        "x = softmax(x)\n",
        "\n",
        "print('Pytorch : x=[4,5,6]----------------------------------------------------')\n",
        "print(f'\\noutput : {x}\\n')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qN-b-pV22uQs",
        "outputId": "6492d0ca-40ad-4f4e-872d-1d03c474fd73"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Pytorch : x=[4,5,6]----------------------------------------------------\n",
            "\n",
            "output : tensor([[0.0145, 0.9855]])\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Assignment 2"
      ],
      "metadata": {
        "id": "wMsy1tz86mMA"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Using NumPy(DONE)"
      ],
      "metadata": {
        "id": "Om9LXOjd6mMB"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "class 정의"
      ],
      "metadata": {
        "id": "BZoBWgH0chN2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "def cross_entropy_error(y, t):\n",
        "    delta = 1e-7\n",
        "    return -np.sum(t * np.log(y + delta)) / y.shape[0]\n",
        "\n",
        "def softmax(a):\n",
        "    c = np.max(a)\n",
        "    exp_a = np.exp(a)\n",
        "    sum_exp_a = np.sum(exp_a)\n",
        "    y = exp_a / sum_exp_a\n",
        "\n",
        "    return y\n",
        "\n",
        "\n",
        "class linear:\n",
        "    def __init__(self, W):\n",
        "        self.W = W\n",
        "        self.x = None\n",
        "        self.dW = None\n",
        "\n",
        "    def forward(self, x):\n",
        "        self.x = x\n",
        "        out = np.dot(x, self.W)\n",
        "\n",
        "        return out\n",
        "\n",
        "    def backward(self, dout):\n",
        "        dx = np.dot(dout, self.W.T)\n",
        "        self.dW = np.dot(self.x.T, dout)\n",
        "\n",
        "        return dx, self.dW\n",
        "\n",
        "\n",
        "class Relu:\n",
        "    def __init__(self):\n",
        "        self.mask = None\n",
        "\n",
        "    def forward(self, x):\n",
        "        self.mask = (x <= 0)\n",
        "        out = x.copy()\n",
        "        out[self.mask] = 0\n",
        "        return out\n",
        "\n",
        "    def backward(self, dout):\n",
        "        dout[self.mask] = 0\n",
        "        dx = dout\n",
        "        return dx\n",
        "\n",
        "class SoftmaxWithloss:\n",
        "    def __init__(self):\n",
        "        self.loss = None\n",
        "        self.y = None\n",
        "        self.t = None\n",
        "\n",
        "    def forward(self, x, t):\n",
        "        self.t = t\n",
        "        self.y = softmax(x)\n",
        "        self.loss = cross_entropy_error(self.y, self.t)\n",
        "\n",
        "        return self.loss\n",
        "\n",
        "    def backward(self, dout=1):\n",
        "        batch_size = self.t.shape[0]\n",
        "        dx = (self.y - self.t) / batch_size\n",
        "\n",
        "        return dx"
      ],
      "metadata": {
        "id": "tfP1KRAhcgLM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x1 → y1"
      ],
      "metadata": {
        "id": "ue9Tvo8cedu7"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "variable 정의"
      ],
      "metadata": {
        "id": "S_tRcn08cnZs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x1 = np.array([[1, 2, 3]])\n",
        "\n",
        "w1=np.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.reshape(3, 4)\n",
        "w2=np.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.reshape(4,2)\n",
        "\n",
        "y1 = np.array([[0, 1]])"
      ],
      "metadata": {
        "id": "IWCihD-scmvi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "linear1 = linear(w1)\n",
        "linear2 = linear(w2)\n",
        "relu1 = Relu()\n",
        "softmaxWithloss = SoftmaxWithloss()\n",
        "\n",
        "# 순전파\n",
        "out1 = linear1.forward(x1)\n",
        "relu_out1 = relu1.forward(out1)\n",
        "out2 = linear2.forward(relu_out1)\n",
        "loss = softmaxWithloss.forward(out2, y1)\n",
        "\n",
        "# 역전파\n",
        "dout = softmaxWithloss.backward()\n",
        "dout1, dw2= linear2.backward(dout)\n",
        "relu_dout1 = relu1.backward(dout1)\n",
        "dx, dw1 = linear1.backward(relu_dout1)\n",
        "\n",
        "print('NumPy : x=[1,2,3]----------------------------------------------------')\n",
        "print(f'\\nloss : {loss}\\n')\n",
        "print(f'w1 gradient : \\n{dw1}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yxqRyY8g6WX6",
        "outputId": "31b54994-89c8-4f63-f8e7-c44859bfbfad"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "NumPy : x=[1,2,3]----------------------------------------------------\n",
            "\n",
            "loss : 0.14201156044285512\n",
            "\n",
            "w1 gradient : \n",
            "[[-0.01323889 -0.01323889 -0.01323889 -0.01323889]\n",
            " [-0.02647777 -0.02647777 -0.02647777 -0.02647777]\n",
            " [-0.03971666 -0.03971666 -0.03971666 -0.03971666]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x2 → y2"
      ],
      "metadata": {
        "id": "pYT1QNxMenw4"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "variable 정의"
      ],
      "metadata": {
        "id": "YH5H-sCmeZ_D"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x2 = np.array([[4, 5, 6]])\n",
        "\n",
        "w1=np.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.reshape(3, 4)\n",
        "w2=np.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.reshape(4,2)\n",
        "\n",
        "y2 = np.array([[1, 0]])"
      ],
      "metadata": {
        "id": "t9P8cXYseZhJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "linear1 = linear(w1)\n",
        "linear2 = linear(w2)\n",
        "relu1 = Relu()\n",
        "softmaxWithloss = SoftmaxWithloss()\n",
        "\n",
        "# 순전파\n",
        "out1 = linear1.forward(x2)\n",
        "relu_out1 = relu1.forward(out1)\n",
        "out2 = linear2.forward(relu_out1)\n",
        "loss = softmaxWithloss.forward(out2, y2)\n",
        "\n",
        "# 역전파\n",
        "dout = softmaxWithloss.backward()\n",
        "dout1, dw2= linear2.backward(dout)\n",
        "relu_dout1 = relu1.backward(dout1)\n",
        "dx, dw1 = linear1.backward(relu_dout1)\n",
        "\n",
        "print('NumPy : x=[4,5,6]----------------------------------------------------')\n",
        "print(f'\\nloss : {loss}\\n')\n",
        "print(f'w1 gradient : \\n{dw1}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NC6L0KcZp1mD",
        "outputId": "96b441b1-8ff7-4818-e332-a4f3333a791f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "NumPy : x=[4,5,6]----------------------------------------------------\n",
            "\n",
            "loss : 4.234584763119424\n",
            "\n",
            "w1 gradient : \n",
            "[[0.39420571 0.39420571 0.39420571 0.39420571]\n",
            " [0.49275714 0.49275714 0.49275714 0.49275714]\n",
            " [0.59130857 0.59130857 0.59130857 0.59130857]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Using PyTorch(DONE)"
      ],
      "metadata": {
        "id": "TL4tE49d6mMC"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "class 정의"
      ],
      "metadata": {
        "id": "jPtKj4wghGni"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "\n",
        "def cross_entropy_error(y, t):\n",
        "    delta = 1e-7\n",
        "    return -torch.sum(t * torch.log(y + delta)) / y.shape[0]\n",
        "\n",
        "def softmax(a):\n",
        "    c = torch.max(a)\n",
        "    exp_a = torch.exp(a)\n",
        "    sum_exp_a = torch.sum(exp_a)\n",
        "    y = exp_a / sum_exp_a\n",
        "\n",
        "    return y\n",
        "\n",
        "\n",
        "class linear:\n",
        "    def __init__(self, W):\n",
        "        self.W = W\n",
        "        self.x = None\n",
        "        self.dW = None\n",
        "\n",
        "    def forward(self, x):\n",
        "        self.x = x\n",
        "        out = torch.matmul(x, self.W)\n",
        "\n",
        "        return out\n",
        "\n",
        "    def backward(self, dout):\n",
        "        dx = torch.matmul(dout, self.W.T)\n",
        "        self.dW = torch.matmul(self.x.T, dout)\n",
        "\n",
        "        return dx, self.dW\n",
        "\n",
        "\n",
        "class Relu:\n",
        "    def __init__(self):\n",
        "        self.mask = None\n",
        "\n",
        "    def forward(self, x):\n",
        "        self.mask = (x <= 0)\n",
        "        out = x.clone()\n",
        "        out[self.mask] = 0\n",
        "        return out\n",
        "\n",
        "    def backward(self, dout):\n",
        "        dout[self.mask] = 0\n",
        "        dx = dout\n",
        "        return dx\n",
        "\n",
        "class SoftmaxWithloss:\n",
        "    def __init__(self):\n",
        "        self.loss = None\n",
        "        self.y = None\n",
        "        self.t = None\n",
        "\n",
        "    def forward(self, x, t):\n",
        "        self.t = t\n",
        "        self.y = softmax(x)\n",
        "        self.loss = cross_entropy_error(self.y, self.t)\n",
        "\n",
        "        return self.loss\n",
        "\n",
        "    def backward(self, dout=1):\n",
        "        batch_size = self.t.shape[0]\n",
        "        dx = (self.y - self.t) / batch_size\n",
        "\n",
        "        return dx"
      ],
      "metadata": {
        "id": "jkmsKxD3hEsH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x1 → y1"
      ],
      "metadata": {
        "id": "3Vyx2c2dfZIs"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x1=torch.tensor([[1.,2.,3.]],dtype=torch.float32)\n",
        "y1=torch.tensor([[0.,1.]],dtype=torch.float32)\n",
        "\n",
        "w1=torch.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.view(3, 4)\n",
        "w2=torch.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.view(4,2)\n",
        "\n",
        "x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True)\n",
        "y1 = torch.tensor(y1, dtype=torch.float32, requires_grad=True)\n",
        "w1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)\n",
        "w2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bW0Wh4fcfkVV",
        "outputId": "b0a29a5d-c401-45f8-b60b-e0a9313563d6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-12-2170b64d7e54>:9: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-12-2170b64d7e54>:10: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  y1 = torch.tensor(y1, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-12-2170b64d7e54>:11: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  w1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-12-2170b64d7e54>:12: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  w2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "linear1 = linear(w1)\n",
        "linear2 = linear(w2)\n",
        "relu1 = Relu()\n",
        "softmaxWithloss = SoftmaxWithloss()\n",
        "\n",
        "# 순전파\n",
        "out1 = linear1.forward(x1)\n",
        "relu_out1 = relu1.forward(out1)\n",
        "out2 = linear2.forward(relu_out1)\n",
        "loss = softmaxWithloss.forward(out2, y1)\n",
        "\n",
        "# 역전파\n",
        "dout = softmaxWithloss.backward()\n",
        "dout1, dw2= linear2.backward(dout)\n",
        "relu_dout1 = relu1.backward(dout1)\n",
        "dx, dw1 = linear1.backward(relu_dout1)\n",
        "\n",
        "print('Pytorch : x=[1,2,3]----------------------------------------------------')\n",
        "print(f'\\nloss : {loss}\\n')\n",
        "print(f'w1 gradient :\\n{dw1}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JewrlRqJ5zbq",
        "outputId": "5716e283-d084-4591-c652-d56b09e36736"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Pytorch : x=[1,2,3]----------------------------------------------------\n",
            "\n",
            "loss : 0.14201155304908752\n",
            "\n",
            "w1 gradient :\n",
            "tensor([[-0.0132, -0.0132, -0.0132, -0.0132],\n",
            "        [-0.0265, -0.0265, -0.0265, -0.0265],\n",
            "        [-0.0397, -0.0397, -0.0397, -0.0397]], grad_fn=<MmBackward0>)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x2 → y2"
      ],
      "metadata": {
        "id": "jZmMZE8Rjk5j"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x2=torch.tensor([[4.,5.,6.]],dtype=torch.float32)\n",
        "y2=torch.tensor([[1.,0.]],dtype=torch.float32)\n",
        "\n",
        "x2 = torch.tensor(x2, dtype=torch.float32, requires_grad=True)\n",
        "y2 = torch.tensor(y2, dtype=torch.float32, requires_grad=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FyKO37mWjiYk",
        "outputId": "18947cd6-b2d3-454c-a9b9-e6b531b4582e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-14-77d6583759b8>:4: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  x2 = torch.tensor(x2, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-14-77d6583759b8>:5: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  y2 = torch.tensor(y2, dtype=torch.float32, requires_grad=True)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "linear1 = linear(w1)\n",
        "linear2 = linear(w2)\n",
        "relu1 = Relu()\n",
        "softmaxWithloss = SoftmaxWithloss()\n",
        "\n",
        "# 순전파\n",
        "out1 = linear1.forward(x2)\n",
        "relu_out1 = relu1.forward(out1)\n",
        "out2 = linear2.forward(relu_out1)\n",
        "loss = softmaxWithloss.forward(out2, y2)\n",
        "\n",
        "# 역전파\n",
        "dout = softmaxWithloss.backward()\n",
        "dout1, dw2= linear2.backward(dout)\n",
        "relu_dout1 = relu1.backward(dout1)\n",
        "dx, dw1 = linear1.backward(relu_dout1)\n",
        "\n",
        "print('Pytorch : x=[4,5,6]----------------------------------------------------')\n",
        "print(f'\\nloss : {loss}\\n')\n",
        "print(f'w1 gradient :\\n{dw1}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rboozR7AjvG2",
        "outputId": "63fb790c-5531-443c-84a6-33f3794bd0e0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Pytorch : x=[4,5,6]----------------------------------------------------\n",
            "\n",
            "loss : 4.234584331512451\n",
            "\n",
            "w1 gradient :\n",
            "tensor([[0.3942, 0.3942, 0.3942, 0.3942],\n",
            "        [0.4928, 0.4928, 0.4928, 0.4928],\n",
            "        [0.5913, 0.5913, 0.5913, 0.5913]], grad_fn=<MmBackward0>)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Assignment 3"
      ],
      "metadata": {
        "id": "G25uTdHK9yyl"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Using NumPy"
      ],
      "metadata": {
        "id": "KySSD9Sb904d"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x1 → y1"
      ],
      "metadata": {
        "id": "3I65j3pWrJX2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "class 정의"
      ],
      "metadata": {
        "id": "G2o6tyx4reoD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "def cross_entropy_error(y, t):\n",
        "    delta = 1e-7\n",
        "    return -np.sum(t * np.log(y + delta)) / y.shape[0]\n",
        "\n",
        "def softmax(a):\n",
        "    c = np.max(a)\n",
        "    exp_a = np.exp(a)\n",
        "    sum_exp_a = np.sum(exp_a)\n",
        "    y = exp_a / sum_exp_a\n",
        "\n",
        "    return y\n",
        "\n",
        "\n",
        "class linear:\n",
        "    def __init__(self, W):\n",
        "        self.W = W\n",
        "        self.x = None\n",
        "        self.dW = None\n",
        "\n",
        "    def forward(self, x):\n",
        "        self.x = x\n",
        "        out = np.dot(x, self.W)\n",
        "\n",
        "        return out\n",
        "\n",
        "    def backward(self, dout):\n",
        "        dx = np.dot(dout, self.W.T)\n",
        "        self.dW = np.dot(self.x.T, dout)\n",
        "\n",
        "        return dx, self.dW\n",
        "\n",
        "\n",
        "class Relu:\n",
        "    def __init__(self):\n",
        "        self.mask = None\n",
        "\n",
        "    def forward(self, x):\n",
        "        self.mask = (x <= 0)\n",
        "        out = x.copy()\n",
        "        out[self.mask] = 0\n",
        "        return out\n",
        "\n",
        "    def backward(self, dout):\n",
        "        dout[self.mask] = 0\n",
        "        dx = dout\n",
        "        return dx\n",
        "\n",
        "class SoftmaxWithloss:\n",
        "    def __init__(self):\n",
        "        self.loss = None\n",
        "        self.y = None\n",
        "        self.t = None\n",
        "\n",
        "    def forward(self, x, t):\n",
        "        self.t = t\n",
        "        self.y = softmax(x)\n",
        "        self.loss = cross_entropy_error(self.y, self.t)\n",
        "\n",
        "        return self.loss\n",
        "\n",
        "    def backward(self, dout=1):\n",
        "        batch_size = self.t.shape[0]\n",
        "        dx = (self.y - self.t) / batch_size\n",
        "\n",
        "        return dx\n",
        "\n",
        "class Dropout:\n",
        "    def __init__(self, p=0.4):\n",
        "        self.p = p\n",
        "        self.mask = None\n",
        "\n",
        "    def forward(self, x, is_training=True):\n",
        "        if is_training:\n",
        "            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)\n",
        "            return x * self.mask\n",
        "        else:\n",
        "            return x\n",
        "\n",
        "    def backward(self, grad_output):\n",
        "        return grad_output * self.mask\n",
        "\n"
      ],
      "metadata": {
        "id": "37TcWOp9mCGw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "variable 정의"
      ],
      "metadata": {
        "id": "RNgO5w-7rZEC"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x1 = np.array([[1, 2, 3]])\n",
        "\n",
        "w1=np.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.reshape(3, 4)\n",
        "w2=np.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.reshape(4,2)\n",
        "\n",
        "y1 = np.array([[0, 1]])"
      ],
      "metadata": {
        "id": "Bxf97gwTrYpi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "num_epochs=100\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "\n",
        "  linear1 = linear(w1)\n",
        "  linear2 = linear(w2)\n",
        "  relu1 = Relu()\n",
        "  softmaxWithloss = SoftmaxWithloss()\n",
        "  dropout=Dropout()\n",
        "\n",
        "  #Forward\n",
        "  out1 = linear1.forward(x1)\n",
        "  relu_out1 = relu1.forward(out1)\n",
        "  dropout_out=dropout.forward(relu_out1)\n",
        "  out2 = linear2.forward(dropout_out)\n",
        "  loss = softmaxWithloss.forward(out2, y1)\n",
        "\n",
        "  #Backward\n",
        "  dout = softmaxWithloss.backward()\n",
        "  dout1, dw2= linear2.backward(dout)\n",
        "  dropout_dout=dropout.backward(dout1)\n",
        "  relu_dout1 = relu1.backward(dropout_dout)\n",
        "  dx, dw1 = linear1.backward(relu_dout1)\n",
        "\n",
        "  w1 = w1-0.01*dw1\n",
        "  w2 = w2-0.01*dw2\n",
        "\n",
        "  # if (epoch+1) % 100 == 0:\n",
        "  #   print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss}')\n",
        "\n",
        "print('NumPy : x=[1,2,3]----------------------------------------------------\\n')\n",
        "print(f'w1 weight : \\n{w1}')\n",
        "print(f'w2 weight : \\n{w2}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5k_f6rA2rjMx",
        "outputId": "72bf4624-dd26-4be1-87cf-eed7d97c999f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "NumPy : x=[1,2,3]----------------------------------------------------\n",
            "\n",
            "w1 weight : \n",
            "[[0.07987577 0.18054097 0.27474415 0.37872284]\n",
            " [0.46657732 0.57176317 0.66517847 0.77560886]\n",
            " [0.85327886 0.96298537 1.05561279 1.17249488]]\n",
            "w2 weight : \n",
            "[[0.14373491 0.35626509]\n",
            " [0.27786081 0.62213919]\n",
            " [0.46648922 0.83351078]\n",
            " [0.67708954 1.02291046]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x2 → y2"
      ],
      "metadata": {
        "id": "FWzkGwvKrRIx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x2 = np.array([[4, 5, 6]])\n",
        "\n",
        "w1=np.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.reshape(3, 4)\n",
        "w2=np.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.reshape(4,2)\n",
        "\n",
        "y2 = np.array([[1, 0]])"
      ],
      "metadata": {
        "id": "kaxyEgdkv9Mg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "num_epochs=100\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "\n",
        "  linear1 = linear(w1)\n",
        "  linear2 = linear(w2)\n",
        "  relu1 = Relu()\n",
        "  softmaxWithloss = SoftmaxWithloss()\n",
        "  dropout=Dropout()\n",
        "\n",
        "  #Forward\n",
        "  out1 = linear1.forward(x2)\n",
        "  relu_out1 = relu1.forward(out1)\n",
        "  dropout_out=dropout.forward(relu_out1)\n",
        "  out2 = linear2.forward(dropout_out)\n",
        "  loss = softmaxWithloss.forward(out2, y2)\n",
        "\n",
        "  #Backward\n",
        "  dout = softmaxWithloss.backward()\n",
        "  dout1, dw2= linear2.backward(dout)\n",
        "  dropout_dout=dropout.backward(dout1)\n",
        "  relu_dout1 = relu1.backward(dropout_dout)\n",
        "  dx, dw1 = linear1.backward(relu_dout1)\n",
        "\n",
        "  w1 = w1-0.01*dw1\n",
        "  w2 = w2-0.01*dw2\n",
        "\n",
        "  # if (epoch+1) % 100 == 0:\n",
        "  #   print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss}')\n",
        "\n",
        "print('NumPy : x=[4,5,6]----------------------------------------------------\\n')\n",
        "print(f'w1 weight : \\n{w1}')\n",
        "print(f'w2 weight : \\n{w2}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L46jRApKv7Dy",
        "outputId": "2492d2d4-951e-4da7-c320-e03d202855cb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "NumPy : x=[4,5,6]----------------------------------------------------\n",
            "\n",
            "w1 weight : \n",
            "[[0.08771028 0.17917451 0.28014444 0.37782504]\n",
            " [0.48224633 0.56903025 0.67597905 0.77381325]\n",
            " [0.87678239 0.958886   1.07181366 1.16980147]]\n",
            "w2 weight : \n",
            "[[0.42651766 0.07348234]\n",
            " [0.5661757  0.3338243 ]\n",
            " [0.85119679 0.44880321]\n",
            " [0.95300839 0.74699161]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Using PyTorch(DONE)"
      ],
      "metadata": {
        "id": "0jNKiB9H94LW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x1 → y1"
      ],
      "metadata": {
        "id": "JBM_AObzwPMF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "class 정의"
      ],
      "metadata": {
        "id": "6DJgTFS1wUpD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "\n",
        "def cross_entropy_error(y, t):\n",
        "    delta = 1e-7\n",
        "    return -torch.sum(t * torch.log(y + delta)) / y.shape[0]\n",
        "\n",
        "def softmax(a):\n",
        "    c = torch.max(a)\n",
        "    exp_a = torch.exp(a)\n",
        "    sum_exp_a = torch.sum(exp_a)\n",
        "    y = exp_a / sum_exp_a\n",
        "\n",
        "    return y\n",
        "\n",
        "\n",
        "class linear:\n",
        "    def __init__(self, W):\n",
        "        self.W = W\n",
        "        self.x = None\n",
        "        self.dW = None\n",
        "\n",
        "    def forward(self, x):\n",
        "        self.x = x\n",
        "        out = torch.matmul(x, self.W)\n",
        "\n",
        "        return out\n",
        "\n",
        "    def backward(self, dout):\n",
        "        dx = torch.matmul(dout, self.W.T)\n",
        "        self.dW = torch.matmul(self.x.T, dout)\n",
        "\n",
        "        return dx, self.dW\n",
        "\n",
        "\n",
        "class Relu:\n",
        "    def __init__(self):\n",
        "        self.mask = None\n",
        "\n",
        "    def forward(self, x):\n",
        "        self.mask = (x <= 0)\n",
        "        out = x.clone()\n",
        "        out[self.mask] = 0\n",
        "        return out\n",
        "\n",
        "    def backward(self, dout):\n",
        "        dout[self.mask] = 0\n",
        "        dx = dout\n",
        "        return dx\n",
        "\n",
        "class SoftmaxWithloss:\n",
        "    def __init__(self):\n",
        "        self.loss = None\n",
        "        self.y = None\n",
        "        self.t = None\n",
        "\n",
        "    def forward(self, x, t):\n",
        "        self.t = t\n",
        "        self.y = softmax(x)\n",
        "        self.loss = cross_entropy_error(self.y, self.t)\n",
        "\n",
        "        return self.loss\n",
        "\n",
        "    def backward(self, dout=1):\n",
        "        batch_size = self.t.shape[0]\n",
        "        dx = (self.y - self.t) / batch_size\n",
        "\n",
        "        return dx"
      ],
      "metadata": {
        "id": "6kmUxJmOwOz3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "variable 정의"
      ],
      "metadata": {
        "id": "jCCi5FvVwZax"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x1=torch.tensor([[1.,2.,3.]],dtype=torch.float32)\n",
        "y1=torch.tensor([[0.,1.]],dtype=torch.float32)\n",
        "\n",
        "w1=torch.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.view(3, 4)\n",
        "w2=torch.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.view(4,2)\n",
        "\n",
        "x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True)\n",
        "y1 = torch.tensor(y1, dtype=torch.float32, requires_grad=True)\n",
        "w1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)\n",
        "w2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KDkbM8sPwcCf",
        "outputId": "4e2fc9f6-1f1d-4fca-ff5b-872d0e83f79d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-22-2170b64d7e54>:9: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-22-2170b64d7e54>:10: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  y1 = torch.tensor(y1, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-22-2170b64d7e54>:11: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  w1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-22-2170b64d7e54>:12: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  w2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "num_epochs=100\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "\n",
        "  linear1 = linear(w1)\n",
        "  linear2 = linear(w2)\n",
        "  relu1 = Relu()\n",
        "  softmaxWithloss = SoftmaxWithloss()\n",
        "\n",
        "  # 순전파\n",
        "  out1 = linear1.forward(x1)\n",
        "  relu_out1 = relu1.forward(out1)\n",
        "  out2 = linear2.forward(relu_out1)\n",
        "  loss = softmaxWithloss.forward(out2, y1)\n",
        "\n",
        "  # 역전파\n",
        "  dout = softmaxWithloss.backward()\n",
        "  dout1, dw2= linear2.backward(dout)\n",
        "  relu_dout1 = relu1.backward(dout1)\n",
        "  dx, dw1 = linear1.backward(relu_dout1)\n",
        "\n",
        "  w1 = w1-0.01*dw1\n",
        "  w2 = w2-0.01*dw2\n",
        "\n",
        "  # if (epoch+1) % 100 == 0:\n",
        "  #   print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss}')\n",
        "\n",
        "print('Pytorch : x=[1,2,3]----------------------------------------------------\\n')\n",
        "print(f'w1 weight : \\n{w1}')\n",
        "print(f'w2 weight : \\n{w2}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ke6SRZzNe22u",
        "outputId": "7d841968-f7c2-4df3-c0cb-97b01fa81b0f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Pytorch : x=[1,2,3]----------------------------------------------------\n",
            "\n",
            "w1 weight : \n",
            "tensor([[0.0903, 0.1892, 0.2881, 0.3871],\n",
            "        [0.4919, 0.5895, 0.6870, 0.7845],\n",
            "        [0.8935, 0.9897, 1.0858, 1.1820]], grad_fn=<SubBackward0>)\n",
            "w2 weight : \n",
            "tensor([[0.1211, 0.3789],\n",
            "        [0.3150, 0.5850],\n",
            "        [0.5088, 0.7912],\n",
            "        [0.7027, 0.9973]], grad_fn=<SubBackward0>)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### x2 → y2"
      ],
      "metadata": {
        "id": "tnYFM7eYyHyK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x2=torch.tensor([[4.,5.,6.]],dtype=torch.float32)\n",
        "y2=torch.tensor([[1.,0.]],dtype=torch.float32)\n",
        "\n",
        "w1=torch.arange(0.1, 1.3, 0.1)\n",
        "w1 = w1.view(3, 4)\n",
        "w2=torch.arange(0.2, 1.0, 0.1)\n",
        "w2 = w2.view(4,2)\n",
        "\n",
        "x2 = torch.tensor(x2, dtype=torch.float32, requires_grad=True)\n",
        "y2 = torch.tensor(y2, dtype=torch.float32, requires_grad=True)\n",
        "w1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)\n",
        "w2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H5epRe6iyDP5",
        "outputId": "28af543c-b3e4-402b-c8e8-863bfdab9c07"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-24-43104bb198e2>:9: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  x2 = torch.tensor(x2, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-24-43104bb198e2>:10: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  y2 = torch.tensor(y2, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-24-43104bb198e2>:11: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  w1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)\n",
            "<ipython-input-24-43104bb198e2>:12: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
            "  w2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "num_epochs=100\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "\n",
        "  linear1 = linear(w1)\n",
        "  linear2 = linear(w2)\n",
        "  relu1 = Relu()\n",
        "  softmaxWithloss = SoftmaxWithloss()\n",
        "\n",
        "  # 순전파\n",
        "  out1 = linear1.forward(x2)\n",
        "  relu_out1 = relu1.forward(out1)\n",
        "  out2 = linear2.forward(relu_out1)\n",
        "  loss = softmaxWithloss.forward(out2, y2)\n",
        "\n",
        "  # 역전파\n",
        "  dout = softmaxWithloss.backward()\n",
        "  dout1, dw2= linear2.backward(dout)\n",
        "  relu_dout1 = relu1.backward(dout1)\n",
        "  dx, dw1 = linear1.backward(relu_dout1)\n",
        "\n",
        "  w1 = w1-0.01*dw1\n",
        "  w2 = w2-0.01*dw2\n",
        "\n",
        "  # if (epoch+1) % 100 == 0:\n",
        "  #   print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss}')\n",
        "\n",
        "print('Pytorch : x=[4,5,6]----------------------------------------------------\\n')\n",
        "print(f'w1 weight : \\n{w1}')\n",
        "print(f'w2 weight : \\n{w2}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vnKQuBKTyLeN",
        "outputId": "7e1bc65b-bf06-4a29-95ae-16ff40f42fd5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Pytorch : x=[4,5,6]----------------------------------------------------\n",
            "\n",
            "w1 weight : \n",
            "tensor([[0.0875, 0.1873, 0.2872, 0.3870],\n",
            "        [0.4862, 0.5856, 0.6850, 0.7845],\n",
            "        [0.8850, 0.9839, 1.0829, 1.1819]], grad_fn=<SubBackward0>)\n",
            "w2 weight : \n",
            "tensor([[0.3040, 0.1960],\n",
            "        [0.5267, 0.3733],\n",
            "        [0.7494, 0.5506],\n",
            "        [0.9722, 0.7278]], grad_fn=<SubBackward0>)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "gAJrHuwZy0n9"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
