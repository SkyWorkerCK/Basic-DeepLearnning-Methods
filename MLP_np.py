import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
def dsigmoid(z):
    return sigmoid(z)(1-sigmoid(z))

class MLP:

    def __init__(self, sizes):
        """
        :param sizes: [784, 30, 10]
        """
        self.sizes = sizes
        # sizes: [784, 30 , 10]
        # w: [ch_out, ch_in]
        # b: [ch_out]
        self.weights = [np.random.randn(ch2, ch1) for ch1, ch2 in zip(sizes[:2], sizes[1:])]
        # self.weights:  [30, 784]   [10, 30]
        self.biases = [np.random.randn(ch) for ch in sizes[1:]]
        # self.biases:  [30, 10]


    def forward(self, x):
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, x) + b
            xx = sigmoid(z)

        return xx

    def backpropagate(self, x, y):

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activations = [x]
        zx = []
        activation = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            zx.append(z)
            activations.append(activation)

        # 反向传播
        # 1. 在输入层计算梯度



def main():
    pass



if __name__ == '__main__':
    main()