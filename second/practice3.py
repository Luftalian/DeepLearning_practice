# import numpy as np

# c = np.array([[1, 0, 0, 0, 0, 0, 0]])
# W = np.random.randn(7, 3)
# h = np.dot(c, W)
# print(h)

import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul

# c = np.array([[1, 0, 0, 0, 0, 0, 0]])
# W = np.random.randn(7, 3)
# layer = MatMul(W)
# h = layer.forward(c)
# print(h)

# c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
# c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# W_in = np.random.randn(7, 3)
# W_out = np.random.randn(3, 7)

# in_layer0 = MatMul(W_in)
# in_layer1 = MatMul(W_in)
# out_layer = MatMul(W_out)

# h0 = in_layer0.forward(c0)
# h1 = in_layer1.forward(c1)
# h = 0.5 * (h0 + h1)
# s = out_layer.forward(h)
# print(s)

from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
# print(corpus)

# print(id_to_word)

def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)

contexts, target = create_contexts_target(corpus, window_size=1)

# print(contexts)

# print(target)

from common.util import convert_one_hot

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

# print(target)

# print(contexts)

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        # Create layers
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        # List of all layers
        self.layers = [self.in_layer0, self.in_layer1, self.out_layer]
        # Combine all weights into a list
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        # Set word vector
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = cross_entropy_error(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.out_layer.backward(dout)
        ds *= 0.5
        self.in_layer1.backward(ds)
        self.in_layer0.backward(ds)
        return None

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
        return None



