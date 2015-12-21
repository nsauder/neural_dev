"""
- allocate train and test data
- initialize random weight
- run things back & forth
"""
import numpy as np

MAX_ITER = 10
BATCH_SIZE = 100
eps = 0.001
num_feat = 100


x_tr = np.random.randn(1000, num_feat)
x_te = np.random.randn(1000, num_feat)
true_weight = np.random.randn(num_feat,)

y_tr = x_tr.dot(true_weight)
y_te = x_te.dot(true_weight)

neural_net = np.dot


class square_loss_layer(object):
    def __init__(self, weights):
        self.weights = weights

    def forward(previous_layer):
        return np.dot(previous_layer, self.weights)

    def loss(labels, previous_layer):
        preds = self.forward(previous_layer, self.weights)
        return np.mean((preds - labels)**2)

    def backward(preds, labels, previous_layer):
        grads = 2 * np.dot(preds - labels, previous_layer)
        self.weights = self.weights - 0.0001 * grads
        return grads


def square_loss(preds, labels):
    return np.mean((preds-labels)**2)


def grad_square_loss(preds, labels, previous_layer):
    return 2 * np.dot(preds - labels, previous_layer)


def train(X, Y, batch_size, max_iter):
    init_weights = np.random.randn(num_feat,)
    
    
    for batch_num in range(MAX_ITER):
        data_section = batch_num % 10
        slc = np.s_[data_section * BATCH_SIZE:
                    (data_section + 1) * BATCH_SIZE]
        
        x_data = x_tr[slc]
        y_data = y_tr[slc]
        
        preds = neural_net(x_data, init_weights)
        
        print square_loss(preds, y_data)
        grads = grad_square_loss(preds, y_data, x_data)
        
        init_weights = init_weights - eps * grads
<<<<<<< HEAD


=======
>>>>>>> b9068356fbf8d6f990ab6ad0b822f40c30752fd0
