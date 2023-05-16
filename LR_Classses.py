import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

############### Decorator Functions for LogisticRegression and ClassificationMetrics classes ###############
def rounder(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        return round(f(self, *args, **kwargs), 2)
    return wrapper

def timer(f):
    import time
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        t1 = time.time()
        f(self,*args, **kwargs)
        t2 = time.time()
        print("-----------------------------------------")
        print('Training took {} seconds'.format(t2-t1))
    return wrapper

###################### ClassificationMetrics class for Binary Classification problem ######################
class ClassificationMetrics:
    def __init__(self, y_hat, y):
        self.y_hat = y_hat
        self.y = y
        # calculate confusion matrix
        self.TP = ((y_hat==1) == (y==1)).sum()
        self.TN = ((y_hat==0) == (y==0)).sum()
        self.FP = ((y_hat==1) == (y==0)).sum()
        self.FN = ((y_hat==0) == (y==1)).sum()
    
    @rounder
    def accuracy(self):
        preds = (self.y_hat > 0.5).astype(int)
        accuracy = round((preds == self.y).sum() / self.y.size, 2)
        return accuracy
    
    @rounder
    def recall(self):
        return self.TP / (self.TP + self.FN)
    
    @rounder
    def precision(self):
        return self.TP / (self.TP + self.FP)
    
    @rounder
    def F1(self):
        return 2 * self.TP / (2*self.TP + self.FP + self.FN)
    
    def __repr__(self):
        return 'ClassificationMetrics()'

####################### LogisticRegression Class for Binary Classification Problem #######################
class LogisticRegression:
    def __init__(self):
        self.W = None
        self.b = None
        self.mean = None
        self.std = None
        self.training_loss_logs = []

    def initialize_weights(self, X):
        _, n_features = X.shape
        W = np.random.randn(n_features, 1)
        b = 1
        return W, b
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def cross_entropy(y_hat, y):
        scale_factor = 1 / y.shape[0]
        total_loss = -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
        return (scale_factor * total_loss).sum()

    @staticmethod
    def accuracy(y_hat, y):
        preds = (y_hat > 0.5).astype(int)
        acc = round((preds == y).sum() / y.size, 2)
        return acc

    def normilizer(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        return X

    @timer
    def fit(self, X, y, num_iterations=5000, learning_rate=0.1, verbose=False):
        # normilize X
        X = self.normilizer(X)
        # initialize weights
        self.W, self.b = self.initialize_weights(X)
        for i in range(num_iterations):
            # get projection of X to W and add bias term
            z = np.matmul(X, self.W)+ self.b
            # obtain prediction
            y_hat = self.sigmoid(z)
            # if verbose == True, then calculate loss
            if verbose:
                loss = self.cross_entropy(y_hat, y)
            # calculate gradients
            dw = (1 / y.size) * np.dot(X.T, (y_hat - y))
            db = (1 / y.size) * np.sum(y_hat - y)
            # update weights (W & b)
            self.W -= learning_rate * dw
            self.b -= learning_rate * db
            # if verbose==True print training results 
            if verbose == True:
                if i % 500 == 0:
                    acc = self.accuracy(y_hat, y)
                    print(f"Epoch: {i+1} | Loss {loss:.3f} | Accuracy {acc:.3f}")
            self.training_loss_logs.append(loss.item())
    
    # returns probabilities
    def __call__(self, X):
        X = (X - self.mean) / self.std
        z = np.matmul(X, self.W)+ self.b
        z = self.sigmoid(z)
        return z
    
    # returns classes as predictions
    def predict(self, X):
        X = (X - self.mean) / self.std
        z = np.matmul(X, self.W)+ self.b
        z = self.sigmoid(z)
        return (z > 0.5).astype(int)
    
    def __repr__(self):
        return 'LogisticRegression()'
    
    def plot_training_losses(self):
        plt.figure(figsize=(7,3))
        plt.title('Training Losses through Iterations')
        plt.plot([*range(len(self.training_loss_logs))], self.training_loss_logs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

####################### DataGenerator Class for Binary Classification Problem #######################
# class DataGenerator:
#     def __init__(self, x, y, batch_size):
#         self.x = x
#         self.y = y
#         self.batch_size = batch_size
    
#     def detect_num_batch(self):
#         length = self.x.shape[0]
#         if length % self.batch_size == 0:
#             return length // self.batch_size
#         else:
#             return length // self.batch_size+1
    
#     def batch_generator(self):
#         idxs = np.arange(self.x.shape[0])
#         batch_num = self.detect_num_batch()
#         np.random.shuffle(idxs)
#         splits = np.array([i * self.batch_size for i in range(1, batch_num)])
#         batches = np.split(idxs, splits)
#         for i in batches:
#             yield self.x[i, :], self.y[i, :]
    
#     def random_batch_generator(self, num_of_batches=100):
#         length = self.x.shape[0] - 1
#         indices = np.arange(length)
#         for i in range(num_of_batches):
#             batch_indices = np.random.choice(indices, self.batch_size)
#             yield self.x[batch_indices, :], self.y[batch_indices, :]