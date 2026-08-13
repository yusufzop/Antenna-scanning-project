# initialization code
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
import faiss
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from Applied_ML.hw1.Retreival_clustering_knn import get_nearest, one_nn, one_nn_exact_faiss

def load_mnist():
  '''
  Loads, reshapes, and normalizes the data
  '''
  (x_train, y_train), (x_test, y_test) = mnist.load_data() # loads MNIST data
  x_train = x_train[::-1]
  y_train = y_train[::-1]
  x_train = np.reshape(x_train, (len(x_train), 28*28))  # reformat to 784-d vectors
  x_test = np.reshape(x_test, (len(x_test), 28*28))
  maxval = x_train.max()
  x_train = x_train/maxval  # normalize values to range from 0 to 1
  x_test = x_test/maxval
  return (x_train, y_train), (x_test, y_test)

def display_mnist(x, subplot_rows=1, subplot_cols=1):
  '''
  Displays one or more examples in a row or a grid
  '''
  if subplot_rows>1 or subplot_cols>1:
    fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(15,15))
    for i in np.arange(len(x)):
      a = i // subplot_cols
      b = i % subplot_cols
      ax[a][b].imshow(np.reshape(x[i], (28,28)), cmap='gray')
      ax[a][b].axis('off')
  else:
      plt.imshow(np.reshape(x, (28,28)), cmap='gray')
      plt.axis('off')
  plt.show()



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = load_mnist()



C = [0.25, 0.5, 1, 2, 4]

for c in C:
    clf = svm.LinearSVC(C=c, max_iter=1000)
    clf.fit(x_train[:1000], y_train[:1000])
    pred = clf.predict(x_train[50000:])
    error = np.mean(pred != y_train[50000:])
    print(f"C: {c}, Error: {error}")
