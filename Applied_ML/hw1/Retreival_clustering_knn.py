# initialization code
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from scipy import stats
import faiss
import time




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


(x_train, y_train), (x_test, y_test) = load_mnist()  # load the data



def get_nearest(X_query, X):
  ''' Return the index of the sample in X that is closest to X_query according
      to L2 distance '''
  # TO DO
  dist = np.linalg.norm(X[0] - X_query)
  ind = 0
  for i in range(1, len(X)):
    if np.linalg.norm(X[i] - X_query) < dist:
      dist = np.linalg.norm(X[i] - X_query)
      ind = i
  return ind


def kmeans(X, K, niter=10):
  '''
  Starting with the first K samples in X as cluster centers, iteratively assign each
  point to the nearest cluster and compute the mean of each cluster.
  Input: X[i] is the ith sample, K is the number of clusters, niter is the number of iterations
  Output: K cluster centers
  '''
  # TO DO -- implement kmeans and add code to display cluster centers at each iteration
  cluster_centers = X[:K].copy()
  
  for i in range(niter):
    clusters = [[] for _ in range(K)]
    for j in range(len(X)):
        nearest = get_nearest(X[j], cluster_centers)
        clusters[nearest].append(X[j])
    for l in range(K):
        if len(clusters[l]) > 0:
            cluster_centers[l] = np.mean(clusters[l], axis=0)
    if i == 1:
        plt.imshow(cluster_centers[4].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()
    if i == niter - 1:
            plt.imshow(cluster_centers[4].reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.show()
        
  return cluster_centers


def one_nn(X_query,y_query, X, y):
    true = 0
    false = 0
    for i in range(len(X_query)):
        nearest = get_nearest(X_query[i], X)
        if y[nearest] == y_query[i]:
            true += 1
        else:
            false += 1

    return false/(true+false)



index = faiss.IndexFlatL2(x_train.shape[1])  # set for exact search
index.add(x_train) # add the data
dist, idx = index.search(x_test[:2],1) # returns index and sq err for each sample






def kmeans_fast(X, K, niter=10):
  '''
  Starting with the first K samples in X as cluster centers, iteratively assign each
  point to the nearest cluster using faiss and compute the mean of each cluster.
  Input: X[i] is the ith sample, K is the number of clusters, niter is the number of iterations
  Output: K cluster centers
  '''

  # TO DO (you can base this on part 1, but use FAISS for search)
  # if you include display code, you need to re-organize the plotting code below

    # TO DO -- implement kmeans and add code to display cluster centers at each iteration
  cluster_centers = X[:K].copy()
  
  
  rmse = []


  for i in range(niter):
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(cluster_centers)
    clusters = [[] for _ in range(K)]
    dist, idx = index.search(X, 1)

    total_dist = 0
    for j in range(len(X)):
        nearest = idx[j][0]
        clusters[nearest].append(X[j])

    rmse.append(np.sqrt(np.mean(dist)))

    for l in range(K):
        if len(clusters[l]) > 0:
            cluster_centers[l] = np.mean(clusters[l], axis=0)


  return cluster_centers, rmse



        
  


"""
K=10
centers, rmse = kmeans_fast(x_train, K, niter=20)
plt.plot(np.arange(len(rmse)), rmse, label='K=10')

K=30
centers, rmse = kmeans_fast(x_train, K, niter=20)
plt.plot(np.arange(len(rmse)), rmse, label='K=30')

K=100
centers, rmse = kmeans_fast(x_train, K, niter=20)
plt.plot(np.arange(len(rmse)), rmse, label='K=100')
plt.legend(), plt.ylabel('RMSE'), plt.xlabel('# iterations')
plt.show()
"""




# 1-NN

nsample = [100, 1000, 10000, 60000]

# TO DO





def one_nn_exact_faiss(X_query,y_query, X, y):
    true = 0
    false = 0
    index = faiss.IndexFlatL2(X_query.shape[1])
    index.add(X)
    dist, idx = index.search(X_query, 1)
    for i in range(len(X_query)):
        nearest = idx[i][0]
        if y[nearest] == y_query[i]:
            true += 1
        else:
            false += 1

    return true/(true+false)







def one_nn_lsh_faiss(X_query,y_query, X, y):
    true = 0
    false = 0
    index = faiss.IndexLSH(X_query.shape[1], X_query.shape[1])  # 128 is the number of bits for LSH
    index.add(X)
    dist, idx = index.search(X_query, 1)
    for i in range(len(X_query)):
        nearest = idx[i][0]
        if y[nearest] == y_query[i]:
            true += 1
        else:
            false += 1

    return true/(true+false)

acc_exact = [one_nn_exact_faiss(x_test, y_test, x_train[:n], y_train[:n]) for n in nsample]
acc_lsh = [one_nn_lsh_faiss(x_test, y_test, x_train[:n], y_train[:n]) for n in nsample]
acc_exact = np.array(acc_exact)
acc_lsh = np.array(acc_lsh)


"""
plt.semilogx(nsample, (1-acc_exact)*100, label='exact')
plt.semilogx(nsample, (1-acc_lsh)*100, label='lsh')
plt.legend(), plt.ylabel('% error'), plt.xlabel('# training samples')
plt.show()
"""



def regress_KNN(X_trn, y_trn, X_tst, K=1):
   dist = np.zeros((len(X_trn)))
   y_pred = np.zeros((len(X_tst)))
   for i in range(len(X_tst)):
      dist = np.linalg.norm(X_trn - X_tst[i], axis=1)
      idx = np.argsort(dist)[:K]