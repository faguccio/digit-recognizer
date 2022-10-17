import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from utilities import accuracy

# Given that inference on this model is way to slow, I just compute
# validation accuracy

def knn_train(X_train, Y_train, X_val, Y_val, k):
    start = time.time()
    k_NN = neighbors.KNeighborsClassifier(n_neighbors=k)
    k_NN.fit(X_train, Y_train)
    val_accuracy = accuracy(k_NN, X_val, Y_val)
    print(f"    model {k}, accuracy: {val_accuracy}")
    print(f"    time: {time.time() - start} sec")
    return (k_NN, val_accuracy)





def best_knn_model(X_train, Y_train, X_val, Y_val):
    knn_models = [] 
    span = 10
    start, stop, step = 5, 300, 50
    
    # First, large span
    for i in range(start, stop, step):
        curr, acc = knn_train(X_train, Y_train, X_val, Y_val, i)
        knn_models.append([i, curr, acc])

    # Now i chose the best, and span 10
    knn_models.sort(key = lambda x: -x[2]) 
    best = knn_models[0]
    print(f"{best} is the best with big steps")
    knn_models = [best]
    start = max(1, best[0] - span)
    stop = best[0] + span
    for i in range(start, stop, 1):
        if i == best[0]:
            continue 
        curr, acc = knn_train(X_train, Y_train, X_val, Y_val, i)
        knn_models.append([i, curr, acc])
    
    knn_models.sort(key = lambda x: -x[2]) 
    return knn_models[0] 



