import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import utilities as ut
#from sklearn.preprocessing import StandardScaler

X = np.load("MNIST-images.npy")
y = np.load("MNIST-labels.npy")
print(f"-----[0: data imoported]-----")


# Image preprocessing (increase contrast maybe)
# sarebbe anche da pulire lo sfondo, ma come?
X = (X/255).astype('float32')


# Train, val and test separation

#seed = 547845

reduction = 0.005
X, _, y, __ = train_test_split(X, y, train_size=reduction) #, random_state=seed)
print(f"Size of toy problem: {len(X), len(y)}")

#oldX = ut.copy_data_set(X)
#assert(X[20][3][15] == oldX[20][3][15])

#X = X.reshape(X.shape[0], 576)
train_to_valtest = 0.70
val_to_test = 0.5
X_train, X_val, Y_train, Y_val = train_test_split(X, y, train_size=train_to_valtest) #, random_state=seed)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, train_size=val_to_test) #, random_state=seed)




print(f"-----[step 1: data splitted]-----")



"""
from neural import selectNN
print(selectNN(X_train, Y_train, X_val, Y_val))
"""

from convnet import selectCNN
print(selectCNN(X_train, Y_train, X_val, Y_val))


