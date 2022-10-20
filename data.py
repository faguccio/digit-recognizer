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

# Making a toy problem with super small size, so exec time is lower
reduction = 0.1
X, _, y, __ = train_test_split(X, y, train_size=reduction) #, random_state=seed)
print(f"Size of toy problem: {len(X), len(y)}")

#oldX = ut.copy_data_set(X)
#assert(X[20][3][15] == oldX[20][3][15])

X = X.reshape(X.shape[0], 576)
train_to_valtest = 0.70
val_to_test = 0.5
X_train, X_val, Y_train, Y_val = train_test_split(X, y, train_size=train_to_valtest) #, random_state=seed)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, train_size=val_to_test) #, random_state=seed)




print(f"-----[step 1: data splitted]-----")


# Model 1 validation gathering: KNN
#print(models.best_knn_model(X_train, Y_train, X_val, Y_val))


from neural import selectNN
#selectNN(X_train, Y_train, X_val, Y_val)
from randomForest import best_rf_model

#print(best_rf_model(X_train, Y_train, X_val, Y_val))
#print(best_nn(X_train, Y_train, X_val, Y_val))



#from convnet import best_convnet
#print(best_convnet(X_train, Y_train, X_val, Y_val))

from svm import best_svm_model
best_svm_model(X_train, Y_train, X_val, Y_val)

