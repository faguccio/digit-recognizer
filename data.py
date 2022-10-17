import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import models

X = np.load("MNIST-images.npy")
y = np.load("MNIST-labels.npy")

print(f"-----[0: data imoported]-----")

# Image preprocessing (increase contrast maybe)

#plt.imshow(X[3], cmap="Greys")
#plt.show()


# Train, val and test separation

X = X.reshape(X.shape[0], 576) 
seed = 547845
train_to_valtest = 0.70
val_to_test = 0.5
X_train, X_val, Y_train, Y_val = train_test_split(X, y, train_size=train_to_valtest, random_state=seed)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, train_size=val_to_test, random_state=seed)

print(f"-----[step 1: data splitted]-----")


# Model 1 validation gathering: KNN

models.best_knn_model(X_train, Y_train, X_val, Y_val)
