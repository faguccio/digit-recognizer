import numpy as np
#import keras
#import tensorflow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import models
import neural

X = np.load("MNIST-images.npy")
y = np.load("MNIST-labels.npy")

print(f"-----[0: data imoported]-----")

# Image preprocessing (increase contrast maybe)
# sarebbe anche da pulire lo sfondo, ma come?


#plt.imshow(X[3], cmap="Greys")
#plt.show()

#show data distribution
"""  #grafo già messo dentro a git. Ci sono pochi 8, ce ne freghiamo?
values, counts = np.unique(y, return_counts = True)
y_pos = np.arange(len(values))
plt.bar(y_pos, counts)  #creates the bars
plt.xticks(y_pos, values) #creates the names on the x-axis
plt.show()  #shows the graph
"""

# Train, val and test separation

seed = 547845
scaler = StandardScaler()

# Making a toy problem with super small size, so exec time is lower
reduction = 0.1
X, _, y, __ = train_test_split(X, y, train_size=reduction, random_state=seed)
print(f"Size of toy problem: {len(X), len(y)}")

X = X.reshape(X.shape[0], 576)
train_to_valtest = 0.70
val_to_test = 0.5
X_train, X_val, Y_train, Y_val = train_test_split(X, y, train_size=train_to_valtest, random_state=seed)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, train_size=val_to_test, random_state=seed)

scaler.fit(X_train)
X_train = scaler.transform(X_train)   #this can be done in the pipeline
X_val = scaler.transform(X_val)        #training time with this preprocessing step takes 1/3 of the time:
X_test = scaler.transform(X_test)       #train time : 53.98358082771301 sec vs 150.37893509864807 sec
                                        #prediction time : 9.5367431640625e-07 sec vs  0.0 sec
                                       #accuracy is also higher: 0.870992963252541 vs 0.7529319781078968
"""
for i in range(len(X_train)):
    X_train[i]/= 255
for j in range(len(X_val)):
    X_val[j]  /= 255
    X_test[j] /=255
"""
print(f"-----[step 1: data splitted]-----")


# Model 1 validation gathering: KNN
#print(models.best_knn_model(X_train, Y_train, X_val, Y_val))


# Neural Network

from neural import best_nn
from randomForest import best_rf_model

#print(best_nn(X_train, Y_train, X_val, Y_val))

print(best_rf_model(X_train, Y_train, X_val, Y_val))



