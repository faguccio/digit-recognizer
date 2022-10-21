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


#there is an abundance of 3 digits and a shortage of 8 digits.
#Preprocessing will therefore include removing most of the 3s (originally 30000) so that they are around the same amount as other digits in the dataset.
#For the 8s, they are removed, and then added 70% to the training set, and the rest is split 50-50 between validation and test sets. This way there is no risk of not having any samples of digit 8 in the train/validation/test sets.
X_three, Y_three = [], []
X_eight, Y_eight = [], []
X_rest, Y_rest = [], []
for i in range(len(y)):
    if y[i] == 3:
        X_three.append(X[i])
        Y_three.append(3)
    elif y[i] == 8:
        X_eight.append(X[i])
        Y_eight.append(8)
    else:
        X_rest.append(X[i])
        Y_rest.append(y[i])

X_three = X_three[:7000]  #keep only 7000 threes
Y_three = Y_three[:7000]

X = X_rest + X_three
y = Y_rest + Y_three

print(f"{len(X_eight)}")


# Train, val and test separation

#seed = 547845

reduction = 0.005
# Making a toy problem with super small size, so exec time is lower

X, _, y, __ = train_test_split(X, y, train_size=reduction) #, random_state=seed)
print(f"Size of toy problem: {len(X), len(y)}")
eights_x, _ , eights_y, __ = train_test_split(X_eight, Y_eight, train_size = reduction)

#oldX = ut.copy_data_set(X)
#assert(X[20][3][15] == oldX[20][3][15])

train_to_valtest = 0.70
val_to_test = 0.5
X_train, X_val, Y_train, Y_val = train_test_split(X, y, train_size=train_to_valtest) #, random_state=seed)

X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, train_size=val_to_test) #, random_state=seed)

eights_x_train, eights_x_valtest, eights_y_train, eights_y_valtest = train_test_split(X_eight, Y_eight, train_size = train_to_valtest)
eights_x_val, eights_x_test, eights_y_val, eights_y_test = train_test_split(eights_x_valtest, eights_y_valtest, train_size = val_to_test)

X_train.extend(eights_x_train)
Y_train.extend(eights_y_train)

X_val.extend(eights_x_val)
Y_val.extend(eights_y_val)

X_test.extend(eights_x_test)
Y_test.extend(eights_y_test)



X = np.array(X)
X = X.reshape(X.shape[0], 576)

print(f"-----[step 1: data splitted]-----")



"""
from neural import selectNN
print(selectNN(X_train, Y_train, X_val, Y_val))
"""

from convnet import selectCNN
print(selectCNN(X_train, Y_train, X_val, Y_val))
#from neural import selectNN
#selectNN(X_train, Y_train, X_val, Y_val)

from randomForest import selectRF

#print(selectRF(X_train, Y_train, X_val, Y_val))

#print(best_nn(X_train, Y_train, X_val, Y_val))




#from convnet import selectCNN
#print(best_convnet(X_train, Y_train, X_val, Y_val))
