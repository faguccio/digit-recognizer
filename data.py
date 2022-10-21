import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import utilities as ut
from sklearn.preprocessing import StandardScaler

X = np.load("MNIST-images.npy")
y = np.load("MNIST-labels.npy")
print(f"-----[0: data imoported]-----")


# Image preprocessing (increase contrast maybe)
# sarebbe anche da pulire lo sfondo, ma come?

#plt.imshow(X[6000], cmap="Greys")
#plt.show()

for i in range(len(X)):
    brightest = X[i].max()
    threshold = brightest // 2
    X[i] = np.where(X[i] > 178, X[i], 0)

#plt.imshow(X[6000], cmap = "Greys")
#plt.show()

X = X.reshape(X.shape[0], 576)
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

reduction = 0.01
# Making a toy problem with super small size, so exec time is lower

X, _, y, __ = train_test_split(X, y, train_size=reduction) #, random_state=seed)
print(f"Size of toy problem: {len(X), len(y)}")
X_eight, _ , Y_eight, __ = train_test_split(X_eight, Y_eight, train_size = reduction)

#oldX = ut.copy_data_set(X)
#assert(X[20][3][15] == oldX[20][3][15])

train_to_valtest = 0.70
val_to_test = 0.5
X_train, X_val, Y_train, Y_val = train_test_split(X, y, train_size=train_to_valtest) #, random_state=seed)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, train_size=val_to_test) #, random_state=seed)


y = y + Y_eight
eights_x_train, eights_x_valtest, eights_y_train, eights_y_valtest = train_test_split(X_eight, Y_eight, train_size = train_to_valtest)
eights_x_val, eights_x_test, eights_y_val, eights_y_test = train_test_split(eights_x_valtest, eights_y_valtest, train_size = val_to_test)

X_train.extend(eights_x_train)
Y_train.extend(eights_y_train)

X_val.extend(eights_x_val)
Y_val.extend(eights_y_val)

X_test.extend(eights_x_test)
Y_test.extend(eights_y_test)


"""   # to show that all numbers are present in the training set in the same proportion
amounts = []
numbers = []
for digit in range (0, 11):
    amounts.append(Y_train.count(digit)/y.count(digit))
    numbers.append(digit)

width = 0.35
fig, ax = plt.subplots()

ax.bar(numbers, amounts)
ax.set_ylabel('percentages')
ax.set_title('all numbers are present in training dataset')
fig.tight_layout()
plt.show()
"""


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_val= np.array(X_val)
Y_val= np.array(Y_val)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(f"-----[step 1: data splitted]-----")
                                  




finalist = []
"""
import convnet
print("**********************\n\n\n")


print("CNN")
start = time.time()
#finalist.append(convnet.selectCNN(X_train, Y_train, X_val, Y_val))
print(f"totale time = {time.time() - start}")
"""
first_start = time.time()


#preprocessing
X_train = X_train.reshape(X_train.shape[0], 576)
X_val = X_val.reshape(X_val.shape[0], 576)
X_test = X_test.reshape(X_test.shape[0], 576)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)   
X_val = scaler.transform(X_val)      
X_test = scaler.transform(X_test)   

 
"""
import neural

print("NeuralNetwork")
start = time.time()
finalist.append(neural.selectNN(X_train, Y_train, X_val, Y_val))
print(f"totale time = {time.time() - start}")
"""

import svm
print("SVC")
start = time.time()
finalist.append(svm.selectSVC(X_train, Y_train, X_val, Y_val))
print(f"totale time = {time.time() - start}")


import randomForest
print("Random Forest")
start = time.time()
finalist.append(randomForest.selectRF(X_train, Y_train, X_val, Y_val))
print(f"totale time = {time.time() - start}")

print(f"totale time for real: {time.time() - first_start}")

for model in finalist:
    predictions = model[0].predict(X_val)
    class = type(model[0])
    ut.printConfMatx(predictions, Y_val, 'model[0]')
    
winner = ut.findBest(finalist)
print(winner)
    

predictions = winner[0].predict(X_test)
ut.printConfMatx(predictions, Y_test, false)
print(f"   {winner[0]} test accuracy is :  ")
test_acc = ut.accuracy(predictions, Y_test)
print(f"{test_acc}")


