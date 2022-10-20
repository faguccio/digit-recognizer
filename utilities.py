from sklearn.metrics import confusion_matrix
import time
import numpy as np

def accuracy(model, X, Y):
    predictions = model.predict(X)
    good_predictions = (predictions == Y)
    accuracy = np.sum(good_predictions) / len(X)
    return accuracy




def train(model, X_train, Y_train):
    start = time.time()
    model.fit(X_train, Y_train)
    return start-time.time()
    

def min_score(confmatx):
    classes = [0]*11
    for i in range(11):
        tot = 0
        for j in range(11):
            tot += confmatx[i][j]

        classes[i] = confmatx[i][i] / tot
    return min(classes), i


def performance(model, X, Y):
    start = time.time()
    y_pred = model.predict(X)
    predict_time = time.time() - start
    conf_matrix = confusion_matrix(Y, np.rint(y_pred))
    min_scr, min_class = min_score(conf_matrix) 
    return model.score(X, Y), min_scr, min_class, predict_time




# Image in format [[], ..., []]
def copy_data_set(X):
    res = []
    for image in X:
        copy = []
        for row in image:
            copy.append(row.copy())
        res.append(copy)
    
    return res
