from sklearn.metrics import confusion_matrix
import time
import numpy as np
import os, sys

def stopPrint(func, *args, **kwargs):
    with open(os.devnull,"w") as devNull:
        original = sys.stdout
        sys.stdout = devNull
        func(*args, **kwargs)
        sys.stdout = original 


def accuracy(y_pred, Y):
    good_predictions = (y_pred == Y)
    accuracy = np.sum(good_predictions) / len(Y)
    return accuracy



def train(model, X_train, Y_train):
    start = time.time()
    model.fit(X_train, Y_train)
    return -start+time.time()
    

def min_score(confmatx):
    classes = [0]*11
    for i in range(len(confmatx)):
        tot = 0
        for j in range(len(confmatx[0])):
            tot += confmatx[i][j]

        if tot == 0:
            classes[i] = 1
        else:
            classes[i] = confmatx[i][i] / tot
    if min(classes) == 0 and classes.index(min(classes)) == 10:
        input(confmatx)
    return min(classes), classes.index(min(classes))


def performance(model, X, Y):
    start = time.time()
    y_pred = model.predict(X)
    predict_time = time.time() - start
    conf_matrix = confusion_matrix(Y, np.rint(y_pred))
    min_scr, min_class = min_score(conf_matrix)
    score = accuracy(y_pred, Y)
    return score, min_scr, min_class, conf_matrix, predict_time



def performanceKeras(model, X, Y):
    start = time.time()
    y_pred = model.predict(X, verbose=0)
    predict_time = time.time() - start
    y_pred = np.array([ list(a).index(max(a)) for a in y_pred])
    Y_val = np.array([ list(a).index(max(a)) for a in Y])
    score = accuracy(y_pred, Y_val)
    conf_matrix = confusion_matrix(Y_val, np.rint(y_pred))
    min_scr, min_class = min_score(conf_matrix) 
    return score, min_scr, min_class, conf_matrix, predict_time





def findBest(models):
    best = None
    for model in models:
        if best == None:
            best = model
            continue
        
        margin = model[1][0] - best[1][0] 
        if margin > 0 and model[1][1] + margin >= best[1][1] - margin:
            best = model

    return best
            


def copy_data_set(X):
    res = []
    for image in X:
        copy = []
        for row in image:
            copy.append(row.copy())
        res.append(copy)
    
    return res



def printByVal(models, key, val):
    summ = 0
    tot = 0
    for model in models:
        if model[2][key] == val:
            summ += model[1][0]
            tot += 1
    print(summ/tot)
