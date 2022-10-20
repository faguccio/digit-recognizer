import time
import numpy as np
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier


    #n_esitmators is the number of trees in the forest
        #other possible hyperparameters are -> gini or entropy or log_loss criterions
                                     #      -> max_depth of tree
                                     #      ->max_leaf_nodes
                                     #      ->bootstrap

def rf_train(X_train, Y_train, X_val, Y_val, hyperparameter, impurity, depth):
    start = time.time()
    rf = RandomForestClassifier(n_estimators = hyperparameter, criterion = impurity, max_depth = depth)
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_val)
    count = 0
    for i in range(len(Y_val)):
        if Y_val[i] != predictions[i]:
            count += 1
    val_accuracy = count/len(Y_val)
    
    print(f"    model {hyperparameter}, accuracy: {val_accuracy}")
    print(f"    time: {time.time() - start} sec")
    return (rf, val_accuracy)
    

def best_rf_model(X_train, Y_train, X_val, Y_val, impurity = 'gini'):  #questo metodo is basically the same as the one per knn. Potremmo farne uno unico?
    rf_models = []
    span = 10
    start, stop, step = 10, 500, 30
    
    # First, large span
    for i in range(start, stop, step):
        curr, acc = rf_train(X_train, Y_train, X_val, Y_val, i, impurity)
        rf_models.append([i, curr, acc])

    #chose the best, and span 10
    rf_models.sort(key = lambda x: -x[2])
    best = rf_models[0]
    print(f"{best} is the best with big steps")
    rf_models = [best]
    start = max(1, best[0] - span)
    stop = best[0] + span
    for i in range(start, stop, 1):
        if i == best[0]:
            continue
        curr, acc = rf_train(X_train, Y_train, X_val, Y_val, i, impurity)
        rf_models.append([i, curr, acc])
    
    rf_models.sort(key = lambda x: -x[2])
    return rf_models[0]
    

