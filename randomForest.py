import time
import numpy as np
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
from utilities import train, performance, findBest, accuracy


    #n_esitmators is the number of trees in the forest
        #other possible hyperparameters are -> gini or entropy or log_loss criterions
                                     #      -> max_depth of tree
                                     #      ->max_leaf_nodes
                                     #      ->bootstrap
def createRF( aNumber , anImpurity):
        rf = RandomForestClassifier(n_estimators = aNumber, criterion = anImpurity)
        return rf

"""
def rf_train(X_train, Y_train, X_val, Y_val, rf):
    start = time.time()
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_val)
    count = 0
    for i in range(len(Y_val)):
        if Y_val[i] == predictions[i]:
            count += 1
    val_accuracy = count/len(Y_val)
    
    print(f"    accuracy: {val_accuracy}")
    print(f"    time: {time.time() - start} sec")
    return val_accuracy
 """

def selectRF(X_train, Y_train, X_val, Y_val):
    models = []
    
    best_acc = 0
    best_combo = (0, '')
    start, stop, step = 5, 20, 1  
    impurityMeasures = ['gini', 'entropy', 'log_loss']
    

    for imp in impurityMeasures:
        for i in range(start, stop, step):
            rf = createRF(i, imp)
            print(f"        time: {train(rf, X_train, Y_train)}")
            models.append((rf, (performance(rf, X_val, Y_val))))
            print(f"        acc: {models[-1][1][0]}")
    return findBest(models)
