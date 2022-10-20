import time
import numpy as np
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier


    #n_esitmators is the number of trees in the forest
        #other possible hyperparameters are -> gini or entropy or log_loss criterions
                                     #      -> max_depth of tree
                                     #      ->max_leaf_nodes
                                     #      ->bootstrap
def create_rf( aNumber , anImpurity):
        rf = RandomForestClassifier(n_estimators = aNumber, criterion = anImpurity)
        return rf

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
    

def best_rf_model(X_train, Y_train, X_val, Y_val):  #questo metodo is basically the same as the one per knn. Potremmo farne uno unico?
    best_acc = 0
    best_combo = (0, '')
    start, stop, step = 10, 500, 50  
    impurityMeasures = ['gini', 'entropy', 'log_loss']
    
    for imp in impurityMeasures:
        for i in range(start, stop, step):
            rf = create_rf(i, imp)
            acc = rf_train(X_train, Y_train, X_val, Y_val, rf)
            if acc > best_acc:
                best_rf = rf
                best_acc = acc;
                best_combo = (i, imp)
    print(f"best rf model has hyperparameters = {best_combo} ")
    print(f"accuracy : {best_acc}")
    return best_rf
