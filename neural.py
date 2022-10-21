import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from utilities import train, performance, findBest, printByVal
import warnings
import itertools

warnings.filterwarnings('ignore') # setting ignore as a parameter


def createNN(layers, alpha, epochs, val_frac):
    early_stopping = False
    if val_frac > 0:
        early_stopping = True

    nn = MLPClassifier(
            hidden_layer_sizes=layers, 
            early_stopping=early_stopping,
            alpha=alpha,
            validation_fraction=val_frac,
            max_iter=epochs
          )    
    
    return nn


def selectNN(X_train, Y_train, X_val, Y_val):
    models = []
    counter = 0

    layers = ((80, 60, 40), (391), (250, 125))
    alphas = (0.0001, 0.001, 0.01)
    epochs = [50]
    val_frac = (0, .10)
    comb = itertools.product(layers, alphas, epochs, val_frac)
    for layer, alpha, epochs, val_frac in comb: 
        counter += 1
        specific = {
                "layers": layer,
                "alpha": alpha,
                "epochs": epochs,
                "val_frac": val_frac
                }
        print(f"    {specific}")
        model = createNN(layer, alpha, epochs, val_frac)
        print(f"        time: {train(model, X_train, Y_train)}")
        models.append((model, (performance(model, X_val, Y_val)), specific))
        print(f"        {models[-1][1][0]}")
    

    print(f"this many NN: {counter}")
 

    return findBest(models)

                    
