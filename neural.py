import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from utilities import train, performance, findBest
import warnings
import itertools
warnings.filterwarnings('ignore') # setting ignore as a parameter

# hyperparam su cui lavorare: 

# alpha https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py





def createNN(layers, alpha, epochs, val_frac):
    early_stopping = False
    if val_frac > 0:
        early_stopping = True

    nn = MLPClassifier(
            hidden_layer_sizes=layers, 
            warm_start=True,
            early_stopping=early_stopping,
            alpha=alpha,
            validation_fraction=val_frac,
            max_iter=epochs
          )    
    
    return nn



def selectNN(X_train, Y_train, X_val, Y_val):
    models = []

    layers = ((80, 60, 40), (391), (250, 120))
    alphas = (0.0001, 0.0005)
    epochs = (1, 50)
    val_frac = (0, .5, .10)
    comb = itertools.product(layers, alphas, epochs, val_frac)
    for layer, alpha, epochs, val_frac in comb: 
        model = createNN(layer, alpha, epochs, val_frac)
        train(model, X_train, Y_train)
        models.append((model, (performance(model, X_val, Y_val))))
        print(models[-1][1][0], layer)
    return findBest(models)

                    
