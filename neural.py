import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from utilities import train, performance

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
    layerss = (50, 50)
    alphas = (0.0001, 0.0005)
    epochss = (100, 50)
    val_fracs = (0, .5, .10)
    for layers in layerss:
        for alpha in alphas:
            for epochs in epochss:
                for val_frac in val_fracs:
                    model = createNN(layers, alpha, epochs, val_frac)
                    train(model, X_train, Y_train)
                    print(performance(model, X_val, Y_val))


                    
