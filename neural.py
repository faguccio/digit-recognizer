import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from utilities import accuracy

# hyperparam su cui lavorare: 

# alpha https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py

# minibatch

# max_iter (def 200) -> epoch

# Optimization
# warm_start=True
# early_stopping=True


def best_nn(X_train, Y_train, X_val, Y_val):
    start = time.time()
    nn = MLPClassifier(
            hidden_layer_sizes=(1000), 
            max_iter=202
            ).fit(X_train, Y_train)
    
    print(f"    train time: {time.time() - start} sec")
    val_acc = accuracy(nn, X_val, Y_val)
    start = time.time()
    print(f"    prediction time: {time.time() - start} sec")
    print(val_acc)


