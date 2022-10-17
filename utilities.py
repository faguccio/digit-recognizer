import numpy as np

def accuracy(model, X, Y):
    predictions = model.predict(X)
    good_predictions = (predictions == Y)
    accuracy = np.sum(good_predictions) / len(X)
    return accuracy


