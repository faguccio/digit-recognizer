import numpy as np

def accuracy(model, X, Y):
    predictions = model.predict(X)
    good_predictions = (predictions == Y)
    accuracy = np.sum(good_predictions) / len(X)
    return accuracy


# Image in format [[], ..., []]
def copy_data_set(X):
    res = []
    for image in X:
        copy = []
        for row in image:
            copy.append(row.copy())
        res.append(copy)
    
    return res
