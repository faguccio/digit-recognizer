import time
from sklearn import svm, metrics
from utilities import accuracy, findBest, performance, train


#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

#https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

#hyperparameters:
    # -> C = regualrization parameter is by default 1.0f
    # -> kernel = rbf (default), poly, sigmoid, precomputed
    # -> gamma = kernel coefficient for rbf, poly and sigmoid
def createSVC(aC, aGamma, aKernel):
    classifier = svm.SVC(C = aC, kernel = aKernel, gamma = aGamma)
    return classifier


def selectSVC(X_train, Y_train, X_val, Y_val):
    models = []
    
    best_acc = 0
    best_combo = ('rbf', 'scale', 1.0)
    kernels = ('rbf', 'poly', 'sigmoid')
    gammas = ('scale', 'auto', 0.01, 0.1, 1.0)
    start, stop, step =  1, 10, 1  #c values
    
    for i in range(start, stop, step):
        for g in gammas:
            for k in kernels:
                svc = createSVC(i, g, k)
                print(f"         time: {train(svc, X_train, Y_train)}")
                models.append((svc, (performance(svc, X_val, Y_val))))
                print(f"        acc: {models[-1][1][0]}")
    
    return findBest(models)
    

def temporary(X_train, Y_train, X_val, Y_val):
    svc = create_svc(5, 'scale', 'rbf')
    print("svc created")
    train(svc, X_train, Y_train)
    print("svc trained")
    predictions = svc.predict(X_val)
    print("predictions made")
    printConfMatx(predictions, Y_val)
