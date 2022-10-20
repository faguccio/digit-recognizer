import time
from sklearn import svm


#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

#https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

def train_svm(X_train, Y_train, aC, aKernel, aGamma):  ###mmmm
    start = time.time()
    classifier = svm.SVC(C = aC, kernel = aKernel, gamma = aGamma)
    clf.fit(X_train, Y_train)
    

def svm_accuracy(X_val, Y_val)
    predictions
    val_acc = accuracy(nn, X_val, Y_val)
    print(val_acc)
