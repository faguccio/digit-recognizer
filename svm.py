import time
from sklearn import svm


def train_svm(X_train, Y_train, X_val, Y_val):  ###mmmm
    start = time.time()
    classifier = svm.SVC()
    clf.fit(X_train, Y_train)
    
    print(f"    train time: {time.time() - start} sec")
    val_acc = accuracy(nn, X_val, Y_val)
    start = time.time()
    print(f"    prediction time: {time.time() - start} sec")
    print(val_acc)
