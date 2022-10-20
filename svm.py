import time
from sklearn import svm

#hyperparameters:
    # -> C = regualrization parameter is by default 1.0f
    # -> kernel = rbf (default), poly, sigmoid, precomputed
    # -> gamma = kernel coefficient for rbf, poly and sigmoid
def train_svc(
                X_train, Y_train, X_val, Y_val,
                aC, aGamma, aKernel):
    start = time.time()
    classifier = svm.SVC(C = aC, kernel = aKernel, gamma = aGamma)
    
    classifier.fit(X_train, Y_train)
    predicted = classifier.predict(X_val)
    
    print(f"    train time: {time.time() - start} sec")
    val_acc = svc_accuracy(predicted, Y_val)
    start = time.time()
    print(f"    prediction time: {time.time() - start} sec")
    print(val_acc)
    return (classifier, val_acc)



def svc_accuracy(predicted, Y_val):
    count = 0
    for i in range(len(predicted)):
        if predicted[i] == Y_val[i]:
            count += 1
    val_acc = count/len(predicted)
    return val_acc
    

def best_svm_model(X_train, Y_train, X_val, Y_val):
    best_acc = 0
    best_combo = ('rbf', 'scale', 1.0)
    kernels = ('rbf', 'poly', 'sigmoid')
    gammas = ('scale', 'auto', 0.001, 0.01, 0.1, 1.0)
    start, stop, step =  1, 10, 1  #c values
    for i in range(start, stop, step):
        for g in gammas:
            for k in kernels:
                svc, accuracy = train_svc(X_train, Y_train, X_val, Y_val, i, g, k)
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_combo = (k, g, i)
    print(f"best svc model has gamma = {g}, kernel = {k}, regularization param = {i} ")
    print(f"accuracy : {best_acc}")
                
