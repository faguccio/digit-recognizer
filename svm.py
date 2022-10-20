import time
from sklearn import svm

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

#https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

#hyperparameters:
    # -> C = regualrization parameter is by default 1.0f
    # -> kernel = rbf (default), poly, sigmoid, precomputed
    # -> gamma = kernel coefficient for rbf, poly and sigmoid
def create_svc(aC, aGamma, aKernel):
    classifier = svm.SVC(C = aC, kernel = aKernel, gamma = aGamma)
    return classifier
    
def train_svc(X_train, Y_train, X_val, Y_val, model):  # train e accuracy in utilities
    start = time.time()
    model.fit(X_train, Y_train)
    predicted = model.predict(X_val)
    
    print(f"    train time: {time.time() - start} sec")
    val_acc = svc_accuracy(predicted, Y_val)
    start = time.time()
    print(f"    prediction time: {time.time() - start} sec")
    print(val_acc)
    return (val_acc)  #e' brutto che train ritorni accuracy forse...



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
                svc = create_svc(i, g, k)
                accuracy = train_svc(X_train, Y_train, X_val, Y_val, svc)
                if accuracy > best_acc:
                    best_svc = svc
                    best_acc = accuracy
                    best_combo = (k, g, i)
    print(f"best svc model has hyperparameters (kernel, gamma, c) =  {best_combo} ")
    print(f"accuracy : {best_acc}")
    
    return best_svc
    

def printConfMatx(predictions, Y_val):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_val, predictions)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()
