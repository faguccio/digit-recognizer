import keras 
import itertools
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D 
import numpy as np
from utilities import performanceKeras, findBest, stopPrint
import time


print("\n\n")

def createCNN(conv_layers):
    cnn = Sequential()
    for l in conv_layers:
        cnn.add(l)
    

    cnn.add(Flatten())
    cnn.add(Dense(11, activation='softmax'))
    cnn.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
    return cnn




def trainCNN(model, X_train, Y_train, X_val, Y_val, batch_size, epochs):
    start = time.time()
    model.fit(
       X_train, Y_train,
       batch_size = batch_size,
       epochs = epochs,
       verbose = 0,
       validation_data = (X_val, Y_val)
    )
    return time.time() - start
    


def selectCNN(X_train, Y_train, X_val, Y_val):
    Y_train = to_categorical(Y_train)
    Y_val = to_categorical(Y_val)
    
    clayers = []
    cflayers = []
    filter_amount = (32, 64)
    kernel_sizes_conv = ((3, 3), (4, 4))
    for cl, ks in itertools.product(filter_amount, kernel_sizes_conv):
        clayers.append( (
                Conv2D(cl, kernel_size=ks, activation='relu') ,
        {"filter": cl,
         "kersize": ks} ) )
        cflayers.append( (
                Conv2D(cl, kernel_size=ks, activation='relu', input_shape=X_train[0].shape),
                {"filter": cl,
                "kersize": ks}  
                ) )
    
    poolayer = MaxPooling2D(pool_size = (2, 2))
    

    final_layers = []

    # just 1 conv layer
    for l in cflayers:
        final_layers.append( ( [l[0]] , [l[1]] ) )
        # with pooling
        pooldict = l[1].copy()
        pooldict["pooling"] = True
        final_layers.append( ( [l[0], poolayer], [pooldict] ))


    # 2 conv layer
    for l in cflayers:
        if l[1]["filter"] == 64:
            continue
        for l2 in clayers:
            final_layers.append( ( [l[0], l2[0]] , [l[1], l2[1]] ) )
            
            # with pooling after the first
            pooldict = l[1].copy()
            pooldict["pooling"] = True
            final_layers.append( ( [l[0], poolayer, l2[0]], [pooldict, l2[1]] ) )
    """
    for fl in final_layers:
        final_layers.append(fl.copy().append(Dropout(0.1)))
    """    
   
    models = []
    print(f"how many cnn: {len(final_layers)}")
    batch_sizes = [128]
    epochss = [6]
    for bs, ep in zip(batch_sizes, epochss): 
        for fl in final_layers:
            print(f"    {fl[1]}")
            model = createCNN(fl[0])
            print(f"        time {trainCNN(model, X_train, Y_train, X_val, Y_val, bs, ep)}")
            models.append((model, (performanceKeras(model, X_val, Y_val)), fl[1]))
            print(f"        acc: {models[-1][1][0]}")  
    return findBest(models)


