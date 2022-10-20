import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K 
import numpy as np


def best_convnet(X_train, Y_train, X_val, Y_val):
    Y_train = to_categorical(Y_train)
    Y_val = to_categorical(Y_val)
     
    
    print(X_train[0].shape)
   
    cnn = Sequential()
    cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=X_train[0].shape))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, kernel_size=3, activation='relu'))
    cnn.add(Flatten())
    cnn.add(Dense(11, activation='softmax'))
   
    cnn.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])

    cnn.fit(
       X_train, Y_train,
       #batch_size = 64,
       epochs = 12,
       verbose = 1,
       validation_data = (X_val, Y_val)
    )
        
    score = cnn.evaluate(x_test, y_test, verbose = 0) 

    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])

