# Deeplearning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import os, shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
data = []
labels = []
classes = 43
cur_path = os.getcwd()
#Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(cur_path,"//content//drive//MyDrive//Colab Notebooks//assignment//Train",str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '//'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")
#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
(10745, 30, 30, 3) (2687, 30, 30, 3) (10745,) (2687,)


y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

y_train
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.utils import  plot_model
plot_model(model, show_shapes=True)


model.summary()
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_8 (Conv2D)           (None, 26, 26, 32)        2432      
                                                                 
 conv2d_9 (Conv2D)           (None, 22, 22, 32)        25632     
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 11, 11, 32)       0         
 2D)                                                             
                                                                 
 dropout_6 (Dropout)         (None, 11, 11, 32)        0         
                                                                 
 conv2d_10 (Conv2D)          (None, 9, 9, 64)          18496     
                                                                 
 conv2d_11 (Conv2D)          (None, 7, 7, 64)          36928     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 3, 3, 64)         0         
 2D)                                                             
                                                                 
 dropout_7 (Dropout)         (None, 3, 3, 64)          0         
                                                                 
 flatten_2 (Flatten)         (None, 576)               0         
                                                                 
 dense_4 (Dense)             (None, 256)               147712    
                                                                 
 dropout_8 (Dropout)         (None, 256)               0         
                                                                 
 dense_5 (Dense)             (None, 43)                11051     
                                                                 
=================================================================
Total params: 242,251
Trainable params: 242,251
Non-trainable params: 0
_________________________________________________________________


epochs = 30
history = model.fit(X_train, y_train, batch_size=62, epochs=epochs, validation_data=(X_test, y_test))
Epoch 1/30
174/174 [==============================] - 53s 300ms/step - loss: 1.9627 - accuracy: 0.5109 - val_loss: 0.3557 - val_accuracy: 0.9300
Epoch 2/30
174/174 [==============================] - 53s 304ms/step - loss: 0.3647 - accuracy: 0.9006 - val_loss: 0.0930 - val_accuracy: 0.9803
Epoch 3/30
174/174 [==============================] - 53s 304ms/step - loss: 0.1963 - accuracy: 0.9461 - val_loss: 0.0448 - val_accuracy: 0.9907
Epoch 4/30
174/174 [==============================] - 51s 291ms/step - loss: 0.1250 - accuracy: 0.9675 - val_loss: 0.0184 - val_accuracy: 0.9959
Epoch 5/30
174/174 [==============================] - 51s 292ms/step - loss: 0.0894 - accuracy: 0.9754 - val_loss: 0.0108 - val_accuracy: 0.9974
Epoch 6/30
174/174 [==============================] - 48s 278ms/step - loss: 0.0742 - accuracy: 0.9789 - val_loss: 0.0202 - val_accuracy: 0.9944
Epoch 7/30
174/174 [==============================] - 53s 303ms/step - loss: 0.0749 - accuracy: 0.9799 - val_loss: 0.0295 - val_accuracy: 0.9940
Epoch 8/30
174/174 [==============================] - 53s 305ms/step - loss: 0.1024 - accuracy: 0.9747 - val_loss: 0.0123 - val_accuracy: 0.9967
Epoch 9/30
174/174 [==============================] - 51s 292ms/step - loss: 0.0783 - accuracy: 0.9800 - val_loss: 0.0098 - val_accuracy: 0.9970
Epoch 10/30
174/174 [==============================] - 52s 301ms/step - loss: 0.0684 - accuracy: 0.9827 - val_loss: 0.0153 - val_accuracy: 0.9937
Epoch 11/30
174/174 [==============================] - 51s 293ms/step - loss: 0.0580 - accuracy: 0.9846 - val_loss: 0.0305 - val_accuracy: 0.9911
Epoch 12/30
174/174 [==============================] - 55s 319ms/step - loss: 0.0570 - accuracy: 0.9850 - val_loss: 0.0079 - val_accuracy: 0.9978
Epoch 13/30
174/174 [==============================] - 51s 291ms/step - loss: 0.0509 - accuracy: 0.9863 - val_loss: 0.0302 - val_accuracy: 0.9885
Epoch 14/30
174/174 [==============================] - 51s 291ms/step - loss: 0.1012 - accuracy: 0.9739 - val_loss: 0.0181 - val_accuracy: 0.9959
Epoch 15/30
174/174 [==============================] - 51s 294ms/step - loss: 0.0557 - accuracy: 0.9870 - val_loss: 0.0288 - val_accuracy: 0.9922
Epoch 16/30
174/174 [==============================] - 51s 293ms/step - loss: 0.0584 - accuracy: 0.9857 - val_loss: 0.0085 - val_accuracy: 0.9967
Epoch 17/30
174/174 [==============================] - 56s 320ms/step - loss: 0.0884 - accuracy: 0.9780 - val_loss: 0.0155 - val_accuracy: 0.9959
Epoch 18/30
174/174 [==============================] - 51s 293ms/step - loss: 0.0473 - accuracy: 0.9894 - val_loss: 0.0061 - val_accuracy: 0.9989
Epoch 19/30
174/174 [==============================] - 51s 296ms/step - loss: 0.0379 - accuracy: 0.9910 - val_loss: 0.0078 - val_accuracy: 0.9974
Epoch 20/30
174/174 [==============================] - 51s 292ms/step - loss: 0.0435 - accuracy: 0.9899 - val_loss: 0.0132 - val_accuracy: 0.9978
Epoch 21/30
174/174 [==============================] - 51s 293ms/step - loss: 0.0486 - accuracy: 0.9876 - val_loss: 0.0337 - val_accuracy: 0.9907
Epoch 22/30
174/174 [==============================] - 55s 319ms/step - loss: 0.0892 - accuracy: 0.9765 - val_loss: 0.0177 - val_accuracy: 0.9974
Epoch 23/30
174/174 [==============================] - 49s 281ms/step - loss: 0.0393 - accuracy: 0.9895 - val_loss: 0.0202 - val_accuracy: 0.9937
Epoch 24/30
174/174 [==============================] - 51s 294ms/step - loss: 0.0549 - accuracy: 0.9854 - val_loss: 0.0119 - val_accuracy: 0.9974
Epoch 25/30
174/174 [==============================] - 51s 293ms/step - loss: 0.0609 - accuracy: 0.9863 - val_loss: 0.0087 - val_accuracy: 0.9974
Epoch 26/30
174/174 [==============================] - 53s 306ms/step - loss: 0.0289 - accuracy: 0.9926 - val_loss: 0.0044 - val_accuracy: 0.9985
Epoch 27/30
174/174 [==============================] - 53s 306ms/step - loss: 0.0320 - accuracy: 0.9916 - val_loss: 0.0094 - val_accuracy: 0.9967
Epoch 28/30
174/174 [==============================] - 51s 293ms/step - loss: 0.0378 - accuracy: 0.9904 - val_loss: 0.0206 - val_accuracy: 0.9978
Epoch 29/30
174/174 [==============================] - 51s 293ms/step - loss: 0.0461 - accuracy: 0.9896 - val_loss: 0.0062 - val_accuracy: 0.9985
Epoch 30/30
174/174 [==============================] - 51s 292ms/step - loss: 0.0501 - accuracy: 0.9895 - val_loss: 0.0123 - val_accuracy: 0.9978


summary = pd.DataFrame(model.history.history)
summary


model.save("//content//drive//MyDrive//Colab Notebooks//lab3//Saved model//my_model.h5")

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
