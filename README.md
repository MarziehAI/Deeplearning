Deep Learning Traffic Sign Classification Project
This project aims to classify traffic signs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The dataset contains 43 different classes of traffic signs.

Project Structure
Data Collection and Pre-processing

Load and resize images
Convert images and labels to numpy arrays
Split data into training and test sets
Model Building and Training

Build a CNN model
Train the model
Evaluate the model's performance
Results Visualization

Plot training and validation accuracy
Plot training and validation loss
Getting Started
Prerequisites
Python 3.6 or higher
Required packages: numpy, pandas, matplotlib, tensorflow, PIL, scikit-learn, keras
Installation
Install the required packages using pip:

bash
Copy code
pip install numpy pandas matplotlib tensorflow pillow scikit-learn keras
Data
Ensure the dataset is placed in the appropriate directory. The dataset should have subdirectories for each class under a main directory called Train.

Running the Project
Data Pre-processing

Retrieve the images and their labels, resize them to (30, 30), and convert them to numpy arrays:

python
Copy code
import numpy as np
from PIL import Image
import os

data = []
labels = []
classes = 43
cur_path = os.getcwd()

for i in range(classes):
    path = os.path.join(cur_path, "//content//drive//MyDrive//Colab Notebooks//assignment//Train", str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '//' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

data = np.array(data)
labels = np.array(labels)
Data Splitting

Split the data into training and test sets:

python
Copy code
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
Model Building

Build the CNN model:

python
Copy code
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
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
Model Training

Train the model:

python
Copy code
epochs = 30
history = model.fit(X_train, y_train, batch_size=62, epochs=epochs, validation_data=(X_test, y_test))
Model Evaluation

Evaluate the model's performance and save it:

python
Copy code
import pandas as pd
import matplotlib.pyplot as plt

model.save("//content//drive//MyDrive//Colab Notebooks//lab3//Saved model//my_model.h5")

# Plotting accuracy and loss
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
Results
The model is trained for 30 epochs.
Accuracy and loss are plotted for both training and validation sets.
The model is saved to a file for future use.
