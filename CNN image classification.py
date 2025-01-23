import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import layers, models
import matplotlib.pyplot as plt

def loadData(dataset):
    images = []
    labels = []
    for item in glob.glob(dataset):
        img = cv2.imread(item)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        images.append(img)
        label = item.split("\\")[-2]
        labels.append(label)

    images = np.array(images)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
    return x_train, x_test, y_train, y_test

def cnnModel():
    cnnModel = models.Sequential([
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 3)),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(11, activation="softmax")
    ])
    cnnModel.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    return cnnModel

def show_result(finalModel):
    plt.style.use('ggplot')
    plt.plot(finalModel.history["accuracy"], label="train accuracy")
    plt.plot(finalModel.history["val_accuracy"], label="test accuracy")
    plt.plot(finalModel.history["loss"], label="train_loss")
    plt.plot(finalModel.history["val_loss"], label="test_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss/accuracy")
    plt.title('Digit Classification')
    plt.show()

x_train, x_test, y_train, y_test = loadData("dataset\\*\\*")
cnn = cnnModel()
print(cnn.summary())
finalModel = cnn.fit(x=x_train, y=y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
show_result(finalModel)
print(y_train.shape)
print(y_test.shape)
# # print(x_train)
cnn.save("FinalModel64.keras")