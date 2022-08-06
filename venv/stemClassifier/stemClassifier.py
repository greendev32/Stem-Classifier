# Stem Classifier
# A sequential CNN model used to classify stem zones based
# on a close-up input image of a strawberry.
# The model is trained on a dataset of just under 4,000
# images and is validated using about 800 images.
# This script includes a GUI that displays the results of the
# model running on new images from the test dataset.

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from future.moves import tkinter
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageTk
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class StemClassifier:
    def __init__(self):
        self.img_width = 0
        self.img_height = 0
        self.batch_size = 32
        self.epochs = 10

        class_names = ['0\u00B0 to 45\u00B0',
                       '45\u00B0 to 90\u00B0',
                       '90\u00B0 to 135\u00B0',
                       '135\u00B0 to 180\u00B0',
                       '180\u00B0 to 225\u00B0',
                       '225\u00B0 to 270\u00B0',
                       '270\u00B0 to 315\u00B0',
                       '315\u00B0 to 360\u00B0']
        self.class_names = np.array(class_names)
        self.num_classes = len(self.class_names)

    def loadDataset(self, dirName, width, height):
        self.img_width = width
        self.img_height = height

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            dirName,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            dirName,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

    def createModel(self):
        self.model = Sequential([
            layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes)
        ])

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.model.summary()

    def createResNet50Model(self):
        input = tf.keras.Input(shape=(self.img_height, self.img_width, 3))
        res_model = tf.keras.applications.resnet50.ResNet50(
                    include_top=False,
                    weights='imagenet',
                    input_tensor=input
        )

        for layer in res_model.layers[:143]:
            layer.trainable = False
            # Check the freezed was done ok
        for i, layer in enumerate(res_model.layers):
            print(i, layer.name, "-", layer.trainable)

        to_res = (self.img_height, self.img_width)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
        model.add(res_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        layers.Dense(self.num_classes)

        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        self.model = model

    def trainModel(self, modelName):
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        # saved to folder C:\Users\green\PycharmProjects\stemClassifier
        self.model.save(modelName)

    def nextButton(self):
        self.imgIdx += 1

        if self.imgIdx > (len(self.onlyfiles) - 1):
            self.imgIdx = 0

        self.updateImage()

    def previousButton(self):
        self.imgIdx -= 1

        if self.imgIdx < 0:
            self.imgIdx = (len(self.onlyfiles) - 1)

        self.updateImage()

    def updateImage(self):
        img = tf.keras.utils.load_img(self.imgDir + self.onlyfiles[self.imgIdx],
                                      target_size=(self.img_height, self.img_width))

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(score)], 100 * np.max(score))
        )

        self.img = ImageTk.PhotoImage(Image.open(self.imgDir + self.onlyfiles[self.imgIdx]))
        imgTk = tkinter.Label(self.content, image=self.img)

        self.imageLabel.destroy() # prevents stacking past labels on top of each other
        labelText = str(self.class_names[np.argmax(score)]) \
                    + " zone"
                    # + " zone with " \
                    # + str(round(100 * np.max(score))) \
                    # + "% confidence"
        self.imageLabel = tkinter.Label(self.content, text=labelText)

        imgTk.grid(column=1, row=0, columnspan=1, rowspan=1, padx=20, pady=20)
        self.imageLabel.grid(column=1, row=1, padx=20, pady=20)

    def classify(self, modelDirName, imgDir):
        # get all image files and model
        self.imgDir = imgDir
        self.onlyfiles = [f for f in listdir(imgDir) if isfile(join(imgDir, f))]
        random.shuffle(self.onlyfiles)
        self.model = tf.keras.models.load_model(modelDirName)
        self.imgIdx = -1
        #model.summary()

        # create a GUI
        self.root = tkinter.Tk()
        self.root.geometry("420x300")
        self.content = tkinter.Frame(self.root)
        self.content.grid(column=0, row=0)
        self.imageLabel = tkinter.Label(self.content, text="")

        # run model on first image
        self.nextButton()

        # define buttons
        button1 = tkinter.Button(self.content, text='Next', command=self.nextButton)
        button2 = tkinter.Button(self.content, text='Previous', command=self.previousButton)

        # put buttons in the display window
        button2.grid(column=0, row=0, padx=20, pady=20)
        button1.grid(column=2, row=0, padx=20, pady=20)

        self.root.title("Stem Classifier")
        self.root.mainloop()  # Start the GUI