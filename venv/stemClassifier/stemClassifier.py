# Stem Classifier
# A sequential CNN model used to classify stem zones based
# on a close-up input image of a strawberry.
# The model is trained on a dataset of just under 4,000
# images and is validated using about 800 images.
# This script includes a GUI that displays the results of the
# model running on new images from the test dataset.

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from future.moves import tkinter
from io import BytesIO
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class StemClassifier:
    def __init__(self):
        self.img_width = 0
        self.img_height = 0
        self.batch_size = 32
        self.epochs = 20
        self.roi = False # ROI mode
        self.originalDataset = True

        # class_names = ['0\u00B0 to 45\u00B0',
        #                '135\u00B0 to 180\u00B0',
        #                '180\u00B0 to 225\u00B0',
        #                '225\u00B0 to 270\u00B0',
        #                '270\u00B0 to 315\u00B0',
        #                '315\u00B0 to 360\u00B0',
        #                '45\u00B0 to 90\u00B0',
        #                '90\u00B0 to 135\u00B0']

        class_names = ['0to45',
                       '135to180',
                       '180to225',
                       '225to270',
                       '270to315',
                       '315to360',
                       '45to90',
                       '90to135']

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
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
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
        accuracy = plt.subplot(1, 2, 1)
        accuracy.set_xlabel('Number of Epochs')
        accuracy.set_ylabel('Accuracy Percentage')
        plt.plot(epochs_range, acc, label='Training Accuracy', color="blue", linewidth=1, marker='.')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy', color="green", linewidth=1, marker='.')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        loss_plt = plt.subplot(1, 2, 2)
        loss_plt.set_xlabel('Number of Epochs')
        loss_plt.set_ylabel('Loss')
        plt.plot(epochs_range, loss, label='Training Loss', color="blue", linewidth=1, marker='.')
        plt.plot(epochs_range, val_loss, label='Validation Loss', color="green", linewidth=1, marker='.')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        # saved to folder C:\Users\green\PycharmProjects\stemClassifier
        self.model.save(modelName)

    def nextButton(self):
        self.imgIdx += 1

        if self.imgIdx > (len(self.onlyfiles) - 1):
            self.imgIdx = 0

        if self.roi:
            self.updateImageROI()
        else:
            self.updateImage()

    def previousButton(self):
        self.imgIdx -= 1

        if self.imgIdx < 0:
            self.imgIdx = (len(self.onlyfiles) - 1)

        if self.roi:
            self.updateImageROI()
        else:
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

        self.img = Image.open(self.imgDir + self.onlyfiles[self.imgIdx])
        self.img  = self.img.resize((400, 400)) # display all images as 400x400px
        self.img = ImageTk.PhotoImage(self.img)
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

    def updateImageROI(self):
        fileName = self.imgDir + self.onlyfiles[self.imgIdx]
        self.img = cv.imread(fileName)  # store original image
        imgDraw = self.img.copy()  # cloned image for drawing purposes

        dilation_size = 5
        erosion_size = 3

        erosion_element = cv.getStructuringElement(cv.MORPH_RECT, (erosion_size, erosion_size))
        dilation_element = cv.getStructuringElement(cv.MORPH_RECT, (dilation_size, dilation_size))

        eroded = cv.erode(imgDraw, erosion_element, iterations=1)
        dilated = cv.dilate(eroded, dilation_element, iterations=1)

        b, g, r = cv.split(dilated)

        # amplify red color
        r = cv.multiply(r, 0.8)

        # reduce blue and green colors
        redProcess = cv.subtract(r, b)
        redProcess = cv.subtract(redProcess, g)

        # threshold using red color
        ret, thresh = cv.threshold(redProcess, 20, 255,
                                   cv.THRESH_BINARY)

        # find red contours
        contours, hierarchies = cv.findContours(
            thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        contourFound = False
        if self.originalDataset:
            # draw valid contours
            for contour in contours:
                moment = cv.moments(contour)
                if moment['m00'] != 0:
                    Array_Size = len(contour)
                    # center of the contour
                    cx = int(moment['m10'] / moment['m00'])
                    cy = int(moment['m01'] / moment['m00'])
                    if (cv.contourArea(contour) > 100) and (cy-100 >= 0) and (cy+100 <= 1280) and (cx-100 >= 0) and (cx+100 <= 720):
                        #cv.drawContours(imgDraw, [contour], -1, (0, 255, 0), 2)
                        roi = self.img[cy - 100:cy + 100, cx - 100:cx + 100]
                        cv.imwrite("temp.png", roi)

                        img = tf.keras.utils.load_img("temp.png",
                                                      target_size=(200, 200))

                        img_array = tf.keras.utils.img_to_array(img)
                        img_array = tf.expand_dims(img_array, 0)  # Create a batch

                        predictions = self.model.predict(img_array)
                        score = tf.nn.softmax(predictions[0])
                        labelText = str(self.class_names[np.argmax(score)])

                        # draw a rectangle centering on each contour
                        cv.rectangle(self.img, (cx-100, cy-100), (cx+100, cy+100), (255, 0, 0), 2)
                        # draw rectangle for label background
                        cv.rectangle(self.img, (cx - 101, cy - 130), (cx + 70, cy - 100), (255, 0, 0), -1)
                        contourFound = True

                        # put label on each ROI
                        pil_image = Image.fromarray(self.img)
                        draw = ImageDraw.Draw(pil_image)
                        font_file = open("C:\Windows\Fonts\\trebucbd.ttf", "rb")
                        bytes_font = BytesIO(font_file.read())
                        draw.text((cx-90, cy-128),
                                  labelText,
                                  font=ImageFont.truetype(bytes_font, 24),
                                  fill="#00FF00")
                        self.img = np.asarray(pil_image)
        else:
            areas = [cv.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            contour = contours[max_index]
            moment = cv.moments(contour)
            if moment['m00'] != 0:
                # center of the contour
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])

                roi = self.img[cy - 200:cy + 200, cx - 200:cx + 200]
                cv.imwrite("temp.png", roi)

                img = tf.keras.utils.load_img("temp.png",
                                              target_size=(400, 400))

                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)  # Create a batch

                predictions = self.model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                labelText = str(self.class_names[np.argmax(score)])

                cv.rectangle(self.img, (cx - 200, cy - 200), (cx + 200, cy + 200), (255, 0, 0), 2)
                cv.rectangle(self.img, (cx - 201, cy - 230), (cx - 30, cy - 200), (255, 0, 0), -1)
                contourFound = True

                # put label on each ROI
                pil_image = Image.fromarray(self.img)
                draw = ImageDraw.Draw(pil_image)
                font_file = open("C:\Windows\Fonts\\trebucbd.ttf", "rb")
                bytes_font = BytesIO(font_file.read())
                draw.text((cx - 190, cy - 228),
                          labelText,
                          font=ImageFont.truetype(bytes_font, 24),
                          fill="#00FF00")
                self.img = np.asarray(pil_image)

        if contourFound:
            self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
            if self.originalDataset == False:
                self.img = self.img[200:1480, 200:920]

            self.img = Image.fromarray(self.img)
            self.img = self.img.resize((360, 640))  # display all images as 360x640px
            self.img = ImageTk.PhotoImage(self.img)
            imgTk = tkinter.Label(self.content, image=self.img)
            imgTk.grid(column=1, row=0, columnspan=1, rowspan=1, padx=20, pady=20)
        else:
            # remove and skip images with no ROIs
            del self.onlyfiles[self.imgIdx]
            self.imgIdx += 1
            self.nextButton()

    def classify(self, modelDirName, imgDir, imgWidth, imgHeight):
        # get all image files and model
        self.imgDir = imgDir
        self.img_width = imgWidth
        self.img_height = imgHeight
        self.onlyfiles = [f for f in listdir(imgDir) if isfile(join(imgDir, f))]
        random.shuffle(self.onlyfiles)
        self.model = tf.keras.models.load_model(modelDirName)
        self.imgIdx = -1
        #model.summary()

        # create a GUI
        self.root = tkinter.Tk()
        self.root.geometry("615x500")
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

    def classifyROI(self, modelDirName, imgDir, originalDataset):
        # get all image files and model
        self.imgDir = imgDir
        self.originalDataset = originalDataset
        self.onlyfiles = [f for f in listdir(imgDir) if isfile(join(imgDir, f))]
        random.shuffle(self.onlyfiles)
        self.model = tf.keras.models.load_model(modelDirName)
        self.imgIdx = -1
        self.roi = True # enable ROI mode

        # create a GUI
        self.root = tkinter.Tk()
        self.root.geometry("575x700")
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