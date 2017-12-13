from PIL import Image
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense

from os import walk
import pickle

import json



#pre-processing methods
def resample_picture(current_gsd, target_gsd, jpg_image):
    jpg_image_size = jpg_image.size
    resized_image = jpg_image.resize(size = (int(jpg_image_size[0] * current_gsd / target_gsd), int(jpg_image_size[1] * current_gsd / target_gsd)))
    return resized_image

def crop(top_left, size, jpg_image):
    cropped_image = jpg_image.crop(box=(top_left[0], top_left[1], top_left[0] + size[0], top_left[1] + size[1]))
    # cropped_image = jpg_image.crop(box=(0, 0, 100, 100))
    return cropped_image

def openJPGImg(img_name):
    return Image.open(img_name)

def create_neural_net(img_width, img_height, dropout_rate):

    model = Sequential()
    model.add(keras.layers.Conv2D(100, (50,50), strides=(20, 20), padding='valid', data_format=None, dilation_rate=(1, 1), activation=Activation("relu"),
    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, input_shape = (img_width, img_height, 3)))
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)

    model.add(keras.layers.Conv2D(50, (10,10), strides=(5, 5), padding='valid', data_format=None, dilation_rate=(1, 1), activation=Activation("relu"),
    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None))
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    model.add(keras.layers.flatten())
    model.add(keras.layers.Dropout(dropout_rate, noise_shape=None, seed=None))
    model.add(Dense(100, activation = "reLu"))
    model.add(keras.layers.Dropout(dropout_rate, noise_shape=None, seed=None))
    model.add(Dense(62, activation='softmax'))

def train_neural_net(neural_net):
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)


# max gsd seem to be around 1.7 for this dataset
def create_training_dataset():
    root = 'D:/fMoW-rgb/val'
    target_gsd = 1.7
    max_width = 0
    max_height = 0

    dict = {}
    index = 0


    for (dirpath, dirnames, filenames) in walk(root):
        for dirname in dirnames:
            training_data = []
            for (dirpath2, dirnames2, filenames2) in walk(os.path.join(root, dirname)):
                for dirname2 in dirnames2:
                    if int(dirname2.split('_')[-1]) <= 100:
                        print (dirname2)
                        for (dirpath3, dirnames3, filenames3) in walk(os.path.join(root, dirname, dirname2)):
                            img_filename = ""
                            meta_filename = ""
                            for filename3 in filenames3:
                                if filename3.split("_")[-1] == 'rgb.jpg':
                                    img_filename = filename3
                                    meta_filename = img_filename[:-4] + '.json'
                                    break

                            meta = json.load(open(os.path.join(root, dirname, dirname2, meta_filename)))
                            bounding_box = meta['bounding_boxes'][0]['box']
                            img = openJPGImg(os.path.join(root, dirname, dirname2, img_filename))
                            img_size = img.size

                            img = crop((bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), img)
                            img = resample_picture(meta['gsd'], 1.7, img)
                            if img.size[0] > max_width:
                                max_width = img.size[0]

                            if img.size[1] > max_height:
                                max_height = img.size[1]


            training_data = np.array(training_data)
            np.save("processed_data/resized_img_data_val_" + dirname, training_data)

        for i in range(24, 26):
            training_data = [[]] * 6
            label_arr = [[]] * 6
            # print i
            for dirname in dirnames:
                # print dirname
                x_sizes = []
                y_sizes = []

                for (dirpath2, dirnames2, filenames2) in walk(os.path.join(root, dirname)):

                    for dirname2 in dirnames2:
                        dir_num = int(dirname2.split('_')[-1])
                        if dir_num == 2*i or dir_num == 2*i + 1:
                            for (dirpath3, dirnames3, filenames3) in walk(os.path.join(root, dirname, dirname2)):
                                img_filename = ""
                                meta_filename = ""
                                for filename3 in filenames3:
                                    if filename3.split("_")[-1] == 'rgb.jpg':
                                        img_filename = filename3
                                        meta_filename = img_filename[:-4] + '.json'
                                        break

                                meta = json.load(open(os.path.join(root, dirname, dirname2, meta_filename)))
                                bounding_box = meta['bounding_boxes'][0]['box']
                                category = meta['bounding_boxes'][0]['category']

                                img = openJPGImg(os.path.join(root, dirname, dirname2, img_filename))
                                img_size = img.size

                                img = crop((bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), img)
                                img = resample_picture(meta['gsd'], 1.7, img)

                                dim = max(img.size[0], img.size[1]) / 500
                                padded_data = np.zeros(((dim + 1) * 500, (dim + 1) * 500, 3))
                                padded_data[:img.size[1], :img.size[0] , :] = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
                                training_data[dim].append(padded_data)
                                if category in dict:
                                    vec = np.zeros(62)
                                    vec[dict[category]] = 1
                                    label_arr[dim].append(vec)
                                else:
                                    dict[category] = index
                                    index += 1
                                    vec = np.zeros(62)
                                    vec[dict[category]] = 1
                                    label_arr[dim].append(vec)
            if len(training_data[0]) > 0:
                np.save("padded_data/resized_img_data_training_" + str(i), training_data)
                np.save("padded_data_labels/resized_img_data_training_" + str(i), label_arr)


def make_training_matrices():
    root = './processed_data'
    training_data = [[]] * 6
    labels = [[]] * 6

    dict = {}
    index = 0
    for (dirpath, dirnames, filenames) in walk(root):
        for filename in filenames:
            if filename[-3:] == 'npy':
                if filename.split('_')[-1] not in dict:
                    dict[filename.split('_')[-1]] = index
                    index += 1

        for filename in filenames:

            training_data.append(np.load(os.path.join(root, filename)))
            labels.append(dict[filename.split('_')[-2]])

        training_data = np.array(training_data)
        training_labels = np.zeros((len(labels), len(dict)))

        training_labels[np.arange(len(labels)), labels] = 1

            loaded_data = np.load(os.path.join(root, filename))
            for i in range(len(loaded_data)):
                training_data[i] += loaded_data[i]
                label = [0] * len(dict)
                label[dict[filename.split('_')[-1]]] = 1
                labels[i] += [label] * len(loaded_data[i])


        np.save('training_data', training_data)
        np.save('training_labels', training_labels)


model = Sequential()
model.add(keras.layers.Conv2D(5, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation="relu",
use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
kernel_constraint=None, bias_constraint=None, input_shape = (500, 500, 3)))
model.add(keras.layers.Conv2D(5, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation="relu",
use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
kernel_constraint=None, bias_constraint=None))
model.add(keras.layers.Conv2D(5, (5,5), strides=(2, 2), padding='valid', data_format=None, dilation_rate=(1, 1), activation="relu",
use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
kernel_constraint=None, bias_constraint=None))
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(100, activation = "relu"))
model.add(keras.layers.Dropout(0.5, noise_shape=None, seed=None))
model.add(Dense(62, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
for j in range(3):
    for i in range(50):
        x_train = np.load('padded_data/' + 'resized_img_data_training_' + str(i) + '.npy', encoding='latin1')[0]
        y_train = np.load('padded_data_labels/' + 'resized_img_data_training_' + str(i) + '.npy')[0]

        y_train_small = np.array([y_train[i] for i in range(len(x_train)) if x_train[i].shape[0] == 500])
        x_train_small = np.array([arr for arr in x_train if arr.shape[0] == 500])

        print(y_train_small.shape)
        print(x_train_small.shape)
        model.fit(x_train_small, y_train_small,
                  batch_size=5,
                  epochs=1,
                  verbose=1)
        model.save('trained_model_1.h5')




