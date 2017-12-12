from PIL import Image
import numpy as np
import os
# import keras
# from keras.models import Sequential
# from keras.layers import Conv2D, Activation, Dense

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
    root = '../train'
    target_gsd = 1.7
    max_width = 0
    max_height = 0


    # 2036 and 2321
    # everything will be padded to size 2500 and 2500, since all pictures in training size should be less than this
    for (dirpath, dirnames, filenames) in walk(root):
        for dirname in dirnames:
            training_data = [[]] * 6
            print dirname
            x_sizes = []
            y_sizes = []

            for (dirpath2, dirnames2, filenames2) in walk(os.path.join(root, dirname)):

                for dirname2 in dirnames2:

                    if int(dirname2.split('_')[-1]) < 200:
                        print dirname2
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
                            # num = int(dirname2.split('_')[-1])
            #                 x_sizes.append((img.size[0], num))
            #                 y_sizes.append((img.size[1], num))
            # print (min(x_sizes), max(x_sizes))
            # print (min(y_sizes), max(y_sizes))
                            # if img.size[0] > max_width:
                            #     max_width = img.size[0]
                            #
                            # if img.size[1] > max_height:
                            #     max_height = img.size[1]
                            # print len(list(resized_img.getdata()))
                            dim = max(img.size[0], img.size[1]) / 500
                            padded_data = np.zeros(((dim + 1) * 500, (dim + 1) * 500, 3))
                            padded_data[:img.size[1], :img.size[0] , :] = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
                            # training_data.append(np.array(img.getdata()).reshape(img.size[1], img.size[0], 3))
                            # training_data.append(np.array(resized_img.getdata()).reshape(resized_img.size[1], resized_img.size[0], 3))
                            training_data[dim].append(padded_data)
                            # # print np.fromstring(resized_img.tobytes(), dtype=np.uint8)

            np.save("padded_data/resized_img_data_training_" + dirname, training_data)

create_training_dataset()

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
            # print filename
            loaded_data = np.load(os.path.join(root, filename))
            for i in range(len(loaded_data)):
                training_data[i] += loaded_data[i]
                label = [0] * len(dict)
                label[dict[filename.split('_')[-1]]] = 1
                labels[i] += [label] * len(loaded_data[i])
            print training_data
            print labels

        # training_data = np.array(training_data)
        # training_labels = np.zeros((len(labels), len(dict)))
        #
        # training_labels[np.arange(len(labels)), labels] = 1

        np.save('training_data', training_data)
        np.save('training_labels', training_labels)
        print dict


# make_training_matrices()
# a = np.load('resized_img_data_training_airport.npy')
# print a.shape
# #

# model = Sequential()
# model.add(keras.layers.Conv2D(5, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=Activation("relu"),
# use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
# kernel_constraint=None, bias_constraint=None, input_shape = (None, None, 3)))
# model.add(keras.layers.Conv2D(5, (3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=Activation("relu"),
# use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
# kernel_constraint=None, bias_constraint=None))
# model.add(Dense(62, activation='softmax'))
#
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# x_train = np.load('training_data.npy', encoding='latin1')
# # print(x_train)
# y_train = np.load('training_labels.npy')
# print(len(x_train))
# print(x_train[0])
# print(x_train[0].shape)
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           verbose=1)
#
# score = model.evaluate(x_train, y_train, verbose=0)
# model.save('trained_model_1')

# create_training_dataset()
