from PIL import Image
import numpy as np
import os
# from keras.models import Sequential
# from keras.layers import Conv2D, Activation

from os import walk
import pickle

import json



#pre-processing methods
def resample_picture(current_gsd, target_gsd, jpg_image):
    jpg_image_size = jpg_image.size
    print (int(jpg_image_size[0] * current_gsd / target_gsd), int(jpg_image_size[1] * current_gsd / target_gsd))
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
    model.add(Dense(26, activation='softmax'))

def train_neural_net(neural_net):
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)


# max gsd seem to be around 1.7 for this dataset
def create_training_dataset():
    root = 'D:/fMoW-rgb/val'
    target_gsd = 1.7
    max_width = 0
    max_height = 0


    # 2036 and 2321
    # everything will be padded to size 2500 and 2500, since all pictures in training size should be less than this
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
                            # # print len(list(resized_img.getdata()))
                            # padded_data = np.zeros((2500, 2500, 3))
                            # padded_data[:resized_img.size[1], :resized_img.size[0] , :] = np.array(resized_img.getdata()).reshape(resized_img.size[1], resized_img.size[0], 3)
                            # training_data.append(padded_data)
                            # training_data.append(np.array(resized_img.getdata()).reshape(resized_img.size[1], resized_img.size[0], 3))
                            # # print np.fromstring(resized_img.tobytes(), dtype=np.uint8)

            training_data = np.array(training_data)
            np.save("processed_data/resized_img_data_val_" + dirname, training_data)


def make_training_matrices():
    root = './processed_data'
    training_data = []
    labels = []

    dict = {}
    index = 0
    for (dirpath, dirnames, filenames) in walk(root):
        for filename in filenames:
            if filename[-3:] == 'npy':
                if filename.split('_')[-2] not in dict:
                    dict[filename.split('_')[-2]] = index
                    index += 1

        for filename in filenames:
            print (filename)
            print (np.load(os.path.join(root, filename)))
            training_data.append(np.load(os.path.join(root, filename)))
            labels.append(dict[filename.split('_')[-2]])

        training_data = np.array(training_data)
        training_labels = np.zeros((len(labels), len(dict)))

        training_labels[np.arange(len(labels)), labels] = 1

        np.save('training_data', training_data)
        np.save('training_labels', training_labels)


# a = np.load('resized_img_data_training_airport.npy')
# print a.shape
# #
def create_neural_net():

    model = Sequential()
    model.add(keras.layers.Conv2D(5, (50,50), strides=(20, 20), padding='valid', data_format=None, dilation_rate=(1, 1), activation=Activation("relu"),
    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, input_shape = (None, None, 3)))
    model.add(keras.layers.Conv2D(5, (10,10), strides=(5, 5), padding='valid', data_format=None, dilation_rate=(1, 1), activation=Activation("relu"),
    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None))
    model.add(Dense(26, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


