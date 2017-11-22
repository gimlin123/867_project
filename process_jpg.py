from PIL import Image
import numpy as np
import os
# from keras.models import Sequential
# from keras.layers import Conv2D, Activation

from os import walk

import json



#pre-processing methods
def resample_picture(current_gsd, target_gsd, jpg_image):
    jpg_image_size = jpg_image.size
    resized_image = jpg_image.resize((int(jpg_image_size[0] * current_gsd / target_gsd), int(jpg_image_size[1] * current_gsd / target_gsd)))
    return resized_image

def crop(bot_left, top_right, jpg_image):
    # print "========"
    # print bot_left
    # print top_right
    # print "========================"
    cropped_image = jpg_image.crop(box=(bot_left[0], bot_left[1], top_right[0], top_right[1]))
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
    root = '../train'

    target_gsd = 1.7
    for (dirpath, dirnames, filenames) in walk(root):
        for dirname in dirnames:
            for (dirpath2, dirnames2, filenames2) in walk(os.path.join(root, dirname)):
                for dirname2 in dirnames2:
                    if int(dirname2.split('_')[-1]) < 100:
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

                            print img_size

                            cropped_img = crop((bounding_box[2] / 2, bounding_box[3] / 2), ((bounding_box[0] + img_size[0]) / 2, (bounding_box[1] + img_size[1]) / 2), img)
                            resized_img = resample_picture(meta['gsd'], 1.7, cropped_img)
                            print np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)
                            # print np.fromstring(resized_img.tobytes(), dtype=np.uint8)

    print gsds
    gsds.sort()
    print gsds[0]
    print gsds[-1]
#
# def create_neural_net(img_width, img_height):
#
#     model = Sequential()
#     model.add(keras.layers.Conv2D(100, (50,50), strides=(20, 20), padding='valid', data_format=None, dilation_rate=(1, 1), activation=Activation("relu"),
#     use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#     kernel_constraint=None, bias_constraint=None, input_shape = (img_width, img_height, 3)))
#     model.add(keras.layers.Conv2D(50, (10,10), strides=(5, 5), padding='valid', data_format=None, dilation_rate=(1, 1), activation=Activation("relu"),
#     use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#     kernel_constraint=None, bias_constraint=None))
#     model.add(Dense(26, activation='softmax'))

create_training_dataset()
>>>>>>> 3187f3b80cec5b5cf727c777326a1cd3dc47f051
