from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Activation


#pre-processing methods
def resample_picture(current_gsd, target_gsd, jpg_image):
    jpg_image_size = jpg_image.size
    resized_image = jpg_image.resize((int(jpg_image_size[0] * current_gsd / target_gsd), int(jpg_image_size[1] * current_gsd / target_gsd)))
    return resized_image

def crop(bot_left, top_right, jpg_image):
    cropped_image = jpg_image.crop((bot_left[0], top_right[1], top_right[0], bot_left[1]))
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
