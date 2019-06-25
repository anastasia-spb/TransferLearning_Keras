"""
The script used to create the model.
"""

from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D


class BehavioralCloneModel:
    """ BehavioralCloneModel_4CNN. Creates the convolutional network model with 4 CNN layers.  """
    def __init__(self, input_shape, top_crop=60, bottom_crop=20, drop_rate=0.5):
        # Create the model
        self.model = Sequential()
        self.model.add(Cropping2D(cropping=((top_crop, bottom_crop), (0, 0)),
                                  input_shape=input_shape))
        # now model.output_shape == (None, 80, 320, 3)
        # Preprocess incoming data, centered around zero with small standard deviation
        self.model.add(Lambda(lambda x: x / 127.5 - 1.))

        # the model architecture is inspired by LeNet
        # Layer 1: Convolutional. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu'))
        self.model.add(MaxPooling2D())

        # Layer 2: Convolutional. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu'))
        self.model.add(MaxPooling2D())

        # Layer 3: Convolutional. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu'))
        self.model.add(MaxPooling2D())

        # Layer 4: Convolutional. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
        self.model.add(MaxPooling2D())

        # Flatten
        self.model.add(Flatten())

        # Layer 5: Fully Connected. Activation: Leaky ReLU
        self.model.add(Dense(units=100, kernel_initializer='normal', activation='elu'))

        # Adding a dropout layer to avoid overfitting
        self.model.add(Dropout(drop_rate))

        # Layer 6: Fully Connected. Activation: Leaky ReLU
        self.model.add(Dense(units=50, kernel_initializer='normal', activation='elu'))

        # Adding a dropout layer to avoid overfitting
        self.model.add(Dropout(drop_rate))

        # Layer 6: Fully Connected. Activation: Leaky ReLU
        self.model.add(Dense(units=10, kernel_initializer='normal', activation='elu'))

        # Layer 7: Fully Connected. One single neuron -  regression
        self.model.add(Dense(units=1, kernel_initializer='normal'))


class BehavioralCloneModel_5CNN:
    """ BehavioralCloneModel_5CNN. Creates the convolutional network model with 5 CNN layers.  """
    def __init__(self, input_shape, top_crop=60, bottom_crop=20, drop_rate=0.5):
        # Create the model
        self.model = Sequential()
        self.model.add(Cropping2D(cropping=((top_crop, bottom_crop), (0, 0)),
                                  input_shape=input_shape))
        # now model.output_shape == (None, 80, 320, 3)
        # Preprocess incoming data, centered around zero with small standard deviation
        self.model.add(Lambda(lambda x: x / 127.5 - 1.))

        # the model architecture is inspired by LeNet
        # Layer 1: Convolutional. Input = 3@80x320. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu'))

        # Layer 2: Convolutional. Input = 24@76x316. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu'))

        # Layer 3: Convolutional. Input = 36@72x312. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu'))

        # Layer 4: Convolutional. Input = 48@68x308. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))

        # Layer 5: Convolutional. Input = 64@66x306. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))

        # Flatten. Input = 64@64x304.
        self.model.add(Flatten())

        # Layer 5: Fully Connected. Input = 1245184. Activation: Leaky ReLU
        self.model.add(Dense(units=100, kernel_initializer='normal', activation='elu'))

        # Adding a dropout layer to avoid overfitting
        self.model.add(Dropout(drop_rate))

        # Layer 6: Fully Connected. Activation: Leaky ReLU
        self.model.add(Dense(units=50, kernel_initializer='normal', activation='elu'))

        # Adding a dropout layer to avoid overfitting
        self.model.add(Dropout(drop_rate))

        # Layer 6: Fully Connected. Activation: Leaky ReLU
        self.model.add(Dense(units=10, kernel_initializer='normal', activation='elu'))

        # Layer 7: Fully Connected. One single neuron -  regression
        self.model.add(Dense(units=1, kernel_initializer='normal'))