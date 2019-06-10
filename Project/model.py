"""
The script used to create and train the model.
"""

from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Lambda, Conv2D, AveragePooling2D, Dropout


class BehavioralCloneModel:
    def __init__(self, input_shape):
        # Create the model
        self.model = Sequential()
        # Preprocess incoming data, centered around zero with small standard deviation
        self.model.add(Lambda(lambda x: x / 127.5 - 1.,
                              input_shape=input_shape,
                              output_shape=input_shape))

        # the model architecture is inspired by LeNet
        # Layer 1: Convolutional. Input = 32x32x3. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='elu'))
        self.model.add(AveragePooling2D())

        # Layer 2: Convolutional. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='elu'))
        self.model.add(AveragePooling2D())

        # Layer 3: Convolutional. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='elu'))
        self.model.add(AveragePooling2D())

        # Layer 4: Convolutional. Activation: Leaky ReLU
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu'))

        # Flatten
        self.model.add(Flatten())

        # Layer 5: Fully Connected. Activation: Leaky ReLU
        self.model.add(Dense(units=120, kernel_initializer='normal', activation='elu'))

        # Adding a dropout layer to avoid overfitting
        self.model.add(Dropout(0.2))

        # Layer 6: Fully Connected. Activation: Leaky ReLU
        self.model.add(Dense(units=84, kernel_initializer='normal', activation='elu'))

        # Layer 7: Fully Connected. One single neuron -  regression
        self.model.add(Dense(units=1, kernel_initializer='normal'))
