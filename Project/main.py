import cv2
import pickle
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import keras

import model as model
import data_reader as dread


class LearningParams:
    '''parameters for model training'''
    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 5


def learnig_pipeline():
    # 0. load parameters
    params = LearningParams()
    # 1. load the data
    paths_list = ['data/clockwise/', 'data/clockwise_1/', 'data/counterclockwise/', 'data/clockcouterwise_1/',
                  'data/counterclockwise_recovery/', 'data/clockwise_recovery/']
    reader = dread.DataReader()
    # compile and train the model using the generator function
    [train_generator, validation_generator] = reader.read_using_generator(paths_list, batch_size=params.batch_size,
                                                                          debug=False)
    input_shape = (reader.image_shape[2], reader.image_shape[0], reader.image_shape[1])
    # 2. Create the model
    clone_model = model.BehavioralCloneModel(reader.image_shape)

    for layer in clone_model.model.layers:
        print(layer.output_shape)

    # 2.2 Compile the model
    clone_model.model.compile(loss='mse', optimizer='adam')
    steps_per_epoch = math.ceil(reader.train_samples_size / params.batch_size)
    validation_steps = math.ceil(reader.validation_samples_size / params.batch_size)

    save_path = "model_checkpoints/saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)

    history_object = clone_model.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                                                     callbacks=[checkpoint],
                                                     validation_data=validation_generator,
                                                     validation_steps=validation_steps, epochs=params.num_epochs,
                                                     verbose=1)

    # saving model
    clone_model.model.save('model.h5')

    # 3. Model visualization. Print the keys contained in the history object
    print(history_object.history.keys())

    # 3.1 Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def store_model_weights():
    x = keras.models.load_model('model.h5')  # runs only on the source machine
    x.save_weights('weights.h5')

if __name__ == '__main__':
    learnig_pipeline()
    store_model_weights()
