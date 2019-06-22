"""
The script used to read data from csv file,
load images and measurements
"""

import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import sklearn


class DataReader:
    """ DataReader class is a set of functions required for manipulating with data.  """

    def __init__(self):
        self.image_shape = (160, 320, 3)
        self.img_size_checked = False
        self.train_samples_size = 0
        self.validation_samples_size = 0

    def read_csv_file(self, file):
        '''read and return lines from input csv file'''
        lines = []
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
        return lines

    def visualize_data(self, frames, angles):
        '''
        Process input video frame by frame
        '''
        for frame, angle in zip(frames, angles):
            cv2.putText(frame, str(angle), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

    def read_using_generator(self, file, path, batch_size=32, debug=False):
        lines = self.read_csv_file(file)
        train_samples, validation_samples = train_test_split(lines, test_size=0.2)
        self.train_samples_size = len(train_samples)
        self.validation_samples_size = len(validation_samples)
        if debug:
            print("Train set size is " + str(self.train_samples_size))
            print("Validation set size is " + str(self.validation_samples_size))
        # compile and train the model using the generator function
        train_generator = self.__generator(train_samples, path, batch_size=batch_size)
        validation_generator = self.__generator(validation_samples, path, batch_size=batch_size)
        return train_generator, validation_generator

    def __generator(self, samples, path, batch_size=32, angle_correction = 0.2):
        '''produces a series of data sets and their labels.'''
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                angles = []  # steering angles
                for batch_sample in batch_samples:
                    # read images
                    path_to_img = 'IMG/'
                    center_name = path + path_to_img + batch_sample[0].split('/')[-1]
                    img_center = cv2.imread(center_name)
                    left_name = path + path_to_img + batch_sample[1].split('/')[-1]
                    img_left = cv2.imread(left_name)
                    right_name = path + path_to_img + batch_sample[2].split('/')[-1]
                    img_right = cv2.imread(right_name)
                    # set angles
                    center_angle = float(batch_sample[3])
                    steering_left = center_angle + angle_correction
                    steering_right = center_angle - angle_correction
                    # add images and angles to data set
                    images.extend([img_center, img_left, img_right])
                    angles.extend([center_angle, steering_left, steering_right])

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

def test_generator():
    path = 'data/clockwise_add/'
    file_name = path + 'driving_log.csv'
    reader = DataReader()
    # Set our batch size
    batch_size = 32
    # compile and train the model using the generator function
    [train_generator, validation_generator] = reader.read_using_generator(file_name, path, batch_size=batch_size, debug=True)
    # call on the generator iterator
    [X_train, y_train] = next(train_generator)
    # print some information about data
    print("Image dimension is: {0}x{1}x{2}".format(str(X_train[0].shape[0]), str(X_train[0].shape[1]),
                                                   str(X_train[0].shape[2])))

def visualize_data():
    path = 'data/clockwise_1/'
    file_name = path + 'driving_log.csv'
    reader = DataReader()
    # Set our batch size
    batch_size = 32
    # compile and train the model using the generator function
    [train_generator, validation_generator] = reader.read_using_generator(file_name, path, batch_size=batch_size, debug=True)
    # call on the generator iterator
    [X_train, y_train] = next(train_generator)
    # store frames as video
    reader.visualize_data(X_train, y_train)

if __name__ == '__main__':
    visualize_data()
