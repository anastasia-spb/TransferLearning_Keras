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
        self.image_shape = (90, 320, 3)
        self.img_size_checked = False
        self.train_samples_size = 0
        self.validation_samples_size = 0

    @staticmethod
    def trim_image(img, bottom_crop=20, top_crop=50):
        '''crop image top and bottom parts. This function changes the image HEIGHT!'''
        height, width, channels = img.shape
        crop_img = img[top_crop:height - bottom_crop, :, :]
        return crop_img

    def read_csv_file(self, file):
        '''read and return lines from input csv file'''
        lines = []
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
        return lines

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

    def __generator(self, samples, path, batch_size=32):
        '''produces a series of data sets and their labels.'''
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                angles = []  # steering angles
                for batch_sample in batch_samples:
                    name = path + 'IMG/' + batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    # trim image to only see section with road
                    center_image = self.trim_image(center_image)
                    if not self.img_size_checked: # check that image shape is the same as expected (default trim parameters)
                        [image_height, image_width, image_channels] = center_image.shape
                        if center_image.shape != self.image_shape:
                            print("Image size differs from the expected one.")
                            raise
                        self.img_size_checked = True
                    images.append(center_image)
                    angles.append(center_angle)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)


def test_generator():
    path = 'data/clockwise_recovery/'
    file_name = path + 'driving_log.csv'
    reader = DataReader()
    # Set our batch size
    batch_size = 32
    # compile and train the model using the generator function
    [train_generator, validation_generator] = reader.read_using_generator(file_name, path, batch_size=batch_size, debug=True)
    # call on the generator iterator
    next(train_generator)
    # print some information about data
    print("Image dimension is: {0}x{1}x{2}".format(str(reader.image_shape[0]), str(reader.image_shape[1]),
                                                   str(reader.image_shape[2])))


if __name__ == '__main__':
    test_generator()
