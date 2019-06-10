"""
The script used to read data from csv file,
load images and measurements
"""

import cv2

class DataReader:
    """ DataReader class reads and stores images and measurements. And converts them into training data and labels. """
    def __init__(self):
        """ Create an empty lists of images and measurements. """
        self.measurements = []
        self.images = []


def test_data_reader():
    reader = DataReader()


if __name__ == '__main__':
    test_data_reader()
