import csv
import cv2
import numpy as np
import glob
from sklearn import preprocessing
import tensorflow as tf
import re
import os
from datetime import datetime
import sys
import shutil
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import math
from sklearn import metrics


class Sdc:

    class DataGenerator():
        def __init__(self):
            self.xx = "hamid"

    def __init__(self):
        self.text = ""

    @staticmethod
    def import_simulated_images(data_path, csv_file, limit=0, augment_data=False):
        csv_lines = []

        with open(csv_file) as csv_file:
            reader = csv.reader(csv_file)

            for csv_line in reader:
                csv_lines.append(csv_line)

        center_images = []
        left_images = []
        right_images = []
        steering_angles = []
        exception_count = 0

        limit_counter = 0

        for line_segments in csv_lines:
            # print(line_segments)

            try:
                steering_angle = float(line_segments[3])
                steering_angles.append(steering_angle)

                image_file = data_path + line_segments[0]
                # use this line in case this code runs on the server
                # (center_image_path, center_image_filename) = os.path.split(center_image_file)

                image = cv2.imread(image_file)
                center_images.append(image)

                if (augment_data):
                    steering_angles.append(steering_angle * -1.0)
                    center_images.append(cv2.flip(image, 1))

            except ValueError:
                if (exception_count > 1):
                    raise Exception("The CSV file is likely corrupted")

                exception_count += 1

            if (limit != 0):
                limit_counter += 1

                if (limit_counter > limit):
                    break

        return (np.asarray(center_images), np.asarray(steering_angles))

    @staticmethod
    def dummy():
        raise NotImplementedError
