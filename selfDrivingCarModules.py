import csv
import cv2
import numpy as np
import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import re
import os
from datetime import datetime
import sys
import shutil
import matplotlib.pyplot as plt
import math
from sklearn import metrics
# from keras.applications.resnet50 import preprocess_input, decode_predictions


class Sdc:

    __CENTER_IMAGES__ = 1
    __LEFT_IMAGES__ = 2
    __RIGHT_IMAGES__ = 3
    __ALL_IMAGES__ = 4

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
    def generate_partition_ids(data_path, csv_file, validation_split=0.2, limit=0, **kwargs):
        """

        :param data_path:
        :param csv_file:
        :param validation_split:
        :param limit:
        :param : (optional) random_state, defaul: None
        :param : (optional) image_series_type, default: all images
        :return: partitions["train"], partitions["validation"]
        """

        image_series_type = kwargs["image_series_type"] if ("image_series_type" in kwargs) else Sdc.__ALL_IMAGES__
        random_state = kwargs["random_state"] if ("random_state" in kwargs) else None
        correction_factor = kwargs["correction_factor"] if ("correction_factor" in kwargs) else 0.1

        csv_lines = []

        with open(csv_file) as csv_file:
            reader = csv.reader(csv_file)

            for csv_line in reader:
                csv_lines.append(csv_line)

        x_partitions = {"train": [], "validation": []}
        # y_partitions = {"train": [], "validation": []}

        image_files = []
        steering_angles = []

        limit_counter = 0

        for line_segments in csv_lines[1:]:
            if (limit != 0):
                limit_counter += 1

                if (limit_counter > limit):
                    break

            # steering_angle = float(line_segments[3])
            steering_angle = float(line_segments[3])

            if (image_series_type == Sdc.__ALL_IMAGES__):
                center_image_file = data_path + line_segments[0]
                left_image_file = data_path + line_segments[1][1:]
                right_image_file = data_path + line_segments[2][1:]

                image_files.extend([center_image_file, left_image_file, right_image_file])
                steering_angles.extend([steering_angle, steering_angle + correction_factor, steering_angle - correction_factor])

            elif (image_series_type == Sdc.__CENTER_IMAGES__):
                # image_file = data_path + line_segments[0]
                image_files.append(data_path + line_segments[0])
                steering_angles.append(steering_angle)
            elif (image_series_type == Sdc.__LEFT_IMAGES__):
                image_files.append(data_path + line_segments[1][1:])
                steering_angles.append(steering_angle + correction_factor)
            elif (image_series_type == Sdc.__RIGHT_IMAGES__):
                image_files.append(data_path + line_segments[2][1:])
                steering_angles.append(steering_angle - correction_factor)

        # print(all_image_files)

        # all_image_files = np.concatenate([center_image_files, left_image_files, right_image_files])
        all_steering_angles = dict(zip(image_files, steering_angles))

        # x_partitions["train"], x_partitions["validation"], y_partitions["train"], y_partitions["validation"] = \
        #     train_test_split(all_image_files, all_steering_angles, test_size=validation_split, random_state=random_state)

        x_partitions["train"], x_partitions["validation"] = \
            train_test_split(image_files, test_size=validation_split, random_state=random_state)

        return (x_partitions["train"], x_partitions["validation"], all_steering_angles)

    @staticmethod
    def dummy():
        raise NotImplementedError
