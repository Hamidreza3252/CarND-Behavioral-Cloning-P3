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
    def __init__(self):
        self.text = ""
