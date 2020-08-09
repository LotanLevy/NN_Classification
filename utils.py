

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import random
import os
import re

SPLIT_FACTOR = "$"

def image_name(image_path):
    regex = ".*[\\/|\\\](.*)[\\/|\\\](.*).jpg"
    m = re.match(regex, image_path)
    return m.group(1) + "_" + m.group(2)


def read_image(path, resize_image=(), augment=False):
    image = Image.open(path, 'r')
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if augment:
        image = get_random_augment(image, resize_image)
    if len(resize_image) > 0:
        image = image.resize(resize_image, Image.NEAREST)
    image = np.array(image).astype(np.float32)
    return image

def read_dataset_map(data_map_path, shuffle=False):
    with open(data_map_path, "r") as lf:
        lines_list = lf.read().splitlines()
        if shuffle:
            random.shuffle(lines_list)
        lines = [line.split(SPLIT_FACTOR) for line in lines_list]
        images, labels = [], []
        if len(lines) > 0:
            images, labels = zip(*lines)
        labels = [int(label) for label in labels]
    return images, np.array(labels).astype(np.int)


def write_dataset_map(output_path, dataset_name,  paths, labels):
    assert len(paths) == len(labels)
    with open(os.path.join(output_path, '{}.txt'.format(dataset_name)), 'w') as df:
        lines = ["{}{}{}\n".format(paths[i], SPLIT_FACTOR, labels[i]) for i in range(len(paths))]
        df.writelines(lines)



