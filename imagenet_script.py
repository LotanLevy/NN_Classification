
import os
import numpy as np
from utils import read_dataset_map
from PIL import Image
import argparse

# LABELS_MAP = "C:/Users/lotan/Documents/datasets/imagenet_map.txt"
#
# NEW_PATH = "C:/Users/lotan/Documents/datasets/imagenet_val_sorted"


def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--labels_map', type=str)

    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')


    return parser.parse_args()

args = get_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)



paths, labels = read_dataset_map(args.labels_map)


# Creates labels directories
dirs = np.unique(labels)
for dir in dirs:
    dir_path = os.path.join(args.output_path, str(dir))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

for i in range(len(paths)):
    p = paths[i]
    if not os.path.exists(p):
        continue
    dir = os.path.join(args.output_path, str(labels[i]))
    image_name = os.path.basename(p)

    output_path = os.path.join(dir, image_name)
    im = Image.open(p, 'r')
    im.save(output_path)



