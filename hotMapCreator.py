
import numpy as np
import tensorflow as tf
import os
import nn_builder
import argparse
import random
import seaborn as sns
import matplotlib.pyplot as plt
from utils import image_name


class HotMapHelper:
    def __init__(self, model, loss_func):
        self.loss_func = loss_func
        self.model = model

    def creates_hotmap_for_classes_directories(self, ds_path, input_size, output_path, kernel_size, stride):
        gen = tf.keras.preprocessing.image.ImageDataGenerator()
        directory_iter = gen.flow_from_directory(ds_path,
                                                class_mode="categorical",
                                                target_size=input_size,
                                                batch_size=1)

        paths = directory_iter.filepaths
        labels = directory_iter.labels
        for i in range(len(directory_iter)):
            image, label = directory_iter[i]
            name = image_name(paths[i])
            output_path= os.path.join(output_path, str(labels[i]))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.test_with_square(name, image, label, kernel_size, stride, output_path)
            print("image {} done".format(paths[i]))



    def test_with_square(self, file_name, image, label, kernel_size, stride, output_path):
        dim_r, dim_h = int((image.shape[1] - kernel_size) / stride), int((image.shape[2] - kernel_size) / stride)
        scores = np.zeros((dim_r, dim_h))
        i, j = 0, 0
        r, c = int(np.floor(kernel_size / 2)), int(np.floor(kernel_size / 2))
        while r < image.shape[1] - int(np.ceil(kernel_size / 2)):
            while c < image.shape[2] - int(np.ceil(kernel_size / 2)):
                image_cp = image.copy()
                k1, k2 = int(np.floor(kernel_size / 2)), int(np.ceil(kernel_size / 2))
                image_cp[0, r - k1: r + k2, c - k1: c + k2, :] = 0

                pred = self.model(image_cp)
                score = -self.loss_func(label, pred)

                scores[i, j] = score
                c += stride
                j += 1
            r += stride
            i += 1
            j = 0
            c = int(np.floor(kernel_size / 2))
        plt.figure()
        ax = sns.heatmap(scores, vmin=np.min(scores), vmax=np.max(scores))
        title = "hot_map_of_{}_with_kernel_{}_and_stride_{}".format(file_name, kernel_size, stride)
        plt.title(title)
        plt.savefig(os.path.join(output_path, title + ".png"))


def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--nntype', default="VGGModel", help='The type of the network')
    parser.add_argument('--ckpt', type=str, default=None)

    parser.add_argument('--cls_num', type=int, required=True)
    parser.add_argument('--kernel_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=16)

    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--ds_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')

    return parser.parse_args()



def main():
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    network = nn_builder.get_network(args.nntype, args.cls_num, args.input_size)
    if args.ckpt is not None:
        network.load_weights(args.ckpt).expect_partial() # expect_partial enables to ignore training information for prediction
    loss = tf.keras.losses.categorical_crossentropy
    hot_map_helper = HotMapHelper(network, loss)
    hot_map_helper.creates_hotmap_for_classes_directories(args.ds_path,
                                                          args.input_size,
                                                          args.output_path,
                                                          args.kernel_size,
                                                          args.stride)




if __name__ == "__main__":




    main()