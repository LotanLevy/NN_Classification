

import numpy as np
import tensorflow as tf
import os
import nn_builder
import argparse
import random
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import matplotlib.pyplot as plt
from PIL import Image
import io
import datetime





def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--nntype', default="VGGModel", help='The type of the network')
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))

    parser.add_argument('--ds_path', type=str, required=True)

    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')
    parser.add_argument('--print_freq', '-pf', type=int, default=10)
    parser.add_argument('--learning_rate1', default=1e-3, type=float)
    parser.add_argument('--learning_rate2', default=1e-5, type=float)


    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='number of batches')

    parser.add_argument('--restart_dataloader_config', type=str, default=None)
    parser.add_argument('--restart_model_path', type=str, default=None)

    return parser.parse_args()

def check_corrupted_images(args):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

                                       validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(args.ds_path,
                                                       subset="training",
                                                       seed=123,
                                                       shuffle=True,
                                                       class_mode="categorical",
                                                       target_size=args.input_size,
                                                       batch_size=args.batch_size)

    validation_generator = train_datagen.flow_from_directory(args.ds_path,
                                             subset="validation",
                                             seed=123,
                                             shuffle=True,
                                             class_mode="categorical",
                                             target_size=args.input_size,
                                             batch_size=args.batch_size)

    for path in train_generator.filepaths:
        print(path )
        Image.open(path)

    for path in validation_generator.filepaths:
        print(path )
        Image.open(path)




def main():
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)




    tf.keras.backend.set_floatx('float32')
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # check_corrupted_images(args)








    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                        rotation_range=20,
                                        zoom_range=0.15,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.15,
                                        horizontal_flip=True,
                                        fill_mode="nearest",
                                       validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(args.ds_path,
                                                       subset="training",
                                                       seed=123,
                                                       shuffle=True,
                                                       class_mode="categorical",
                                                       target_size=args.input_size,
                                                       batch_size=args.batch_size)

    validation_generator = train_datagen.flow_from_directory(args.ds_path,
                                             subset="validation",
                                             seed=123,
                                             shuffle=True,
                                             class_mode="categorical",
                                             target_size=args.input_size,
                                             batch_size=args.batch_size)

    cls_num = len(train_generator.class_indices.keys())




    network = nn_builder.get_network(args.nntype, cls_num, args.input_size)
    network.update_output_path(args.output_path)

    network.freeze_status()

    log_dir = os.path.join(
        os.path.join(args.output_path, "amazon_logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    chackpoint_path = os.path.join(os.path.join(args.output_path, "checkpoint"))

    checkpoint = ModelCheckpoint(chackpoint_path, monitor='val_accuracy', save_best_only=True,
                                 save_weights_only=False, mode='max')
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    csv_logger = CSVLogger(os.path.join(args.output_path, 'log.csv'), append=True, separator=';')

    hist = network.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator), validation_data=validation_generator, validation_steps=10,
                               epochs=args.num_epochs, callbacks=[checkpoint, early, csv_logger, tensorboard_callback],
                                 workers=args.workers, use_multiprocessing=True)



if __name__ == "__main__":




    main()
