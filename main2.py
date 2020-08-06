
import argparse
import os
import tensorflow as tf
import numpy as np
import random
import nn_builder
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from DataGenerator import DataGenerator
import datetime






def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--nntype', default="VGGModel", help='The type of the network')
    parser.add_argument('--input_size', type=int, nargs=2, default=(32, 32))


    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')
    parser.add_argument('--print_freq', '-pf', type=int, default=10)
    parser.add_argument('--learning_rate1', default=0.1, type=float)
    parser.add_argument('--learning_rate2', default=1e-5, type=float)


    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='number of batches')

    parser.add_argument('--restart_dataloader_config', type=str, default=None)
    parser.add_argument('--restart_model_path', type=str, default=None)

    return parser.parse_args()



def tfdata_generator(images, labels, cls_num, shuffle, batch_size, seed, image_size):
  '''Construct a data generator using `tf.Dataset`. '''
  def map_fn(image, label):
      '''Preprocess raw data to trainable input. '''
      x = tf.reshape(tf.cast(image, tf.float32), (image_size[0], image_size[1], 3))
      y = tf.one_hot(tf.cast(label, tf.uint8), cls_num)
      return x, y

  dataset = tf.data.Dataset.from_tensor_slices((images, labels))

  if shuffle:
    dataset = dataset.shuffle(len(images), seed=seed)  # depends on sample size
  dataset = dataset.map(map_fn)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

def main():
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)


    tf.keras.backend.set_floatx('float32')
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    log_dir = os.path.join(os.path.join(args.output_path, "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


    cls_num = len(np.unique(y_train))






    train_generator = DataGenerator(images=x_train, labels=y_train, batch_size=args.batch_size,
                                    dim=args.input_size, n_channels = 3, n_classes =cls_num, shuffle = True)

    validation_generator = DataGenerator(images=x_test, labels=y_test, batch_size=args.batch_size,
                                    dim=args.input_size, n_channels=3, n_classes=cls_num, shuffle=True)


    network = nn_builder.get_network(args.nntype, cls_num, args.input_size)
    network.update_output_path(args.output_path)

    network.freeze_status()

    optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)

    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    chackpoint_path = os.path.join(os.path.join(args.output_path, "vgg16_1.h5"))

    checkpoint = ModelCheckpoint(chackpoint_path, monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', save_freq=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

    csv_logger = CSVLogger(os.path.join(args.output_path, 'log.csv'), append=True, separator=';')

    hist = network.fit_generator(generator=train_generator, steps_per_epoch=len(train_generator), validation_data=validation_generator, validation_steps=10,
                               epochs=args.num_epochs, callbacks=[checkpoint, tensorboard_callback])




if __name__ == "__main__":




    main()
