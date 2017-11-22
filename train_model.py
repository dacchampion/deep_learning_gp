import numpy as np
import argparse
import os
from unet_model import create_model
from keras.callbacks import ModelCheckpoint

np.random.seed(103)

HEIGHT, WIDTH = 256, 256
CHANNELS = 4
CLASSES = 2
TRANSFORMATION = 3
BATCH_SIZE = 16
NUM_EPOCHS = 200


def args_parser():
    parser = argparse.ArgumentParser(description="Train the chosen model with splitted tiles then takes the best")
    parser.add_argument('--train-dir', required=True, help="Directory with training data set elements")
    parser.add_argument('--target-dir', required=True, help="Directory with target data set elements")
    parser.add_argument("--xv-dir", type=str, required=False, help="Directory with cross-validation data set elements")
    return parser.parse_args()


def get_training_data(train_dir, target_dir):
    input_file_names = os.listdir(train_dir)
    num_samples = len(input_file_names) * TRANSFORMATION
    x_train = np.zeros((num_samples, HEIGHT, WIDTH, CHANNELS))
    y_train = np.zeros((num_samples, HEIGHT, WIDTH))
    sample_idx = 0
    for file_name in input_file_names:
        if file_name == ".DS_Store":
            continue
        x = np.load(os.path.join(train_dir, file_name))
        y = np.load(os.path.join(target_dir, file_name))
        x_train[sample_idx, :, :, :] = x
        y_train[sample_idx, :, :] = y
        sample_idx += 1
        # transformations
        yy = np.reshape(y, (HEIGHT, WIDTH))
        # left-to-right mirror
        x_aug = np.fliplr(x)
        y_aug = np.fliplr(yy)
        y_aug = np.reshape(y_aug, (HEIGHT, WIDTH))
        x_train[sample_idx, :, :, :] = x_aug
        y_train[sample_idx, :, :] = y_aug
        sample_idx += 1
        # up to down mirror
        x_aug = np.flipud(x)
        y_aug = np.flipud(yy)
        y_aug = np.reshape(y_aug, (HEIGHT, WIDTH))
        x_train[sample_idx, :, :, :] = x_aug
        y_train[sample_idx, :, :] = y_aug
        sample_idx += 1
    return x_train, y_train

xv_set = False
arguments = args_parser()
if not os.path.isdir(arguments.train_dir):
    raise RuntimeError("Train directory does not exist '%s'" % arguments.train_dir)
if not os.path.isdir(arguments.target_dir):
    raise RuntimeError("Target directory does not exist '%s'" % arguments.target_dir)
if arguments.xv_dir:
    xv_set = True
    if not os.path.isdir(arguments.xv_dir):
        raise RuntimeError("The specified cross-validation directory does not exist '%s'" % arguments.xv_dir)

if not xv_set:
    train_set, target_set = get_training_data(arguments.train_dir, arguments.target_dir)
    model = create_model()
    model.fit(train_set, target_set, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2,
              callbacks=[ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)])