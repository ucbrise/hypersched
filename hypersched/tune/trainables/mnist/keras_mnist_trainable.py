from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import ray
from ray.tune import tune
from ray.tune.trial import Resources
from .utils import create_parser

from hypersched.tune import ResourceTrainable, ResourceExecutor

mnist.load_data()  # we do this because it's not threadsafe

DEFAULT_CONFIG = {
    "starting_lr": 0.1,
    "momentum": 0.9,
    "batch_size": 128,
    "max_batch_size": 8192,
    "primary_resource": "extra_cpu",
}


class MNISTTrainable(ResourceTrainable):

    metric = "mean_accuracy"

    def _setup(self, config):
        # We set threads here to avoid contention, as Keras
        # is heavily parallelized across multiple cores.
        self.config = config or DEFAULT_CONFIG
        parser = create_parser()
        args = parser.parse_known_args()[0]
        vars(args).update(config)

        # Assign number of threads by looking at resources
        print("Got {} atoms.".format(self.atoms))
        self.num_threads = self.atoms
        if config.get("override_threads"):
            print("WARNING: Overriding threads.")
            self.num_threads = config["override_threads"]
        print("Using {} threads".format(self.num_threads))

        # Set params for Keras
        K.set_session(
            K.tf.Session(
                config=K.tf.ConfigProto(
                    intra_op_parallelism_threads=self.num_threads,
                    inter_op_parallelism_threads=self.num_threads,
                )
            )
        )
        self.batch_size = min(
            config["batch_size"] * self.num_threads, config["max_batch_size"]
        )
        self.num_batches_per_step = config.get("num_batches_per_step", 1)
        self.val_batches_per_step = max(int(self.num_batches_per_step / 4), 1)
        self.lr = (
            config.get("starting_lr") * self.batch_size / config["batch_size"]
        )

        print("BatchSize is: {}".format(self.batch_size))

        # Start model def
        num_classes = 10

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == "channels_first":
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train /= 255
        x_test /= 255
        # print('x_train shape:', x_train.shape)
        # print(x_train.shape[0], 'train samples')
        # print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        self.data = x_train, y_train, x_test, y_test

        model = Sequential()
        model.add(
            Conv2D(
                32,
                kernel_size=(args.kernel1, args.kernel1),
                activation="relu",
                input_shape=input_shape,
            )
        )
        model.add(Conv2D(64, (args.kernel2, args.kernel2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(args.poolsize, args.poolsize)))
        model.add(Dropout(args.dropout1))
        model.add(Flatten())
        model.add(Dense(args.hidden, activation="relu"))
        model.add(Dropout(args.dropout2))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(lr=self.lr, momentum=args.momentum),
            metrics=["accuracy"],
        )
        self.model = model

    def _train(self):
        x_train, y_train, x_test, y_test = self.data
        randoms = np.random.choice(
            len(x_train), self.batch_size * self.num_batches_per_step
        )
        val_randoms = np.random.choice(
            len(x_test), self.batch_size * self.val_batches_per_step
        )
        total_samples = randoms.size + val_randoms.size
        result = self.model.fit(
            x_train[randoms],
            y_train[randoms],
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
            validation_data=(x_test[val_randoms], y_test[val_randoms]),
        )
        res = {k: v[-1] for k, v in result.history.items()}
        res["mean_accuracy"] = res["val_acc"]
        res.update(samples=total_samples)
        return res

    def _save(self, checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, "model")
        self.model.save_weights(file_path)
        return file_path

    def _restore(self, path):
        self.model.load_weights(path)

    @classmethod
    def name_creator(cls, trial):
        return "{}_{:0.4E}".format(
            trial.trainable_name, trial.config.get("starting_lr")
        )

    @classmethod
    def to_atoms(cls, resources):
        return int(resources.cpu)

    @classmethod
    def to_resources(cls, atoms):
        return Resources(cpu=atoms, gpu=0)
