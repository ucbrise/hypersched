from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import ray

from ray import tune
from ray.tune.trial import Resources
from ray.tune.examples.utils import set_keras_threads
from .babi_helpers import get_and_parse_babi_data

from hypersched.tune import ResourceTrainable

# this just loads the file
path = get_file(
    "babi-tasks-v1-2.tar.gz",
    origin="https://s3.amazonaws.com/text-datasets/"
    "babi_tasks_1-20_v1-2.tar.gz",
)

DEFAULT_CONFIG = {
    "lstm_size": 128,
    "embedding": 64,
    "dropout": 0.3,
    "opt": "rmsprop",
    "challenge_type": "single_supporting_fact_10k",
    "threads": 4,
    "batch_size": 16,
}

DEFAULT_HSPACE = {
    "lstm_size": tune.uniform(4, 128),
    "embedding": tune.uniform(32, 128),
    "dropout": tune.uniform(0.1, 0.5),
    "opt": tune.choice(["rmsprop", "adam", "sgd"]),
}

DEFAULT_SCALING = {
    1: 1,
    2: 1.62,
    4: 2.027,
    8: 2.75,
    16: 2.48,
    32: 2.48,
}

DEFAULT_MULTIJOB_CONFIG = {
    "min_time_allocation": 20,
    "global_deadline": 180,  # this is in clock time
    "clock_time_attr": "time_total_s",
}


class BABITrainable(ResourceTrainable):

    metric = "mean_accuracy"

    def _setup(self, config):
        # Assign number of threads by looking at resources
        print("Got {} atoms.".format(self.atoms))
        self.num_threads = self.atoms
        set_keras_threads(self.num_threads)
        all_data = get_and_parse_babi_data(config)
        self.batch_size = self.config["batch_size"]
        self.num_batches_per_step = config.get("num_batches_per_step", 1)
        self.val_batches_per_step = max(int(self.num_batches_per_step / 4), 1)
        self.training_dataset = all_data[:3]
        self.testing_dataset = all_data[3:6]
        vocab_size, story_maxlen, query_maxlen = all_data[6:]
        input_sequence = Input((story_maxlen,))
        question = Input((query_maxlen,))

        # encoders
        # embed the input sequence into a sequence of vectors
        input_encoder_m = Sequential()
        input_encoder_m.add(
            Embedding(
                input_dim=vocab_size, output_dim=int(config["embedding"]),
            )
        )
        input_encoder_m.add(Dropout(config["dropout"]))
        # output: (samples, story_maxlen, embedding_dim)

        # embed the input into a sequence of vectors of size query_maxlen
        input_encoder_c = Sequential()
        input_encoder_c.add(
            Embedding(input_dim=vocab_size, output_dim=query_maxlen)
        )
        input_encoder_c.add(Dropout(config["dropout"]))
        # output: (samples, story_maxlen, query_maxlen)

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=int(config["embedding"]),
                input_length=query_maxlen,
            )
        )
        question_encoder.add(Dropout(config["dropout"]))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation("softmax")(match)

        # add the match matrix with the second input vector sequence
        response = add(
            [match, input_encoded_c]
        )  # (samples, story_maxlen, query_maxlen)
        response = Permute((2, 1))(
            response
        )  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = LSTM(int(config["lstm_size"]))(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = Dropout(config["dropout"])(answer)
        answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = Activation("softmax")(answer)

        # build the final model
        model = Model([input_sequence, question], answer)
        model.compile(
            optimizer=config["opt"],
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model

    def _train(self):
        # x_train, y_train, x_test, y_test = self.data
        (inputs_train, queries_train, answers_train,) = self.training_dataset
        inputs_test, queries_test, answers_test = self.testing_dataset
        randoms = np.random.choice(
            len(inputs_train), self.batch_size * self.num_batches_per_step,
        )
        val_randoms = np.random.choice(
            len(inputs_test), self.batch_size * self.val_batches_per_step,
        )
        total_samples = randoms.size + val_randoms.size
        result = self.model.fit(
            [inputs_train[randoms], queries_train[randoms]],
            answers_train[randoms],
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
            validation_data=(
                [inputs_test[val_randoms], queries_test[val_randoms]],
                answers_test[val_randoms],
            ),
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
        name = f"{trial.trainable_name}"
        params = "_".join(
            [
                f"{k}={trial.config[k]:3.3f}"
                for k in ["lstm_size", "embedding", "dropout"]
            ]
        )
        params += f"{trial.config['opt']}"
        return name + params

    @classmethod
    def to_atoms(cls, resources):
        return int(resources.cpu)

    @classmethod
    def to_resources(cls, atoms):
        return Resources(cpu=atoms, gpu=0)
