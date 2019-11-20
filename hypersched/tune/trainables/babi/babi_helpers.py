"""
Originally from: https://github.com/keras-team/keras/blob/master/examples/babi_memnn.py
#Trains a memory network on the bAbI dataset.

References:

- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  ["Towards AI-Complete Question Answering:
  A Set of Prerequisite Toy Tasks"](http://arxiv.org/abs/1502.05698)

- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  ["End-To-End Memory Networks"](http://arxiv.org/abs/1503.08895)

Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
"""
from __future__ import print_function

from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences

import tarfile
import numpy as np
import re


def tokenize(sent):
    """Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    return [x.strip() for x in re.split(r"(\W+)?", sent) if x and x.strip()]


def parse_stories(lines, only_supporting=False):
    """Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences
    that support the answer are kept.
    """
    data = []
    story = []
    for line in lines:
        line = line.decode("utf-8").strip()
        nid, line = line.split(" ", 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if "\t" in line:
            q, a, supporting = line.split("\t")
            q = tokenize(q)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append("")
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    """Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    data = [
        (sum(story, []), q, answer)
        for story, q, answer in data
        if not max_length or len(sum(story, [])) < max_length
    ]
    return data


challenges = {
    # QA1 with 10,000 samples
    "single_supporting_fact_10k": "tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt",
    # QA2 with 10,000 samples
    "two_supporting_facts_10k": "tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt",
}

NUM_THREADS = 4


def get_and_parse_babi_data(config):
    challenge_type = config["challenge_type"]
    challenge = challenges[challenge_type]
    path = get_file(
        "babi-tasks-v1-2.tar.gz",
        origin="https://s3.amazonaws.com/text-datasets/"
        "babi_tasks_1-20_v1-2.tar.gz",
    )

    with tarfile.open(path) as tar:
        extracted_train = tar.extractfile(challenge.format("train"))
        extracted_test = tar.extractfile(challenge.format("test"))
        train_stories = get_stories(extracted_train)
        test_stories = get_stories(extracted_test)

    vocab = set()
    for story, q, answer in train_stories + test_stories:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    story_maxlen = max(
        map(len, (x for x, _, _ in train_stories + test_stories))
    )
    query_maxlen = max(
        map(len, (x for _, x, _ in train_stories + test_stories))
    )

    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    def vectorize_stories(data):
        inputs, queries, answers = [], [], []
        for story, query, answer in data:
            inputs.append([word_idx[w] for w in story])
            queries.append([word_idx[w] for w in query])
            answers.append(word_idx[answer])
        return (
            pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers),
        )

    inputs_train, queries_train, answers_train = vectorize_stories(
        train_stories
    )
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories)
    return (
        inputs_train,
        queries_train,
        answers_train,
        inputs_test,
        queries_test,
        answers_test,
        vocab_size,
        story_maxlen,
        query_maxlen,
    )

    # train
    # model.fit(
    #     [inputs_train, queries_train],
    #     answers_train,
    #     batch_size=32,
    #     epochs=120,
    #     validation_data=([inputs_test, queries_test], answers_test),
    #     verbose=0,
    #     callbacks=[TuneKerasCallback(reporter)])


# trainable = train_babi
# experiment_name = f"nlp-benchmark"
# reward_attr = "episode_reward_mean"
# time = "training_iteration"
# stop = {time: 120, "time_total_s": 800}
# reuse_actors = False

# trial_name = tune.function(_trial_name)
