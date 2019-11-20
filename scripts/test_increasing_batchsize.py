#!/usr/bin/env python
# This file is to validate and experiment with increasming batch sizes.
import sys
import argparse
import numpy as np
import pandas as pd

import ray
import time
from ray import tune
from ray.tune.trial import Resources
from ray.tune.logger import pretty_print

from hypersched.tune import ResourceExecutor
from hypersched.utils import timestring
from hypersched.make_parser import make_parser
from hypersched.tune.trainables import get_trainable_cls

parser = make_parser()
parser.add_argument("--write", action="store_true", help="WRite or not.")
parser.add_argument("--reverse", action="store_true", help="Reverse atoms.")
args = parser.parse_args()


def name_creator(trial):
    return "{}_{:0.4E}".format(
        trial.trainable_name, trial.config.get("starting_lr")
    )


if __name__ == "__main__":

    ray.init(redis_address=args.redis_address)
    # get trainable
    trainablecls, cfg = get_trainable_cls(args.trainable_id)
    # warmup trainable (if needed)

    # measure throughput
    results = []
    print("Using {} atoms!".format(atoms))
    trainable = trainablecls(cfg, None, trainablecls.to_resources(atoms))
    print("Class instantiated.")
    atom_results = []
    for i in range(5):
        print("Training Iteration: {}".format(i))
        result = trainable.train()
        print(pretty_print())
        atom_results += [result]

    df = pd.DataFrame(atom_results)
    print("Average throughput: ", df["mean_throughput"].mean())
    print("Median throughput: ", np.median(df["mean_throughput"]))
    print("Max throughput: ", np.max(df["mean_throughput"]))
    print("Min throughput: ", np.min(df["mean_throughput"]))

    results += [df]
    import ipdb

    ipdb.set_trace()
    trainable.stop()

    final_df = pd.DataFrame(results)
    if args.write:
        fname = f"throughput_{str(type(trainable).__name__)}_{timestring()}.csv"
        final_df.to_csv(fname)
        print("wrote to ", fname)
