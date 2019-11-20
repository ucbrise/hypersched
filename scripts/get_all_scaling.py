#!/usr/bin/env python
# This file is to measure throughput on a given trainable.

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
from hypersched.tune.trainables.pytorch import (
    get_data_creator,
    get_model_creator,
)

# from hypersched.tune.pytorch.trainables.pytorch_helpers import prefetch

# ray submit cluster_cfg/sgd.yaml get_all_scaling.py --args="--trainable-id pytorch --num-iters 5 --write"
parser = make_parser()
parser.add_argument("--write", action="store_true", help="WRite or not.")
parser.add_argument("--reverse", action="store_true", help="Reverse atoms.")
parser.add_argument(
    "--model-string", default="resnet50", type=str, help="The model to use"
)
parser.add_argument(
    "--data", choices=["cifar", "imagenet", None], help="The data to use"
)
args = parser.parse_args()


def name_creator(trial):
    return "{}_{:0.4E}".format(
        trial.trainable_name, trial.config.get("starting_lr")
    )


PROFILED_MODELS = [
    "shufflenet",
    "resnet152",
    "resnet101",
    "resnet50",
    "resnet18",
]

if __name__ == "__main__":
    ray.init(redis_address=args.redis_address, temp_dir=args.ray_logs)
    # get trainable
    trainablecls, cfg = get_trainable_cls(args.trainable_id)

    # if args.model_string:
    #     cfg.update(model_string=args.model_string)

    for modelstr in PROFILED_MODELS:
        cfg.update(
            model_creator=get_model_creator(args.model_string, args.data)
        )
        cfg.update(data_creator=get_data_creator(args.data))
        # measure throughput
        results = []
        atom_list = [1, 2, 4, 8]
        if args.reverse:
            atom_list = reversed(atom_list)
        for atoms in atom_list:
            print("Using {} atoms!".format(atoms))
            trainable = trainablecls(
                cfg, None, trainablecls.to_resources(atoms)
            )
            print("Class instantiated.")
            atom_results = []
            for i in range(args.num_iters + 1):
                print("Training Iteration: {}".format(i))
                result = trainable.train()
                if i != 0:
                    atom_results += [result]  # Warmup iter

            df = pd.DataFrame(atom_results)
            print("Average throughput: ", df["mean_throughput"].mean())
            print("Median throughput: ", np.median(df["mean_throughput"]))
            print("Max throughput: ", np.max(df["mean_throughput"]))
            print("Min throughput: ", np.min(df["mean_throughput"]))

            results += [df]
            trainable.stop()

        final_df = pd.concat(results)
        if args.write:
            fname = f"throughput_{str(type(trainable).__name__)}_{str(modelstr)}_{timestring()}.csv"
            final_df.to_csv(fname)
            print("wrote to ", fname)
