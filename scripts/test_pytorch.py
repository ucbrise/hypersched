#!/usr/bin/env python

import sys
import argparse
import numpy as np
import json
import ray
import time
from ray import tune
from ray.tune.trial import Resources
from ray.tune.logger import pretty_print

from hypersched.tune import ResourceExecutor
from hypersched.utils import timestring
from hypersched.tune.trainables.pytorch.pytorch_trainable import (
    PytorchSGD,
    DEFAULT_CONFIG,
)
from hypersched.tune.trainables.pytorch.pytorch_helpers import (
    prefetch,
    get_cifar_dataset,
)
from hypersched.tune.trainables.pytorch.cifar_models import MODEL_DICT
from hypersched.make_parser import make_parser

import logging

logging.basicConfig(level="INFO")

parser = make_parser()
parser.add_argument(
    "--min-batch-size", default=128, type=int, help="Batch per device."
)
parser.add_argument(
    "--num-workers", default=1, type=int, help="Number of workers."
)
# There is a difference between this and refresh_freq.
parser.add_argument(
    "--num-jobs", default=32, type=int, help="Seconds before termination."
)
parser.add_argument(
    "--placement", default="[]", type=json.loads, help="How to split clusters"
)
parser.add_argument(
    "--model-string", default="resnet50", type=str, help="The model to use"
)
parser.add_argument(
    "--dataset", default="cifar", type=str, help="dataset to use"
)
parser.add_argument("--profile", action="store_true", help="Use profiler")

args = parser.parse_args()
ray.init(redis_address=args.redis_address)

config = DEFAULT_CONFIG.copy()


def name_creator(trial):
    return "{}_{:0.4E}".format(
        trial.trainable_name, trial.config.get("starting_lr")
    )


if __name__ == "__main__":
    config["num_workers"] = args.num_workers
    config["target_batch_size"] = args.batch_size
    config["min_batch_size"] = args.min_batch_size
    config["gpu"] = args.gpu
    config["placement"] = args.placement

    res = "extra_gpu" if args.gpu else "extra_cpu"
    total_resources = ray.cluster_resources()["GPU"]
    config["model_creator"] = MODEL_DICT[args.model_string]
    config["data_creator"] = get_cifar_dataset

    if args.prefetch:
        # This launches data downloading for all the experiments.
        prefetch(args.dataset)

    if not args.tune:
        config["verbose"] = True
        sgd = PytorchSGD(
            config=config,
            resources=Resources(**{"cpu": 0, "gpu": 0, res: args.num_workers}),
        )
        from ray.tune.logger import pretty_print

        for i in range(args.num_iters):
            print(pretty_print(sgd.train()))
        # checkpoint = sgd.save()
        # sgd.stop()
        # config["placement"] = None
        # sgd2 = PytorchSGD(
        #     config=config,
        #     resources=Resources(
        #     **{"cpu": 0, "gpu": 0, res: 2}))
        # sgd2.restore(checkpoint)
        # for i in range(args.num_iters):
        #     print(pretty_print(sgd2.train()))
    else:
        # This is the deadline (s). The resource executor will kill all jobs
        # after deadline.

        config["starting_lr"] = tune.grid_search(
            list(np.logspace(-6, -1, args.num_jobs))
        )
        # config["weight_decay"] = tune.grid_search([1e-2, 1e-3, 1e-4])
        config["steps_per_iteration"] = args.steps_per_iter
        deadline = args.deadline
        schedulers = [None]

        for sched in schedulers:
            tune.run(
                PytorchSGD,
                name="pytorch-sgd-schedule-{}".format(timestring()),
                max_failures=3,
                config=config,
                trial_name_creator=tune.function(name_creator),
                stop={
                    "mean_accuracy": 90,
                    "training_iteration": args.num_iters,
                },
                resources_per_trial={"cpu": 0, "gpu": 0, "extra_gpu": 1},
                local_dir="~/ray_results/{}/".format(
                    str(type(sched).__name__)
                ),
                verbose=1,
                scheduler=sched,
                queue_trials=True,
                trial_executor=ResourceExecutor(deadline_s=deadline,),
            )
