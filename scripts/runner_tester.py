# ray submit cluster_cfg/local-steropes.yaml evaluate_dynamic_asha.py --args="--num-atoms=8 --trainable-id pytorch --num-jobs=50"
import random
import os
import ray
from ray import tune
import numpy as np
import sys

from hypersched.make_parser import make_parser

if __name__ == "__main__":
    parser = make_parser()
    parser.add_argument(
        "--num-atoms",
        default=1,
        type=int,
        help="Number of atoms to launch Ray with.",
    )
    parser.add_argument(
        "--num-jobs", default=1, type=int, help="Number of jobs to launch."
    )
    parser.add_argument(
        "--target-batch-size", default=None, help="Target batch size"
    )
    parser.add_argument("--sched", default=None, help="Scheduler")
    parser.add_argument(
        "--min-time",
        default=None,
        help="Min time. Needed for DynResourceTimeASHA.",
    )
    parser.add_argument(
        "--global-deadline", default=1200, type=int, help="Target deadline."
    )
    parser.add_argument(
        "--model-string", default="resnet50", type=str, help="The model to use"
    )
    parser.add_argument(
        "--grid",
        default=None,
        type=str,
        help="To use grid if running distributed.",
    )
    parser.add_argument("--gloo", action="store_true", help="Use Tune.")
    parser.add_argument(
        "--seed", default=0, type=int, help="Number of jobs to launch."
    )
    parser.add_argument(
        "--result-path", default="", type=str, help="Result path"
    )
    args = parser.parse_args(sys.argv[1:])

    if args.result_path:
        args.result_path = os.path.expanduser(args.result_path)
        print("ASD")
        print(args.result_path)
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)

    print(vars(args))
    result_log_path = os.path.join(args.result_path, f"result-summary.txt")
    print(result_log_path)
    with open(result_log_path, "a") as f:
        print("Writing it to disk")
        f.write("Test.\n")
