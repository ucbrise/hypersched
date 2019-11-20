from hypersched.tune import ResourceExecutor
from hypersched.utils import timestring

# from hypersched.tune.dynamic_asha import DynamicHyperBandScheduler, DynResourceTimeASHA
from hypersched.tune.hypersched import HyperSched, SCALING_MAP
from hypersched.tune.summary import Summary
from hypersched.tune.dyn_allocator import DynamicAllocator
from hypersched.make_parser import make_parser
from hypersched.tune import ray_init
from hypersched.tune import ray_init
from hypersched.tune.trainables import (
    get_trainable_cls,
    get_multijob_config,
    get_scaling,
)
import logging
import sys
from ray import tune
import numpy as np
import random
import os
import logging

logger = logging.getLogger(__name__)

# python [thisscript.py] --trainable-id optimus --global-deadline 30  --num-atoms=4 --num-jobs=10

if __name__ == "__main__":

    parser = make_parser()
    parser.add_argument(
        "--num-atoms",
        default=1,
        type=int,
        help="Number of atoms to launch Ray with.",
    )
    parser.add_argument("--config", default="", type=str, help="Config name.")
    parser.add_argument(
        "--num-jobs", default=1, type=int, help="Number of jobs to launch."
    )
    parser.add_argument(
        "--delay",
        default=0.1,
        type=float,
        help="Seconds of delay per step. Divided by scaling",
    )
    parser.add_argument(
        "--startup-delay",
        default=0.001,
        type=float,
        help="Seconds of delay per for startup",
    )
    parser.add_argument(
        "--global-deadline", default=50, type=int, help="Target deadline."
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="Number of jobs to launch."
    )
    parser.add_argument("--max-t", type=int, help="Max epochs/iterations.")
    parser.add_argument(
        "--strategy",
        default="UNIFORM",
        choices=DynamicAllocator.POLICIES,
        help="Dictates how stuff is allocated across trials.",
    )
    # We want to show that ignoring the job overhead can result in
    # significantly worse performance. This is due to
    # degradations of decisions made - namely the mass accumulation
    # of overhead and a decrease of training iterations on the top trial(s).
    parser.add_argument(
        "--ignore-overhead",
        action="store_true",
        help="Does not track startup overhead.",
    )
    # We want to show that limiting different amounts of jobs
    # will result in degraded performance.
    parser.add_argument(
        "--no-job-limit", action="store_true", help="Does not limit jobs run."
    )
    # We want to show that the linear assumption can really hurt
    #
    parser.add_argument(
        "--assume-linear", action="store_true", help="Assumes linear scaling."
    )
    # We want to show that a more exploitative policy
    # is not better
    parser.add_argument(
        "--fixed-exploration",
        action="store_true",
        help="Fix the exploration time.",
    )
    parser.add_argument(
        "--exploration-ratio", type=float, help="Ratio of exploration time."
    )
    parser.add_argument(
        "--scaling",
        default="LINEAR",
        choices=list(SCALING_MAP),
        help="Scaling for trainable. Defaults to linear scaling on Trainable.",
    )
    parser.add_argument(
        "--no-speculation", action="store_true", help="No speculation."
    )

    parser.add_argument("--result-file", default="./ablation.log", type=str)
    parser.add_argument("--dummy", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    assert args.trainable_id == "optimus"
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"SEEDED TO {args.seed}")
    trainable, config = get_trainable_cls(args.trainable_id, search=True)
    if args.delay is not None:
        config.update(delay=args.delay)

    if args.startup_delay is not None:
        config.update(startup_delay=args.startup_delay)

    if args.dummy:
        config.update(dummy=args.dummy)

    # Only used for abblations.
    config.update(scaling=SCALING_MAP[args.scaling])

    multijob_config = get_multijob_config(args.trainable_id)
    num_atoms = args.num_atoms

    num_cpus = trainable.to_resources(num_atoms).cpu_total() or None
    num_gpus = trainable.to_resources(num_atoms).gpu_total()
    ray_init(
        redis_address=args.redis_address,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        local_mode=False,
        temp_dir=args.ray_logs,
    )
    summary = Summary(trainable.metric)
    hypersched_params = dict(
        deadline=args.global_deadline,
        resource_policy=args.strategy,
        time_attr=multijob_config["time_attr"],
        metric=trainable.metric,
        grace_period=multijob_config["min_allocation"],
        max_t=args.max_t or multijob_config["max_allocation"],
        _no_speculation=args.no_speculation,
        _ignore_overhead=args.ignore_overhead,
        _no_job_limit=args.no_job_limit,
        _assume_linear=args.assume_linear,
        _fixed_exploration=args.fixed_exploration,
        _exploration_ratio=args.exploration_ratio,
    )
    sched = HyperSched(
        num_atoms,
        scaling_dict=get_scaling(
            args.trainable_id,
            scaling="LINEAR" if args.assume_linear else args.scaling,
        ),
        **hypersched_params,
    )

    trials = tune.run(
        trainable,
        name=f"{type(sched).__name__}-{timestring()}-jobs={args.num_jobs}-atoms={args.num_atoms}",
        num_samples=args.num_jobs,
        config=config,
        return_trials=True,
        verbose=1,
        local_dir=f"~/socc_playground/",
        scheduler=sched,
        resources_per_trial=trainable.to_resources(1)._asdict(),
        trial_executor=ResourceExecutor(
            deadline_s=args.global_deadline, hooks=[summary]
        ),
    )

    params = hypersched_params.copy()
    params.update(
        seed=args.seed,
        num_atoms=num_atoms,
        num_jobs=args.num_jobs,
        trainable_id=args.trainable_id,
        delay=args.delay,
        startup_delay=args.startup_delay,
        scaling=args.scaling,
    )
    summary.summarize(params, os.path.expanduser(args.result_file))
