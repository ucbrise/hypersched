import random
import logging

logger = logging.getLogger(__name__)
import numpy as np

from hypersched.tune import ResourceExecutor
from hypersched.utils import timestring
from hypersched.tune.dynamic_asha import (
    DynamicHyperBandScheduler,
    DynResourceTimeASHA,
)
from hypersched.tune.hypersched import HyperSched
from hypersched.make_parser import make_parser
from hypersched.tune import ray_init
from hypersched.tune import ray_init
from hypersched.tune.trainables import (
    get_trainable_cls,
    get_multijob_config,
    get_scaling,
)
import sys
from ray import tune

# python [thisscript.py] --trainable-id optimus --global-deadline 30  --num-atoms=4 --num-jobs=10

if __name__ == "__main__":

    parser = make_parser()
    parser.add_argument(
        "--num-atoms",
        default=1,
        type=int,
        help="Number of atoms to launch Ray with.",
    )
    parser.add_argument(
        "--num-jobs", default=1, type=int, help="Number of jobs to launch.",
    )
    parser.add_argument(
        "--delay",
        default=1,
        type=float,
        help="Seconds of delay per step. Divided by scaling",
    )
    parser.add_argument(
        "--startup-delay",
        default=2,
        type=float,
        help="Seconds of delay per for startup",
    )
    parser.add_argument(
        "--global-deadline", default=1200, type=int, help="Target deadline.",
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="Number of jobs to launch.",
    )

    args = parser.parse_args(sys.argv[1:])
    assert args.trainable_id == "optimus"
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"SEEDED TO {args.seed}")
    trainable, config = get_trainable_cls(args.trainable_id, search=True)
    if args.delay is not None:
        config.update(delay=args.delay)

    if args.startup_delay is not None:
        config.update(startup_delay=args.startup_delay)

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
    sched = HyperSched(
        num_atoms,
        scaling_dict=get_scaling(args.trainable_id),
        deadline=args.global_deadline,
        resource_policy="UNIFORM",
        time_attr=multijob_config["time_attr"],
        metric=trainable.metric,
        grace_period=multijob_config["min_allocation"],
        max_t=multijob_config["max_allocation"],
    )

    trials = tune.run(
        trainable,
        name=f"{type(sched).__name__}-{timestring()}-jobs={args.num_jobs}-atoms={args.num_atoms}",
        **{"num_samples": args.num_jobs, "config": config, "verbose": 1,},
        local_dir=f"~/socc_playground/",
        scheduler=sched,
        resources_per_trial=trainable.to_resources(1)._asdict(),
        trial_executor=ResourceExecutor(deadline_s=args.global_deadline),
    )
