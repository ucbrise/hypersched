# ray submit cluster_cfg/local-steropes.yaml evaluate_dynamic_asha.py --args="--num-atoms=8 --trainable-id pytorch --num-jobs=50"
import random
import os
import ray
from ray import tune
from ray.tune.trial import Resources
from ray.tune.schedulers import AsyncHyperBandScheduler
import numpy as np
import sys
import time
import uuid

from hypersched.tune import ResourceExecutor
from hypersched.utils import timestring

# from hypersched.tune.dynamic_asha import DynamicHyperBandScheduler, DynResourceTimeASHA
from hypersched.tune.ashav2 import ASHAv2
from hypersched.tune.hypersched import HyperSched
from hypersched.tune.summary import Summary
from hypersched.make_parser import make_parser
from hypersched.tune import ray_init
from hypersched.tune.trainables import (
    get_trainable_cls,
    get_multijob_config,
    get_scaling,
)
import logging

logger = logging.getLogger(__name__)

# ray submit cluster_cfg/sgd.yaml --tmux evaluate_dynamic_asha.py --args="--num-atoms=8 --trainable-id pytorch --num-jobs=20 --global-deadline=900 --sched hyper --model-string=resnet18 --result-path='~/socc_playground/resnet18/'"

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
        "--max-t", type=int, default=0, help="Max epochs/iterations."
    )
    parser.add_argument(
        "--global-deadline", default=1200, type=int, help="Target deadline."
    )
    parser.add_argument(
        "--model-string", default="", type=str, help="The model to use"
    )
    parser.add_argument(
        "--data", choices=["cifar", "imagenet", None], help="The data to use"
    )
    parser.add_argument(
        "--grid",
        default=None,
        type=str,
        help="To use grid if running distributed.",
    )
    parser.add_argument("--gloo", action="store_true", help="Use Gloo.")
    parser.add_argument(
        "--no-job-limit", action="store_true", help="No Job Limit."
    )
    parser.add_argument(
        "--no-speculation", action="store_true", help="No speculation."
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="Number of jobs to launch."
    )
    parser.add_argument(
        "--result-path", default="", type=str, help="Result path"
    )
    parser.add_argument(
        "--result-file", type=str, help="manual override summary file."
    )

    args = parser.parse_args(sys.argv[1:])
    if args.result_path:
        args.result_path = os.path.expanduser(args.result_path)
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path, exist_ok=True)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"SEEDED TO {args.seed}")
    trainable, config = get_trainable_cls(args.trainable_id, search=True)
    if args.target_batch_size:
        config.update(target_batch_size=int(args.target_batch_size))

    if args.model_string:
        from hypersched.tune.trainables.pytorch import (
            get_model_creator,
            get_data_creator,
        )

        assert args.trainable_id == "pytorch"
        print("Updating model creator)")
        config.update(
            model_creator=get_model_creator(
                args.model_string, args.data, wrap=True
            )
        )
        config.update(data_creator=get_data_creator(args.data, wrap=True))
    else:
        assert not args.trainable_id == "pytorch"

    # if args.redis_address:
    #     assert args.gloo, "Need gloo if distributed."
    #     config.update(use_nccl=False)

    multijob_config = get_multijob_config(args.trainable_id)

    if args.data == "imagenet":
        worker_config = {}
        worker_config.update(
            data_loader_pin=True,
            data_loader_workers=4,
            max_train_steps=100,
            max_val_steps=20,
            decay=True,
        )
        config.update(worker_config=worker_config)
        multijob_config.update(min_allocation=10)

        class PyTorchImageNet(trainable):
            @classmethod
            def to_atoms(cls, resources):
                return int(resources.extra_gpu / 2)

            @classmethod
            def to_resources(cls, atoms):
                return Resources(0, 0, extra_gpu=int(2 * atoms))

        trainable = PyTorchImageNet

    num_atoms = args.num_atoms

    num_cpus = trainable.to_resources(num_atoms).cpu_total() or None
    num_gpus = trainable.to_resources(num_atoms).gpu_total()
    ray_init(
        redis_address=args.redis_address,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        log_to_driver=False,
        local_mode=False,
        temp_dir=args.ray_logs,
    )

    scheduler_name = args.sched
    if scheduler_name == "asha":
        raise ValueError("use asha2")
        sched = AsyncHyperBandScheduler(
            time_attr=multijob_config["time_attr"],
            mode="max",
            metric=trainable.metric,
            grace_period=multijob_config["min_allocation"],
            max_t=args.max_t or multijob_config["max_allocation"],
        )
    elif scheduler_name == "asha2":
        sched = ASHAv2(
            time_attr=multijob_config["time_attr"],
            mode="max",
            metric=trainable.metric,
            grace_period=multijob_config["min_allocation"],
            max_t=args.max_t or multijob_config["max_allocation"],
        )
    elif scheduler_name == "hyper":
        grid = None
        if args.grid:
            assert args.redis_address, "Need redis address if setting grid."
            grid = int(args.grid)
        print(multijob_config)
        sched = HyperSched(
            num_atoms,
            scaling_dict=get_scaling(
                args.trainable_id, args.model_string, args.data
            ),
            deadline=args.global_deadline,
            resource_policy="UNIFORM",
            allocation_grid=grid,
            time_attr=multijob_config["time_attr"],
            mode="max",
            metric=trainable.metric,
            grace_period=multijob_config["min_allocation"],
            max_t=args.max_t or multijob_config["max_allocation"],
            _no_speculation=args.no_speculation,
            _no_job_limit=args.no_job_limit,
        )
        if args.no_speculation:
            scheduler_name += "-no-spec"
        if args.no_job_limit:
            scheduler_name += "-no-joblim"
    else:
        raise NotImplementedError

    summary = Summary(trainable.metric)

    trials = tune.run(
        trainable,
        name=f"{scheduler_name}-"
        f"{args.model_string}-"
        f"{args.seed}-"
        f"maxt={args.max_t}-"
        f"deadline={args.global_deadline}-"
        f"jobs={args.num_jobs}-"
        f"atoms={args.num_atoms}-"
        f"{uuid.uuid4().hex[:8]}",
        **{"num_samples": args.num_jobs, "config": config, "verbose": 1},
        local_dir=args.result_path
        if args.result_path and os.path.exists(args.result_path)
        else None,
        global_checkpoint_period=600,  # avoid checkpointing completely.
        scheduler=sched,
        resources_per_trial=trainable.to_resources(1)._asdict(),
        trial_executor=ResourceExecutor(
            deadline_s=args.global_deadline, hooks=[summary]
        ),
    ).trials

    # if not args.log:
    #     sys.exit(0)

    params = dict(
        deadline=args.global_deadline,
        metric=trainable.metric,
        model_string=args.model_string,
        sched=args.sched,
        maxt=args.max_t,
        seed=args.seed,
        atoms=args.num_atoms,
    )

    if args.result_file:
        result_summary_path = args.result_file
    else:
        result_summary_path = os.path.join(args.result_path, "summary.log")
    summary.summarize(params, os.path.expanduser(result_summary_path))

    result_log_path = os.path.join(
        args.result_path, f"result-{args.model_string}-test.txt"
    )
    with open(result_log_path, "a") as f:
        print("Writing it to disk")
        f.write("Just finished " + " ".join(sys.argv) + "\n")
        f.write("Args %s" % (str(vars(args))) + "\n")
        f.write("Results written at " + trials[0].local_dir + "\n")
        top_score = str(
            max([t.last_result.get(trainable.metric, 0) for t in trials])
        )
        top_titer = str(
            max([t.last_result.get("training_iteration", 0) for t in trials])
        )
        f.write(f"Top score: {top_score} @ iter {top_titer}\n")
        if not ray.available_resources() == ray.cluster_resources():
            f.write(str(ray.available_resources()))
            f.write(" RESOURCes NOT RELEASED.\n")
        f.write("\n")
