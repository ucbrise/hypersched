from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import math
import time
import sys
import logging
import tempfile
import torch
import torchvision
import os
import random
from contextlib import closing
import pandas as pd
import numpy as np
import socket

import ray
from ray.tune.trial import Resources
from ..utils import TimerStat, create_colocated

import ray
from ray import tune
from hypersched.tune import ResourceTrainable

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .cifar_models import ResNet18
from .pytorch_helpers import (
    train,
    validate,
    state_from_cuda,
    state_to_cuda,
    Adjuster,
    adjust_learning_rate,
    mass_download,
    get_cifar_dataset,
)


DEFAULT_CONFIG = {
    # Arguments to pass to the optimizer
    "starting_lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    # Pins actors to cores
    "pin": False,
    "target_batch_size": 128,
    "model_creator": None,
    "min_batch_size": 128,
    "data_creator": None,
    "loss_creator": nn.CrossEntropyLoss,
    "placement": None,
    "use_nccl": True,
    "devices_per_worker": 1,
    "primary_resource": "extra_gpu",
    "gpu": False,
    "verbose": False,
    "worker_config": {
        "data_loader_workers": 2,
        "data_loader_pin": False,
        "max_train_steps": None,
        "fp16": False,
        "max_val_steps": None,
        "decay": True,
    },
    "lr_config": {"mode": "log", "steps": 5},
}

DEFAULT_HSPACE = {
    "starting_lr": tune.choice(
        [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    ),
    "weight_decay": tune.choice([0.0001, 0.0005, 0.001, 0.005]),
    "momentum": tune.choice([0.9, 0.95, 0.99, 0.997]),
}

DEFAULT_MULTIJOB_CONFIG = {
    # Model setup time can be 20, overall first epoch setup can take up to 100
    "min_allocation": 2,
    "max_allocation": 300,  # this doesn't really matter
    "time_attr": "training_iteration",
}

# DEFAULT_SCALING = {
#     # 1024, 512 * 2, 512 * 4, 512 * 8
#     # V100, 1 machine
#     1: 1,
#     2: 1.6,
#     4: 2.37,
#     8: 2.72
# }


logger = logging.getLogger(__name__)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def initialization_hook(runner):
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    print("NCCL DEBUG SET")
    # Need this for avoiding a connection restart issue
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["NCCL_DEBUG"] = "INFO"
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)


class PyTorchRunner(object):
    def __init__(
        self,
        batch_size,
        momentum=0.9,
        weight_decay=5e-4,
        model_creator=None,
        data_creator=None,
        loss_creator=None,
        verbose=False,
        use_nccl=False,
        worker_config=None,
        lr_config=None,
    ):
        initialization_hook(self)
        self.batch_size = batch_size
        self.epoch = 0
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.use_nccl = use_nccl
        self.model_creator = model_creator
        self.data_creator = data_creator
        self.loss_creator = loss_creator
        self.verbose = verbose
        self.config = worker_config or {}
        self._timers = {
            "setup_proc": TimerStat(window_size=1),
            "setup_model": TimerStat(window_size=1),
            "get_state": TimerStat(window_size=1),
            "set_state": TimerStat(window_size=1),
            "validation": TimerStat(window_size=1),
            "training": TimerStat(window_size=1),
        }
        self.local_rank = None
        self.lr_config = lr_config or {"initial_lr": 0.1}
        assert isinstance(lr_config, dict), type(lr_config)
        self.adjuster = None

    def set_device(self, original_cuda_id=None):
        self.local_rank = int(
            original_cuda_id or os.environ["CUDA_VISIBLE_DEVICES"]
        )

    def setup_proc_group(self, dist_url, world_rank, world_size):
        # self.try_stop()
        import torch.distributed as dist

        with self._timers["setup_proc"]:
            self.world_rank = world_rank
            if self.verbose:
                print(
                    f"Inputs to process group: dist_url: {dist_url} "
                    f"world_rank: {world_rank} world_size: {world_size}"
                )

            # Turns out NCCL is TERRIBLE, do not USE - does not release resources
            backend = "nccl" if self.use_nccl else "gloo"
            logger.warning(f"using {backend}")
            dist.init_process_group(
                backend=backend,
                init_method=dist_url,
                rank=world_rank,
                world_size=world_size,
            )

    def setup_model(self):
        with self._timers["setup_model"]:
            trainset, valset = self.data_creator()

            # Create DistributedSampler to handle distributing the dataset across nodes when training
            # This can only be called after torch.distributed.init_process_group is called
            logger.warning("Creating distributed sampler")
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset
            )
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(
                valset
            )

            # Create the Dataloaders to feed data to the training and validation steps
            logger.warning(f"Using a batch-size of {self.batch_size}")
            self.train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=self.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=self.config.get("data_loader_workers", 4),
                pin_memory=self.config.get("data_loader_pin", False),
                sampler=self.train_sampler,
            )

            # self._train_iterator = iter(self.train_loader)
            self.val_loader = torch.utils.data.DataLoader(
                valset,
                batch_size=self.batch_size,
                shuffle=(self.val_sampler is None),
                num_workers=self.config.get("data_loader_workers", 4),
                pin_memory=self.config.get("data_loader_pin", False),
                sampler=self.val_sampler,
            )

            # self._train_iterator = iter(self.train_loader)
            self.model = self.model_creator().cuda()
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr_config["initial_lr"],
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

            if self.config.get("fp16"):
                assert (
                    torch.backends.cudnn.enabled
                ), "Amp requires cudnn backend to be enabled."
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level="O2"
                )

            # Make model DistributedDataParallel
            logger.warning("Creating DDP Model")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            logger.warning("Finished creating model.")

            # define loss function (criterion) and optimizer
            self.criterion = self.loss_creator().cuda()
        logger.warning("Finished Setup model.")
        return len(self.train_loader.dataset) + len(self.val_loader.dataset)

    def _set_adjuster(self):
        if not self.lr_config:
            return
        if self.adjuster:
            logger.warning("Overriding adjuster.")
        self.adjuster = Adjuster(
            self.lr_config["initial_lr"],
            self.lr_config.get("target_lr"),
            steps=self.lr_config.get("steps"),
            mode=self.lr_config.get("mode"),
        )

    def apply_adjusted_lr(self):
        if not self.adjuster:
            self._set_adjuster()

        return self.adjuster.adjust()

    def step(self):
        # Set epoch count for DistributedSampler
        self.train_sampler.set_epoch(self.epoch)

        # Adjust learning rate according to schedule
        lr = self.apply_adjusted_lr()
        # Decay is applied at runtime
        adjust_learning_rate(
            lr, self.optimizer, self.epoch, decay=self.config["decay"]
        )

        # train for one self.epoch

        if self.verbose:
            print("\nBegin Training Epoch {}".format(self.epoch + 1))
        train_stats = train(
            self.train_loader,
            self.model,
            self.criterion,
            self.optimizer,
            self.epoch,
            max_steps=self.config.get("max_train_steps"),
            fp16=self.config.get("fp16"),
        )

        # evaluate on validation set

        if self.verbose:
            print("Begin Validation @ Epoch {}".format(self.epoch + 1))
        val_stats = validate(
            self.val_loader,
            self.model,
            self.criterion,
            max_steps=self.config.get("max_val_steps"),
        )
        self.epoch += 1
        train_stats.update(val_stats)
        train_stats.update(self.stats())

        if self.verbose:
            print({k: type(v) for k, v in train_stats.items()})
        return train_stats

    def stats(self):
        stats = {}
        for k, t in self._timers.items():
            stats[k + "_time_mean"] = t.mean
            stats[k + "_time_total"] = t.sum
            t.reset()
        return stats

    def get_state(self, ckpt_path):
        with self._timers["get_state"]:
            try:
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
            except OSError:
                logger.exception("failed making dirs")
            if self.verbose:
                print("getting state")
            state_dict = {}
            tmp_path = os.path.join(
                ckpt_path, ".state{}".format(self.world_rank)
            )
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "opt": self.optimizer.state_dict(),
                },
                tmp_path,
            )

            with open(tmp_path, "rb") as f:
                state_dict["model_state"] = f.read()

            os.unlink(tmp_path)
            state_dict["epoch"] = self.epoch

            state_dict["learning_rate"] = self.adjuster.current_lr
            if self.verbose:
                print("Got state.")

        return state_dict

    def set_state(self, state_dict, ckpt_path):
        with self._timers["set_state"]:
            if self.verbose:
                print("setting state for {}".format(self.world_rank))
            try:
                os.makedirs(ckpt_path)
            except OSError:
                print("failed making dirs")
            tmp_path = os.path.join(
                ckpt_path, ".state{}".format(self.world_rank)
            )

            with open(tmp_path, "wb") as f:
                f.write(state_dict["model_state"])

            checkpoint = torch.load(tmp_path)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["opt"])

            os.unlink(tmp_path)
            # self.model.train()
            self.epoch = state_dict["epoch"]
            self.lr_config["initial_lr"] = state_dict["learning_rate"]
            self._set_adjuster()
            if self.verbose:
                print("Loaded state.")

    def try_stop(self):
        logger.debug("Stopping worker.")
        try:
            import torch.distributed as dist

            dist.destroy_process_group()
        except Exception:
            logger.exception("Stop failed.")

    def get_host(self):
        return os.uname()[1]

    def node_ip(self):
        return ray.services.get_node_ip_address()

    def find_port(self):
        return find_free_port()


class NodeColocatorActor:
    """Object that is called when launching the different nodes

    Should take in N number of gpus in the node (and the location of the cluster?)
    and create N actors with num_gpu=0 and place them on the cluster.
    """

    def __init__(self, batch_size, num_gpus, config):
        RemotePyTorchRunner = ray.remote(PyTorchRunner)
        logger.warning(f"Colocator launched on: {os.uname()[1]}")
        args = [
            config["batch_per_device"],
            config["momentum"],
            config["weight_decay"],
            config["model_creator"],
            config["data_creator"],
            config["loss_creator"],
            config["verbose"],
            config["use_nccl"],
            config["worker_config"],
            config["lr_config"],
        ]
        self.remote_workers = create_colocated(
            RemotePyTorchRunner, args, count=num_gpus
        )
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        logger.warning(f"Colocator sharing {gpu_ids}")

        assert len(gpu_ids) == len(self.remote_workers)
        for dev_id, worker in zip(gpu_ids, self.remote_workers):
            worker.set_device.remote(dev_id)

    def get_workers(self):
        return self.remote_workers


class PytorchSGD(ResourceTrainable):
    ADDRESS_TMPL = "tcp://{ip}:{port}"

    metric = "mean_accuracy"

    @property
    def num_workers(self):
        return self.resources._asdict()["extra_gpu"]

    def _setup_impl(self, config):
        global logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        # devices_per_worker = self.config["devices_per_worker"]
        self._initial_timing = True  # This is set on first restart.
        self.t1 = None  # This is set on first restart
        self._next_iteration_start = time.time()
        self._time_so_far = 0
        self._data_so_far = 0
        self.resource_time = 0

        self.config = config or DEFAULT_CONFIG
        print("workers", self.num_workers)

        self._placement_set = self.config["placement"] or [
            1 for i in range(self.num_workers)
        ]

        config["batch_per_device"] = max(
            int(config["target_batch_size"] / self.num_workers),
            config["min_batch_size"],
        )

        batch_scaling = (
            self.num_workers
            * config["batch_per_device"]
            / config["target_batch_size"]
        )
        target_lr = config["starting_lr"] * np.sqrt(batch_scaling)
        logger.warning(f"Scaling learning rate to {target_lr}")
        config["lr_config"].update(
            initial_lr=config["starting_lr"], target_lr=target_lr
        )

        assert sum(self._placement_set) == self.num_workers
        self.colocators = []
        self.remote_workers = []
        logger.warning(f"Placement set is {self._placement_set}")
        for actors_in_node in self._placement_set:
            if not actors_in_node:
                continue
            if actors_in_node == 1:
                self.remote_workers += [self._create_single_worker(config)]
            else:
                self.remote_workers += self._create_group_workers(
                    actors_in_node, config
                )
        logger.warning(
            "MASTER: {} workers started. Syncing.".format(
                len(self.remote_workers)
            )
        )
        self._sync_all_workers()
        logger.warning("MASTER: Finished syncing.")

    def _create_single_worker(self, config):
        logger.warning("AVAIL CLUSTER RES {}".format(ray.cluster_resources()))
        RemotePyTorchRunner = ray.remote(num_gpus=1)(PyTorchRunner)
        worker = RemotePyTorchRunner.remote(
            config["batch_per_device"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            model_creator=config["model_creator"],
            data_creator=config["data_creator"],
            loss_creator=config["loss_creator"],
            verbose=config["verbose"],
            use_nccl=config["use_nccl"],
            worker_config=config["worker_config"],
            lr_config=config["lr_config"],
        )
        worker.set_device.remote()
        return worker

    def _create_group_workers(self, actors_in_node, config):
        RemoteColocator = ray.remote(num_gpus=int(actors_in_node))(
            NodeColocatorActor
        )
        colocator = RemoteColocator.remote(
            self.config["target_batch_size"], int(actors_in_node), self.config
        )
        self.colocators += [colocator]

        return ray.get(colocator.get_workers.remote())

    def _sync_all_workers(self):
        setup_futures = []
        master_ip = None
        master_port = None
        logger.warning("MASTER: Running ack for all workers.")
        ray.get([w.get_host.remote() for w in self.remote_workers])
        logger.warning("MASTER: Received ack from all workers.")

        for world_rank, worker in enumerate(self.remote_workers):
            if not master_ip:
                master_ip = ray.get(worker.node_ip.remote())
                master_port = ray.get(worker.find_port.remote())
            setup_futures += [
                worker.setup_proc_group.remote(
                    self.ADDRESS_TMPL.format(ip=master_ip, port=master_port),
                    world_rank,
                    len(self.remote_workers),
                )
            ]

        ray.get(setup_futures)
        logger.warning("MASTER: Setting up models.")
        data_size = ray.get(
            [worker.setup_model.remote() for worker in self.remote_workers][0]
        )
        self._data_size = data_size

    def _setup(self, config):
        self.session_timer = {
            "setup": TimerStat(window_size=1),
            "train": TimerStat(window_size=2),
        }
        self.metric_stats = {"accuracy": [], "loss": []}
        with self.session_timer["setup"]:
            self._setup_impl(config)

    def _train(self):
        worker_stats = ray.get([w.step.remote() for w in self.remote_workers])
        # res = self._fetch_metrics_from_remote_workers()
        df = pd.DataFrame(worker_stats)
        results = df.mean().to_dict()
        self._data_so_far += self._data_size

        results.update(samples=self._data_size)  # TODO: doublecheck this
        # results.update(total_data_processed=self._data_so_far)
        # if self.config["dataset"] == "CIFAR":
        #     results.update(epochs_processed=self._data_so_far / 50000)
        results.update(num_workers=self.num_workers)
        self.metric_stats["accuracy"] += [results["val_accuracy"]]
        self.metric_stats["loss"] += [results["val_loss"]]
        results.update(
            mean_accuracy=np.mean(self.metric_stats["accuracy"][-5:]),
            mean_loss=np.mean(self.metric_stats["loss"][-5:]),
        )
        return results

    def _save(self, ckpt):
        return {
            "worker_state": ray.get(
                self.remote_workers[0].get_state.remote(ckpt)
            ),
            "metric_stats": self.metric_stats,
            "time_so_far": self._time_so_far,
            "data_so_far": self._data_so_far,
            "resource_time": self.resource_time,
            "ckpt_path": ckpt,
            "t1": self.t1 or self.session_timer["train"].mean,
        }

    def _restore(self, ckpt):
        self._time_so_far = ckpt["time_so_far"]
        self._data_so_far = ckpt["data_so_far"]
        self.t1 = ckpt["t1"]
        self.metric_stats = ckpt["metric_stats"]
        self.resource_time = ckpt["resource_time"]
        self._initial_timing = False
        worker_state = ray.put(ckpt["worker_state"])
        states = []

        for worker in self.remote_workers:
            states += [worker.set_state.remote(worker_state, ckpt["ckpt_path"])]

        ray.get(states)
        self._next_iteration_start = time.time()

    def _stop(self):
        logger.warning("Calling stop on this trainable.")
        stops = []
        for colocator in self.colocators:
            stops += [colocator.__ray_terminate__.remote()]

        for worker in self.remote_workers:
            stops += [worker.try_stop.remote()]
            stops += [worker.__ray_terminate__.remote()]
        logger.warning("Stop signals sent.")

    @classmethod
    def to_atoms(cls, resources):
        return int(resources.extra_gpu)

    @classmethod
    def to_resources(cls, atoms):
        return Resources(0, 0, extra_gpu=atoms)
