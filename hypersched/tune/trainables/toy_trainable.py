import numpy as np
import json
import os
import time
import logging

from hypersched.tune import ResourceTrainable
from hypersched.utils import check
from hypersched.function import OptimusFunction

from ray import tune
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.trial import Resources, Trial

logger = logging.getLogger(__name__)

DEFAULT_SCALING = {1: 1, 2: 2, 4: 4, 8: 8, 16: 16}


DEFAULT_CONFIG = {
    "seed": None,
    "delay": 0.1,
    "startup_delay": 0.001,
    "param1": 0.1,
    "param2": 0.1,
    "param3": 0.1,
    "scaling": None,
    "dummy": False,
}

DEFAULT_HSPACE = {
    "param1": tune.sample_from(lambda spec: np.random.exponential(0.1)),
    "param2": tune.sample_from(lambda _: np.random.rand()),
    "param3": tune.sample_from(lambda _: np.random.rand()),
}

DEFAULT_MULTIJOB_CONFIG = {
    "min_allocation": 5,  # Model setup time can be 20, overall first epoch setup can take up to 100
    "max_allocation": 500,
    "time_attr": "training_iteration",
}

import cloudpickle


class OptimusTrainable(ResourceTrainable):
    dummy = False
    metric = "mean_accuracy"

    @classmethod
    def to_atoms(cls, resource):
        return int(resource.cpu)

    @classmethod
    def to_resources(cls, atoms):
        return Resources(cpu=atoms, gpu=0)

    def _setup(self, config):
        self.iter = 0
        self._next_iteration_start = time.time()
        self._time_so_far = 0
        if config.get("dummy"):
            self.dummy = True
        if config.get("seed"):
            np.random.seed(config["seed"])
        self._delay = config["delay"]
        time.sleep(config.get("startup_delay", 0))
        params = [config["param1"], config["param2"], config["param3"]]
        self._initial_samples_per_step = 500
        self.func = OptimusFunction(
            params=params, scaling=self.config["scaling"]
        )

    def _train(self):
        time.sleep(self._delay / self.func.scaling(self.atoms))
        self.iter += 1
        if self.dummy:
            return {
                "mean_loss": -self.iter,
                "mean_accuracy": self.iter,
                "samples": self._initial_samples_per_step,
            }
        new_loss = self.func.step(1, self.iter)
        return {
            "mean_loss": float(new_loss),
            "mean_accuracy": (2 - new_loss) / 2,
            "samples": self._initial_samples_per_step,
        }

    def _save(self, checkpoint_dir):
        return {
            "func": cloudpickle.dumps(self.func),
            "seed": np.random.get_state(),
        }

    def _restore(self, checkpoint):
        self.func = cloudpickle.loads(checkpoint["func"])
        np.random.set_state(checkpoint["seed"])
