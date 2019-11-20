import ray
import time
import numpy as np
from ray.tune import Trainable
from .utils import TimerStat


class ResourceTrainable(Trainable):
    def __init__(self, config=None, logger_creator=None, resources=None):
        import os

        cwd = os.getcwd()
        self._resources = resources
        self._sample_counter = TimerStat(window_size=5)
        if ray.worker._mode() == ray.worker.LOCAL_MODE:
            os.chdir(cwd)
        self._session_start = time.time()
        self._warmup_period = None
        super(ResourceTrainable, self).__init__(config, logger_creator)

    def train(self):
        if self._warmup_period is None:
            self._warmup_period = time.time() - self._session_start
        with self._sample_counter:
            result = super(ResourceTrainable, self).train()
        assert (
            "samples" in result
        ), "Need to return 'samples' - abstraction is broken!"
        self._sample_counter.push_units_processed(result["samples"])
        del result["config"]
        result.update(mean_throughput=self._sample_counter.mean_throughput)
        result.update(atoms=self.atoms)
        result.update(setup_time=self._warmup_period)
        print(result)
        return result

    @classmethod
    def to_atoms(cls, resources):
        raise NotImplementedError

    @classmethod
    def to_resources(cls, atoms):
        raise NotImplementedError

    @property
    def atoms(self):
        return self.to_atoms(self.resources)

    @property
    def resources(self):
        return self._resources
