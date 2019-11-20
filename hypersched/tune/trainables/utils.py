import time
import numpy as np
import os
import ray
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TimerStat(object):
    """A running stat for conveniently logging the duration of a code block.

    Example:
        wait_timer = TimerStat()
        with wait_timer:
            ray.wait(...)

    Note that this class is *not* thread-safe.
    """

    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    def push_units_processed(self, n):
        self._units_processed.append(n)
        if len(self._units_processed) > self._window_size:
            self._units_processed.pop(0)

    @property
    def mean(self):
        return np.mean(self._samples)

    @property
    def median(self):
        return np.median(self._samples)

    @property
    def sum(self):
        return np.sum(self._samples)

    @property
    def max(self):
        return np.max(self._samples)

    @property
    def first(self):
        return self._samples[0] if self._samples else None

    @property
    def last(self):
        return self._samples[-1] if self._samples else None

    @property
    def size(self):
        return len(self._samples)

    @property
    def mean_units_processed(self):
        return float(np.mean(self._units_processed))

    @property
    def mean_throughput(self):
        time_total = sum(self._samples)
        if not time_total:
            return 0.0
        return sum(self._units_processed) / time_total

    def reset(self):
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0


def drop_colocated(actors):
    colocated, non_colocated = split_colocated(actors)
    for a in colocated:
        a.__ray_terminate__.remote()
    return non_colocated


def split_colocated(actors):
    localhost = os.uname()[1]
    hosts = ray.get([a.get_host.remote() for a in actors])
    local = []
    non_local = []
    for host, a in zip(hosts, actors):
        if host == localhost:
            local.append(a)
        else:
            non_local.append(a)
    return local, non_local


def try_create_colocated(cls, args, count):
    actors = [cls.remote(*args) for _ in range(count)]
    local, rest = split_colocated(actors)
    logger.info("Got {} colocated actors of {}".format(len(local), count))
    for a in rest:
        a.__ray_terminate__.remote()
    return local


def create_colocated(cls, args, count):
    logger.info("Trying to create {} colocated actors".format(count))
    ok = []
    i = 1
    while len(ok) < count and i < 10:
        attempt = try_create_colocated(cls, args, count * i)
        ok.extend(attempt)
        i += 1
    if len(ok) < count:
        raise Exception("Unable to create enough colocated actors, abort.")
    for a in ok[count:]:
        a.__ray_terminate__.remote()
    return ok[:count]


class EarlyStopper:
    def __init__(
        self, perf_metric="episode_reward_mean", min_delta=1e-4, patience=5
    ):
        self._min_delta = min_delta
        self._patience = patience
        self.perf_metric = perf_metric
        self.monitor_op = lambda a, b: np.greater(a, b + self._min_delta)
        self._best = {}
        self._wait_counter = defaultdict(int)

    def should_kill(self, trial, result):
        if trial in self._best:
            if not self.monitor_op(result[self.perf_metric], self._best[trial]):
                self._wait_counter[trial] += 1
                if self._wait_counter[trial] > self._patience:
                    logger.warning("Killing %s", str(trial))
                    return True
            else:
                self._wait_counter[trial] = 0
            self._best[trial] = max(result[self.perf_metric], self._best[trial])
        else:
            self._best[trial] = result[self.perf_metric]
        return False
