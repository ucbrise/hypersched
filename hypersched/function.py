#!/usr/bin/env python
import logging
import pandas as pd
from hypersched.utils import check
import numpy as np
import uuid

logger = logging.getLogger(__name__)


def variance_check(values, delta=1e-6):
    return np.var(values) < 1e-6


class Function:
    def __init__(self):
        self._learning_fn = self._generate_learning_func()
        # self._scale_factor = self._generate_speed_func()
        self.setup()

    def step(self, resources, itr):
        check(
            not (resources > 0 and self._done), "Should not step on done job!"
        )
        self._progress += int(self._scale_factor(resources))
        self._iteration += 1
        score = self.score()
        self._scores += [(self._progress, score)]
        if resources:
            self._progressed_scores += [(score, itr)]
        return score

    def score(self):
        return self._learning_fn(self._progress)

    def derivative(self, last_n=10):
        progress, scores = zip(*self._scores[-last_n:])
        return (scores[-1] - scores[-last_n]) / (
            progress[-1] - progress[-last_n]
        )

    def done(self):
        raise NotImplementedError

    def terminate(self, done_info):
        check(bool("terminate" not in self.info))
        self.running = False
        self._done = True
        self.info["terminate"] = done_info

    def setup(self):
        self._iteration = 0
        self._progress = 0  # some equivalent to epochs
        self._progressed_scores = []
        self._scores = []
        self.info = {}
        self.running = True
        self._reported = False
        self._done = False
        self.early_stop = True
        self.fid = uuid.uuid4().hex[:20]

    @property
    def cpu_time(self):
        return self._progress

    def debug_string(self):
        return "{}: progress {} - score {} - {}".format(
            self,
            self.cpu_time,
            self.score(),
            "running" if self.running else "",
        )

    def evaluate(self, cpu_time):
        return self._learning_fn(cpu_time)

    def __str__(self):
        return "Fn [{}]".format(self.fid)

    def to_dataframe(self, max_cpu_time):
        return pd.DataFrame([self.evaluate(i) for i in range(max_cpu_time)])

    def scaling(self, resources):
        return self._scale_factor(resources)


class BasicFunction(Function):
    def _generate_learning_func(self):
        b, c = np.random.rand(2)
        logger.debug("using %f, %f" % (b, c))
        return lambda x: (b * 0.01 * x + 0.1) ** (-0.2 * c)

    def _generate_speed_func(self):
        return lambda res: res - (0.1 * res) ** 2 if res >= 1 else res

    def done(self):
        if self._done:
            return True
        elif self._progress <= 5:
            return False
        if self.early_stop:
            if variance_check(
                [self._learning_fn(p) for p in self._progress - np.r_[:5]]
            ):
                self._done = True
        else:
            self._done = self._progress > 2000
        if self._done:
            self.running = False
        return self._done


class DiscreteFunction(Function):
    def step(self, resources, itr):
        # isinstance(np.dtype('int8'), numbers.Integral)
        # check(type(resources) is int)
        return Function.step(self, resources, itr)

    def cpu_time(self):
        raise DeprecationWarning("CPUTime for discrete functions not a thing.")

    def evaluate(self, steps):
        return self._learning_fn(steps)


class OptimusFunction(DiscreteFunction):
    """Use Modeling functions."""

    def __init__(self, params=None, scaling={1: 1, 2: 2}):
        self.params = params or np.random.rand(3)
        self._learning_fn = self._generate_learning_func()
        self._scale_factor = self._generate_speed_func(scaling)
        self.setup()

    def _generate_learning_func(self, max_t=10000):
        """k ranges from 0 to 1000.
        TODO: should range greater.

        Returns:
            fn: steps (int) -> loss (reals)"""
        b0, b1, b2 = self.params
        noise = np.random.normal(size=max_t) * 0.005

        def _loss_curve(k, add_noise=True):
            assert type(k) is int, type(k)
            score = (b0 * k / 100 + 0.1 * b1 + 0.5) ** (-1) + b2 * 0.01
            if add_noise:
                return score + abs(noise[k])
            else:
                return score

        self.progress_needed = None
        for i in range(0, max_t, 5):
            if variance_check(
                [
                    _loss_curve(int(r), add_noise=False)
                    for r in np.r_[i : i + 5]
                ]
            ):
                self.progress_needed = i
                break

        return _loss_curve

    def _generate_speed_func(self, speed_config):
        """Generates a speed function..

        Returns:
            Function: resources (Z) -> steps/sec (Q), with non-linear scaling.
        """
        # a, b = np.random.rand(2)
        if speed_config:

            def scaling_function(query):
                return np.interp(
                    query, list(speed_config), list(speed_config.values())
                )

            return scaling_function
        raise NotImplementedError("Need speed_config")

    def done(self):
        if self._done:
            return True
        elif self._progress <= 5:
            return False
        if self.early_stop:
            if variance_check(
                [
                    self._learning_fn(p, add_noise=False)
                    for p in self._progress - np.r_[:5]
                ]
            ):
                self._done = True
        else:
            self._done = self._progress > 2000
        if self._done:
            self.running = False
        return self._done
