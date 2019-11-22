# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time

import ray
from ray.tune.logger import NoopLogger
from ray.tune.ray_trial_executor import RayTrialExecutor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ResourceExecutor(RayTrialExecutor):
    """An implemention of TrialExecutor based on Ray."""

    def __init__(self, deadline_s=None, hooks=None, **kwargs):
        self.deadline_s = deadline_s or float("inf")
        self._start = time.time()
        self._trials = set()
        self._hooks = hooks
        super(ResourceExecutor, self).__init__(**kwargs)

    def _setup_runner(self, trial, reuse_allowed=False):
        cls = ray.remote(
            num_cpus=trial.resources.cpu,
            num_gpus=trial.resources.gpu,
            resources=trial.resources.custom_resources,
        )(trial._get_trainable_cls())

        trial.init_logger()
        # We checkpoint metadata here to try mitigating logdir duplication
        self.try_checkpoint_metadata(trial)
        remote_logdir = trial.logdir

        def logger_creator(config):
            # Set the working dir in the remote process, for user file writes
            if not os.path.exists(remote_logdir):
                os.makedirs(remote_logdir)
            os.chdir(remote_logdir)
            return NoopLogger(config, remote_logdir)

        # Logging for trials is handled centrally by TrialRunner, so
        # configure the remote runner to use a noop-logger.
        return cls.remote(
            config=trial.config,
            logger_creator=logger_creator,
            resources=trial.resources,
        )

    def fetch_result(self, trial):
        result = super().fetch_result(trial)
        for h in self._hooks:
            h.on_result(trial, result)
        return result

    def on_step_end(self, runner):
        if time.time() - self._start > self.deadline_s:
            logger.warning("Killing all trials - deadline hit.")
            for trial in runner.get_trials():
                self.stop_trial(trial)

            for i in range(20):
                if not ray.available_resources() == ray.cluster_resources():
                    print("Resources not released yet.")
                    time.sleep(5)

    def continue_training(self, trial):
        """Continues the training of this trial."""
        if trial not in set(self._running.values()):
            self._train(trial)
        else:
            logger.warning(
                "Trial {} found in running set. NOOP.".format(trial)
            )
