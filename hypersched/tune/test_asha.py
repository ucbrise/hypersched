from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random
import unittest
import numpy as np
import sys
import tempfile
import shutil
import ray

from ray.tune.result import TRAINING_ITERATION
from ray.tune.schedulers import TrialScheduler
from hypersched.tune.ashav2 import ASHAv2

from ray.tune.schedulers.pbt import explore
from ray.tune.trial import Trial, Checkpoint
from ray.tune.trial_executor import TrialExecutor
from ray.tune.resources import Resources

from ray.rllib import _register_all

_register_all()

if sys.version_info >= (3, 3):
    from unittest.mock import MagicMock
else:
    from mock import MagicMock


def result(t, rew):
    return dict(
        time_total_s=t, episode_reward_mean=rew, training_iteration=int(t)
    )


class ASHAv2Suite(unittest.TestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        ray.shutdown()
        _register_all()  # re-register the evicted objects

    def testAshaGrace(self):
        rule = ASHAv2(grace_period=3, reduction_factor=2)
        t1 = Trial("__fake")  # mean is 450, max 900, t_max=10
        t2 = Trial("__fake")  # mean is 450, max 450, t_max=5
        for i in range(2):
            self.assertEqual(
                rule.on_trial_result(None, t1, result(i, 0)),
                TrialScheduler.CONTINUE,
            )
        self.assertEqual(
            rule.on_trial_result(None, t1, result(i, 0)), TrialScheduler.PAUSE
        )
        for i in range(5):
            self.assertEqual(
                rule.on_trial_result(None, t2, result(i, 450)),
                TrialScheduler.CONTINUE,
            )
