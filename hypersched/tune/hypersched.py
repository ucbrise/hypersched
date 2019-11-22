from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time

import numpy as np

from hypersched.tune.dyn_allocator import DynamicAllocator
from hypersched.tune.ashav2 import _Bracket as ASHAv2Bracket
from hypersched.tune.ashav2 import ASHAv2
from ray.tune.trial import Trial
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from hypersched.utils import check

logger = logging.getLogger(__name__)

TERM_STATES = (Trial.ERROR, Trial.TERMINATED)

SCALING_MAP = {
    "NONE": {1: 1, 2: 1, 4: 1, 8: 1, 16: 1, 32: 1},
    "SQRT": {1: 1, 2: 1.4, 4: 2, 8: 2.8, 16: 4, 32: 5.7},
    "HALF": {1: 1, 2: 1, 4: 2, 8: 4, 16: 8, 32: 16},
    "LINEAR": dict((i, i) for i in range(1, 32)),
}


def scaling_function_from_dict(throughput_map, total_atoms=None):
    def scaling_function(query):
        return np.interp(
            query, list(throughput_map.keys()), list(throughput_map.values())
        )

    return scaling_function


class HyperSched(FIFOScheduler):
    """Implements the Async Successive Halving with retro kill.

    Args:
        total_atoms (int): Logical representation of total resources available.
        resource_policy: How to allocate free resources. Choose among
            "UNIFORM", "RANDOM".
        scaling dict: can be normalized.
        deadline: When the experiment finishes.
        allocation_grid (int): Let `allocation_grid` be Z.
            Do not place/resize job unless the trial allocation < Z or
            trial allocation % Z == 0.
        use_pausing: Pause trials instead of killing.
        grace_period (int): "r" in ASHA. See ASHAv2.py
        max_t (float): "R" in ASHA. See ASHAv2.py
        reduction_factor (float): Parameter for exploration/exploit.
            See ASHAv2.py.
        time_attr: The metric to use forr tracking time allocation per trial.
        metric: Optimization metric. Assume INCREASING.
        _no_speculation (bool): Used for ablation studies. See code for more
            details.
        _ignore_overhead (bool): Used for ablation studies. See code for more
            details.
        _no_job_limit (bool): Used for ablation studies. See code for more
            details.
        _assume_linear (bool): Used for ablation studies. See code for more
            details.
        _fixed_exploration (bool): Used for ablation studies. See code for more
            details.
        _exploration_ratio: Used for ablation studies. See code for more
            details.

    """

    def __init__(
        self,
        total_atoms,
        resource_policy="UNIFORM",
        scaling_dict=SCALING_MAP["LINEAR"],
        deadline=np.inf,
        allocation_grid=None,
        use_pausing=True,
        grace_period=1,
        reduction_factor=4,
        max_t=100,
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        _no_speculation=False,
        _ignore_overhead=False,
        _no_job_limit=False,
        _assume_linear=False,
        _fixed_exploration=False,
        _exploration_ratio=1.0,
    ):
        # Arguments for ablative study
        self._no_speculation = _no_speculation  # stored
        self._ignore_overhead = _ignore_overhead  # stored
        self._no_job_limit = _no_job_limit  # stored
        self._assume_linear = _assume_linear
        self._fixed_exploration = _fixed_exploration
        self._exploration_ratio = _exploration_ratio
        FIFOScheduler.__init__(self)

        self.use_pausing = use_pausing
        self._num_paused = 0
        self._num_stopped = 0
        self._reduction_factor = reduction_factor
        self._max_t = max_t
        self._metric = metric
        self._time_attr = time_attr
        if mode == "max":
            self._metric_op = 1.0
        elif mode == "min":
            self._metric_op = -1.0

        if self._no_speculation:
            self._brackets = [
                ASHAv2Bracket(
                    min_t=grace_period,
                    max_t=self._max_t,
                    reduction_factor=self._reduction_factor,
                    s=0,
                )
            ]
        else:
            self._brackets = [
                _DeadlineBracket(
                    self._reduction_factor,
                    max_t=self._max_t,
                    min_t=grace_period,
                    use_pausing=self.use_pausing,
                )
            ]

        if self._fixed_exploration:
            logger.warning(
                f"FIXED EXPLORATION TIME OF {self._exploration_ratio}"
            )

        if self._fixed_exploration:
            logger.warning(
                f"FIXED EXPLORATION TIME OF {self._exploration_ratio}"
            )

        self.grace_period = grace_period
        self.start_time = time.time()
        self._deadline = deadline
        self._deadline_time = deadline + time.time()
        self._longest_duration = -1
        check(self._deadline_time > self.start_time)

        self.total_atoms = total_atoms
        self.allocator = DynamicAllocator(
            self.total_atoms,
            policy=resource_policy,
            allocation_grid=allocation_grid,
            recharge_period=5,
            metric=self._metric,
            metric_op=self._metric_op,
        )

        if self._assume_linear:
            logger.warning("ABLATION: ASSUMING LINEAR SCALING.")
            scaling_dict = SCALING_MAP["LINEAR"]
        self.scaling_fn = scaling_function_from_dict(scaling_dict)
        self._startup_times = set()
        #: Time it takes for a single iteration
        self._single_atom_iteration_times = []

    def on_trial_result(self, trial_runner, trial, result):
        """First check ASHA

        Nothing should occur bc of plateau.
        """
        decision = ASHAv2.on_trial_result(self, trial_runner, trial, result)
        primary = decision
        # if self.early_stopper.should_kill(trial, result):
        #     decision = TrialScheduler.STOP
        self._startup_times.add(result["setup_time"])
        if result["atoms"] == 1:
            self._single_atom_iteration_times += [result["time_this_iter_s"]]
        self._longest_duration = max(
            self._longest_duration, result["time_total_s"]
        )
        self.allocator.populate_if_needed(self.get_live_trials(trial_runner))

        # This is commented out because there is no "safe" way to argue
        # for this.
        # if self.time_left < self._deadline / self._reduction_factor:
        #     logger.warning("Since time left is less than reduction factor, "
        #                    f"POLICY IS TOP_JOB")
        #     self.allocator.set_policy("TOP_JOB")

        # We reallocate according to the current decision. This should be
        # the last change of decision, and will only happen if
        decision = self.allocator.on_result(
            trial_runner, trial, decision, execute=False
        )

        if decision == TrialScheduler.CONTINUE:
            # Check if with the new allocation, there is improvement
            check(self.allocator.get_proposed_atoms(trial))
            if self._improved_progress_after_reallocation(trial):
                self.allocator.try_execute_update(trial, trial_runner)

        if self._no_speculation and self.allocator._policy == "NONE":
            check(primary == decision)

        return decision

    def _improved_progress_after_reallocation(self, trial):
        """Checks the proposed allocation.

        If the benefit of allocation outweighs the cost, return True.
        """
        if self.time_left < 0:
            return False
        new_allocation = self.allocator.get_proposed_atoms(trial)
        current_allocation = self.allocator.get_current_atoms(trial)
        if new_allocation < current_allocation:
            # check(current_allocation - new_allocation > 1)
            logger.warning(
                f"Proposed deallocation from {current_allocation}"
                f"to {new_allocation} - executing."
            )
            return True
        new_progress = self.scaling_fn(new_allocation) * (
            self.time_left - self.startup_time
        )
        old_progress = self.time_left * self.scaling_fn(current_allocation)
        execute = new_progress / old_progress > 1.1
        if execute:
            logger.debug(
                f"Projected {new_progress:0.3f}  > {old_progress:0.3f}. "
                "Seeing if we should update."
            )
        else:
            logger.debug(
                f"Projected {new_progress:0.3f} ~< {old_progress:0.3f}. "
                "Not updating."
            )
        return execute

    def choose_trial_to_run(self, trial_runner):
        fair_allocation = (
            self._longest_duration - self.startup_time
        ) * self._reduction_factor
        fair_allocation = min(fair_allocation, self.expected_trial_duration)
        if self._fixed_exploration:
            fair_allocation = self._deadline * (1 - self._exploration_ratio)

        if self._no_job_limit:
            logger.warning("ABLATION: NO JOB LIMIT")
            if self._deadline_time - time.time() < 0:
                for t in self.get_pending(trial_runner):
                    t.status = Trial.TERMINATED

                for t in self.get_paused(trial_runner):
                    self._brackets[0].unpause_trial(t)
                    t.status = Trial.TERMINATED
                return None

        elif self.time_left < fair_allocation:
            logger.debug(
                f"Time left: {self.time_left}. "
                f"Fair Allocation: {fair_allocation}. "
                "Not running new trial."
            )
            for t in self.get_pending(trial_runner):
                t.status = Trial.TERMINATED

            for t in self.get_paused(trial_runner):
                self._brackets[0].unpause_trial(t)
                t.status = Trial.TERMINATED
            return None

        if self.use_pausing:
            return ASHAv2.choose_trial_to_run(self, trial_runner)
        return super().choose_trial_to_run(trial_runner)

    def debug_string(self):
        title = (
            f"Hypersched: {self._deadline:.3f}. "
            "Time left: {self._deadline_time - time.time():.3f}"
        )
        startup_time = f"Calculated startup time: {self.startup_time:.3f}"
        expected_time = (
            f"Expected New Trial Duration: {self.expected_trial_duration:.3f}"
        )
        status = " | ".join([title, startup_time, expected_time])
        return "\n".join([status, self._brackets[0].debug_str()])

    def get_live_trials(self, runner):
        return [t for t in runner.get_trials() if t.status not in TERM_STATES]

    def get_pending(self, trial_runner):
        return [
            t for t in trial_runner.get_trials() if t.status == Trial.PENDING
        ]

    def get_paused(self, trial_runner):
        return [
            t for t in trial_runner.get_trials() if t.status == Trial.PAUSED
        ]

    @property
    def startup_time(self):
        if self._ignore_overhead:
            # logger.warning("ABLATION: IGNORING STARTUP TIME.")
            return 0
        if self._startup_times:
            return np.percentile(list(self._startup_times), 90)
        else:
            return -1

    @property
    def time_left(self):
        return self._deadline_time - time.time()

    @property
    def expected_trial_duration(self):
        return (self._max_t) * np.median(self._single_atom_iteration_times)


# NOTE: This is different from dynhyperband.py
class _Rung:
    def __init__(self, rf, use_pausing=False):
        self.recorded = {}
        self.trials_kept = set()
        self._use_pausing = use_pausing
        self.paused = []
        self.rf = rf
        self.milestone = np.inf

    def set_milestone(self, milestone):
        logger.warning(f"Milestone set at {milestone}")
        self.milestone = milestone

    def put(self, trial):
        self.trials_kept.add(trial)

    def pop(self, trial):
        if trial in self.trials_kept:
            self.trials_kept.remove(trial)
        if self._use_pausing:
            self.paused += [trial]

    def record_if_needed(self, trial, cur_rew):
        return self.recorded.setdefault(trial, cur_rew)

    def descending_paused(self):
        for t in self.paused:
            check(t in self.recorded)
        return sorted(
            self.paused, key=lambda t: self.recorded[t], reverse=True
        )

    def cutoff(self):
        if len(self.recorded) > 1:
            return np.percentile(
                list(self.recorded.values()), (1 - 1 / self.rf) * 100
            )

    def check_to_kill(self, trial, cur_rew):
        cutoff = self.cutoff()
        # Check if the reward already was evaluated; if not, use current_reward
        evaluted_reward = self.record_if_needed(trial, cur_rew)
        retrospective = not evaluted_reward == cur_rew
        kill = cutoff is not None and evaluted_reward < cutoff
        if kill and retrospective:
            logger.info(
                "Recommending to kill {} retrospectively.".format(trial)
            )
        return kill

    def __str__(self):
        return "[{:.3f}]: {:0.3f} ({})".format(
            self.milestone, self.cutoff() or 0, len(self.trials_kept)
        )


# NOTE: This is different from dynhyperband.py
class _DeadlineBracket:
    """Bracket inspired by HyperBand.

    Reduce the bracket 3 times.
    """

    # TODO: Implement Rung quantities?
    # TODO: Implement thresholding based off of max effective res time
    def __init__(
        self, reduction_factor, max_t=100, min_t=1, use_pausing=False
    ):
        self.rf = reduction_factor

        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(reduction_factor) + 1)
        self._use_pausing = use_pausing
        self._rungs = []
        for i in reversed(range(MAX_RUNGS + 1)):
            rung = _Rung(reduction_factor, use_pausing=use_pausing)
            rung.set_milestone(min_t * self.rf ** (i))
            self._rungs += [rung]

    def on_result(self, trial, cur_iter, cur_rew):
        action = TrialScheduler.CONTINUE
        for rung in self._rungs:
            if cur_iter >= rung.milestone:
                if rung.check_to_kill(trial, cur_rew):
                    rung.pop(trial)
                    if self._use_pausing:
                        logger.info(f"{trial} is less than cutoff! Pause.")
                        return TrialScheduler.PAUSE
                    logger.info(
                        "{} is less than cutoff! Stopping.".format(trial)
                    )
                    return TrialScheduler.STOP
                rung.put(trial)

        return action

    def debug_str(self):
        iters = " | ".join([f"{rung}" for rung in self._rungs])
        return "Bracket: " + iters

    def promotable_trials(self):
        assert self._use_pausing
        for rung in self._rungs:
            if rung.cutoff() is None:
                continue
            for trial in rung.descending_paused():
                if rung.recorded[trial] >= rung.cutoff():
                    yield trial

    def unpause_trial(self, trial):
        assert self._use_pausing
        found = False
        for rung in self._rungs:
            paused = rung.paused
            if trial in paused:
                found = True
                paused.pop(paused.index(trial))
            assert trial not in paused
        return found
