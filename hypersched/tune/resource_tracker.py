from collections import defaultdict
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EffectiveResourceTimer:
    def __init__(
        self, deadline, target_util, scaling_fn, optimal_atoms, debug_mode=False
    ):
        self.deadline = deadline
        self.debug_mode = debug_mode
        self.scaling_fn = scaling_fn
        self._start_time = time.time()
        self._trial_times = defaultdict(int)
        self._trial_scores = {}
        self.best_eff_scale = self.scaling_fn(optimal_atoms)

        if self.debug_mode:
            self.current_time = 0
        else:
            self.current_time = time.time() - self._start_time

        self.target_eff_time_allocation = (
            target_util * self.deadline * self.best_eff_scale
        )
        self.exploit_time = (
            1 - target_util
        ) * self.deadline + self.current_time

    def on_trial_result(self, trial, score, time_this_iter, atoms):
        """Tracks all trials and their current progress."""
        if self.debug_mode:
            self.current_time += 1
            time_this_iter = 1
        else:
            self.current_time = time.time() - self._start_time

        logger.debug("Current Time {}".format(self.current_time))

        self._trial_times[trial] += self.scaling_fn(atoms) * time_this_iter
        self._trial_scores[trial] = score
        assert len(self._trial_scores) == len(self._trial_times)

    def pop(self, trial):
        assert len(self._trial_scores) == len(self._trial_times)
        if trial in self._trial_scores:
            self._trial_scores.pop(trial)
            self._trial_times.pop(trial)

    def check_to_exploit(self):
        """Checks that the best trial is on track to hitting target effective time."""

        eff_resource_time_left = self.best_eff_scale * (
            self.deadline - self.current_time
        )
        eff_resource_time_needed = (
            self.target_eff_time_allocation - self.best_eff_time_covered
        )
        exploit = eff_resource_time_left < eff_resource_time_needed

        if exploit:
            logger.warning(
                f"Full exploit time: {self.current_time} > {self.exploit_time}"
            )
        return exploit

    @property
    def time_left_till_exploit(self):
        wall_time_needed = (
            self.target_eff_time_allocation - self.best_eff_time_covered
        ) / self.best_eff_scale

        return (self.deadline - self.current_time) - wall_time_needed

    @property
    def best_eff_time_covered(self):
        if self._trial_scores:
            best_trial = max(self._trial_scores.items(), key=lambda t: t[1])[0]
            return self._trial_times[best_trial]
        else:
            return 0


if __name__ == "__main__":
    deadline = 60
    target_usage = 0.7
    scaling_fn = lambda x: x
    optimal_atoms = 10
    tracker = EffectiveResourceTimer(
        deadline, target_usage, scaling_fn, optimal_atoms, debug_mode=False
    )

    for i in range(6):
        time.sleep(i)
        tracker.on_trial_result("test", 10, 1, 1)
        tracker.on_trial_result("test2", 10, 5, 2)
        print(f"Time left till exploit: {tracker.time_left_till_exploit}")
        print(f"Best Time: {tracker.best_eff_time_covered}")
