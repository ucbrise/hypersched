from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
import time

import numpy as np

import ray

# from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.trial import Trial, Resources
from hypersched.utils import check

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def add_resources(original, to_add):
    cpu = original.cpu + to_add.cpu
    gpu = original.gpu + to_add.gpu
    extra_cpu = original.extra_cpu + to_add.extra_cpu
    extra_gpu = original.extra_gpu + to_add.extra_gpu
    all_resources = set(original.custom_resources).union(
        set(to_add.custom_resources)
    )
    new_custom_res = {
        k: original.custom_resources.get(k, 0)
        + to_add.custom_resources.get(k, 0)
        for k in all_resources
    }
    extra_custom_res = {
        k: original.extra_custom_resources.get(k, 0)
        + to_add.extra_custom_resources.get(k, 0)
        for k in all_resources
    }
    return Resources(
        cpu, gpu, extra_cpu, extra_gpu, new_custom_res, extra_custom_res
    )


def linear_scaling(total_atoms):
    """Dummy scaling"""
    return lambda x: x, total_atoms


def determine_scaling(trainable, config, total_atoms, metric):
    """Automatically detects scaling"""
    max_log = int(np.ceil(np.log2(total_atoms)))
    import ray

    result_map = {}
    for atom_count in [min(2 ** i, total_atoms) for i in range(max_log + 1)]:
        resources = trainable.to_resources(atom_count)
        trainable_cls = ray.remote(trainable)
        handle = trainable_cls.remote(config, resources=resources)
        for _ in range(3):
            result = handle.train.remote()
        handle.stop.remote()
        print(result, atom_count)
        result_map[atom_count] = result

    xy = {}
    for atom_count, result_handle in result_map.items():
        result = ray.get(result_handle)
        xy[atom_count] = result[metric]

    normalizer = xy[1]
    for i in xy:
        xy[i] /= normalizer

    def scaling_function(query):
        return np.interp(query, list(xy.keys()), list(xy.values()))

    max_scale = max(np.r_[1 : total_atoms + 1], key=scaling_function)
    logger.info("Detected max atoms per trial as {}".format(max_scale))
    return scaling_function, max_scale


class DynamicAllocator:
    POLICIES = ["TOP_JOB", "RANDOM", "UNIFORM", "SOFTMAX", "NONE"]

    def __init__(
        self,
        total,
        policy="UNIFORM",
        allocation_grid=None,
        recharge_period=2,  # This is set because learning rate takes a while to adjust
        metric="mean_accuracy",
        metric_op=1,
    ):
        """Dynamically allocates stuff."""
        self._resource_allocations = {}
        self.reallocation_timer = {}
        self._max_atoms_per_trial = total
        self._total_atoms = total
        self.committed = 0
        self.recharge_period = recharge_period
        self._policy = policy
        self._initialized = False
        self._metric = metric
        self._metric_op = metric_op  # Assume max == 1
        self._allocation_grid = allocation_grid

    def populate_if_needed(self, trials):
        for trial in trials:
            if (
                trial not in self._resource_allocations
                and trial.status == "RUNNING"
            ):
                self._resource_allocations[trial] = self.get_current_atoms(
                    trial
                )
                logger.warning(f"Adding {trial} to track.")
                self.reallocation_timer[trial] = self.recharge_period
        for trial in list(self._resource_allocations):
            if trial not in trials:
                self._resource_allocations.pop(trial)
                self.reallocation_timer.pop(trial)
        check(
            len(self._resource_allocations) <= self._total_atoms,
            f"Cannot track more than {self.total_atoms} atoms.",
        )
        self._initialized = True
        # logger.info(f"Initialized with {len(self._resource_allocations)} tracked.")

    @property
    def total_atoms(self):
        return self._total_atoms

    def on_result(self, trial_runner, trial, decision, execute=False):
        if trial.status == "TERMINATED":
            return
        if decision in ["PAUSE", "STOP"]:
            self._resource_allocations.pop(trial)
            self.reallocation_timer.pop(trial)
            execute = False
        else:
            self.reallocation_timer[trial] -= 1

        self.reallocate(self._policy)

        if self._resource_allocations.get(trial) == 0:
            self._resource_allocations.pop(trial)
            self.reallocation_timer.pop(trial)
            return "STOP"

        if execute:
            self.try_execute_update(trial, trial_runner)
        return decision

    def try_execute_update(self, trial, trial_runner):
        check(self._resource_allocations.get(trial))
        if self.should_execute_resource_update(trial, trial_runner):
            logger.info("Committing resource update.")
            new_resources = trial._get_trainable_cls().to_resources(
                self._resource_allocations[trial]
            )
            self._commit_resource_update(
                trial, trial_runner.trial_executor, new_resources
            )

    def should_execute_resource_update(self, trial, trial_runner):
        # logger.info("Relying on allocation timer ONLY to admiss update.")
        proposed_atoms = self._resource_allocations[trial]
        current_atoms = self.get_current_atoms(trial)
        if current_atoms == proposed_atoms:
            return False
        elif current_atoms < proposed_atoms:
            diff_resources = trial._get_trainable_cls().to_resources(
                proposed_atoms - current_atoms
            )
            logger.debug(f"Diff of resources - {diff_resources}")
            if not trial_runner.has_resources(diff_resources):
                logger.info(f"No resources for {trial} DRA.")
                return False
        elif self._allocation_grid:
            if (
                proposed_atoms > self._allocation_grid
                and proposed_atoms % self._allocation_grid
            ):
                logger.info(
                    f"{proposed_atoms} does not fit into {self._allocation_grid}."
                )
                return False
        # # We can consider allocating resources more aggressively than this.
        # if new_resources != trial.resources:

        #     good = (trial_runner.has_resources(diff_resources))
        return self.reallocation_timer[trial] < 0

    def _commit_resource_update(self, trial, executor, updated_resources):
        executor.save(trial, storage="memory")
        executor.stop_trial(trial, stop_logger=False)
        logger.info("Stopped trial.")
        request = updated_resources.gpu_total()
        if request:
            new_res = ray.available_resources().get("GPU", 0)
            while new_res < request:
                time.sleep(0.01)
                new_res = ray.available_resources()["GPU"]
                logger.info(f"Free GPUs: {new_res}. Waiting for {request}.")
            logger.info(f"Obtained GPUs: {new_res}.")
        trial.status = Trial.PENDING
        trial.resources = updated_resources
        logger.info("Starting new trial.")
        logger.debug("Cluster resources: " + str(ray.cluster_resources()))
        executor.start_trial(trial)
        self.reallocation_timer[trial] = self.recharge_period
        logger.info(
            "Committed res {}".format(
                executor._committed_resources.summary_string()
            )
        )
        logger.info(
            "Avail {}".format(executor._avail_resources.summary_string())
        )

    def reallocate(self, policy):
        """
        Dynamic resource reallocation - allocate the available resources from the stopped trial.
        :param policy:  Either of 'UNIFORM', 'TOP_JOB', 'RANDOM'
        """
        # Other running trials across the experiment
        sorted_remaining_trials = sorted(
            list(self._resource_allocations),
            key=lambda t: -self._metric_op
            * t.last_result.get(self._metric, np.inf),
        )
        scores = [
            t.last_result.get(self._metric, np.inf)
            for t in sorted_remaining_trials
        ]

        if not sorted_remaining_trials:
            return

        new_allocation = {trial: 1 for trial in self._resource_allocations}
        remaining_atoms = self.total_atoms - sum(new_allocation.values())
        if remaining_atoms:
            logger.info("Yay we have leftover atoms!")

        # Redistribute resources to sorted_remaining_trials according to policy
        if policy == "RANDOM":
            # All resources to random trial
            for i in range(remaining_atoms):
                t = random.choice(sorted_remaining_trials)
                new_allocation[t] += 1
        elif policy == "TOP_JOB":
            # All leftover resources to best trial
            rem_trial = sorted_remaining_trials[0]
            new_allocation[rem_trial] += remaining_atoms
        elif policy == "UNIFORM":
            # All resources randomly split among remaining trials
            for i in range(remaining_atoms):
                for j in range(len(sorted_remaining_trials)):
                    t = sorted_remaining_trials[
                        (i + j) % len(sorted_remaining_trials)
                    ]
                    if new_allocation[t] < self._max_atoms_per_trial:
                        new_allocation[t] += 1
                        break
                if j > 1 and j == len(sorted_remaining_trials) - 1:
                    logger.warning("All trials filled up!")
                    break
        elif policy == "SOFTMAX":
            vals = np.array(scores)
            logits = np.exp((vals - vals.max()))
            normalized = logits / logits.sum()
            softmax_remaining = ((normalized) * remaining_atoms).astype(int)
            while sum(softmax_remaining) < remaining_atoms:
                softmax_remaining[softmax_remaining.argmax()] += 1

            for trial, alloc in zip(sorted_remaining_trials, softmax_remaining):
                new_allocation[trial] += alloc
            logger.info(f"Softmax remaining allocation: {softmax_remaining}")
        elif policy == "NONE":
            pass

        else:
            raise NotImplementedError()
        self.validate_allocation(new_allocation)
        self._resource_allocations = new_allocation

    def set_policy(self, policy):
        if policy not in DynamicAllocator.POLICIES:
            raise ValueError
        else:
            self._policy = policy

    def status_string(self):
        out = ["Proposed allocation:"]
        out += [
            "{} - {}".format(k, v)
            for k, v in self._resource_allocations.items()
        ]
        out += [
            "Allocation total: {}".format(
                sum(self._resource_allocations.values())
            )
        ]
        return "\n".join(out)

    def profile(self, trainable, trainable_cfg, throughput_attr):
        if trainable:
            logger.info("Trainable detected. Profiling scaling...")
            scaling_fn, optimal_atoms = determine_scaling(
                trainable, trainable_cfg, self.total_atoms, throughput_attr
            )
        else:
            logger.info("Trainble not provided. Assuming scaling is linear.")
            scaling_fn, optimal_atoms = linear_scaling(self.total_atoms)
        self._max_atoms_per_trial = optimal_atoms
        return scaling_fn, optimal_atoms

    def get_current_atoms(self, trial):
        return trial._get_trainable_cls().to_atoms(trial.resources)

    def get_proposed_atoms(self, trial):
        return self._resource_allocations.get(trial)

    def validate_allocation(self, allocation_dict):
        check(
            len(allocation_dict) <= self.total_atoms,
            f"Cannot track more than {self.total_atoms} atoms.",
        )
        check(
            sum(allocation_dict.values()) <= self.total_atoms,
            f"Cannot allocate more than {self.total_atoms} atoms.",
        )


if __name__ == "__main__":

    class MockTrial:
        pass

    allocator = DynamicAllocator(20, 9, policy="UNIFORM", recharge_period=1)
    allocator.populate_if_needed(list(range(12)))
    # allocator.reallocate()
    print(allocator.status_string())
    for i in range(12):
        allocator.on_result(None, i, "CONTINUE")
    allocator.on_result(None, 0, "STOP")
    allocator.on_result(None, 1, "CONTINUE")
    allocator.on_result(None, 1, "CONTINUE")
    allocator.on_result(None, 1, "CONTINUE")
    allocator.on_result(None, 1, "CONTINUE")
    allocator.on_result(None, 1, "CONTINUE")

    allocator.on_result(None, 6, "CONTINUE")
    allocator.on_result(None, 6, "CONTINUE")
    allocator.on_result(None, 1, "STOP")
    allocator.on_result(None, 2, "STOP")
    allocator.on_result(None, 3, "STOP")
    print(allocator.status_string())

    allocator.on_result(None, 5, "CONTINUE")
    print(allocator.status_string())

    for i in range(4, 8):
        allocator.on_result(None, i, "STOP")
    allocator.on_result(None, 11, "CONTINUE")
    print(allocator.status_string())
