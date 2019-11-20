from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hypersched.tune.resource_ray_executor import ResourceExecutor
from hypersched.tune.trainables.resource_trainable import ResourceTrainable

__all__ = [
    "ResourceExecutor",
    "ResourceTrainable",
]

import ray
from unittest.mock import patch


def ray_init(
    num_cpus=None,
    num_gpus=None,
    local_mode=True,
    log_to_driver=False,
    temp_dir=None,
    redis_address=None,
):
    ray.shutdown()
    if local_mode:
        obj = patch(
            "ray.services.check_and_update_resources",
            return_value={"CPU": num_cpus or 8, "GPU": num_gpus or 0},
        )
        obj.start()
        ray.init(
            local_mode=True,
            log_to_driver=log_to_driver,
            object_store_memory=int(1e10),
        )
    elif not redis_address:
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            log_to_driver=log_to_driver,
            temp_dir=temp_dir,
            object_store_memory=int(1e10),
        )
    else:
        ray.init(redis_address=redis_address)
