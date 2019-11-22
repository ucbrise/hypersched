def get_trainable_cls(trainable_id, search=False):
    if trainable_id == "pytorch":
        from hypersched.tune.trainables.pytorch.pytorch_trainable import (
            PytorchSGD,
            DEFAULT_CONFIG,
            DEFAULT_HSPACE,
        )

        config = DEFAULT_CONFIG.copy()
        if search:
            config.update(DEFAULT_HSPACE)
        return PytorchSGD, config
    elif trainable_id == "mnist":
        from hypersched.tune.trainables.mnist.keras_mnist_trainable import (
            MNISTTrainable,
            DEFAULT_CONFIG,
        )

        return MNISTTrainable, DEFAULT_CONFIG
    elif trainable_id == "babi":
        from hypersched.tune.trainables.babi import (
            BABITrainable,
            DEFAULT_CONFIG,
        )

        return BABITrainable, DEFAULT_CONFIG
    elif trainable_id == "optimus":
        from hypersched.tune.trainables.toy_trainable import (
            OptimusTrainable,
            DEFAULT_HSPACE,
            DEFAULT_CONFIG,
        )

        config = DEFAULT_CONFIG.copy()
        if search:
            config.update(DEFAULT_HSPACE)
        return OptimusTrainable, config
    else:
        raise NotImplementedError


def get_multijob_config(trainable_id):
    if trainable_id == "pytorch":
        from hypersched.tune.trainables.pytorch.pytorch_trainable import (
            DEFAULT_MULTIJOB_CONFIG,
        )

        config = DEFAULT_MULTIJOB_CONFIG.copy()
        return config
    elif trainable_id == "optimus":
        from hypersched.tune.trainables.toy_trainable import (
            DEFAULT_MULTIJOB_CONFIG,
        )

        config = DEFAULT_MULTIJOB_CONFIG.copy()
        return config
    elif trainable_id == "babi":
        from hypersched.tune.trainables.babi import DEFAULT_MULTIJOB_CONFIG

        config = DEFAULT_MULTIJOB_CONFIG.copy()
        return config
    else:
        raise NotImplementedError


def get_scaling(
    trainable_id, model_string=None, data_string=None, scaling="LINEAR",
):
    if trainable_id == "pytorch":
        if data_string == "cifar":
            from hypersched.tune.trainables.pytorch.cifar_models import (
                get_scaling_for_model_string,
            )

            config = get_scaling_for_model_string(model_string).copy()
        elif data_string == "imagenet":
            config = {
                # resnet50
                # 128, 128 * 2, 128 * 4, 128 * 8
                # V100, 1 machine
                1: 1,
                2: 1.77,
                4: 2.93,
                8: 4.29,
            }
        return config
    elif trainable_id == "optimus":
        from hypersched.tune.hypersched import SCALING_MAP

        scaling_dict = SCALING_MAP[scaling].copy()
        return scaling_dict
    elif trainable_id == "babi":
        from hypersched.tune.trainables.babi import DEFAULT_SCALING

        config = DEFAULT_SCALING.copy()
        return config
    else:
        raise NotImplementedError
