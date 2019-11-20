from ray import tune
from .cifar_models import MODEL_DICT
from .pytorch_helpers import get_cifar_dataset, imagenet_dataset


def get_model_creator(model_string, data_string, wrap=False):
    if data_string == "cifar":
        fn = MODEL_DICT[model_string]
        if wrap:
            fn = tune.function(fn)
        return fn
    elif data_string == "imagenet":
        import torchvision.models as models

        return models.__dict__[model_string]
    else:
        raise NotImplementedError


def get_data_creator(data_string, wrap=False):
    if data_string == "cifar":
        fn = get_cifar_dataset
        if wrap:
            fn = tune.function(fn)
        return fn
    elif data_string == "imagenet":
        fn = imagenet_dataset
        if wrap:
            fn = tune.function(fn)
        return fn
    else:
        raise NotImplementedError
