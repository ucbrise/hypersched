import logging

from .vgg import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *

MODEL_DICT = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "shufflenet": ShuffleNetG2,
    "lenet": LeNet,
}

SCALING_PROFILES = {
    "linear_scaling": {1: 1.0, 2: 2.0, 4: 4.0, 8: 8.0},
    "resnet18": {
        # 128, 128 * 2, 128 * 4, 128 * 8
        # AWS p3.16xlarge, V100, 1 machine
        1: 1.0,
        2: 1.853960396039604,
        4: 3.4262376237623764,
        8: 5.398019801980198,
    },
    "resnet50": {
        # 128, 128 * 2, 128 * 4, 128 * 8
        # AWS p3.16xlarge, V100, 1 machine
        1: 1.0,
        2: 1.8698727015558698,
        4: 3.4214992927864216,
        8: 5.598302687411598,
    },
    "resnet101": {
        # 128, 128 * 2, 128 * 4, 128 * 8
        # AWS p3.16xlarge, V100, 1 machine
        1: 1.0,
        2: 1.8923444976076556,
        4: 3.6315789473684212,
        8: 6.672248803827751,
    },
    "resnet152": {
        # 128, 128 * 2, 128 * 4, 128 * 8
        # AWS p3.16xlarge, V100, 1 machine
        1: 1.0,
        2: 1.796204620462046,
        4: 3.448019801980198,
        8: 6.541254125412541,
    },
    "shufflenet": {
        # 128, 128 * 2, 128 * 4, 128 * 8
        # AWS p3.16xlarge, V100, 1 machine
        1: 1.0,
        2: 1.9200988467874793,
        4: 3.6054365733113674,
        8: 6.0840197693574956,
    },
}

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_scaling_for_model_string(model_string):
    if model_string not in SCALING_PROFILES:
        logger.warning(
            "%s not found in profiles - default to resnet50", model_string,
        )
        model_string = "resnet50"
    return SCALING_PROFILES[model_string].copy()
