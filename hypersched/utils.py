import numpy as np
import os
from datetime import datetime


def check(condition, msg=""):
    if not condition:
        if os.environ.get("TUNE_SESSION"):
            raise Exception("Erroring out because Tune: {}".format(msg))
        import ipdb

        ipdb.set_trace()


def trunc(x, dec=5):
    return np.trunc(x * (10 ** dec)) / (10 ** dec)


def timestring():
    return datetime.today().strftime("%m-%d_%H-%M-%S")
