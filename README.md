<div align="center">
   <p align="center"> <img src="figs/hypersched-logo.png" height=240p weight=320px><br></p>
</div>

# HyperSched

An experimental scheduler for accelerated hyperparameter tuning.

**People**: Richard Liaw, Romil Bhardwaj, Lisa Dunlap, Yitian Zou, Joseph E. Gonzalez, Ion Stoica, Alexey Tumanov

## Overview

HyperSched a dynamic application-level resource scheduler to track, identify, and preferentially allocate resources to the best performing trials to maximize accuracy by the deadline.

HyperSched is implemented as a `TrialScheduler` of [Ray Tune](http://tune.io/).

<div align="center">
   <p align="center"> <img src="figs/scheduler.png" height=240p><br></p>
</div>

HyperSched does so by resizing Trials.

### Terminology:

**Trial**: One training run of a (randomly sampled) hyperparameter configuration

**Experiment**: A collection of trials.

## Quick Start

This code has been tested with PyTorch 1.13 and Ray 0.7.6.
Install with:

```bash
pip install ray==0.7.6
git clone https://github.com/ucbrise/hypersched && cd hypersched
pip install -e .
```

Then:

```bash

python scripts/evaluate_dynamic_asha.py \
    --num-atoms=8 \
    --num-jobs=100 \
    --seed=1 \
    --sched hyper \
    --result-file="some-test.log" \
    --max-t=200 \
    --global-deadline=1800 \
    --trainable-id pytorch \
    --model-string resnet18 \
    --data cifar
```


## Advanced Usage


#### Viewing Results
The `hypersched.tune.Summary` object will log both a text file and also a CSV for "experiment-level" statistics.

#### HyperSched Imagenet Training on AWS

1. Create an EBS volume with ImageNet (https://github.com/pytorch/examples/tree/master/imagenet)
2. Set the EBS volume for all nodes of your cluster. For example, as seen in `scripts/imagenet.yaml`;

```yaml
head_node:
    InstanceType: p3.16xlarge
    ImageId: ami-0d96d570269578cd7
    BlockDeviceMappings:
      - DeviceName: "/dev/sdm"
        Ebs:
          VolumeType: "io1"
          Iops: 10000
          DeleteOnTermination: True
          VolumeSize: 250
          SnapshotId: "snap-01838dca0cbffad5c"

```

3. Launch the cluster. If you modify the yaml, you can then launch a cluster using `ray up scripts/imagenet.yaml`. Beware, this will cost some money. If you use the YAML, cluster will then setup a Ray cluster among the nodes launched.

3. Run the following command:

```bash
python ~/sosp2019/scripts/evaluate_dynamic_asha.py \
    --redis-address="localhost:6379" \
    --num-atoms=16 \
    --num-jobs=200 \
    --seed=0 \
    --sched hyper \
    --result-file="~/MY_LOG_FILE.log" \
    --max-t=500 \
    --global-deadline=7200 \
    --trainable-id pytorch \
    --model-string resnet50 \
    --data imagenet \
```

You can use the autoscaler to launch the experiment.

```
ray exec [CLUSTER.YAML] "<your python command here>"
```

**Note**: You may see that for imagenet, HyperSched does not isolate trials effectively (2 trials running by deadline). This is because we set the following parameters:

```python
    if args.data == "imagenet":
        worker_config = {}
        worker_config.update(
            data_loader_pin=True,
            data_loader_workers=4,
            max_train_steps=100,
            max_val_steps=20,
            decay=True,
        )
        config.update(worker_config=worker_config)
```

This indicates that for the ImageNet experiment, 1 "Trainable iteration" is defined as 100 SGD updates. HyperSched depends on the ASHA adaptive allocation to terminate trials, and a particular setup of ImageNet will not trigger the ASHA termination. Feel free to push a patch for this (or raise an issue if you want me to fix it :).

## TODOs

- [ ] Move PyTorch Trainable onto `ray.experimental.sgd`

## Cite

```
@inproceedings{Liaw:2019:HDR:3357223.3362719,
 author = {Liaw, Richard and Bhardwaj, Romil and Dunlap, Lisa and Zou, Yitian and Gonzalez, Joseph E. and Stoica, Ion and Tumanov, Alexey},
 title = {HyperSched: Dynamic Resource Reallocation for Model Development on a Deadline},
 booktitle = {Proceedings of the ACM Symposium on Cloud Computing},
 series = {SoCC '19},
 year = {2019},
 isbn = {978-1-4503-6973-2},
 location = {Santa Cruz, CA, USA},
 pages = {61--73},
 numpages = {13},
 url = {http://doi.acm.org/10.1145/3357223.3362719},
 doi = {10.1145/3357223.3362719},
 acmid = {3362719},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Distributed Machine Learning, Hyperparameter Optimization, Machine Learning Scheduling},
}
```
