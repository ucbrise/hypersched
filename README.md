<div align="center">
   <p align="center"> <img src="figs/hypersched-logo.png"><br></p>
</div>

# HyperSched

An experimental scheduler for accelerated hyperparameter tuning.

People: Richard Liaw, Romil Bhardwaj, Lisa Dunlap, Yitian Zou, Joseph E. Gonzalez, Ion Stoica, Alexey Tumanov

## Overview

HyperSched a dynamic application-level resource scheduler to track, identify, and preferentially allocate resources to the best performing trials to maximize accuracy by the deadline.

HyperSched is implemented as a `TrialScheduler` of Ray Tune.
Published at SOCC 2019.

## Quick Start

Install with:

```bash
pip install ray
git clone https://github.com/ucbrise/hypersched && cd hypersched
pip install -e .
```

## Advanced Usage




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
