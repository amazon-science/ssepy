# `ssepy`: A Library for Efficient Model Evaluation through <ins>S</ins>tratification, <ins>S</ins>ampling, and <ins>E</ins>stimation in <ins>Py</ins>thon

<p align="center">
    <a href="https://arxiv.org/pdf/2406.07320"><img src="https://img.shields.io/badge/paper-arXiv-red" alt="Paper"></a>
            <a style="text-decoration:none !important;" href="https://pypi.org/project/ssepy/" alt="package management"> <img src="https://img.shields.io/badge/pip-package-blue" /></a>
        <img src="https://img.shields.io/github/license/amazon-science/ssepy" alt="Apache-2.0">
</p>

**Given an unlabeled dataset and model predictions, how can we select which
instances to annotate in one go to maximize the precision of our estimates of
model performance on the entire dataset?**

The ssepy package helps you do that! The implementation of
the ssepy package revolves around the following sequential framework:

1. **Predict**: Predict the expected model performance for each
   example.
2. **Stratify**: Divide the dataset into strata using the base predictions.
3. **Sample**: Sample a data subset using the chosen sampling method.
4. **Annotate**: Acquire annotations for the sampled subset.
5. **Estimate**: Estimate model performance.

See our paper [here](https://arxiv.org/pdf/2406.07320) for a technical overview of the framework.

# Getting started

In order to intall the package, run 
```python
pip install ssepy
```

Alternatively, clone the repo, `cd` into it, and run

```python
pip install .
```

You may want to initialize a conda environment before running this operation.

Test your setup using this example, which demonstrates data stratification,
n allocation for annotation via proportional allocation, sampling via
stratified simple random sampling, and estimation using the Horvitz-Thompson
estimator:

```python
import numpy as np
from sklearn.cluster import KMeans
from ssepy import ModelPerformanceEvaluator

np.random.seed(0)
# Generate data
N = 100000
Y = np.random.normal(0, 1, N) # Ground truth

# Unobserved target
print(np.mean(Y))

n = 100 # Annotation n
# 1. Proxy for ground truth
Yh = Y + np.random.normal(0, 0.1, N)
evaluator = ModelPerformanceEvaluator(Yh = Yh, budget = n) # Initialize evaluator
# 2. Stratify on Yh
evaluator.stratify_data(clustering_algo=KMeans(n_clusters=5, random_state=0, n_init="auto"), X=Yh) # 5 strata
# 3. Allocate n with proportional allocation and sample
evaluator.allocate_budget(allocation_type="proportional")
sampled_idx = evaluator.sample()
# 4. Annotate
Yl = Y[sampled_idx]
# 5. Estimate target and variance of estimate
estimate, variance_estimate = evaluator.compute_estimate(Yl, estimator="ht")
print(estimate, variance_estimate)
```

For the difference estimator under simple random sampling, run

```python
evaluator = ModelPerformanceEvaluator(Yh=Yh, budget=n) # initialize sampler
sampled_idx = evaluator.sample(sampling_method="srs") # 3. sample
Yl = Y[sampled_idx] # 4. annotate
estimate, variance_estimate = evaluator.compute_estimate(Yl, estimator="df") # 5. estimate
print(estimate, variance_estimate)
```

See also some examples in the associated folder. 

# Features

The supported sample designs are: (SRS) simple random sampling without
replacement, (SSRS) stratified simple random sampling without replacement with
proportional and optimal/Neyman allocation, (Poisson) sampling. All sampling
methods have associated (HT) Horvitz-Thompson and (DF) difference estimators.

# Bugs and contribute

Feel free to reach out if you find any bugs or you would like other features to
be implemented in the package.
