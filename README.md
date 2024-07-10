# `ssepy`: A Library for Efficient Model Evaluation through <ins>S</ins>tratification, <ins>S</ins>ampling, and <ins>E</ins>stimation in <ins>Py</ins>thon

<p align="center">
    <a href="https://arxiv.org/pdf/2406.07320"><img src="https://img.shields.io/badge/paper-arXiv-red" alt="Paper"></a>
            <a style="text-decoration:none !important;" href="https://pypi.org/project/ppi-python/" alt="package management"> <img src="https://img.shields.io/badge/pip-package-blue" /></a>
        <img src="https://img.shields.io/github/license/awslabs/cis-matching-tasks" alt="Apache-2.0">
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

In order to intall the package, download the repo, cd into it, and run

```python
pip install .
```

You may want to initialize a conda environment before running this operation.

Test your setup using this example, which demonstrates data stratification,
budget allocation for annotation via proportional allocation, sampling via
stratified simple random sampling, and estimation using the Horvitz-Thompson
estimator:

```python
import numpy as np
from sklearn.cluster import KMeans
from ssepy import ModelPerformanceEvaluator

np.random.seed(0)
# Generate data
total_samples = 100000
true_performance = np.random.normal(0, 1, total_samples) # Ground truth

# Unobserved target
print(np.mean(true_performance))

annotation_budget = 100 # Annotation budget
# 1. Proxy for ground truth
proxy_performance = true_performance + np.random.normal(0, 0.1, total_samples)
evaluator = ModelPerformanceEvaluator(proxy_performance=proxy_performance, budget=annotation_budget) # Initialize evaluator
# 2. Stratify on proxy_performance
evaluator.stratify_data(clustering_algorithm=KMeans(n_clusters=5, random_state=0, n_init="auto"), features=proxy_performance) # 5 strata
# 3. Allocate budget with proportional allocation and sample
evaluator.allocate_budget(allocation_type="proportional")
sample_indices = evaluator.sample_data(sampling_method="ssrs")
# 4. Annotate
sampled_performance = true_performance[sample_indices]
# 5. Estimate target and variance of estimate
estimate, variance_estimate = evaluator.compute_estimate(sampled_performance, estimator="ht")
print(estimate, variance_estimate)
```

For the difference estimator under simple random sampling, run

```python
evaluator = ModelPerformanceEvaluator(proxy_performance=proxy_performance, budget=annotation_budget) # initialize sampler
sample_indices = evaluator.sample_data(sampling_method="srs") # 2. sample
sampled_performance = true_performance[sample_indices] # 4. annotate
estimate, variance_estimate = evaluator.compute_estimate(sampled_performance, estimator="df") # 5. estimate
print(estimate, variance_estimate)
```

The difference estimator is also implemented in the [ppi_py
package](https://github.com/aangelopoulos/ppi_py). They implement the
prediction-powered estimator that corresponds to the difference estimator in
case of mean estimation. Their package has more functionalities than what we
offer, so check it out if you're interested in using this estimator.

# Examples

The repo comes with a series of examples contained in the `examples` folder:

- `sampling-and-estimation.ipynb` for an example on how to stratify, sample, and estimate
- `oracle-estimation.ipynb` on the computation of the (oracle) efficiency of the
  estimators under various survey designs, assuming we had access to all ground
  truth variables. This file contains the core part of the code underlying the
  results in the paper

# Features

The supported sample designs are: (SRS) simple random sampling without
replacement, (SSRS) stratified simple random sampling without replacement with
proportional and optimal/Neyman allocation, (Poisson) sampling. All sampling
methods have associated (HT) Horvitz-Thompson and (DF) difference estimators.

# Bugs and contribute

Feel free to reach out if you find any bugs or you would like other features to
be implemented in the package.
