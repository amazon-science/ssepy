import numpy as np
import random
from sklearn.cluster import KMeans


class ModelPerformanceEvaluator:
    """
    ModelPerformanceEvaluator class for performing stratified simple random sampling, simple random sampling, and poisson sampling,
    and estimating model performance via the Horvitz-Thompson and difference estimators.

    Attributes:
        proxy_performance (np.ndarray): Array of predicted performance values used for stratification.
        budget (int): Number of samples to annotate.
    """

    def __init__(self, proxy_performance, budget):
        self.total_samples = len(proxy_performance)
        self.proxy_performance = proxy_performance
        self.budget = budget
        self.strata_labels = None

    def stratify_data(self, clustering_algorithm, features=None):
        """
        Stratify the dataset using the provided clustering algorithm.

        Args:
            clustering_algorithm (object): Clustering algorithm instance (e.g., KMeans).
            features (np.ndarray, optional): Feature array for clustering. Defaults to None, which uses proxy_performance.
        """
        if features is None:
            features = self.proxy_performance
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        clusters = clustering_algorithm.fit(features)
        self.strata_labels = clusters.labels_ + 1

    def allocate_budget(self, variances_by_strata=None, allocation_type="proportional"):
        """
        Allocate the annotation budget across strata.

        Args:
            variances_by_strata (np.ndarray, optional): Variance estimates for each stratum. Required if allocation_type is "neyman".
            allocation_type (str): Type of allocation ("proportional" or "neyman").

        Raises:
            ValueError: If the budget allocation is invalid.
        """
        unique_strata, counts = np.unique(self.strata_labels, return_counts=True)
        if allocation_type == "proportional":
            samples_per_stratum = (counts / self.total_samples * self.budget).astype(
                int
            )
        elif allocation_type == "neyman":
            strata_std = np.sqrt(variances_by_strata)
            weights = strata_std * counts
            samples_per_stratum = (weights / np.sum(weights) * self.budget).astype(int)
        if all(samples_per_stratum > 0) and all(samples_per_stratum <= counts):
            self.samples_per_stratum = samples_per_stratum
            self.total_stratum_sizes = np.unique(
                self.strata_labels, return_counts=True
            )[1]
        else:
            print(
                samples_per_stratum
                / np.unique(self.strata_labels, return_counts=True)[1]
            )
            raise ValueError("Invalid budget allocation!")

    def sample_data(self, sampling_method="srs", probabilities=None):
        """
        Sample data based on the specified sampling method.

        Args:
            sampling_method (str): Sampling method ("srs", "ssrs", or "poisson").
            probabilities (np.ndarray, optional): Sampling probabilities for Poisson sampling.

        Returns:
            list: Indices of the sampled data.

        Raises:
            ValueError: If the sampling method is unsupported or probabilities are not provided for Poisson sampling.
        """
        self.sampling_method = sampling_method
        indices = range(self.total_samples)
        if sampling_method == "srs":
            sampled_indices = random.sample(indices, self.budget)
        elif sampling_method == "ssrs":
            unique_strata = np.unique(self.strata_labels)
            sampled_indices = []
            for i, stratum in enumerate(unique_strata):
                stratum_indices = [
                    idx
                    for idx, label in zip(indices, self.strata_labels)
                    if label == stratum
                ]
                sampled_indices.extend(
                    random.sample(stratum_indices, self.samples_per_stratum[i])
                )
        elif sampling_method == "poisson":
            if probabilities is None:
                raise ValueError("Provide sampling probabilities!")
            self.probabilities = (
                probabilities * self.budget / np.sum(probabilities)
            )  # normalization step
            bernoulli_trials = np.random.binomial(
                1, p=self.probabilities, size=self.total_samples
            )
            sampled_indices = [
                i for i in range(self.total_samples) if bernoulli_trials[i] == 1
            ]
        else:
            raise ValueError("Unsupported sampling method!")
        self.sampled_indices = sampled_indices
        return sampled_indices

    def compute_estimate(self, observed_performance, estimator="ht"):
        """
        Compute the estimate of the performance and its variance.

        Args:
            observed_performance (np.ndarray): Observed performance values for the sampled data.
            estimator (str): Estimator type ("ht" for Horvitz-Thompson, "df" for Difference).

        Returns:
            tuple: Estimate of the performance and the variance of the estimate.

        Raises:
            ValueError: If the sampling method or estimator is unsupported.
        """
        observed_performance = np.array(observed_performance)
        if self.sampling_method in ["ssrs", "srs"]:
            if not isinstance(self.strata_labels, np.ndarray):
                self.strata_labels = np.zeros_like(self.proxy_performance)
                self.samples_per_stratum, self.total_stratum_sizes = np.array(
                    self.budget
                ), np.array(len(self.proxy_performance))
            sampled_strata = self.strata_labels[self.sampled_indices]
            unique_strata, stratum_counts = np.unique(
                self.strata_labels, return_counts=True
            )
            stratum_counts = {
                unique_strata[i]: stratum_counts[i] for i in range(len(unique_strata))
            }
            weights = self.total_stratum_sizes / self.total_samples
            if estimator == "ht":
                estimate = np.sum(
                    weights
                    * np.array(
                        [
                            np.mean(observed_performance[sampled_strata == s])
                            for s in unique_strata
                        ]
                    )
                )
                stratum_variances = np.array(
                    [
                        np.var(observed_performance[sampled_strata == s], ddof=1)
                        for s in unique_strata
                    ]
                )
            elif estimator == "df":
                sampled_proxy_performance = self.proxy_performance[self.sampled_indices]
                estimate = np.sum(
                    weights
                    * np.array(
                        [
                            (
                                np.sum(self.proxy_performance[self.strata_labels == s])
                                + np.sum(
                                    observed_performance[sampled_strata == s]
                                    - sampled_proxy_performance[sampled_strata == s]
                                )
                                * np.sum(self.strata_labels == s)
                                / np.sum(sampled_strata == s)
                            )
                            / np.sum(self.strata_labels == s)
                            for s in unique_strata
                        ]
                    )
                )
                stratum_variances = np.array(
                    [
                        np.var(
                            observed_performance[sampled_strata == s]
                            - sampled_proxy_performance[sampled_strata == s],
                            ddof=1,
                        )
                        for s in unique_strata
                    ]
                )
            stratum_variances = (
                (1 - self.samples_per_stratum / self.total_stratum_sizes)
                * stratum_variances
                / self.samples_per_stratum
            )
            variance_estimate = np.sum(weights**2 * stratum_variances)
        elif self.sampling_method == "poisson":
            if estimator == "ht":
                estimate = (
                    np.sum(
                        observed_performance / self.probabilities[self.sampled_indices]
                    )
                    / self.total_samples
                )
                variance_estimate = np.sum(
                    (self.probabilities[self.sampled_indices] - 1)
                    * (observed_performance**2)
                    / self.probabilities[self.sampled_indices]
                ) / (self.total_samples**2)
            elif estimator == "df":
                sampled_proxy_performance = self.proxy_performance[self.sampled_indices]
                estimate = (
                    np.sum(self.proxy_performance)
                    + np.sum(
                        (observed_performance - sampled_proxy_performance)
                        / self.probabilities[self.sampled_indices]
                    )
                ) / self.total_samples
                variance_estimate = np.sum(
                    (self.probabilities[self.sampled_indices] - 1)
                    * ((observed_performance - sampled_proxy_performance) ** 2)
                    / self.probabilities[self.sampled_indices]
                ) / (self.total_samples**2)
        else:
            raise ValueError("Unsupported sampling method!")

        return estimate, variance_estimate


if __name__ == "__main__":
    total_samples, budget = 100000, 100
    proxy_performance = 1 / (1 + np.exp(-np.random.normal(0, 2, total_samples)))
    performance = np.array(proxy_performance < np.random.uniform(0, 1, total_samples))

    evaluator = ModelPerformanceEvaluator(
        proxy_performance=proxy_performance, budget=budget
    )
    indices = evaluator.sample_data(sampling_method="srs")
    estimate_ht_srs, _ = evaluator.compute_estimate(
        performance[indices], estimator="ht"
    )
    estimate_df_srs, _ = evaluator.compute_estimate(
        performance[indices], estimator="df"
    )

    evaluator = ModelPerformanceEvaluator(
        proxy_performance=proxy_performance, budget=budget
    )
    evaluator.stratify_data(
        KMeans(n_clusters=5, random_state=0, n_init="auto"), proxy_performance
    )
    evaluator.allocate_budget(allocation_type="proportional")
    indices = evaluator.sample_data(sampling_method="ssrs")
    estimate_ht_ssrs_prop, _ = evaluator.compute_estimate(
        performance[indices], estimator="ht"
    )
    estimate_df_ssrs_prop, _ = evaluator.compute_estimate(
        performance[indices], estimator="df"
    )

    evaluator = ModelPerformanceEvaluator(
        proxy_performance=proxy_performance, budget=budget
    )
    evaluator.stratify_data(
        KMeans(n_clusters=5, random_state=0, n_init="auto"), proxy_performance
    )
    evaluator.allocate_budget(
        variances_by_strata=[
            np.mean(evaluator.proxy_performance[evaluator.strata_labels == s])
            * (1 - np.mean(evaluator.proxy_performance[evaluator.strata_labels == s]))
            for s in np.unique(evaluator.strata_labels)
        ],
        allocation_type="neyman",
    )
    indices = evaluator.sample_data(sampling_method="ssrs")
    estimate_ht_ssrs_ney, _ = evaluator.compute_estimate(
        performance[indices], estimator="ht"
    )
    estimate_df_ssrs_ney, _ = evaluator.compute_estimate(
        performance[indices], estimator="df"
    )
