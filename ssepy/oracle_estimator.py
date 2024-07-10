import numpy as np
from .model_performance_evaluator import ModelPerformanceEvaluator


class OracleEstimator:
    """
    OracleEstimator class for evaluating variance under different sampling and estimation strategies.

    Attributes:
        performance (np.ndarray): Array of actual performance values.
        proxy_performance (np.ndarray): Array of predicted performance values used for stratification.
        total_samples (int): Total number of samples in the dataset.
        budget (int): Number of samples to annotate.
        evaluator (ModelPerformanceEvaluator): Instance of ModelPerformanceEvaluator for performing sampling and estimation.
    """

    def __init__(self, performance, proxy_performance, total_samples, budget):
        self.total_samples, self.budget = total_samples, budget
        self.performance, self.proxy_performance = performance, proxy_performance
        self.evaluator = ModelPerformanceEvaluator(
            proxy_performance=self.proxy_performance, budget=self.budget
        )

    def get_ssrs_neyman_variance(self, outcomes):
        """
        Compute the variance of the estimates using stratified sampling with Neyman allocation.

        Args:
            outcomes (np.ndarray): The performance for the HT estimator and performance - proxy_performance for the DF estimator.

        Returns:
            float: The variance of the estimator.
        """
        total_stratum_sizes, samples_per_stratum = (
            self.evaluator.total_stratum_sizes,
            self.evaluator.samples_per_stratum,
        )
        unique_strata, _ = np.unique(self.evaluator.strata_labels, return_counts=True)

        variance = 0
        for stratum, total_stratum_size, sample_size in zip(
            unique_strata, total_stratum_sizes, samples_per_stratum
        ):
            outcomes_stratum = outcomes[self.evaluator.strata_labels == stratum]
            if len(outcomes_stratum) > 1:
                stratum_variance = (
                    np.var(outcomes_stratum, ddof=1)
                    * (total_stratum_size - 1)
                    / total_stratum_size
                )
                variance += (
                    (total_stratum_size / self.total_samples) ** 2
                    * (total_stratum_size - sample_size)
                    / total_stratum_size
                    * stratum_variance
                    / sample_size
                )

        return variance

    def get_ssrs_proportional_variance(self, outcomes):
        """
        Compute the variance of the estimates using stratified sampling with proportional allocation.

        Args:
            outcomes (np.ndarray): The performance for the HT estimator and performance - proxy_performance for the DF estimator.

        Returns:
            float: The variance of the estimator.
        """
        within_strata_variance, _ = self._get_within_between_strata_variance(
            self.evaluator.strata_labels, outcomes
        )
        return (1 / self.budget - 1 / self.total_samples) * within_strata_variance

    def _get_within_between_strata_variance(self, strata_labels, outcomes):
        unique_strata, counts = np.unique(strata_labels, return_counts=True)

        stratum_means = [outcomes[strata_labels == s].mean() for s in unique_strata]
        stratum_variances = [
            (
                np.var(outcomes[strata_labels == s], ddof=1)
                if len(outcomes[strata_labels == s]) > 1
                else 0
            )
            for s in unique_strata
        ]

        weighted_variances = np.array(stratum_variances) * counts
        weighted_means = (
            np.array([(mean - outcomes.mean()) ** 2 for mean in stratum_means]) * counts
        )

        within_strata_variance = weighted_variances.sum() / counts.sum()
        between_strata_variance = weighted_means.sum() / counts.sum()

        return within_strata_variance, between_strata_variance

    def get_srs_variance(self, outcomes):
        """
        Compute the variance of the estimates using simple random sampling.

        Args:
            outcomes (np.ndarray):  The performance for the HT estimator and performance - proxy_performance for the DF estimator.

        Returns:
            float: The variance of the estimator.
        """
        return np.var(outcomes) / self.budget * (1 - self.budget / self.total_samples)


if __name__ == "__main__":
    total_samples, budget = 100000, 100
    proxy_performance = 1 / (1 + np.exp(-np.random.normal(0, 2, total_samples)))
    performance = np.array(proxy_performance < np.random.uniform(0, 1, total_samples))

    estimator = OracleEstimator(
        performance=performance,
        proxy_performance=proxy_performance,
        total_samples=total_samples,
        budget=budget,
    )
    print(estimator.get_srs_variance(outcomes=estimator.performance))
    print(
        estimator.get_srs_variance(
            outcomes=estimator.performance - estimator.proxy_performance
        )
    )

    from sklearn.cluster import KMeans

    estimator.evaluator.stratify_data(
        KMeans(n_clusters=5, random_state=0, n_init="auto"), estimator.proxy_performance
    )

    estimator.evaluator.allocate_budget(allocation_type="proportional")
    print(estimator.get_ssrs_proportional_variance(outcomes=estimator.performance))

    estimator.evaluator.allocate_budget(
        variances_by_strata=[
            np.mean(estimator.proxy_performance[estimator.evaluator.strata_labels == s])
            * (
                1
                - np.mean(
                    estimator.proxy_performance[estimator.evaluator.strata_labels == s]
                )
            )
            for s in np.unique(estimator.evaluator.strata_labels)
        ],
        allocation_type="neyman",
    )
    print(estimator.get_ssrs_neyman_variance(outcomes=estimator.performance))
