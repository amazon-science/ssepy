import numpy as np
from .model_performance_evaluator import ModelPerformanceEvaluator


class OracleEstimator:
    # Evaluates variance under different sampling and estimation strategies
    def __init__(self, Yl, Yhl, total_samples, budget):
        # Keep references to actual (Yl) and predicted (Yhl) performance values
        self.total_samples = total_samples
        self.budget = budget
        self.Yl = Yl
        self.Yhl = Yhl
        # ModelPerformanceEvaluator uses Yhl for sampling
        self.evaluator = ModelPerformanceEvaluator(Yh=self.Yhl, budget=self.budget)

    def get_ssrs_neyman_variance(self, outcomes):
        # Stratified sampling variance using Neyman allocation
        total_stratum_sizes = self.evaluator.total_stratum_sizes
        samples_per_stratum = self.evaluator.samples_per_stratum
        unique_strata, _ = np.unique(self.evaluator.strata_labels, return_counts=True)

        variance = 0
        # Accumulate variance per stratum
        for stratum, total_stratum_size, sample_size in zip(
            unique_strata, total_stratum_sizes, samples_per_stratum
        ):
            outcomes_stratum = outcomes[self.evaluator.strata_labels == stratum]
            if len(outcomes_stratum) > 1:
                # Uses finite population correction
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
        # Stratified sampling variance using proportional allocation
        within_strata_variance, _ = self._get_within_between_strata_variance(
            self.evaluator.strata_labels, outcomes
        )
        # Standard formula for proportional allocation
        return (1 / self.budget - 1 / self.total_samples) * within_strata_variance

    def _get_within_between_strata_variance(self, strata_labels, outcomes):
        # Compute separate within-strata and between-strata components
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
        # Compare each stratum mean to overall mean
        weighted_means = (
            np.array([(mean - outcomes.mean()) ** 2 for mean in stratum_means]) * counts
        )

        within_strata_variance = weighted_variances.sum() / counts.sum()
        between_strata_variance = weighted_means.sum() / counts.sum()
        return within_strata_variance, between_strata_variance

    def get_srs_variance(self, outcomes):
        # Simple random sampling variance
        return np.var(outcomes) / self.budget * (1 - self.budget / self.total_samples)


if __name__ == "__main__":
    # Example usage
    total_samples, budget = 100000, 100
    Yhl = 1 / (1 + np.exp(-np.random.normal(0, 2, total_samples)))
    Yl = np.array(Yhl < np.random.uniform(0, 1, total_samples))

    estimator = OracleEstimator(
        Yl=Yl,
        Yhl=Yhl,
        total_samples=total_samples,
        budget=budget,
    )

    print(estimator.get_srs_variance(outcomes=estimator.Yl))
    print(estimator.get_srs_variance(outcomes=estimator.Yl - estimator.Yhl))

    from sklearn.cluster import KMeans

    # Stratify data and allocate budget
    estimator.evaluator.stratify_data(
        KMeans(n_clusters=5, random_state=0, n_init="auto"), estimator.Yhl
    )
    estimator.evaluator.allocate_budget(allocation_type="proportional")
    print(estimator.get_ssrs_proportional_variance(outcomes=estimator.Yl))

    variances = [
        np.mean(estimator.Yhl[estimator.evaluator.strata_labels == s])
        * (1 - np.mean(estimator.Yhl[estimator.evaluator.strata_labels == s]))
        for s in np.unique(estimator.evaluator.strata_labels)
    ]
    estimator.evaluator.allocate_budget(allocation_type="neyman", variances=variances)
    print(estimator.get_ssrs_neyman_variance(outcomes=estimator.Yl))
