import numpy as np
import random
from sklearn.cluster import KMeans
from collections import Counter


def ensure_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def compute_lambda_star(Yhu, Yl, Yhl):
    # Compute a separate lambda for each feature by comparing:
    #  - labeled data (Yl) with its predictions (Yhl)
    #  - unlabeled data's predictions (Yhu)
    n_features = Yl.shape[1]
    lambda_vec = np.zeros(n_features)
    for j in range(n_features):
        Yl_j = Yl[:, j]
        Yhl_j = Yhl[:, j]
        Yhu_j = Yhu[:, j]

        # Identify valid (non-NaN) entries in labeled and unlabeled arrays
        mask_labeled = ~np.isnan(Yl_j) & ~np.isnan(Yhl_j)
        mask_unlabeled = ~np.isnan(Yhu_j)

        Yl_j_valid = Yl_j[mask_labeled]
        Yhl_j_valid = Yhl_j[mask_labeled]
        Yhu_j_valid = Yhu_j[mask_unlabeled]

        # If there's insufficient valid data to compute covariance or variance, use lambda=0
        if len(Yl_j_valid) == 0 or len(Yhu_j_valid) == 0:
            lambda_j = 0.0
        else:
            # Covariance between Yl_j and Yhl_j for labeled data
            cov = np.cov(Yl_j_valid, Yhl_j_valid)[0, 1]
            # Ratio of labeled to unlabeled valid counts
            r = len(Yl_j_valid) / len(Yhu_j_valid)
            # Combine Yhu_j and Yhl_j for variance calculation
            combined = np.concatenate([Yhu_j_valid, Yhl_j_valid])
            den = np.var(combined, ddof=1) if len(combined) > 1 else 0.0
            # Lambda is the scaled ratio of covariance to variance
            lambda_j = cov / ((1 + r) * den) if den != 0 else 0.0
            lambda_j = np.clip(lambda_j, 0, 1)
        lambda_vec[j] = lambda_j
    return lambda_vec


class StratifiedSampler:
    def __init__(self, Yh, budget):
        self.Yh = ensure_2d(Yh)
        self.budget = budget
        self.total_samples = len(Yh)
        self.strata_labels = None
        self.samples_per_stratum = None
        self.total_stratum_sizes = None
        self.sampled_indices = None
        self.sampling_method = None
        self.probabilities = None

    def stratify(self, clustering_algo, X=None, X_train=None):
        # Cluster the data into strata using a specified algorithm. Optionally, a separate X_train can be used for fitting.
        if X is None:
            X = self.Yh
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        data = X_train if X_train is not None else X
        model = clustering_algo.fit(data)
        labels = model.predict(X)
        self.strata_labels = labels + 1

        # Merge clusters with fewer than 10 samples into the nearest large cluster
        counts = Counter(labels)
        small = [lab for lab, cnt in counts.items() if cnt < 10]
        if small:
            if hasattr(model, "cluster_centers_"):
                centers = model.cluster_centers_
            else:
                centers = np.array([X[labels == lab].mean(axis=0) for lab in counts])
            large = [lab for lab in counts if lab not in small]
            if not large:
                self.strata_labels = np.ones_like(labels)
                return
            for lab in small:
                dists = np.linalg.norm(centers[large] - centers[lab], axis=1)
                nearest = large[np.argmin(dists)]
                self.strata_labels[self.strata_labels == (lab + 1)] = nearest + 1
            # Reindex strata labels so they remain consecutive integers
            uniq = np.unique(self.strata_labels)
            mapping = {old: new + 1 for new, old in enumerate(uniq)}
            self.strata_labels = np.vectorize(mapping.get)(self.strata_labels)

    @staticmethod
    def _allocate_leftover(samples, strata_sizes, total_budget):
        # If we still have leftover samples after allocation, distribute them among strata that haven't reached their size limit.
        leftover = total_budget - samples.sum()
        eligible = np.where(samples < strata_sizes)[0]
        while leftover > 0 and eligible.size > 0:
            chosen = np.random.choice(eligible)
            samples[chosen] += 1
            leftover -= 1
            eligible = np.where(samples < strata_sizes)[0]
        return samples

    def allocate_budget(self, allocation_type="proportional", variances=None):
        # Decide how many samples to draw from each stratum, using either proportional or Neyman allocation.
        uniq, sizes = np.unique(self.strata_labels, return_counts=True)
        mask = ~np.isnan(self.Yh)
        strata_bool = self.strata_labels[:, None] == uniq
        # valid[i] is the average number of non-NaN entries for stratum i
        valid = np.dot(strata_bool.T.astype(int), mask.astype(int)).mean(axis=1)

        if allocation_type == "proportional":
            sp = np.round(valid / valid.sum() * self.budget)
        elif allocation_type == "neyman":
            # If variances not provided, approximate using mean*(1 - mean) in each stratum
            if variances is None:
                variances = []
                for s in uniq:
                    stratum_mask = self.strata_labels == s
                    means = np.nanmean(self.Yh[stratum_mask], axis=0)
                    v = np.clip(means * (1 - means), 0.05, np.inf)
                    variances.append(np.nanmean(v))
                variances = np.array(variances)
            stds = np.sqrt(variances)
            weights = stds * valid
            sp = np.round(weights / weights.sum() * self.budget)
        else:
            raise ValueError("Invalid allocation type")

        # Clip allocations to stratum sizes and then allocate leftovers
        sp = np.clip(sp, 0, sizes).astype(int)
        self.samples_per_stratum = self._allocate_leftover(sp, sizes, self.budget)
        self.total_stratum_sizes = sizes

    def sample(self, sampling_method="srs", probabilities=None):
        # Choose how we select samples: Simple Random (srs), Stratified Simple Random (ssrs), or Poisson.
        self.sampling_method = sampling_method
        indices = np.arange(self.total_samples)
        if sampling_method == "srs":
            samp = random.sample(list(indices), self.budget)
        elif sampling_method == "ssrs":
            samp = []
            uniq = np.unique(self.strata_labels)
            # For each stratum, pick the predetermined number of samples at random
            for i, s in enumerate(uniq):
                mask = self.strata_labels == s
                idxs = indices[mask]
                n = self.samples_per_stratum[i]
                samp.extend(random.sample(list(idxs), n))
        elif sampling_method == "poisson":
            # For Poisson sampling, each sample is included with probability p_i, adjusted so that total expected sample size = budget
            self.probabilities = probabilities * self.budget / probabilities.sum()
            trials = np.random.binomial(1, self.probabilities, size=self.total_samples)
            samp = np.where(trials == 1)[0]
        else:
            raise ValueError("Unsupported sampling method")
        self.sampled_indices = np.sort(samp)
        return self.sampled_indices


class Estimator:
    def __init__(
        self,
        Yh,
        strata_labels=None,
        sampled_indices=None,
        sampling_method="ssrs",
        samples_per_stratum=None,
        total_stratum_sizes=None,
        budget=None,
        probabilities=None,
    ):
        self.Yh = ensure_2d(Yh)
        # If no strata are given, treat all samples as belonging to one stratum
        self.strata_labels = (
            np.ones(len(Yh), int) if strata_labels is None else strata_labels
        )
        self.sampled_indices = sampled_indices
        self.sampling_method = sampling_method
        self.unique_strata, self.total_stratum_sizes = np.unique(
            self.strata_labels, return_counts=True
        )
        self.strata_mapping = {s: i for i, s in enumerate(self.unique_strata)}

        # Use the provided samples_per_stratum if available; otherwise, compute from sampled_indices
        if samples_per_stratum is None and sampled_indices is not None:
            sampled_strata = self.strata_labels[self.sampled_indices]
            self.samples_per_stratum = np.array(
                [np.sum(sampled_strata == s) for s in self.unique_strata]
            )
        else:
            self.samples_per_stratum = samples_per_stratum

        self.budget = budget
        self.total_samples = len(Yh)
        self.probabilities = probabilities
        # Map sampled index -> row position in the sampled subset
        self.sample_index_map = (
            {idx: i for i, idx in enumerate(self.sampled_indices)}
            if self.sampled_indices is not None
            else {}
        )

    def compute(self, Yl, estimator="ht", tune_power=False):
        # Estimate mean or total under either stratified or Poisson sampling.
        # 'estimator' can be 'ht' (Horvitz-Thompson) or 'df' (Difference estimator).
        Yl = ensure_2d(Yl)
        if self.sampling_method in ["ssrs", "srs"]:
            return self._compute_ssrs(Yl, estimator, tune_power)
        elif self.sampling_method == "poisson":
            return self._compute_poisson(Yl, estimator)
        raise ValueError("Unsupported method")

    def _compute_ssrs(self, Yl, estimator, tune_power):
        # Compute estimates under stratified sampling or simple random sampling.
        # If 'df' and tune_power=True, compute lambdas based on unlabeled data.
        sample_mask = np.isin(np.arange(self.total_samples), self.sampled_indices)
        non_sample_mask = ~sample_mask
        Yh_sampled = self.Yh[self.sampled_indices]

        # lambda_vec is either zero (HT) or estimated (DF)
        if estimator == "ht":
            lambda_vec = np.zeros(Yl.shape[1])
        elif estimator == "df":
            if tune_power:
                # Use compute_lambda_star to find optimal lambdas based on unlabeled Yh
                Yhu = self.Yh[non_sample_mask]
                lambda_vec = compute_lambda_star(Yhu, Yl, Yh_sampled)
            else:
                lambda_vec = np.ones(Yl.shape[1])
        else:
            raise ValueError("Invalid estimator")

        est_total = np.zeros(Yl.shape[1])
        var_total = np.zeros(Yl.shape[1])
        # Sum estimates across strata, then divide by the total sample size for the final mean
        for s in self.unique_strata:
            s_idx = self.strata_mapping[s]
            N_h = self.total_stratum_sizes[s_idx]  # Total stratum size
            n_h = self.samples_per_stratum[
                s_idx
            ]  # Number of samples drawn from this stratum
            if n_h == 0:
                continue
            w = N_h / n_h  # Weight applied to corrections for this stratum

            # Identify indices in current stratum that were sampled
            stratum_mask = self.strata_labels == s
            stratum_sample = self.sampled_indices[
                np.isin(self.sampled_indices, np.where(stratum_mask)[0])
            ]
            # Convert those sampled indices into row positions in Yl
            Yl_s = Yl[[self.sample_index_map[idx] for idx in stratum_sample]]
            Yh_s = self.Yh[stratum_sample]

            # valid_mask tracks non-NaN entries in both Yl_s and Yh_s
            valid_mask = ~np.isnan(Yl_s) & ~np.isnan(Yh_s)
            # Mean of Yh for the entire stratum, used in difference estimator
            mYh = np.nanmean(self.Yh[stratum_mask], axis=0)
            # The 'diff' term is (Yl - lambda*Yh) for all valid data
            diff = np.where(valid_mask, Yl_s - lambda_vec * Yh_s, 0)
            correction = w * np.sum(diff, axis=0)

            # For DF, we add (mean(Yh)*N_h*lambda_vec) plus the correction
            # For HT, since lambda_vec=0, only correction adds up.
            est_total += mYh * N_h * lambda_vec + correction

            # Approximate variance in the stratum
            data_var = np.where(valid_mask, Yl_s - lambda_vec * Yh_s, np.nan)
            s2 = np.nanvar(data_var, axis=0, ddof=1)
            count = np.sum(valid_mask, axis=0)
            s2 = np.where(count > 1, s2, 0)
            # (N_h^2) * (1 - n_h/N_h) * s2 / n_h is a standard finite population correction
            var_total += (N_h**2) * (1 - n_h / N_h) * s2 / n_h

        return est_total / self.total_samples, var_total / (self.total_samples**2)

    def _compute_poisson(self, Yl, estimator):
        # Compute estimates for Poisson sampling. Each sample is drawn with probability pi.
        pi = self.probabilities[self.sampled_indices]
        Yl_s = Yl[[self.sample_index_map[idx] for idx in self.sampled_indices]]

        if estimator == "ht":
            # Horvitz-Thompson: multiply each sampled observation by 1/pi
            w = 1 / pi[:, None]
            est = np.nansum(Yl_s * w, axis=0) / self.total_samples
            # Variance includes (1-pi)/pi^2 term for each observation
            var = np.nansum(((1 - pi) / (pi**2)) * (Yl_s**2), axis=0) / (
                self.total_samples**2
            )
            return est, var
        elif estimator == "df":
            # Difference estimator adjusts the mean of Yh by a correction term
            Yh_mean = np.nanmean(self.Yh, axis=0)
            w_sum = np.nansum(1 / pi)
            correction = (
                np.nansum((Yl_s - self.Yh[self.sampled_indices]) / pi[:, None], axis=0)
                / w_sum
            )
            return Yh_mean + correction, None
        raise ValueError("Invalid estimator")


class ModelPerformanceEvaluator:
    def __init__(self, Yh, budget):
        self.sampler = StratifiedSampler(Yh, budget)

    @property
    def strata_labels(self):
        return self.sampler.strata_labels

    @property
    def samples_per_stratum(self):
        return self.sampler.samples_per_stratum

    @property
    def total_stratum_sizes(self):
        return self.sampler.total_stratum_sizes

    def stratify_data(self, clustering_algo, X=None, X_train=None):
        # Run stratification using the provided clustering algorithm.
        self.sampler.stratify(clustering_algo, X, X_train)

    def allocate_budget(self, allocation_type="proportional", variances=None):
        # Ensure stratification has been performed.
        if self.sampler.strata_labels is None:
            raise ValueError("Strata labels not set. Run stratify_data() first.")
        self.sampler.allocate_budget(allocation_type, variances)

    def sample(self, sampling_method="srs", probabilities=None):
        # For ssrs, ensure that budget allocation (samples_per_stratum) is done.
        if sampling_method == "ssrs" and self.sampler.samples_per_stratum is None:
            raise ValueError(
                "Samples per stratum not set. Run allocate_budget() first."
            )
        return self.sampler.sample(sampling_method, probabilities)

    def compute_estimate(self, Yl, estimator="ht", tune_power=False):
        # Ensure that sampling has been performed.
        if self.sampler.sampled_indices is None:
            raise ValueError("Sampled indices not set. Run sample() first.")
        est_obj = Estimator(
            Yh=self.sampler.Yh,
            strata_labels=self.sampler.strata_labels,
            sampled_indices=self.sampler.sampled_indices,
            sampling_method=self.sampler.sampling_method,
            samples_per_stratum=self.sampler.samples_per_stratum,
            total_stratum_sizes=self.sampler.total_stratum_sizes,
            budget=self.sampler.budget,
            probabilities=self.sampler.probabilities,
        )
        return est_obj.compute(Yl, estimator, tune_power)


if __name__ == "__main__":
    total_samples, budget = 10000, 100
    # Generate proxy performance (Yh) and true performance (Y) as 2D arrays
    Yh = 1 / (1 + np.exp(-np.random.normal(0, 2, total_samples)))
    Yh = Yh.reshape(-1, 1)
    Y = (Yh < np.random.uniform(0, 1, total_samples)).astype(float)
    Y = Y.reshape(-1, 1)

    # Test 1: SRS sampling with HT and DF estimators
    evaluator = ModelPerformanceEvaluator(Yh=Yh, budget=budget)
    indices = evaluator.sample(sampling_method="srs")
    estimate_ht_srs, _ = evaluator.compute_estimate(Y[indices], estimator="ht")
    estimate_df_srs, _ = evaluator.compute_estimate(Y[indices], estimator="df")
    print("SRS HT:", estimate_ht_srs)
    print("SRS DF:", estimate_df_srs)

    # Test 2: SSRS sampling with proportional allocation
    evaluator = ModelPerformanceEvaluator(Yh=Yh, budget=budget)
    evaluator.stratify_data(KMeans(n_clusters=5, random_state=0, n_init="auto"), X=Yh)
    evaluator.allocate_budget(allocation_type="proportional")
    indices = evaluator.sample(sampling_method="ssrs")
    estimate_ht_ssrs_prop, _ = evaluator.compute_estimate(Y[indices], estimator="ht")
    estimate_df_ssrs_prop, _ = evaluator.compute_estimate(Y[indices], estimator="df")
    print("SSRS Proportional HT:", estimate_ht_ssrs_prop)
    print("SSRS Proportional DF:", estimate_df_ssrs_prop)

    # Test 3: SSRS sampling with Neyman allocation
    evaluator = ModelPerformanceEvaluator(Yh=Yh, budget=budget)
    evaluator.stratify_data(KMeans(n_clusters=5, random_state=0, n_init="auto"), X=Yh)
    # Compute variances for Neyman allocation.
    variances = [
        np.mean(Yh[evaluator.sampler.strata_labels == s])
        * (1 - np.mean(Yh[evaluator.sampler.strata_labels == s]))
        for s in np.unique(evaluator.sampler.strata_labels)
    ]
    evaluator.allocate_budget(variances=variances, allocation_type="neyman")
    indices = evaluator.sample(sampling_method="ssrs")
    estimate_ht_ssrs_ney, _ = evaluator.compute_estimate(Y[indices], estimator="ht")
    estimate_df_ssrs_ney, _ = evaluator.compute_estimate(Y[indices], estimator="df")
    print("SSRS Neyman HT:", estimate_ht_ssrs_ney)
    print("SSRS Neyman DF:", estimate_df_ssrs_ney)
