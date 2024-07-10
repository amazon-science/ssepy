import unittest
import numpy as np
from sklearn.cluster import KMeans
from cascade.model_performance_evaluator import ModelPerformanceEvaluator
from cascade.oracle_estimator import OracleEstimator
from tqdm import tqdm

np.random.seed(0)

class TestModelPerformanceEvaluator(unittest.TestCase):
    
    def setUp(self):
        self.total_samples, self.budget = 100000, 100
        self.proxy_performance = np.random.normal(0, 10, self.total_samples)
        self.performance = self.proxy_performance + np.random.normal(0, 1, self.total_samples)
        self.evaluator = ModelPerformanceEvaluator(proxy_performance=self.proxy_performance, budget=self.budget)
        self.target = np.mean(self.performance)
        self.oracle = OracleEstimator(performance=self.performance, proxy_performance=self.proxy_performance, total_samples=self.total_samples, budget=self.budget)
        
    def test_srs(self):
        estimates_ht, variances_ht = [], []
        estimates_df, variances_df = [], []
        for _ in tqdm(range(100)):
            indices = self.evaluator.sample_data(sampling_method="srs")
            ht, var_ht = self.evaluator.compute_estimate(self.performance[indices], estimator="ht")
            df, var_df = self.evaluator.compute_estimate(self.performance[indices], estimator="df")
            estimates_ht.append(ht)
            variances_ht.append(var_ht)
            estimates_df.append(df)
            variances_df.append(var_df)
        
        oracle_variance_ht = self.oracle.get_srs_variance(outcomes=self.oracle.performance)
        self.assertTrue(np.mean(np.abs(self.target - estimates_ht) / np.sqrt(oracle_variance_ht) < 2) >= 0.9)
        
        oracle_variance_df = self.oracle.get_srs_variance(outcomes=self.oracle.performance - self.oracle.proxy_performance)
        self.assertTrue(np.mean(np.abs(self.target - estimates_df) / np.sqrt(oracle_variance_df) < 2) >= 0.9)
        
    def test_ssrs_proportional(self):
        estimates_ht, variances_ht = [], []
        estimates_df, variances_df = [], []
        
        self.evaluator.stratify_data(KMeans(n_clusters=5, random_state=0, n_init="auto"), self.proxy_performance)
        self.evaluator.allocate_budget(allocation_type="proportional")
        
        self.oracle.evaluator.strata_labels = self.evaluator.strata_labels
        self.oracle.evaluator.allocate_budget(allocation_type="proportional")
        
        for _ in tqdm(range(100)):
            indices = self.evaluator.sample_data(sampling_method="ssrs")
            ht, var_ht = self.evaluator.compute_estimate(self.performance[indices], estimator="ht")
            df, var_df = self.evaluator.compute_estimate(self.performance[indices], estimator="df")
            estimates_ht.append(ht)
            variances_ht.append(var_ht)
            estimates_df.append(df)
            variances_df.append(var_df)
        
        oracle_variance_ht = self.oracle.get_ssrs_proportional_variance(outcomes=self.oracle.performance)
        self.assertTrue(np.mean(np.abs(self.target - estimates_ht) / np.sqrt(oracle_variance_ht) < 2) >= 0.9)
        
        oracle_variance_df = self.oracle.get_ssrs_proportional_variance(outcomes=self.oracle.performance - self.oracle.proxy_performance)
        self.assertTrue(np.mean(np.abs(self.target - estimates_df) / np.sqrt(oracle_variance_df) < 2) >= 0.9)
        
    def test_ssrs_neyman(self):
        estimates_ht, variances_ht = [], []
        estimates_df, variances_df = [], []
        
        self.evaluator.stratify_data(KMeans(n_clusters=5, random_state=0, n_init="auto"), self.proxy_performance)
        unique_strata, counts = np.unique(self.evaluator.strata_labels, return_counts=True)
        variances_by_strata = np.array([np.var(self.performance[self.evaluator.strata_labels == s]) for s in unique_strata])
        self.evaluator.allocate_budget(allocation_type="neyman", variances_by_strata=variances_by_strata)
        
        self.oracle.evaluator.strata_labels = self.evaluator.strata_labels
        self.oracle.evaluator.allocate_budget(allocation_type="neyman", variances_by_strata=variances_by_strata)
        
        for _ in tqdm(range(100)):
            indices = self.evaluator.sample_data(sampling_method="ssrs")
            ht, var_ht = self.evaluator.compute_estimate(self.performance[indices], estimator="ht")
            df, var_df = self.evaluator.compute_estimate(self.performance[indices], estimator="df")
            estimates_ht.append(ht)
            variances_ht.append(var_ht)
            estimates_df.append(df)
            variances_df.append(var_df)
        
        oracle_variance_ht = self.oracle.get_ssrs_neyman_variance(outcomes=self.oracle.performance)
        oracle_variance_df = self.oracle.get_ssrs_neyman_variance(outcomes=self.oracle.performance - self.oracle.proxy_performance)
        
        self.assertTrue(np.mean(np.abs(self.target - estimates_ht) / np.sqrt(oracle_variance_ht) < 2) >= 0.9)
        self.assertTrue(np.mean(np.abs(self.target - estimates_df) / np.sqrt(oracle_variance_df) < 2) >= 0.9)
        
if __name__ == "__main__":
    unittest.main()

