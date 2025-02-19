
import numpy as np
import random
from sklearn.cluster import KMeans
from ssepy import ModelPerformanceEvaluator
from tqdm import tqdm 

def run_resampling_test():
    total_samples, budget = 10000, 100
    d = 5
    
    # # Generate proxy performance (Yh) and binary outcome (Y)
    # Yh = 1 / (1 + np.exp(-np.random.normal(3, 2, (total_samples, d))))
    # Y = (Yh < np.random.uniform(0, 2, (total_samples, d))).astype(float)

    num_strata = 1  # Number of strata
    strata_sizes = np.random.multinomial(total_samples, [1/num_strata] * num_strata)  # Distribute samples across strata

    # Define different means for each stratum
    strata_means = np.array([0.2, 0.5, 0.8])  # Different mean performances per stratum
    strata_vars = np.array([0.05, 0.08, 0.1])  # Variance per stratum

    Y_list = []
    Yh_list = []

    for i in range(num_strata):
        n = strata_sizes[i]
        
        # Generate true binary outcome Y using stratified means
        Y_stratum = np.random.binomial(1, strata_means[i], (n, d))

        # Generate proxy Yh as a slightly noisy version of Y
        noise = np.random.normal(0, np.sqrt(strata_vars[i]), (n, d))
        Yh_stratum = 1 / (1 + np.exp(- (Y_stratum + noise)))  # Sigmoid transformation

        Y_list.append(Y_stratum)
        Yh_list.append(Yh_stratum)

    # Combine all strata
    Y = np.vstack(Y_list)
    Yh = np.vstack(Yh_list)



    # True performance (mean and variance) computed from entire population
    true_mean = np.nanmean(Y, axis=0)

    sampling_methods = ["srs", "ssrs"]
    estimators = ["ht", "df"]
    num_iterations = 1000

    # check the first dimension
    results = {}

    for samp_method in sampling_methods:
        for est in estimators:
            est_vals = []
            var_est_vals = []
            evaluator = ModelPerformanceEvaluator(Yh, budget)
            # For SSRS, stratify the data and allocate budget
            if samp_method == "ssrs":
                evaluator.stratify_data(KMeans(n_clusters=10, random_state=42, n_init="auto"), X=Yh)
                evaluator.allocate_budget(allocation_type="proportional")
                # evaluator.allocate_budget(allocation_type="neyman")

            for i in tqdm(range(num_iterations)):
                # For Poisson, you may either stratify or not; here we do not.
                indices = evaluator.sample(sampling_method=samp_method)
                # Compute estimate using sampled indices
                est_val, var_val = evaluator.compute_estimate(Y[indices], estimator=est)
                # est_val and var_val are arrays; we take the first element for simplicity.
                est_vals.append(est_val)
                if var_val is not None:
                    var_est_vals.append(var_val)
            avg_est = np.mean(est_vals, axis = 0)
            avg_var_est = np.mean(var_est_vals, axis = 0) if var_est_vals else None
            results[(samp_method, est)] = (avg_est, avg_var_est)
            print(f"Method: {samp_method}, Estimator: {est}")
            print(f"    Mean Estimate: {avg_est[0]:.4f} (True Mean: {true_mean[0]:.4f})")
            if avg_var_est is not None:
                print(f"    Estimated Variance: {avg_var_est[0]:.4f} (True Variance: {np.var(est_vals, axis = 0)[0]:.4f})")
            else:
                print("     Estimated Variance: None")
            # print(np.mean((est_vals)))
    return results

if __name__ == "__main__":
    run_resampling_test()
