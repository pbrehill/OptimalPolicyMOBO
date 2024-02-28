from fit_functions import fit_frontier
import pandas as pd

NUM_TRIALS = 200

df = pd.read_csv('kenya_dr.csv')
X = pd.read_csv('kenya_covars.csv')
X_lim = pd.read_csv('kenya_limited_covars.csv')
# r_dataframeX = pandas2ri.py2rpy(df[['hhh_gender', 'hhh_age', 'hhh_literacy',
#        'age', 'female', 'monthly_spending']])
# r_dataframeG1 = pandas2ri.py2rpy(df[['any_drops2-1', 'any_drops3-1', 'any_drops4-1']] * 0.1446128)
G1 = df.filter(regex='yes_treated$', axis=1)
G2 = df.filter(regex='no_treated$', axis=1)
X_vlim = X_lim[["dist_clinic", "LOG_patient_age1", "subfarm"]]


# Depth 1
## All vars
df_results, time_taken, frontier1, _ = fit_frontier(X, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=1, bs_replicates=80)
df_results.to_csv("vbd1.csv")

## Limited vars
df_results, time_taken, frontier2, _ = fit_frontier(X_lim, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=1, bs_replicates=80)
df_results.to_csv("vsd1.csv")

## Very limited vars
df_results, time_taken, frontier3, _ = fit_frontier(X_vlim, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=1, bs_replicates=80)
df_results.to_csv("vtd1.csv")

# Depth 2
## All vars
df_results, time_taken, frontier4, _ = fit_frontier(X, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=2, bs_replicates=80)
df_results.to_csv("vbd2.csv")

## Limited vars
df_results, time_taken, frontier5, _ = fit_frontier(X_lim, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=2, bs_replicates=80)
df_results.to_csv("vsd2.csv")

## Very limited vars
df_results, time_taken, frontier6, _ = fit_frontier(X_vlim, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=2, bs_replicates=80)
df_results.to_csv("vtd2.csv")

# Depth 3
## All vars
df_results, time_taken, frontier7, _ = fit_frontier(X, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=3, bs_replicates=80)
df_results.to_csv("vbd3.csv")

## Limited vars
df_results, time_taken, frontier8, _ = fit_frontier(X_lim, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=3, bs_replicates=80)
df_results.to_csv("vsd3.csv")

## Very limited vars
df_results, time_taken, frontier9, _ = fit_frontier(X_vlim, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=3, bs_replicates=80)
df_results.to_csv("vtd3.csv")