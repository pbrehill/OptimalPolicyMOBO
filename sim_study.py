import pandas as pd
from fit_functions import fit_frontier
from time import time
import random
from ax.plot.pareto_frontier import scatter_plot_with_hypervolume_trace_plotly

random.seed(1993)

# Set up

NUM_TRIALS = 100

df = pd.read_csv('kenya_dr.csv')
X = pd.read_csv('kenya_covars.csv')
X_lim = pd.read_csv('kenya_limited_covars.csv')
# r_dataframeX = pandas2ri.py2rpy(df[['hhh_gender', 'hhh_age', 'hhh_literacy',
#        'age', 'female', 'monthly_spending']])
# r_dataframeG1 = pandas2ri.py2rpy(df[['any_drops2-1', 'any_drops3-1', 'any_drops4-1']] * 0.1446128)
G1 = df.filter(regex='yes_treated$', axis=1)
G2 = df.filter(regex='no_treated$', axis=1)

# Test number of bootstraps
bs_times = []
for i in [2, 5, 10, 20, 40, 80, 160, 320]:
    start_time = time()
    df_results, time_taken, _, hvs = fit_frontier(X_lim, G1, G2, num_trials=NUM_TRIALS, search_depth=1, depth=3, bs_replicates=i)
    end_time = time()
    hvs.to_csv(f"sim_tests/bootstraps/test_bootstraps{i}.csv")
    bs_times.append(start_time - end_time)
pd.DataFrame(bs_times).to_csv("time_bs.csv")

# Test number of trials (marginal time increase, decrease in HVI)
df_results, time_taken, iter_times, hvs1 = fit_frontier(X_lim, G1, G2, num_trials=400, search_depth=1, depth=3, bs_replicates=10)
df_results.to_csv(f"sim_tests/diminishing_returns.csv")
pd.DataFrame(iter_times).to_csv(f"sim_tests/diminishing_times.csv")
hvs1.to_csv(f"sim_tests/hv400.csv")

# Test hybrid vs optimal
df_results, time_taken, iter_times, hvs2 = fit_frontier(X_lim, G1, G2, num_trials=100, search_depth=2, depth=3, bs_replicates=10)
df_results.to_csv(f"sim_tests/opt_returns.csv")
pd.DataFrame(iter_times).to_csv(f"sim_tests/opt_times.csv")
hvs2.to_csv(f"sim_tests/hv100.csv")

