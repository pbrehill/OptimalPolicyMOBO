from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import time

import torch
import pandas as pd

# Plotting imports and initialization
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
# init_notebook_plotting()
import plotly.offline as py
from tqdm import tqdm

import random
import subprocess

subprocess.run(["Rscript", "get_dr_scores.R"])

random.seed(11)

# Load our sample 2-objective problem
from botorch.test_functions.multi_objective import BraninCurrin

# Read in R functions
from rpy2.robjects.packages import importr
import rpy2.robjects
from rpy2.robjects import pandas2ri
from rpy2.rinterface import RRuntimeWarning
import warnings
warnings.filterwarnings("ignore", category=RRuntimeWarning)
pandas2ri.activate()
import rpy2.robjects as ro
# ro.r('''source('tree_utility.r')''')
#
# evaluate_tree = ro.globalenv['evaluate_tree']

NUM_TRIALS = 50

from rpy2.robjects.packages import STAP
#Read the file with the R code snippet
with open('tree_utility.R', 'r') as f:
    string = f.read()
#Parse using STAP
evaluate_tree = STAP(string, "evaluate_tree")
honest_pt = STAP(string, "honest_pt")


def new_experiment():
    ax_client = AxClient()
    ax_client.create_experiment(
        name="morocco_experiment",
        parameters=[
            {
                "name": f"y{i+1}_weight",
                "type": "range",
                "bounds": [0.0, 1.0],
            }
            for i in range(1)
        ],
        objectives={
            # `threshold` arguments are optional
            "a": ObjectiveProperties(minimize=False),
            "b": ObjectiveProperties(minimize=False)
        },
        overwrite_existing_experiment=True,
        is_test=True,
    )
    return ax_client


def evaluate(parameters, X, gamma1, gamma2, search_depth):
    evaluation = evaluate_tree.evaluate_tree(
        X, gamma1, gamma2,
        parameters.get("y1_weight"), search_depth,
    )
    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    return {"a": (evaluation[0], evaluation[2]), "b": (evaluation[1], evaluation[3])}

def evaluate_cost(parameters, X, gamma1, cost, search_depth):
    evaluation = evaluate_tree.evaluate_tree_cost(
        X, gamma1, cost *
        parameters.get("y1_weight"), search_depth,
    )
    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    return {"a": (evaluation[0], evaluation[2]), "b": (evaluation[1], evaluation[3])}


df = pd.read_csv('mopol_data.csv')
# r_dataframeX = pandas2ri.py2rpy(df[['hhh_gender', 'hhh_age', 'hhh_literacy',
#        'age', 'female', 'monthly_spending']])
r_dataframeG1 = pandas2ri.py2rpy(df[['any_drops2-1', 'any_drops3-1', 'any_drops4-1']] * 0.1446128)
r_dataframeG2 = pandas2ri.py2rpy(df[['maths2-1', 'maths3-1', 'maths4-1']] * 0.9120552)
r_dataframeX = pandas2ri.py2rpy(df[["hhh_gender", "hhh_age", "monthly_spending", "hhh_literacy", "age", "gender", "benef", "female", "est_num_kids", "f4", "school_spending"]])
r_dataframeG2 = pandas2ri.py2rpy(df[['maths2-1', 'maths3-1', 'maths4-1']] * 0.9120552)
# r_dataframe_cost = pandas2ri.py2rpy()

total_time = []

# Greedy
time_loop = []
start_greedy = time.time()
ax_client = new_experiment()
for i in tqdm(range(NUM_TRIALS)):
    start = time.time()
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, r_dataframeX, r_dataframeG1, r_dataframeG2, 1))
    end = time.time()
    time_loop.append(end - start)

objectives = ax_client.experiment.optimization_config.objective.objectives
frontier = compute_posterior_pareto_frontier(
    experiment=ax_client.experiment,
    data=ax_client.experiment.fetch_data(),
    primary_objective=objectives[0].metric,
    secondary_objective=objectives[1].metric,
    absolute_metrics=["a", "b"],
    num_points=NUM_TRIALS,
)

plt_stuff = plot_pareto_frontier(frontier, CI_level=0.95)
py.plot(plt_stuff.data, filename='simple-lineCFG.html')

y1_weights = [x['y1_weight'] for x in frontier.param_dicts]
a_mean = frontier.means['a']
b_mean = frontier.means['b']

pd.DataFrame(
    {
        'parameter': y1_weights,
        'a_mean': a_mean,
        'b_mean': b_mean,
        'a_sem': frontier.sems['a'],
        'b_sem': frontier.sems['b']
    }
).to_csv('pareto_resultsCFG.csv')

with open('greedy_time.txt', 'w') as f:
    for line in time_loop:
        f.write(f"{line}\n")
total_time.append(time.time() - start_greedy)


# Hybrid
start_hybrid = time.time()
time_loop = []
ax_client = new_experiment()
for i in tqdm(range(NUM_TRIALS)):
    start = time.time()
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, r_dataframeX, r_dataframeG1, r_dataframeG2, search_depth=2))
    end = time.time()
    time_loop.append(end - start)

objectives = ax_client.experiment.optimization_config.objective.objectives
frontier = compute_posterior_pareto_frontier(
    experiment=ax_client.experiment,
    data=ax_client.experiment.fetch_data(),
    primary_objective=objectives[0].metric,
    secondary_objective=objectives[1].metric,
    absolute_metrics=["a", "b"],
    num_points=NUM_TRIALS,
)

plt_stuff = plot_pareto_frontier(frontier, CI_level=0.95)
py.plot(plt_stuff.data, filename='simple-lineCFH.html')

y1_weights = [x['y1_weight'] for x in frontier.param_dicts]
a_mean = frontier.means['a']
b_mean = frontier.means['b']

pd.DataFrame(
    {
        'parameter': y1_weights,
        'a_mean': a_mean,
        'b_mean': b_mean,
        'a_sem': frontier.sems['a'],
        'b_sem': frontier.sems['b']
    }
).to_csv('pareto_resultsCFH.csv')

with open('hybrid_time.txt', 'w') as f:
    for line in time_loop:
        f.write(f"{line}\n")

total_time.append(time.time() - start_hybrid)

with open('total_loops.txt', 'w') as f:
    for line in total_time:
        f.write(f"{line}\n")