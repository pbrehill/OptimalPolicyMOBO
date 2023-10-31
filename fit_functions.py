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

subprocess.run(["Rscript", "kenya_dr.R"])

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
import numpy as np
import rpy2.robjects as ro
# ro.r('''source('tree_utility.r')''')
#
# evaluate_tree = ro.globalenv['evaluate_tree']

NUM_TRIALS = 400

from rpy2.robjects.packages import STAP
#Read the file with the R code snippet
with open('tree_utility.R', 'r') as f:
    string = f.read()
#Parse using STAP
evaluate_tree = STAP(string, "evaluate_tree")
honest_pt = STAP(string, "honest_pt")

import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)

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


def evaluate(parameters, X, gamma1, gamma2, search_depth, depth, bs_replicates):
    evaluation = evaluate_tree.evaluate_tree(
        X, gamma1, gamma2,
        parameters.get("y1_weight"), search_depth = search_depth, depth = depth, bs = bs_replicates
    )
    return {"a": (evaluation[0], evaluation[2]), "b": (evaluation[1], evaluation[3])}

def evaluate_cost(parameters, X, gamma1, cost, search_depth):
    evaluation = evaluate_tree.evaluate_tree_cost(
        X, gamma1, cost *
        parameters.get("y1_weight"), search_depth,
    )
    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    return {"a": (evaluation[0], evaluation[2]), "b": (evaluation[1], evaluation[3])}

def set_values(row):
    if row['female']:
        return row
    else:
        return 0

def oracle_values(G1, G2, weight):
    G1 = G1.to_numpy()
    G2 = G2.to_numpy()
    gammax = (weight * G1) + ((1- weight) * G2)
    max_val = np.argmax(gammax, axis = 1)
    gamma1_value = np.choose(max_val, G1.T).mean()
    gamma2_value = np.choose(max_val, G2.T).mean()
    return gamma1_value, gamma2_value

def blunt_values(G1, G2, weight):
    G1 = G1.to_numpy()
    G2 = G2.to_numpy()
    gammax = (weight * G1) + ((1 - weight) * G2)
    max_val = np.argmax(gammax.mean(axis = 0))
    gamma1_value = np.choose(max_val, G1.T).mean()
    gamma2_value = np.choose(max_val, G2.T).mean()
    return gamma1_value, gamma2_value


def fit_frontier(X, G1, G2, search_depth, num_trials, depth, bs_replicates, time_iter = False):
    G1_r = pandas2ri.py2rpy(G1)
    X_r = pandas2ri.py2rpy(X)
    G2_r = pandas2ri.py2rpy(G2)

    start_hybrid = time.time()
    time_loop = []
    ax_client = new_experiment()
    for i in tqdm(range(num_trials)):
        start = time.time()
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, X_r, G1_r, G2_r, search_depth=search_depth, depth=depth, bs_replicates = bs_replicates))
        end = time.time()
        time_loop.append(end - start)

    objectives = ax_client.experiment.optimization_config.objective.objectives
    frontier = compute_posterior_pareto_frontier(
        experiment=ax_client.experiment,
        data=ax_client.experiment.fetch_data(),
        primary_objective=objectives[0].metric,
        secondary_objective=objectives[1].metric,
        absolute_metrics=["a", "b"],
        num_points=num_trials,
    )

    plt_stuff = plot_pareto_frontier(frontier, CI_level=0.95)
    py.plot(plt_stuff.data, filename='simple-lineCFH.html')

    y1_weights = [x['y1_weight'] for x in frontier.param_dicts]
    a_mean = frontier.means['a']
    b_mean = frontier.means['b']

    oracles = [oracle_values(G1, G2, weight) for weight in y1_weights]
    blunt = [blunt_values(G1, G2, weight) for weight in y1_weights]


    df_out = pd.DataFrame(
        {
            'parameter': y1_weights,
            'a_mean': a_mean,
            'b_mean': b_mean,
            'a_sem': frontier.sems['a'],
            'b_sem': frontier.sems['b'],
            'a_oracle': np.array(oracles)[:,0],
            'b_oracle': np.array(oracles)[:,1],
            'a_blunt': np.array(blunt)[:,0],
            'b_blunt': np.array(blunt)[:,1]
        }
    )

    if time_iter:
        return df_out, time.time() - start_hybrid, time_loop
    else:
        return df_out, time.time() - start_hybrid
