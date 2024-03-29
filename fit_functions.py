from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from time import time

import torch
import pandas as pd

# Plotting imports and initialization
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.pareto_utils import compute_posterior_pareto_frontier, get_observed_pareto_frontiers
from ax.plot.pareto_frontier import plot_pareto_frontier, scatter_plot_with_hypervolume_trace_plotly
# init_notebook_plotting()
import plotly.offline as py
from tqdm import tqdm

import random
import subprocess

subprocess.run(["Rscript", "kenya_dr.R"])

random.seed(11)
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
import pickle

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
    """
    Initialises a new Ax client to control the MOBO process.
    :return: A new Ax client
    """
    ax_client = AxClient()
    ax_client.create_experiment(
        name="experiment",
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
    """
    This takes the parameters for fitting a policy tree and passes them to the R implementation of policytree.
    It then estimates the value of this tree for each outcome and standard errors for these estimates.
    :param parameters: This gets the hyperparameter settings for the evaluation generated by the Ax client.
    :param X: Covariates to for fitting the policy tree.
    :param gamma1: Doubly robust scores for outcome 1.
    :param gamma2: Doubly robust scores for outcome 2.
    :param search_depth: The number of levels across which the learner will optimise each split (recomend this not be above 3).
    :param depth: The total depth of the decision tree.
    :param bs_replicates: The number of replicates used to bootstrap standard errors.
    :return: The point estimate and standard errors for each outcome for the policy tree fit on these parameters.
    """
    evaluation = evaluate_tree.evaluate_tree(
        X, gamma1, gamma2,
        parameters.get("y1_weight"), search_depth = search_depth, depth = depth, bs = bs_replicates
    )
    return {"a": (evaluation[0], evaluation[2]), "b": (evaluation[1], evaluation[3])}

# def evaluate_cost(parameters, X, gamma1, cost, search_depth):
#     evaluation = evaluate_tree.evaluate_tree_cost(
#         X, gamma1, cost *
#         parameters.get("y1_weight"), search_depth,
#     )
#     # In our case, standard error is 0, since we are computing a synthetic function.
#     # Set standard error to None if the noise level is unknown.
#     return {"a": (evaluation[0], evaluation[2]), "b": (evaluation[1], evaluation[3])}

# def set_values(row):
#     if row['female']:
#         return row
#     else:
#         return 0

def oracle_values(G1, G2, weight):
    """
    Gives the oracle values for a given weighting of outcomes. That is, outcomes if we could choose treatment at the
    individual level and knew the doubly robust scores for each individual.
    :param G1: Doubly robust scores for outcome 1.
    :param G2: Doubly robust scores for outcome 2.
    :param weight: The weight put on G1.
    :return: Expected value for each outcome.
    """
    G1 = G1.to_numpy()
    G2 = G2.to_numpy()
    gammax = (weight * G1) + ((1- weight) * G2)
    max_val = np.argmax(gammax, axis = 1)
    gamma1_value = np.choose(max_val, G1.T).mean()
    gamma2_value = np.choose(max_val, G2.T).mean()
    return gamma1_value, gamma2_value

def blunt_values(G1, G2, weight):
    """
    Returns the expected value for each outcome if we cannot discriminate in treatments at all, everyone gets the same one.
    :param G1: Doubly robust scores for outcome 1.
    :param G2: Doubly robust scores for outcome 2.
    :param weight: The weight put on G1.
    :return: Expected value for each outcome.
    """
    G1 = G1.to_numpy()
    G2 = G2.to_numpy()
    gammax = (weight * G1) + ((1 - weight) * G2)
    max_val = np.argmax(gammax.mean(axis = 0))
    gamma1_value = np.choose(max_val, G1.T).mean()
    gamma2_value = np.choose(max_val, G2.T).mean()
    return gamma1_value, gamma2_value


def fit_frontier(X, G1, G2, search_depth, num_trials, depth, bs_replicates, posterior = True):
    """
    Fits a Pareto frontier of models using multi-objective Bayesian optimisation based on policy trees of a given type.
    :param X: Covariates to for fitting the policy tree.
    :param G1: Doubly robust scores for outcome 1.
    :param G2: Doubly robust scores for outcome 2.
    :param search_depth: The number of levels across which the learner will optimise each split (recomend this not be above 3).
    :param num_trials: The number of iterations to run in fitting the frontier.
    :param depth: The total depth of the decision tree.
    :param bs_replicates: The number of replicates used to bootstrap standard errors.
    :param time_iter: Whether to return the time taken for each loop.
    :return: A Pareto frontier of parameters, estimated value for those points, standard errors of those estimates,
    oracle and blunt expected values for those points and optionally time taken for each loop.
    """
    G1_r = pandas2ri.py2rpy(G1)
    X_r = pandas2ri.py2rpy(X)
    G2_r = pandas2ri.py2rpy(G2)

    start_hybrid = time()
    time_loop = []
    ax_client = new_experiment()
    evaluations = []
    for i in tqdm(range(num_trials)):
        start = time()
        parameters, trial_index = ax_client.get_next_trial()
        evaluation = evaluate(parameters, X_r, G1_r, G2_r, search_depth=search_depth, depth=depth, bs_replicates=bs_replicates)
        print(evaluation)
        evaluations.append(evaluation)
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluation)
        end = time()
        time_loop.append(end - start)

    objectives = ax_client.experiment.optimization_config.objective.objectives

    frontier = compute_posterior_pareto_frontier(
        experiment=ax_client.experiment,
        data=ax_client.experiment.fetch_data(),
        primary_objective=objectives[0].metric,
        secondary_objective=objectives[1].metric,
        absolute_metrics=["a", "b"],
        num_points=100,
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

    hv_plot = scatter_plot_with_hypervolume_trace_plotly(ax_client.experiment)
    hvs = pd.DataFrame(hv_plot.data[0]["y"], columns=['HV'])

    with open('evaluations.pkl', 'wb') as file:
        pickle.dump(evaluations, file)


    return df_out, time() - start_hybrid, time_loop, hvs
