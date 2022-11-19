from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

import torch
import pandas as pd

# Plotting imports and initialization
# from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
# init_notebook_plotting()

# Load our sample 2-objective problem
from botorch.test_functions.multi_objective import BraninCurrin

# Read in R functions
from rpy2.robjects.packages import importr
import rpy2.robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects as ro
# ro.r('''source('tree_utility.r')''')
#
# evaluate_tree = ro.globalenv['evaluate_tree']

from rpy2.robjects.packages import STAP
#Read the file with the R code snippet
with open('tree_utility.R', 'r') as f:
    string = f.read()
#Parse using STAP
evaluate_tree = STAP(string, "evaluate_tree")



ax_client = AxClient()
ax_client.create_experiment(
    name="morocco_experiment",
    parameters=[
        {
            "name": f"y{i+1}_weight",
            "type": "range",
            "bounds": [0.0, 1.0],
        }
        for i in range(2)
    ],
    objectives={
        # `threshold` arguments are optional
        "a": ObjectiveProperties(minimize=False),
        "b": ObjectiveProperties(minimize=False)
    },
    overwrite_existing_experiment=True,
    is_test=True,
)


def evaluate(parameters, X, gamma1, gamma2):
    evaluation = evaluate_tree.evaluate_tree(
        X, gamma1, gamma2,
        parameters.get("y1_weight"),
        parameters.get("y2_weight")
    )
    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    return {"a": (evaluation[0], evaluation[2]), "b": (evaluation[1], evaluation[3])}


df = pd.read_csv('train_data1.csv')
r_dataframeX = pandas2ri.py2rpy(df[['hhh_gender', 'hhh_age', 'hhh_literacy',
       'age', 'female', 'monthly_spending']])
r_dataframeG1 = pandas2ri.py2rpy(df[['any_drops2-1', 'any_drops3-1', 'any_drops4-1']])
r_dataframeG2 = pandas2ri.py2rpy(df[['maths2-1', 'maths3-1', 'maths4-1']])



for i in range(100):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, r_dataframeX, r_dataframeG1, r_dataframeG2))

objectives = ax_client.experiment.optimization_config.objective.objectives
frontier = compute_posterior_pareto_frontier(
    experiment=ax_client.experiment,
    data=ax_client.experiment.fetch_data(),
    primary_objective=objectives[1].metric,
    secondary_objective=objectives[0].metric,
    absolute_metrics=["a", "b"],
    num_points=100,
)

plot_pareto_frontier(frontier, CI_level=0.90)
