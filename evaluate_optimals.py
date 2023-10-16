import pandas as pd
import random

from rpy2.robjects import pandas2ri
from rpy2.rinterface import RRuntimeWarning
import warnings
warnings.filterwarnings("ignore", category=RRuntimeWarning)
pandas2ri.activate()
from tqdm import tqdm


from rpy2.robjects.packages import STAP
#Read the file with the R code snippet
with open('tree_utility.R', 'r') as f:
    string = f.read()
#Parse using STAP
evaluate_tree = STAP(string, "evaluate_tree")


def evaluate(parameter, X, gamma1, gamma2, search_depth, se=True, depth=3):
    evaluation = evaluate_tree.evaluate_tree(
        X, gamma1, gamma2,
        parameter, search_depth, se=se, depth=depth
    )
    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    if se:
        return {"a": (evaluation[0], evaluation[2]), "b": (evaluation[1], evaluation[3])}
    else:
        return {"a": evaluation[0], "b": evaluation[1]}


# Load data
df = pd.read_csv('kenya_dr.csv')
X = pd.read_csv('kenya_covars.csv')
X_lim = pd.read_csv('kenya_limited_covars.csv')
X_vlim = X_lim[["dist_clinic", "LOG_patient_age1", "subfarm"]]
# r_dataframeX = pandas2ri.py2rpy(df[['hhh_gender', 'hhh_age', 'hhh_literacy',
#        'age', 'female', 'monthly_spending']])
# r_dataframeG1 = pandas2ri.py2rpy(df[['any_drops2-1', 'any_drops3-1', 'any_drops4-1']] * 0.1446128)
G1 = df.filter(regex='yes_treated$', axis=1)
G2 = df.filter(regex='no_treated$', axis=1)
r_dataframeG1 = pandas2ri.py2rpy(G1)
r_dataframeX = pandas2ri.py2rpy(X)
r_dataframeX_lim = pandas2ri.py2rpy(X_lim)
r_dataframeX_vlim = pandas2ri.py2rpy(X_vlim)
r_dataframeG2 = pandas2ri.py2rpy(G2)
y1_weights = pd.read_csv('pareto_resultsCFG.csv')["parameter"]

# Selected optimal points
eval_weights = random.sample(y1_weights.to_list(), 20)
opt_evals = [evaluate(parameter, r_dataframeX_vlim, r_dataframeG1, r_dataframeG2, 3, se=False, depth = 3) for parameter in tqdm(eval_weights)]
opt_df = pd.DataFrame.from_records(opt_evals)
opt_df["parameter"] = eval_weights
opt_df.to_csv("opt_evaluations_small.csv")

