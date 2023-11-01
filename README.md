# OptimalPolicyMOBO

This repository implements the Multi-Objective Policy Learning (MOPOL) method proposed by Rehill and Biddle (2023).
This includes an example looking at designing a subsidy program for anti-malarial drugs in Kenya.
This approach uses R and Python packages and so while the multi-objective Bayesian optimisation is carried out in Python,
the best implementation of policy learning is in the *policytree* package in R.

The main functions to run MOPoL are found in `fit_functions.py`.
`tree_utility.R` provides the R function that are used under the hood to implement policy learning. 
`kenya_dr.R` estimates doubly robust scores for use in the example (and can be adapted for use in other applications).
`evaluate_depths.py` and `evaluate_optimals.py` make up most of the application in the paper.
`sim_study.py` covers the specification tests from the paper.
`frontier_function.R` provides an R function to find the area under a 2D Pareto frontier.
`plot_output.R` plots the output of to MOPoL optimisation.
`specification_tests.Rmd` provides the plots for the specification tests.

This repository does not represent a general-purpose package for MOPoL.
The functions are designed to take two outcomes and a single hyperparameter weighting these outcomes,
however, MOPoL can in theory take an arbitrary number of objectives and use hyperparameters other than just weighting of objectives to control the trade-off.
For example, it is possible to add tree hyperparameters like maximum depth to the MOBO and see the trade-off between
outcomes and model characteristics. 
Adapting this code would require changing the functions in `fit_functions.py` and `tree_utility.R`
to code in the hyperparameters that will be used in the MOBO.