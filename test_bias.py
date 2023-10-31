import pandas as pd
from fit_functions import fit_frontier
from time import time
import numpy as np

NUM_TRIALS = 100
N_TREAT = 5
N_X = 50
N_Y = 1

# Set up

np.random.seed(250693)

# Create a DataFrame with 50 random variables
data = np.random.rand(1000, N_X)  # 100 rows and 50 columns of random values between 0 and 1

# Create column names for the DataFrame
column_names = [f"Variable_{i}" for i in range(N_X)]

# Create Y
dataY = np.random.rand(1000, N_Y)
column_namesY = [f"Outcome_{i}" for i in range(N_Y)]


# Create the DataFrame
X = pd.DataFrame(data, columns=column_names)
W = np.random.randint(0, high=N_TREAT, size=1000, dtype=int)
X = pd.DataFrame(W, columns="Treatment")
Y = pd.DataFrame(dataY, columns=column_namesY)

X.to_csv("synth_dataX.csv")
Y.to_csv("synth_dataY.csv")
W.to_csv("synth_dataW.csv")


# Get DR scores from R grf package
import subprocess; subprocess.run(["Rscript", "dr_scores_synth.R"])

# Get DR scores
gamma = pd.read_csv("synth_gamma.csv")



