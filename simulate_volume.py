from main import run_mopol
import pandas as pd
from rpy2.robjects import pandas2ri
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


dr_df = pd.read_csv('mopol_data.csv')
pred_df = pd.read_csv('mopol_data.csv')

assert len(dr_df) == len(pred_df)

# Calculate the number of rows to sample (half of the rows)
num_rows_to_sample = len(dr_df) // 2

# Generate random row indices to sample
random_indices = np.random.choice(len(dr_df), num_rows_to_sample, replace=False)

# Sample rows from both data frames using the same indices
dr_df = dr_df.iloc[random_indices]
pred_df = pred_df.iloc[random_indices]

other_indices = np.setdiff1d(np.arange(len(dr_df)), random_indices)
dr_df_test = dr_df.iloc[other_indices]
pred_df_test = pred_df.iloc[other_indices]

def split_trial(df):
    train_df, test_df = train_test_split(df, test_size=0.5, random_state=99)

    # r_dataframeX = pandas2ri.py2rpy(df[['hhh_gender', 'hhh_age', 'hhh_literacy',
    #        'age', 'female', 'monthly_spending']])
    r_dataframeG1 = pandas2ri.py2rpy(train_df[['any_drops2-1', 'any_drops3-1', 'any_drops4-1']] * 0.1446128)
    r_dataframeG2 = pandas2ri.py2rpy(train_df[['maths2-1', 'maths3-1', 'maths4-1']] * 0.9120552)
    r_dataframeX = pandas2ri.py2rpy(train_df[["hhh_gender", "hhh_age", "monthly_spending", "hhh_literacy", "age", "gender", "benef", "female", "est_num_kids", "f4", "school_spending"]])
    r_dataframeG2 = pandas2ri.py2rpy(train_df[['maths2-1', 'maths3-1', 'maths4-1']] * 0.9120552)
    # r_dataframe_cost = pandas2ri.py2rpy()

    return run_mopol(r_dataframeX, r_dataframeG1, r_dataframeG2, 40)

with open("dr_trial.pkl", 'wb') as file:
    pickle.dump(split_trial(dr_df), file)

with open("pred_trial.pkl", 'wb') as file:
    pickle.dump(split_trial(pred_df), file)
