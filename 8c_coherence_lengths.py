import joblib
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from helper import LP_filter_data
import random
import math
# from helper import kcluster, ward_agglom
from pandas.plotting import parallel_coordinates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


def bin_eq_num(df, var, bins):  # bin for equal num of points per bin

    bin_centers = []
    medians = []

    df = df.sort_values(by=[var], ignore_index=True)

    frac = int(len(df) / bins)  # adjust here for number of bins

    for i in range(bins):

        set = df.loc[frac * i:]
        set = set.loc[:frac * (i + 1)]

        bin_centers.append((set[var].max() + set[var].min()) / 2)

        median = set['r'].median()
        medians.append(median)

    return bin_centers, medians

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

coherence_cases = joblib.load('vlim_coherence_cases.pkl')
# ['direction', 'd_ab', 'r_ab',
# 'd_ac', 'r_ac',
# 'd_bc', 'r_bc',
# 'v0_std', 'v1_std', 'shock_speed']

####################################### PREP COHERENCE FALLOFF DATA #######################################

falloff_df = pd.DataFrame(columns=['direction', 'd', 'r', 'Velocity Std. Dev.', 'Shock Speed'])

falloff_df['direction'] = np.concatenate([coherence_cases['direction'], coherence_cases['direction']])
falloff_df['d'] = np.concatenate([coherence_cases['d_ab'], coherence_cases['d_ac']])
falloff_df['r'] = np.concatenate([coherence_cases['r_ab'], coherence_cases['r_ac']])
falloff_df['Velocity Std. Dev.'] = np.concatenate([coherence_cases['v0_std'], coherence_cases['v1_std']])
falloff_df['Shock Speed'] = np.concatenate([coherence_cases['shock_speed'], coherence_cases['shock_speed']])

# All data
bins_all, medians_all = bin_eq_num(falloff_df, 'd', bins=8)


###################################### FF vs FR #######################################

ff = falloff_df.loc[falloff_df['direction'] == 'ff']
fr = falloff_df.loc[falloff_df['direction'] == 'fr']
bins_ff, medians_ff = bin_eq_num(ff, 'd', bins=7)
bins_fr, medians_fr = bin_eq_num(fr, 'd', bins=7)


####################################### SPEED STRATIFICATION #######################################

sorter = 'Shock Speed'

q1 = np.percentile(falloff_df[sorter], 25)
q2 = np.percentile(falloff_df[sorter], 50)
q3 = np.percentile(falloff_df[sorter], 75)

data_q1 = falloff_df.loc[falloff_df[sorter] < q1]
data_q2 = falloff_df.loc[(falloff_df[sorter] >= q1) & (falloff_df[sorter] < q2)]
data_q3 = falloff_df.loc[(falloff_df[sorter] >= q2) & (falloff_df[sorter] < q3)]
data_q4 = falloff_df.loc[falloff_df[sorter] >= q3]

bins_q1, medians_q1 = bin_eq_num(data_q1, 'd', 4)
bins_q2, medians_q2 = bin_eq_num(data_q2, 'd', 4)
bins_q3, medians_q3 = bin_eq_num(data_q3, 'd', 4)
bins_q4, medians_q4 = bin_eq_num(data_q4, 'd', 4)

df = pd.DataFrame(columns=['d', 'r'])
df['d'] = bins_q4  ####### Change depending on which set
df['r'] = medians_q4

# print(bins_all)
# print(medians_all)

################################ FIT EXPONENTIAL DECAY FUNCTION FOR COHERENCE LENGTH ################################

def fit_exp(data, dimension):
    data = data.dropna()

    x = abs(data[dimension].to_numpy()).reshape((-1, 1)).flatten()
    y = (data['r'].to_numpy()).flatten()

    # fit to e^kx + c
    popt, pcov = curve_fit(
        lambda t, k, c: np.exp(k * t) + c,
        x, y
    )

    return popt[0], popt[1]

def exp(x, k, c):
    return np.exp(k * x) + c

def plot_exp(data, dimension):

    plt.scatter(x=abs(data[dimension]), y=data['r'], s=50, linewidths=0, marker='o')

    k, c = fit_exp(data, dimension)

    print('Length:', 1/k)

    fit = exp(abs(data[dimension]), k, c)

    r_sq = r2_score(data['r'], fit)
    print('R-squared:', r_sq)

    plt.plot(abs(data[dimension]), fit, c='red')
    plt.show()

plot_exp(df, 'd')
