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

def bin_eq_num(df, var, bins):  # bin for equal num of points per bin

    bin_centers = []
    medians = []
    conf_intervals = []

    df = df.sort_values(by=[var], ignore_index=True)

    frac = int(len(df) / bins)

    for i in range(bins):

        set = df.loc[frac * i:]
        set = set.loc[:frac * (i + 1)]

        bin_centers.append((set[var].max() + set[var].min()) / 2)

        median = set['r'].median()
        medians.append(median)
        std_err = set['r'].std() / np.sqrt(len(set))
        conf_int = 1.96 * std_err  # 95% confidence interval

        conf_intervals.append(conf_int)

    print('{} points per bin'.format(round(len(df) / bins)))
    return bin_centers, medians, conf_intervals

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

####################################### PREP COHERENCE FALLOFF DATA #######################################

coherence_cases = joblib.load('vlim_coherence_cases.pkl')

falloff_df = pd.DataFrame(columns=['direction', 'd', 'r', 'Velocity Std. Dev.', 'Shock Speed'])

falloff_df['direction'] = np.concatenate([coherence_cases['direction'], coherence_cases['direction']])
falloff_df['d'] = np.concatenate([coherence_cases['d_ab'], coherence_cases['d_ac']])
falloff_df['r'] = np.concatenate([coherence_cases['r_ab'], coherence_cases['r_ac']])
falloff_df['Velocity Std. Dev.'] = np.concatenate([coherence_cases['v0_std'], coherence_cases['v1_std']])
falloff_df['Shock Speed'] = np.concatenate([coherence_cases['shock_speed'], coherence_cases['shock_speed']])

####################################### PLOT COHERENCE FALLOFF #######################################

fig = plt.figure(figsize=(8, 3))

####################################### all shocks - use 8 bins #######################################

bins_all, medians_all, c_ints_all = bin_eq_num(falloff_df, 'd', 8)

plt.plot(bins_all, medians_all, c='red')
c_ints_all = np.array(c_ints_all)
plt.fill_between(bins_all, medians_all - c_ints_all, medians_all + c_ints_all, alpha=0.1, color='red', linewidth=0)
plt.title('All Shocks - Coherence Falloff Over Distance')

####################################### speed-stratified - use 6 bins #######################################

sorter = 'Shock Speed'
bins = 6

q1 = np.percentile(falloff_df[sorter], 25)
q2 = np.percentile(falloff_df[sorter], 50)
q3 = np.percentile(falloff_df[sorter], 75)

print(q1, q2, q3)

data_q1 = falloff_df.loc[falloff_df[sorter] < q1]
data_q2 = falloff_df.loc[(falloff_df[sorter] >= q1) & (falloff_df[sorter] < q2)]
data_q3 = falloff_df.loc[(falloff_df[sorter] >= q2) & (falloff_df[sorter] < q3)]
data_q4 = falloff_df.loc[falloff_df[sorter] >= q3]

bins_q1, medians_q1, c_ints_q1 = bin_eq_num(data_q1, 'd', bins)
bins_q2, medians_q2, c_ints_q2 = bin_eq_num(data_q2, 'd', bins)
bins_q3, medians_q3, c_ints_q3 = bin_eq_num(data_q3, 'd', bins)
bins_q4, medians_q4, c_ints_q4 = bin_eq_num(data_q4, 'd', bins)


plt.title('Coherence Falloff - Stratified by {}'.format(sorter))

plt.plot(bins_q4, medians_q4, label='4th Quartile', color='red')
c_ints_q4 = np.array(c_ints_q4)
plt.fill_between(bins_q4, medians_q4 - c_ints_q4, medians_q4 + c_ints_q4, alpha=0.1, color='red', linewidth=0)

plt.plot(bins_q3, medians_q3, label='3rd Quartile', color='green')
c_ints_q3 = np.array(c_ints_q3)
plt.fill_between(bins_q3, medians_q3 - c_ints_q3, medians_q3 + c_ints_q3, alpha=0.1, color='green', linewidth=0)

plt.plot(bins_q2, medians_q2, label='2nd Quartile', color='orange')
c_ints_q2 = np.array(c_ints_q2)
plt.fill_between(bins_q2, medians_q2 - c_ints_q2, medians_q2 + c_ints_q2, alpha=0.1, color='orange', linewidth=0)

plt.plot(bins_q1, medians_q1, label='1st Quartile', color='blue')
c_ints_q1 = np.array(c_ints_q1)
plt.fill_between(bins_q1, medians_q1 - c_ints_q1, medians_q1 + c_ints_q1, alpha=0.1, color='blue', linewidth=0)

####################################### ff vs fr shocks - use 7 bins #######################################

bins = 7

ff = falloff_df.loc[falloff_df['direction'] == 'ff']
fr = falloff_df.loc[falloff_df['direction'] == 'fr']

bins_ff, medians_ff, c_ints_ff = bin_eq_num(ff, 'd', bins)
bins_fr, medians_fr, c_ints_fr = bin_eq_num(fr, 'd', bins)

plt.plot(bins_ff, medians_ff, label='FF', color='red')
c_ints_ff = np.array(c_ints_ff)
plt.fill_between(bins_ff, medians_ff - c_ints_ff, medians_ff + c_ints_ff, color='red', alpha=0.1, linewidth=0)

plt.plot(bins_fr, medians_fr, label='FR', color='blue')
c_ints_fr = np.array(c_ints_fr)
plt.fill_between(bins_fr, medians_fr - c_ints_fr, medians_fr + c_ints_fr, color='blue', alpha=0.1, linewidth=0)

plt.title('Coherence Falloff - FF vs FR')


plt.legend()
plt.grid()
plt.xlabel('Distance (Re)')
plt.ylabel('Coherence (r)')
plt.tight_layout()
# plt.savefig('5_coherence_falloff/dir_sep_shocks.png', dpi=300, format='png')
plt.show()
