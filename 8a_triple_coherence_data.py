import joblib
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from helper import LP_filter_data
import random

# trim time series
def cut_time_series(data, start, end):
    data = data.loc[data.index <= end]
    data = data.loc[data.index >= start]

    return data

# calculate correlation between two offset time-series
def calc_offset_corr(arr_1, arr_2, offset):
    arr_1_overlap = arr_1[offset:]
    arr_2_overlap = arr_2[:-offset]

    correlation = np.corrcoef(arr_1_overlap, arr_2_overlap)[0, 1]

    return correlation

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

######################################### DATA AND SETUP #########################################


corrected_cases = joblib.load('vlim_corrected_cases.pkl')
all_data = joblib.load('all_data.pkl')

# corrected_cases.drop(columns=['orbit_normal', 'v_vect'], inplace=True)
print(corrected_cases.keys())

coherence_cases = pd.DataFrame(columns=['direction', 'd_ab', 'r_ab',
                                        'd_ac', 'r_ac',
                                        'd_bc', 'r_bc',
                                        'v0_std', 'v1_std', 'shock_speed', 'alfvenicity'])

######################################### ITERATE OVER SHOCKS #########################################

for i in range(len(corrected_cases)):

    ######################################### pull shock parameters #########################################

    prog = i / len(corrected_cases) * 100
    if prog % 10 == 0:
        print('Progress:', prog, '%')
    case = corrected_cases.iloc[i]

    v_vect = case['v_corrected']
    speed = np.linalg.norm(v_vect)
    n_hat = v_vect / speed

    start = case['start']
    end = start + datetime.timedelta(minutes=2*int(case['tot_window']))
    cut = cut_time_series(all_data, start, end)

    sat_1, sat_2, sat_3 = case['sat_a'], case['sat_b'], case['sat_c']
    lag_ab, lag_ac = case['lag_ab'], case['lag_ac']
    lag_bc = lag_ac - lag_ab

    sat_1_b = np.vstack((cut[sat_1+'_bx'], cut[sat_1+'_by'], cut[sat_1+'_bz'])).T
    sat_2_b = np.vstack((cut[sat_2+'_bx'], cut[sat_2+'_by'], cut[sat_2+'_bz'])).T
    sat_3_b = np.vstack((cut[sat_3+'_bx'], cut[sat_3+'_by'], cut[sat_3+'_bz'])).T

    sat_1_par = np.dot(sat_1_b, n_hat)
    sat_2_par = np.dot(sat_2_b, n_hat)
    sat_3_par = np.dot(sat_3_b, n_hat)

    ################################# calculate coherence between time-lagged signals #################################

    cutoff = case['tot_window'] / 4
    sat_1_par = LP_filter_data(data=sat_1_par, cutoff1=cutoff, order=5)
    sat_2_par = LP_filter_data(data=sat_2_par, cutoff1=cutoff, order=5)
    sat_3_par = LP_filter_data(data=sat_3_par, cutoff1=cutoff, order=5)

    r_ab = calc_offset_corr(sat_1_par, sat_2_par, lag_ab)
    r_ac = calc_offset_corr(sat_1_par, sat_3_par, lag_ac)
    r_bc = calc_offset_corr(sat_2_par, sat_3_par, lag_bc)

    a = case[['pos_a_x', 'pos_a_y', 'pos_a_z']].to_numpy()
    b = case[['pos_b_x', 'pos_b_y', 'pos_b_z']].to_numpy()
    c = case[['pos_c_x', 'pos_c_y', 'pos_c_z']].to_numpy()

    d_ab = abs(np.dot(b - a, n_hat))
    d_ac = abs(np.dot(c - a, n_hat))
    d_bc = abs(np.dot(c - b, n_hat))

    sat_1_v = np.vstack((cut[sat_1+'_vx'], cut[sat_1+'_vy'], cut[sat_1+'_vz'])).T
    sat_2_v = np.vstack((cut[sat_2+'_vx'], cut[sat_2+'_vy'], cut[sat_2+'_vz'])).T

    sat_1_v_par = np.dot(sat_1_v, n_hat)
    sat_2_v_par = np.dot(sat_2_v, n_hat)

    sat_1_v_par_std = np.std(sat_1_v_par)
    sat_2_v_par_std = np.std(sat_2_v_par)

    alfvenicity_1 = np.corrcoef(sat_1_par, sat_1_v_par)[0, 1]
    alfvenicity_2 = np.corrcoef(sat_2_par, sat_2_v_par)[0, 1]
    alfvenicity = (alfvenicity_1 + alfvenicity_2) / 2


    # (columns=['direction', 'd_ab', 'r_ab',
    #                        'd_ac', 'r_ac',
    #                        'd_bc', 'r_bc',
    #                        'v0_std', 'v1_std', 'shock_speed', 'alfvenicity'])
    coherence_cases.loc[len(coherence_cases)] = [case['direction'], d_ab, r_ab, d_ac, r_ac, d_bc, r_bc, sat_1_v_par_std, sat_2_v_par_std, speed, alfvenicity]

# print(coherence_cases.head())

joblib.dump(coherence_cases, 'vlim_coherence_cases.pkl')
