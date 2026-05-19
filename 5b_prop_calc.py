import joblib
import datetime
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# trim time series
def cut_time_series(data, start, end):
    data = data.loc[data.index <= end]
    data = data.loc[data.index >= start]

    return data

# plot step functions
def plot_b(data, title):

    fig, ax = plt.subplots(2)

    ax[0].plot(data['ace_b'], label='ACE')
    ax[0].plot(data['wind_b'], label='Wind')
    ax[0].plot(data['dsc_b'], label='DSC')
    ax[0].set_ylabel('|B| (nT)')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax[0].grid(True)

    ax[1].plot(abs(data['ace_v']), label='ACE')
    ax[1].plot(abs(data['wind_v']), label='Wind')
    ax[1].plot(abs(data['dsc_v']), label='DSC')
    ax[1].set_ylabel('Proton speed (km/s)')
    ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))

    plt.suptitle(title)
    plt.xlabel('Time (UTC)')
    plt.grid(True)
    plt.show()

######################################### DATA PULL, SET UP DATAFRAMES #########################################

cases = joblib.load('all_cases.pkl')
all_data = joblib.load('all_data.pkl')
print(len(cases))
# columns=['t_a', 'lag_ab', 'lag_ac',
#                               'pos_a_x', 'pos_a_y', 'pos_a_z',
#                               'pos_b_x', 'pos_b_y', 'pos_b_z',
#                               'pos_c_x', 'pos_c_y', 'pos_c_z',
#                               'orb_norm_x', 'orb_norm_y', 'orb_norm_z',
#                               'v_true_x', 'v_true_y', 'v_true_z',
#                               'v_calc_x', 'v_calc_y', 'v_calc_z'])

sunward_props = pd.DataFrame(columns=['start', 'tot_window', 'direction',
                                      'sat_a', 'sat_b', 'sat_c',
                                      't_a', 'lag_ab', 'lag_ac',
                                      'pos_a_x', 'pos_a_y', 'pos_a_z',
                                      'pos_b_x', 'pos_b_y', 'pos_b_z',
                                      'pos_c_x', 'pos_c_y', 'pos_c_z',
                                      'orb_norm_x', 'orb_norm_y', 'orb_norm_z',
                                      'v_calc_x', 'v_calc_y', 'v_calc_z'])
earthward_props = pd.DataFrame(columns=['start', 'tot_window', 'direction',
                                        'sat_a', 'sat_b', 'sat_c',
                                        't_a', 'lag_ab', 'lag_ac',
                                        'pos_a_x', 'pos_a_y', 'pos_a_z',
                                        'pos_b_x', 'pos_b_y', 'pos_b_z',
                                        'pos_c_x', 'pos_c_y', 'pos_c_z',
                                        'orb_norm_x', 'orb_norm_y', 'orb_norm_z',
                                        'v_calc_x', 'v_calc_y', 'v_calc_z'])

######################################## ITERATE THROUGH STEP-IDENTIFIED CASES #########################################

for i in range(len(cases)):
    case = cases.iloc[i]

    start = case['time']
    end = start + datetime.timedelta(minutes=int(case['window']))
    cut = cut_time_series(all_data, start, end)

    sat_a, sat_b, sat_c = case['sat_a'], case['sat_b'], case['sat_c']
    lag_ab, lag_ac = case['lag_ab'], case['lag_ac']

    ######################################## Plane wave fit ########################################

    try:
        # Sensor positions (3D coordinates)
        a = np.array([cut[sat_a + '_x'].iloc[case['t_a']], cut[sat_a + '_y'].iloc[case['t_a']],
                      cut[sat_a + '_z'].iloc[case['t_a']]])
        b = np.array([cut[sat_b + '_x'].iloc[case['t_a'] + lag_ab], cut[sat_b + '_y'].iloc[case['t_a'] + lag_ab],
                      cut[sat_b + '_z'].iloc[case['t_a'] + lag_ab]])
        c = np.array([cut[sat_c + '_x'].iloc[case['t_a'] + lag_ac], cut[sat_c + '_y'].iloc[case['t_a'] + lag_ac],
                      cut[sat_c + '_z'].iloc[case['t_a'] + lag_ac]])
    except:
        # print('Time lag out of bounds')
        continue

    orbit_normal = np.cross(b - a, c - a)
    orbit_normal = orbit_normal / np.linalg.norm(orbit_normal)

    D = np.vstack((b - a, c - a))  # distances in Re
    t = np.array([lag_ab, lag_ac])  # time lags in minutes

    # Solve for k*D=t coefficient (k=1/v)
    n, residuals, rank, s = np.linalg.lstsq(D, t, rcond=None)

    # Normalize to get the unit direction vector
    n_hat = n / np.linalg.norm(n)
    # print('Unit direction of propagation:', n_hat)

    try:
        v = 1. / float(np.linalg.norm(n))  # v in Re/min
    except:
        # print('error in plane wave fit - skipping')
        continue

    v_kms = v * 6378. / 60.
    prop_speed = np.linalg.norm(v_kms)
    # print('Propagation speed (km/s):', prop_speed)

    v_vect = prop_speed * n_hat

    ######################################## VALIDATE AND SAVE ########################################

    ########## Checking if solution reaches points at correct times (will skip if false) ##########
    d_par = np.dot(D[0], n_hat)
    v_t = v * t[0]
    val_1 = abs(d_par - v_t) / d_par <= 0.05  # check within 5%

    d_par = np.dot(D[1], n_hat)
    v_t = v * t[1]
    val_2 = abs(d_par - v_t) / d_par <= 0.05

    if val_1 == False or val_2 == False:
        # print('Fit incorrect - skipping')
        continue

    tot_window = case['window']
    direction = case['direction']
    t_a = case['t_a']

    # (columns=['start', 'tot_window', 'direction',
    # 't_a', 'lag_ab', 'lag_ac',
    # 'pos_a_x', 'pos_a_y', 'pos_a_z',
    # 'pos_b_x', 'pos_b_y', 'pos_b_z',
    # 'pos_c_x', 'pos_c_y', 'pos_c_z'
    # 'orb_norm_x', 'orb_norm_y', 'orb_norm_z',
    # 'v_calc_x', 'v_calc_y', 'v_calc_z'])

    row = [start, tot_window, direction,
           sat_a, sat_b, sat_c,
           t_a, lag_ab, lag_ac,
           a[0], a[1], a[2],
           b[0], b[1], b[2],
           c[0], c[1], c[2],
           orbit_normal[0], orbit_normal[1], orbit_normal[2],
           v_vect[0], v_vect[1], v_vect[2]]

    if v_vect[0] > 0:
        sunward_props.loc[len(sunward_props)] = row
    else:
        earthward_props.loc[len(earthward_props)] = row

joblib.dump(sunward_props, 'raw_sunward_props.pkl')
joblib.dump(earthward_props, 'raw_earthward_props.pkl')
