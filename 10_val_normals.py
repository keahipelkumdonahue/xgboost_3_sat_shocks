import joblib
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# predict plane wave arrival times based on position and velocity data
def plane_prop(pos_a, pos_b, pos_c, v_vect):

    # Calculate distances from start point along normal direction
    speed = np.linalg.norm(v_vect)
    normal = v_vect / speed

    pos_a, pos_b, pos_c = np.array(pos_a*6378), np.array(pos_b*6378), np.array(pos_c*6378)

    dist_b = abs(np.dot(normal, pos_b - pos_a))
    dist_c = abs(np.dot(normal, pos_c - pos_a))

    time_b = dist_b / speed
    time_c = dist_c / speed

    time_b, time_c = time_b / 60, time_c / 60

    return time_b, time_c

# trim time-series
def cut_time_series(data, start, end):
    data = data.loc[data.index <= end]
    data = data.loc[data.index >= start]

    return data

# convert times from string to datetime
def convert_time_string(time_str):
    time_str = str(int(time_str)).zfill(4)
    hours = int(time_str[:2])
    minutes = int(time_str[2:])
    t = datetime.time(hours, minutes)
    return t.strftime('%H:%M')

# convert harvard database times to datetimes
def hvd_convert_date(obs_time):
    month = obs_time[:2]
    day = obs_time[3:5]
    year = obs_time[6:10]
    time = convert_time_string(obs_time[22:])

    time = pd.to_datetime(year + '-' + month + '-' + day) + pd.to_timedelta(time + ':00')
    return time

corrected_cases = joblib.load('vlim_corrected_cases.pkl')

####################################### LOAD HARVARD DATA #######################################

hvd = pandas.read_csv('wind_shocks_2017_2023.csv')
hvd['time'] = hvd['observation_time'].apply(hvd_convert_date)
hvd = hvd.drop(['observation_time', 'event_url', 'year'], axis=1)
hvd = hvd.set_index('time')
hvd['vx'] = hvd['median_Nx']  * hvd['median_shock_speed']
hvd['vy'] = hvd['median_Ny']  * hvd['median_shock_speed']
hvd['vz'] = hvd['median_Nz']  * hvd['median_shock_speed']
hvd = hvd.drop(['median_Nx', 'median_Ny', 'median_Nz', 'median_shock_speed'], axis=1)

####################################### LOAD HELSINKI DATA #######################################

fin = pd.read_csv('shocks_20250726_044117.dat', delimiter=',')
fin['time'] = pd.to_datetime(fin[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
fin = fin.loc[fin['Spacecraft'] == fin['Spacecraft'][0]]
# print(fin[['Shock normal X', 'Shock type']][:25])
fin = fin[['time', 'Shock normal X', 'Shock normal Y', 'Shock normal Z', 'Shock speed (km/s)']]
fin.columns = ['time', 'vx', 'vy', 'vz', 'speed']
fin = fin.loc[fin['time'] >= datetime.datetime(2017, 1, 1)].reset_index(drop=True)
fin = fin.loc[fin['time'] < datetime.datetime(2024, 1, 1)].reset_index(drop=True)
fin = fin.set_index('time')
fin['vx'] = fin['vx']  * fin['speed']
fin['vy'] = fin['vy']  * fin['speed']
fin['vz'] = fin['vz']  * fin['speed']
fin.drop(['speed'], axis=1, inplace=True)

####################################### COMPARE HARVARD AND HELSINKI #######################################

comp_df = pd.DataFrame(columns = ['start',
                                  'lag_ab', 'xgb_lag_ab', 'hvd_lag_ab', 'fin_lag_ab',
                                  'lag_ac', 'xgb_lag_ac', 'hvd_lag_ac', 'fin_lag_ac',
                                  'xg_vx', 'xg_vy', 'xg_vz',
                                  'hvd_vx', 'hvd_vy', 'hvd_vz',
                                  'fin_vx', 'fin_vy', 'fin_vz'])

fin_overlap = 0
hvd_overlap = 0
hvd_fin_overlap = 0

for i in range(len(hvd)):
    t = hvd.index[i]
    start = t - datetime.timedelta(minutes=15)
    end = t + datetime.timedelta(minutes=15)

    fin_incl = cut_time_series(fin, start, end)

    if len(fin_incl) > 0:
        hvd_fin_overlap += 1

print('Helsinki-Harvard overlap:', hvd_fin_overlap)

####################################### COMPARE THIS STUDY, HARVARD, HELSINKI #######################################

# ITERATE OVER CASES, CHECKING IF SHOWN IN HARVARD, HELSINKI LISTS
for i in range(len(corrected_cases)):
    shock = corrected_cases.iloc[i]
    start = shock['start']
    end = start + datetime.timedelta(minutes=int(shock['tot_window']))

    hvd_set = cut_time_series(hvd, start, end)
    fin_set = cut_time_series(fin, start, end)

    if len(fin_set) > 0:
        fin_overlap += 1
    if len(hvd_set) > 0:
        hvd_overlap += 1

    if len(hvd_set) > 0 and len(fin_set) > 0:

        if len(hvd_set) > 1 or len(fin_set) > 1:
            print('Multiple shocks found')
            continue

        xg_vx, xg_vy, xg_vz = shock['v_corrected']

        fin_vx = fin_set['vx'].to_numpy()[0]
        fin_vy = fin_set['vy'].to_numpy()[0]
        fin_vz = fin_set['vz'].to_numpy()[0]

        hvd_vx = hvd_set['vx'].to_numpy()[0]
        hvd_vy = hvd_set['vy'].to_numpy()[0]
        hvd_vz = hvd_set['vz'].to_numpy()[0]

        lag_ab = shock['lag_ab']
        lag_ac = shock['lag_ac']

        shock_pos_a = shock[['pos_a_x', 'pos_a_y', 'pos_a_z']].to_numpy()
        shock_pos_b = shock[['pos_b_x', 'pos_b_y', 'pos_b_z']].to_numpy()
        shock_pos_c = shock[['pos_c_x', 'pos_c_y', 'pos_c_z']].to_numpy()

        xgb_v = [xg_vx, xg_vy, xg_vz]
        hvd_v = [hvd_vx, hvd_vy, hvd_vz]
        fin_v = [fin_vx, fin_vy, fin_vz]

        xgb_lag_ab, xgb_lag_ac = plane_prop(shock_pos_a, shock_pos_b, shock_pos_c, xgb_v)
        hvd_lag_ab, hvd_lag_ac = plane_prop(shock_pos_a, shock_pos_b, shock_pos_c, hvd_v)
        fin_lag_ab, fin_lag_ac = plane_prop(shock_pos_a, shock_pos_b, shock_pos_c, fin_v)

        comp_df.loc[len(comp_df)] = [start,
                                     lag_ab, xgb_lag_ab, hvd_lag_ab, fin_lag_ab,
                                     lag_ac, xgb_lag_ac, hvd_lag_ac, fin_lag_ac,
                                     xg_vx, xg_vy, xg_vz,
                                     hvd_vx, hvd_vy, hvd_vz,
                                     fin_vx, fin_vy, fin_vz]

print('XGBoost:', len(corrected_cases))
print('Helsinki:', len(fin))
print('Harvard:', len(hvd))
print('Triple-Match:', len(comp_df))

# print(corrected_cases)

print('Helsinki overlap:', fin_overlap)
print('Harvard overlap:', hvd_overlap)

# COMPARE VIA CCC 1:1 CORRESPONDENCE SCORES
def concordance_correlation_coefficient(y_true, y_pred):
    """CCC measures agreement with y=x line"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

xg_ab_ccc = concordance_correlation_coefficient(comp_df['lag_ab'], comp_df['xgb_lag_ab'])
hvd_ab_ccc = concordance_correlation_coefficient(comp_df['lag_ab'], comp_df['hvd_lag_ab'])
fin_ab_ccc = concordance_correlation_coefficient(comp_df['lag_ab'], comp_df['fin_lag_ab'])

xg_ac_ccc = concordance_correlation_coefficient(comp_df['lag_ac'], comp_df['xgb_lag_ac'])
hvd_ac_ccc = concordance_correlation_coefficient(comp_df['lag_ac'], comp_df['hvd_lag_ac'])
fin_ac_ccc = concordance_correlation_coefficient(comp_df['lag_ac'], comp_df['fin_lag_ac'])

print('XGBoost: CCC for lag_ab:', xg_ab_ccc)
print('Helsinki: CCC for lag_ab:', hvd_ab_ccc)
print('Harvard: CCC for lag_ab:', fin_ab_ccc)
print()
print('XGBoost: CCC for lag_ac:', xg_ac_ccc)
print('Helsinki: CCC for lag_ac:', hvd_ac_ccc)
print('Harvard: CCC for lag_ac:', fin_ac_ccc)

# PLOT ARRIVAL TIMES WITH CCC SCORES

fig = plt.figure(figsize=(6, 6))

plt.subplot(2, 3, 1)
plt.scatter(comp_df['lag_ab'], comp_df['xgb_lag_ab'], label='CCC: {}'.format(round(xg_ab_ccc, 2)), s=20)
plt.axline((0, 0), slope=1, color='red', linestyle='--')
plt.xlabel('Lag (min)')
plt.ylabel('Predicted Time to Sat B (min)')
plt.title('Our Method')
plt.legend(handlelength=0, handletextpad=0, markerscale=0, loc='upper left')
plt.grid()
plt.ylim([-1,35])

plt.subplot(2, 3, 2)
plt.scatter(comp_df['lag_ab'], comp_df['fin_lag_ab'], label='CCC: {}'.format(round(fin_ab_ccc, 2)), s=20)
plt.axline((0, 0), slope=1, color='red', linestyle='--')
plt.xlabel('Lag (min)')
plt.title('Helsinki')
plt.grid()
plt.legend(handlelength=0, handletextpad=0, markerscale=0, loc='upper left')
plt.ylim([-1,35])

plt.subplot(2, 3, 3)
plt.scatter(comp_df['lag_ab'], comp_df['hvd_lag_ab'], label='CCC: {}'.format(round(hvd_ab_ccc, 2)), s=20)
plt.axline((0, 0), slope=1, color='red', linestyle='--')
plt.xlabel('Lag (min)')
plt.title('Harvard')
plt.grid()
plt.legend(handlelength=0, handletextpad=0, markerscale=0, loc='upper left')
plt.ylim([-1,35])

plt.subplot(2, 3, 4)
plt.scatter(comp_df['lag_ac'], comp_df['xgb_lag_ac'], label='CCC: {}'.format(round(xg_ac_ccc, 2)), s=20)
plt.axline((0, 0), slope=1, color='red', linestyle='--')
plt.xlabel('Lag (min)')
plt.ylabel('Predicted Time to Sat C (min)')
plt.grid()
plt.legend(handlelength=0, handletextpad=0, markerscale=0, loc='upper left')
plt.ylim([-1,60])

plt.subplot(2, 3, 5)
plt.scatter(comp_df['lag_ac'], comp_df['fin_lag_ac'], label='CCC: {}'.format(round(fin_ac_ccc, 2)), s=20)
plt.axline((0, 0), slope=1, color='red', linestyle='--')
plt.xlabel('Lag (min)')
plt.grid()
plt.legend(handlelength=0, handletextpad=0, markerscale=0, loc='upper left')
plt.ylim([-1,60])

plt.subplot(2, 3, 6)
plt.scatter(comp_df['lag_ac'], comp_df['hvd_lag_ac'], label='CCC: {}'.format(round(hvd_ac_ccc, 2)), s=20)
plt.axline((0, 0), slope=1, color='red', linestyle='--')
plt.xlabel('Lag (min)')
plt.grid()
plt.legend(handlelength=0, handletextpad=0, markerscale=0, loc='upper left')
plt.ylim([-1,60])

plt.tight_layout()
# plt.savefig('6_time_lags/time_lags.png', dpi=300, format='png')
plt.show()
