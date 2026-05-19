import datetime
from datetime import timedelta
import joblib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd
import math
import scipy
import random
from helper import LP_filter_data
from scipy.optimize import curve_fit, OptimizeWarning
import sklearn
import warnings

# output/view settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore', 'overflow')
warnings.filterwarnings("ignore", category=OptimizeWarning)

def cut_time_series(data, start, end):
    left_idx = data.index.searchsorted(start)
    right_idx = data.index.searchsorted(end)
    return data.iloc[left_idx:right_idx]

##########################################################################################
############### This code creates the example shock and log fit plot #####################
##########################################################################################
def plot_b(data, title, amp, vert, rate, midpoint1, midpoint2, midpoint3):

    fig, ax = plt.subplots(2)

    ax[0].plot(data['wind_b'], label='Wind', c='blue')
    wind_y = logistic(np.arange(0, len(data)), amp, vert, rate, midpoint1)
    ax[0].plot(data['wind_b'].index, wind_y, c='red', linestyle='--', alpha=0.7)

    ax[0].plot(data['ace_b'], label='ACE', c='orange')
    ace_y = logistic(np.arange(0, len(data)), amp, vert, rate, midpoint2)
    ax[0].plot(data['ace_b'].index, ace_y, c='red', linestyle='--', alpha=0.7)

    ax[0].plot(data['dsc_b'], label='DSC', c='green')
    dsc_y = logistic(np.arange(0, len(data)), amp, vert, rate, midpoint3)
    ax[0].plot(data['dsc_b'].index, dsc_y, c='red', linestyle='--', label='Time-Shifted Logistic Fit', alpha=0.7)

    ax[0].set_ylabel('B (nT)')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=4, frameon=False)
    ax[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax[0].grid(True)
    # ax[0].set_xticklabels([])  # Removes x-axis tick labels

    ax[1].plot(abs(data['wind_v']), label='Wind', c='blue')
    ax[1].plot(abs(data['ace_v']), label='ACE', c='orange')
    ax[1].plot(abs(data['dsc_v']), label='DSC', c='green')
    ax[1].set_ylabel('Plasma speed (km/s)')
    ax[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))

    plt.suptitle(title)
    plt.xlabel('Time (UTC)')
    plt.grid(True)
    # plt.savefig('shock_log_ex_fig.png', dpi=300, format='png')
    plt.show()

# logistic function for step-function fitting
def logistic(x, a, b, c, d):
    # a is amplitude
    # b is vertical shift
    # c is growth rate
    # d is x value of midpoint
    return (a / (1+np.exp(c*(x-d))))+b

# fit logisitc function to data
def log_reg_fit(data, var, plot):
    x = np.arange(0, len(data))
    y = data[var].to_numpy()

    a0 = abs(max(y) - min(y))
    b0 = min(y)
    c0 = 1
    d0 = len(x) / 2
    params, cov = curve_fit(f=logistic, xdata=x, ydata=y, p0=[a0, b0, c0, d0])

    ypred = logistic(x, *params)
    r2 = sklearn.metrics.r2_score(y, ypred)

    amp = params[0]
    vert = params[1]
    rate = params[2]
    midpoint = params[3]

    if plot:
        plt.title('R2: {}, midpoint: {}'.format(round(r2, 2), round(midpoint,2)))
        plt.plot(x, y)
        plt.plot(x, ypred)
        plt.show()

    return r2, amp, vert, rate, midpoint

######################################### DATA AND PARAMETERS #########################################

tot_windows = [20, 40, 60, 80, 100, 120, 140, 160, 180]  # choice of window lengths
attempts = 5000
# tot_windows = [180] # for example figure
# attempts = 50 # for example figure

cases = pd.DataFrame(columns=['time', 'window', 'direction',
                              'sat_a', 'sat_b', 'sat_c',
                              't_a', 'lag_ab', 'lag_ac'])

all_data = joblib.load('all_data.pkl')

start = '01-01-2017'
end = '01-01-2024'

# start = '08-05-2023' # for example figure
# end = '08-06-2023'

date_range = pd.date_range(start=start, end=end, freq='d')

for i in range(len(date_range)-1):

    all_data_month = cut_time_series(all_data, date_range[i], date_range[i+1])

    print('\n.............................Pulling cases for {} ......................'.format(date_range[i]))

    locs_pulled = pd.DataFrame(columns=['time', 'margin'])

    attempts_tracker = []
    success_tracker = []

    successes = 0

    for i in range(attempts):

        attempts_tracker.append(i)
        success_tracker.append(successes)

        # Calculate percentage and create visual bar
        percentage = (i / attempts) * 100
        bar_length = 30
        filled_length = int(bar_length * (i / attempts))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        print(f"\r|{bar}| {percentage:.1f}% of {attempts} attempts - {successes} successes", end="", flush=True)

        ######################################## Random interval selection ########################################

        tot_window = random.choice(tot_windows)  # pick random window length from given range
        max_index = len(all_data_month) - int(tot_window)
        t_index = random.randint(0, max_index)

        start = all_data_month.index[t_index]
        end = start + datetime.timedelta(minutes=tot_window)

        cut = cut_time_series(all_data_month, start, end)

        encountered = False
        for i in range(len(locs_pulled)):

            time = locs_pulled.iloc[i]['time']
            margin = locs_pulled.iloc[i]['margin']

            bottom_margin = time - timedelta(minutes=int(tot_window))
            top_margin = time + timedelta(minutes=int(margin))

            if start > bottom_margin and start < top_margin:
                # print('Window already pulled successfully')
                encountered = True
                continue
        if encountered == True:
            continue

        ######################################## Filtering to window length ########################################

        cutoff = len(cut) / 4

        filtered_data = pd.DataFrame()
        filtered_data.index = cut.index

        ace_b = (cut['ace_bx'] ** 2 + cut['ace_by'] ** 2 + cut['ace_bz'] ** 2).pow(0.5)
        wind_b = (cut['wind_bx']**2 + cut['wind_by']**2 + cut['wind_bz']**2).pow(0.5)
        dsc_b = (cut['dsc_bx']**2 + cut['dsc_by']**2 + cut['dsc_bz']**2).pow(0.5)

        ace_v = (cut['ace_vx']**2 + cut['ace_vy']**2 + cut['ace_vz']**2).pow(0.5)
        wind_v = (cut['wind_vx']**2 + cut['wind_vy']**2 + cut['wind_vz']**2).pow(0.5)
        dsc_v = (cut['dsc_vx']**2 + cut['dsc_vy']**2 + cut['dsc_vz']**2).pow(0.5)

        filtered_data['ace_b'] = LP_filter_data(data=ace_b, cutoff1=cutoff, order=5)
        filtered_data['wind_b'] = LP_filter_data(data=wind_b, cutoff1=cutoff, order=5)
        filtered_data['dsc_b'] = LP_filter_data(data=dsc_b, cutoff1=cutoff, order=5)

        filtered_data['ace_v'] = LP_filter_data(data=ace_v, cutoff1=cutoff, order=5)
        filtered_data['wind_v'] = LP_filter_data(data=wind_v, cutoff1=cutoff, order=5)
        filtered_data['dsc_v'] = LP_filter_data(data=dsc_v, cutoff1=cutoff, order=5)


        ######################################## B-field Logistic Regression ########################################

        try:
            ace_r2, ace_amp, ace_vert, ace_rate, ace_t = log_reg_fit(filtered_data, 'ace_b', False)
        except:
            # print('Inital fit not possible - skipping')
            continue

        if ace_r2 < 0.75 or ace_rate < 0.1:
            # print('Intial fit not satisfied - skipping')
            continue

        def fixed_v_logistic(x, d):
            return (ace_amp / (1 + np.exp(ace_rate * (x - d)))) + ace_vert

        def fit_fixed_logistic(data, var, plot):
            x = np.arange(0, len(data))
            y = data[var].to_numpy()

            d0 = len(x) / 2

            params, cov = curve_fit(f=fixed_v_logistic, xdata=x, ydata=y, p0=[d0])

            ypred = fixed_v_logistic(x, *params)
            r2 = sklearn.metrics.r2_score(y, ypred)

            midpoint = params[0]

            if plot:
                plt.title('R2: {}, midpoint: {}'.format(round(r2, 2), round(midpoint, 2)))
                plt.plot(x, y)
                plt.plot(x, ypred)
                plt.show()

            return r2, midpoint

        try:
            wind_r2, wind_t = fit_fixed_logistic(data=filtered_data, var='wind_b', plot=False)
            dsc_r2, dsc_t = fit_fixed_logistic(data=filtered_data, var='dsc_b', plot=False)
        except:
            # print('Subsequent fits not possible - skipping')
            continue

        if wind_r2 < 0.7 or dsc_r2 < 0.7:
            # print('Subsequent fits not satisfied - skipping')
            continue

        times = {'ace': ace_t, 'wind': wind_t, 'dsc': dsc_t}

        ######################################## v-step correlation condition ########################################

        ace_r = np.corrcoef(filtered_data['ace_v'], filtered_data['ace_b'])[0, 1]
        wind_r = np.corrcoef(filtered_data['wind_v'], filtered_data['wind_b'])[0, 1]
        dsc_r = np.corrcoef(filtered_data['dsc_v'], filtered_data['dsc_b'])[0, 1]

        avg_r = (ace_r + wind_r + dsc_r) / 3

        if abs(avg_r) < 0.8:
            # print('v-step correlation not satisfied - skipping')
            continue

        ### ensure v steps up
        filtered_data['v_avg'] = filtered_data[['ace_v', 'wind_v', 'dsc_v']].mean(axis=1)
        try:
            ace_v_r2, ace_v_amp, ace_v_vert, ace_v_rate, ace_v_t = log_reg_fit(filtered_data, 'v_avg', False)
        except:
            continue
        if ace_v_rate*ace_v_amp >= 0:
            # print('v stepping down - not a shock')
            continue


        ### ensure v step >= 25 km/s at all three sats
        ace_dv = max(filtered_data['ace_v']) - min(filtered_data['ace_v'])
        wind_dv = max(filtered_data['wind_v']) - min(filtered_data['wind_v'])
        dsc_dv = max(filtered_data['dsc_v']) - min(filtered_data['dsc_v'])

        if ace_dv < 25 or dsc_dv < 25 or wind_dv < 25:
            # print('v step less than 25 km/s - skipping')
            continue

        if avg_r < 0:
            direction = 'fr'
        else:
            direction = 'ff'

        keys = list(times.keys())
        values = list(times.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}

        t_a = int(list(sorted_dict.values())[0])
        t_b = int(list(sorted_dict.values())[1])
        t_c = int(list(sorted_dict.values())[2])

        # ensure time lags at least 2 minutes apart
        if (t_b - t_a) < 1 or (t_c - t_b) < 1:
            # print('Lags too close - skipping')
            continue

        lag_ab = t_b - t_a
        lag_ac = t_c - t_a

        sat_order = list(sorted_dict.keys())
        sat_a = sat_order[0]
        sat_b = sat_order[1]
        sat_c = sat_order[2]

        # print(sat_order)
        # print(lag_ab, lag_ac)

        # validate w/ max-step func
        # lag_ab_corr, lag_ac_corr = calc_lags(filtered_data, sat_order[0], sat_order[1], sat_order[2], tot_window*2, plot=False)

       #  print(lag_ab_corr, lag_ac_corr)

        '''
        print('\nACE_B R2:', ace_r2, 'Wind_B R2:', wind_r2, 'DSC_B R2:', dsc_r2)
        print('ACE time', ace_t, 'Wind time', wind_t, 'DSC time', dsc_t)
        print('Direction:', direction)
        print('Lags:', lag_ab, lag_ac)
        print('Time:', start)
        print('Window:', tot_window)
        print('----------------------------------------')
        '''
        # plot_b(filtered_data, 'Triple-Satellite |B| and |v|', ace_amp, ace_vert, ace_rate, wind_t, ace_t, dsc_t)

        # print('time of successful case:', start)
        successes += 1

        success_tracker[-1] = successes

        locs_pulled.loc[len(locs_pulled)] = [start, tot_window]

        #################### Saving to dataframe ####################

        # columns=['time', 'window', 'direction', 'sat_a', 'sat_b', 'sat_c', 't_a', 'lag_ab', 'lag_ac'])
        cases.loc[len(cases)] = [start, tot_window, direction,
                                 sat_a, sat_b, sat_c,
                                 t_a, lag_ab, lag_ac]

print(len(cases))
joblib.dump(cases, 'vlim_all_cases.pkl')