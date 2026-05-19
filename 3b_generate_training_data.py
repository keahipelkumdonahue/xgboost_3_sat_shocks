import joblib
import random
import datetime
import numpy as np
from typing import List, Tuple
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import math
import seaborn as sns
from visualizers import plot_hemisphere_heatmap

# calculate predicted arrival times of plane wave given inital position, velocity
def calculate_plane_motion(ace: np.ndarray, wind: np.ndarray, dsc: np.ndarray,
                           v: np.ndarray, start_point: str) -> Tuple[List[float], List[str]]:

    # Map string to actual points
    points = {'ace': ace, 'wind': wind, 'dsc': dsc}
    point_names = ['ace', 'wind', 'dsc']

    # Get starting point
    start_pos = points[start_point]

    # Calculate signed distances from start point along normal direction
    speed = np.linalg.norm(v)
    normal = v / speed

    distances = {}
    for name in point_names:
        if name != start_point:
            # Signed distance along normal direction
            distances[name] = np.dot(normal, points[name] - start_pos)

    # Calculate times (distance / speed)
    times = {}
    for name, dist in distances.items():
        times[name] = dist / speed

    # Sort by time to get order
    sorted_points = sorted(times.keys(), key=lambda x: times[x])
    sorted_times = [times[point] for point in sorted_points]

    return sorted_times, sorted_points

# starting with one satellite at random, send plane wave through orbits and record simluated detection times
def solve_random_start(ace: np.ndarray, wind: np.ndarray, dsc: np.ndarray, v: np.ndarray) -> dict:

    start_options = ['ace', 'wind', 'dsc']
    random_start = random.choice(start_options)

    pos_seq_found = False

    while pos_seq_found == False:
        times, order = calculate_plane_motion(ace, wind, dsc, v, random_start)

        has_negative = any(num < 0 for num in times)

        if has_negative:
            # print('negative time - selecting different starting satellite')
            random_start = random.choice(start_options)
        else:
            pos_seq_found = True

    return random_start, times, order

######################################## Testing synthetic plane dist ########################################

def rand_plane_earthward():
    vx = np.random.normal(loc=-385.1128877862596, scale=166.9922363980802)
    vy = np.random.normal(loc=-18.030785672342923, scale=160.01024587944588)
    vz = np.random.normal(loc=8.204893247210805, scale=165.25638025111013)

    while vx > 0:  # ensure earthward planes are actually earthward (not in sunward dist tail)
        vx = np.random.normal(loc=-385.1128877862596, scale=166.9922363980802)

    v = [vx, vy, vz]

    return np.array(v)

def rand_plane_sunward():
    vx = np.random.normal(loc=336.6440181944444, scale=167.4704796909948)
    vy = np.random.normal(loc=37.25733888888889, scale=138.70533575827326)
    vz = np.random.normal(loc=-0.22048993055555427, scale=147.34612839420467)

    while vx < 0:
        vx = np.random.normal(loc=336.6440181944444, scale=167.4704796909948)

    v = [vx, vy, vz]

    return np.array(v)

######################################## Generating velocity vectors ########################################

vs = []
speeds = []

for i in range(10000):
    v = rand_plane_sunward()  # change this line to pull from sunward vs earthward distribution

    vs.append(v)
    speeds.append(np.linalg.norm(v))

# vs = np.array(vs)
# # plt.hist(vs[:, 2], bins=100)
# plt.hist(speeds, bins=100)
# plt.show()
#
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
# plot_hemisphere_heatmap(vectors=vs, ax=ax, title='V-TRUE')
# plt.show()


######################################## Generating test cases ########################################


all_data = joblib.load('all_data.pkl')

cases = pd.DataFrame(columns=['t_a', 'lag_ab', 'lag_ac',
                              'pos_a_x', 'pos_a_y', 'pos_a_z',
                              'pos_b_x', 'pos_b_y', 'pos_b_z',
                              'pos_c_x', 'pos_c_y', 'pos_c_z',
                              'orb_norm_x', 'orb_norm_y', 'orb_norm_z',
                              'v_true_x', 'v_true_y', 'v_true_z',
                              'v_calc_x', 'v_calc_y', 'v_calc_z'])

for i in range(100000):

    ######################################## Pulling random orbital config ########################################

    max_index = len(all_data)
    t_index = random.randint(0, max_index-1)

    time = all_data.index[t_index]
    snapshot = all_data.loc[time]

    ace_pos = np.array([snapshot['ace_x'], snapshot['ace_y'], snapshot['ace_z']])
    wind_pos = np.array([snapshot['wind_x'], snapshot['wind_y'], snapshot['wind_z']])
    dsc_pos = np.array([snapshot['dsc_x'], snapshot['dsc_y'], snapshot['dsc_z']])

    ace_pos = ace_pos * 6378
    wind_pos = wind_pos * 6378
    dsc_pos = dsc_pos * 6378

    ######################################## Sending random plane wave ########################################

    v_true = rand_plane_sunward()

    start, times, order = solve_random_start(ace_pos, wind_pos, dsc_pos, v_true)

    # print(start)
    # print(times)
    # print(order)

    sat_a = start
    sat_b, sat_c = order[0], order[1]

    t_a = 0
    t_b, t_c = times[0], times[1]

    # print(sat_a, sat_b, sat_c)
    # print(t_a, t_b, t_c)

    ######################################## Back-calculating plane wave via lst squ ########################################

    lag_ab = t_b - t_a
    lag_ac = t_c - t_a

    a = np.array([snapshot[sat_a + '_x'], snapshot[sat_a + '_y'], snapshot[sat_a + '_z']])
    b = np.array([snapshot[sat_b + '_x'], snapshot[sat_b + '_y'], snapshot[sat_b + '_z']])
    c = np.array([snapshot[sat_c + '_x'], snapshot[sat_c + '_y'], snapshot[sat_c + '_z']])

    D = np.vstack((b - a, c - a))  # distances in Re
    t = np.array([lag_ab, lag_ac])  # time lags in seconds

    D = D * 6378  # km
    t = t  # sec

    # Solve for k*D=t coefficient (k=1/v)
    n, residuals, rank, s = np.linalg.lstsq(D, t, rcond=None)
    n_hat_calc = n / np.linalg.norm(n)

    try:
        v = 1. / float(np.linalg.norm(n))  # v in km/s
    except:
        # print('error in plane wave fit - skipping')
        continue
    v_calc = v * n_hat_calc

    orbit_normal = np.cross(b-a, c-a)

    orbit_normal = orbit_normal / np.linalg.norm(orbit_normal)
    # print('Orbit normal:', orbit_normal)

    # cases = pd.DataFrame(columns=['t_a', 'lag_ab', 'lag_ac',
    #                               'pos_a_x', 'pos_a_y', 'pos_a_z',
    #                               'pos_b_x', 'pos_b_y', 'pos_b_z',
    #                               'pos_c_x', 'pos_c_y', 'pos_c_z',
    #                               'orb_norm_x', 'orb_norm_y', 'orb_norm_z',
    #                               'v_true_x', 'v_true_y', 'v_true_z',
    #                               'v_calc_x', 'v_calc_y', 'v_calc_z'])
    case = [time, lag_ab, lag_ac,
            a[0], a[1], a[2],
            b[0], b[1], b[2],
            c[0], c[1], c[2],
            orbit_normal[0], orbit_normal[1], orbit_normal[2],
            v_true[0], v_true[1], v_true[2],
            v_calc[0], v_calc[1], v_calc[2]]
    # print(case)
    cases.loc[len(cases)] = case

    if i % 10000 == 0:
        print('Progress:', i/100000 * 100)

joblib.dump(cases, 'training_data/training_data_sunward.pkl')
