import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.ma.extras import flatnotmasked_edges
from visualizers import plot_hemisphere_heatmap, flat_vect_proj
import seaborn as sns

################################### Load Earthward events for separated vis ###################################

direction = 'earthward'

xgb_model = joblib.load('{}_xgb_model.pkl'.format(direction))

X = joblib.load('training_data/test_data/{}_x.pkl'.format(direction))
y = joblib.load('training_data/test_data/{}_y.pkl'.format(direction))
y_pred = joblib.load('training_data/test_data/{}_y_pred.pkl'.format(direction))

X = X.to_numpy()
y = y.to_numpy()

v_obs_arr = []
v_pred_arr = []
v_true_arr = []

while len(v_obs_arr) < 1000:
    i = np.random.randint(0, len(X))
    row = X[i]

    v_ob = row[3:]
    v_true = y[i]
    v_pred = y_pred[i]

    v_obs_arr.append(v_ob)
    v_pred_arr.append(v_pred)
    v_true_arr.append(v_true)

v_obs_arr_E = np.array(v_obs_arr)
v_pred_arr_E = np.array(v_pred_arr)
v_true_arr_E = np.array(v_true_arr)


# Hemisphere heatmap
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), subplot_kw=dict(projection='polar'))
plot_hemisphere_heatmap(vectors=v_obs_arr_E, ax=ax[0], title='')
plot_hemisphere_heatmap(vectors=v_pred_arr_E, ax=ax[1], title='')
plot_hemisphere_heatmap(vectors=v_true_arr_E, ax=ax[2], title='')
plt.subplots_adjust(wspace=0.2)
plt.margins(x=0.2, y=0.2)
plt.tight_layout()
plt.savefig('3_xgb_test/xgb_test_E_yz.png', dpi=300, format='png')
plt.show()


################################### Load Sunward events for total-vis ###################################

direction = 'sunward'

xgb_model = joblib.load('{}_xgb_model.pkl'.format(direction))
# training_data = joblib.load('training_data/training_data_{}.pkl'.format(direction))

X = joblib.load('training_data/test_data/{}_x.pkl'.format(direction))
y = joblib.load('training_data/test_data/{}_y.pkl'.format(direction))
y_pred = joblib.load('training_data/test_data/{}_y_pred.pkl'.format(direction))

X = X.to_numpy()
y = y.to_numpy()

v_obs_arr = []
v_pred_arr = []
v_true_arr = []

while len(v_obs_arr) < 1000:
    i = np.random.randint(0, len(X))
    row = X[i]

    v_ob = row[3:]
    v_true = y[i]
    v_pred = y_pred[i]

    v_obs_arr.append(v_ob)
    v_pred_arr.append(v_pred)
    v_true_arr.append(v_true)

v_obs_arr_S = np.array(v_obs_arr)
v_pred_arr_S = np.array(v_pred_arr)
v_true_arr_S = np.array(v_true_arr)

# Combine earthward and sunward
v_obs_arr = np.concatenate((v_obs_arr_E, v_obs_arr_S))
v_pred_arr = np.concatenate((v_pred_arr_E, v_pred_arr_S))
v_true_arr = np.concatenate((v_true_arr_E, v_true_arr_S))

######################################### SPEED HISTORGRAMS #########################################

speed_obs = np.linalg.norm(v_obs_arr, axis=1)
speed_corrected = np.linalg.norm(v_pred_arr, axis=1)
speed_true = np.linalg.norm(v_true_arr, axis=1)

fig = plt.figure(figsize=(8, 3))
sns.kdeplot(speed_obs, label='Observed Shock Speed')
sns.kdeplot(speed_corrected, label='Model-Corrected Speed')
sns.kdeplot(speed_true, label='True Speed')
plt.xscale('log')
# plt.title('Shock Speed Distributions')
plt.xlabel('Shock Speed (km/s)')
plt.ylabel('Prob. Density')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid()
plt.savefig('3_xgb_test/xgb_test_speeds.png', dpi=300, format='png')
plt.show()

######################################### ECLIPTIC PROJECTIONS #########################################

fig = plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
flat_vect_proj(v_obs_arr, '', 'xy')
plt.xlim([-6000, 6000])
plt.ylim([-8000, 6000])

plt.subplot(1, 3, 2)
flat_vect_proj(v_pred_arr, '', 'xy')

plt.subplot(1, 3, 3)
flat_vect_proj(v_true_arr, '', 'xy')

plt.tight_layout()
plt.savefig('3_xgb_test/xgb_test_xy.png', dpi=300, format='png')
plt.show()

######################################### XZ PROJECTIONS #########################################

fig = plt.figure(figsize=(8, 4))

plt.subplot(1, 3, 1)
flat_vect_proj(v_obs_arr, '', 'xz')
plt.xlim([-6000, 6000])
plt.ylim([-8000, 6000])

plt.subplot(1, 3, 2)
flat_vect_proj(v_pred_arr, '', 'xz')

plt.subplot(1, 3, 3)
flat_vect_proj(v_true_arr, '', 'xz')

plt.tight_layout()
plt.savefig('3_xgb_test/xgb_test_xz.png', dpi=300, format='png')
plt.show()
