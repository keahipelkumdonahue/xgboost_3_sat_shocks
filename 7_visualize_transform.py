import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from visualizers import plot_hemisphere_heatmap, flat_vect_proj

pd.set_option('display.max_columns', None)

######################################### DATA #########################################

corrected_cases = joblib.load('vlim_corrected_cases.pkl')


v_obs = corrected_cases[['v_calc_x', 'v_calc_y', 'v_calc_z']].to_numpy()
v_pred = np.stack(corrected_cases['v_corrected'])

sunward_v_obs = v_obs[v_obs[:, 0] > 0]
earthward_v_obs = v_obs[v_obs[:, 0] <= 0]

sunward_v_pred = v_pred[v_pred[:, 0] > 0]
earthward_v_pred = v_pred[v_pred[:, 0] <= 0]

######################################### HEMISPHERE HEATMAPS #########################################

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), subplot_kw=dict(projection='polar'))
plot_hemisphere_heatmap(vectors=earthward_v_obs, ax=ax[0, 0], title='')
plot_hemisphere_heatmap(vectors=earthward_v_pred, ax=ax[0, 1], title='')
plot_hemisphere_heatmap(vectors=sunward_v_obs, ax=ax[1, 0], title='')
plot_hemisphere_heatmap(vectors=sunward_v_pred, ax=ax[1, 1], title='')
plt.tight_layout()
plt.savefig('4_xgb_results/yz_heatmaps.png', dpi=300, format='png')
plt.show()

######################################### SPEED HISTOGRAMS #########################################

# Speed histograms
fig = plt.figure(figsize=(8, 3))

speed_obs = np.linalg.norm(v_obs, axis=1)
speed_corrected = np.linalg.norm(v_pred, axis=1)

sns.kdeplot(speed_obs, label='Observed Shock Speed')
sns.kdeplot(speed_corrected, label='Model-Corrected Speed')
plt.xscale('log')
plt.xlabel('Shock Speed (km/s)')
plt.ylabel('Prob. Density')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid()
plt.savefig('4_xgb_results/speed_dist.png', dpi=300, format='png')
plt.show()

######################################### ECLIPTIC PROJECTIONS #########################################

fig = plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
flat_vect_proj(v_obs, '', 'xy')
plt.xlim([-2500, 2500])
plt.ylim([-5000, 3000])

plt.subplot(1, 2, 2)
flat_vect_proj(v_pred, '', 'xy')

plt.tight_layout()
plt.savefig('4_xgb_results/xy_proj.png', dpi=300, format='png')
plt.show()

######################################### XZ PROJECTIONS #########################################

fig = plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
flat_vect_proj(v_obs, '', 'xz')
plt.xlim([-2500, 2500])
plt.ylim([-5000, 3000])

plt.subplot(1, 2, 2)
flat_vect_proj(v_pred, '', 'xz')

plt.tight_layout()
plt.savefig('4_xgb_results/xz_proj.png', dpi=300, format='png')
plt.show()

######################################### ROTATING 3D SHOCK NORMALS PLOT #########################################

# '''
# #################################### 3D sphere plot ####################################
#
# normalize = True
#
# sunward = corrected_cases.loc[corrected_cases['v_corrected'].apply(lambda x: x[0]) > 0]
# earthward = corrected_cases.loc[corrected_cases['v_corrected'].apply(lambda x: x[0]) <= 0]
#
# sunward = np.array(sunward['v_corrected'].values.tolist())
# earthward = np.array(earthward['v_corrected'].values.tolist())
#
# if normalize:
#     sunward_magnitudes = np.linalg.norm(sunward, axis=1)
#     sunward = sunward / sunward_magnitudes[:, np.newaxis]
#
#     earthward_magnitudes = np.linalg.norm(earthward, axis=1)
#     earthward = earthward / earthward_magnitudes[:, np.newaxis]
#
# sunward_xs, sunward_ys, sunward_zs = sunward[:, 0], sunward[:, 1], sunward[:, 2]
# earthward_xs, earthward_ys, earthward_zs = earthward[:, 0], earthward[:, 1], earthward[:, 2]
#
# ax = plt.figure().add_subplot(projection='3d')
#
# ax.scatter(sunward_xs, sunward_ys, sunward_zs, c='red', label='Sunward', alpha=0.2)
# ax.scatter(earthward_xs, earthward_ys, earthward_zs, c='blue', label='Earthward', alpha=0.2)
# ax.quiver(0, 0, 0, -2, 0, 0, length=1, arrow_length_ratio=0.2, color='black', label='GSE -X')
# ax.set_xlabel('GSE X')
# ax.set_ylabel('GSE Y')
# ax.set_zlabel('GSE Z')
# plt.axis('equal')
# plt.legend()
#
#
# # Rotating graph visualization
# for angle in range(0, 360*10 + 1):  # for each angle in 10 rotations...
#     # Normalize the angle to the range [-180, 180] for display
#     angle_norm = (angle + 180) % 360 - 180
#
#     azim = roll = 0  # set azimuth and roll to 0
#     elev = 15  # set elevation to 15
#
#     if angle <= 360*10:  # if within the range to keep rotating, do so
#         azim = angle_norm
#
#     # Update the axis view and title
#     ax.view_init(elev, azim, roll)
#     # plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))
#
#     plt.draw()  # show current version
#     plt.pause(.001)  # wait before next frame
# '''
