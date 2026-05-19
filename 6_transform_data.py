import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from visualizers import plot_hemisphere_heatmap, flat_vect_proj
import datetime

all_data = joblib.load('all_data.pkl')

# given input shock velocities and orbital geometries, corrects via XGBoost model
def transform_data(data, model):
    # X_test = test[['orb_norm_x', 'orb_norm_y', 'orb_norm_z',
    #                'v_calc_x', 'v_calc_y', 'v_calc_z']]

    # columns=['start', 'tot_window', 'direction',
    #                                         'sat_a', 'sat_b', 'sat_c',
    #                                         't_a', 'lag_ab', 'lag_ac',
    #                                         'pos_a_x', 'pos_a_y', 'pos_a_z',
    #                                         'pos_b_x', 'pos_b_y', 'pos_b_z',
    #                                         'pos_c_x', 'pos_c_y', 'pos_c_z',
    #                                         'orb_norm_x', 'orb_norm_y', 'orb_norm_z',
    #                                         'v_calc_x', 'v_calc_y', 'v_calc_z'])

    input_data = data[['orb_norm_x', 'orb_norm_y', 'orb_norm_z',
                       'v_calc_x', 'v_calc_y', 'v_calc_z']]

    input_data_np = input_data.to_numpy()

    v_pred = model.predict(input_data_np)

    shock_pos_a = data[['pos_a_x', 'pos_a_y', 'pos_a_z']].to_numpy()
    shock_pos_b = data[['pos_b_x', 'pos_b_y', 'pos_b_z']].to_numpy()
    shock_pos_c = data[['pos_c_x', 'pos_c_y', 'pos_c_z']].to_numpy()
    lag_ab = data['lag_ab'].to_numpy()
    lag_ac = data['lag_ac'].to_numpy()

    # v_pred_proj = proj_feasible_set(shock_pos_a, shock_pos_b, shock_pos_c, lag_ab, lag_ac, v_pred)

    return v_pred

######################################## Earthward case ########################################

earthward_cases = joblib.load('vlim_earthward_props.pkl')
earthward_xgb = joblib.load('earthward_xgb_model.pkl')

earthward_v_pred = transform_data(earthward_cases, earthward_xgb)

######################################## Sunward case ########################################

sunward_cases = joblib.load('vlim_sunward_props.pkl')
sunward_xgb = joblib.load('sunward_xgb_model.pkl')

sunward_v_pred = transform_data(sunward_cases, sunward_xgb)

######################################## Combining and saving ########################################

v_pred = np.concatenate((earthward_v_pred, sunward_v_pred), axis=0)

all_cases = pd.concat([earthward_cases, sunward_cases], ignore_index=True)
all_cases['v_corrected'] = list(v_pred)

joblib.dump(all_cases, 'vlim_corrected_cases.pkl')
