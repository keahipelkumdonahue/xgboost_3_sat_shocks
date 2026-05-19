import pandas as pd
import joblib
from cdasws.datarepresentation import DataRepresentation as dr
from cdasws import CdasWs
from cdflib.xarray import cdf_to_xarray
cdas = CdasWs() # initiate
import matplotlib.pyplot as plt

# function from NOAA for downloading ACE/Wind data
def framebuilder(label, array, idx):
    f = {str(label): pd.Series(array[:], index=idx[:])}
    f = pd.DataFrame(f)
    f = f.sort_index()
    return f

# retrieve and format cdaweb dataset for time range
def get_and_process_data(starttime, endtime, quant, dataset, variable):  # retrieve data and put it into df
    start_time_str = starttime.strftime('%Y%m%dT%H:%M:00Z')
    end_time_str = endtime.strftime('%Y%m%dT%H:%M:00Z')

    retrieved_data = cdas.get_data(dataset, [variable], start_time_str, end_time_str, dataRepresentation=dr.XARRAY)[1]
    dim = len(retrieved_data.dims)

    if variable == 'B3GSE':
        time = retrieved_data.coords['Epoch3']  # Epoch3 is the time-key for Wind B data
    else:
        time = retrieved_data.coords['Epoch']

    if dim == 2:  # 3 components
        a = getattr(retrieved_data, variable)[:, 0].to_numpy()
        b = getattr(retrieved_data, variable)[:, 1].to_numpy()
        c = getattr(retrieved_data, variable)[:, 2].to_numpy()

        # Process each component with framebuilder
        A = framebuilder(quant, a, time)
        B = framebuilder(quant, b, time)
        C = framebuilder(quant, c, time)

        return A, B, C

    elif dim == 1:  # single dimension data
        a = getattr(retrieved_data, variable).to_numpy()
        A = framebuilder(quant, a, time)
        return A

def load_ace_int(start, end):
    ace_x, ace_y, ace_z = get_and_process_data(start, end, 'pos', 'AC_OR_SSC', 'XYZ_GSE')
    ace_bx, ace_by, ace_bz = get_and_process_data(start, end, 'B', 'AC_H3_MFI', 'BGSEc')
    ace_vx, ace_vy, ace_vz = get_and_process_data(start, end, 'v', 'AC_H0_SWE', 'V_GSE')

    ace_pos = pd.concat([ace_x, ace_y, ace_z], axis=1, ignore_index=True)
    ace_pos.rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=True)

    ace_b = pd.concat([ace_bx, ace_by, ace_bz], axis=1, ignore_index=True)
    ace_b.rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=True)

    ace_v = pd.concat([ace_vx, ace_vy, ace_vz], axis=1, ignore_index=True)
    ace_v.rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=True)

    return ace_pos, ace_b, ace_v

def load_wind_int(start, end):
    wind_x, wind_y, wind_z = get_and_process_data(start, end, 'pos', 'WI_OR_PRE', 'GSE_POS_t')
    wind_bx, wind_by, wind_bz = get_and_process_data(start, end, 'B', 'WI_H0_MFI', 'B3GSE')
    wind_vx, wind_vy, wind_vz = get_and_process_data(start, end, 'v', 'WI_PLSP_3DP', 'MOM.P.VELOCITY')

    wind_pos = pd.concat([wind_x, wind_y, wind_z], axis=1, ignore_index=True)
    wind_pos.rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=True)

    wind_b = pd.concat([wind_bx, wind_by, wind_bz], axis=1, ignore_index=True)
    wind_b.rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=True)

    wind_v = pd.concat([wind_vx, wind_vy, wind_vz], axis=1, ignore_index=True)
    wind_v.rename(columns={0: 'x', 1: 'y', 2: 'z'}, inplace=True)

    return wind_pos, wind_b, wind_v


# time parameters for data pull
start = '01-01-2017'
end = '01-01-2024'
date_range = pd.date_range(start=start, end=end, freq='MS')  # monthly pull windows

for i in range(len(date_range)):
    print('{}: {}'.format(i, date_range[i]))

############################################## GET DATA ##############################################

for i in range(len(date_range) - 1):
    print('loading data for', date_range[i], 'to', date_range[i + 1])
    data_int = []
    try:
        ace_data = load_ace_int(date_range[i], date_range[i+1])
        wind_data = load_wind_int(date_range[i], date_range[i+1])
    except:
        print('Gap in data - skipping', date_range[i], 'to', date_range[i + 1])
    else:
        ace_pos, ace_b, ace_v = ace_data
        wind_pos, wind_b, wind_v = wind_data

        joblib.dump(ace_pos,'raw_data/aw_monthly/{}_ace_pos.pkl'.format(i))
        joblib.dump(ace_b,'raw_data/aw_monthly/{}_ace_b.pkl'.format(i))
        joblib.dump(ace_v,'raw_data/aw_monthly/{}_ace_v.pkl'.format(i))

        joblib.dump(wind_pos,'raw_data/aw_monthly/{}_wind_pos.pkl'.format(i))
        joblib.dump(wind_b,'raw_data/aw_monthly/{}_wind_b.pkl'.format(i))
        joblib.dump(wind_v,'raw_data/aw_monthly/{}_wind_v.pkl'.format(i))


######################################### COMBINE MONTHLY DATA FILES AND SAVE #########################################

ace_pos = pd.DataFrame()
ace_b = pd.DataFrame()
ace_v = pd.DataFrame()

wind_pos = pd.DataFrame()
wind_b = pd.DataFrame()
wind_v = pd.DataFrame()

for i in range(84):
    ace_pos_mod = joblib.load('raw_data/aw_monthly/{}_ace_pos.pkl'.format(i))
    ace_b_mod = joblib.load('raw_data/aw_monthly/{}_ace_b.pkl'.format(i))
    ace_v_mod = joblib.load('raw_data/aw_monthly/{}_ace_v.pkl'.format(i))

    wind_pos_mod = joblib.load('raw_data/aw_monthly/{}_wind_pos.pkl'.format(i))
    wind_b_mod = joblib.load('raw_data/aw_monthly/{}_wind_b.pkl'.format(i))
    wind_v_mod = joblib.load('raw_data/aw_monthly/{}_wind_v.pkl'.format(i))

    if i == 0:
        ace_pos, ace_b, ace_v = ace_pos_mod, ace_b_mod, ace_v_mod
        wind_pos, wind_b, wind_v = wind_pos_mod, wind_b_mod, wind_v_mod
    else:
        ace_pos = pd.concat([ace_pos, ace_pos_mod])
        ace_b = pd.concat([ace_b, ace_b_mod])
        ace_v = pd.concat([ace_v, ace_v_mod])

        wind_pos = pd.concat([wind_pos, wind_pos_mod])
        wind_b = pd.concat([wind_b, wind_b_mod])
        wind_v = pd.concat([wind_v, wind_v_mod])

joblib.dump(ace_pos, 'raw_data/ace_pos.pkl')
joblib.dump(ace_b, 'raw_data/ace_b.pkl')
joblib.dump(ace_v, 'raw_data/ace_v.pkl')

joblib.dump(wind_pos, 'raw_data/wind_pos.pkl')
joblib.dump(wind_b, 'raw_data/wind_b.pkl')
joblib.dump(wind_v, 'raw_data/wind_v.pkl')