import joblib
from helper import interpolate
import pandas as pd

######################################### GET DATA #########################################

ace_pos = joblib.load('raw_data/ace_pos.pkl')
ace_b = joblib.load('raw_data/ace_b.pkl')
ace_v = joblib.load('raw_data/ace_v.pkl')

wind_pos = joblib.load('raw_data/wind_pos.pkl')
wind_b = joblib.load('raw_data/wind_b.pkl')
wind_v = joblib.load('raw_data/wind_v.pkl')

dsc_pos = joblib.load('raw_data/dsc_pos.pkl')
dsc_b = joblib.load('raw_data/dsc_b.pkl')
dsc_v = joblib.load('raw_data/dsc_v.pkl')

######################################### REFORMATTING DSC DATA TO SAME FORMAT #########################################

dsc_pos = dsc_pos.sort_index()
dsc_b = dsc_b.sort_index()
dsc_v = dsc_v.sort_index()

dsc_pos.index.name = None
dsc_v.index.name = None
dsc_b.index.name = None

dsc_pos.columns = ['x', 'y', 'z']
dsc_b.columns = ['x', 'y', 'z']
dsc_v.columns = ['x', 'y', 'z']

######################################### CONVERTING POSITION TO RE #########################################

ace_pos = ace_pos.div(6378)
wind_pos = wind_pos.div(6378)
dsc_pos = dsc_pos.div(6378)


######################################### UNIFYING TO 1 MIN RESOLUTION #########################################

print('\n.................... ACE Pos ....................')

print('Before:', len(ace_pos))

ace_pos = interpolate(ace_pos, '1s', False)
ace_pos = ace_pos.loc[ace_pos.index.second == 30]

print('After:', len(ace_pos))


print('\n.................... ACE B ....................')

print('Before:', len(ace_b))

# print((len(ace_b) - len(ace_b.dropna()))/len(ace_b) * 100)

# Plotting lines for checking interpolation/resampling working as desired
# plt.plot(ace_b, label='Original (1 sec)', c='blue')

ace_b.index = pd.to_datetime(ace_b.index).round('1s')  # round off nanoseconds
ace_b = interpolate(ace_b, '1s', False)  # interpolate missing values
ace_b = ace_b.resample('min').mean()
ace_b = ace_b.shift(0.5, freq='min')
# plt.legend(loc='upper left')
# plt.grid()
# plt.title('ACE Bx data')
print('After:', len(ace_b))

# plt.show()

print('\n.................... ACE v ....................')

ace_v.index = pd.to_datetime(ace_v.index).round('1s')  # round off nanoseconds

print('Before:', len(ace_v))

ace_v = interpolate(ace_v, '1s', False)
ace_v = ace_v.loc[ace_v.index.second == 30]  # B data resampled at shifted to center of interval

print('After:', len(ace_v))


print('\n.................... Wind Pos ....................')

wind_pos.index = pd.to_datetime(wind_pos.index).round('1s')  # round off nanoseconds

print('Before:', len(wind_pos))

wind_pos = interpolate(wind_pos, '1s', False)
wind_pos = wind_pos.loc[wind_pos.index.second == 30]

print('After:', len(wind_pos))


print('\n.................... Wind B ....................')

print('Before:', len(wind_b))

wind_b.index = pd.to_datetime(wind_b.index).round('1s')  # round off nanoseconds
wind_b = interpolate(wind_b, '1s', False)  # interpolate up to 1-sec res
wind_b = wind_b.resample('min').mean()
wind_b = wind_b.shift(0.5, freq='min')

print('After:', len(wind_b))

print('\n.................... Wind v ....................')

wind_v.index = pd.to_datetime(wind_v.index).round('1s')  # round off nanoseconds

print('Before:', len(wind_v))

wind_v = interpolate(wind_v, '1s', False)
wind_v = wind_v.loc[wind_v.index.second == 30]

print('After:', len(wind_v))


print('\n.................... DSC Pos ....................')

dsc_pos.index = pd.to_datetime(dsc_pos.index).round('1s')  # round off nanoseconds

print('Before:', len(dsc_pos))

dsc_pos = interpolate(dsc_pos, '1s', False)
dsc_pos = dsc_pos.loc[dsc_pos.index.second == 30]

print('After:', len(dsc_pos))


print('\n.................... DSC B ....................')

print('Before:', len(dsc_b))

dsc_b.index = pd.to_datetime(dsc_b.index).round('1s')  # round off nanoseconds
dsc_b = interpolate(dsc_b, '1s', False)  # interpolate up to 1-sec res
dsc_b = dsc_b.resample('min').mean()
dsc_b = dsc_b.shift(0.5, freq='min')

print('After:', len(wind_b))


print('\n.................... DSC v ....................')

dsc_v.index = pd.to_datetime(dsc_v.index).round('1s')  # round off nanoseconds

print('Before:', len(dsc_v))

dsc_v = interpolate(dsc_v, '1s', False)
dsc_v = dsc_v.loc[dsc_v.index.second == 30]

print('After:', len(dsc_v))


######################################### COMBINING INTO DATAFRAME #########################################

print('ACE Pos:', ace_pos.index[0], 'to', ace_pos.index[-1])
print('ACE B:', ace_b.index[0], 'to', ace_b.index[-1])
print('ACE v:', ace_v.index[0], 'to', ace_v.index[-1])

print('Wind Pos:', wind_pos.index[0], 'to', wind_pos.index[-1])
print('Wind B:', wind_b.index[0], 'to', wind_b.index[-1])
print('Wind v:', wind_v.index[0], 'to', wind_v.index[-1])

print('DSC Pos:', dsc_pos.index[0], 'to', dsc_pos.index[-1])
print('DSC B:', dsc_b.index[0], 'to', dsc_b.index[-1])
print('DSC v:', dsc_v.index[0], 'to', dsc_v.index[-1])

starts = [ace_pos.index[0], ace_b.index[0], ace_v.index[0],
          wind_pos.index[0], wind_b.index[0], wind_v.index[0],
          dsc_pos.index[0], dsc_b.index[0], dsc_v.index[0]]
latest_start = max(starts)

print('Latest start:', latest_start)
ends = [ace_pos.index[-1], ace_b.index[-1], ace_v.index[-1],
        wind_pos.index[-1], wind_b.index[-1], wind_v.index[-1],
        dsc_pos.index[-1], dsc_b.index[-1], dsc_v.index[-1]]
earliest_end = min(ends)

print('Earliest end:', earliest_end)


############ Trimming ############

def trim(data):
    trimmed = data.loc[data.index >= latest_start]
    trimmed = trimmed.loc[trimmed.index <= earliest_end]
    return trimmed

ace_pos, ace_b, ace_v = trim(ace_pos), trim(ace_b), trim(ace_v)
wind_pos, wind_b, wind_v = trim(wind_pos), trim(wind_b), trim(wind_v)
dsc_pos, dsc_b, dsc_v = trim(dsc_pos), trim(dsc_b), trim(dsc_v)

print('\nAdjusted lengths:')
print('ACE Pos:', len(ace_pos))
print('ACE B:', len(ace_b))
print('ACE v:', len(ace_v))

print('Wind pos:', len(wind_pos))
print('Wind B:', len(wind_b))
print('Wind v:', len(wind_v))

print('DSC pos:', len(dsc_pos))
print('DSC B:', len(dsc_b))
print('DSC v:', len(dsc_v))

print(False in ace_pos.index == wind_b.index)  # check that all datetime indices are identical (Should print False)
print(False in ace_b.index == ace_v.index)

print(False in wind_pos.index == wind_b.index)
print(False in wind_b.index == wind_v.index)

print(False in dsc_pos.index == dsc_b.index)
print(False in dsc_b.index == dsc_v.index)

ace_x = ace_pos['x']
ace_y = ace_pos['y']
ace_z = ace_pos['z']

wind_x = wind_pos['x']
wind_y = wind_pos['y']
wind_z = wind_pos['z']

dsc_x = dsc_pos['x']
dsc_y = dsc_pos['y']
dsc_z = dsc_pos['z']

ace_bx = ace_b['x']
ace_by = ace_b['y']
ace_bz = ace_b['z']

wind_bx = wind_b['x']
wind_by = wind_b['y']
wind_bz = wind_b['z']

dsc_bx = dsc_b['x']
dsc_by = dsc_b['y']
dsc_bz = dsc_b['z']

ace_vx = ace_v['x']
ace_vy = ace_v['y']
ace_vz = ace_v['z']

wind_vx = wind_v['x']
wind_vy = wind_v['y']
wind_vz = wind_v['z']

dsc_vx = dsc_v['x']
dsc_vy = dsc_v['y']
dsc_vz = dsc_v['z']

all_data = pd.concat([ace_x, ace_y, ace_z, ace_bx, ace_by, ace_bz, ace_vx, ace_vy, ace_vz,
                      wind_x, wind_y, wind_z, wind_bx,wind_by, wind_bz, wind_vx, wind_vy, wind_vz,
                      dsc_x, dsc_y, dsc_z, dsc_bx, dsc_by, dsc_bz, dsc_vx, dsc_vy, dsc_vz], axis=1, ignore_index=True)
all_data.rename(columns={0:'ace_x', 1:'ace_y', 2:'ace_z',
                         3:'ace_bx', 4:'ace_by', 5:'ace_bz',
                         6:'ace_vx', 7:'ace_vy', 8:'ace_vz',
                         9:'wind_x', 10:'wind_y', 11:'wind_z',
                         12:'wind_bx', 13:'wind_by', 14:'wind_bz',
                         15:'wind_vx', 16:'wind_vy', 17:'wind_vz',
                         18:'dsc_x', 19:'dsc_y', 20:'dsc_z',
                         21:'dsc_bx', 22:'dsc_by', 23:'dsc_bz',
                         24:'dsc_vx', 25:'dsc_vy', 26:'dsc_vz'}, inplace=True)

print('length before dropna:', len(all_data))
all_data = all_data.dropna()
print('length after dropna:', len(all_data))

print(all_data)

######################################### SAVE #########################################

joblib.dump(all_data, 'all_data.pkl')