import requests
import pandas as pd
import joblib

######################################### LOAD CMES #########################################

# request the data
url = "https://kauai.ccmc.gsfc.nasa.gov/CMEscoreboard/WS/get/predictions"

# if there's no response at this time, print warning
response = requests.get(url)
if response.status_code != 200:
    print('cannot successfully get an http response')

# read the data
print("Getting data from", url)
df = pd.read_json(url)

df['arrivalTime'] = pd.to_datetime(df['arrivalTime'], utc=True)
df = df.loc[df['noArrivalObserved'] == False]

df = df.drop(columns=['noArrivalObserved', 'observedTime', 'maxKP', 'dstMin', 'dstMinTime', 'cmeNote', 'predictions'])
cmes = df[::-1].reset_index(drop=True)

joblib.dump(cmes, 'cmes.pkl')

######################################### LOAD SIRS #########################################

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

sirs = pd.read_csv('UCLA_SIRS_list.txt', sep='\t') # downloaded SIR list in txt format

sirs = sirs.drop(columns=['hybrid_event_marker', 'ambiguous_event_marker', 'STEREO', 'Ptmax', 'Bmax', 'Npmax', 'Vmin', 'Vmax'])

datetime_columns = ['SIR_start_time', 'SIR_end time', 'Ptmax_time']

for col in datetime_columns:
    if col in sirs.columns:
        sirs[col] = pd.to_datetime(sirs[col], format='%Y-%m-%dT%H:%M:%S.%fZ')

print(sirs)

joblib.dump(sirs, 'sirs.pkl')

