import joblib
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from matplotlib.dates import DateFormatter
import numpy as np
import math
from helper import LP_filter_data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def cut_time_series(data, start, end):
    data = data.loc[data.index <= end]
    data = data.loc[data.index >= start]

    return data

cases = joblib.load('vlim_corrected_cases.pkl')
cmes = joblib.load('cmes.pkl')
sirs = joblib.load('sirs.pkl')

####################################### MATCH CME-SIR ASSOCIATED EVENTS #######################################

cmes['arrivalTime'] = cmes['arrivalTime'].dt.tz_localize(None)
cmes = cmes.set_index('arrivalTime')

labels = []

for i in range(len(cases)):
    case = cases.iloc[i]

    # CME?
    start = case['start'] - datetime.timedelta(minutes=10)
    end = start + datetime.timedelta(minutes=int(case['tot_window']))
    cmes_window = cut_time_series(cmes, start, end)

    # SIR?
    sir = sirs.loc[sirs['SIR_start_time'] < case['start']]
    sir = sir.loc[sir['SIR_end time'] > case['start']]

    if len(cmes_window) > 0:
        label = 'CME'
    elif len(sir) > 0:
        label = 'SIR'
    else:
        label = 'Unclear'

    labels.append(label)

cases['type'] = labels

####################################### ANALYSIS #######################################

# cases = cases.loc[cases['type'] != 'Unclear']

ff = cases.loc[cases['direction'] == 'ff']
fr = cases.loc[cases['direction'] == 'fr']
cme = cases.loc[cases['type'] == 'CME']
sir = cases.loc[cases['type'] == 'SIR']

ff_cme = cme.loc[cme['direction'] == 'ff']
fr_cme = cme.loc[cme['direction'] == 'fr']

ff_sir = sir.loc[sir['direction'] == 'ff']
fr_sir = sir.loc[sir['direction'] == 'fr']

ff_event = ff.loc[(ff['type'] == 'CME') | (ff['type'] == 'SIR')]
fr_event = fr.loc[(fr['type'] == 'CME') | (fr['type'] == 'SIR')]

print('\n{}% of shock-associated CMEs have FF'.format(round(len(ff_cme)/len(cme)*100, 4)))
print('{}% of shock-associated CMEs have FR'.format(round(len(fr_cme)/len(cme)*100, 4)))

print('\n{}% of shock-associated SIRs have FF'.format(round(len(ff_sir)/len(sir)*100, 4)))
print('{}% of shock-associated SIRs have FR'.format(round(len(fr_sir)/len(sir)*100, 4)))

print('\n{}% of event-associated FFs are associated with CMEs'.format(round(len(ff_cme)/len(ff_event)*100, 4)))
print('{}% of event-associated FFs are associated with SIRs'.format(round(len(ff_sir)/len(ff_event)*100, 4)))

print('\n{}% of event-associated FRs are associated with CMEs'.format(round(len(fr_cme)/len(fr_event)*100, 4)))
print('{}% of event-associated FRs are associated with SIRs'.format(round(len(fr_sir)/len(fr_event)*100, 4)))
