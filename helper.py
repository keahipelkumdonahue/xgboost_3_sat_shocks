import matplotlib.pyplot as plt
from scipy import signal

def LP_filter_data(data, cutoff1, order):
    '''this is a fix filter for this data set only. also only in pandas'''
    fs = (1 / 60)
    nyq = 0.5 * fs

    cutoff_freq1 = (1 / (cutoff1 * 60)) / nyq
    b_high, a_high = signal.butter(order, cutoff_freq1, 'low')
    LOW = signal.filtfilt(b_high, a_high, data)

    return LOW

def interpolate(df, resolution, verbose):

    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'Time'})
    df = df.drop_duplicates(subset=['Time'])  # drop duplicate rows (start or end)
    df.set_index('Time', inplace=True)  # set index to time

    if verbose:
        print('before interpolation:')
        print(df)

    df_resampled = df.resample(resolution)
    df_interp = df_resampled.interpolate(method='linear')

    if verbose:
        print('after interpolation:')
        print(df_interp)

        plt.plot(df, label='original')
        plt.plot(df_interp, label='interpolated')
        plt.legend()
        plt.show() # plots should be identical

    return df_interp

