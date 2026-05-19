import joblib
import wget
import gzip
import os
import shutil
import pandas as pd
from netCDF4 import Dataset
import xarray
from dsc_urls import get_urls  # method containing all necessary DSCOVR wget urls
import joblib

def gz_extract(directory):
    extension = ".gz"
    os.chdir(directory)
    for item in os.listdir(directory):  # loop through items in dir
        if item.endswith(extension):  # check for ".gz" extension
            gz_name = os.path.abspath(item)  # get full path of files
            file_name = (os.path.basename(gz_name)).rsplit('.', 1)[0]  # get file name for file within
            with gzip.open(gz_name, "rb") as f_in, open(file_name, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_name)  # delete zipped file

urls = get_urls()
urls_list = urls.split()
dir = 'raw_data/dsc_daily'

############################################ Download and unzip ############################################

for url in urls_list:
    filename = wget.download(url, dir)

with gzip.open(filename, 'rb') as f_in:
    decompressed_file_content = f_in.read()

gz_extract(dir)

############################################ Convert to df ############################################

dsc_pos = pd.DataFrame()
dsc_b = pd.DataFrame()
dsc_v = pd.DataFrame()

os.chdir(dir)
for item in os.listdir(dir):
    if item.endswith('.nc'):  # check for extension
        ds = xarray.open_dataset(dir+'/'+item)

        if 'pop' in item:
            df = ds[['time', 'sat_x_gse', 'sat_y_gse', 'sat_z_gse']].to_dataframe()
            dsc_pos = dsc_pos._append(df)
        elif 'm1s' in item:
            df = ds[['time', 'bx_gse', 'by_gse', 'bz_gse']].to_dataframe()
            dsc_b = dsc_b._append(df)
        else:
            df = ds[['time', 'proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse']].to_dataframe()
            dsc_v = dsc_v._append(df)

joblib.dump(dsc_pos, 'raw_data/dsc_pos.pkl')
joblib.dump(dsc_b, 'raw_data/dsc_b.pkl')
joblib.dump(dsc_v, 'raw_data/dsc_v.pkl')
