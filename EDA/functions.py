import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time

import warnings
warnings.filterwarnings("ignore")

def exe_time(func, print_time=True):
    """
    This function returns execution time.
    """
    def new_func(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        if print_time:
            print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back

    return new_func

path = '/Users/parkerjiang/Documents/GitHub/Fall2022CapstoneProject/data/'

@exe_time
def preproc(household_df, timeseries_df, weather_df):
    household = household_df.copy()

    new_timeseries_cL = []
    for c in timeseries_df.columns:
        if c[:2] == 'DE' or c in ["utc_timestamp", "cet_cest_timestamp"]:
            new_timeseries_cL.append(c)
    timeseries = timeseries_df[new_timeseries_cL]

    new_weather_cL = []
    for c in weather_df.columns:
        if c[:2] == 'DE' or c in ["utc_timestamp"]:
            new_weather_cL.append(c)
    weather = weather_df[new_weather_cL]
    return household, timeseries, weather

def return_time(df):
    return pd.to_datetime(df["utc_timestamp"])

def get_year(df):
    return df.utc_timestamp.year


@exe_time
def clean_data(household_):
    household_['utc_timestamp'] = household_.apply(return_time, axis = 1)
    date_after = datetime.date(2016, 1, 1)
    date_before = datetime.date(2017, 2, 28)
    household = household_[(household_['utc_timestamp'].dt.date < date_before) &
                           (household_['utc_timestamp'].dt.date > date_after)]
    industrial1_cL = ['utc_timestamp']
    industrial2_cL = ['utc_timestamp']
    industrial3_cL = ['utc_timestamp']
    public1_cL = ['utc_timestamp']
    public2_cL = ['utc_timestamp']
    residential1_cL = ['utc_timestamp']
    residential2_cL = ['utc_timestamp']
    residential3_cL = ['utc_timestamp']
    residential4_cL = ['utc_timestamp']
    residential5_cL = ['utc_timestamp']
    residential6_cL = ['utc_timestamp']
    for c in household.columns:
        if 'industrial1' in c:
            industrial1_cL.append(c)
        if 'industrial2' in c:
            industrial2_cL.append(c)
        if 'industrial3' in c:
            industrial3_cL.append(c)
        if 'public1' in c:
            public1_cL.append(c)
        if 'public2' in c:
            public2_cL.append(c)
        if 'residential1' in c:
            residential1_cL.append(c)
        if 'residential2' in c:
            residential2_cL.append(c)
        if 'residential3' in c:
            residential3_cL.append(c)
        if 'residential4' in c:
            residential4_cL.append(c)
        if 'residential5' in c:
            residential5_cL.append(c)
        if 'residential6' in c:
            residential6_cL.append(c)
    industrial1 = household[industrial1_cL]
    industrial2 = household[industrial2_cL]
    industrial3 = household[industrial3_cL]
    public1 = household[public1_cL]
    public2 = household[public2_cL]
    residential1 = household[residential1_cL]
    residential2 = household[residential2_cL]
    residential3 = household[residential3_cL]
    residential4 = household[residential4_cL]
    residential5 = household[residential5_cL]
    residential6 = household[residential6_cL]

    dirty_household_dfL = [industrial1, industrial2, industrial3,
                          public1, public2,
                          residential1, residential2, residential3, residential4, residential5, residential6]
    nameL = ['industrial1', 'industrial2', 'industrial3',
            'public1', 'public2',
            'residential1', 'residential2', 'residential3', 'residential4', 'residential5', 'residential6']

    cum_household_dfL = []
    household_dfL = []
    for i,df in enumerate(dirty_household_dfL):
        clean_df = df.dropna(axis=0, subset=[c for c in df.columns if c not in ['utc_timestamp', 'cet_cest_timestamp']], how='all')
        # clean_df['utc_timestamp'] = clean_df.apply(return_time, axis = 1)
        clean_df['day'] = clean_df.utc_timestamp.dt.date
        clean_df = clean_df.drop('utc_timestamp', axis=1)
        cum_by_day = clean_df.groupby('day').max().reset_index()
        cum_by_day.name = nameL[i]

        ex_consL = [c for c in list(cum_by_day) if c != "day"
                    and 'grid_import' not in c and ('charge' not in c or 'decharge' in c)
                    and 'pv' not in c]
        im_genL = [c for c in list(cum_by_day) if c != "day"
                      and ('grid_import' in c or 'charge' in c or 'pv' in c)
                      and 'decharge' not in c]
        cum_by_day['export_consumption_sum'] = cum_by_day[ex_consL].sum(axis=1)
        cum_by_day['import_generation_sum'] = cum_by_day[im_genL].sum(axis=1) 
        
        clean_diff_df = clean_df.set_index('day').diff()
        diff_by_day = clean_diff_df.groupby('day').sum().reset_index()
        diff_by_day.name = nameL[i]
        
        diff_by_day['export_consumption_sum'] = diff_by_day[ex_consL].sum(axis=1)
        diff_by_day['import_generation_sum'] = diff_by_day[im_genL].sum(axis=1) 
        
        cum_household_dfL.append(cum_by_day)
        household_dfL.append(diff_by_day)
    
    return household_dfL, cum_household_dfL


def plot_line(df, metrics):
    date = df["day"]
    cm = plt.get_cmap('gist_rainbow')
    if len(metrics) == 0:
        return
    fig, ax = plt.subplots(figsize=(20, 6))
    color_map = [cm(1.*i/len(metrics)) for i in range(len(metrics))]
    for i,c in enumerate(metrics):
        value = df[c]
        ax.plot(date, value, color = color_map[i], label = c)
    ax.set_title("{} summary".format(df.name), fontsize = 15)
    ax.set_xlabel("timestamp", fontsize = 10)
    ax.set_ylabel("kWh", fontsize = 10)
    ax.legend(loc="upper left")
    plt.show()





