# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:12:48 2025

@author: sbecker14
"""
# note this script requires the packages installed in an environment containing requirements2

from datetime import datetime
import os
import glob
import pandas as pd
import numpy as np
import re
from sklearn.neighbors import LocalOutlierFactor


# set up directory
stamp = datetime.now().strftime("%Y%m%d")
inDir = 'C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Code_for_Github'
outDir = os.path.normpath(inDir + os.sep + '\\FilteredSnowFreeData_output'+stamp) + '\\'   # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

os.chdir(inDir)

# load corrected bare neutron counts
file_directory_crns = os.path.normpath(inDir + os.sep + '\\mock_Bare_output20250812')
file_pattern_bare = f'{file_directory_crns}\\Bare*'
file_paths_bare = glob.glob(file_pattern_bare, recursive=True)

bare_names = [re.search(r'_(.*?)\.', os.path.basename(fp)).group(1) for fp in file_paths_bare] # get the text in between the _ and .
bare_names_clean = [s.replace(" ", "") for s in bare_names]
print(f'Bare detector site names are {bare_names}')
bare_names_dict = dict(zip(bare_names, file_paths_bare))

# load weighted TDR data (Includes corrected moderated counts as well)
TDR_dir  = "CombineDataWithFunction_output20250813"
TDR_pattern = f'{TDR_dir}/*.csv'
TDR_paths = glob.glob(TDR_pattern, recursive = True)
TDR_dfs = [pd.read_csv(file_path) for file_path in TDR_paths]
site_names = [os.path.basename(fp)[0:2] for fp in TDR_paths]
print(f'Site names are {site_names}')

TDR_df_dict = dict(zip(site_names, TDR_dfs))
print(TDR_df_dict.keys())

# load site-specific variables to calculate total water (bd, soc water, lattice water)
site_var = pd.read_excel("Data\\Mock Calibration Summary.xlsx")

# dataframe used to convert between site naming conventions
sitenames_df = pd.read_excel('Data\\Data_Release_2024\\Network_paper_site_names.xlsx')
name_dict = sitenames_df.set_index('network_paper_new_name')['Short_name'].to_dict()


# TDR and sample offset information: 

Cal_bias_df = pd.read_csv("Data\\Site_specific_N0_values.csv")


for s in site_names:
    THIS_SITE_new = s
   
    print(f'Starting {THIS_SITE_new}')
    
    # get old network name
    THIS_SITE_old =  name_dict[THIS_SITE_new]
    
    tdr_df = TDR_df_dict[THIS_SITE_new]
    
    # make sure date column is in datetime format
    
    tdr_df.loc[:,'date'] =  pd.to_datetime(tdr_df['Date (aggregated based on UTC)']).dt.date
    
    '''
    Index(['Date (aggregated based on UTC)', 'Raw_Moderated_cph',
           'Raw_Mod_Coeff_of_Var', 'Raw_Mod_sqrt', 'Corrected_Mod_cph_for_Des',
           'SWC_Des_cm3_cm3', 'Corrected_Mod_cph_for_UTS', 'SWC_UTS_cm3_cm3',
           'SWC_5', 'SWC_10', 'SWC_20', 'SWC_50', 'WeightedTDR_SWC',
           'WeightedTDR_SD', 'WeightedTDR_SE', 'sT_5', 'sT_10', 'sT_20', 'sT_50',
           'PRISM_ppt_mm', 'BaroPress', 'airRH', 'airT'],
          dtype='object')
    '''
    
    # filter out snow dates:
    # replace snow-covered dates with na
    # Define the date range for which you want to consider possible snow
    start_date = pd.Timestamp('2023-9-01').date()
    end_date = pd.Timestamp('2024-7-01').date()
    
    # find porosity 
    site_bd = site_var[THIS_SITE_old][24]
    
    site_por = 1 - (site_bd/2.65) # site porosity from particle density of 2.65 g/cm^3
    
    cal_bias = Cal_bias_df[Cal_bias_df['Site']==THIS_SITE_new]['GRAV_TDR_offset'].item()
    cal_bias_vwc = cal_bias*site_bd
    TDR_max = site_por - cal_bias_vwc
    print(f'porosity for {THIS_SITE_new} is {site_por}')
    
    # make sure CRNS_SWC_UTS_f3 is numeric
    tdr_df['CRNS'] = pd.to_numeric(tdr_df['SWC_UTS_cm3_cm3'])
    
    snow_cover = tdr_df[((tdr_df['SWC_UTS_cm3_cm3'] >= site_por-0.05) |
                         (tdr_df['SWC_Des_cm3_cm3'] >= site_por-0.05) |
                         (tdr_df['WeightedTDR_SWC'] >= TDR_max-0.1) |
                         (tdr_df['sT_5'] <= 3) |(tdr_df['sT_50'] <= 3))
                        & (tdr_df['date']>start_date) & (tdr_df['date']< end_date)]
    first_snow_cover = snow_cover['date'].min()
    last_snow_cover = snow_cover['date'].max()
    print(f'First snow cover date is: {first_snow_cover}')
    print(f'Last snow cover date is: {last_snow_cover}')
    
    # drop snow dates using saturation value for snow cover ID
    tdr_df_filt = tdr_df.drop(tdr_df[(tdr_df['date']>= first_snow_cover) & (tdr_df['date']<= last_snow_cover)].index)
    
    # filter moderated counts with min/max
    # step 1
    tdr_df_filt.drop(tdr_df_filt[(tdr_df_filt['Corrected_Mod_cph_for_Des']<1200) | (tdr_df_filt['Corrected_Mod_cph_for_Des']>6000)].index, inplace = True)
    
    # step 2
    mod_max = np.nanmean(tdr_df_filt['Corrected_Mod_cph_for_Des'])+ 2* np.nanstd(tdr_df_filt['Corrected_Mod_cph_for_Des'])
    mod_min = np.nanmean(tdr_df_filt['Corrected_Mod_cph_for_Des'])- 2* np.nanstd(tdr_df_filt['Corrected_Mod_cph_for_Des'])
    tdr_df_filt.drop(tdr_df_filt[(tdr_df_filt['Corrected_Mod_cph_for_Des']<mod_min) | (tdr_df_filt['Corrected_Mod_cph_for_Des']>mod_max)].index, inplace = True)
    tdr_df_filt.dropna(axis = 1, how = 'all', inplace = True) # drop nan columns first
    tdr_df_filt.dropna(subset = ['Corrected_Mod_cph_for_Des','WeightedTDR_SWC'], inplace = True)
    
    #ADD IN BARE COUNTS
    
    bare_df = pd.read_csv(bare_names_dict[THIS_SITE_old])
       
    # create date column and then filter out snow days
    bare_df['Datetime'] = pd.to_datetime(bare_df['datetime'], format = '%Y-%m-%d %H:%M:%S')
    # Step 1: Localize to UTC first (if it's originally UTC)
    bare_df['Datetime'] = bare_df['Datetime'].dt.tz_localize('UTC')
    # Step 2: Convert to Colorado (America/Denver)
    bare_df['Datetime'] = bare_df['Datetime'].dt.tz_convert('America/Denver')
       
    bare_df['date'] = bare_df['Datetime'].dt.date
    # drop snow covered dates
    bare_df= bare_df.drop(bare_df[(bare_df['date']>= first_snow_cover) & (bare_df['date']<= last_snow_cover)].index)
    
    # filter with min/max
    # step 1
    bare_df.drop(bare_df[(bare_df['bare_nc_cph']<600) | (bare_df['bare_nc_cph']>3000)].index, inplace = True)
    # step 2
    bare_max = np.nanmean(bare_df['bare_nc_cph'])+ 3* np.nanstd(bare_df['bare_nc_cph'])
    bare_min = np.nanmean(bare_df['bare_nc_cph'])- 3* np.nanstd(bare_df['bare_nc_cph'])
    bare_df.drop(bare_df[(bare_df['bare_nc_cph']<bare_min) | (bare_df['bare_nc_cph']>bare_max)].index, inplace = True)
    bare_df.dropna(inplace = True)
    
    daily_bare_series = bare_df.groupby(bare_df['date'])['bare_nc_cph'].mean()
    
    daily_bare_df = pd.concat([daily_bare_series], axis = 1, join = 'inner')
    daily_bare_df.columns = ['Bare']
    # make date the index of tdr_df_filt: 
    tdr_df_filt.index = tdr_df_filt['date']
    
    matching_dates = tdr_df_filt.index.intersection(daily_bare_df.index)
    print(f"Matching dates count: {len(matching_dates)}")
    print(f"tdr_df_filt unique dates: {tdr_df_filt.index.nunique()}")
    print(f"daily_bare_df unique dates: {daily_bare_df.index.nunique()}")
    
    tdr_df_join = pd.concat([tdr_df_filt, daily_bare_df], join = 'outer', axis = 1)
    
    
    # Identify outliers
    # select columns to consider
    
    sel_cols = ['Corrected_Mod_cph_for_Des',
           'SWC_Des_cm3_cm3',
           'Corrected_Mod_cph_for_UTS', 
           'SWC_UTS_cm3_cm3',
            'WeightedTDR_SWC', 
            'Bare']
    tdr_df_join_filt = tdr_df_join.dropna()
    find_outliers = tdr_df_join_filt[sel_cols]
    
    def estimate_contamination(df):
        """Estimate contamination using IQR method"""
        outlier_counts = []
        for col in find_outliers.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts.append(outliers)
        
        return np.mean(outlier_counts) / len(df)
    
    # Estimate contamination
    estimated_contamination = estimate_contamination(find_outliers)
    print(f"Estimated contamination: {estimated_contamination:.3f}")
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    outliers = lof.fit_predict(find_outliers)  # -1 indicates outliers
   
    #outlier dates: 
    outlier_rows = tdr_df_join_filt[outliers == -1]
    print(f'{THIS_SITE_new} outliers are: ')
    print(outlier_rows)
    
    rm_dates = outlier_rows['date']
    
    # remove outlier date that behaves strangely with precip and where Bare is already nan: 
    date1 = datetime.strptime('2024-08-15', '%Y-%m-%d').date()
    date2 = datetime.strptime('2024-08-14', '%Y-%m-%d').date()
    date3 = datetime.strptime('2023-10-12', '%Y-%m-%d').date()
    
    extra_dates = [date1, date2, date3]
    all_rm_dates = pd.Series(list(set(rm_dates.tolist() + extra_dates)))
    
    tdr_df_join = tdr_df_join[~tdr_df_join['date'].isin(all_rm_dates)]

    tdr_df_join.to_csv(f'{outDir}\\{THIS_SITE_old}_{THIS_SITE_new}_FilteredAndSnowFree_{stamp}.csv')
    
    print(f'Saved {THIS_SITE_new}. ')