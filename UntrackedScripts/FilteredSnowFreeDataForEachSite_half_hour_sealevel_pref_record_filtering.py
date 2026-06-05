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
inDir = 'C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO'
outDir = os.path.normpath(inDir + os.sep + '\\FilteredSnowFreeData_output_sealevel_pref'+stamp) + '\\'   # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory
logDir = os.path.join(outDir, 'filter_logs')
if not os.path.exists(logDir): os.makedirs(logDir)

os.chdir(inDir)

# load corrected bare neutron counts
file_directory_crns = os.path.normpath(inDir + os.sep + '\\mock_Bare_output20260210')
file_pattern_bare = f'{file_directory_crns}\\Bare*'
file_paths_bare = glob.glob(file_pattern_bare, recursive=True)

bare_names = [re.search(r'_(.*?)\.', os.path.basename(fp)).group(1) for fp in file_paths_bare] # get the text in between the _ and .
bare_names_clean = [s.replace(" ", "") for s in bare_names]
print(f'Bare detector site names are {bare_names}')
bare_names_dict = dict(zip(bare_names, file_paths_bare))

# load weighted TDR data (Includes corrected moderated counts as well)
TDR_dir  = "CombineDataWithFunction_half_hr_output_sealevel_pref_20260210"
TDR_pattern = f'{TDR_dir}/*.csv'
TDR_paths = glob.glob(TDR_pattern, recursive = True)
TDR_dfs = [pd.read_csv(file_path) for file_path in TDR_paths]
site_names_old = [os.path.basename(fp).split('_')[0] for fp in TDR_paths]
print(f'Old site names are {site_names_old}')
site_names_new = [os.path.basename(fp).split('_')[1] for fp in TDR_paths]
print(f'New site names are {site_names_new}')


TDR_df_dict = dict(zip(site_names_new, TDR_dfs))
print(TDR_df_dict.keys())

# load site-specific variables to calculate total water (bd, soc water, lattice water)
site_var = pd.read_excel("Data\\Mock Calibration Summary_20251106.xlsx")

# dataframe used to convert between site naming conventions
sitenames_df = pd.read_excel('Data\\Data_Release_2024_b\\Network_paper_site_names.xlsx')
name_dict = sitenames_df.set_index('network_paper_new_name')['Short_name'].to_dict()


# TDR and sample offset information: 

Cal_bias_df = pd.read_csv("Data\\Site_specific_N0_values.csv")

# store all filter decisions here
filter_records = []

def normalize_timestamp(ts):
    """Convert timestamps to tz-naive UTC (or keep naive) for consistent sorting."""
    if pd.isna(ts):
        return pd.NaT
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            return ts.tz_convert('UTC').tz_localize(None)
        return ts
    try:
        ts_parsed = pd.to_datetime(ts, errors='coerce')
        if pd.isna(ts_parsed):
            return pd.NaT
        if ts_parsed.tzinfo is not None:
            return ts_parsed.tz_convert('UTC').tz_localize(None)
        return ts_parsed
    except Exception:
        return pd.NaT

for s in site_names_new:
    THIS_SITE_new = s
   
    print(f'Starting {THIS_SITE_new}')
    # marker to split log per site
    start_len = len(filter_records)

    def add_logs(df, reason, detail, ts_col='DateTime', date_col='date'):
        """Append rows to filter_records with site info and normalized timestamp/date"""
        if df is None or df.empty:
            return
        for _, r in df.iterrows():
            ts_val = r[ts_col] if ts_col in r else pd.NaT
            ts_norm = normalize_timestamp(ts_val)
            date_val = r[date_col] if date_col in r else pd.NaT
            if isinstance(date_val, pd.Timestamp):
                date_val = date_val.date()
            elif pd.isna(date_val) and not pd.isna(ts_norm):
                date_val = ts_norm.date()
            filter_records.append({
                'site_old': THIS_SITE_old,
                'site_new': THIS_SITE_new,
                'timestamp': ts_norm,
                'date': date_val,
                'reason': reason,
                'detail': detail
            })
    
    # get old network name
    THIS_SITE_old =  name_dict[THIS_SITE_new]
    
    tdr_df = TDR_df_dict[THIS_SITE_new]
    
    # make sure date column is in datetime format (be defensive about column name/contents)
    if 'DateTime' in tdr_df.columns:
        dt_series = pd.to_datetime(tdr_df['DateTime'], errors='coerce')
    elif 'Date (aggregated based on UTC)' in tdr_df.columns:
        dt_series = pd.to_datetime(tdr_df['Date (aggregated based on UTC)'], errors='coerce')
    else:
        raise KeyError("No datetime column found (expected 'DateTime' or 'Date (aggregated based on UTC)').")

    tdr_df.loc[:, 'DateTime'] = dt_series
    tdr_df.loc[:,'date'] =  dt_series.dt.date
    
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
    
    # track individual reasons before we drop anything
    temp_mask = ((tdr_df['sT_5'] <= 3) | (tdr_df['sT_50'] <= 3)) & (tdr_df['date']>start_date) & (tdr_df['date']< end_date)
    water_mask = (((tdr_df['SWC_UTS_cm3_cm3'] >= site_por-0.05) |
                   (tdr_df['SWC_Des_cm3_cm3'] >= site_por-0.05) |
                   (tdr_df['WeightedTDR_SWC'] >= TDR_max-0.1))
                  & (tdr_df['date']>start_date) & (tdr_df['date']< end_date))

    # log temp-based removals (half-hour rows)
    add_logs(tdr_df.loc[temp_mask, ['DateTime','date']], 'temperature', 'Soil temperature <=3C')

    # log water-content-based removals (half-hour rows)
    add_logs(tdr_df.loc[water_mask, ['DateTime','date']], 'water_content_high', 'Soil water near saturation in possible snow months')

    snow_cover = tdr_df[(temp_mask | water_mask)]
    # make exception for site R1 because calibration date is 10/18/23
    if (THIS_SITE_new == 'R1') and (snow_cover['date'].min() < pd.Timestamp('2023-10-19').date()):
        first_snow_cover = pd.Timestamp('2023-10-19').date()
    else:
        first_snow_cover = snow_cover['date'].min()
       
    last_snow_cover = snow_cover['date'].max()
    print(f'First snow cover date is: {first_snow_cover}')
    print(f'Last snow cover date is: {last_snow_cover}')
    
    # drop snow dates using saturation value for snow cover ID
    snow_range_mask = (tdr_df['date']>= first_snow_cover) & (tdr_df['date']<= last_snow_cover)
    # log all rows removed by the snow window
    add_logs(tdr_df.loc[snow_range_mask, ['DateTime','date']], 'winter_window', f'{first_snow_cover} to {last_snow_cover}')

    tdr_df_filt = tdr_df.drop(tdr_df[snow_range_mask].index)
    
    # filter moderated counts with min/max
    # step 1
    mod_mask_step1 = (tdr_df_filt['Corrected_Mod_cph_for_Des']<1200) | (tdr_df_filt['Corrected_Mod_cph_for_Des']>6000)
    add_logs(tdr_df_filt.loc[mod_mask_step1, ['DateTime','date']], 'mod_counts_minmax', 'Des moderated counts outside 1200-6000')
    tdr_df_filt.drop(tdr_df_filt[mod_mask_step1].index, inplace = True)
    
    # step 2
    mod_max = np.nanmean(tdr_df_filt['Corrected_Mod_cph_for_Des'])+ 2* np.nanstd(tdr_df_filt['Corrected_Mod_cph_for_Des'])
    mod_min = np.nanmean(tdr_df_filt['Corrected_Mod_cph_for_Des'])- 2* np.nanstd(tdr_df_filt['Corrected_Mod_cph_for_Des'])
    mod_mask_step2 = (tdr_df_filt['Corrected_Mod_cph_for_Des']<mod_min) | (tdr_df_filt['Corrected_Mod_cph_for_Des']>mod_max)
    add_logs(tdr_df_filt.loc[mod_mask_step2, ['DateTime','date']], 'mod_counts_sd', f'Des moderated counts outside {mod_min:.1f}-{mod_max:.1f}')
    tdr_df_filt.drop(tdr_df_filt[mod_mask_step2].index, inplace = True)
    tdr_df_filt.dropna(axis = 1, how = 'all', inplace = True) # drop nan columns first
    tdr_df_filt.dropna(subset = ['Corrected_Mod_cph_for_Des'], inplace = True) #don't drop rows with nan in TDR data yet
    
    #ADD IN BARE COUNTS
    
    bare_df = pd.read_csv(bare_names_dict[THIS_SITE_old])
       
    # create date column and then filter out snow days
    bare_df['Datetime'] = pd.to_datetime(bare_df['datetime'], format = '%Y-%m-%d %H:%M:%S')
    # Step 1: Localize to UTC first (if it's originally UTC)
    bare_df['Datetime'] = bare_df['Datetime'].dt.tz_localize('UTC')
    # Step 2: Convert to Colorado (America/Denver)
    bare_df['Datetime'] = bare_df['Datetime'].dt.tz_convert('America/Denver')
       
    bare_df['date'] = bare_df['Datetime'].dt.date
    # basic stats before filtering
    print(
        f"Bare stats before filter for {THIS_SITE_new}: "
        f"min={bare_df['bare_nc_cph'].min():.2f}, "
        f"max={bare_df['bare_nc_cph'].max():.2f}, "
        f"median={bare_df['bare_nc_cph'].median():.2f}, "
        f"std={bare_df['bare_nc_cph'].std():.2f}"
    )
   
    # drop snow covered dates
    bare_df= bare_df.drop(bare_df[(bare_df['date']>= first_snow_cover) & (bare_df['date']<= last_snow_cover)].index)
    
    # filter with min/max
    # step 1
    bare_mask_step1 = (bare_df['bare_nc_cph']<500) | (bare_df['bare_nc_cph']>20000)
    add_logs(bare_df.loc[bare_mask_step1, ['Datetime','date']].rename(columns={'Datetime':'DateTime'}), 'bare_counts_minmax', 'Bare counts outside 500-20000', ts_col='DateTime')
    bare_df.drop(bare_df[bare_mask_step1].index, inplace = True)
    # step 2
    bare_max = np.nanmean(bare_df['bare_nc_cph'])+ 3* np.nanstd(bare_df['bare_nc_cph'])
    bare_min = np.nanmean(bare_df['bare_nc_cph'])- 3* np.nanstd(bare_df['bare_nc_cph'])
    bare_mask_step2 = (bare_df['bare_nc_cph']<bare_min) | (bare_df['bare_nc_cph']>bare_max)
    add_logs(bare_df.loc[bare_mask_step2, ['Datetime','date']].rename(columns={'Datetime':'DateTime'}), 'bare_counts_sd', f'Bare counts outside {bare_min:.1f}-{bare_max:.1f}', ts_col='DateTime')
    bare_df.drop(bare_df[bare_mask_step2].index, inplace = True)
    bare_df.dropna(inplace = True)
    
    # basic stats before filtering
    print(
        f"Bare stats after filter for {THIS_SITE_new}: "
        f"min={bare_df['bare_nc_cph'].min():.2f}, "
        f"max={bare_df['bare_nc_cph'].max():.2f}, "
        f"median={bare_df['bare_nc_cph'].median():.2f}, "
        f"std={bare_df['bare_nc_cph'].std():.2f}"
    )
    
    daily_bare_series = bare_df.groupby(bare_df['date'])['bare_nc_cph'].mean()
    
    daily_bare_df = pd.concat([daily_bare_series], axis = 1, join = 'inner')
    daily_bare_df.columns = ['Bare']
    # merge bare counts onto half-hour data by date (allowing repeated dates in tdr_df_filt)
    matching_dates = set(tdr_df_filt['date']).intersection(set(daily_bare_df.index))
    print(f"Matching dates count: {len(matching_dates)}")
    print(f"tdr_df_filt unique dates: {tdr_df_filt['date'].nunique()}")
    print(f"daily_bare_df unique dates: {daily_bare_df.index.nunique()}")
    
    tdr_df_join = pd.merge(tdr_df_filt, daily_bare_df, left_on='date', right_index=True, how='left')
    
    
    # Identify outliers
    # select columns to consider
    
    sel_cols = ['Corrected_Mod_cph_for_Des',
           'SWC_Des_cm3_cm3',
           'Corrected_Mod_cph_for_UTS', 
           'SWC_UTS_cm3_cm3',
            'WeightedTDR_SWC', 'Bare'
            ] # removed bare
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
    rm_dates = set(outlier_rows['date'].tolist())
    
    # remove outlier date that behaves strangely with precip and where Bare is already nan: 
    date1 = datetime.strptime('2024-08-15', '%Y-%m-%d').date()
    date2 = datetime.strptime('2024-08-14', '%Y-%m-%d').date()
    date3 = datetime.strptime('2023-10-12', '%Y-%m-%d').date()
    
    manual_dates = {date1, date2, date3}
    all_rm_dates = sorted(list(rm_dates.union(manual_dates)))

    # log all rows removed because of outlier decisions (half-hour level)
    rows_removed_outlier = tdr_df_join_filt[tdr_df_join_filt['date'].isin(all_rm_dates)]
    for _, r in rows_removed_outlier.iterrows():
        d = r['date']
        if d in rm_dates and d in manual_dates:
            reason = 'outlier_lof_manual'
        elif d in rm_dates:
            reason = 'outlier_lof'
        else:
            reason = 'outlier_manual'
        add_logs(pd.DataFrame([{'DateTime': r['DateTime'], 'date': d}]), reason, 'Removed entire date due to outlier screening')
    
    tdr_df_join = tdr_df_join_filt[~tdr_df_join_filt['date'].isin(all_rm_dates)]

    tdr_df_join.to_csv(f'{outDir}\\{THIS_SITE_old}_{THIS_SITE_new}_FilteredAndSnowFree_half_hr_{stamp}.csv')
    
    # save site-specific filter log
    site_log_df = pd.DataFrame(filter_records[start_len:])
    # sort for readability (timestamp is half-hour resolution)
    if not site_log_df.empty and 'date' in site_log_df.columns:
        site_log_df['date_sort'] = pd.to_datetime(site_log_df['date'], errors='coerce')
        if 'timestamp' in site_log_df.columns:
            site_log_df['timestamp'] = site_log_df['timestamp'].apply(normalize_timestamp)
            site_log_df['timestamp_sort'] = pd.to_datetime(site_log_df['timestamp'], utc=True, errors='coerce').dt.tz_convert(None)
            site_log_df = site_log_df.sort_values(['date_sort','timestamp_sort'])
        else:
            site_log_df = site_log_df.sort_values(['date_sort'])
        site_log_df.drop(columns=['date_sort','timestamp_sort'], errors='ignore', inplace=True)
    site_log_path = f'{logDir}\\{THIS_SITE_old}_{THIS_SITE_new}_FilterLog_{stamp}.csv'
    if not site_log_df.empty:
        site_log_df.to_csv(site_log_path, index=False)
        print(f'Saved filter log for {THIS_SITE_new} to {site_log_path}')
    else:
        print(f'No filtered rows logged for {THIS_SITE_new}')

    # save temperature-only log for this site
    site_temp_df = site_log_df[site_log_df['reason'] == 'temperature'][['site_old','site_new','date','reason','detail']].drop_duplicates().sort_values(['site_new','date']) if not site_log_df.empty else pd.DataFrame()
    temp_log_path = f'{logDir}\\{THIS_SITE_old}_{THIS_SITE_new}_TempFilterLog_{stamp}.csv'
    if not site_temp_df.empty:
        site_temp_df.to_csv(temp_log_path, index=False)
        print(f'Saved temperature-only log for {THIS_SITE_new} to {temp_log_path}')
    else:
        print(f'No temperature filters logged for {THIS_SITE_new}')

    # save environmental (temp/water/snow) log for this site (date-level, priority temp>water>winter/snow)
    env_reasons = ['temperature','water_content_high','winter_window','snow_window']
    if not site_log_df.empty:
        site_env_df = site_log_df[site_log_df['reason'].isin(env_reasons)][['site_old','site_new','date','reason','detail']].copy()
        priority = {'temperature':0,'water_content_high':1,'winter_window':2,'snow_window':2}
        site_env_df['priority'] = site_env_df['reason'].map(priority).fillna(99)
        site_env_df = site_env_df.sort_values(['site_new','date','priority'])
        site_env_df = site_env_df.drop_duplicates(subset=['site_old','site_new','date'], keep='first')
        site_env_df = site_env_df.drop(columns=['priority'])
        site_env_df = site_env_df.sort_values(['site_new','date'])
    else:
        site_env_df = pd.DataFrame()
    env_log_path = f'{logDir}\\{THIS_SITE_old}_{THIS_SITE_new}_EnvFilterLog_{stamp}.csv'
    if not site_env_df.empty:
        site_env_df.to_csv(env_log_path, index=False)
        print(f'Saved env-filter log for {THIS_SITE_new} to {env_log_path}')
    else:
        print(f'No env filters logged for {THIS_SITE_new}')

    print(f'Saved {THIS_SITE_new}. ')

# save combined filter log for all sites
if filter_records:
    all_logs_df = pd.DataFrame(filter_records)
    all_logs_df['date_sort'] = pd.to_datetime(all_logs_df['date'], errors='coerce')
    if 'timestamp' in all_logs_df.columns:
        all_logs_df['timestamp'] = all_logs_df['timestamp'].apply(normalize_timestamp)
        all_logs_df['timestamp_sort'] = pd.to_datetime(all_logs_df['timestamp'], utc=True, errors='coerce').dt.tz_convert(None)
        all_logs_df = all_logs_df.sort_values(['site_new','date_sort','timestamp_sort'])
    else:
        all_logs_df = all_logs_df.sort_values(['site_new','date_sort'])
    all_logs_df.drop(columns=['date_sort','timestamp_sort'], errors='ignore', inplace=True)
    all_log_path = f'{logDir}\\AllSites_FilterLog_half_hr_{stamp}.csv'
    all_logs_df.to_csv(all_log_path, index=False)
    print(f'Saved combined filter log to {all_log_path}')
    # combined temperature-only log
    temp_all = all_logs_df[all_logs_df['reason'] == 'temperature'][['site_old','site_new','date','reason','detail']].drop_duplicates().sort_values(['site_new','date'])
    if not temp_all.empty:
        temp_all_log_path = f'{logDir}\\AllSites_TempFilterLog_{stamp}.csv'
        temp_all.to_csv(temp_all_log_path, index=False)
        print(f'Saved combined temperature-only log to {temp_all_log_path}')
    # combined environmental log (temp/water/snow) with priority temp>water>winter/snow
    env_reasons = ['temperature','water_content_high','winter_window','snow_window']
    env_all = all_logs_df[all_logs_df['reason'].isin(env_reasons)][['site_old','site_new','date','reason','detail']].copy()
    if not env_all.empty:
        priority = {'temperature':0,'water_content_high':1,'winter_window':2,'snow_window':2}
        env_all['priority'] = env_all['reason'].map(priority).fillna(99)
        env_all = env_all.sort_values(['site_new','date','priority'])
        env_all = env_all.drop_duplicates(subset=['site_old','site_new','date'], keep='first')
        env_all = env_all.drop(columns=['priority'])
        env_all = env_all.sort_values(['site_new','date'])
    if not env_all.empty:
        env_all_log_path = f'{logDir}\\AllSites_EnvFilterLog_{stamp}.csv'
        env_all.to_csv(env_all_log_path, index=False)
        print(f'Saved combined env-filter log to {env_all_log_path}')
else:
    print('No filter records were recorded; combined log not written.')
