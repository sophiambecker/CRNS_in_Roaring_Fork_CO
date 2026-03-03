# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:40:29 2025

@author: sbecker14
"""

# load summary of site data used for calibration: 
# libraries
import os
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir('C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\')

from UTS_helpers import convert_neutrons_to_soil_moisture_uts

import datetime as dt
from datetime import datetime
import pandas as pd

import glob
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from sklearn.neighbors import LocalOutlierFactor


stamp = dt.datetime.now().strftime("%Y%m%d") 
Dir = os.getcwd()
outFold = '\\Site_specific_predictions_output_{}'.format(stamp)
                                                 
outDir = os.path.normpath(Dir + os.sep + outFold) + '\\'    # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory


# load portable calibration data
directory_path_port_des = 'Data\\MockPortableData\\Processed_ALlRF_Des_output20260210'
directory_path_port_uts = 'Data\\MockPortableData\\Processed_ALlRF_UTS_output20260210'
port_uts_file_pattern = f'{directory_path_port_uts}\\*.csv'
port_des_file_pattern = f'{directory_path_port_des}\\*.csv'
port_uts_paths = glob.glob(port_uts_file_pattern, recursive = True)
port_des_paths = glob.glob(port_des_file_pattern, recursive = True)
site_names_old_port_uts = [os.path.basename(fp).split('_')[2] for fp in port_uts_paths]
site_names_old_port_des = [os.path.basename(fp).split('_')[2] for fp in port_des_paths]
dfs_port_des_ls = [pd.read_csv(file_path).copy() for file_path in port_des_paths]
dfs_port_uts_ls = [pd.read_csv(file_path).copy() for file_path in port_uts_paths]
dict_port_des = dict(zip(site_names_old_port_des, dfs_port_des_ls))
dict_port_uts = dict(zip(site_names_old_port_uts, dfs_port_uts_ls))

# Use data that's been filtered for outliers and snow:
directory_path = 'CombineDataWithFunction_half_hr_output_sealevel_pref_20260210'
file_pattern = f'{directory_path}\\*.csv'
file_paths = glob.glob(file_pattern, recursive=True)
dfs_ls = [pd.read_csv(file_path).copy() for file_path in file_paths]
site_names_old = [os.path.basename(fp).split('_')[0] for fp in file_paths]
print(f'Old site names are {site_names_old}')
site_names_new = [os.path.basename(fp).split('_')[1] for fp in file_paths]
print(f'New site names are {site_names_new}')

# make dictionary of dataframes using new site names 
df_dict = dict(zip(site_names_new, dfs_ls))

# load reference detector information:
reference_detector = pd.read_csv("Data\\Reference_Detector_summary.csv")
reference_dict = dict(zip(reference_detector['Site'], reference_detector['Raw_Portable_Mod_cph']))

# load landcover descriptions: 
RF_veginfo = pd.read_csv("Data\\RoaringFork_CRNS_metadata.csv")
RF_veginfo.replace('RF 5', 'RF5', inplace=True) 
veg_dict = dict(zip(RF_veginfo['Original_ID'], RF_veginfo['Land Cover ']))

# dictionary to get old site name based on new site name: 

new_to_old_name = dict(zip(site_names_new, site_names_old))
    
# load calibration day data: 

site_var = pd.read_excel("Data\\Mock Calibration Summary_20251106.xlsx")

# Extract row 16 and columns 1–15
sample_series_start = site_var.iloc[16, 1:16]
sample_series_end = site_var.iloc[17, 1:16]
# Convert values to just the date, keep column names (keys) as strings

sample_dt_start = {
    col: pd.to_datetime(val)
           #.tz_localize('UTC')             # 1. Say it's in UTC
           .floor('min')                      # floor to minute
           #.date()                         # 2. Extract just the date
           for col, val in sample_series_start.items()
}

sample_dt_end = {
    col: pd.to_datetime(val)
           #.tz_localize('UTC')             # 1. Say it's in UTC
           .floor('min')                      # floor to minute
           #.date()                         # 2. Extract just the date
           for col, val in sample_series_end.items()
}
# load forest/non-forest classification:
f_clas = pd.read_excel("Data\\VisualSiteClassification.xlsx")
f_clas_dict = dict(zip(f_clas['Site Original'], f_clas['Imagery']))

# pull BWE estimates 
directory_path = 'Data\\MockBWEestimatesToUse'
file_pattern = f'{directory_path}/**/*.csv'
file_paths = glob.glob(file_pattern, recursive=True)
dataframes = [pd.read_csv(file_path).copy() for file_path in file_paths]

# Create list of information you want    
TotBWE_ls = [df[['Site', 'BWE Representing 200 m Radius Footprint (mm)','BWE Uncertainty (mm)']].copy() for df in dataframes]
 
# Concatenate the list of DataFrames into a single DataFrame
TotBWE_df = pd.concat(TotBWE_ls, ignore_index=True)
TotBWE_df.replace('RF 5', 'RF5', inplace=True)
out_df = pd.read_csv("Calibration_AnalysisWithKGE_output_20260210\\Site_data_summary.csv")

N0_df = pd.read_csv("Calibration_AnalysisWithKGE_output_20260210\\Parameter_fit.csv")

# load daily precip (PRISM) for snow screening
precip_df = pd.read_csv("Data\\DailyPRISM_Ppt.csv")
precip_df['date'] = pd.to_datetime(precip_df['Date']).dt.normalize()

for n in site_names_new:
    
            
    THIS_SITE_new = n
    
    sitedata = out_df[out_df['NewName']==THIS_SITE_new].copy()
    print(sitedata.shape)
    
    
    site_bd = sitedata['bd'].item()
    # estimate porosity from bulk density (assume particle density 2.65 g/cm3)
    porosity = max(0.05, min(0.8, 1 - (site_bd / 2.65)))
    lc = sitedata['Canopy'].iloc[0]
    
    site_lw = sitedata['lw'].item()
    site_soc = sitedata['soc_water'].item()
    
    # get universal parameters from df:
    N0_univ = N0_df['Fit'][0]
    ND_univ = N0_df['Fit'][1]
    
    # Assuming df is your pandas DataFrame with 'Avg_N0', 'theta_tot', and 'Npvi'
    
    theta_tot = sitedata['Sample_total_swc_g'].astype(float).item()
    N_Des = sitedata['N_pvisd_Des'].item()
    N_UTS = sitedata['N_pisd_UTS'].item()
    
    # Find N0 based on single gravimetric sample value with Desilets method
    N0_fit_Des = sitedata['N0_fit_Des'].item()
    
    print(f'N0 from Desilets fit for {THIS_SITE_new} ( {lc} ) is: {N0_fit_Des}')
    
    # use N0_fitted to predict swc at site
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    # Find weighted TDR value corresponding to gravimetric sampling day
    site_df = df_dict[THIS_SITE_new].copy().drop(columns = ['Date'], errors = 'ignore').copy() # use  unfiltered data
    
    # estimate soil moisture for all snow-free days using Desilets equation
    tau = site_lw +site_soc
    
    f_d = None
    
    f_d = sitedata['N_ratio_raw'].item()
    print(f'f_d = {f_d} for Site {THIS_SITE_new}')
    
    #Check to see if the predicted SWC at calibration time is correct:
    cal_dt_first = sample_dt_start[THIS_SITE_old]
    cal_dt_last = sample_dt_end[THIS_SITE_old]
    
    if 'DateTime' not in site_df.columns:
        raise KeyError(f"Expected column 'DateTime' not found in site_df for {THIS_SITE_new}.")
    
    site_df['date'] = pd.to_datetime(site_df['DateTime']).dt.floor('min')
    
    # --- Snow screening: combine counts + temp + daily precip ---
    precip_col = THIS_SITE_old
    site_df['date_day'] = site_df['date'].dt.normalize()
    if precip_col in precip_df.columns:
        site_df = site_df.merge(
            precip_df[['date', precip_col]],
            left_on='date_day',
            right_on='date',
            how='left',
            suffixes=('', '_ppt')
        )
        site_df.rename(columns={precip_col: 'ppt_daily_mm'}, inplace=True)
        site_df.drop(columns=['date_ppt'], inplace=True, errors='ignore')
    else:
        site_df['ppt_daily_mm'] = np.nan

    # rolling median baseline for moderated counts (3 days @ 30 min ≈ 144 samples)
    site_df['mod_med_3d'] = site_df['Corrected_Mod_cph_for_Des'].rolling(
        window=144, min_periods=48, center=True
    ).median()
    site_df['mod_drop'] = site_df['Corrected_Mod_cph_for_Des'] < site_df['mod_med_3d'] * 0.90

    # choose a soil temperature proxy if available
    soil_temp_cols = [c for c in ['sT_5', 'sT_10', 'sT_20', 'sT_50'] if c in site_df.columns]
    if soil_temp_cols:
        site_df['soil_temp_proxy'] = site_df[soil_temp_cols[0]]
    else:
        site_df['soil_temp_proxy'] = np.nan

    site_df['cold_air'] = site_df['airT'] <= 1.5
    site_df['cold_soil'] = site_df['soil_temp_proxy'] <= 1.0
    site_df['recent_snow_precip'] = site_df['ppt_daily_mm'].fillna(0) > 2.0

    # accumulate cold-precip over past 3 days
    daily_cold_precip = (
        site_df.groupby('date_day')
        .apply(lambda g: g['ppt_daily_mm'].iloc[0] if (g['airT'] <= 1.5).mean() > 0.3 else 0)
        .rename('cold_ppt_mm')
        .reset_index()
    )
    site_df = site_df.merge(daily_cold_precip, on='date_day', how='left')
    site_df['snow_cold_budget'] = site_df['cold_ppt_mm'].fillna(0).rolling(window=3, min_periods=1).sum()

    # flag unrealistically high SWC relative to porosity when temps are cold
    swc_cols = [c for c in ['WeightedTDR_SWC', 'SWC_Des_cm3_cm3', 'SWC_UTS_cm3_cm3'] if c in site_df.columns]
    swc_high = pd.Series(False, index=site_df.index)
    for c in swc_cols:
        swc_high |= site_df[c] > porosity
    cold_temps = (site_df['airT'] <= 1.0) & (site_df['soil_temp_proxy'] <= 1.0)

    site_df['snow_flag_raw'] = (
        site_df['mod_drop'] &
        (site_df['cold_air'] | site_df['cold_soil']) &
        (site_df['recent_snow_precip'] | (site_df['snow_cold_budget'] > 2.0))
    ) | (swc_high & cold_temps)

    # melt/unmask indicators
    site_df['warming'] = site_df['airT'].rolling(48, min_periods=12).mean() >= 2.0  # ~1 day window
    site_df['counts_rebound'] = site_df['Corrected_Mod_cph_for_Des'] >= site_df['mod_med_3d'] * 0.96

    # smooth snow flag to avoid single-bin flips
    site_df['snow_flag'] = (
        site_df['snow_flag_raw']
        .rolling(window=5, center=True, min_periods=1)
        .median()
        .round()
        .astype(bool)
    )
    site_df.loc[site_df['warming'] & site_df['counts_rebound'], 'snow_flag'] = False

    # drop snow-flagged rows
    site_df = site_df[~site_df['snow_flag']].copy()
    # --- end snow screening ---

    if not (site_df['date'] == cal_dt_first).any():
        print(f"{THIS_SITE_new} Calibration date {cal_dt_first} not found. Finding nearest date...")
        
        site_df['date_diff'] = (site_df['date'] - cal_dt_first).abs()
    
        nearest_row = site_df.loc[site_df['date_diff'].idxmin()]
        cal_dt_first = nearest_row['date']
        site_df.drop(columns='date_diff', inplace=True)
        print(f'Nearest Date is {cal_dt_first} for {THIS_SITE_new}.')
    
        QA1 = 'FALSE'
    else: QA1 = 'TRUE' # True if the calibration sample date is in the stationary data
    
    if not (site_df['date'] == cal_dt_last).any():
        
        cal_dt_last = cal_dt_first + pd.Timedelta(hours=2)
        print(f"Setting last sample time to 2 hours after cal_dt_first: {cal_dt_last}")
        
    # get stationary raw counts right away (calibration period might change slightly below based on in airRH and airT availability)
    cal_data_1 = site_df[(site_df['date'] >= cal_dt_first) & (site_df['date'] <= cal_dt_last)].copy()
    
    check_cols = ['Corrected_Mod_cph_for_Des','Corrected_Mod_cph_for_UTS','airRH','airT','Raw_Moderated_cph'] # for now it's okay if TDR is NaN, we are just calibrating with samples
       
    # check the above columns for nan values. If any column contains an nan value, use the nearest date without nan values as the calibration date. 
    # Check if any NaNs in the row for the starting calibration datetime
    if site_df.loc[site_df['date'] == cal_dt_first, check_cols].isnull().any(axis=1).any():
        
        # Filter to find rows with no NaNs in the desired columns
        valid_rows = site_df[site_df[check_cols].notna().all(axis=1)].copy()
    
        # Compute absolute difference
        valid_rows['date_diff'] = (valid_rows['date'] - cal_dt_first).abs()
    
        # Find the nearest datetime
        nearest_row = valid_rows.loc[valid_rows['date_diff'].idxmin()]
        nearest_dt = nearest_row['date']
    
        # If the gap is more than 12 hours, try to match the original time of day
        if abs(nearest_dt - cal_dt_first) > pd.Timedelta(hours=12):
            target_time = cal_dt_first.time()  # desired hour and minute
            same_time_rows = valid_rows[valid_rows['date'].dt.time == target_time]
            if not same_time_rows.empty:
                same_time_rows['date_diff'] = (same_time_rows['date'] - cal_dt_first).abs()
                nearest_dt = same_time_rows.loc[same_time_rows['date_diff'].idxmin(), 'date']
    
        # Update cal_dt_first
        cal_dt_first = nearest_dt
        
        print(f"Nearest Starting DateTime with values for corrected counts, RH, and Temp is {cal_dt_first} for {THIS_SITE_new}.")
        cal_dt_last = cal_dt_first + pd.Timedelta(hours=2)
        print(f"Setting last sample time to 2 hours after cal_dt_first: {cal_dt_last}")
        QA2 = 'FALSE'  # Different datetime was used
    
    else: QA2 = 'TRUE'  # Original datetime is valid
     
    # Now extract cal_data
    cal_data = site_df[(site_df['date'] >= cal_dt_first) & (site_df['date'] <= cal_dt_last)].copy()
    # Filter out rows with abnormal raw/mod counts relative to the 2-hour calibration window (per site)
    count_cols = ['Corrected_Mod_cph_for_Des', 'Corrected_Mod_cph_for_UTS', 'Raw_Moderated_cph']
    cal_stats = {}
    for col in count_cols:
        if col in cal_data.columns and not cal_data[col].dropna().empty:
            med = cal_data[col].median()
            mad = (cal_data[col] - med).abs().median()
            # fallback tolerance to avoid zero/NaN MAD
            if pd.isna(mad) or mad == 0:
                mad = abs(med) * 0.01 + 1e-6
            cal_stats[col] = (med, mad)
    if cal_stats:
        def _is_outlier(row):
            for col, (med, mad) in cal_stats.items():
                val = row.get(col, np.nan)
                if pd.isna(val):
                    continue
                if abs(val - med) > 6 * mad:
                    return True
            return False
        # identify rows within calibration window
        calib_mask = (site_df['date'] >= cal_dt_first) & (site_df['date'] <= cal_dt_last)
        # drop only those calibration-window rows flagged as outliers from the overall dataframe
        drop_idx = site_df.loc[calib_mask].apply(_is_outlier, axis=1)
        site_df = site_df.loc[~(calib_mask & drop_idx)].copy()
        # refresh cal_data after filtering
        cal_data = site_df[(site_df['date'] >= cal_dt_first) & (site_df['date'] <= cal_dt_last)].copy()
        
        # length of cal_data: 
            
    # Aggregate using a bin that is exactly the calibration interval and anchored to cal_dt_first
    start_time = pd.to_datetime(cal_dt_first)
    end_time   = pd.to_datetime(cal_dt_last)
    
    interval = end_time - start_time
    print("Interval length:", interval)
    
    # Convert interval to pandas frequency string (e.g., 2 hours -> '2H')
    if interval.total_seconds() % 3600 == 0:
        freq_str = f"{int(interval.total_seconds()//3600)}h"
    elif interval.total_seconds() % 60 == 0:
        freq_str = f"{int(interval.total_seconds()//60)}T"
    else:
        freq_str = f"{int(interval.total_seconds())}S"
    
    print("Resampling frequency string:", freq_str)
    
    
    
    # do some filtering before the resampling:
    
     # step 1
     
     # Step 1: Make extreme values NaN instead of dropping
    mask1 = (site_df['Corrected_Mod_cph_for_Des'] < 1200) | \
             (site_df['Corrected_Mod_cph_for_Des'] > 6000)
     
    site_df.loc[mask1, ['Corrected_Mod_cph_for_Des', 'Corrected_Mod_cph_for_UTS']] = np.nan
     
     
     # Step 2: Compute mean and SD ignoring NaN
    mod_max = np.nanmean(site_df['Corrected_Mod_cph_for_Des']) + 2 * np.nanstd(site_df['Corrected_Mod_cph_for_Des'])
    mod_min = np.nanmean(site_df['Corrected_Mod_cph_for_Des']) - 2 * np.nanstd(site_df['Corrected_Mod_cph_for_Des'])
     
    mask2 = (site_df['Corrected_Mod_cph_for_Des'] < mod_min) | \
             (site_df['Corrected_Mod_cph_for_Des'] > mod_max)
     
    site_df.loc[mask2, ['Corrected_Mod_cph_for_Des', 'Corrected_Mod_cph_for_UTS', 
                         ]] = np.nan
    '''
    site_df.drop(site_df[(site_df['Corrected_Mod_cph_for_Des']<1200) | (site_df['Corrected_Mod_cph_for_Des']>6000)].index, inplace = True)
    
    site_df.drop(site_df[(site_df['Raw_Moderated_cph']<4000) | (site_df['Raw_Moderated_cph']>30000)].index, inplace = True) 
    
     # step 2
    mod_max = np.nanmean(site_df['Corrected_Mod_cph_for_Des'])+ 2* np.nanstd(site_df['Corrected_Mod_cph_for_Des'])
    mod_min = np.nanmean(site_df['Corrected_Mod_cph_for_Des'])- 2* np.nanstd(site_df['Corrected_Mod_cph_for_Des'])
    site_df.drop(site_df[(site_df['Corrected_Mod_cph_for_Des']<mod_min) | (site_df['Corrected_Mod_cph_for_Des']>mod_max)].index, inplace = True)
    site_df.dropna(axis = 1, how = 'all', inplace = True) # drop nan columns first
    site_df.dropna(subset = ['Corrected_Mod_cph_for_Des'], inplace = True) #don't drop rows with nan in TDR data yet
    '''
    
    sel_cols = ['Corrected_Mod_cph_for_Des',
           'SWC_Des_cm3_cm3',
           'Corrected_Mod_cph_for_UTS', 
           'SWC_UTS_cm3_cm3',
            'WeightedTDR_SWC', 
            ]
    
    df_outlier_filt = site_df.dropna(subset = sel_cols).copy()
    find_outliers = df_outlier_filt[sel_cols].copy()
  
    
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
    outlier_rows = df_outlier_filt[outliers == -1]
    print(f'{THIS_SITE_new} outliers are: ')
    print(outlier_rows)
    
    # LOF outliers: drop exact timestamps; manual dates: drop entire days
    rm_dates = outlier_rows['date']  # keep full timestamps
    
    # remove outlier date that behaves strangely with precip and where Bare is already nan: 
    extra_dates = pd.to_datetime(['2024-08-16','2024-08-15', '2024-08-14', '2024-08-13']).normalize()
    
    # drop rows matching exact outlier timestamps OR any row on the extra dates
    site_df = site_df[
        ~(site_df['date'].isin(rm_dates) |
          site_df['date'].dt.normalize().isin(extra_dates))
    ]
    
    # Select numeric columns
    numeric_cols = site_df.select_dtypes(include='number').columns
    
    # Resample using 'date' as index, anchoring bins to the calibration start so the calibration window is one full bin
    resampled_df = (
        site_df
        .set_index('date')
        .resample(rule=freq_str, origin=start_time, label='left', closed='left')[numeric_cols]
        .mean()
        .reset_index()
    )
    
    
    
    # -----------------------------
    # Show results
    # -----------------------------
    print(resampled_df.head())
    resampled_df['scaled_stationary_N_Des'] = resampled_df['Corrected_Mod_cph_for_Des']*f_d
    resampled_df['scaled_stationary_N_UTS'] = resampled_df['Corrected_Mod_cph_for_UTS']*f_d
    
    resampled_df['theta_pred_tot_g_Des'] = (0.0808 / (resampled_df['scaled_stationary_N_Des'] / N0_fit_Des - 0.372) - 0.115)  # total gravimetric water content
    resampled_df['theta_pred_pore_vol_Des'] = (resampled_df['theta_pred_tot_g_Des']- tau) * site_bd
    
    resampled_df['TDR_pore_swc_g'] = resampled_df['WeightedTDR_SWC']/site_bd
    resampled_df['TDR_tot_swc_g'] = resampled_df['TDR_pore_swc_g'] + tau
    
    resampled_df['TDR_tot_swc_g_resid_Des'] = resampled_df['TDR_tot_swc_g'] - resampled_df['theta_pred_tot_g_Des'] 
    
    # filter out rows with nan
    resampled_df.dropna(subset = ['TDR_tot_swc_g_resid_Des'], inplace = True)
    resampled_df['NewName'] = THIS_SITE_new
    
    # now do universal N0 prediction 
    resampled_df['swc_univ_N0_pred_tot_g_Des'] =  (0.0808 / (resampled_df['scaled_stationary_N_Des'] / N0_univ - 0.372) - 0.115)  # totalgravimetric water content
    resampled_df['swc_univ_N0_pred_pore_vol_Des'] = (resampled_df['swc_univ_N0_pred_tot_g_Des'] - tau) * site_bd
       
    samp_date = pd.to_datetime(sitedata['CalStart']).iloc[0].floor('min')
    
    #resampled_df['datetime'] = pd.to_datetime(resampled_df['Date']).dt.floor('min')
    
    # -----------------------------
    # CALIBRATION CHECK USING MEAN OVER CALIBRATION INTERVAL
    # -----------------------------
    
    # Extract calibration interval
    cal_data = site_df[(site_df['date'] >= cal_dt_first) & (site_df['date'] <= cal_dt_last)].copy()
    
    if cal_data.empty:
        print(f"⚠️ No data found in calibration interval for {THIS_SITE_new}.")
    else:
        # Compute mean counts over calibration interval
        mean_N_Des = cal_data['Corrected_Mod_cph_for_Des'].mean()
        mean_N_UTS = cal_data['Corrected_Mod_cph_for_UTS'].mean()
    
        # Apply scaling factor f_d
        scaled_mean_N_Des = mean_N_Des * f_d
        scaled_mean_N_UTS = mean_N_UTS * f_d
    
        # Predict gravimetric soil water content using Desilets N0
        theta_pred_cal_Des = 0.0808 / (scaled_mean_N_Des / N0_fit_Des - 0.372) - 0.115
    
        # Predict pore volumetric SWC if needed
        theta_pred_pore_Des = (theta_pred_cal_Des - tau) * site_bd
    
        # Compare to observed gravimetric SWC
        obs_swc = sitedata['Sample_total_swc_g'].item()
        diff = theta_pred_cal_Des - obs_swc
    
        print(f"Calibration interval mean counts (Desilets) = {mean_N_Des:.2f}")
        print(f"Scaled mean counts = {scaled_mean_N_Des:.2f}")
        print(f"Predicted SWC (total gravimetric, g/g) at calibration = {theta_pred_cal_Des:.4f}")
        print(f"Observed SWC at calibration = {obs_swc:.4f}")
        print(f"Difference = {diff:.4f}")
        
        N0_fit_UTS = None
        N0_fit_UTS = sitedata['N0_fit_UTS'].item()
        # Optional: check UTS method
        theta_pred_cal_UTS = convert_neutrons_to_soil_moisture_uts(
            neutron_count=scaled_mean_N_UTS,
            n0=N0_fit_UTS,
            air_humidity=cal_data['Rhov_g_cm3'].mean(),  # mean humidity over interval
            bulk_density=site_bd,
            lattice_water=site_lw*site_bd,
            water_equiv_soil_organic_carbon=site_soc*site_bd,
            method="Mar21_mcnp_drf"
        )
        theta_pred_tot_UTS = (theta_pred_cal_UTS / site_bd) + tau
        diff_UTS = theta_pred_tot_UTS - obs_swc
    
        print(f"Predicted SWC (UTS) at calibration = {theta_pred_tot_UTS:.4f}")
        print(f"Difference (UTS) = {diff_UTS:.4f}")
    
    '''
    matches = resampled_df.loc[resampled_df['date'] == samp_date, 'swc_univ_N0_pred_tot_g_Des']
    
    if matches.empty:
        print(f"⚠️ No match for SampDate {samp_date} in daily data for {THIS_SITE_new}.")
       # continue
    elif len(matches) > 1:
        print(f"⚠️ Multiple matches for SampDate at site {THIS_SITE_new}. Using first one.")
    
    pred_twc = matches.iloc[0]
    obs_twc = sitedata['Sample_total_swc_g'].item()
    diff = pred_twc - obs_twc
    print(f"Difference between predicted and observed (g/g) at cal time = {diff:.4f}")
    
    #check for difference in counts used: 
    
    counts_matches = resampled_df.loc[resampled_df['date'] == samp_date, 'scaled_stationary_N_Des']
    
    if counts_matches.empty:
        print(f"⚠️ No match for SampDate {samp_date} in daily data for {THIS_SITE_new}.")
        #continue
    elif len(counts_matches) > 1:
        print(f"⚠️ Multiple matches for SampDate at site {THIS_SITE_new}. Using first one.")
    
    pred_counts = counts_matches.iloc[0]
    obs_counts= sitedata['N_pvisd_Des'].item()
    diff_counts = pred_counts - obs_counts
    print(f"Difference between counts in timeseries and calibration df at cal time = {diff_counts:.4f}")
    '''
    # now each site with UTS method
    
    # Best-fit N0 UTS value
    print(f'N0 from UTS fit method at Site {THIS_SITE_new} ( {lc} ) is: {N0_fit_UTS:.2f}')
     # filter out rows with nan in airRH or airT
    resampled_df.dropna(subset = ['airRH', 'airT'], inplace = True)
    
    theta_pred_pore_volumetric = resampled_df.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['scaled_stationary_N_UTS'],
            n0=N0_fit_UTS,
            air_humidity=row['Rhov_g_cm3'],
            bulk_density= site_bd,
            lattice_water=site_lw* site_bd,
            water_equiv_soil_organic_carbon= site_soc * site_bd,
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    ) 
    
    resampled_df['theta_pred_tot_g_UTS'] = (theta_pred_pore_volumetric/site_bd) + tau
    resampled_df['theta_pred_pore_vol_UTS'] = theta_pred_pore_volumetric
    
    resampled_df['TDR_tot_swc_g_resid_UTS'] = resampled_df['TDR_tot_swc_g'] - resampled_df['theta_pred_tot_g_UTS'] 
    
    
    resampled_df['NewName'] = THIS_SITE_new
    
    # now do universal ND prediction
    theta_pred_pore_volumetric_univ = resampled_df.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['scaled_stationary_N_UTS'],
            n0=ND_univ,
            air_humidity=row['Rhov_g_cm3'],
            bulk_density= site_bd,
            lattice_water=site_lw* site_bd,
            water_equiv_soil_organic_carbon= site_soc * site_bd,
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    ) 
    
    resampled_df['swc_univ_ND_pred_tot_g_UTS'] = theta_pred_pore_volumetric_univ/site_bd + site_lw+ site_soc
    resampled_df['swc_univ_ND_pred_pore_vol_UTS'] = theta_pred_pore_volumetric_univ
    
    import os
    
    filename = f"{THIS_SITE_new}_UTS_and_Des_{lc}_SiteSpecificN0_predictions.csv"
    filepath = os.path.join(outDir, filename)
    
    resampled_df.to_csv(filepath, index=False)
    print(f"Saved file to: {filepath}")
    
    
    print(f'Saved predictions for {THIS_SITE_new}')
