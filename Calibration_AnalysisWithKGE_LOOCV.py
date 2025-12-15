# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:10:36 2025

@author: sbecker14
"""

'''
note this script requires the packages installed in an environment containing requirements1
 
Do a leave-one-out cross-validation analysis using the soil sample data (for Desilets and UTS methods)
- Calculate single N0 from gravimetric data and stationary detector (scaled to portable)
- identify outlier sites

Predict SWC from moderated counts (scaled to portable) using both Desilets and UTS methods
for the time series at all the sites




'''
# libraries
import os
os.chdir('C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO')
from config import Config
from watervapor import calculate_watervapor
from UTS_helpers import convert_neutrons_to_soil_moisture_uts

import datetime as dt
import pandas as pd

import glob
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr

stamp = dt.datetime.now().strftime("%Y%m%d") 
Dir = os.getcwd()
outFold = '\\Calibration_AnalysisWithKGE_output_{}'.format(stamp)
                                                 
outDir = os.path.normpath(Dir + os.sep + outFold) + '\\'    # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory


# load portable calibration data
directory_path_port_des = 'Data\\MockPortableData\\Processed_ALlRF_Des_output20251106'
directory_path_port_uts = 'Data\\MockPortableData\\Processed_ALlRF_UTS_output20251106'
port_uts_file_pattern = f'{directory_path_port_uts}\\*.csv'
port_des_file_pattern = f'{directory_path_port_des}\\*.csv'
port_uts_paths = glob.glob(port_uts_file_pattern, recursive = True)
port_des_paths = glob.glob(port_des_file_pattern, recursive = True)
site_names_old_port_uts = [os.path.basename(fp).split('_')[2] for fp in port_uts_paths]
site_names_old_port_des = [os.path.basename(fp).split('_')[2] for fp in port_des_paths]
dfs_port_des_ls = [pd.read_csv(file_path) for file_path in port_des_paths]
dfs_port_uts_ls = [pd.read_csv(file_path) for file_path in port_uts_paths]
dict_port_des = dict(zip(site_names_old_port_des, dfs_port_des_ls))
dict_port_uts = dict(zip(site_names_old_port_uts, dfs_port_uts_ls))

# Use data that's been filtered for outliers and snow:
directory_path = 'FilteredSnowFreeData_output20251106'
file_pattern = f'{directory_path}\\*.csv'
file_paths = glob.glob(file_pattern, recursive=True)
dfs_ls = [pd.read_csv(file_path) for file_path in file_paths]
site_names_old = [os.path.basename(fp).split('_')[0] for fp in file_paths]
print(f'Old site names are {site_names_old}')
site_names_new = [os.path.basename(fp).split('_')[1] for fp in file_paths]
print(f'New site names are {site_names_new}')

# make dictionary of dataframes using new site names 
df_dict = dict(zip(site_names_new, dfs_ls))

# load reference detector information:
reference_detector = pd.read_csv("Data\\Reference_Detector_summary.csv")
reference_dict = dict(zip(reference_detector['Site'], reference_detector['Portable_to_Stationary_Raw_Cts']))

# load landcover descriptions: 
RF_veginfo = pd.read_csv("Data\\RoaringFork_CRNS_metadata.csv")
RF_veginfo.replace('RF 5', 'RF5', inplace=True) 
veg_dict = dict(zip(RF_veginfo['Original_ID'], RF_veginfo['Land Cover ']))

# dictionary to get old site name based on new site name: 

new_to_old_name = dict(zip(site_names_new, site_names_old))
    
# load calibration day data: 

site_var = pd.read_excel("Data\\Mock Calibration Summary_20251106.xlsx")

# Extract row 16 and columns 1–15
sample_series = site_var.iloc[16, 1:16]

# Convert values to just the date, keep column names (keys) as strings

sample_date = {
    col: pd.to_datetime(val)
           .tz_localize('UTC')             # 1. Say it's in UTC
           .date()                         # 2. Extract just the date
           for col, val in sample_series.items()
}

# load forest/non-forest classification:
f_clas = pd.read_excel("Data\\VisualSiteClassification.xlsx")
f_clas_dict = dict(zip(f_clas['Site Original'], f_clas['Imagery']))

# pull BWE estimates 
directory_path = 'Data\\MockBWEestimatesToUse'
file_pattern = f'{directory_path}/**/*.csv'
file_paths = glob.glob(file_pattern, recursive=True)
dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Create list of information you want    
TotBWE_ls = [df[['Site', 'BWE Representing 200 m Radius Footprint (mm)','BWE Uncertainty (mm)']] for df in dataframes]
 
# Concatenate the list of DataFrames into a single DataFrame
TotBWE_df = pd.concat(TotBWE_ls, ignore_index=True)
TotBWE_df.replace('RF 5', 'RF5', inplace=True)


# define functions for evaluation criteria:

def r_squared_from_definition(observed, predicted):
    
    # Remove NaNs from both arrays
    mask = ~np.isnan(predicted) & ~np.isnan(observed)
    pred = np.array(predicted)[mask]
    obs = np.array(observed)[mask]
    
    ss_res = np.sum((obs - pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)  # Total sum of squares
    
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# function for KGE
def kling_gupta_efficiency(predicted, observed):
    # Remove NaNs from both arrays
    mask = ~np.isnan(predicted) & ~np.isnan(observed)
    pred = np.array(predicted)[mask]
    obs = np.array(observed)[mask]
    
    # Components
    r, _ = pearsonr(pred, obs)
    mean_pred, mean_obs = np.mean(pred), np.mean(obs)
    std_pred, std_obs = np.std(pred), np.std(obs)
    
    beta = mean_pred / mean_obs
    gamma = (std_pred / mean_pred) / (std_obs / mean_obs)
    
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    # Additional metrics
    r_squared = r_squared_from_definition(obs, pred)
    rmse = np.sqrt(np.mean((pred - obs)**2))
    
    # Bias
    #bias = mean_pred - mean_obs

    # Unbiased RMSE
    ubrmse = np.sqrt(np.mean(((pred-mean_pred)-(obs-mean_obs))**2))
    #np.sqrt(np.mean((residuals - bias) ** 2))
    
    return pd.DataFrame([{
        'KGE': kge,
        'r': r,
        'beta': beta,   
        'gamma': gamma,
        'r_squared': r_squared,
        'RMSE': rmse,
        'ubRMSE': ubrmse,
        'std_pred': std_pred,
        'std_obs': std_obs,
        'std_bias':np.abs(std_pred-std_obs)
    }])

'''
KGE Interpretation:
γ Value	Interpretation
1.0	Perfect variability match
< 1	Model is less variable than observations (too smooth)
> 1	Model is more variable than observations (too jumpy)

β Value	Interpretation
1.0	Model has no bias in the mean (perfect)
< 1	Model underestimates the average value
> 1	Model overestimates the average value


'''

# create data set of Portable corrected counts, Stationary corrected N, and gravimetric water content for all sites

rows = [ ]

for s in site_names_new:
    df = None
    Ncal_Des_p = None
    Ncal_UTS_p = None
    Ncal_UTS_st_d = None
    Ncal_Des_st = None 
    Ncal_UTS_st = None
    N_ratio_raw = None
    
    site_bd = None
    
    swc_g = None
    lw = None
    soc = None
    GRAV_swc_tot_g = None
    
    Rhov_cal_g = None
    
    THIS_SITE_new = s
    
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    df = df_dict[THIS_SITE_new]
    df['date'] = pd.to_datetime(df['date']).dt.date # make sure date column is in date format
    
    p_uts_df = dict_port_uts[THIS_SITE_old]
    p_des_df = dict_port_des[THIS_SITE_old]
    
    cal_date = sample_date[THIS_SITE_old]
    
    if not (df['date'] == cal_date).any():
        print(f"{THIS_SITE_new} Calibration date {cal_date} not found. Finding nearest date...")
        
        df['date_diff'] = df['date'].apply(lambda d: abs(d - cal_date))
        nearest_row = df.loc[df['date_diff'].idxmin()]
        cal_date = nearest_row['date']
        df.drop(columns='date_diff', inplace=True)
        print(f'Nearest Date is {cal_date} for {THIS_SITE_new}.')
        QA1 = 'FALSE'
    else: QA1 = 'TRUE' # True if the calibration sample date is in the stationary data
        
    check_cols = ['Corrected_Mod_cph_for_Des','Corrected_Mod_cph_for_UTS','airRH','airT'] # for now it's okay if TDR is NaN, we are just calibrating with samples
    
    # check the above columns for nan values. If any column contains an nan value, use the nearest date without nan values as the calibration date. 
    # Check if any NaNs in the row for the calibration date
    if df.loc[df['date'] == cal_date, check_cols].isnull().any(axis=1).any():
        
        # Filter to find rows with no NaNs in the desired columns
        valid_rows = df[df[check_cols].notna().all(axis=1)].copy()
    
        # Find the row with the minimum absolute date difference
        valid_rows['date_diff'] = valid_rows['date'].apply(lambda d: abs(d - cal_date))
        nearest_row = valid_rows.loc[valid_rows['date_diff'].idxmin()]
        
        # Update cal_date
        cal_date = nearest_row['date']
        QA2 = 'FALSE' # False if there were NaNs in the stationary data for the sampling time and a different datetime is used instead
    else: QA2 = 'TRUE' # True if there aren't any NaNs in the stationary data for the sampling calibration time
    
    # Now extract cal_data
    cal_data = df[df['date'] == cal_date]
    
    cal_swc = cal_data['WeightedTDR_SWC'].item()
    
    Ncal_Des_p = p_des_df['mod_nc_cph'].mean() # corrected moderated counts for Desilets method, using f3 intesity correction
    
    Ncal_UTS_p = p_uts_df['mod_nc_cph'].mean() # corrected moderated counts for UTS method, using f3 intensity correction
    
    #Ncal_p_raw = p_des_df[''] # could edit portable processing code to also output raw data
    
    Ncal_Des_st = cal_data['Corrected_Mod_cph_for_Des'].item() # stationary data during soil sampling
    Ncal_UTS_st = cal_data['Corrected_Mod_cph_for_UTS'].item() # stationary data during soil sampling
    
    N_ratio_raw = reference_dict[THIS_SITE_new] # f_d in manuscript, ratio of raw portable to stationary counts
    
    Ncal_Des_st_d = Ncal_Des_st * N_ratio_raw # scale stationary counts by sensitivity compared to portable
    Ncal_UTS_st_d = Ncal_UTS_st * N_ratio_raw
    
    # Get other site-specific variables to calculate total gravimetric water content
    
    site_bd = site_var[THIS_SITE_old][24]
    
    swc_g = site_var[THIS_SITE_old][18] # pore water (g/g)
    lw = site_var[THIS_SITE_old][26] # lattice water (g/g)
    soc = site_var[THIS_SITE_old][29] # soc water (g/g)
    
    porosity = 1-(site_bd/2.65)
    
    print(f"{THIS_SITE_new}: bd={site_bd}, lw={lw}, soc={soc}")
    
    GRAV_swc_tot_g = swc_g + lw + soc
    
    # pore gravimetric water content from TDR = (volumetric water content)/bd
    TDR_Pore_g = cal_swc/site_bd
    
    TDR_swc_tot_g = TDR_Pore_g + lw + soc
    
    elev = site_var[THIS_SITE_old][6]
    
    # add forest/non-forest column:
    Canopy = f_clas_dict[THIS_SITE_old]
    LC = veg_dict[THIS_SITE_old]
    
    #N0_ratio = site_var[THIS_SITE_old][42] # portable/stationary # this isn't used for anything
    
    # Extract fitted N0
    N0_Des = None
    N0_Des =  Ncal_Des_st_d / (0.0808 / (GRAV_swc_tot_g + 0.115) + 0.372)  # N0 (cph)
    
    RH_cal = cal_data['airRH'].item()
    T_cal = cal_data['airT'].item()
    
    Rhov_cal_g = cal_data['Rhov_g_cm3'].item() # in g/cm^3
    
    objective_singlerow = None
    
    def objective_singlerow(N0):
        
            try:
                swc_pred = convert_neutrons_to_soil_moisture_uts(
                    neutron_count=Ncal_UTS_st_d,
                    n0=N0,
                    air_humidity=Rhov_cal_g,
                    bulk_density=site_bd,
                    lattice_water=lw * site_bd,
                    water_equiv_soil_organic_carbon=soc * site_bd,
                    method="Mar21_mcnp_drf"
                )
            except:
                return np.inf  # Return a large error if conversion fails
            
            actual_swc = swc_g * site_bd # pore gravimetric water content
            
            return abs(swc_pred - actual_swc)  # absolute error
    
    # Run the minimizer
    res = None
    res = minimize_scalar(objective_singlerow, bounds=(Ncal_Des_st_d, Ncal_Des_st_d + 6000), method='bounded')
    
    # Best-fit N0 value
    N0_UTS = None
    N0_UTS = res.x
    
    # might need to come back and add BWE estimates
    
    BWE = TotBWE_df[TotBWE_df['Site']==THIS_SITE_old]['BWE Representing 200 m Radius Footprint (mm)'].item()
    BWE_uncer = TotBWE_df[TotBWE_df['Site']==THIS_SITE_old]['BWE Uncertainty (mm)'].item()
    
    row = {'N_pvisd_Des': Ncal_Des_st_d, 'N_pisd_UTS': Ncal_UTS_st_d, 'Sample_total_swc_g': GRAV_swc_tot_g, 'TDR_total_swc_g': TDR_swc_tot_g, 'bd': site_bd, 
           'lw': lw, 'soc_water': soc, 'Porosity': porosity, 'Elev': elev, 'Canopy': Canopy, 'landCoverClass': LC, 'OldName': THIS_SITE_old, 'NewName': THIS_SITE_new, 
           'CalDate': cal_date, 'SampDate': sample_date[THIS_SITE_old], 'airRH': p_uts_df['airRH'].mean(), 'airT': p_uts_df['airT'].mean(), 'Rhov_cal_g_cm3': Rhov_cal_g,
            'WeightedTDR': cal_data['WeightedTDR_SWC'].item(), 'ppt': cal_data['PRISM_ppt_mm'].item(), 
           'Bare': cal_data['Bare'].item(), 'BWE_mm': BWE, 'BWE_uncer': BWE_uncer,
            'Raw_Mod_cv':cal_data['Raw_Mod_Coeff_of_Var'].item(), 'Raw_Mod_sqrt':cal_data['Raw_Mod_sqrt'].item(),
           'N0_fit_Des': N0_Des, 'N0_fit_UTS': N0_UTS, 
           'N_ratio_raw': N_ratio_raw,
           'QA1': QA1, 'QA2': QA2}
    
    rows.append(row)

out_df = pd.DataFrame(rows)

# CALIBRATE N0 WITH LOOCV

possible = list(range(len(out_df)))

for i in possible:
    select = [j for j in possible if j != i] # select all but one index
    subset = out_df.iloc[select, :]
    
    # make sure values are reset for each loop:
    theta_tot = None
    N = None
    N0_fit_Des = None
    
    # DESILETS METHOD, calibrated with portable data

    # Assuming df is your pandas DataFrame with 'Avg_N0', 'theta_tot', and 'Npvi'
    N0_start = subset['N_pvisd_Des'].mean()

    theta_tot = subset['Sample_total_swc_g'].astype(float).values
    N = subset['N_pvisd_Des'].values

    # Define the model function: f(N, N0)
    def model(N, N0):
        return 0.0808 / (N / N0 - 0.372) - 0.115
    
    popt = None
    # Fit the model using curve_fit (nonlinear least squares)
    popt, pcov = curve_fit(model, N, theta_tot, p0=[N0_start])
    
    # Extract fitted N0
    N0_fit_Des = popt[0]
    
    print(f'N0 from Desilets fit is: {N0_fit_Des}')

    # evaluate with site left out
    N_val = None
    theta_tot_val = None
    theta_pred = None
    residual = None
    
    N_val = out_df.iloc[i, :]['N_pvisd_Des']
    theta_tot_val = float(out_df.iloc[i, :]['Sample_total_swc_g'])
    theta_pred = model(N_val, *popt)
    residual = theta_tot_val - theta_pred

    out_df.loc[i,'Des_resid_loocv_N0'] = residual
    out_df.loc[i,'Des_pred_loocv_N0'] = theta_pred
    out_df.loc[i,'Des_obs_loocv_N0'] = theta_tot_val
    
# Calculate stats (RMSE, ubRMSE, R2, KGE)

Des_loocv_fit = kling_gupta_efficiency(out_df['Des_pred_loocv_N0'], out_df['Des_obs_loocv_N0'])

# calculate stats without outliers

# find outliers using inter quartile range
Q1 = np.percentile(out_df['Des_resid_loocv_N0'], 25)
Q3 = np.percentile(out_df['Des_resid_loocv_N0'], 75)
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
IQR = Q3 - Q1
print(f"IQR: {IQR}")
lower_iqr_bound = Q1 - 1.5 * IQR
upper_iqr_bound = Q3 + 1.5 * IQR
print(f"Lower Bound: {lower_iqr_bound}")
print(f"Upper Bound: {upper_iqr_bound}")

iqr_outliers = [x for x in out_df['Des_resid_loocv_N0'] if x < lower_iqr_bound or x > upper_iqr_bound]
print(f"Outliers from IQR: {iqr_outliers}")

# find outliers in the residuals # keep working on these: want to just find 1d outliers from residuals now

sd_resid = np.std(out_df['Des_resid_loocv_N0'])
mean_resid = np.mean(out_df['Des_resid_loocv_N0'])
upper = mean_resid + 2*sd_resid
lower = mean_resid - 2*sd_resid

outliers = out_df[((out_df['Des_resid_loocv_N0']>upper) | (out_df['Des_resid_loocv_N0']<lower))]
print(f"Outliers from standard deviation: {outliers}")

keep_df = out_df.drop(outliers.index).copy()
keep_df2 = keep_df.reset_index()

possible_keep = list(range(len(keep_df2)))

for i in possible_keep:
    select = [j for j in possible_keep if j != i] # select all but one index
    print(select)
    subset = keep_df2.iloc[select, :]
    
    theta_tot = None
    N = None
    N0_fit_Des = None
    
    # DESILETS METHOD, calibrated with corrected stationary data

    # Assuming df is your pandas DataFrame with 'Avg_N0', 'theta_tot', and 'Npvi'
    N0_start = subset['N_pvisd_Des'].mean()

    theta_tot = subset['Sample_total_swc_g'].astype(float).values
    N = subset['N_pvisd_Des'].values


    # Define the model function: f(N, N0)
    def model(N, N0):
        return 0.0808 / (N / N0 - 0.372) - 0.115

    popt = None
    # Fit the model using curve_fit (nonlinear least squares)
    popt, pcov = curve_fit(model, N, theta_tot, p0=[N0_start])
    
    # Extract fitted N0
    N0_fit_Des = popt[0]
    
    print(f'N0 from Desilets fit is: {N0_fit_Des}')

    # evaluate with site left out
    N_val = None
    theta_tot_val = None
    theta_pred = None
    residual = None
    
    N_val = keep_df2.iloc[i, :]['N_pvisd_Des']
    theta_tot_val = float(keep_df2.iloc[i, :]['Sample_total_swc_g'])
    theta_pred = model(N_val, *popt)
    residual = theta_tot_val - theta_pred

    keep_df2.loc[i,'Des_resid_loocv_N0_2'] = residual
    keep_df2.loc[i,'Des_pred_loocv_N0_2'] = theta_pred
    keep_df2.loc[i,'Des_obs_loocv_N0_2'] = theta_tot_val

Des_loocv_fit_keep = kling_gupta_efficiency(keep_df2['Des_pred_loocv_N0_2'], keep_df2['Des_obs_loocv_N0_2'])

'''
# plot residuals vs obs swc
label each point
put outliers in different color
label mean

'''
'''
plt.scatter(N, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('N')
plt.ylabel('Residuals')
plt.title('Desilets Residual Plot')
plt.show()
'''
# single N0 to save using all data points
theta_tot = out_df['Sample_total_swc_g'].astype(float).values
N = out_df['N_pvisd_Des'].values

# Bootstrap parameters
n_bootstrap = 1000  # Number of resamples
N0_bootstrap = []

# Perform bootstrap resampling
for _ in range(n_bootstrap):
    sample_indices = np.random.choice(len(theta_tot), len(N), replace=True)
    x_sample = N[sample_indices]
    y_sample = theta_tot[sample_indices]

    try:
        popt, _ = curve_fit(model, x_sample, y_sample, p0=[x_sample.min()], maxfev=1000)
        N0_bootstrap.append(popt[0])
    except:
        continue  # Skip failed fits

# Compute confidence intervals from percentiles
CI_lower_Des, CI_upper_Des = np.percentile(N0_bootstrap, [2.5, 97.5])

print(f"95% Bootstrap Confidence Interval for N0: ({CI_lower_Des:.4f}, {CI_upper_Des:.4f})")

# Define the model function: f(N, N0)
def model(N, N0):
    return 0.0808 / (N / N0 - 0.372) - 0.115

popt = None
N0_fit_Des = None

# Fit the model using curve_fit (nonlinear least squares)
popt, pcov = curve_fit(model, N, theta_tot, p0=[N0_start])

# Extract fitted N0
N0_fit_Des = popt[0]

Des_N0 = ['Des_N0', N0_fit_Des, CI_lower_Des, CI_upper_Des]

# find residuals for predictions from N0 fit to all samples
pred = model(N, *popt) # predict total water content (g/g)
Des_resid = theta_tot - pred
out_df['Des_resid_N0_fit_to_all'] = Des_resid

# UTS METHOD #############################################################################
'''
# need absolute air humidity in g/cm^3
RH_cal = out_df['airRH']
T_cal = out_df['airT']
Rhov_cal, _ = calculate_watervapor(
RH_cal, T_cal, Config.gama
) # output in kg/m^3

out_df['Rhov_cal_g_cm3'] = Rhov_cal/1000 # in g/cm^3
'''
# Objective function to minimize: total absolute difference over all rows

def objective2(N0, df, verbose=False):
    swc_pred = []
    invalid_rows = 0

    for idx, row in df.iterrows():
        try:
            neutron_count = row['N_pisd_UTS']
            air_humidity = row['Rhov_cal_g_cm3']
            bd = row['bd']
            lw = row['lw'] * row['bd']
            soc = row['soc_water'] * row['bd']

            # Sanity checks
            if any([
                not np.isfinite(neutron_count),
                not np.isfinite(air_humidity),
                not np.isfinite(bd),
                not np.isfinite(lw),
                not np.isfinite(soc)
            ]):
                raise ValueError(f"Non-finite input in row {idx}")

            swc_val = convert_neutrons_to_soil_moisture_uts(
                neutron_count=neutron_count,
                n0=N0,
                air_humidity=air_humidity,
                bulk_density=bd,
                lattice_water=lw,
                water_equiv_soil_organic_carbon=soc,
                method="Mar21_mcnp_drf"
            )

            if not np.isfinite(swc_val) or swc_val < 0:
                raise ValueError(f"Invalid SWC value computed in row {idx}: {swc_val}")

        except Exception as e:
            if verbose:
                print(f"[Row {idx}] Skipping due to error: {e}")
            swc_val = np.nan
            invalid_rows += 1

        swc_pred.append(swc_val)

    swc_pred = np.array(swc_pred)
    actual_swc = (df['Sample_total_swc_g'].values - df['lw'] - df['soc_water']) * df['bd']
    valid = ~np.isnan(swc_pred)

    if np.sum(valid) == 0:
        if verbose:
            print("No valid data points in this sample. Returning np.inf.")
        return np.inf
    
    total_diff = np.sum(np.abs(swc_pred[valid] - actual_swc[valid]))

    if verbose:
        print(f"Valid rows: {np.sum(valid)}, Invalid rows: {invalid_rows}, Objective: {total_diff:.4f}")

    return total_diff

  
# Do LOOCV with soil samples for UTS:

possible = list(range(len(out_df)))

for i in possible:
    select = [j for j in possible if j != i] # select all but one index
    subset = out_df.iloc[select, :]

    # Assuming df is your pandas DataFrame with 'Avg_N0', 'theta_tot', and 'Npvi'
    N0_start = subset['N_pisd_UTS'].mean()

    theta_tot = subset['Sample_total_swc_g'].astype(float).values
    N = subset['N_pisd_UTS'].values

    # Run the minimizer
    res = None
    res = minimize_scalar(objective2, args=(subset), bounds=(N0_start, N0_start + 6000), method='bounded')

    # Best-fit N0 value
    N0_UTS = None
    N0_UTS = res.x
    print(f"Best-fitting N0 from UTS fit is: {N0_UTS:.2f}")

    # Use best-fit N0 to generate predictions
    val = out_df.iloc[i, :].to_frame().T # data used to validate
    theta_tot_val = out_df.iloc[i, :]['Sample_total_swc_g'] #swc used to validate
    
    theta_pred_pore_volumetric = val.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['N_pisd_UTS'],
            n0=N0_UTS,
            air_humidity=row['Rhov_cal_g_cm3'],
            bulk_density= row['bd'],
            lattice_water=row['lw'] * row['bd'],
            water_equiv_soil_organic_carbon= row['soc_water'] * row['bd'],
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    ) 

    theta_pred_UTS = theta_pred_pore_volumetric/val['bd'] + val['lw'] + val['soc_water']
    residual_uts = theta_tot_val - theta_pred_UTS.item()
    
    out_df.loc[i,'UTS_Resid_loocv_N0'] = residual_uts
    out_df.loc[i,'UTS_pred_loocv_N0'] = theta_pred_UTS.item()
    out_df.loc[i,'UTS_obs_loocv_N0'] = theta_tot_val
    
UTS_loocv_fit = kling_gupta_efficiency(out_df['UTS_pred_loocv_N0'], out_df['UTS_obs_loocv_N0'])

# identify outliers

# find outliers using inter quartile range
Q1 = np.percentile(out_df['UTS_Resid_loocv_N0'], 25)
Q3 = np.percentile(out_df['UTS_Resid_loocv_N0'], 75)
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
IQR = Q3 - Q1
print(f"IQR: {IQR}")
lower_iqr_bound = Q1 - 1.5 * IQR
upper_iqr_bound = Q3 + 1.5 * IQR
print(f"Lower Bound: {lower_iqr_bound}")
print(f"Upper Bound: {upper_iqr_bound}")

iqr_outliers = [x for x in out_df['UTS_Resid_loocv_N0'] if x < lower_iqr_bound or x > upper_iqr_bound]
print(f"Outliers from IQR: {iqr_outliers}")

# find outliers in the residuals # keep working on these: want to just find 1d outliers from residuals now

sd_resid_uts = np.std(out_df['UTS_Resid_loocv_N0'])
mean_resid_uts = np.mean(out_df['UTS_Resid_loocv_N0'])
upper = mean_resid_uts + 2*sd_resid_uts
lower = mean_resid_uts - 2*sd_resid_uts

outliers = out_df[((out_df['UTS_Resid_loocv_N0']>upper) | (out_df['UTS_Resid_loocv_N0']<lower))]
print(f"Outliers from standard deviation: {outliers}")

keep_df_uts = out_df.drop(outliers.index).copy()
keep_df2_uts = keep_df_uts.reset_index()

# Do LOOCV with soil samples for UTS with outlier removed:

possible = list(range(len(keep_df2_uts)))

for i in possible:
    select = [j for j in possible if j != i] # select all but one index
    subset = keep_df2_uts.iloc[select, :]
    
    # DESILETS METHOD, calibrated with portable data

    # Assuming df is your pandas DataFrame with 'Avg_N0', 'theta_tot', and 'Npvi'
    N0_start = subset['N_pisd_UTS'].mean()

    theta_tot = subset['Sample_total_swc_g'].astype(float).values
    N = subset['N_pisd_UTS'].values

    # Run the minimizer
    res = None
    res = minimize_scalar(objective2, args=(subset), bounds=(N0_start, N0_start + 6000), method='bounded')

    # Best-fit N0 value
    N0_UTS = None
    N0_UTS = res.x
    print(f"Best-fitting N0 from UTS fit is: {N0_UTS:.2f}")

    # Use best-fit N0 to generate predictions
    val = keep_df2_uts.iloc[i, :].to_frame().T # data used to validate
    theta_tot_val = keep_df2_uts.iloc[i, :]['Sample_total_swc_g'] #swc used to validate
    
    theta_pred_pore_volumetric = val.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['N_pisd_UTS'],
            n0=N0_UTS,
            air_humidity=row['Rhov_cal_g_cm3'],
            bulk_density= row['bd'],
            lattice_water=row['lw'] * row['bd'],
            water_equiv_soil_organic_carbon= row['soc_water'] * row['bd'],
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    ) 

    theta_pred_UTS = theta_pred_pore_volumetric/val['bd'] + val['lw'] + val['soc_water']
    residual_uts = theta_tot_val - theta_pred_UTS.item()
    
    keep_df2_uts.loc[i,'UTS_Resid_loocv_N0'] = residual_uts
    keep_df2_uts.loc[i,'UTS_pred_loocv_N0'] = theta_pred_UTS.item()
    keep_df2_uts.loc[i,'UTS_obs_loocv_N0'] = theta_tot_val

    
UTS_loocv_fit_keep = kling_gupta_efficiency(keep_df2_uts['UTS_pred_loocv_N0'], keep_df2_uts['UTS_obs_loocv_N0'])

# single N0 to save using all data points
theta_tot = out_df['Sample_total_swc_g'].astype(float).values
N = out_df['N_pisd_UTS'].values

# Bootstrap parameters
n_bootstrap = 1000  # Number of resamples
N0_bootstrap_UTS = []

# Perform bootstrap resampling
for i in range(n_bootstrap):
    sample_df = out_df.sample(n=len(out_df), replace=True)
    try:
        res_bs = minimize_scalar(objective2, bounds=(N0_start, N0_start + 6000), method='bounded', args=(sample_df, False))
        if res_bs.success and np.isfinite(res_bs.fun):
            N0_bootstrap_UTS.append(res_bs.x)
        else:
            print(f"[Bootstrap {i}] Minimization failed or invalid result. fun: {res_bs.fun}, success: {res_bs.success}")
    except Exception as e:
        print(f"[Bootstrap {i}] Exception: {e}")

# Compute confidence intervals from percentiles
CI_lower_UTS, CI_upper_UTS = np.percentile(N0_bootstrap_UTS, [2.5, 97.5])

print(f"95% Bootstrap Confidence Interval for N0: ({CI_lower_UTS:.4f}, {CI_upper_UTS:.4f})")

res_single_N0 = minimize_scalar(objective2, bounds=(N0_start, N0_start + 6000), method='bounded', args=(out_df, False))
single_UTS_N0 = res_single_N0.x

UTS_ND = ['UTS_ND', single_UTS_N0, CI_lower_UTS, CI_upper_UTS]

# get residuals from predictions using N0 fit to all data
pred_pore_volumetric = out_df.apply(
     lambda row: convert_neutrons_to_soil_moisture_uts(
         neutron_count=row['N_pisd_UTS'],
         n0=N0_UTS,
         air_humidity=row['Rhov_cal_g_cm3'],
         bulk_density= row['bd'],
         lattice_water=row['lw'] * row['bd'],
         water_equiv_soil_organic_carbon= row['soc_water'] * row['bd'],
         method="Mar21_mcnp_drf",
     ), 
     axis=1
 ) 
pred_tot_g_uts = pred_pore_volumetric/out_df['bd'] + out_df['lw'] + out_df['soc_water']
UTS_resid = theta_tot - pred_tot_g_uts
out_df['UTS_resid_N0_fit_to_all'] = UTS_resid

# add to df to save
N0_ls = [Des_N0, UTS_ND]
colnames = ['Param', 'Fit', 'Lower 95 CI', 'Upper 95 CI']
N0_df = pd.DataFrame(N0_ls, columns=colnames)
N0_df.to_csv(f'{outDir}\\Parameter_fit.csv')

'''SAVE THE LOOCV FIT STATS'''

save_dict = {'Des_full':Des_loocv_fit, 'Des_sans_outliers': Des_loocv_fit_keep,
             'UTS_full': UTS_loocv_fit, 'UTS_sans_outliers': UTS_loocv_fit_keep} 
save_ls= [Des_loocv_fit, Des_loocv_fit_keep, UTS_loocv_fit, UTS_loocv_fit_keep]
save_df = pd.concat(save_ls, ignore_index=True)
save_df['label'] = ['Des_full', 'Des_sans_outliers', 'UTS_full', 'UTS_sans_outliers']

save_df.to_csv(f'{outDir}\\LOOCV_stats.csv')

# SOLVE N0 FOR EACH SITE INDIVIDUALLY

N0_df_ls = []
for n in site_names_new:
    
    #n = 'C4'
            
    THIS_SITE_new = n
    
    sitedata = out_df[out_df['NewName']==THIS_SITE_new]
    print(sitedata.shape)
    
    offset = sitedata['Sample_total_swc_g'].item()-sitedata['TDR_total_swc_g'].item()
    TDR_Grav_ratio = sitedata['TDR_total_swc_g'].item()/sitedata['Sample_total_swc_g'].item()
    
    site_bd = sitedata['bd'].item()
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
    site_df = df_dict[THIS_SITE_new]
    # estimate soil moisture for all snow-free days using Desilets equation
    tau = site_lw +site_soc
    
    site_df['scaled_stationary_N_Des'] = site_df['Corrected_Mod_cph_for_Des']*sitedata['N_ratio_raw'].item()
    site_df['scaled_stationary_N_UTS'] = site_df['Corrected_Mod_cph_for_UTS']*sitedata['N_ratio_raw'].item()
    
    site_df['theta_pred_tot_g_Des'] = (0.0808 / (site_df['scaled_stationary_N_Des'] / N0_fit_Des - 0.372) - 0.115)  # totalgravimetric water content
    site_df['theta_pred_pore_vol_Des'] = (site_df['theta_pred_tot_g_Des']- tau) * site_bd
    
    site_df['TDR_pore_swc_g'] = site_df['WeightedTDR_SWC']/site_bd
    site_df['TDR_tot_swc_g'] = site_df['TDR_pore_swc_g'] + tau
    
    site_df['TDR_tot_swc_g_resid_Des'] = site_df['TDR_tot_swc_g'] - site_df['theta_pred_tot_g_Des'] 
    
    # filter out rows with nan
    site_df.dropna(subset = ['TDR_tot_swc_g_resid_Des'], inplace = True)
    site_df['NewName'] = THIS_SITE_new
    
    fit_stat_df = kling_gupta_efficiency(site_df['theta_pred_tot_g_Des'], site_df['TDR_tot_swc_g'])
    fit_stat_df.loc[0,'Site'] = THIS_SITE_new
    fit_stat_df.loc[0,'Method'] = 'Desilets, site-specific N0'
    fit_stat_df.loc[0,'LandCover'] = lc
    fit_stat_df.loc[0,'N0_fit'] = N0_fit_Des
    
    # now do universal N0 prediction 
    site_df['swc_univ_N0_pred_tot_g_Des'] =  (0.0808 / (site_df['scaled_stationary_N_Des'] / N0_univ - 0.372) - 0.115)  # totalgravimetric water content
    site_df['swc_univ_N0_pred_pore_vol_Des'] = (site_df['swc_univ_N0_pred_tot_g_Des'] - site_lw - site_soc) * site_bd
    
    # now each site with UTS method
    
    # Best-fit N0 UTS value
    N0_fit_UTS = None
    N0_fit_UTS = sitedata['N0_fit_UTS'].item()
    print(f'N0 from UTS fit method at Site {THIS_SITE_new} ( {lc} ) is: {N0_fit_UTS:.2f}')
     
    '''
    # need absolute air humidity in g/cm^3 for all data in prediction
    all_RH_cal = site_df['airRH']
    all_T_cal = site_df['airT']
    all_Rhov_cal, _ = calculate_watervapor(
    all_RH_cal, all_T_cal, Config.gama
    ) # output in kg/m^3
    
    print(f'Gama is {Config.gama} for {THIS_SITE_new}')
    
    site_df['Rhov_cal_g_cm3'] = all_Rhov_cal/1000 # in g/cm^3
    '''
    
    theta_pred_pore_volumetric = site_df.apply(
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
    
    site_df['theta_pred_tot_g_UTS'] = (theta_pred_pore_volumetric/site_bd) + tau
    site_df['theta_pred_pore_vol_UTS'] = theta_pred_pore_volumetric
    
    site_df['TDR_pore_swc_g'] = site_df['WeightedTDR_SWC']/site_bd
    site_df['TDR_tot_swc_g'] = site_df['TDR_pore_swc_g'] + site_lw + site_soc
    
    site_df['TDR_tot_swc_g_resid_UTS'] = site_df['TDR_tot_swc_g'] - site_df['theta_pred_tot_g_UTS'] 
    
    # filter out rows with nan in airRH or airT
    site_df.dropna(subset = ['airRH', 'airT'], inplace = True)
    site_df['NewName'] = THIS_SITE_new
    
    
    fit_stat_df_UTS = kling_gupta_efficiency(site_df['theta_pred_tot_g_UTS'], site_df['TDR_tot_swc_g'])
    fit_stat_df_UTS['Site'] = THIS_SITE_new
    fit_stat_df_UTS['Method'] = 'UTS, site-specific N0'
    fit_stat_df_UTS['LandCover'] = lc
    fit_stat_df_UTS['N0_fit'] = N0_fit_UTS
    
    fit_stat_df_out = pd.concat([fit_stat_df, fit_stat_df_UTS] )
    
    # now do universal ND prediction
    theta_pred_pore_volumetric_univ = site_df.apply(
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
    
    site_df['swc_univ_ND_pred_tot_g_UTS'] = theta_pred_pore_volumetric_univ/site_bd + site_lw+ site_soc
    site_df['swc_univ_ND_pred_pore_vol_UTS'] = theta_pred_pore_volumetric_univ
    
    site_df.to_csv(f'{outDir}\\{THIS_SITE_new}_UTS_and_Des_{lc}_SiteSpecificN0_predictions.csv')
    fit_stat_df.to_csv(f'{outDir}\\{THIS_SITE_new}_{THIS_SITE_old}_SiteSpecificN0_FitStats.csv')
    
    # plot residuals
    
    # Create figure and axis
    fig, ax = plt.subplots()
    
    ax.scatter(site_df['scaled_stationary_N_Des'], 
               site_df['TDR_tot_swc_g_resid_Des'], color='blue', label='Desilets', alpha=0.5)
    
    ax.scatter(site_df['scaled_stationary_N_UTS'], 
               site_df['TDR_tot_swc_g_resid_UTS'], color='orange', label='UTS', alpha = 0.5)
    
    #ax.scatter(N, 0, marker='o', facecolors='none', edgecolors='black', label='calibration point')
    #ax.scatter(N_UTS, 0, marker='o', facecolors='none', edgecolors='black')  # no label to avoid duplicate
    
    # Add horizontal line at y = 0
    ax.axhline(0, color='red', linestyle='--')
    
    # Axis labels and title
    ax.set_xlabel('Moderated Neutron Counts (cph)')
    ax.set_ylabel(r'Residuals ($\mathrm{g\ g}^{-1}$)')
    ax.set_title(rf'{THIS_SITE_new} Residual Plot ( $N_{{0}}$ = {N0_fit_Des:.0f})')
    
    # Get unique legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Method', bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    # Tight layout
    fig.tight_layout()
    
    # Save and show
    fig.savefig(f'{outDir}\\{THIS_SITE_new}_resid_SiteSpecific_N0_Des_{lc}.png', dpi=300)
    plt.show()
    
    print(f'Saved individual analysis for {THIS_SITE_new}')
    
    N0_row = pd.DataFrame({'Site': THIS_SITE_new, 'N0_Desilets': N0_fit_Des, 
                           'N0_UTS': N0_fit_UTS, 'GRAV_TDR_offset':offset,
                           'TDR_Grav_ratio':TDR_Grav_ratio}, index = [0])
    N0_df_ls.append(N0_row)
N0_df_out = pd.concat(N0_df_ls, ignore_index=True)
N0_df_out.to_csv(f'{outDir}\\Site_specific_N0_values.csv')

# combine with outdf: 

site_summaries = pd.merge(out_df, N0_df_out, left_on='NewName', right_on='Site', how='inner')
site_summaries.to_csv(f'{outDir}\\Site_Calibration_data_summary.csv')
