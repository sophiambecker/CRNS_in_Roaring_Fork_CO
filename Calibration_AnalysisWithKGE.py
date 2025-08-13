# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:10:36 2025

@author: sbecker14
"""

'''
note this script requires the packages installed in an environment containing requirements1
 
Calculate single N0 from gravimetric data and stationary detector

Predict SWC from moderated counts after applying the ratio of the 
portable/moderated counts using both Desilets and UTS methods
for the time series at all the sites

Find the KGE for each of the sites

Repeat all the above for 3 major land cover groups

Repeat all the above for site-specific N0 values
'''
# libraries

from config import Config
from watervapor import calculate_watervapor
from UTS_helpers import convert_neutrons_to_soil_moisture_uts

import datetime as dt
import pandas as pd
import os
import glob
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr

stamp = dt.datetime.now().strftime("%Y%m%d") 
Dir = "C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Code_for_Github"
outFold = '\\Calibration_AnalysisWithKGE_output_{}'.format(stamp)
                                                 
outDir = os.path.normpath(Dir + os.sep + outFold) + '\\'    # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

os.chdir(Dir)

# load portable calibration data
directory_path_port_des = 'Data\\MockPortableData\\Processed_Portable_Des'
directory_path_port_uts = 'Data\\MockPortableData\\Processed_Portable_UTS'
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
directory_path = 'FilteredSnowFreeData_output20250813'
file_pattern = f'{directory_path}\\*.csv'
file_paths = glob.glob(file_pattern, recursive=True)
dfs_ls = [pd.read_csv(file_path) for file_path in file_paths]
site_names_old = [os.path.basename(fp).split('_')[0] for fp in file_paths]
print(f'Old site names are {site_names_old}')
site_names_new = [os.path.basename(fp).split('_')[1] for fp in file_paths]
print(f'New site names are {site_names_new}')

# make dictionary of dataframes using new site names 
df_dict = dict(zip(site_names_new, dfs_ls))

# load landcover descriptions: 
RF_veginfo = pd.read_csv("Data\\RoaringFork_CRNS_metadata.csv")
RF_veginfo.replace('RF 5', 'RF5', inplace=True) 
veg_dict = dict(zip(RF_veginfo['Original_ID'], RF_veginfo['Land Cover ']))

# dictionary to get old site name based on new site name: 

new_to_old_name = dict(zip(site_names_new, site_names_old))
    
# load calibration day data: 

site_var = pd.read_excel("Data\\Mock Calibration Summary.xlsx")
# Extract row 16 and columns 1–15
sample_series = site_var.iloc[17, 1:16]

# Convert values to just the date, keep column names (keys) as strings

sample_date = {
    col: pd.to_datetime(val)
           .tz_localize('UTC')             # 1. Say it's in UTC
           .tz_convert('America/Denver')   # 2. Convert to Colorado time
           .date()                         # 3. Extract just the date
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
    THIS_SITE_new = s

    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    # Find weighted TDR value corresponding to gravimetric sampling day
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
        
    check_cols = ['Corrected_Mod_cph_for_Des','Corrected_Mod_cph_for_UTS', 'WeightedTDR_SWC', 'Bare']
    
    # check the above columns for nan values. If any column contains an nan value, use the nearest date without nan values as the calibration date. 
    # Check if any NaNs in the row for the calibration date
    if df.loc[df['date'] == cal_date, check_cols].isnull().any(axis=1).any():
        # Filter to rows with no NaNs in the desired columns
        valid_rows = df[df[check_cols].notna().all(axis=1)].copy()
    
        # Find the row with the minimum absolute date difference
        valid_rows['date_diff'] = valid_rows['date'].apply(lambda d: abs(d - cal_date))
        nearest_row = valid_rows.loc[valid_rows['date_diff'].idxmin()]
        
        # Update cal_date
        cal_date = nearest_row['date']
    
    # Now extract cal_data
    cal_data = df[df['date'] == cal_date]
    
    cal_swc = cal_data['WeightedTDR_SWC'].item()
    
    Ncal_Des_p = p_des_df['mod_nc_cph'].mean() # corrected moderated counts for Desilets method, using f3 intesity correction
    
    Ncal_UTS_p = p_uts_df['mod_nc_cph'].mean() # corrected moderated counts for UTS method, using f3 intensity correction
    
    Ncal_Des_st = cal_data['Corrected_Mod_cph_for_Des'].item()
    Ncal_UTS_st = cal_data['Corrected_Mod_cph_for_UTS'].item()
    
    N_ratio_Des = Ncal_Des_p/Ncal_Des_st
    N_ratio_UTS = Ncal_UTS_p/Ncal_UTS_st
    
    # Get other site-specific variables to calculate total gravimetric water content
    
    site_bd = site_var[THIS_SITE_old][24]
    
    swc_g = site_var[THIS_SITE_old][18] # pore water (g/g)
    lw = site_var[THIS_SITE_old][26] # lattice water (g/g)
    soc = site_var[THIS_SITE_old][29] # soc water (g/g)
    
    print(f"{THIS_SITE_new}: bd={site_bd}, lw={lw}, soc={soc}")

    GRAV_swc_tot_g = swc_g + lw + soc
    
    # pore gravimetric water content = (volumetric water content)/bd
    TDR_Pore_g = cal_swc/site_bd
    
    TDR_swc_tot_g = TDR_Pore_g + lw + soc
    
    elev = site_var[THIS_SITE_old][6]
    
    # add forest/non-forest column:
    Canopy = f_clas_dict[THIS_SITE_old]
    LC = veg_dict[THIS_SITE_old]
    
    N0_ratio = site_var[THIS_SITE_old][42] # portable/stationary # this isn't used for anything
    N0_Des = p_des_df['N0_fit'][0]
    N0_UTS = p_uts_df['N0_fit'][0]
    # might need to come back and add BWE estimates
    
    BWE = TotBWE_df[TotBWE_df['Site']==THIS_SITE_old]['BWE Representing 200 m Radius Footprint (mm)'].item()
    BWE_uncer = TotBWE_df[TotBWE_df['Site']==THIS_SITE_old]['BWE Uncertainty (mm)'].item()
    
    row = {'N_Correct_Des': Ncal_Des_p, 'N_Correct_UTS': Ncal_UTS_p, 'Sample_total_swc_g': GRAV_swc_tot_g, 'TDR_total_swc_g': TDR_swc_tot_g, 'bd': site_bd, 
           'lw': lw, 'soc_water': soc, 'Elev': elev, 'Canopy': Canopy, 'landCoverClass': LC, 'OldName': THIS_SITE_old, 'NewName': THIS_SITE_new, 
           'CalDate': cal_date, 'airRH': p_uts_df['airRH'].mean(), 'airT': p_uts_df['airT'].mean(),
            'WeightedTDR': cal_data['WeightedTDR_SWC'].item(), 'ppt': cal_data['PRISM_ppt_mm'].item(), 
           'Bare': cal_data['Bare'].item(), 'BWE_mm': BWE, 'BWE_uncer': BWE_uncer,
           'N0_port_to_stat' : N0_ratio, 'Raw_Mod_cv':cal_data['Raw_Mod_Coeff_of_Var'].item(), 'Raw_Mod_sqrt':cal_data['Raw_Mod_sqrt'].item(),
           'N0_fit_Des': N0_Des, 'N0_fit_UTS': N0_UTS, 
           'N_ratio_Des': N_ratio_Des, 'N_ratio_UTS': N_ratio_UTS}
    
    rows.append(row)

out_df = pd.DataFrame(rows)

# CALIBRATE N0 WITH ENTIRE DATA SET

# DESILETS METHOD, calibrated with portable data

# Assuming df is your pandas DataFrame with 'Avg_N0', 'theta_tot', and 'Npvi'
N0_start = out_df['N_Correct_Des'].mean()

theta_tot = out_df['Sample_total_swc_g'].astype(float).values
N = out_df['N_Correct_Des'].values

# Define the model function: f(N, N0)
def model(N, N0):
    return 0.0808 / (N / N0 - 0.372) - 0.115

# Fit the model using curve_fit (nonlinear least squares)
popt, pcov = curve_fit(model, N, theta_tot, p0=[N0_start])

# Extract fitted N0
N0_fit_Des = popt[0]

print(f'N0 from Desilets fit is: {N0_fit_Des}')

theta_pred = model(N, *popt)
residuals = theta_tot - theta_pred

out_df['Des_resid_singleN0'] = residuals

# plot residuals
plt.scatter(N, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('N')
plt.ylabel('Residuals')
plt.title('Desilets Residual Plot')
plt.show()

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

# use N0_fitted to predict swc at all sites

all_Des_Pred_SingleN0 = []
all_Des_SingleN0_SiteFitStats = []

for s in site_names_new: 
    bd=None
    sitedata = None
    df = None
    
    THIS_SITE_new = s
  
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    # Find weighted TDR value corresponding to gravimetric sampling day
    df = df_dict[THIS_SITE_new]
    sitedata = out_df[out_df['NewName']==THIS_SITE_new]
    df.loc[:,'scaled_stationary_N_Des'] = df['Corrected_Mod_cph_for_Des']*sitedata['N_ratio_Des'].item()
    
    df['site_theta_pred'] = model(df['scaled_stationary_N_Des'], *popt) # prediction of total gravimetric water content
    
    offset = sitedata['Sample_total_swc_g'].item()-sitedata['TDR_total_swc_g'].item()
    
    bd = sitedata['bd'].item()
    
    df['TDR_pore_swc_g'] = df['WeightedTDR_SWC']/bd
    df['TDR_tot_swc_g'] = df['TDR_pore_swc_g'] + sitedata['lw'].item() + sitedata['soc_water'].item()
    
    df['TDR_tot_swc_g_resid_Des'] = df['TDR_tot_swc_g'] - df['site_theta_pred'] 
    
    site_por = 1 - (bd/2.65) # site porosity from particle density of 2.65 g/cm^3
    
    # filter out TDR values over saturation
    
    cal_bias_vwc = offset*bd
    TDR_max = site_por - cal_bias_vwc
    
    df = df[df['WeightedTDR_SWC'] <= TDR_max - 0.05]
    
    # filter out rows with nan
    df.dropna(subset = ['TDR_tot_swc_g_resid_Des'], inplace = True)
    
    if df.empty:
        print(f"Warning: {THIS_SITE_new} has no valid rows after filtering.")
        continue

    df['NewName'] = THIS_SITE_new
    
    df['Des_upper_95_pred'] = model(df['scaled_stationary_N_Des'], CI_upper_Des)
    df['Des_lower_95_pred'] = model(df['scaled_stationary_N_Des'], CI_lower_Des)
    
    all_Des_Pred_SingleN0.append(df)
    
    fit_stat_df = kling_gupta_efficiency(df['site_theta_pred'], df['TDR_tot_swc_g'])
    fit_stat_df['Site'] = THIS_SITE_new
    fit_stat_df['Method'] = 'Desilets, single N0 for all sites'
    fit_stat_df['N0_fit'] = N0_fit_Des
    fit_stat_df['N0_CI_upper'] = CI_upper_Des
    fit_stat_df['N0_CI_lower'] = CI_lower_Des
    all_Des_SingleN0_SiteFitStats.append(fit_stat_df)
    
all_Des_Pred_SingleN0_df = pd.concat(all_Des_Pred_SingleN0, ignore_index=True)
all_Des_SingleN0_SiteFitStats_df = pd.concat(all_Des_SingleN0_SiteFitStats, ignore_index = True)

all_Des_Pred_SingleN0_df.to_csv(f'{outDir}\\all_Des_SingleN0_predictions.csv')
all_Des_SingleN0_SiteFitStats_df.to_csv(f'{outDir}\\all_Des_SingleN0_SiteFitStats.csv')

# Assuming 'all_UTS_Pred_SingleN0_df' is your DataFrame
all_Des_Pred_SingleN0_df['category_num'] = pd.factorize(all_Des_Pred_SingleN0_df['NewName'])[0]

# Use the 'tab20' colormap (it has 20 distinct colors)
cmap = plt.cm.tab20  # Choose the colormap (tab20 has 20 distinct colors)
colors = cmap(all_Des_Pred_SingleN0_df['category_num'] / max(all_Des_Pred_SingleN0_df['category_num']))  # Normalize for colormap

# Scatter plot
plt.scatter(all_Des_Pred_SingleN0_df['scaled_stationary_N_Des'], 
            all_Des_Pred_SingleN0_df['TDR_tot_swc_g_resid_Des'],
            c=colors, alpha=0.5)

# Add horizontal line at y=0
plt.axhline(0, color='red', linestyle='--')

# Add labels and title
plt.xlabel('Moderated Neutron Counts (cph)')
plt.ylabel(r'$\text{Residuals (g g}^{-1})$')
plt.title(r'Desilets Residual Plot (All Data, single $N_{0}$)')

plt.ylim(-2.05, 0.3)
# Create one label for each category in the dataframe
handles = []
for category in all_Des_Pred_SingleN0_df['NewName'].unique():
    handle = plt.Line2D([0], [0], marker='o', color='w', 
                        markerfacecolor=cmap(all_Des_Pred_SingleN0_df['category_num'][all_Des_Pred_SingleN0_df['NewName'] == category].iloc[0] / max(all_Des_Pred_SingleN0_df['category_num'])),
                        markersize=10, label=category)
    handles.append(handle)

# Add the legend with category names, positioned off to the right of the plot
plt.legend(handles=handles, title="Site", loc='center left', bbox_to_anchor=(1.05, 0.5))

# Adjust layout to ensure the plot doesn't get cut off
plt.tight_layout()

plt.savefig(f'{outDir}\\Site_resid_single_N0_Des_{stamp}', dpi = 300)
# Show plot
plt.show()


# UTS METHOD #############################################################################

# need absolute air humidity in g/cm^3
RH_cal = out_df['airRH']
T_cal = out_df['airT']
Rhov_cal, _ = calculate_watervapor(
RH_cal, T_cal, Config.gama
) # output in kg/m^3

out_df['Rhov_cal_g_cm3'] = Rhov_cal/1000 # in g/cm^3


# Objective function to minimize: total absolute difference over all rows
'''def objective(N0):
    swc_pred = []
    for idx, row in out_df.iterrows():
        try:
            swc_val = convert_neutrons_to_soil_moisture_uts(
                neutron_count=row['N_Correct_UTS'],
                n0=N0,
                air_humidity=row['Rhov_cal_g_cm3'],
                bulk_density=row['bd'],
                lattice_water=row['lw'] * row['bd'],
                water_equiv_soil_organic_carbon= row['soc_water'] * row['bd'],
                method="Mar21_mcnp_drf"
            )
        except:
            swc_val = np.nan
        swc_pred.append(swc_val)
    
    swc_pred = np.array(swc_pred)
    actual_swc = (out_df['Sample_total_swc_g'].values - out_df['lw']-out_df['soc_water'])*out_df['bd'] # in volumetric water content
    valid = ~np.isnan(swc_pred)
    
    return np.sum(np.abs(swc_pred[valid] - actual_swc[valid]))
'''
def objective2(N0, df, verbose=False):
    swc_pred = []
    invalid_rows = 0

    for idx, row in df.iterrows():
        try:
            neutron_count = row['N_Correct_UTS']
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
  
    '''
    residuals = swc_pred[valid] - actual_swc[valid]
    bias = np.mean(residuals)
    unbiased_rmse = np.sqrt(np.mean((residuals - bias) ** 2))

    if verbose:
        print(f"Valid rows: {np.sum(valid)}, Invalid rows: {invalid_rows}, Unbiased RMSE: {unbiased_rmse:.4f}")

    return unbiased_rmse
    '''
    '''
    residuals = swc_pred[valid] - actual_swc[valid]
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    if verbose:
        print(f"Valid rows: {np.sum(valid)}, Invalid rows: {invalid_rows}, RMSE: {rmse:.4f}")
    
    return rmse
    '''
    '''
    residuals = swc_pred[valid] - actual_swc[valid]
    sse = np.sum(residuals ** 2)
    
    if verbose:
        print(f"Valid rows: {np.sum(valid)}, Invalid rows: {invalid_rows}, RMSE: {sse:.4f}")
    
    return sse
    '''
# Run the minimizer
res = minimize_scalar(objective2, args=(out_df,), bounds=(N0_start, N0_start + 6000), method='bounded')

# Best-fit N0 value
N0_UTS = None
N0_UTS = res.x
print(f"Best-fitting N0 from UTS fit is: {N0_UTS:.2f}")

# Use best-fit N0 to generate predictions

theta_pred_pore_volumetric = out_df.apply(
    lambda row: convert_neutrons_to_soil_moisture_uts(
        neutron_count=row['N_Correct_UTS'],
        n0=N0_UTS,
        air_humidity=row['Rhov_cal_g_cm3'],
        bulk_density= row['bd'],
        lattice_water=row['lw'] * row['bd'],
        water_equiv_soil_organic_carbon= row['soc_water'] * row['bd'],
        method="Mar21_mcnp_drf",
    ), 
    axis=1
) 

theta_pred_UTS = theta_pred_pore_volumetric/out_df['bd'] + out_df['lw'] + out_df['soc_water']
residuals_uts = theta_tot - theta_pred_UTS
out_df['UTS_Resid_singleN0'] = residuals_uts
# plot residuals
plt.scatter(N, residuals_uts)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('N')
plt.ylabel('Residuals')
plt.title('UTS Residual Plot')
plt.show()

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


all_UTS_Pred_SingleN0 = []
all_UTS_SingleN0_SiteFitStats = []
for s in site_names_new:
    sitedata = None
    df = None
    
    THIS_SITE_new = s
     
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    # Find weighted TDR value corresponding to gravimetric sampling day
    df = df_dict[THIS_SITE_new]
    
    sitedata = out_df[out_df['NewName']==THIS_SITE_new]
    offset = sitedata['Sample_total_swc_g'].item()-sitedata['TDR_total_swc_g'].item()
    df.loc[:,'scaled_stationary_N_UTS'] = df['Corrected_Mod_cph_for_UTS']*sitedata['N_ratio_UTS'].item()
    
    site_bd = sitedata['bd'].item()
    site_lw = sitedata['lw'].item()
    site_soc = sitedata['soc_water'].item()
    
    site_por = 1 - (site_bd/2.65) # site porosity from particle density of 2.65 g/cm^3
    
    # filter out TDR values over saturation
    
    cal_bias_vwc = offset*site_bd
    TDR_max = site_por - cal_bias_vwc
    df = df[df['WeightedTDR_SWC'] <= TDR_max - 0.05]
    
    print(f"{THIS_SITE_new}: bd={site_bd}, lw={site_lw}, soc={site_soc}")

    # need absolute air humidity in g/cm^3
    RH_cal = df['airRH']
    T_cal = df['airT']
    Rhov_cal, _ = calculate_watervapor(
    RH_cal, T_cal, Config.gama
    ) # output in kg/m^3
    
    df.loc[:, 'Rhov_cal_g_cm3'] = Rhov_cal/1000 # in g/cm^3
    
    theta_pred_pore_volumetric = df.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['scaled_stationary_N_UTS'],
            n0=N0_UTS,
            air_humidity=row['Rhov_cal_g_cm3'],
            bulk_density= site_bd,
            lattice_water=site_lw* site_bd,
            water_equiv_soil_organic_carbon= site_soc * site_bd,
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    ) 
    
    df['theta_pred_tot_g_UTS'] = theta_pred_pore_volumetric/site_bd + site_lw+ site_soc
    
    df['TDR_pore_swc_g'] = df['WeightedTDR_SWC']/site_bd
    df['TDR_tot_swc_g'] = df['TDR_pore_swc_g'] + site_lw + site_soc
    
    df['TDR_tot_swc_g_resid_UTS'] = df['TDR_tot_swc_g'] - df['theta_pred_tot_g_UTS'] # wo: "With offest"

    # filter out rows with nan
    df.dropna(subset = ['TDR_tot_swc_g_resid_UTS'], inplace = True)
    df['NewName'] = THIS_SITE_new

    all_UTS_Pred_SingleN0.append(df)
    
    fit_stat_df = kling_gupta_efficiency(df['theta_pred_tot_g_UTS'], df['TDR_tot_swc_g'])
    fit_stat_df['Site'] = THIS_SITE_new
    fit_stat_df['Method'] = 'UTS, single N0 for all sites'
    fit_stat_df['N0_fit'] = N0_UTS
    fit_stat_df['N0_upper_CI'] = CI_upper_UTS
    fit_stat_df['N0_lower_CI'] = CI_lower_UTS
    
    all_UTS_SingleN0_SiteFitStats.append(fit_stat_df)
    
all_UTS_Pred_SingleN0_df = pd.concat(all_UTS_Pred_SingleN0, ignore_index=True)
all_UTS_SingleN0_SiteFitStats_df = pd.concat(all_UTS_SingleN0_SiteFitStats, ignore_index = True)
all_UTS_r_sq = r_squared_from_definition(observed = all_UTS_Pred_SingleN0_df['TDR_tot_swc_g'], 
                                         predicted = all_UTS_Pred_SingleN0_df['theta_pred_tot_g_UTS'])

all_UTS_Pred_SingleN0_df.to_csv(f'{outDir}\\all_UTS_SingleN0_predictions.csv')
all_UTS_SingleN0_SiteFitStats_df.to_csv(f'{outDir}\\all_UTS_SingleN0_SiteFitStats.csv')

# plot residuals
# Automatically map categorical 'category' values to integers using factorize

# Assuming 'all_UTS_Pred_SingleN0_df' is your DataFrame
all_UTS_Pred_SingleN0_df['category_num'] = pd.factorize(all_UTS_Pred_SingleN0_df['NewName'])[0]

# Use the 'tab20' colormap (it has 20 distinct colors)
cmap = plt.cm.tab20  # Choose the colormap (tab20 has 20 distinct colors)
colors = cmap(all_UTS_Pred_SingleN0_df['category_num'] / max(all_UTS_Pred_SingleN0_df['category_num']))  # Normalize for colormap

# Scatter plot
plt.scatter(all_UTS_Pred_SingleN0_df['scaled_stationary_N_UTS'], 
            all_UTS_Pred_SingleN0_df['TDR_tot_swc_g_resid_UTS'],
            c=colors, alpha=0.5)

# Add horizontal line at y=0
plt.axhline(0, color='red', linestyle='--')

# Add labels and title
plt.xlabel('Moderated Neutron Counts (cph)')
plt.ylabel(r'$\text{Residuals (g g}^{-1})$')
plt.title(r'UTS Residual Plot (All Data, single $N_{0}$)')
plt.ylim(-2.05, 0.3)
# Create one label for each category in the dataframe
handles = []
for category in all_UTS_Pred_SingleN0_df['NewName'].unique():
    handle = plt.Line2D([0], [0], marker='o', color='w', 
                        markerfacecolor=cmap(all_UTS_Pred_SingleN0_df['category_num'][all_UTS_Pred_SingleN0_df['NewName'] == category].iloc[0] / max(all_UTS_Pred_SingleN0_df['category_num'])),
                        markersize=10, label=category)
    handles.append(handle)

# Add the legend with category names, positioned off to the right of the plot
plt.legend(handles=handles, title="Site", loc='center left', bbox_to_anchor=(1.05, 0.5))

# Adjust layout to ensure the plot doesn't get cut off
plt.tight_layout()
plt.savefig(f'{outDir}\\Site_resid_single_N0_UTS_{stamp}', dpi = 300)
# Show plot
# Show plot
plt.show()

#### NOW REPEAT FITS FOR FOREST, HALF-FOREST, AND NON-FOREST GROUPS ##########

f_df = out_df.loc[out_df['Canopy']=='forest',]  
half_df =   out_df.loc[out_df['Canopy']=='half-forest',]  
non_f_df = out_df.loc[out_df['Canopy']=='non-forest',]  

out_df_ls = [f_df, half_df, non_f_df]

for a_df in out_df_ls:
    
    lc = a_df['Canopy'].iloc[0]
    # Assuming df is your pandas DataFrame with 'Avg_N0', 'theta_tot', and 'Npvi'
    N0_start = a_df['N_Correct_Des'].mean()
    
    theta_tot = a_df['Sample_total_swc_g'].astype(float).values
    N = None
    N = a_df['N_Correct_Des'].values
    
    # Define the model function: f(N, N0)
    def model(N, N0):
        return 0.0808 / (N / N0 - 0.372) - 0.115
    
    # Fit the model using curve_fit (nonlinear least squares)
    popt, pcov = curve_fit(model, N, theta_tot, p0=[N0_start])
    
    # Extract fitted N0
    N0_fit_Des = popt[0]
    
    print(f'N0 from Desilets fit for {lc} is: {N0_fit_Des}')
    
    theta_pred = model(N, *popt)
    residuals = theta_tot - theta_pred
    
    # plot residuals
    plt.scatter(N, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('N')
    plt.ylabel('Residuals')
    plt.title(f'{lc} Desilets Residual Plot')
    plt.show()
    
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

    print(f"95% Bootstrap Confidence Interval for N0 in {lc}: ({CI_lower_Des:.4f}, {CI_upper_Des:.4f})")
    
    # use N0_fitted to predict swc at all sites
    
    all_Des_Pred_SingleN0 = []
    all_Des_SingleN0_SiteFitStats = []
    
    a_site_names_new = a_df['NewName']
    for s in a_site_names_new: 
        df = None
        sitedata = None
        bd = None
        THIS_SITE_new = s
         
        THIS_SITE_old = new_to_old_name[THIS_SITE_new]
        sitedata = out_df[out_df['NewName']==THIS_SITE_new]
         
        # Find weighted TDR value corresponding to gravimetric sampling day
        df = df_dict[THIS_SITE_new]
        df.loc[:,'scaled_stationary_N_Des'] = df['Corrected_Mod_cph_for_Des']*sitedata['N_ratio_Des'].item()
        
        df['site_theta_pred'] = model(df['scaled_stationary_N_Des'], *popt) # prediction of total gravimetric water content
        
       
        offset = sitedata['Sample_total_swc_g'].item()-sitedata['TDR_total_swc_g'].item()
        
        bd = sitedata['bd'].item()
        
        df['TDR_pore_swc_g'] = df['WeightedTDR_SWC']/bd
        df['TDR_tot_swc_g'] = df['TDR_pore_swc_g'] + sitedata['lw'].item() + sitedata['soc_water'].item()
        
        df['TDR_tot_swc_g_resid_Des'] = df['TDR_tot_swc_g'] - df['site_theta_pred'] 
        
        # filter out rows with nan
        df.dropna(subset = ['TDR_tot_swc_g_resid_Des'], inplace = True)
        df['NewName'] = THIS_SITE_new
    
        all_Des_Pred_SingleN0.append(df)
        
        fit_stat_df = kling_gupta_efficiency(df['site_theta_pred'], df['TDR_tot_swc_g'])
        fit_stat_df['Site'] = THIS_SITE_new
        fit_stat_df['Method'] = 'Desilets, single N0 for forest, half-forest, or non-forest'
        fit_stat_df.loc[0,'N0_fit'] = N0_fit_Des
        fit_stat_df['N0_CI_upper'] = CI_upper_Des
        fit_stat_df['N0_CI_lower'] = CI_lower_Des
        all_Des_SingleN0_SiteFitStats.append(fit_stat_df)
        
    all_Des_Pred_SingleN0_df = pd.concat(all_Des_Pred_SingleN0, ignore_index=True)
    all_Des_SingleN0_SiteFitStats_df = pd.concat(all_Des_SingleN0_SiteFitStats, ignore_index = True)
    
    all_Des_Pred_SingleN0_df.to_csv(f'{outDir}\\all_Des_{lc}_SingleN0_predictions.csv')
    all_Des_SingleN0_SiteFitStats_df.to_csv(f'{outDir}\\all_Des_{lc}_SingleN0_SiteFitStats.csv')
    
    # Assuming 'all_UTS_Pred_SingleN0_df' is your DataFrame
    all_Des_Pred_SingleN0_df['category_num'] = pd.factorize(all_Des_Pred_SingleN0_df['NewName'])[0]
    
    # Use the 'tab20' colormap (it has 20 distinct colors)
    cmap = plt.cm.tab20  # Choose the colormap (tab20 has 20 distinct colors)
    colors = cmap(all_Des_Pred_SingleN0_df['category_num'] / max(all_Des_Pred_SingleN0_df['category_num']))  # Normalize for colormap
    
    # Scatter plot
    plt.scatter(all_Des_Pred_SingleN0_df['scaled_stationary_N_Des'], 
                all_Des_Pred_SingleN0_df['TDR_tot_swc_g_resid_Des'],
                c=colors, alpha=0.5)
    
    # Add horizontal line at y=0
    plt.axhline(0, color='red', linestyle='--')
    
    # Add labels and title
    plt.xlabel('Moderated Neutron Counts (cph)')
    plt.ylabel(r'$\text{Residuals (g g}^{-1})$')
    plt.title(rf'{lc} Desilets Residual Plot (All Data, single $N_{{0}}$)')
    
    plt.ylim(-0.2, 0.2)
    # Create one label for each category in the dataframe
    handles = []
    for category in all_Des_Pred_SingleN0_df['NewName'].unique():
        handle = plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=cmap(all_Des_Pred_SingleN0_df['category_num'][all_Des_Pred_SingleN0_df['NewName'] == category].iloc[0] / max(all_Des_Pred_SingleN0_df['category_num'])),
                            markersize=10, label=category)
        handles.append(handle)
    
    # Add the legend with category names, positioned off to the right of the plot
    plt.legend(handles=handles, title="Site", loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    # Adjust layout to ensure the plot doesn't get cut off
    plt.tight_layout()
    
    plt.savefig(f'{outDir}\\Site_resid_single_N0_Des_{lc}__{stamp}', dpi = 300)
    # Show plot
    plt.show()

    # now each landcover with UTS method
    
    # need absolute air humidity in g/cm^3
    RH_cal = a_df['airRH']
    T_cal = a_df['airT']
    Rhov_cal, _ = calculate_watervapor(
    RH_cal, T_cal, Config.gama
    ) # output in kg/m^3
    
    a_df.loc[:, 'Rhov_cal_g_cm3'] = Rhov_cal/1000 # in g/cm^3
    
    
    # Run the minimizer
    res = minimize_scalar(objective2, args=(a_df,), bounds=(N0_start, N0_start + 6000), method='bounded')
    
    # Best-fit N0 value
    N0_UTS = res.x
    print(f"Best-fitting N0 from UTS fit for {lc} is: {N0_UTS:.2f}")
    
    # Use best-fit N0 to generate predictions for calibration days
    
    theta_pred_pore_volumetric = a_df.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['N_Correct_UTS'],
            n0=N0_UTS,
            air_humidity=row['Rhov_cal_g_cm3'],
            bulk_density= row['bd'],
            lattice_water=row['lw'] * row['bd'],
            water_equiv_soil_organic_carbon= row['soc_water'] * row['bd'],
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    ) 
    
    theta_pred_UTS = theta_pred_pore_volumetric/a_df['bd'] + a_df['lw'] + a_df['soc_water']
    residuals = theta_tot - theta_pred_UTS
    
    # plot residuals
    plt.scatter(N, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('N')
    plt.ylabel('Residuals')
    plt.title(f'UTS {lc}_Residual Plot')
    plt.show()
    
    # Bootstrap parameters
    n_bootstrap = 1000  # Number of resamples
    N0_bootstrap_UTS = []

    # Perform bootstrap resampling
    for i in range(n_bootstrap):
        sample_df = a_df.sample(n=len(out_df), replace=True)
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
    
    all_UTS_Pred_SingleN0 = []
    all_UTS_SingleN0_SiteFitStats = []
    for s in a_site_names_new:
        df = None
        sitedata = None
        THIS_SITE_new = s
         
        THIS_SITE_old = new_to_old_name[THIS_SITE_new]
        
        # Find weighted TDR value corresponding to gravimetric sampling day
        df = df_dict[THIS_SITE_new]
        
        sitedata = out_df[out_df['NewName']==THIS_SITE_new]
        offset = sitedata['Sample_total_swc_g'].item()-sitedata['TDR_total_swc_g'].item()
        
        df.loc[:,'scaled_stationary_N_UTS'] = df['Corrected_Mod_cph_for_UTS']*sitedata['N_ratio_UTS'].item()
    
        site_bd = sitedata['bd'].item()
        site_lw = sitedata['lw'].item()
        site_soc = sitedata['soc_water'].item()
        
        # need absolute air humidity in g/cm^3
        RH_cal = df['airRH']
        T_cal = df['airT']
        Rhov_cal, _ = calculate_watervapor(
        RH_cal, T_cal, Config.gama
        ) # output in kg/m^3
        
        df['Rhov_cal_g_cm3'] = Rhov_cal/1000 # in g/cm^3
        
        theta_pred_pore_volumetric = df.apply(
            lambda row: convert_neutrons_to_soil_moisture_uts(
                neutron_count=row['scaled_stationary_N_UTS'],
                n0=N0_UTS,
                air_humidity=row['Rhov_cal_g_cm3'],
                bulk_density= site_bd,
                lattice_water=site_lw* site_bd,
                water_equiv_soil_organic_carbon= site_soc * site_bd,
                method="Mar21_mcnp_drf",
            ), 
            axis=1
        ) 
        
        df['theta_pred_tot_g_UTS'] = theta_pred_pore_volumetric/site_bd + site_lw+ site_soc
        
        df['TDR_pore_swc_g'] = df['WeightedTDR_SWC']/site_bd
        df['TDR_tot_swc_g'] = df['TDR_pore_swc_g'] + site_lw + site_soc
        
        df['TDR_tot_swc_g_resid_UTS'] = df['TDR_tot_swc_g'] - df['theta_pred_tot_g_UTS'] 
    
        # filter out rows with nan
        df.dropna(subset = ['TDR_tot_swc_g_resid_UTS'], inplace = True)
        df['NewName'] = THIS_SITE_new
    
        all_UTS_Pred_SingleN0.append(df)
        
        fit_stat_df = kling_gupta_efficiency(df['theta_pred_tot_g_UTS'], df['TDR_tot_swc_g'])
        fit_stat_df['Site'] = THIS_SITE_new
        fit_stat_df['Method'] = 'UTS, single N0 for forest, half-forest, or non-forest'
        fit_stat_df.loc[0,'N0_fit'] = N0_UTS
        fit_stat_df['N0_upper_CI'] = CI_upper_UTS
        fit_stat_df['N0_lower_CI'] = CI_lower_UTS
        all_UTS_SingleN0_SiteFitStats.append(fit_stat_df)
        
    all_UTS_Pred_SingleN0_df = pd.concat(all_UTS_Pred_SingleN0, ignore_index=True)
    all_UTS_SingleN0_SiteFitStats_df = pd.concat(all_UTS_SingleN0_SiteFitStats, ignore_index = True)
    all_UTS_r_sq = r_squared_from_definition(observed = all_UTS_Pred_SingleN0_df['TDR_tot_swc_g'], 
                                             predicted = all_UTS_Pred_SingleN0_df['theta_pred_tot_g_UTS'])
    
    all_UTS_Pred_SingleN0_df.to_csv(f'{outDir}\\all_UTS_{lc}_SingleN0_predictions.csv')
    all_UTS_SingleN0_SiteFitStats_df.to_csv(f'{outDir}\\all_UTS_{lc}_SingleN0_SiteFitStats.csv')
    
    # plot residuals
    # Automatically map categorical 'category' values to integers using factorize
    
    # Assuming 'all_UTS_Pred_SingleN0_df' is your DataFrame
    all_UTS_Pred_SingleN0_df['category_num'] = pd.factorize(all_UTS_Pred_SingleN0_df['NewName'])[0]
    
    # Use the 'tab20' colormap (it has 20 distinct colors)
    cmap = plt.cm.tab20  # Choose the colormap (tab20 has 20 distinct colors)
    colors = cmap(all_UTS_Pred_SingleN0_df['category_num'] / max(all_UTS_Pred_SingleN0_df['category_num']))  # Normalize for colormap
    
    # Scatter plot
    plt.scatter(all_UTS_Pred_SingleN0_df['Corrected_Mod_cph_for_UTS'], 
                all_UTS_Pred_SingleN0_df['TDR_tot_swc_g_resid_UTS'],
                c=colors, alpha=0.5)
    
    # Add horizontal line at y=0
    plt.axhline(0, color='red', linestyle='--')
    
    # Add labels and title
    plt.xlabel('Moderated Neutron Counts (cph)')
    plt.ylabel(r'$\text{Residuals (g g}^{-1})$')
    plt.title(rf' {lc} UTS Residual Plot (All Data, single $N_{{0}}$)')
    plt.ylim(-0.2, 0.2)
    # Create one label for each category in the dataframe
    handles = []
    for category in all_UTS_Pred_SingleN0_df['NewName'].unique():
        handle = plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=cmap(all_UTS_Pred_SingleN0_df['category_num'][all_UTS_Pred_SingleN0_df['NewName'] == category].iloc[0] / max(all_UTS_Pred_SingleN0_df['category_num'])),
                            markersize=10, label=category)
        handles.append(handle)
    
    # Add the legend with category names, positioned off to the right of the plot
    plt.legend(handles=handles, title="Site", loc='center left', bbox_to_anchor=(1.05, 0.5))
    
    # Adjust layout to ensure the plot doesn't get cut off
    plt.tight_layout()
    plt.savefig(f'{outDir}\\Site_resid_single_N0_UTS_{lc}_{stamp}', dpi = 300)
    # Show plot
    # Show plot
    plt.show()
    

# SOLVE N0 FOR EACH SITE INDIVIDUALLY

N0_df_ls = []
for n in site_names_new:

    #n = 'F2' # these plots do not look good
            
    THIS_SITE_new = n
    
    sitedata = out_df[out_df['NewName']==THIS_SITE_new]
    print(sitedata.shape)
    
    offset = sitedata['Sample_total_swc_g'].item()-sitedata['TDR_total_swc_g'].item()
    TDR_Grav_ratio = sitedata['TDR_total_swc_g'].item()/sitedata['Sample_total_swc_g'].item()
    
    site_bd = sitedata['bd'].item()
    lc = sitedata['Canopy'].iloc[0]
    
    site_lw = sitedata['lw'].item()
    site_soc = sitedata['soc_water'].item()
    
    # Assuming df is your pandas DataFrame with 'Avg_N0', 'theta_tot', and 'Npvi'
    
    theta_tot = sitedata['Sample_total_swc_g'].astype(float).item()
    N_Des = sitedata['N_Correct_Des'].item()
    N_UTS = sitedata['N_Correct_UTS'].item()
    
    # Find N0 based on single gravimetric sample value with Desilets method
    N0_fit_Des = sitedata['N0_fit_Des'].item()
    
    print(f'N0 from Desilets fit for {THIS_SITE_new} ( {lc} ) is: {N0_fit_Des}')
    
    # use N0_fitted to predict swc at site
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    # Find weighted TDR value corresponding to gravimetric sampling day
    site_df = df_dict[THIS_SITE_new]
    # estimate soil moisture for all snow-free days using Desilets equation
    tau = site_lw +site_soc
    
    site_df['scaled_stationary_N_Des'] = site_df['Corrected_Mod_cph_for_Des']*sitedata['N_ratio_Des'].item()
    site_df['scaled_stationary_N_UTS'] = site_df['Corrected_Mod_cph_for_UTS']*sitedata['N_ratio_UTS'].item()
    
    site_df['theta_pred_tot_g_Des'] = (0.0808 / (site_df['scaled_stationary_N_Des'] / N0_fit_Des - 0.372) - 0.115)  # totalgravimetric water content
    
    site_df['TDR_pore_swc_g'] = site_df['WeightedTDR_SWC']/site_bd
    site_df['TDR_tot_swc_g'] = site_df['TDR_pore_swc_g'] + sitedata['lw'].item() + sitedata['soc_water'].item()
    
    site_df['TDR_tot_swc_g_resid_Des'] = site_df['TDR_tot_swc_g'] - site_df['theta_pred_tot_g_Des'] 
    
    # filter out rows with nan
    site_df.dropna(subset = ['TDR_tot_swc_g_resid_Des'], inplace = True)
    site_df['NewName'] = THIS_SITE_new
    
    fit_stat_df = kling_gupta_efficiency(site_df['theta_pred_tot_g_Des'], site_df['TDR_tot_swc_g'])
    fit_stat_df.loc[0,'Site'] = THIS_SITE_new
    fit_stat_df.loc[0,'Method'] = 'Desilets, site-specific N0'
    fit_stat_df.loc[0,'LandCover'] = lc
    fit_stat_df.loc[0,'N0_fit'] = N0_fit_Des
    
    # now each site with UTS method
    
    # Best-fit N0 UTS value
    N0_fit_UTS = sitedata['N0_fit_UTS'].item()
    print(f'N0 from UTS fit method at Site {THIS_SITE_new} ( {lc} ) is: {N0_fit_UTS:.2f}')
       
    # need absolute air humidity in g/cm^3 for all data in prediction
    all_RH_cal = site_df['airRH']
    all_T_cal = site_df['airT']
    all_Rhov_cal, _ = calculate_watervapor(
    all_RH_cal, all_T_cal, Config.gama
    ) # output in kg/m^3
    
    print(f'Gama is {Config.gama} for {THIS_SITE_new}')
    
    site_df['Rhov_cal_g_cm3'] = all_Rhov_cal/1000 # in g/cm^3
    
    theta_pred_pore_volumetric = site_df.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['scaled_stationary_N_UTS'],
            n0=N0_UTS,
            air_humidity=row['Rhov_cal_g_cm3'],
            bulk_density= site_bd,
            lattice_water=site_lw* site_bd,
            water_equiv_soil_organic_carbon= site_soc * site_bd,
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    ) 
    
    site_df['theta_pred_tot_g_UTS'] = theta_pred_pore_volumetric/site_bd + site_lw+ site_soc
    
    site_df['TDR_pore_swc_g'] = site_df['WeightedTDR_SWC']/site_bd
    site_df['TDR_tot_swc_g'] = site_df['TDR_pore_swc_g'] + site_lw + site_soc
    
    site_df['TDR_tot_swc_g_resid_UTS'] = site_df['TDR_tot_swc_g'] - site_df['theta_pred_tot_g_UTS'] 
    
    # filter out rows with nan
    site_df.dropna(subset = ['TDR_tot_swc_g_resid_UTS'], inplace = True)
    site_df['NewName'] = THIS_SITE_new
    
    
    fit_stat_df_UTS = kling_gupta_efficiency(site_df['theta_pred_tot_g_UTS'], site_df['TDR_tot_swc_g'])
    fit_stat_df_UTS['Site'] = THIS_SITE_new
    fit_stat_df_UTS['Method'] = 'UTS, site-specific N0'
    fit_stat_df_UTS['LandCover'] = lc
    fit_stat_df_UTS['N0_fit'] = N0_fit_UTS
    
    fit_stat_df_out = pd.concat([fit_stat_df, fit_stat_df_UTS] )
    
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
