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
outFold = '\\Calibration_AnalysisWithCorrelation_output_{}'.format(stamp)
                                                 
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

# Round numeric values to a fixed number of significant figures.
def round_sigfig(value, sig=2):
    if pd.isna(value):
        return value
    try:
        return float(f"{value:.{sig}g}")
    except (ValueError, TypeError):
        return value

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

# load summary of site data used for calibration: 

out_df = pd.read_csv("Calibration_AnalysisWithKGE_output_20251214\\Site_data_summary.csv")

# CALIBRATE N0 WITH ENTIRE DATA SET

# DESILETS METHOD, calibrated with portable data

# Assuming df is your pandas DataFrame with 'Avg_N0', 'theta_tot', and 'Npvi'
N0_start = out_df['N_pvisd_Des'].mean()

theta_tot = out_df['Sample_total_swc_g'].astype(float).values
N = out_df['N_pvisd_Des'].values

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
    df.loc[:,'scaled_stationary_N_Des'] = df['Corrected_Mod_cph_for_Des']*sitedata['N_ratio_raw'].item()
    
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
    df.loc[:,'scaled_stationary_N_UTS'] = df['Corrected_Mod_cph_for_UTS']*sitedata['N_ratio_raw'].item()
    
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
# Collect publication summary rows: start with universal calibration rows
landcover_summary_rows = [
    {
        'Land Cover': 'Universal',
        'Method': 'Des',
        'Mean r': all_Des_SingleN0_SiteFitStats_df['r'].mean(),
        'Mean N0': N0_fit_Des
    },
    {
        'Land Cover': 'Universal',
        'Method': 'UTS',
        'Mean r': all_UTS_SingleN0_SiteFitStats_df['r'].mean(),
        'Mean N0': N0_UTS
    }
]


# Build publication-ready correlation table for network-wide calibrations (Des & UTS)
corr_des = all_Des_SingleN0_SiteFitStats_df[["Site", "r"]].rename(columns={"r": "r_Des"})
corr_uts = all_UTS_SingleN0_SiteFitStats_df[["Site", "r"]].rename(columns={"r": "r_UTS"})
correlation_table = (
    pd.merge(corr_des, corr_uts, on="Site", how="outer")
      .sort_values("Site")
      .reset_index(drop=True)
)
summary_rows = pd.DataFrame({
    "Site": ["Mean", "Median", "Max", "Min"],
    "r_Des": [
        correlation_table["r_Des"].mean(),
        correlation_table["r_Des"].median(),
        correlation_table["r_Des"].max(),
        correlation_table["r_Des"].min(),
    ],
    "r_UTS": [
        correlation_table["r_UTS"].mean(),
        correlation_table["r_UTS"].median(),
        correlation_table["r_UTS"].max(),
        correlation_table["r_UTS"].min(),
    ],
})
correlation_table = pd.concat([correlation_table, summary_rows], ignore_index=True)
for col in ["r_Des", "r_UTS"]:
    correlation_table[col] = correlation_table[col].apply(lambda v: round_sigfig(v, 3))
correlation_table.to_csv(f"{outDir}CorrelationCoefficients_networkwide_publication_ready.csv", index=False)

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
    N0_start = a_df['N_pvisd_Des'].mean()
    
    theta_tot = a_df['Sample_total_swc_g'].astype(float).values
    N = None
    N = a_df['N_pvisd_Des'].values
    
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
        df.loc[:,'scaled_stationary_N_Des'] = df['Corrected_Mod_cph_for_Des']*sitedata['N_ratio_raw'].item()
        
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
    landcover_summary_rows.append({
        'Land Cover': lc,
        'Method': 'Des',
        'Mean r': all_Des_SingleN0_SiteFitStats_df['r'].mean(),
        'Mean N0': N0_fit_Des
    })
    
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
    
    # Run the minimizer
    res = minimize_scalar(objective2, args=(a_df,), bounds=(N0_start, N0_start + 6000), method='bounded')
    
    # Best-fit N0 value
    N0_UTS = res.x
    print(f"Best-fitting N0 from UTS fit for {lc} is: {N0_UTS:.2f}")
    
    # Use best-fit N0 to generate predictions for calibration days
    
    theta_pred_pore_volumetric = a_df.apply(
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
        
        df.loc[:,'scaled_stationary_N_UTS'] = df['Corrected_Mod_cph_for_UTS']*sitedata['N_ratio_raw'].item()
    
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
    landcover_summary_rows.append({
        'Land Cover': lc,
        'Method': 'UTS',
        'Mean r': all_UTS_SingleN0_SiteFitStats_df['r'].mean(),
        'Mean N0': N0_UTS
    })
    
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
    
# Save publication-ready summary table after land-cover-specific calibrations
publication_summary = pd.DataFrame(landcover_summary_rows)
publication_summary['Mean r'] = publication_summary['Mean r'].apply(lambda v: round_sigfig(v, 3))
publication_summary['Mean N0'] = publication_summary['Mean N0'].apply(lambda v: round_sigfig(v, 4))
publication_summary = publication_summary[['Land Cover', 'Method', 'Mean r', 'Mean N0']]
publication_summary.to_csv(f'{outDir}\\LandCover_mean_r_and_N0_publication_ready.csv', index=False)

