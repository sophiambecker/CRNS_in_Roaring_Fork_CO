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

from UTS_helpers import convert_neutrons_to_soil_moisture_uts

import datetime as dt
from datetime import datetime
import pandas as pd

import glob
import numpy as np

from scipy.optimize import minimize_scalar

from sklearn.neighbors import LocalOutlierFactor

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
directory_path = 'FilteredSnowFreeData_output20251120'
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

# load summary of site data used for calibration: 

out_df = pd.read_csv("Calibration_AnalysisWithKGE_output_20251204\\Site_data_summary_withC4.csv")


# SOLVE N0 with probe data FOR EACH SITE 

for n in site_names_new:

    #n = 'C2' 
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
    
    # use N0_fitted to predict swc at site
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    site_df = df_dict[THIS_SITE_new]
    
    # estimate soil moisture for all snow-free days using Desilets equation
    tau = site_lw +site_soc
    
    sel_cols = ['Corrected_Mod_cph_for_Des',
           'SWC_Des_cm3_cm3',
           'Corrected_Mod_cph_for_UTS', 
           'SWC_UTS_cm3_cm3',
            'WeightedTDR_SWC', 
            'Bare', 'Rhov_g_cm3']
    df_outlier_filt = site_df.dropna(subset=sel_cols).copy()
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
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
    outliers = lof.fit_predict(find_outliers)  # -1 indicates outliers
   
    #outlier dates: 
    outlier_rows = df_outlier_filt[outliers == -1]
    print(f'{THIS_SITE_new} outliers are: ')
    print(outlier_rows)
    
    rm_dates = outlier_rows['date']
    
    # remove outlier date that behaves strangely with precip and where Bare is already nan: 
    date1 = datetime.strptime('2024-08-15', '%Y-%m-%d').date()
    date2 = datetime.strptime('2024-08-14', '%Y-%m-%d').date()
    date3 = datetime.strptime('2023-10-12', '%Y-%m-%d').date()
    
    extra_dates = [date1, date2, date3]
    all_rm_dates = pd.Series(list(set(rm_dates.tolist() + extra_dates)))
    
    site_df = site_df[~site_df['date'].isin(all_rm_dates)]
    
    site_df['scaled_stationary_N_Des'] = site_df['Corrected_Mod_cph_for_Des']*sitedata['N_ratio_raw'].item()
    site_df['scaled_stationary_N_UTS'] = site_df['Corrected_Mod_cph_for_UTS']*sitedata['N_ratio_raw'].item()
    
    # Calculate N0 for every time step with probe data
    
    swc1 = site_df['WeightedTDR_SWC']
    tw1 = swc1 + tau
    
    Ncal1 = site_df['scaled_stationary_N_Des']
    
    site_df['N0'] = Ncal1 / (0.0808 / (tw1 + 0.115) + 0.372)
    
    
    site_df['TDR_pore_swc_g'] = site_df['WeightedTDR_SWC']/site_bd
    site_df['TDR_tot_swc_g'] = site_df['TDR_pore_swc_g'] + sitedata['lw'].item() + sitedata['soc_water'].item()
    
    
    site_df['NewName'] = THIS_SITE_new
    
    # now with UTS method
    
    Starting_N0_UTS = site_df['scaled_stationary_N_UTS'].mean() # just mean of corrected counts
    
    # filter out bad temperature and RH values
    #bad_rows =  ~(site_df['airT'] == -35)|(site_df['airRH'] == -100)
    
    # Iterate over rows of df_clean to access both swc and the corresponding date
    def compute_ND(row):
        # --- Extract inputs ---
        swc1 = None
        swc1  = row['WeightedTDR_SWC']
        rhov  = row['Rhov_g_cm3']
        Ncal1 = row['scaled_stationary_N_UTS']
        temp = row['airT']
        rh = row['airRH']
        
        # --- Return NaN if ANY input is missing ---
        if np.isnan(swc1) or np.isnan(rhov) or np.isnan(Ncal1) or np.isnan(temp) or np.isnan(rh):
            return np.nan
        
        if temp == -35:
            return np.nan
        
        if rh == 100: 
            return np.nan
        
    
        # --- Objective function for this row ---
        def objective_single(N0):
            try:
                swc_pred = convert_neutrons_to_soil_moisture_uts(
                    neutron_count=Ncal1,
                    n0=N0,
                    air_humidity=rhov,
                    bulk_density=site_bd,
                    lattice_water=site_lw * site_bd,
                    water_equiv_soil_organic_carbon=site_soc * site_bd,
                    method="Mar21_mcnp_drf"
                )
            except Exception:
                return np.inf
            
            return abs(swc_pred - swc1)
    
        # --- Perform bounded search ---
        # some sites need narrower bounds
        if (THIS_SITE_new == 'F1')|(THIS_SITE_new == 'R2'):
            bound_upper = Starting_N0_UTS + 1.4*Starting_N0_UTS
        elif(THIS_SITE_new == 'F2'):
            bound_upper = Starting_N0_UTS + 1.6*Starting_N0_UTS
        elif(THIS_SITE_new == 'F4')|(THIS_SITE_new == 'R1'):
            bound_upper = Starting_N0_UTS + 1.7*Starting_N0_UTS
        else:
            bound_upper = Starting_N0_UTS + 2*Starting_N0_UTS 
            
        res = minimize_scalar(
            objective_single,
            bounds=(Starting_N0_UTS, bound_upper), # needed to narrow inversion bounds to prevent unrealistic ND solutions that are caused by combo of low swc and high rh
            method="bounded"
        )
    
        # If minimizer fails, return NaN
        if not res.success:
            return np.nan
    
        return res.x
    
    
    # ---- Apply to DataFrame ----
    site_df["ND"] = site_df.apply(compute_ND, axis=1)
    
    
    site_df.to_csv(f'{outDir}\\{THIS_SITE_new}_Probe_Based_Parameters.csv')
    
    print(f'Saved {THIS_SITE_new}')
    
    print(f'Max ND at {THIS_SITE_new} is {site_df["ND"].max()}')
    print(f'Mean ND at {THIS_SITE_new} is {site_df["ND"].mean()}')
    print(f'Minimum ND at {THIS_SITE_new} is {site_df["ND"].min()}')
    # print date of minimum ND
    
    min_date = site_df['date'][site_df['ND']==site_df["ND"].min()]
    print(f'Date for minimum ND at {THIS_SITE_new} is {min_date}')
    
    max_date = site_df['date'][site_df['ND']==site_df["ND"].max()]
    print(f'Date for max ND at {THIS_SITE_new} is {max_date}')
    
    '''
    print(f'Max swc at {THIS_SITE_new} is {site_df["WeightedTDR_SWC"].max()}')
    print(f'Minimum SWC at {THIS_SITE_new} is {site_df["WeightedTDR_SWC"].min()}')
    
    print(f'Max mod counts at {THIS_SITE_new} is {site_df["scaled_stationary_N_UTS"].max()}')
    print(f'Mean mod counts at {THIS_SITE_new} is {site_df["scaled_stationary_N_UTS"].mean()}')
    print(f'Minimum mod counts at {THIS_SITE_new} is {site_df["scaled_stationary_N_UTS"].min()}')

    print(f'Max rhov at {THIS_SITE_new} is {site_df["Rhov_g_cm3"].max()}')
    print(f'Mean rhov at {THIS_SITE_new} is {site_df["Rhov_g_cm3"].mean()}')
    print(f'Minimum rhov at {THIS_SITE_new} is {site_df["Rhov_g_cm3"].min()}')
    '''
