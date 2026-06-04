# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:32:23 2025

@author: sbecker14
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from datetime import datetime
from scipy.stats import linregress, pearsonr
from pandas.plotting import scatter_matrix

# helpers
def round_sigfig(value, sig=2):
    if pd.isna(value):
        return value
    try:
        return float(f"{value:.{sig}g}")
    except (ValueError, TypeError):
        return value

# set up output directory

stamp = datetime.now().strftime("%Y%m%d")
inDir = 'C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis'
outDir = os.path.normpath(inDir + os.sep + 'N0_vs_SiteSpecificVariables\\output_no_fs_'+stamp) + '\\'   # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

site_var = pd.read_csv("C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Calibration_AnalysisWithKGE_output_20260210\\Site_Calibration_data_summary.csv")

site_var_excel = pd.read_excel("C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Data\\RoaringFork_CRNS_metadata_from_GoogleDrive_20250617.xlsx", sheet_name = 'site_metadata')

site_CR = site_var_excel[['Site_ID','Cutoff_Rigidity_GV']]# get site name and cutoff rigidity
# d
'''
site_var columns are: 
    
    ['Unnamed: 0', 'N_Correct_Des', 'N_Correct_UTS', 'Sample_total_swc_g',
           'TDR_total_swc_g', 'bd', 'lw', 'soc_water', 'Elev', 'Canopy',
           'landCoverClass', 'OldName', 'NewName', 'CalDate', 'airRH', 'airT',
           'WeightedTDR', 'ppt', 'Bare', 'BWE_mm', 'BWE_uncer', 'N0_port_to_stat',
           'Raw_Mod_cv', 'Raw_Mod_sqrt', 'N0_fit_Des', 'N0_fit_UTS', 'N_ratio_Des',
           'N_ratio_UTS', 'Des_resid_singleN0', 'Rhov_cal_g_cm3',
           'UTS_Resid_singleN0', 'Site', 'N0_Desilets', 'N0_UTS',
           'GRAV_TDR_offset', 'TDR_Grav_ratio'],
          dtype='object')

'''

sel_col = ['NewName', 'N0_fit_Des', 'N0_fit_UTS',
           'Elev',
           'bd', 
           'lw','soc_water',
           'BWE_mm'
   ]
'''
# Rename columns for display
pretty_labels = {
    'N_Correct_Des': r'$SM_{0-5}$',
    'N_Correct_Des': r'$SM_{5-15}$',
    'Temp_5cm': r'$T_{5\ cm}$'
}
df = df.rename(columns=pretty_labels)

'''
site_data = site_var[sel_col]

site_data = pd.merge(site_data, site_CR, left_on = 'NewName', right_on = 'Site_ID')

# remove site name columns

site_data_clean = site_data.drop(['NewName', 'Site_ID'], axis=1)

site_corr = site_data_clean.corr()

# Plot scatter matrix
scatter_matrix(site_data, figsize=(12, 12), diagonal='hist', alpha=0.7)
plt.suptitle("Scatter Matrix")
plt.show()


# Dependent reference columns
ref_cols = ['N0_fit_Des', 'N0_fit_UTS']

# Store results
results = []

for col in site_data_clean.columns:
    if col not in ref_cols:
        result = {'independent_var': col}
        for ref in ref_cols:
            # Drop NaNs
            valid_data = site_data_clean[[col, ref]].dropna()
            if len(valid_data) >= 2:  # linregress and pearsonr require at least 2 points
                # Linear regression: ref ~ col
                slope, intercept, r_value, p_value, std_err = linregress(valid_data[col], valid_data[ref])
                r, _ = pearsonr(valid_data[col], valid_data[ref])
                
                
                result[f'r_vs_{ref}'] = r
                result[f'slope_vs_{ref}'] = slope
                result[f'p_val_vs_{ref}'] = p_value
            else:
                result[f'r_vs_{ref}'] = None
                result[f'slope_vs_{ref}'] = None
                result[f'p_val_vs_{ref}'] = None
        results.append(result)

# Convert to DataFrame
summary_df = pd.DataFrame(results)

# View results
print(summary_df)

summary_df.to_csv(f'{outDir}\\N0_vs_sitespecific_variables_{stamp}.csv')

# Publication-ready summary
pub_df = summary_df.rename(columns={
    'independent_var': 'independent variable',
    'r_vs_N0_fit_Des': 'N0 r',
    'slope_vs_N0_fit_Des': 'N0 Slope',
    'p_val_vs_N0_fit_Des': 'N0 P-value',
    'r_vs_N0_fit_UTS': 'ND r',
    'slope_vs_N0_fit_UTS': 'ND slope',
    'p_val_vs_N0_fit_UTS': 'ND p-value'
})
ordered_cols = ['independent variable', 'N0 r', 'N0 Slope', 'N0 P-value', 'ND r', 'ND slope', 'ND p-value']
pub_df = pub_df[[c for c in ordered_cols if c in pub_df.columns]]
numeric_cols = pub_df.select_dtypes(include=[np.number]).columns
pub_df[numeric_cols] = pub_df[numeric_cols].applymap(lambda v: round_sigfig(v, 2))
pub_df.to_csv(f'{outDir}\\N0_vs_sitespecific_variables_publication_ready_{stamp}.csv', index=False)
