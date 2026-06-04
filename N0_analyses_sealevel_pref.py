# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 21:02:51 2025

@author: sbecker14
"""

# libraries
import os
os.chdir('C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO')

import datetime as dt
import pandas as pd

import glob
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D

from UTS_helpers import convert_neutrons_to_soil_moisture_uts

stamp = dt.datetime.now().strftime("%Y%m%d") 
Dir = os.getcwd()
outFold = '\\N0_analyses_output_no_fs{}'.format(stamp)
                                                 
outDir = os.path.normpath(Dir + os.sep + outFold) + '\\'    # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

# load portable calibration data
directory_path_N0= 'Calibration_AnalysisWithCorrelation_output_20260211'

N0_file_pattern = f'{Dir}\\{directory_path_N0}\\*Probe_Based_Parameters.csv'

N0_paths = glob.glob(N0_file_pattern, recursive = True)

site_names_new = [os.path.basename(fp).split('_')[0] for fp in N0_paths]

dfs_N0_ls = [pd.read_csv(file_path) for file_path in N0_paths]

dict_N0 = dict(zip(site_names_new, dfs_N0_ls))


 
# load calibration day data: 

site_var = pd.read_excel("Data\\Mock Calibration Summary_20251106.xlsx")

# dictionary to get old site name based on new site name: 
names = pd.read_excel('Data\\Data_Release_2024_b\\Network_paper_site_names.xlsx')
names['Short_name'] = names['Short_name'].astype(str).str.strip().str.replace('RF 5', 'RF5')

new_to_old_name = dict(zip(names['network_paper_new_name'], names['Short_name']))
old_to_new_name = dict(zip(names['Short_name'], names['network_paper_new_name']))

# get biomass sample dates:

metadata_df = pd.read_csv('Data\\RoaringFork_CRNS_metadata.csv')
metadata_df.replace('RF 5', 'RF5', inplace=True)
date_series = metadata_df['Biomass Sampling Date']

# Convert values to just the date, keep column names (keys) as strings

sample_date = {
    col: pd.to_datetime(val)
           .tz_localize('UTC')             # 1. Say it's in UTC
           .date()                         # 2. Extract just the date
           for col, val in date_series.items()
}

oldname_to_date = dict(zip(metadata_df['Original_ID'], sample_date.values()))

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

# add columns with new name, bio sample date

TotBWE_df['SiteNew'] = TotBWE_df['Site'].map(old_to_new_name)
TotBWE_df['BioSampDate'] = TotBWE_df['Site'].map(oldname_to_date)

# load forest/non-forest classification:
f_clas = pd.read_excel("Data\\VisualSiteClassification.xlsx")
f_clas_dict = dict(zip(f_clas['Site Original'], f_clas['Imagery']))

rows = [ ]

stats_ls = []

for s in site_names_new:
    df = None
    
    THIS_SITE_new = s
    
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    df = dict_N0[THIS_SITE_new]
    df['date'] = pd.to_datetime(df['date']).dt.date # make sure date column is in date format
    
    bio_date = TotBWE_df['BioSampDate'][TotBWE_df['SiteNew'] == THIS_SITE_new].item()
    
    if not (df['date'] == bio_date).any():
        print(f"{THIS_SITE_new} Calibration date {bio_date} not found. Finding nearest date...")
        
        df['date_diff'] = df['date'].apply(lambda d: abs(d - bio_date))
        nearest_row = df.loc[df['date_diff'].idxmin()]
        bio_date = nearest_row['date']
        df.drop(columns='date_diff', inplace=True)
        print(f'Nearest Date is {bio_date} for {THIS_SITE_new}.')
        QA1 = 'FALSE'
    else: QA1 = 'TRUE' # True if the calibration sample date is in the stationary data
        
    check_cols = ['Corrected_Mod_cph_for_Des','Corrected_Mod_cph_for_UTS','airRH','airT'] # for now it's okay if TDR is NaN, we are just calibrating with samples
    
    # check the above columns for nan values. If any column contains an nan value, use the nearest date without nan values as the calibration date. 
    # Check if any NaNs in the row for the calibration date
    if df.loc[df['date'] == bio_date, check_cols].isnull().any(axis=1).any():
        
        # Filter to find rows with no NaNs in the desired columns
        valid_rows = df[df[check_cols].notna().all(axis=1)].copy()
    
        # Find the row with the minimum absolute date difference
        valid_rows['date_diff'] = valid_rows['date'].apply(lambda d: abs(d - bio_date))
        nearest_row = valid_rows.loc[valid_rows['date_diff'].idxmin()]
        
        # Update bio_date
        bio_date = nearest_row['date']
        QA2 = 'FALSE' # False if there were NaNs in the stationary data for the sampling time and a different datetime is used instead
    else: QA2 = 'TRUE' # True if there aren't any NaNs in the stationary data for the sampling calibration time
    
    # Now extract cal_data
    cal_data = df[df['date'] == bio_date]
    
    # half-hourly data produce many rows per date; pick the row closest to noon to avoid .item() errors
    if len(cal_data) > 1:
        cal_data = cal_data.copy()
        cal_data['DateTime'] = pd.to_datetime(cal_data['DateTime'])
        target_dt = pd.Timestamp.combine(bio_date, pd.Timestamp('12:00').time())
        cal_data['time_diff'] = (cal_data['DateTime'] - target_dt).abs()
        cal_data = cal_data.sort_values('time_diff').head(1)
        print(f"{THIS_SITE_new}: multiple rows on {bio_date}, using {cal_data['DateTime'].iloc[0]}")
    elif cal_data.empty:
        raise ValueError(f"No calibration data found for {THIS_SITE_new} on {bio_date}")
    
    cal_row = cal_data.iloc[0]
    
    N0_probe = cal_row['N0']
    ND_probe = cal_row['ND']
   
    #Ncal_p_raw = p_des_df[''] # could edit portable processing code to also output raw data
    
    Ncal_Des_st = cal_row['Corrected_Mod_cph_for_Des'] # stationary data during soil sampling
    Ncal_UTS_st = cal_row['Corrected_Mod_cph_for_UTS'] # stationary data during soil sampling
    
    SWC_probe_vol = cal_row['WeightedTDR_SWC']
    
    Rhov_cal_g = cal_row['Rhov_g_cm3']
    
    # Get other site-specific variables to calculate total gravimetric water content
    
    site_bd = site_var[THIS_SITE_old][24]
    
    swc_g = site_var[THIS_SITE_old][18] # pore water (g/g)
    lw = site_var[THIS_SITE_old][26] # lattice water (g/g)
    soc = site_var[THIS_SITE_old][29] # soc water (g/g)
    
    porosity = 1-(site_bd/2.65)
    
    print(f"{THIS_SITE_new}: bd={site_bd}, lw={lw}, soc={soc}")
    
    GRAV_swc_tot_g = swc_g + lw + soc
    
    elev = site_var[THIS_SITE_old][6]
    
    
    
    row = {'Sample_total_swc_g': GRAV_swc_tot_g, 'Probe_swc_vol': SWC_probe_vol, 'Probe_N0':N0_probe, 'Probe_ND': ND_probe, 'bd': site_bd, 
           'lw': lw, 'soc_water': soc, 'Porosity': porosity, 'Elev': elev, 'Site': THIS_SITE_old, 'NewName': THIS_SITE_new, 
           'BioSampDate': bio_date, 'Rhov_cal_g_cm3': Rhov_cal_g,
            'WeightedTDR': cal_row['WeightedTDR_SWC'], 'ppt': cal_row['PRISM_ppt_mm'], 
           'Bare': cal_row['Bare'], 
            'Raw_Mod_cv':cal_row['Raw_Mod_Coeff_of_Var'], 'Raw_Mod_sqrt':cal_row['Raw_Mod_sqrt'],
           'QA1': QA1, 'QA2': QA2}
    
    rows.append(row)
    
    # also calculate summary statistics for 2024 N0 series May 1 to Oct 1): 
        
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['doy'] = pd.to_datetime(df['date']).dt.dayofyear


    year_int = int(2024)
    start_date = pd.to_datetime(f'{year_int}-05-01').date() 
    end_date = pd.to_datetime(f'{year_int}-10-1').date() 
        
        
    df_24 = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date) & 
        (df['year'] == year_int)
    ].copy()
        
    
    mean_N0_cts = df_24['scaled_stationary_N_Des'].mean()
    mean_swc = df_24['WeightedTDR_SWC'].mean()
    
    mean_nd = df_24['ND'].mean()
    mean_n0 = df_24['N0'].mean()
    
    sd_nd = df_24['ND'].std()
    sd_n0 = df_24['N0'].std()
    
    max_nd = df_24['ND'].max()
    max_n0 = df_24['N0'].max()
    
    min_nd = df_24['ND'].min()
    min_n0 = df_24['N0'].min()
    
    range_nd = max_nd - min_nd
    range_n0 = max_n0 - min_n0
    
    cov_nd = sd_nd/mean_nd
    cov_n0 = sd_n0/mean_n0
    
    xy_nd = df_24[['ND', 'doy']].dropna()
    y_all_24_nd = xy_nd['ND']
    x_all_24_nd = xy_nd['doy']
    
    xy_n0 = df_24[['N0', 'doy']].dropna()
    y_all_24_n0 = xy_n0['N0']
    x_all_24_n0 = xy_n0['doy']
    
    slope_nd, intercept_nd, r_value_nd, p_value_nd, std_err_nd = linregress(x_all_24_nd, y_all_24_nd)
    slope_n0, intercept_n0, r_value_n0, p_value_n0, std_err_n0 = linregress(x_all_24_n0, y_all_24_n0)

    # propagate uncertainty:

    max_twc =  0.0808 / (df_24['scaled_stationary_N_Des'] / (df_24['N0']+2*sd_n0) - 0.372) - 0.115
    max_swc = (max_twc - lw - soc)*site_bd

    min_twc =  0.0808 / (df_24['scaled_stationary_N_Des'] / (df_24['N0']-2*sd_n0) - 0.372) - 0.115
    min_swc = (min_twc - lw - soc)*site_bd

   
    min_resid = min_swc - df_24['WeightedTDR_SWC']
    
    mean_min_resid = min_resid.mean()
    
    min_rmse = np.sqrt((min_resid**2).mean())
    
    max_resid = max_swc - df_24['WeightedTDR_SWC']
    
    mean_max_resid = max_resid.mean()
    
    max_rmse = np.sqrt((max_resid**2).mean())
    
    # now for uts: 
        
    max_swc_uts = df_24.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['scaled_stationary_N_UTS'],
            n0=row['ND']+2*sd_nd,
            air_humidity=row['Rhov_g_cm3'],
            bulk_density= site_bd,
            lattice_water=lw* site_bd,
            water_equiv_soil_organic_carbon= soc * site_bd,
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    ) 
    
    max_resid_uts = max_swc_uts - df_24['WeightedTDR_SWC']
    mean_max_resid_uts = max_resid_uts.mean()
    max_rmse_uts = np.sqrt((max_resid_uts**2).mean())
    
    min_swc_uts = df_24.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['scaled_stationary_N_UTS'],
            n0=row['ND']-2*sd_nd,
            air_humidity=row['Rhov_g_cm3'],
            bulk_density= site_bd,
            lattice_water=lw* site_bd,
            water_equiv_soil_organic_carbon= soc * site_bd,
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    ) 
    
    min_resid_uts = min_swc_uts - df_24['WeightedTDR_SWC']
    mean_min_resid_uts = min_resid_uts.mean()
    min_rmse_uts = np.sqrt((min_resid_uts**2).mean())
    
    # get land cover group: 
    
    lc = f_clas_dict[THIS_SITE_old]
    
    nd_stats = {'Site': THIS_SITE_new, 'Group':lc, 'Param': 'ND', 'Mean': mean_nd, 'STD': sd_nd, 'CoV': cov_nd, 'Max': max_nd, 
                "Min": min_nd, 'Range': range_nd, 'Slope': slope_nd, 'intercept': intercept_nd,
                'r':r_value_nd, 'slope pval': p_value_nd,
                'mean_max_resid': mean_max_resid_uts, 'mean_min_resid': mean_min_resid_uts, 'RMSE_upper_param':max_rmse_uts, 'RMSE_lower_param':min_rmse_uts
                }
    

    n0_stats = {'Site': THIS_SITE_new, 'Group': lc, 'Param': 'N0', 'Mean': mean_n0, 'STD': sd_n0, 'CoV': cov_n0, 'Max': max_n0, 
                "Min": min_n0, 'Range': range_n0, 'Slope': slope_n0, 'intercept': intercept_n0,
                'r':r_value_n0, 'slope pval': p_value_n0,
                'mean_max_resid': mean_max_resid, 'mean_min_resid': mean_min_resid, 'RMSE_upper_param':max_rmse, 'RMSE_lower_param':min_rmse
                }
    
    
    
    stats_ls.append(nd_stats)
    stats_ls.append(n0_stats)
    
out_df = pd.DataFrame(rows)

stats_df = pd.DataFrame(stats_ls)

# Build publication-ready stats (2 sig figs) with overall and group means (excluding Cw)
def _sigfig(val, sig=2):
    try:
        return float(f"{val:.{sig}g}")
    except Exception:
        return val

exclude_sites_for_mean = {'none'}
numeric_cols = stats_df.select_dtypes(include=[np.number]).columns

stats_df_filtered = stats_df[~stats_df['Site'].isin(exclude_sites_for_mean)].copy()

stats_df_pub = stats_df.copy()
stats_df_pub[numeric_cols] = stats_df_pub[numeric_cols].applymap(_sigfig)

overall_means = stats_df_filtered[numeric_cols].mean()
overall_row = {'Site': 'Overall ', 'Param': 'All'}
overall_row.update({col: _sigfig(overall_means[col]) for col in numeric_cols})

param_rows = []
for param, grp in stats_df_filtered.groupby('Param'):
    means = grp[numeric_cols].mean()
    row = {'Site': f'{param} mean', 'Param': param}
    row.update({col: _sigfig(means[col]) for col in numeric_cols})
    param_rows.append(row)

# Landcover-group means (by Group and Param)
group_param_rows = []
for (group_name, param), grp in stats_df_filtered.groupby(['Group', 'Param']):
    means = grp[numeric_cols].mean()
    row = {
        'Site': f'{group_name} {param} mean ',
        'Param': param,
        'Group': group_name
    }
    row.update({col: _sigfig(means[col]) for col in numeric_cols})
    group_param_rows.append(row)

rows_to_append = [overall_row] + param_rows + group_param_rows

if rows_to_append:
    stats_df_pub = pd.concat([stats_df_pub, pd.DataFrame(rows_to_append)], ignore_index=True)
else:
    stats_df_pub = pd.concat([stats_df_pub, pd.DataFrame([overall_row])], ignore_index=True)

stats_df.to_csv(f'{outDir}\\Seasonal_2024_N0_ND_stats.csv')
stats_df_pub.to_csv(f'{outDir}\\Seasonal_2024_N0_ND_stats_publication.csv', index=False)

TotBWE_df = pd.merge(TotBWE_df, out_df, on = 'Site')

# filter out row with nan (R1, probe wasn't working during biomass sampling)
TotBWE_df = TotBWE_df.dropna(subset = 'Probe_N0')

# Make basic plots

fig, ax1 = plt.subplots(figsize=(4, 4))
ax1.scatter(TotBWE_df['BWE Representing 200 m Radius Footprint (mm)'], 
            TotBWE_df['Probe_N0'], color = '#44AA99', s = 20)

ax1.scatter(TotBWE_df['BWE Representing 200 m Radius Footprint (mm)'], 
            TotBWE_df['Probe_ND'], color = "#6699CC", s = 20)

ax1.set_xlabel('BWE (mm)', fontsize = 14)
ax1.set_ylabel(r'$N_{0}$ or $N_{D}$ value (cph)', fontsize=14)

marker_handles = [
    Line2D([0], [0], marker='o', color='#44AA99', label=r'$N_0$',
           markerfacecolor='#44AA99', markersize=8, linestyle='None', linewidth=1),
    Line2D([0], [0], marker='o', color='#6699CC', label=r'$N_D$',
           markerfacecolor='#6699CC',alpha = 0.4, markersize=8, linestyle='None', linewidth=1),
]

# Extract labels from the handles
marker_labels = [h.get_label() for h in marker_handles]

fig.legend(handles = marker_handles, labels = marker_labels,
           bbox_to_anchor=(0.80, 0.5), 
           loc='center left', frameon=False,
           prop={'size': 11})
plt.subplots_adjust(right=0.79, left = 0.12)  # Shrinks the plot width to leave space on the right

plt.show()

# fit line to these
# fit a linear regression, predict BWE from Bare.mod
x = TotBWE_df['BWE Representing 200 m Radius Footprint (mm)'].to_numpy()
y_N0 = TotBWE_df['Probe_N0'].to_numpy()
# Run linear regression
result_N0 = linregress(x, y_N0)

# Unpack results
slope_N0      = result_N0.slope
intercept_N0  = result_N0.intercept
r_value_N0    = result_N0.rvalue
p_value_N0    = result_N0.pvalue     # <-- p-value of the slope
stderr_N0     = result_N0.stderr

# Compute RMSE (linregress does NOT compute it)
y_pred_N0 = slope_N0 * x + intercept_N0
rmse_N0   = np.sqrt(mean_squared_error(y_N0, y_pred_N0))

print(f"coefficient of determination for N0 line: {r_value_N0 **2}")

# Calculate the RMSE
rmse_N0 = np.sqrt(mean_squared_error(y_N0, y_pred_N0))

# Print the equation of the linear model
print(f"Equation of the N0 line: y = {slope_N0:.2f}x + {intercept_N0:.2f}")

# Run linear regression
y_ND = np.array(TotBWE_df['Probe_ND'])
result_ND = linregress(x, y_ND)

# Unpack results
slope_ND      = result_ND.slope
intercept_ND  = result_ND.intercept
r_value_ND    = result_ND.rvalue
p_value_ND    = result_ND.pvalue     # <-- p-value of the slope
stderr_ND     = result_ND.stderr

# Compute RMSE (linregress does NOT compute it)
y_pred_ND = slope_ND * x + intercept_ND
rmse_ND   = np.sqrt(mean_squared_error(y_ND, y_pred_ND))

y_ND = np.array(TotBWE_df['Probe_ND'])

print(f"coefficient of determination for ND line: {r_value_ND**2}")

# Calculate the RMSE
rmse_ND = np.sqrt(mean_squared_error(y_ND, y_pred_ND))

# Print the equation of the linear model
print(f"Equation of the ND line: y = {slope_ND:.2f}x + {intercept_ND:.2f}")

eta_N0 = slope_N0/intercept_N0 
eta_ND = slope_ND/intercept_ND 

print(f"Eta from the N0 line = {eta_N0}")
print(f"Eta from the ND line = {eta_ND}")

# make predictions to plot
x_plot =np.array(pd.Series(range(110))).reshape((-1, 1))
y_plot_N0 = slope_N0 * x_plot + intercept_N0
y_plot_ND = slope_ND * x_plot + intercept_ND

# Add this info to plot:

fig, ax1 = plt.subplots(figsize=(7, 4))

# Extract x, x-errors
x_vals = TotBWE_df['BWE Representing 200 m Radius Footprint (mm)']
x_err  = TotBWE_df['BWE Uncertainty (mm)']

# Y values
y_N0_vals = TotBWE_df['Probe_N0']
y_ND_vals = TotBWE_df['Probe_ND']

# Horizontal error bars
ax1.errorbar(
    x_vals, y_N0_vals,
    xerr=x_err,
    fmt='o',
    linestyle='None',
    markerfacecolor='#44AA99',
    markeredgecolor='#44AA99',
    ecolor='black',      # error bar color
    elinewidth=0.9,
    capsize=3,
    markersize = 4
)

ax1.errorbar(
    x_vals, y_ND_vals,
    xerr=x_err,
    fmt='^',
    linestyle='None',
    markerfacecolor="#6699CC",
    markeredgecolor="#6699CC",
    ecolor = 'black',
    capsize=3,
    markersize=4
)

ax1.plot(x_plot, y_plot_N0, color = '#44AA99', alpha = 0.7)
ax1.plot(x_plot, y_plot_ND, color = "#6699CC", alpha = 0.7)
ax1.set_xlabel('BWE (mm)', fontsize = 14)
ax1.set_ylabel(r'$N_{0}$ or $N_{D}$ value (cph)', fontsize=14)

marker_handles = [
    Line2D([0], [0], marker='^', color='#6699CC', label=r'$N_D$',
           markerfacecolor='#6699CC', markersize=3, linestyle='None', linewidth=1),
    Line2D([0], [0], marker='o', color='#44AA99', label=r'$N_0$',
           markerfacecolor='#44AA99', markersize=3, linestyle='None', linewidth=1)
]

# Extract labels from the handles
marker_labels = [h.get_label() for h in marker_handles]

ax1.annotate(rf'$N_D$ $R^2$ = {r_value_ND**2:.2f}, p-val = {p_value_ND:.2g}', xy=(0.69, .6), xycoords='figure fraction',
                xytext=(0.68, .6), textcoords='figure fraction', fontsize = 10,
                ha='left', va='bottom'
                )
ax1.annotate(r'$N_{D}$ ' + f"y = {slope_ND:.2g}x + {intercept_ND:.2g}", xy = (0.69, .53), 
             fontsize =10,  xycoords='figure fraction',
                xytext=(0.68, .53), textcoords='figure fraction',
                ha='left', va='bottom'
                )
ax1.annotate(r'$N_{D}$ ' + f"η = {eta_ND: .2g}", xy = (0.69, .46), 
             fontsize =10, xycoords='figure fraction',
                xytext=(0.69, .46), textcoords='figure fraction',
                ha='left', va='bottom')
ax1.annotate(f'$N_0$ $R^2$ = {r_value_N0**2:.2f}, p-val = {p_value_N0:.2g}', xy=(0.69, .36), xycoords='figure fraction',
                xytext=(0.69, .36), textcoords='figure fraction',fontsize = 10,
                ha='left', va='bottom'
                )
ax1.annotate(r'$N_{0}$ '+ f"y = {slope_N0:.2g}x + {intercept_N0:.2g}", xy = (0.69, .3), 
             fontsize =10,  xycoords='figure fraction',
                xytext=(0.69, .3), textcoords='figure fraction',
                ha='left', va='bottom'
                )
ax1.annotate(r'$N_{0}$ '+ f"η = {eta_N0: .2g}", xy = (0.69, .24), 
             fontsize =10, xycoords='figure fraction',
                xytext=(0.69, .24), textcoords='figure fraction',
                ha='left', va='bottom')
'''
plt.annotate('N0 rsq = {}'.format(round(r_sq_N0, 2)), xy = (200,70), fontsize =12 )
plt.annotate('ND rsq = {}'.format(round(r_sq_ND, 2)), xy = (0.22,70), fontsize =12 )

plt.annotate('RMSE = {} mm'.format(round(rmse,2)), xy = (0.22,65), fontsize =12)
plt.annotate(f"y = {a:.2f}x + {b:.2f}", xy = (0.22, 60), fontsize =12)
'''
fig.legend(handles = marker_handles, labels = marker_labels,
           bbox_to_anchor=(0.69, 0.75), 
           loc='center left', frameon=False,
           prop={'size': 12})

# add other info
plt.subplots_adjust(right=0.67, left = 0.12)  # Shrinks the plot width to leave space on the right
#fig.savefig(f'{outDir}\\N0_ND_vs_BWE_300dpi', dpi = 300)
plt.show()

################## now need to identify outliers ############################

# now do with elevation and N0 from gravimetric sample

df_grav = pd.read_csv("Calibration_AnalysisWithKGE_output_20260210\\Site_data_summary.csv")

# fit line to these
# fit a linear regression, predict BWE from Bare.mod
x = df_grav['Elev'].to_numpy()
y_N0 = df_grav['N0_fit_Des'].to_numpy()
# Run linear regression
result_N0 = linregress(x, y_N0)

# Unpack results
slope_N0      = result_N0.slope
intercept_N0  = result_N0.intercept
r_value_N0    = result_N0.rvalue
p_value_N0    = result_N0.pvalue     # <-- p-value of the slope
stderr_N0     = result_N0.stderr

# Compute RMSE (linregress does NOT compute it)
y_pred_N0 = slope_N0 * x + intercept_N0
rmse_N0   = np.sqrt(mean_squared_error(y_N0, y_pred_N0))

print(f"coefficient of determination for N0 line: {r_value_N0 **2}")

# Calculate the RMSE
rmse_N0 = np.sqrt(mean_squared_error(y_N0, y_pred_N0))

# Print the equation of the linear model
print(f"Equation of the N0 line: y = {slope_N0:.2f}x + {intercept_N0:.2f}")

# Run linear regression
y_ND = np.array(df_grav['N0_fit_UTS'])
result_ND = linregress(x, y_ND)

# Unpack results
slope_ND      = result_ND.slope
intercept_ND  = result_ND.intercept
r_value_ND    = result_ND.rvalue
p_value_ND    = result_ND.pvalue     # <-- p-value of the slope
stderr_ND     = result_ND.stderr

# Compute RMSE (linregress does NOT compute it)
y_pred_ND = slope_ND * x + intercept_ND
rmse_ND   = np.sqrt(mean_squared_error(y_ND, y_pred_ND))

print(f"coefficient of determination for ND line: {r_value_ND**2}")

# Calculate the RMSE
rmse_ND = np.sqrt(mean_squared_error(y_ND, y_pred_ND))

# Print the equation of the linear model
print(f"Equation of the ND line: y = {slope_ND:.2f}x + {intercept_ND:.2f}")

eta_N0 = slope_N0/intercept_N0 *100
eta_ND = slope_ND/intercept_ND *100

print(f"Eta from the N0 line = {eta_N0}")
print(f"Eta from the ND line = {eta_ND}")

# make predictions to plot
x_plot =np.array(pd.Series(range(int(df_grav['Elev'].min()), int(df_grav['Elev'].max())))).reshape((-1, 1))
y_plot_N0 = slope_N0 * x_plot + intercept_N0
y_plot_ND = slope_ND * x_plot + intercept_ND

# Add this info to plot:

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.scatter(df_grav['Elev'], 
            df_grav['N0_fit_Des'], color = '#44AA99', s = 14, marker = 'o')

ax1.scatter(df_grav['Elev'], 
            df_grav['N0_fit_UTS'], color = "#6699CC", s = 14, marker = '^')
ax1.plot(x_plot, y_plot_N0, color = '#44AA99', alpha = 0.7)
ax1.plot(x_plot, y_plot_ND, color = "#6699CC", alpha = 0.7)
ax1.set_xlabel('Elevation (m)', fontsize = 14)
ax1.set_ylabel(r'$N_{0}$ or $N_{D}$ value (cph)', fontsize=14)

marker_handles = [
    Line2D([0], [0], marker='^', color='#6699CC', label=r'$N_D$',
           markerfacecolor='#6699CC', markersize=3, linestyle='None', linewidth=1),
    Line2D([0], [0], marker='o', color='#44AA99', label=r'$N_0$',
           markerfacecolor='#44AA99', markersize=3, linestyle='None', linewidth=1)
]

# Extract labels from the handles
marker_labels = [h.get_label() for h in marker_handles]

ax1.annotate(rf'$N_D$ $R^2$ = {r_value_ND**2:.2f}, p-val = {p_value_ND:.2g}', xy=(0.69, .6), xycoords='figure fraction',
                xytext=(0.68, .6), textcoords='figure fraction', fontsize = 10,
                ha='left', va='bottom'
                )
ax1.annotate(r'$N_{D}$ ' + f"y = {slope_ND:.2g}x + {intercept_ND:.2g}", xy = (0.69, .53), 
             fontsize =10,  xycoords='figure fraction',
                xytext=(0.68, .53), textcoords='figure fraction',
                ha='left', va='bottom'
                )

ax1.annotate(f'$N_0$ $R^2$ = {r_value_N0**2:.2f}, p-val = {p_value_N0:.2g}', xy=(0.69, .36), xycoords='figure fraction',
                xytext=(0.69, .36), textcoords='figure fraction',fontsize = 10,
                ha='left', va='bottom'
                )
ax1.annotate(r'$N_{0}$ '+ f"y = {slope_N0:.2g}x + {slope_N0:.2g}", xy = (0.69, .3), 
             fontsize =10,  xycoords='figure fraction',
                xytext=(0.69, .3), textcoords='figure fraction',
                ha='left', va='bottom'
                )

'''
plt.annotate('N0 rsq = {}'.format(round(r_sq_N0, 2)), xy = (200,70), fontsize =12 )
plt.annotate('ND rsq = {}'.format(round(r_sq_ND, 2)), xy = (0.22,70), fontsize =12 )

plt.annotate('RMSE = {} mm'.format(round(rmse,2)), xy = (0.22,65), fontsize =12)
plt.annotate(f"y = {a:.2f}x + {b:.2f}", xy = (0.22, 60), fontsize =12)
'''
fig.legend(handles = marker_handles, labels = marker_labels,
           bbox_to_anchor=(0.69, 0.75), 
           loc='center left', frameon=False,
           prop={'size': 12})

# add other info
plt.subplots_adjust(right=0.67, left = 0.12)  # Shrinks the plot width to leave space on the right
fig.savefig(f'{outDir}\\N0_ND_vs_Elev_300dpi', dpi = 300)
plt.show()

# Add plot of latitude vs N0_fit_Des
lat_lookup = dict(zip(metadata_df['Original_ID'], metadata_df['Lat DD']))
df_grav['Lat'] = df_grav['OldName'].map(lat_lookup)

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.scatter(df_grav['Lat'], df_grav['N0_fit_Des'], color='#44AA99', s=20, marker='o')
ax1.set_xlabel('Latitude (decimal degrees)', fontsize=14)
ax1.set_ylabel(r'$N_{0}$ value (cph)', fontsize=14)
fig.tight_layout()
fig.savefig(f'{outDir}\\N0_vs_Latitude_300dpi', dpi=300)
plt.show()
