# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:14:33 2025

@author: sbecker14

"""
# note this script requires the packages from the environment containing requirements2

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from datetime import datetime
import numpy as np

# set up output directory
stamp = datetime.now().strftime("%Y%m%d")
inDir = 'C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO'
outDir = os.path.normpath(inDir + os.sep + 'Calibration_Plots_output'+stamp) + '\\'   # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

os.chdir(inDir)

# load landcover descriptions: 
RF_veginfo = pd.read_csv("Data\\RoaringFork_CRNS_metadata.csv")
RF_veginfo.replace('RF 5', 'RF5', inplace=True) 
veg_dict = dict(zip(RF_veginfo['Original_ID'], RF_veginfo['Grouped Land Cover']))

# load calibration day data: 
site_var = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20251212\\Site_Calibration_data_summary.csv'))

#load parameter fits
Param = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20251212\\Parameter_fit.csv'))

# get uncertainty values for gravimetric samples: 
cal_data = pd.read_excel("Data\\Mock Calibration Summary_20251106.xlsx")
sd_row = cal_data.loc[22, site_var['OldName'].values]

lw_2rows = cal_data.loc[27:28, site_var['OldName'].values]

soc_2rows = cal_data.loc[30:31, site_var['OldName'].values]

# Build DataFrame; convert all to float arrays to avoid alignment issues
sd_df = pd.DataFrame({
    'lw_mean': lw_2rows.mean(axis=0).astype(float),
    'lw_std':  lw_2rows.std(axis=0).astype(float),
    'soc_mean': soc_2rows.mean(axis=0).astype(float),
    'soc_std': soc_2rows.std(axis=0).astype(float),
    'gwc_std': sd_row.astype(float).values   # <-- convert Series to aligned array
})

# Columns to propagate
cols = ['lw_std', 'soc_std', 'gwc_std']

# Convert selected columns to a NumPy array explicitly
std_array = sd_df[cols].to_numpy()

# Compute propagated uncertainty
sigma_S_values = np.sqrt(np.sum(std_array**2, axis=1))

# Wrap back into a Series with the same index as sd_df
sigma_S = pd.Series(sigma_S_values, index=sd_df.index, name='sigma_S')

site_var['sigma_S'] = site_var['OldName'].map(sigma_S)

site_var['N_pvisd_sigma'] = site_var['N_pvisd_Des']*site_var['Raw_Mod_cv']

# Define the model function: f(N, N0)
def Des_model(N, N0):
    return 0.0808 / (N / N0 - 0.372) - 0.115

#set figure font sizes: 
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

# Plot 1
# get outlier metrics

# find outliers using inter quartile range
Q1_Des = np.percentile(site_var['Des_resid_N0_fit_to_all'], 25)
Q3_Des = np.percentile(site_var['Des_resid_N0_fit_to_all'], 75)
print(f"Q1_Des: {Q1_Des}")
print(f"Q3_Des: {Q3_Des}")
IQR_Des = Q3_Des - Q1_Des
print(f"IQR_Des: {IQR_Des}")
lower_iqr_bound_Des = Q1_Des - 1.5 * IQR_Des
upper_iqr_bound_Des = Q3_Des + 1.5 * IQR_Des
print(f"Lower Bound: {lower_iqr_bound_Des}")
print(f"Upper Bound: {upper_iqr_bound_Des}")

iqr_outliers_Des = [x for x in site_var['Des_resid_N0_fit_to_all'] if x < lower_iqr_bound_Des or x > upper_iqr_bound_Des]
print(f"Des Outliers: {iqr_outliers_Des}")

# find outliers in the residuals # keep working on these: want to just find 1d outliers from residuals now

sd_resid_des = np.std(site_var['Des_resid_N0_fit_to_all'])
mean_resid_des = np.mean(site_var['Des_resid_N0_fit_to_all'])
upper_des = mean_resid_des + 2*sd_resid_des
lower_des = mean_resid_des - 2*sd_resid_des

# find outliers using inter quartile range
Q1_UTS = np.percentile(site_var['UTS_resid_N0_fit_to_all'], 25)
Q3_UTS = np.percentile(site_var['UTS_resid_N0_fit_to_all'], 75)
print(f"Q1_UTS: {Q1_UTS}")
print(f"Q3_UTS: {Q3_UTS}")
IQR_UTS = Q3_UTS - Q1_UTS
print(f"IQR_UTS: {IQR_UTS}")
lower_iqr_bound_UTS = Q1_UTS - 1.5 * IQR_UTS
upper_iqr_bound_UTS = Q3_UTS + 1.5 * IQR_UTS
print(f"Lower Bound: {lower_iqr_bound_UTS}")
print(f"Upper Bound: {upper_iqr_bound_UTS}")

iqr_outliers_UTS = [x for x in site_var['UTS_resid_N0_fit_to_all'] if x < lower_iqr_bound_UTS or x > upper_iqr_bound_UTS]
print(f"Outliers: {iqr_outliers_UTS}")

# find outliers in the residuals # keep working on these: want to just find 1d outliers from residuals now

sd_resid_uts = np.std(site_var['UTS_resid_N0_fit_to_all'])
mean_resid_uts = np.mean(site_var['UTS_resid_N0_fit_to_all'])
upper_uts = mean_resid_uts + 2*sd_resid_uts
lower_uts = mean_resid_uts - 2*sd_resid_uts

plt.hist(site_var['UTS_resid_N0_fit_to_all'])
plt.show()

plt.hist(site_var['Des_resid_N0_fit_to_all'])
plt.show()
# plot 1a
# total water vs. moderated counts for Desilets
# plot gravimetric water content
# plot TDR data
# plot Desilets equation
# 1b
# Residuals of universal calibration using UTS

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(7, 9), sharey=False)

# === First axis: scatter plots ===

categories = site_var['NewName'].astype("category").cat.categories
cat_codes_site = site_var['NewName'].astype("category").cat.codes

# Choose colormap
# 15-color Paul Tol palette (color-blind-friendly)
tol_colors_15 = [
    "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77",
    "#CC6677", "#882255", "#AA4499", "#DDDDDD", "#332288",
    "#661110", "#CC79A7", "#6699CC", "#33BBEE", "#FFCC00"
]
cmap = mcolors.ListedColormap(tol_colors_15)

# Plot site_var on top
ax1.errorbar(
    site_var['N_pvisd_Des'],
    site_var['Sample_total_swc_g'],
    yerr=site_var['sigma_S'],
    xerr=site_var['N_pvisd_sigma'],
    fmt='none',      # no markers
    ecolor='#333333',
    elinewidth=1,
    capsize=2,
    zorder=3
)
sc2 = ax1.scatter(
    site_var['N_pvisd_Des'], site_var['Sample_total_swc_g'],
    c=cat_codes_site, cmap=cmap, zorder = 4,
    alpha=1, s=45, marker='D', edgecolor='black', linewidth=1.2, label='Soil Samples'
)

# Plot Desilets equation 
N0_Des1N0 = Param['Fit'][0]
N0_Des_upper = Param['Upper 95 CI'][0]
N0_Des_lower = Param['Lower 95 CI'][0]
# set N range for predicting total water: 
N = range(1850, int(N0_Des1N0))  # Convert to integer

pred_Des  = Des_model(N, N0_Des1N0)
pred_Des_upper = Des_model(N, N0_Des_upper)
pred_Des_lower = Des_model(N, N0_Des_lower)

ax1.fill_between(N, pred_Des_lower, pred_Des_upper, color='black', alpha=0.2, label="N0 95% CI", zorder=2)
ax1.plot(N, pred_Des, linewidth = 1, color = 'black', label = 'Desilets et al. (2011)', zorder = 3)

# Add category legend
handles = [plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=cmap(i), label=cat,
                      markersize=15)
           for i, cat in enumerate(categories)]

fig.legend(handles=handles, title="Site", bbox_to_anchor=(0.81, 0.5), 
           prop={'size': 11}, loc='center left', frameon=False)  # Removes the border

marker_handles = [
    Line2D([0], [0], marker='D', color='black', label='Soil \nSamples',
           markerfacecolor='gray', markersize=7, linestyle='None', linewidth=1.2),
    Line2D([0], [0], marker='s', color='black', label=r'$N_0$ 95% CI',
           markerfacecolor='black',alpha = 0.2, markersize=10, linestyle='None', linewidth=1),
]

# Dummy error bar for legend
errorbar_handle = ax1.errorbar(
    [0], [0], yerr=[1], fmt='none',
    ecolor='#333333', elinewidth=1, capsize=3,
    label='1 SD error bars'
)
fig.legend(handles=[errorbar_handle], labels=['1 SD \nerror bars'], loc='center left', bbox_to_anchor = (0.8, 0.81), frameon=False, prop={'size': 11})

# Extract labels from the handles
marker_labels = [h.get_label() for h in marker_handles]

fig.legend(handles = marker_handles, labels = marker_labels,
           bbox_to_anchor=(0.80, 0.87), 
           loc='center left', frameon=False,
           prop={'size': 11})

ax1.set_title("(a)", fontsize=14, fontweight='bold')
ax1.set_xlabel(r'Daily Avg. $N_{pvisd}$ (cph)', fontsize = 14)
ax1.set_ylabel(r'$\theta_p + \theta_{lw} + \theta_{SOC}$ (g g$^{-1}$)', fontsize=14)
ax1.set_ylim(-0.01, 0.75)
ax1.set_xlim(2000, 4000)

# === Second axis plot (Desilets) ===


# sample residual export this from previous code????
#pred_samples = 

ax2.scatter(site_var['N_pvisd_Des'], site_var['Des_resid_N0_fit_to_all'],
            c=cat_codes_site, cmap=cmap,zorder = 3,
            alpha=1.0, s=45, marker='D', edgecolor='black', linewidth=1.2, label='Soil Samples')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.axhline(y=mean_resid_des, color='red', linewidth=1)
#ax2.fill_between(N, lower_des, upper_des, color='red', alpha=0.3, zorder=2)
ax2.axhline(y = upper_iqr_bound_Des, color = 'red', linestyle='--', linewidth=1)
ax2.axhline(y = lower_iqr_bound_Des, color = 'red', linestyle='--', linewidth=1)
ax2.set_title("(b)", fontsize=14, fontweight='bold')
ax2.set_xlabel(r'Daily Avg. $N_{pvisd}$ (cph)', fontsize = 14)
ax2.set_ylabel(r'Obs. - Pred. (g g$^{-1}$)', fontsize=14)
ax2.set_ylim(-0.2, 0.2)
ax2.set_xlim(2000, 4000)
# Third axis plot (UTS)

# sample residual export this from previous code????
#pred_samples = 

ax3.scatter(site_var['N_pisd_UTS'], site_var['UTS_resid_N0_fit_to_all'],
            c=cat_codes_site, cmap=cmap,zorder = 3,
            alpha=1.0, s=45, marker='D', edgecolor='black', linewidth=1.2, label='Soil Samples')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.axhline(y=mean_resid_uts, color='red', linewidth=1)
#ax3.fill_between(N, lower_uts, upper_uts, color='red', alpha=0.3, zorder=2)
ax3.axhline(y = upper_iqr_bound_UTS, color = 'red', linestyle = '--', linewidth = 1)
ax3.axhline(y = lower_iqr_bound_UTS, color = 'red', linestyle = '--', linewidth = 1, label = 'IQR Bound')
ax3.set_title("(c)", fontsize=14, fontweight='bold')
ax3.set_xlabel(r'Daily Avg. $N_{pisd}$ (cph)', fontsize = 14)
ax3.set_ylabel(r'Obs. - Pred. (g g$^{-1}$)', fontsize=14)
ax3.set_ylim(-0.2, 0.2)
ax3.set_xlim(2000, 4000)

fig.legend(handles = [Line2D([0], [0], marker = 'none',  color='red', label="mean",
                                        linewidth=1),
                      #Line2D([0], [0], marker='s', color='red', label='2 st. dev',
                            # markerfacecolor='red', alpha = 0.3, markersize=10, linestyle='None', linewidth=1), 
                       
                      Line2D([0], [0], marker = 'none', linestyle='--', color='red', label="IQR Bound",
                             linewidth=1)],
                       
           labels = ['Mean', "IQR \nOutlier \nThreshold",],
           bbox_to_anchor=(0.80, 0.20),  
           loc='center left', frameon=False,
           prop={'size': 11})
# make x limits match the left hand plot
plt.tight_layout()
plt.subplots_adjust(right=0.79, left = 0.12)  # Shrinks the plot width to leave space on the right

fig.savefig(f'{outDir}\\SingleN0_A_Des_B_Des_C_UTS_300dpi', dpi = 300)
plt.show()

'''
###############################################################################
### repeat for forest fit: 
###############################################################################

Des_FN0 = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20250813\\all_Des_forest_SingleN0_predictions.csv'))
Des_FN0_stats = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20250813\\all_Des_forest_SingleN0_SiteFitStats.csv'))

UTS_FN0 = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20250813\\all_UTS_forest_SingleN0_predictions.csv'))

site_var_F = site_var[site_var['Canopy']=='forest']

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(7, 9), sharey=False)

# === First axis: scatter plots ===
# Combine categories from both DataFrames
combined = pd.concat([Des_FN0['NewName'], site_var_F['Site']])
categories = combined.astype("category").cat.categories
cat_codes_Des = Des_FN0['NewName'].astype("category").cat.codes
cat_codes_site = site_var_F['Site'].astype("category").cat.codes
cat_codes_UTS = UTS_FN0['NewName'].astype('category').cat.codes
# Choose colormap
# 15-color Paul Tol palette (color-blind-friendly)
selected_colors = [
    "#88CCEE",  # Light blue
    "#117733",  # Dark green
    "#DDCC77",  # Sand yellow
    "#CC6677",  # Muted red
    "#332288"   # Dark blue
]
cmap = mcolors.ListedColormap(selected_colors)

# Plot Des_1N0 on ax1
sc1 = ax1.scatter(
    Des_FN0['scaled_stationary_N_Des'], Des_FN0['TDR_tot_swc_g'],
    c=cat_codes_Des, cmap=cmap,
    alpha=0.5, s=40, marker='o', label='TDR'
)

# Plot site_var on top
sc2 = ax1.scatter(
    site_var_F['N_Correct_Des'], site_var_F['Sample_total_swc_g'],
    c=cat_codes_site, cmap=cmap,zorder = 4,
    alpha=1.0, s=140, marker='^', edgecolor='black', linewidth=1.6, label='Soil Samples'
)

# Plot Desilets equation 
N0_DesFN0 = Des_FN0_stats['N0_fit'].mean()
N0_Des_upperF = Des_FN0_stats['N0_CI_upper'].mean()
N0_Des_lowerF = Des_FN0_stats['N0_CI_lower'].mean()
# set N range for predicting total water: 
N = range(1850, int(N0_Des1N0))  # Convert to integer

pred_Des  = Des_model(N, N0_DesFN0)
pred_Des_upper = Des_model(N, N0_Des_upperF)
pred_Des_lower = Des_model(N, N0_Des_lowerF)

ax1.fill_between(N, pred_Des_lower, pred_Des_upper, color='black', alpha=0.4, label="N0 95% CI", zorder=2)
ax1.plot(N, pred_Des, linewidth = 1, color = 'black', label = 'Desilets et al. (2011)', zorder = 3)
# Add category legend
handles = [plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=cmap(i), label=cat,
                      markersize=15)
           for i, cat in enumerate(categories)]

fig.legend(handles=handles, title="Site", bbox_to_anchor=(0.81, 0.65), 
           prop={'size': 11}, loc='center left', frameon=False)  # Removes the border

marker_handles = [
    Line2D([0], [0], marker='o', color='gray', label='TDR',
           markersize=8, linestyle='None', alpha=0.5),
    Line2D([0], [0], marker='^', color='black', label='Soil \nSamples',
           markerfacecolor='gray', markersize=11, linestyle='None', linewidth=1.6),
    Line2D([0], [0], marker='s', color='black', label=r'$N_0$ 95% CI',
           markerfacecolor='black',alpha = 0.4, markersize=11, linestyle='None', linewidth=1),
]

# Extract labels from the handles
marker_labels = [h.get_label() for h in marker_handles]

fig.legend(handles = marker_handles, labels = marker_labels, title="Data Type",
           bbox_to_anchor=(0.8, 0.85),  # lower than the Site legend
           loc='center left', frameon=False,
           prop={'size': 11})

ax1.set_title("(a)", fontsize=14, fontweight='bold')
ax1.set_xlabel(r'Daily Avg. $N_{pvisd}$ (cph)', fontsize = 14)
ax1.set_ylabel(r'$\theta_p + \theta_{lw} + \theta_{SOC}$ (g g$^{-1}$)', fontsize=14)
ax1.set_ylim(-0.01, 0.76)
ax1.set_xlim(1850, 4000)
# === Second axis: new plot ===
# Example: a line plot or any other plot you want
# Replace with your own plot logic
ax2.scatter(Des_FN0['scaled_stationary_N_Des'], Des_FN0['TDR_tot_swc_g_resid_Des'], 
            c=cat_codes_UTS, cmap=cmap,
            alpha=0.5, s=40, marker='o', label='TDR')

ax2.scatter(site_var_F['N_Correct_Des'], site_var_F['Des_resid_singleN0'],
            c=cat_codes_site, cmap=cmap,zorder = 3,
            alpha=1.0, s=140, marker='^', edgecolor='black', linewidth=1.6, label='Soil Samples')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_title("(b)", fontsize=14, fontweight='bold')
ax2.set_xlabel(r'Daily Avg. $N_{pvisd}$ (cph)', fontsize = 14)
ax2.set_ylabel(r'Obs. - Pred. (g g$^{-1}$)', fontsize=14)
ax2.set_ylim(-0.4 ,0.2)
ax2.set_xlim(1850, 4000)

# Third axis plot
ax3.scatter(UTS_FN0['scaled_stationary_N_UTS'], UTS_FN0['TDR_tot_swc_g_resid_UTS'], 
            c=cat_codes_UTS, cmap=cmap,
            alpha=0.5, s=40, marker='o', label='TDR')

ax3.scatter(site_var_F['N_Correct_UTS'], site_var_F['UTS_Resid_singleN0'],
            c=cat_codes_site, cmap=cmap,zorder = 3,
            alpha=1.0, s=140, marker='^', edgecolor='black', linewidth=1.6, label='Soil Samples')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.set_title("(c)", fontsize=14, fontweight='bold')
ax3.set_xlabel(r'Daily Avg. $N_{pisd}$ (cph)', fontsize = 14)
ax3.set_ylabel(r'Obs. - Pred. (g g$^{-1}$)', fontsize=14)
ax3.set_ylim(-0.4, 0.2)
ax3.set_xlim(1850, 4000)
plt.tight_layout()
plt.subplots_adjust(right=0.81, left = 0.12)  # Shrinks the plot width to leave space on the right

fig.savefig(f'{outDir}\\Forest_SingleN0_A_Des_B_Des_C_UTS_300dpi', dpi = 300)
plt.show()

##############################################################################
# repeat for half-forest
###############################################################################
Des_HFN0 = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20250813\\all_Des_half-forest_SingleN0_predictions.csv'))
Des_HFN0_stats = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20250813\\all_Des_half-forest_SingleN0_SiteFitStats.csv'))

UTS_HFN0 = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20250813\\all_UTS_half-forest_SingleN0_predictions.csv'))

site_var_HF = site_var[site_var['Canopy']=='half-forest']

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(7, 9), sharey=False)

# === First axis: scatter plots ===
# Combine categories from both DataFrames
combined = pd.concat([Des_HFN0['NewName'], site_var_HF['Site']])
categories = combined.astype("category").cat.categories
cat_codes_Des = Des_HFN0['NewName'].astype("category").cat.codes
cat_codes_site = site_var_HF['Site'].astype("category").cat.codes
cat_codes_UTS = UTS_HFN0['NewName'].astype('category').cat.codes
# Choose colormap
# 15-color Paul Tol palette (color-blind-friendly)
selected_colors = [ 
    "#88CCEE",  # Light blue
    "#117733",  # Dark green
    "#DDCC77",  # Sand yellow
    "#CC6677",  # Muted red
    "#332288",  # Dark blue
    "#AA4499",  # Purple-magenta
    "#999933"   # Olive-ish green
]
cmap = mcolors.ListedColormap(selected_colors)

# Plot Des_1N0 on ax1
sc1 = ax1.scatter(
    Des_HFN0['scaled_stationary_N_Des'], Des_HFN0['TDR_tot_swc_g'],
    c=cat_codes_Des, cmap=cmap,
    alpha=0.5, s=40, marker='o', label='TDR'
)

# Plot site_var on top
sc2 = ax1.scatter(
    site_var_HF['N_Correct_Des'], site_var_HF['Sample_total_swc_g'],
    c=cat_codes_site, cmap=cmap,zorder = 4,
    alpha=1.0, s=140, marker='^', edgecolor='black', linewidth=1.6, label='Soil Samples'
)

# Plot Desilets equation 
N0_DesHFN0 = Des_HFN0_stats['N0_fit'].mean()
N0_Des_upperHF = Des_HFN0_stats['N0_CI_upper'].mean()
N0_Des_lowerHF = Des_HFN0_stats['N0_CI_lower'].mean()
# set N range HFor predicting total water: 
N = range(1850, int(N0_Des1N0))  # Convert to integer

pred_Des  = Des_model(N, N0_DesHFN0)
pred_Des_upper = Des_model(N, N0_Des_upperHF)
pred_Des_lower = Des_model(N, N0_Des_lowerHF)

ax1.fill_between(N, pred_Des_lower, pred_Des_upper, color='black', alpha=0.4, label="N0 95% CI", zorder=2)
ax1.plot(N, pred_Des, linewidth = 1, color = 'black', label = 'Desilets et al. (2011)', zorder = 3)
# Add category legend
handles = [plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=cmap(i), label=cat,
                      markersize=15)
           for i, cat in enumerate(categories)]

fig.legend(handles=handles, title="Site", bbox_to_anchor=(0.81, 0.6), 
           prop={'size': 11}, loc='center left', frameon=False)  # Removes the border

marker_handles = [
    Line2D([0], [0], marker='o', color='gray', label='TDR',
           markersize=8, linestyle='None', alpha=0.5),
    Line2D([0], [0], marker='^', color='black', label='Soil \nSamples',
           markerfacecolor='gray', markersize=11, linestyle='None', linewidth=1.6),
    Line2D([0], [0], marker='s', color='black', label=r'$N_0$ 95% CI',
           markerfacecolor='black',alpha = 0.4, markersize=11, linestyle='None', linewidth=1),
]

# Extract labels from the handles
marker_labels = [h.get_label() for h in marker_handles]

fig.legend(handles = marker_handles, labels = marker_labels, title="Data Type",
           bbox_to_anchor=(0.8, 0.85),  # lower than the Site legend
           loc='center left', frameon=False,
           prop={'size': 11})

ax1.set_title("(a)", fontsize=14, fontweight='bold')
ax1.set_xlabel(r'Daily Avg. $N_{pvisd}$ (cph)', fontsize = 14)
ax1.set_ylabel(r'$\theta_p + \theta_{lw} + \theta_{SOC}$ (g g$^{-1}$)', fontsize=14)
ax1.set_ylim(-0.01, 0.76)
ax1.set_xlim(1850, 4000)

# === Second axis: new plot ===
ax2.scatter(Des_HFN0['scaled_stationary_N_Des'], Des_HFN0['TDR_tot_swc_g_resid_Des'], 
            c=cat_codes_UTS, cmap=cmap,
            alpha=0.5, s=40, marker='o', label='TDR')
# sample residual export this from previous code????
#pred_samples = 

ax2.scatter(site_var_HF['N_Correct_Des'], site_var_HF['Des_resid_singleN0'],
            c=cat_codes_site, cmap=cmap,zorder = 3,
            alpha=1.0, s=140, marker='^', edgecolor='black', linewidth=1.6, label='Soil Samples')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_title("(b)", fontsize=14, fontweight='bold')
ax2.set_xlabel(r'Daily Avg. $N_{pvisd}$ (cph)', fontsize = 14)
ax2.set_ylabel(r'Obs. - Pred. (g g$^{-1}$)', fontsize=14)
ax2.set_ylim(-0.25, 0.2)
ax2.set_xlim(1850, 4000)

# Third axis plot:
ax3.scatter(UTS_HFN0['scaled_stationary_N_UTS'], UTS_HFN0['TDR_tot_swc_g_resid_UTS'], 
            c=cat_codes_UTS, cmap=cmap,
            alpha=0.5, s=40, marker='o', label='TDR')
ax3.scatter(site_var_HF['N_Correct_UTS'], site_var_HF['UTS_Resid_singleN0'],
            c=cat_codes_site, cmap=cmap,zorder = 3,
            alpha=1.0, s=140, marker='^', edgecolor='black', linewidth=1.6, label='Soil Samples')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.set_title("(c)", fontsize=14, fontweight='bold')
ax3.set_xlabel(r'Daily Avg. $N_{pisd}$ (cph)', fontsize = 14)
ax3.set_ylabel(r'Obs. - Pred. (g g$^{-1}$)', fontsize=14)
ax3.set_ylim(-0.25, 0.2)
ax3.set_xlim(1850, 4000)
plt.tight_layout()
plt.subplots_adjust(right=0.81, left = 0.12)  # Shrinks the plot width to leave space on the right

fig.savefig(f'{outDir}\\HalfForest_SingleN0_A_Des_B_Des_C_UTS_300dpi', dpi = 300)
plt.show()

################################################################################
# repeat for non-forest
###############################################################################
Des_NFN0 = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20250813\\all_Des_non-forest_SingleN0_predictions.csv'))
Des_NFN0_stats = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20250813\\all_Des_non-forest_SingleN0_SiteFitStats.csv'))

UTS_NFN0 = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20250813\\all_UTS_non-forest_SingleN0_predictions.csv'))

site_var_NF = site_var[site_var['Canopy']=='non-forest']

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(7, 9), sharey=False)

# === First axis: scatter plots ===
# Combine categories from both DataFrames
combined = pd.concat([Des_NFN0['NewName'], site_var_NF['Site']])
categories = combined.astype("category").cat.categories
cat_codes_Des = Des_NFN0['NewName'].astype("category").cat.codes
cat_codes_site = site_var_NF['Site'].astype("category").cat.codes
cat_codes_UTS = UTS_NFN0['NewName'].astype('category').cat.codes
# Choose colormap
# 15-color Paul Tol palette (color-blind-friendly)
selected_colors = [ 
  "#88CCEE",  # Light blue
    "#117733",  # Dark green
    "#CC6677"   # Muted red
]
cmap = mcolors.ListedColormap(selected_colors)

# Plot Des_1N0 on ax1
sc1 = ax1.scatter(
    Des_NFN0['scaled_stationary_N_Des'], Des_NFN0['TDR_tot_swc_g'],
    c=cat_codes_Des, cmap=cmap,
    alpha=0.5, s=40, marker='o', label='TDR'
)

# Plot site_var on top
sc2 = ax1.scatter(
    site_var_NF['N_Correct_Des'], site_var_NF['Sample_total_swc_g'],
    c=cat_codes_site, cmap=cmap,zorder = 4,
    alpha=1.0, s=140, marker='^', edgecolor='black', linewidth=1.6, label='Soil Samples'
)

# Plot Desilets equation 
N0_DesNFN0 = Des_NFN0_stats['N0_fit'].mean()
N0_Des_upperNF = Des_NFN0_stats['N0_CI_upper'].mean()
N0_Des_lowerNF = Des_NFN0_stats['N0_CI_lower'].mean()
# set N range HFor predicting total water: 
N = range(1850, int(N0_Des1N0))  # Convert to integer

pred_Des  = Des_model(N, N0_DesNFN0)
pred_Des_upper = Des_model(N, N0_Des_upperNF)
pred_Des_lower = Des_model(N, N0_Des_lowerNF)

ax1.fill_between(N, pred_Des_lower, pred_Des_upper, color='black', alpha=0.4, label="N0 95% CI", zorder=2)
ax1.plot(N, pred_Des, linewidth = 1, color = 'black', label = 'Desilets et al. (2011)', zorder = 3)
# Add category legend
handles = [plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=cmap(i), label=cat,
                      markersize=15)
           for i, cat in enumerate(categories)]

fig.legend(handles=handles, title="Site", bbox_to_anchor=(0.81, 0.68), 
           prop={'size': 11}, loc='center left', frameon=False)  # Removes the border

marker_handles = [
    Line2D([0], [0], marker='o', color='gray', label='TDR',
           markersize=8, linestyle='None', alpha=0.5),
    Line2D([0], [0], marker='^', color='black', label='Soil \nSamples',
           markerfacecolor='gray', markersize=11, linestyle='None', linewidth=1.6),
    Line2D([0], [0], marker='s', color='black', label=r'$N_0$ 95% CI',
           markerfacecolor='black',alpha = 0.4, markersize=11, linestyle='None', linewidth=1),
]

# Extract labels from the handles
marker_labels = [h.get_label() for h in marker_handles]

fig.legend(handles = marker_handles, labels = marker_labels, title="Data Type",
           bbox_to_anchor=(0.8, 0.85),  # lower than the Site legend
           loc='center left', frameon=False,
           prop={'size': 11})

ax1.set_title("(a)", fontsize=14, fontweight='bold')
ax1.set_xlabel(r'Daily Avg. $N_{pvisd}$ (cph)', fontsize = 14)
ax1.set_ylabel(r'$\theta_p + \theta_{lw} + \theta_{SOC}$ (g g$^{-1}$)', fontsize=14)
ax1.set_ylim(-0.01, 0.76)
ax1.set_xlim(1850, 4000)

# === Second axis: new plot ===
# Example: a line plot or any other plot you want
# Replace with your own plot logic
ax2.scatter(Des_NFN0['scaled_stationary_N_Des'], Des_NFN0['TDR_tot_swc_g_resid_Des'], 
            c=cat_codes_Des, cmap=cmap,
            alpha=0.5, s=40, marker='o', label='TDR')
# sample residual export this from previous code????
#pred_samples = 

ax2.scatter(site_var_NF['N_Correct_Des'], site_var_NF['Des_resid_singleN0'],
            c=cat_codes_site, cmap=cmap,zorder = 3,
            alpha=1.0, s=140, marker='^', edgecolor='black', linewidth=1.6, label='Soil Samples')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_title("(b)", fontsize=14, fontweight='bold')
ax2.set_xlabel(r'Daily Avg. $N_{pvisd}$ (cph)', fontsize = 14)
ax2.set_ylabel(r'Obs. - Pred. (g g$^{-1}$)', fontsize=14)
ax2.set_ylim(-0.16, 0.11)
ax2.set_xlim(1850, 4000)

# Third axis plot
ax3.scatter(UTS_NFN0['scaled_stationary_N_UTS'], UTS_NFN0['TDR_tot_swc_g_resid_UTS'], 
            c=cat_codes_UTS, cmap=cmap,
            alpha=0.5, s=40, marker='o', label='TDR')
# sample residual export this from previous code????
#pred_samples = 

ax3.scatter(site_var_NF['N_Correct_UTS'], site_var_NF['UTS_Resid_singleN0'],
            c=cat_codes_site, cmap=cmap,zorder = 3,
            alpha=1.0, s=140, marker='^', edgecolor='black', linewidth=1.6, label='Soil Samples')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.set_title("(c)", fontsize=14, fontweight='bold')
ax3.set_xlabel(r'Daily Avg. $N_{pisd}$ (cph)', fontsize = 14)
ax3.set_ylabel(r'Obs. - Pred. (g g$^{-1}$)', fontsize=14)
ax3.set_ylim(-0.16, 0.11)
ax3.set_xlim(1850, 4000)
plt.tight_layout()
plt.subplots_adjust(right=0.81, left = 0.12)  # Shrinks the plot width to leave space on the right

fig.savefig(f'{outDir}\\NonForest_SingleN0_A_Des_B_Des_C_UTS_300dpi', dpi = 300)
plt.show()

'''