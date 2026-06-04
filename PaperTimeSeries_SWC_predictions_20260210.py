# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:27:14 2025

@author: sbecker14

Using data from Climate Engine
Include precipitation in plot

"""
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import glob
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# set up directory
stamp = datetime.now().strftime("%Y%m%d")
inDir = 'C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO'
outDir = os.path.normpath(inDir + os.sep + 'PaperTimeseries\\TimeSeries_output'+stamp) + '\\'   # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

directory_path = 'C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Site_specific_predictions_output_20260211'
file_pattern = f'{directory_path}\\*SiteSpecificN0_predictions.csv'
file_paths = glob.glob(file_pattern, recursive=True)
df_ls = [pd.read_csv(file_path, parse_dates=["date"]) for file_path in file_paths]

site_names_new = [os.path.basename(fp).split('_')[0] for fp in file_paths]
print(f'New site names are {site_names_new}')

# make dictionary of dataframes using new site names 

df_dict = dict(zip(site_names_new, df_ls))

# get landcover information:
RF_veginfo = pd.read_csv("C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Data\RoaringFork_CRNS_metadata.csv")
RF_veginfo.replace('RF 5', 'RF5', inplace=True) 

#RF_veginfo['Name'] = [s.replace(" ", "") for s in RF_veginfo['Original_ID']]
RF_veginfo['Name'] = RF_veginfo['Original_ID']
# create dictionary:
veg_dict = dict(zip(RF_veginfo['Name'], RF_veginfo['Land Cover ']))

# load calibration day data: 
site_var = pd.read_csv(
    os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20260210\\Site_data_summary.csv'),
    parse_dates=['SampDate']
)

#load parameter fits
Param = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20260210\\Parameter_fit.csv'))

#set up dictionary to convert site names
sitenames_df = pd.read_excel('C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Data\\Data_Release_2024_b\\Network_paper_site_names.xlsx')

new_to_old_name = dict(zip(sitenames_df['network_paper_new_name'], sitenames_df['Short_name']))

# PRISM data
precip = pd.read_csv(os.path.join(inDir, "Data\\DailyPRISM_Ppt.csv"), parse_dates=['Date'])

#THIS_SITE_new = 'R1'


for s in site_names_new:
    THIS_SITE_new = s
    
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    
    df_with_skips = df_dict[THIS_SITE_new].copy()
    
    df_with_skips['date'] = pd.to_datetime(df_with_skips['date'])  # keep as datetime64
    df_with_skips['doy'] = df_with_skips['date'].dt.dayofyear    
    
    # Define the full time range (daily, hourly, etc.)
    full_time = pd.DataFrame({
        "date": pd.date_range(df_with_skips["date"].min(), df_with_skips["date"].max(), freq="D")
    })
    
    # Outer merge to insert missing rows with NaN (matplotlib with automatically skip over/leave gaps for nan values)
    
    df= pd.merge(full_time, df_with_skips, on="date", how="outer").sort_values("date")
    
    # apply filtering
    # Define the date range for which you want to consider possible snow
    start_date = pd.Timestamp('2023-09-01')
    end_date = pd.Timestamp('2024-07-01')
    
    # find porosity 
    sitedata = site_var[site_var['NewName']==THIS_SITE_new].copy()
    site_bd = sitedata['bd'].item()
    
    site_por = 1 - (site_bd/2.65) # site porosity from particle density of 2.65 g/cm^3
    
       
    print(f'porosity for {THIS_SITE_new} is {site_por}')
    
    # make sure CRNS_SWC_UTS_f3 is numeric
    df['CRNS'] = pd.to_numeric(df['SWC_UTS_cm3_cm3'])
    
    if 'sT_50' in df.columns:
        
        snow_cover = df[((df['SWC_UTS_cm3_cm3'] >= site_por-0.05) |
                         (df['SWC_Des_cm3_cm3'] >= site_por-0.05)|
                         (df['sT_5'] <2) |(df['sT_50'] < 1)| (df['airT']<1))
                        & (df['date']>start_date) & (df['date']< end_date)]
    elif 'sT_10' in df.columns:
        snow_cover = df[((df['SWC_UTS_cm3_cm3'] >= site_por-0.05) |
                            (df['SWC_Des_cm3_cm3'] >= site_por-0.05))
                            & (df['sT_5'] <2) |(df['sT_10'] <1)| (df['airT']<1)
                           & (df['date']>start_date) & (df['date']< end_date)] 
        
    else: snow_cover = df[((df['SWC_UTS_cm3_cm3'] >= site_por-0.05) |
                        (df['SWC_Des_cm3_cm3'] >= site_por-0.05))
                        &(df['sT_5'] <2)| (df['airT']<1)
                       & (df['date']>start_date) & (df['date']< end_date)] 
    
    # make exception for site R1 because calibration date is 10/18/23
    if (THIS_SITE_new == 'R1') and (pd.Timestamp(snow_cover['date'].min()) < pd.Timestamp('2023-10-19')):
        first_snow_cover = pd.Timestamp('2023-10-19')
    else:
        first_snow_cover = snow_cover['date'].min()
       
    last_snow_cover = snow_cover['date'].max()
    print(f'First snow cover date is: {first_snow_cover}')
    print(f'Last snow cover date is: {last_snow_cover}')
    
    # drop snow dates using saturation value for snow cover ID
    #df_filt = df.drop(df[(df['date']>= first_snow_cover) & (df['date']<= last_snow_cover)].index)
    # just drop snow_cover days individually:
    match_dates = snow_cover['date'].unique()

    df_filt = df.copy()
    df_filt.loc[df_filt['date'].isin(match_dates), :] = np.nan

    # filter moderated counts with min/max
    '''
    # Step 1: Make extreme values NaN instead of dropping
    mask1 = (df_filt['Corrected_Mod_cph_for_Des'] < 1200) | \
            (df_filt['Corrected_Mod_cph_for_Des'] > 6500)
    
    df_filt.loc[mask1, ['theta_pred_pore_vol_Des', 'theta_pred_pore_vol_UTS', 
                        'swc_univ_N0_pred_pore_vol_Des', 'swc_univ_ND_pred_tot_g_UTS']] = np.nan
    
    
    # Step 2: Compute mean and SD ignoring NaN
    mod_max = np.nanmean(df_filt['scaled_stationary_N_Des']) + 2 * np.nanstd(df_filt['scaled_stationary_N_Des'])
    mod_min = np.nanmean(df_filt['scaled_stationary_N_Des']) - 2 * np.nanstd(df_filt['scaled_stationary_N_Des'])
    
    mask2 = (df_filt['scaled_stationary_N_Des'] < mod_min) | \
            (df_filt['scaled_stationary_N_Des'] > mod_max)
    
    df_filt.loc[mask2, ['theta_pred_pore_vol_Des', 'theta_pred_pore_vol_UTS', 
                        'swc_univ_N0_pred_pore_vol_Des', 'swc_univ_ND_pred_tot_g_UTS']] = np.nan
    
    '''
    # Clean up: drop all-NaN columns but keep rows
    df_filt.dropna(axis=1, how='all', inplace=True)
    
    # Keep NaN rows—do NOT drop rows with NaNs
    
    # aggregate and add ppt
    numeric_cols = df_filt.select_dtypes(include='number').columns
    # Aggregate numeric columns by date
    daily_df = (
    df_filt
    .groupby(df_filt['date'].dt.floor('D'))[numeric_cols]  # floor to midnight -> datetime64[ns]
    .mean()
    .reset_index()  # makes the grouping key a column again
    )
    
    sixhr_df = (
        df_filt
        .set_index('date')
        .resample(rule='6h', label='left', closed='left')[numeric_cols]
        .mean()
        .reset_index()
    )
    
    
    ppt = precip[['Date',THIS_SITE_old]].copy()
    ppt.columns = ['Date', 'ppt']
    ppt['date'] = pd.to_datetime(ppt['Date'], format='%m/%d/%Y')
    
    # Make sure both 'date' columns are datetime
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    ppt['date'] = pd.to_datetime(ppt['date'])
    
    df_daily_merge = (
        daily_df.merge(ppt, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )
    
    df_plot = sixhr_df.copy()
    
    xlims = (
        (pd.Timestamp('2023-07-01'), pd.Timestamp('2023-11-01')),
        (pd.Timestamp('2024-06-01'), pd.Timestamp('2024-10-01'))
    )
    
    plt.rcParams.update({'font.size': 12})  # set global font size
    
    fig = plt.figure(figsize = (6,3.7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.3, 1], width_ratios=[1, 1], wspace=0.18, hspace = 0.1)  
    
    # Create subplots
    # Initialize axes array as a 2D numpy array
    axes = np.empty((2, 2), dtype=object)  # Create empty 2x2 array for axes
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])  # Assign each subplot
            axes[i,j] = ax
            
    #first row
    axes[0,0].bar(df_daily_merge['date'], df_daily_merge['ppt'], width=0.9, color='#0072B2', align="center", label = 'Precip')
    axes[0,1].bar(df_daily_merge['date'], df_daily_merge['ppt'], width=0.9, color='#0072B2', align="center", label = 'Precip')
    
    
    # Invert y-axis to make bars come down from the top
    axes[0,0].yaxis.set_inverted(True)
    axes[0,1].yaxis.set_inverted(True)
    
    # set y limits
    #axes[0,0].set_ylim(np.nanmax(df['ppt'])+1, 0)
    #axes[0,1].set_ylim(np.nanmax(df['ppt'])+1, 0)
    
    axes[0,0].set_ylim(33, 0)
    axes[0,1].set_ylim(33, 0)
    
    
    
    #second row
    
    axes[1,0].plot(df_plot['date'], df_plot['WeightedTDR_SWC'], linestyle = '-', color="gray", 
              linewidth = 2, label="CS655", alpha = 0.7)
    axes[1,1].plot(df_plot['date'], df_plot['WeightedTDR_SWC'], linestyle = '-', color="gray", 
              linewidth = 2, label="CS655", alpha = 0.7)
    
    axes[1,0].plot(df_plot['date'], df_plot['swc_univ_N0_pred_pore_vol_Des'], linestyle = '-', color="#AA4499", 
              linewidth = 1, label="Network-wide \nDesilets", alpha = 0.75 )
    axes[1,1].plot(df_plot['date'], df_plot['swc_univ_N0_pred_pore_vol_Des'], linestyle = '-', color="#AA4499", 
              linewidth = 1, label="Network-wide \nDesilets", alpha = 0.75 )
    
    axes[1,0].plot(df_plot['date'], df_plot['swc_univ_ND_pred_pore_vol_UTS'], linestyle = '-', color="#999933", 
              linewidth = 1, label="Network-wide \nUTS", alpha = 0.75 )
    axes[1,1].plot(df_plot['date'], df_plot['swc_univ_ND_pred_pore_vol_UTS'], linestyle = '-', color="#999933", 
              linewidth = 1, label="Network-wide \nUTS", alpha = 0.75 )
    
    axes[1,0].plot(df_plot['date'], df_plot['theta_pred_pore_vol_Des'], linestyle = '-', color="#332288", 
              linewidth = 1, label="Site-specific \nDesilets", alpha = 0.75 )
    axes[1,1].plot(df_plot['date'], df_plot['theta_pred_pore_vol_Des'], linestyle = '-', color="#332288", 
              linewidth = 1, label="Site-specific \nDesilets", alpha = 0.75 )
    
    axes[1,0].plot(df_plot['date'], df_plot['theta_pred_pore_vol_UTS'], linestyle = '-', color="#117733", 
              linewidth = 1, label="Site-specific \nUTS", alpha = 0.75 )
    axes[1,1].plot(df_plot['date'], df_plot['theta_pred_pore_vol_UTS'], linestyle = '-', color="#117733", 
              linewidth = 1, label="Site-specific \nUTS", alpha = 0.75 )
    
    # get soil sample point
    site_row=None
    site_row = site_var[site_var['NewName'] == THIS_SITE_new].copy()
    site_pore_swc_vol = (site_row['Sample_total_swc_g'] - site_row['lw'] - site_row['soc_water'])*site_row['bd']
    
    axes[1,0].plot(pd.to_datetime(site_row['SampDate']), site_pore_swc_vol, marker= 'x', color = 'red', markersize = 9, markeredgewidth=2)
    axes[1,1].plot(pd.to_datetime(site_row['SampDate']), site_pore_swc_vol, marker= 'x', color = 'red', markersize = 9, markeredgewidth=2)
    
    
    axes[1,0].set_ylim(0, 0.45)
    axes[1,1].set_ylim(0, 0.45)
    
    '''axes[2,0].set_ylim(np.nanmin(df['Mod_cph_Des_f3'])-300, 
                  np.nanmax(df['Mod_cph_Des_f3'])+10)
    axes[2,1].set_ylim(np.nanmin(df['Mod_cph_Des_f3'])-300, 
                  np.nanmax(df['Mod_cph_Des_f3'])+10)'''
    
    
    # hide the spines between ax and ax2
    for r in range(axes.shape[0]):
        axes[r,0].set_xlim(xlims[0])
        axes[r,1].set_xlim(xlims[1])
        
        axes[r,0].spines.right.set_visible(False)
        axes[r,1].spines.left.set_visible(False)
        axes[r,1].tick_params(labelleft=False)  # don't put tick labels at the top
        axes[r,1].set_yticks([])
    
        # Now, let's turn towards the cut-out slanted lines.
        # We create line objects in axes coordinates, in which (0,0), (0,1),
        # (1,0), and (1,1) are the four corners of the Axes.
        # The slanted lines themselves are markers at those locations, such that the
        # lines keep their angle and position, independent of the Axes size or scale
        # Finally, we need to disable clipping.
        
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        axes[r,0].plot([1, 1], [0, 1], transform=axes[r,0].transAxes, **kwargs)
        axes[r,1].plot([0, 0], [0, 1], transform=axes[r,1].transAxes, **kwargs)
        
        #axes[r,1].legend()
    
    # only label bottom x-axis
    
    for ax in [axes[0,0], axes[0,1]]: 
        # Format the x-axis to show only the first of each month
        ax.xaxis.set_major_locator(mdates.MonthLocator()) 
        ax.xaxis.set_ticklabels([])
        
    for ax in [axes[1,0],axes[1,1]]:
        
        # Format the x-axis to show only the first of each month
        ax.xaxis.set_major_locator(mdates.MonthLocator())  # Show first of every month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%b'))  # Format as year-month-day
        
        # Rotate the x-axis labels for readability
        ax.tick_params(axis='x', rotation=45)
    
    #ax[0,0].set_ylim(max(precip) + 5, 0)
    axes[1,0].set_ylabel(r'SWC  (cm$^3$ cm$^{-3}$)', color ='black')  
    axes[1,0].tick_params(axis = 'y', colors = 'black')  
    
    axes[0,0].set_ylabel("Ppt (mm)")
    
    line_handles = [
        Line2D([0], [0], linestyle = '-', color="gray", label='in situ probe', linewidth=2),
        Line2D([0], [0], linestyle = '-', color="#AA4499", label='Network-wide \nDesilets', linewidth=1, alpha = 0.75),
        Line2D([0], [0], linestyle = '-', color="#999933", label='Network-wide \nUTS', linewidth=1, alpha = 0.75),
        Line2D([0], [0], linestyle = '-', color="#332288", label='Site-specific \nDesilets', linewidth=1, alpha = 0.75),
        Line2D([0], [0], linestyle = '-', color="#117733", label='Site-specific \nUTS', linewidth=1, alpha = 0.75),
        Line2D([0], [0], linestyle = 'none', color="red", label='Soil sample', marker = 'x', markersize = 9, markeredgewidth=2),
    ]
    
    # Extract labels from the handles
    line_labels = [h.get_label() for h in line_handles]
    
    fig.legend(handles = line_handles, labels = line_labels,
               bbox_to_anchor=(0.81, 0.45), 
               loc='center left', frameon=False,
               prop={'size': 11})
    
    
    # Label whole plot with site name and vegetation type
    fig.text(0.4, 0.95, f'{THIS_SITE_new}, {veg_dict[THIS_SITE_old]}', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.80, left = 0.12, top = 0.91)
    
    plt.savefig(f'{outDir}\\{THIS_SITE_new}_TimeSeries.png', dpi=300, bbox_inches='tight')
    
    print(f'Saved figure for {THIS_SITE_new}')
    
