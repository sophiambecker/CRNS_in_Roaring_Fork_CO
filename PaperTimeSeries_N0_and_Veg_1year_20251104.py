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
outDir = os.path.normpath(inDir + os.sep + 'PaperTimeseries\\N0_TimeSeries_output'+stamp) + '\\'   # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory



# get landcover information:
RF_veginfo = pd.read_csv("C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Data\RoaringFork_CRNS_metadata.csv")
RF_veginfo.replace('RF 5', 'RF5', inplace=True) 

#RF_veginfo['Name'] = [s.replace(" ", "") for s in RF_veginfo['Original_ID']]
RF_veginfo['Name'] = RF_veginfo['Original_ID']
# create dictionary:
veg_dict = dict(zip(RF_veginfo['Name'], RF_veginfo['Land Cover ']))


#set up dictionary to convert site names
sitenames_df = pd.read_excel('C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Data\\Data_Release_2024\\Network_paper_site_names.xlsx')

new_to_old_name = dict(zip(sitenames_df['network_paper_new_name'], sitenames_df['Short_name']))

directory_path_N0= 'C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Calibration_AnalysisWithCorrelation_output_20251205'

N0_file_pattern = f'{directory_path_N0}\\*Probe_Based_Parameters.csv'

N0_paths = glob.glob(N0_file_pattern, recursive = True)

site_names_new = [os.path.basename(fp).split('_')[0] for fp in N0_paths]

dfs_N0_ls = [pd.read_csv(file_path) for file_path in N0_paths]

dict_N0 = dict(zip(site_names_new, dfs_N0_ls))
 
# read in GDD data:

gdd_directory = os.path.join(inDir, 'Data', 'External', 'GDD', 'output20251120')
gdd_pattern = f'{gdd_directory}\\*seasonal.csv'
gdd_paths = glob.glob(gdd_pattern, recursive=True)
gdd_ls = [pd.read_csv(file_path) for file_path in gdd_paths]
site_names_gdd = [os.path.basename(fp).split('_')[0] for fp in gdd_paths]
print(f'GDD site names are {site_names_gdd}')
gdd_dict = dict(zip(site_names_gdd, gdd_ls))

# read in GSI data:
gsi_directory = os.path.join(inDir, 'Data', 'External', 'GSI', 'output20251120')
gsi_pattern = f'{gsi_directory}\\*seasonal.csv'
gsi_paths = glob.glob(gsi_pattern, recursive=True)
gsi_ls = [pd.read_csv(file_path) for file_path in gsi_paths]
site_names_gsi = [os.path.basename(fp).split('_')[0] for fp in gsi_paths]
print(f'GSI site names are {site_names_gsi}')
gsi_dict = dict(zip(site_names_gsi, gsi_ls))  
    
THIS_SITE_new = 'C1'


for s in site_names_new:
    THIS_SITE_new = s
    
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    df_with_skips = dict_N0[THIS_SITE_new]
    
    df_with_skips['date'] = pd.to_datetime(df_with_skips['date'])  # keep as datetime64
    df_with_skips['doy'] = df_with_skips['date'].dt.dayofyear    
    
    # Define the full time range (daily, hourly, etc.)
    full_time = pd.DataFrame({
        "date": pd.date_range(df_with_skips["date"].min(), df_with_skips["date"].max(), freq="D")
    })
    
    # Outer merge to insert missing rows with NaN (matplotlib with automatically skip over/leave gaps for nan values)
    
    df= pd.merge(full_time, df_with_skips, on="date", how="outer").sort_values("date")
    
    #convert time to local CO time in df
    
    #df['date_local'] = df['date'].dt.tz_localize('America/Denver')
    
    # now merge GSI, GDD, VPD with rest of df
    
    gsi_df = gsi_dict[THIS_SITE_old]
    gdd_df = gdd_dict[THIS_SITE_old]
    
    gsi_df['date'] =  pd.to_datetime(gsi_df['date'])
    gdd_df['date'] = pd.to_datetime(gdd_df['date'])
    
    all_df = df.merge(gsi_df, on='date', how = 'outer').merge(gdd_df, on='date', how = 'outer')
    
    bare_mean = np.nanmean(all_df['Bare'])
    bare_rolling = all_df['Bare'].rolling(window = 3).mean()
    all_df['Bare_norm'] = bare_rolling/bare_mean
    bare_rolling7 = all_df['Bare'].rolling(window = 14).mean()
    all_df['Bare_norm7'] = bare_rolling7/bare_mean
    
    max_VPD = all_df['VPD'].max()
    print(f'Maximum VPD for {THIS_SITE_new} is {max_VPD}')
    all_df['VPD_kPa'] = all_df['VPD']/1000
    
    all_df['N0_7'] = all_df['N0'].rolling(window = 14).mean()
    all_df['ND_7'] = all_df['ND'].rolling(window = 14).mean()
    all_df['VPD_7'] = all_df['VPD_kPa'].rolling(window = 14).mean()
    
    # keep only the 2024 portion for plotting
    all_df = all_df[all_df['date'].dt.year == 2024].copy()
    
    plt.rcParams.update({'font.size': 18})  # set global font size
    
    fig = plt.figure(figsize = (7,9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1], hspace = 0.05)
    
    # Create subplots (one column, three rows)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    
    # first row: Bare + VPD
    axes[0].plot(all_df['date'], all_df['Bare'], color='#0072B2', markersize = 2 , linewidth = 0.8, linestyle = '-', marker = 'o')
    axes[0].set_ylim(all_df['Bare'].min()-20, all_df['Bare'].max()+20)
    
    ax0b = axes[0].twinx()  # Create secondary y-axis
    ax0b.plot(all_df['date'], all_df['VPD']/1000, markersize = 2 ,linewidth = 0.8, linestyle = '-', marker = 'o', color='#661110', label = 'VPD')
    ax0b.tick_params(axis = 'y', colors = '#661110')
    ax0b.set_ylabel("VPD (kPa)", color = '#661110')
    ax0b.set_ylim(0,3)
    
    # second row: N0 and ND
    axes[1].plot(all_df['date'], all_df['N0'], linestyle = '-', color="#999933", 
              linewidth = 1, label="N0", marker = 'o', markersize = 2 )
    axes[1].plot(all_df['date'], all_df['ND'], linestyle = '-', color="#117733", 
              linewidth = 1, label="ND" , marker = 'o', markersize = 2 )
    
    # third row: GSI + GDD
    axes[2].plot(all_df['date'], all_df['cumulative_GSI'], linestyle = '-', color="#33BBEE", 
              linewidth = 1, label="Cumm. GSI")
    axes[2].set_ylim(0, all_df['cumulative_GSI'].max()+50)
    
    ax2b= axes[2].twinx()  # Create secondary y-axis
    ax2b.plot(all_df['date'], all_df['cumulative_GDD'], linestyle='-', color='#AA4499', label = 'Cum. GDD')
    ax2b.tick_params(axis = 'y', colors = '#AA4499')
    ax2b.set_ylabel("Cum. GDD", color = '#AA4499')
    ax2b.set_ylim(0, all_df['cumulative_GDD'].max()+50)
    
    # Formatting shared axes
    x_start = pd.Timestamp('2024-04-01')
    x_end = pd.Timestamp('2024-10-01')
    for ax in axes:
        ax.set_xlim(x_start, x_end)
        ax.xaxis.set_major_locator(mdates.MonthLocator())  # Show first of every month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%b'))  # Format as year-month
    
    # Only bottom panel gets x labels
    for ax in axes[:-1]:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    axes[-1].tick_params(axis='x', rotation=45)
    
    axes[2].set_ylabel('Cum. GSI', color ="#33BBEE")  
    axes[2].tick_params(axis = 'y', colors = "#33BBEE")  
    
    axes[0].set_ylabel("Bare (cph)", color = '#0072B2')
    axes[0].tick_params(axis = 'y', colors = '#0072B2')
    
    axes[1].set_ylabel(r'$N_{0}$ or $N_{D}$ (cph)')
    
    line_handles = [
        Line2D([0], [0], linestyle = '-', color="#661110", label='VPD',markersize = 2, marker = 'o', linewidth = 0.8),
        Line2D([0], [0], marker = 'o', color="#0072B2", label='Bare', markersize = 2, linestyle = '-', linewidth = 0.8),
        Line2D([0], [0], linestyle = '-', color="#999933", label=r'$N_{0}$', linewidth=1, marker = 'o', markersize = 2 ),
        Line2D([0], [0], linestyle = '-', color="#117733", label=r'$N_{D}$', linewidth=1, marker = 'o', markersize = 2 ),
        Line2D([0], [0], linestyle = '-', color="#33BBEE", label='Cum. GSI', linewidth=1),
        Line2D([0], [0], linestyle = '-', color="#AA4499", label='Cum. GDD', linewidth=1)
    ]
    
    # Extract labels from the handles
    line_labels = [h.get_label() for h in line_handles]
    
    fig.legend(handles = line_handles, labels = line_labels,
               bbox_to_anchor=(0.81, 0.5), 
               loc='center left', frameon=False,
               prop={'size': 18})
    
    
    # Label whole plot with site name and vegetation type
    fig.text(0.4, 0.95, f'{THIS_SITE_new}, {veg_dict[THIS_SITE_old]}', va='center', fontsize=14)
    
    plt.subplots_adjust(right=0.79, left = 0.12, top = 0.92)
    
    plt.savefig(f'{outDir}\\{THIS_SITE_new}_N0_TimeSeries.png', dpi=300, bbox_inches='tight')
    
    print(f'Saved figure for {THIS_SITE_new}')
    
    all_df.to_csv(f'{outDir}\\{THIS_SITE_new}_veg_variables.csv')




