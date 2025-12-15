# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 16:22:14 2025

@author: sbecker14
"""

from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import glob
import re
import numpy as np
#from brokenaxes import brokenaxes
#import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# set up directory
stamp = datetime.now().strftime("%Y%m%d")
inDir = 'C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO'
outDir = os.path.normpath(inDir + os.sep + 'DryPeriods\\output'+stamp) + '\\'   # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

directory_path = 'C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Site_specific_predictions_output_20251215'
file_pattern = f'{directory_path}\\*SiteSpecificN0_predictions.csv'
file_paths = glob.glob(file_pattern, recursive=True)
df_ls = [pd.read_csv(file_path) for file_path in file_paths]

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
veg_dict = dict(zip(RF_veginfo['Name'], RF_veginfo['Grouped Land Cover']))

# load calibration day data: 
site_var = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20251214\\Site_Calibration_data_summary.csv'))

#load parameter fits
Param = pd.read_csv(os.path.join(inDir, 'Calibration_AnalysisWithKGE_output_20251214\\Parameter_fit.csv'))

#set up dictionary to convert site names
sitenames_df = pd.read_excel('C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Data\\Data_Release_2024\\Network_paper_site_names.xlsx')

new_to_old_name = dict(zip(sitenames_df['network_paper_new_name'], sitenames_df['Short_name']))


for s in site_names_new:
    
    THIS_SITE_new = s
    
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    df_with_skips = df_dict[THIS_SITE_new]
    
    df_with_skips['date'] = pd.to_datetime(df_with_skips['DateTime'])  # keep as datetime64
    df_with_skips['doy'] = df_with_skips['date'].dt.dayofyear    
    
    # Define the full time range (daily, hourly, etc.)
    full_time = pd.DataFrame({
        "date": pd.date_range(df_with_skips["date"].min(), df_with_skips["date"].max(), freq="D")
    })
    
    # Outer merge to insert missing rows with NaN (matplotlib with automatically skip over/leave gaps for nan values)
    
    df= pd.merge(full_time, df_with_skips, on="date", how="outer").sort_values("date")
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # find minimum SWC
    min_probe = df['WeightedTDR_SWC'].min()
    min_Des = df['theta_pred_pore_vol_Des'].min()
    min_UTS = df['theta_pred_pore_vol_UTS'].min()
    
    df['SWC_level'] = np.where(
        (df['theta_pred_pore_vol_Des'] < 0.06) | 
        (df['theta_pred_pore_vol_UTS'] < 0.06),
        'VeryDry',
        'NotVeryDry'
    )
    
    # Step 2: identify transitions into and out of 'VeryDry'
    df['is_verydry'] = (df['SWC_level'] == 'VeryDry')
    
    # Detect changes (True where a new period starts)
    df['change'] = df['is_verydry'].ne(df['is_verydry'].shift()).cumsum()
    
    # Step 3: group by each continuous period
    periods = (
        df[df['is_verydry']]  # only consider VeryDry periods
        .groupby('change')
        .agg(start_date=('date', 'min'),
             end_date=('date', 'max'),
             length_days=('date', lambda x: (x.max() - x.min()).days + 1))
        .reset_index(drop=True)
    )
    
    print(THIS_SITE_new)
    print(periods)
    periods['length_days'] = pd.to_numeric(periods['length_days'], errors = 'coerce')
    
    periods_filt = periods[periods['length_days']>=7]
    
    periods_filt.to_csv(f'{outDir}\\{THIS_SITE_new}_DryPeriods.csv')
    
    # make a plot for every dry period
    
    for i in range(len(periods_filt)):
        print(periods_filt.iloc[i,0])
        xlims = (
            pd.Timestamp(periods_filt.iloc[i,0]).date(), pd.Timestamp(periods_filt.iloc[i,1]).date()
            
        )
        
        plt.rcParams.update({'font.size': 12})  # set global font size
        
        fig = plt.figure(figsize = (6,5))
        
        # Create subplots
        # Initialize axes array as a 2D numpy array
        axes = plt.subplot()
        
        #second row
        
        axes.plot(df['date'], df['WeightedTDR_SWC'], linestyle = '-', color="gray", 
                  linewidth = 2, label="CS655", alpha = 1)
        
        axes.plot(df['date'], df['swc_univ_N0_pred_pore_vol_Des'], linestyle = '-', color="#AA4499", 
                  linewidth = 1, label="Network-wide \nDesilets", alpha = 0.6 )

        
        axes.plot(df['date'], df['swc_univ_ND_pred_pore_vol_UTS'], linestyle = '-', color="#999933", 
                  linewidth = 1, label="Network-wide \nUTS", alpha = 0.76 )
 
        
        axes.plot(df['date'], df['theta_pred_pore_vol_Des'], linestyle = '-', color="#332288", 
                  linewidth = 1, label="Site-specific \nDesilets", alpha = 0.6 )
        
        axes.plot(df['date'], df['theta_pred_pore_vol_UTS'], linestyle = '-', color="#117733", 
                  linewidth = 1, label="Site-specific \nUTS", alpha = 0.6 )

        
        axes.set_ylim(0, 0.2)
        
        '''axes[2,0].set_ylim(np.nanmin(df['Mod_cph_Des_f3'])-300, 
                      np.nanmax(df['Mod_cph_Des_f3'])+10)
        axes[2,1].set_ylim(np.nanmin(df['Mod_cph_Des_f3'])-300, 
                      np.nanmax(df['Mod_cph_Des_f3'])+10)'''
        
        
            
            #axes[r,1].legend()
       
        # only label bottom x-axis

            
        axes.set_xlim(xlims)
            
        # Format the x-axis to show only the first of each month
        axes.xaxis.set_major_locator(mdates.AutoDateLocator())
        axes.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes.xaxis.get_major_locator()))

        # Rotate the x-axis labels for readability
        axes.tick_params(axis='x', rotation=45)
        
        #ax[0,0].set_ylim(max(precip) + 5, 0)
        axes.set_ylabel(r'SWC  (cm$^3$ cm$^{-3}$)', color ='black')  
        axes.tick_params(axis = 'y', colors = 'black')  

        
        line_handles = [
            Line2D([0], [0], linestyle = '-', color="gray", label='CS655', linewidth=2),
            Line2D([0], [0], linestyle = '-', color="#AA4499", label='Network-wide \nDesilets', linewidth=1, alpha = 0.75),
            Line2D([0], [0], linestyle = '-', color="#999933", label='Network-wide \nUTS', linewidth=1, alpha = 0.75),
            Line2D([0], [0], linestyle = '-', color="#332288", label='Site-specific \nDesilets', linewidth=1, alpha = 0.75),
            Line2D([0], [0], linestyle = '-', color="#117733", label='Site-specific \nUTS', linewidth=1, alpha = 0.75)
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
        
        plt.savefig(f'{outDir}\\{THIS_SITE_new}_DryPeriod_{i}_TimeSeries.png', dpi=300, bbox_inches='tight')
        
        print(f'Saved figure for {THIS_SITE_new} Dry Period {i}')