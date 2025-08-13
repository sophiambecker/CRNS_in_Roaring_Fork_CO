# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:39:27 2025

@author: sbecker14
"""

#import libraries
import os
import pandas as pd
import glob
from datetime import datetime
import math
import numpy as np

# set up directory
stamp = datetime.now().strftime("%Y%m%d")
inDir = 'C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Code_for_Github'
outDir = os.path.normpath(inDir + os.sep + 'CombineDataWithFunction_output'+stamp) + '\\'   # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

os.chdir(inDir)

# load data

# locally processed CRNS swc (Desilets and f_inc_3)
directory_path_nc = 'ModCountsProcessing_Des_output20250812'
file_pattern_nc = f'{directory_path_nc}/Mod*'
file_paths = glob.glob(file_pattern_nc, recursive=True)
dataframes_loc_Des_f3 = [pd.read_csv(file_path) for file_path in file_paths]

# locally processed CRNS swc (UTS and f_inc_3)
directory_path_nc = 'ModCountsProcessing_UTS_output20250813'
file_pattern_nc = f'{directory_path_nc}/Mod*'
file_paths = glob.glob(file_pattern_nc, recursive=True)
dataframes_loc_UTS_f3 = [pd.read_csv(file_path) for file_path in file_paths]

# USGS data release
directory_path_usgs = '\\Data\\Data_Release_2024'
file_pattern_usgs = f'{directory_path_usgs}/*v1.csv'
file_paths_usgs = glob.glob(file_pattern_usgs, recursive=True)
dataframes_usgs = [pd.read_csv(file_path) for file_path in file_paths_usgs]

# PRISM data
precip = pd.read_csv("Data\\DailyPRISM_Ppt.csv")

# Extract the first two characters of each file name
sitenames_usgs_list = [os.path.basename(path)[:2] for path in file_paths_usgs]

sitenames_loc_list = [os.path.splitext(os.path.basename(path))[0][4:] for path in file_paths]

# Pair site names with dataframes
dataframe_dict_usgs = dict(zip(sitenames_usgs_list, dataframes_usgs))
dataframe_dict_loc_Des_f3 = dict(zip(sitenames_loc_list, dataframes_loc_Des_f3))
dataframe_dict_loc_UTS_f3 = dict(zip(sitenames_loc_list, dataframes_loc_UTS_f3))

#set up dictionary to convert site names
sitenames_df = pd.read_excel('Data\\Data_Release_2024\\Network_paper_site_names.xlsx')

# read site-specific variables
site_var = pd.read_excel("Data\\Mock Calibration Summary.xlsx")

# time period: 
# find the earliest date with CRNS at all RF sites
# find the start date for each site

start_dates = []
for a in dataframe_dict_loc_Des_f3.values():
    date = a.iloc[1,0]
    start_dates.append(date)
    
# Format of the date strings (YYYY-MM-DD in this case)
date_format = '%Y-%m-%d %H:%M:%S'

# Convert list of strings to datetime objects
start_dates_dt = [datetime.strptime(date_str, date_format) for date_str in start_dates]

# most recent date 
#choose_start = max(start_dates_dt)

BEGIN = min(start_dates_dt)
# end date (set as end of the water year)
END = datetime.strptime('2024-09-30 00:00:00', date_format)

# filter data
'''
# function inputs: 
    locally processed df
    usgs df

# function must:
    filter to correct date range
    compute daily averages 
    depth-weight TDR data with available data
    combine CRNS SWC, TDR SWC, timestamp in single data frame
# a dataframe with timestamp, CRNS SWC, TDR SWC, sitename columns (old and new)
    
'''
# function that filters the locally processed dataframes and calculated the daily average: 

def cleanlocal_df(start, end, df_loc):
    df_loc['datetime_date'] = pd.to_datetime(df_loc['datetime'], format = 'mixed')#change from 'date_format' to 'mixed'
    df_loc_filt = df_loc[(df_loc['datetime_date'] >= start) & (df_loc['datetime_date'] <= end)]
    # compute daily averages
    # columns to compute mean on:
    agg_col_loc = ['mod_nc_cph', 'swc', 'Mod_cv', 'Mod_sqrt']
    daily_avg_loc = df_loc_filt.groupby(df_loc_filt['datetime_date'].dt.date)[agg_col_loc].mean()
    return(daily_avg_loc)



def combinedfs(oldname_str, newname_str, choose_start, choose_end):
    df_loc_Des_f3 = dataframe_dict_loc_Des_f3[oldname_str]
    df_loc_UTS_f3 = dataframe_dict_loc_UTS_f3[oldname_str]
    daily_avg_Des_f3 = cleanlocal_df(start = choose_start, end = choose_end, 
                                     df_loc = df_loc_Des_f3)
    daily_avg_UTS_f3 = cleanlocal_df(start = choose_start, end = choose_end, 
                                      df_loc = df_loc_UTS_f3)

    # test function
    #df_loc = dataframe_dict_loc['Crys 1']
    df_usgs = dataframe_dict_usgs[newname_str]

    df_usgs['datetime_date'] = pd.to_datetime(df_usgs['datetime_UTC'], format = '%Y-%m-%d %H:%M')
    df_usgs_filt = df_usgs[(df_usgs['datetime_date'] >= choose_start) & (df_usgs['datetime_date'] <= choose_end)]


    # compute daily averages
    # columns to compute mean on:

    agg_col_usgs = ['BaroPress', 'airRH', 'airT', 'N1_cts', 'N2_cts', 'SWC_5', 'SWC_10','SWC_20', 'SWC_50',
                    'sT_5', 'sT_10', 'sT_20', 'sT_50']

    daily_avg_usgs = df_usgs_filt.groupby(df_usgs_filt['datetime_date'].dt.date)[agg_col_usgs].mean()

    #clean precip data

    ppt = precip[['Date',oldname_str]]
    ppt.columns = ['Date', 'ppt']
    ppt = ppt.copy()  # fix SettingWithCopyWarning
    ppt['Date'] = pd.to_datetime(ppt['Date'], format='%m/%d/%Y')

    df_ppt_filt = ppt[(ppt['Date'] >= choose_start) & (ppt['Date'] <= choose_end)].copy()
    df_ppt_filt['Date'] = df_ppt_filt['Date'].dt.date 
    df_ppt = df_ppt_filt.set_index('Date')
    df_ppt.index.name = 'datetime_date'

    # Merge DataFrames based on index with outer join
    daily_comb = pd.concat([daily_avg_Des_f3, 
                           daily_avg_UTS_f3, daily_avg_usgs], axis=1, join='outer')

    #rename columns
    daily_comb.columns = [
        'Mod_cph_Des_f3', 'CRNS_SWC_Des_f3', 'Mod_cv', 'Mod_sqrt',
        'Mod_cph_UTS_f3', 'CRNS_SWC_UTS_f3','Mod_cv', 'Mod_sqrt'
    ] + list(daily_comb.columns[8:])

    # Select only the first occurrence of each column name
    daily_comb = daily_comb.loc[:, ~daily_comb.columns.duplicated()]

    # now do depth weighting with whatever TDR depths are available

    # How far away are the TDR sensors from the CRNS?
    r = 1 # 1 meter?

    # need site-specific bulk-density
    site_variables = {'lat':site_var[oldname_str][3], 'long': site_var[oldname_str][4], 'elev':site_var[oldname_str][6],
          'lw': site_var[oldname_str][26], 'soc': site_var[oldname_str][29], 'bda': site_var[oldname_str][24], 'swc_weighted':site_var[oldname_str][18]}

    bda = site_variables['bda']

    '''
    # FUNCTIONS FOR DEPTH-WEIGHTING FROM 
    Schrön, M., Köhli, M., Scheiffele, L., Iwema, J., Bogena, H. R., Lv, L., Martini, E.,
    Baroni, G., Rosolem, R., Weimar, J., Mai, J., Cuntz, M., Rebmann, C., Oswald, S. E.,
    Dietrich, P., Schmidt, U., and Zacharias, S.: Improving calibration and validation 
    of cosmic-ray neutron sensors in the light of spatial sensitivity, 
    Hydrol. Earth Syst. Sci., 21, 5009–5030, https://doi.org/10.5194/hess-21-5009-2017, 2017
    AND
    https://github.com/danpower101/crspy/blob/master/crspy/n0_calibration.py
    '''
    def D86(r, bd, y):
        """D86 Calculates the depth of sensor measurement (taken as the depth from which
        86% of neutrons originate)

        Parameters
        ----------
        r : float, int
            radial distance from sensor (m)
        bd : float
            bulk density (g/cm^3)
        y : float
            Soil Moisture from 0.02 to 0.50 in m^3/m^3
        """

        return(1/bd*(8.321+0.14249*(0.96655+np.exp(-0.01*r))*(20+y)/(0.0429+y)))

    def Wd(d, r, bd, y):
        """Wd Weighting function to be applied on samples to calculate weighted impact of 
        soil samples based on depth.

        Parameters
        ----------
        d : float
            depth of sample (cm)
        r : float,int
            radial distance from sensor (m)
        bd : float
            bulk density (g/cm^3)
        y : float
            Soil Moisture from 0.02 to 0.50 in m^3/m^3
        """

        return(np.exp(-2*d/D86(r, bd, y)))

    # weighting function is for VWC on the range from 0.02 to 0.5

    swc_cols = ['SWC_5', 'SWC_10','SWC_20', 'SWC_50']

    daily_comb_swcfilt = daily_comb[(daily_comb[swc_cols] >= 0.02) & (daily_comb[swc_cols] <= 0.5)]

    # create SWC dataframe
    swc = daily_comb_swcfilt[swc_cols].dropna(how = 'all') # drop dates where all sensor values are na
    swcT = swc.T

    # remove rows will all nan values
    swcT = swcT.dropna(how = 'all')

    # create a depth column
    mapping = {'SWC_5': 5, 'SWC_10': 10,'SWC_20':20, 'SWC_50':50}
    swcT['Depth_cm'] = swcT.index.map(mapping)

    # find arithmetic average of theta to serve as initial estimate and add as new row
    swcT.loc['CalibTheta'] = swcT.mean()

    CalibTheta = swcT.loc['CalibTheta'].iloc[0]

    # define the desired accuracy as less than 1 percent error:
    defineaccuracy = 0.01

    # Initialize Accuracy
    Accuracy = 1

    while Accuracy > defineaccuracy:
        # Initial Theta
        theta_init = CalibTheta
        
        # find weights for each layer in profile. Penetration depth, Dp = D86 is calculated within the function, Wd
        swcT.loc[:,'WdL'] = swcT.apply(lambda row: Wd(row['Depth_cm'], r, bda, theta_init), axis = 1)
        
        Wd_tot = swcT['WdL'][:-1].sum()
        
        # Calculate the depth-weighted average for each profile
        
        swcT.loc[:,'thetweight'] = swcT.iloc[:,0] * swcT['WdL']/Wd_tot
        
        FinalTheta = swcT['thetweight'][:-1].sum()
        
        CalibTheta = FinalTheta
        
        # Compare CalibTheta to theta_init using percent error
        Accuracy = abs((CalibTheta - theta_init) /
                   theta_init)
        print("Current Accuracy:", Accuracy)
        
    # use the same WdL for the remainder of the timestamps

    # exclude columns

    exclude_col = ['WdL', 'Depth_cm','thetweight' ]

    final_theta = {}
    theta_sd = {}
    theta_se = {}

    for column in swcT.columns:
        if column not in exclude_col:
            thetweight = swcT[column] * swcT['WdL'] / Wd_tot
            final_theta[column] = thetweight[:-1].sum()
            
            # Find sd and se of the weighted profiles
            Obs = final_theta[column]
            Weights = swcT['WdL'] / Wd_tot #normalized weights
            N_eff = sum(Weights)**2/sum(Weights**2) #https://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
        
            numerator = sum(Weights*((Obs-CalibTheta)**2))
            denominator = sum(Weights)
            if denominator ==0 or N_eff <= 1:
                theta_sd[column] = np.nan
                theta_se[column] = np.nan
            else:
                sd_w = math.sqrt(numerator / denominator * (N_eff / (N_eff - 1)))
                theta_sd[column] = sd_w
                theta_se[column] = sd_w/math.sqrt(N_eff)

    final_theta_series = pd.Series(final_theta)
    final_theta_series.name = 'WeightedTDR_SWC'

    theta_sd_series = pd.Series(theta_sd)
    theta_sd_series.name = 'WeightedTDR_SD'
    theta_se_series = pd.Series(theta_se)
    theta_se_series.name = 'WeightedTDR_SE'

    # Merge depth weighted TDR with daily_comb
    daily_comb_TDR = pd.concat([daily_comb, final_theta_series, theta_sd_series, 
                                theta_se_series, df_ppt], axis=1, join='outer')

    daily_comb_TDR = daily_comb_TDR.sort_index()


    # simplified dataframe

    keep = ['N1_cts', 'Mod_cv', 'Mod_sqrt', 'Mod_cph_Des_f3', 'CRNS_SWC_Des_f3', 'Mod_cph_UTS_f3','CRNS_SWC_UTS_f3',
            'SWC_5', 'SWC_10', 'SWC_20', 'SWC_50','WeightedTDR_SWC', 
            'WeightedTDR_SD', 'WeightedTDR_SE', 'sT_5', 'sT_10', 'sT_20', 'sT_50', 'ppt', 'BaroPress', 'airRH', 'airT',]

    daily_simple = daily_comb_TDR[keep]
    
    #rename colums
    
    new_col_names = ['Raw_Moderated_cph', 'Raw_Mod_Coeff_of_Var', 'Raw_Mod_sqrt', 'Corrected_Mod_cph_for_Des', 'SWC_Des_cm3_cm3', 
                     'Corrected_Mod_cph_for_UTS', 'SWC_UTS_cm3_cm3', 'SWC_5', 'SWC_10', 'SWC_20', 'SWC_50','WeightedTDR_SWC', 
                     'WeightedTDR_SD', 'WeightedTDR_SE', 'sT_5', 'sT_10', 'sT_20', 'sT_50', 'PRISM_ppt_mm', 'BaroPress', 'airRH', 'airT']

    daily_simple.columns = new_col_names
    daily_simple.index.name = 'Date (aggregated based on UTC)'
    daily_simple.to_csv(f'{outDir}{newname_str}_CRNS_Site_Data.csv')
    print(f'{oldname_str} finished')
    
# loop through all of the sites with the function above

for a in range(len(sitenames_df)):
    name1 = sitenames_df['Short_name'][a]
    #print(name1)
    name2 = sitenames_df['network_paper_new_name'][a]
    #print(name2)
    combinedfs(name1, name2, BEGIN, END)
