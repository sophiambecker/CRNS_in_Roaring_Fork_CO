# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:41:18 2024

@author: sbecker14

"""

from __future__ import annotations

import os
os.chdir('C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Code_for_Github')
os.getcwd()

from config import Config
from helpers import get_correction_parameters, get_fi_over_time, remove_utc_from_string_date, datenum, clean_data_local, half_hour_string
from watervapor import calculate_watervapor
from UTS_helpers import convert_neutrons_to_soil_moisture_uts

import datetime as dt
from collections import OrderedDict

import numpy as np
import pandas as pd
from dateutil import parser
import glob
from scipy.optimize import minimize_scalar

stamp = dt.datetime.now().strftime("%Y%m%d") 
Dir = "C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Code_for_Github\\"
outFold = '\\ModCountsProcessing_UTS_output{}'.format(stamp)
                                             
outDir = os.path.normpath(Dir + os.sep + outFold) + '\\'    # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

# load dataframe with site specific variables (calibration water content, lattice water, etc. )
site_var = pd.read_excel("Data\\Mock Calibration Summary.xlsx")

# define processing function for a single data frame and then use it to loop through all data frames
def process_local(df: pd.DataFrame) -> None:

    # Extract the substring for the start date
    start_date = remove_utc_from_string_date(df['Message Date'][df.shape[0]-1]) #[0:10]
    
    SiteID = {202: 'PEP Mod BF3', 203: 'DE723 Mod BF3', 261: 'KAN Mod BF3', 266: 'RF 6', 
      267: 'Fry 4', 269: 'Crys 6', 270: 'Crys 1', 271: 'RF5',
      272: 'RF 7', 273: 'RF 8', 274: 'RF 3', 275: 'RF 2',
      277: 'Fry 5', 278: 'RF 1', 279: 'Crys 2', 280: 'Crys 3',
      283: 'Fry 6', 294: 'RF 9'}
    
    
    df['site'] = df['Site ID'].map(SiteID)
    Site = df['site'][0]
    site_id = df['Site ID'][0]
    site_variables = {'lat':site_var[Site][3], 'long': site_var[Site][4], 'elev':site_var[Site][6],
          'lw': site_var[Site][26], 'soc': site_var[Site][29], 'bda': site_var[Site][24], 
          'swc_weighted':site_var[Site][18], 'CalDate':site_var[Site][16]}

    df.rename(columns={"Message Date": "message_date", "T7 C":"t7_c", "H7 %":"h7_pct", 
               "N1/Samp": "n1_persamp", "N2/Samp": "n2_persamp", "N3/Samp": "n3_persamp", "E1": "e1", "P1 (mb)":"p1_mb"}, inplace=True) 
    
    df['message_date']= df['message_date'].apply(remove_utc_from_string_date)
    df['message_date']= df['message_date'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    
    # clean_data_local is a function defines in helpers.py
    # combine site-specific variables in dataframe with raw sensor data
    # may need to edit clean_data_local function depending on setup of input dataframes
    
    df_clean = clean_data_local(df, site_variables)
    
    if len(df_clean) == 0:
        print(f"No data found for site {site_id} on {start_date}")
        #return
    
    calday = df_clean['caldate'][0]
    
    long = df_clean.longi.mean()
    lat = df_clean.lati.mean()
    elev = df_clean.elevation.mean()
    yeari = df_clean.message_date.dt.year.min()
    
    lw = df_clean.lattice_water.mean()
    soc = df_clean.soil_organic_carbon.mean()
    bda = df_clean.bulk_density.mean()
    tau = lw + soc
    
    H = (elev * Config.r0) / (elev + Config.r0)
    pref = Config.P0 * (1 + Config.L / Config.trefK * H) ** Config.alpha
    fscal, _, Lpt = get_correction_parameters(pref, lat, long, yeari)
    
    # TODO: figure out what time this is updated for previous day
    # Use midnight value for the next day
    fsol_df = get_fi_over_time(pref, lat, long)
    
    text = df_clean.t7_c.values
    pext = df_clean.p1_mb.values
    rhext = df_clean.h7_pct.values
    mod = df_clean.n1_persamp_cph.values
    bare = df_clean.n2_persamp_cph.values
    
    if np.isfinite(mod).mean() == 0 or np.isfinite(bare).mean() == 0:
        print(f"No valid data for site {site_id} on {start_date}")
        #return
    
    st = datenum(start_date)
    en = datenum(dt.datetime.now())
    interval = 1 / 48  #could change this to 1/48 for half-hour data
    
    # Compute daily average water content and site average
    # TODO: the goal of this merging/for loop is to fill in time step gaps
    # It might be possible to do it another way
    ti = np.arange(st, en, interval) # hourly range from calibration date to present
    data = pd.DataFrame(
    {
    "tdat": df_clean.tdat,
    "message_date": df_clean.message_date,
    "lati": df_clean.lati,
    "longi": df_clean.longi,
    "Mod": mod,
    "Bare": bare,
    "Text": text,
    "Pext": pext,
    "RHext": rhext,
    }
    )
    data["ti"] = pd.cut(data["tdat"], bins=ti, labels=ti[:-1])
    solardate_df = pd.DataFrame(
    {"solardate": fsol_df.date.map(datenum), "fsol": fsol_df.fsol}
    )
    solardate_df["ti"] = pd.cut(solardate_df["solardate"], bins=ti, labels=ti[:-1])
    merged = pd.merge_asof(
    data.sort_values("tdat"),
    solardate_df.sort_values("solardate"),
    left_on="tdat",
    right_on="solardate",
    direction="nearest",
    )
    merged = merged.dropna(subset=["ti_x"])
    # Calculate the mean for each time interval
    grouped = (
    merged.groupby("ti_x", observed=False)
    .agg(
    {
        "message_date": "first",
        "lati": "mean",
        "longi": "mean",
        "Mod": "mean",
        "Bare": "mean",
        "Text": "mean",
        "Pext": "mean",
        "RHext": "mean",
        "fsol": "mean",
    }
    )
    .reset_index()
    )
    grouped["ti_x"] = grouped.ti_x.astype(float)
    grouped["Mod_sqrt"] = np.sqrt(grouped["Mod"])
    grouped["Mod_cv"] = grouped["Mod_sqrt"] / grouped["Mod"]
    grouped["BWE"] = Config.BWEi  # Set to BWEi for all rows
    # Add solar intensity correction
    merged_solardate = pd.merge_asof(
    solardate_df.sort_values("solardate"),
    data.sort_values("tdat"),
    left_on="solardate",
    right_on="tdat",
    direction="nearest",
    )
    merged_solardate = merged_solardate.dropna(subset=["ti_x"])
    grouped["fidum"] = grouped["fsol"]
    grouped["mind"] = np.abs(merged_solardate["solardate"] - merged_solardate["tdat"])
    grouped.replace([np.inf, -np.inf], np.nan, inplace=True)
    grouped = grouped.dropna(subset=["Mod"])
    
    # Correction for air pressure variations, Zreda 2012, Desilets 2010
    fp = np.exp((grouped.Pext.values - pref) / Lpt)
    
    # Correction for variations in atmopsheric water vapor, Rosolem 2014 JOH
    rhovref, _ = calculate_watervapor(Config.trefC, Config.rhref, Config.gama)
    rhov, _ = calculate_watervapor(
    grouped.Text.values, grouped.RHext.values, Config.gama
    )
    
    grouped['rhov_g_cm3'] = rhov/1000
    
    # correction in g/m3
    fv = 1.0 #1.0 + 0.0054 * (rhov - rhovref) * 1000 # set water vapor correction to 1, no water vapor correction should be used on raw counts for UTS soln
    fi = grouped.fidum.values
    fb = 1 / (1 - Config.eta * grouped.BWE)
    Npvibs = grouped.Mod * fp * fv * fi * fb * fscal
    grouped["Npvibs"] = Npvibs
    
    
    swc1 = df_clean.swc.mean() #use weighted gravimetric swc already calculated and stored in spreadsheet
    tw1 = swc1 + tau
    
    cd1 = datenum(calday)
    
    mask1 = np.logical_and(
        grouped.ti_x >= (cd1 - 1/24),
        grouped.ti_x <= (cd1 + 5/24)
    )
    
    # Check if there is valid data in the mask
    if not np.any(mask1):
        print(f"No valid data for site {site_id} on {start_date} (no times in mask)")
        # return or handle differently
    
    # Extract masked values
    Npvibs_masked = grouped.Npvibs[mask1]
    RHext_masked = grouped.RHext[mask1]
    Text_masked = grouped.Text[mask1]
    
    # Check if masked arrays have finite values
    if not np.isfinite(Npvibs_masked).any():
        print("No valid Npvibs in mask, using nearest available value.")
        Ncal1 = grouped.Npvibs.iloc[(grouped.ti_x - cd1).abs().argsort()].dropna().iloc[0]
    else:
        Ncal1 = np.nanmean(Npvibs_masked)
    
    if not np.isfinite(RHext_masked).any():
        print("No valid RHext in mask, using nearest available value.")
        RH_cal = grouped.RHext.iloc[(grouped.ti_x - cd1).abs().argsort()].dropna().iloc[0]
    else:
        RH_cal = np.nanmean(RHext_masked)
    
    if not np.isfinite(Text_masked).any():
        print("No valid Text in mask, using nearest available value.")
        T_cal = grouped.Text.iloc[(grouped.ti_x - cd1).abs().argsort()].dropna().iloc[0]
    else:
        T_cal = np.nanmean(Text_masked)
    
    Rhov_cal, _ = calculate_watervapor(
    RH_cal, T_cal, Config.gama
    ) # output in kg/m^3
    Rhov_cal_g = Rhov_cal/1000 # in g/cm^3
    
    # Calculate N0 value using UTS from Köhli et al., 2021
    def objective_singlerow(N0):
    
        try:
            swc_pred = convert_neutrons_to_soil_moisture_uts(
                neutron_count=Ncal1,
                n0=N0,
                air_humidity=Rhov_cal_g,
                bulk_density=bda,
                lattice_water=lw * bda,
                water_equiv_soil_organic_carbon=soc * bda,
                method="Mar21_mcnp_drf"
            )
        except:
            return np.inf  # Return a large error if conversion fails
        
        actual_swc = (tw1 - tau) * bda
        
        return abs(swc_pred - actual_swc)  # Only one value, so just absolute error
    
    
    # Run the minimizer
    res = minimize_scalar(objective_singlerow, bounds=(Ncal1, Ncal1 + 6000), method='bounded')
    
    # Best-fit N0 value
    N0_cal = res.x
    
    grouped['swc'] = grouped.apply(
        lambda row: convert_neutrons_to_soil_moisture_uts(
            neutron_count=row['Npvibs'],
            n0=N0_cal,
            air_humidity=row['rhov_g_cm3'],
            bulk_density=bda,
            lattice_water=lw * bda,
            water_equiv_soil_organic_carbon=soc * bda,
            method="Mar21_mcnp_drf",
        ), 
        axis=1
    )
    
    
    grouped["half_hour"] = grouped.message_date.map(half_hour_string)
    gb = grouped.groupby("half_hour")
    df2 = pd.DataFrame(
    OrderedDict(
    datetime=gb.half_hour.first().map(parser.parse),
    lat=gb.lati.mean(),
    long=gb.longi.mean(),
    mod_nc_cph=gb.Npvibs.mean(),
    swc=gb.swc.mean(),
    bwe=gb.BWE.mean(),
    Mod_cv = gb.Mod_cv.mean(),
    Mod_sqrt = gb.Mod_sqrt.mean(),
    N0_fit = N0_cal
    )
    )
    df2["site"] = str(site_id)
    df2['SiteName']= Site
    
    print(f"Success for site {site_id} on calibration date {calday}")
    
    return(df2, Site)


# directory path for raw data

directory_path = 'Data\\MockRawStationaryData'

# Use glob with the ** wildcard to search recursively through subdirectories for .csv files
file_pattern = f'{directory_path}/*.csv'

# Use glob to get a list of all .csv file paths recursively
file_paths = glob.glob(file_pattern, recursive=True)

# Read the files into DataFrames and store them in a list
dataframes = [pd.read_csv(file_path) for file_path in file_paths]

for s in range(len(dataframes)):
    result = process_local(dataframes[s])
    if result is None:
       print(f"Warning: process_local failed for index {s}. Skipping this iteration.")
       continue  # Skip the current iteration if result is NoneProcessed, name = process_local(dataframes[s])
    Processed, name = result
    Processed.to_csv(f'{outDir}Mod_{name}.csv')
    Processed = None
    name = None
