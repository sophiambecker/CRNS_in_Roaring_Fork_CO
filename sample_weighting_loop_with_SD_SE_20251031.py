# -*- coding: utf-8 -*-
"""
Created on Tues May  14 11:17:33 2024
@author: Sophia Becker
Code adapted from https://github.com/danpower101/crspy/blob/master/crspy/n0_calibration.py

The below code is based on the work of Schron et al., (2017). This work established
a new weighting scheme to be applied in the calibration of Cosmic Ray Neutron Sensors.
The functions have been taken from the supplementary data.  
References:
    Schrön, M., Köhli, M., Scheiffele, L., Iwema, J., Bogena, H. R., Lv, L., Martini, E.,
    Baroni, G., Rosolem, R., Weimar, J., Mai, J., Cuntz, M., Rebmann, C., Oswald, S. E.,
    Dietrich, P., Schmidt, U., and Zacharias, S.: Improving calibration and validation 
    of cosmic-ray neutron sensors in the light of spatial sensitivity, 
    Hydrol. Earth Syst. Sci., 21, 5009–5030, https://doi.org/10.5194/hess-21-5009-2017, 2017.  
"""

# Load up the packages needed at the begining of the code
import pandas as pd
import numpy as np
from datetime import datetime
import math

"""
I am not sure what the format of the calibration sample tables will be, but they need to include the 
following columns for each sample:
    Distance_m, Bearing_deg, Depth_cm, GWC_g_g
Other columns are unnecessary. 
"""
# output path
output_path = 'C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Data\\'

# import calibration sampling data and make sure dtype is float
file_path = 'C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Data\\UCOL CRNS calibration updated 20250617 edited to remove strikethroughs.xlsx'
read_sheets = ['Crys-1', 'Crys-2', 'Crys3', 'Crys6', 'Fry4', 'Fry 5', 'Fry 6',
               'RF-1', 'RF2','RF3', 'RF5', 'RF6', 'RF7', 'RF8', 'RF9']

# Define the data types for the columns of interest
dtype_mapping = {'Distance (m)': float, 'Bearing (deg)': float, 'Depth (cm)': float, 'GWC (g/g)': float}

sheets = pd.read_excel(file_path, sheet_name = read_sheets, skiprows=3, 
                          dtype=dtype_mapping)

# read in site specific humidity, pressure, bulk density data
sitespec = pd.read_excel('C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Data\\SiteSpecificWeightingVariables_20251106.xlsx' )
# just sites with calibration sampling data:
sitespec_filt = sitespec.iloc[0:15,]

# Ensure the length of the dictionary matches the length of the DataFrame
if len(sheets) != len(sitespec_filt):
    raise ValueError("The dictionary and DataFrame must have the same number of elements")


""" 
Functions from Shcron et al 2017
     
"""

def WrX(r, x, y):
    """WrX Radial Weighting function for point measurements taken within 5m of sensor

    Parameters
    ----------
    r : float
        rescaled distance from sensor (see rscaled function below)
    x : float
        Air Humidity from 0.1 to 50 in g/m^3
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    """

    x00 = 3.7
    a00 = 8735
    a01 = 22.689
    a02 = 11720
    a03 = 0.00978
    a04 = 9306
    a05 = 0.003632
    a10 = 2.7925e-002
    a11 = 6.6577
    a12 = 0.028544
    a13 = 0.002455
    a14 = 6.851e-005
    a15 = 12.2755
    a20 = 247970
    a21 = 23.289
    a22 = 374655
    a23 = 0.00191
    a24 = 258552
    a30 = 5.4818e-002
    a31 = 21.032
    a32 = 0.6373
    a33 = 0.0791
    a34 = 5.425e-004

    x0 = x00
    A0 = (a00*(1+a03*x)*np.exp(-a01*y)+a02*(1+a05*x)-a04*y)
    A1 = ((-a10+a14*x)*np.exp(-a11*y/(1+a15*y))+a12)*(1+x*a13)
    A2 = (a20*(1+a23*x)*np.exp(-a21*y)+a22-a24*y)
    A3 = a30*np.exp(-a31*y)+a32-a33*y+a34*x

    return((A0*(np.exp(-A1*r))+A2*np.exp(-A3*r))*(1-np.exp(-x0*r)))

def WrA(r, x, y):
    """WrA Radial Weighting function for point measurements taken within 50m of sensor

    Parameters
    ----------
    r : float
         rescaled distance from sensor (see rscaled function below)
    x : float
         Air Humidity from 0.1 to 50 in g/m^3
    y : float
         Soil Moisture from 0.02 to 0.50 in m^3/m^3
    """

    a00 = 8735
    a01 = 22.689
    a02 = 11720
    a03 = 0.00978
    a04 = 9306
    a05 = 0.003632
    a10 = 2.7925e-002
    a11 = 6.6577
    a12 = 0.028544
    a13 = 0.002455
    a14 = 6.851e-005
    a15 = 12.2755
    a20 = 247970
    a21 = 23.289
    a22 = 374655
    a23 = 0.00191
    a24 = 258552
    a30 = 5.4818e-002
    a31 = 21.032
    a32 = 0.6373
    a33 = 0.0791
    a34 = 5.425e-004

    A0 = (a00*(1+a03*x)*np.exp(-a01*y)+a02*(1+a05*x)-a04*y)
    A1 = ((-a10+a14*x)*np.exp(-a11*y/(1+a15*y))+a12)*(1+x*a13)
    A2 = (a20*(1+a23*x)*np.exp(-a21*y)+a22-a24*y)
    A3 = a30*np.exp(-a31*y)+a32-a33*y+a34*x
    
    return(A0*(np.exp(-A1*r))+A2*np.exp(-A3*r))

def WrB(r, x, y):
    """WrB Radial Weighting function for point measurements taken over 50m of sensor

    Parameters
    ----------
    r : float
        rescaled distance from sensor (see rscaled function below)
    x : float
        Air Humidity from 0.1 to 50 in g/m^3
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    """
    b00 = 39006
    b01 = 15002337
    b02 = 2009.24
    b03 = 0.01181
    b04 = 3.146
    b05 = 16.7417
    b06 = 3727
    b10 = 6.031e-005
    b11 = 98.5
    b12 = 0.0013826
    b20 = 11747
    b21 = 55.033
    b22 = 4521
    b23 = 0.01998
    b24 = 0.00604
    b25 = 3347.4
    b26 = 0.00475
    b30 = 1.543e-002
    b31 = 13.29
    b32 = 1.807e-002
    b33 = 0.0011
    b34 = 8.81e-005
    b35 = 0.0405
    b36 = 26.74

    B0 = (b00-b01/(b02*y+x-0.13))*(b03-y)*np.exp(-b04*y)-b05*x*y+b06
    B1 = b10*(x+b11)+b12*y
    B2 = (b20*(1-b26*x)*np.exp(-b21*y*(1-x*b24))+b22-b25*y)*(2+x*b23)
    B3 = ((-b30+b34*x)*np.exp(-b31*y/(1+b35*x+b36*y))+b32)*(2+x*b33)

    return(B0*(np.exp(-B1*r))+B2*np.exp(-B3*r))

# Vertical

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

# Rescaled distance
def rscaled(r, p, Hveg, y):
    """rscaled rescales the radius based on below parameters

    Parameters
    ----------
    r : float
        radius from sensor (m)
    p : float
        pressure at site (mb)
    Hveg : float
        height of vegetation during calibration period (m)
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    """
    Fp = 0.4922/(0.86-np.exp(-p/1013.25))
    Fveg = 1-0.17*(1-np.exp(-0.41*Hveg))*(1+np.exp(-9.25*y))
    return(r / Fp / Fveg)

def calculate_watervapor(tair, rh):
    # Constants
    R = 8.31432  # Universal gas constant (J mol-1 K-1)
    Mvap = 18.01528  # Molar mass water vapor (g mol-1)
    Rvap = R / (Mvap * 1E-3)  # Gas constant for water vapor (J K-1 kg-1)

    # Converting some parameters to SI units
    T0 = tair + 273.15  # Kelvin
    RH = rh / 100  # Fraction

    # (1) Calculate saturated vapor pressure at surface es0 based on
    #     formulae used by NOAA/NWS
    es0 = 6.112 * np.exp((17.67 * (T0 - 273.15)) / (243.5 + (T0 - 273.15)))  # hPa
    es0 = es0 * 100  # Pa

    # (2) Calculate actual vapor pressure at surface using relative humidity
    e0 = RH * es0  # Pa

    # (3) Absolute humidity
    rho0 = e0 / (Rvap * T0)


    # Output variables
    rhov = rho0  # Absolute humidity, in kg/m3
    return rhov

# Example usage
tair = np.array([25])  # Air temperature in Celsius
press = np.array([1013])  # Pressure in hPa
rh = np.array([50])  # Relative humidity in percentage
gama = np.array([0.01])  # Some site-specific parameter

rhov = calculate_watervapor(tair, rh)
##############################################################################
   
# Ensure the length of the dictionary matches the length of the DataFrame
if len(sheets) != len(sitespec_filt):
    raise ValueError("The dictionary and DataFrame must have the same number of elements")

# define the desired accuracy as less than 1 percent error:
defineaccuracy = 0.01

list_of_dict = []

for (sheet_name, theta_cal),(index, row) in zip(sheets.items(), sitespec_filt.iterrows()):
    
    # assign  variables from the site specific dataframe
    bda = row['Bulk Density'] # site specific bulk density (g/cm^3)
    p = row['Pressure (mb)'] # site specific pressure (mb)
    tair = row['AirTemp (C)'] # air temperature in Deg. C
    rh = row['PercentRelativeHumidity'] # relative humidity in %
    rhov = calculate_watervapor(tair, rh) # (rhov is an output  in kg/m^3 from the function in Trenton's script, "calculate_watervapor")
    x = rhov*1000 # site specific absolute humidity from 0.1 to 50 in g/m^3
    Hveg = 0 # site specific vegetation height in m

    #rename columns to valid python identifiers
    theta_cal.rename(columns={'Distance (m)': 'Distance_m', 'Bearing (deg)':'Bearing_deg',
                             'Depth (cm)': 'Depth_cm', 'GWC (g/g)': 'GWC_g_g'}, inplace=True)
    # convert gravimetric water content values to total soil water content in m^3/m^3

    theta_cal['VWC_m3_m3']= theta_cal['GWC_g_g']*bda

    #filter out nan values 
    columns_to_check = ['GWC_g_g', "Distance_m","Bearing_deg", "Depth_cm"]
    theta_cal = theta_cal.dropna(subset = columns_to_check)

    # if values in 'VWC (m^3/m^3)' are greater than 0.5, replace them with 0.5
    # replace values < 0.02 with 0.02
    # weighting function is for VWC on the range from 0.02 to 0.5

    theta_cal.loc[theta_cal['VWC_m3_m3'] > 0.5, 'VWC_m3_m3'] = 0.5
    theta_cal.loc[theta_cal['VWC_m3_m3'] < 0.02, 'VWC_m3_m3'] = 0.02
    # find arithmetic average of theta to serve as initial estimate

    CalibTheta = theta_cal['VWC_m3_m3'].mean() 

    # Initialize Accuracy
    Accuracy = 1

    while Accuracy > defineaccuracy:
        # Initial Theta
        theta_init = CalibTheta

        # calculate scaled r for each sample
        theta_cal.loc[:,'rscaled']= theta_cal.apply(lambda row: rscaled(
            row['Distance_m'], p, Hveg, theta_init), axis=1)

        # find weights for each layer in profile. Penetration depth, Dp = D86 is calculated within the function, Wd
        theta_cal.loc[:,'WdL'] = theta_cal.apply(lambda row: Wd(row['Depth_cm'], row['rscaled'], bda, theta_init), axis = 1)

        # Calculate the depth-weighted average for each profile  

        theta_cal.loc[:,'thetweight'] = theta_cal['VWC_m3_m3'] * theta_cal['WdL']

        # Create profile identifier from Bearing and Distance
        theta_cal.loc[:,'profile_Dist_Bear']  = theta_cal.apply(lambda row: f"{row['Distance_m']} - {row['Bearing_deg']}", axis=1)

        # Create a table with the weighted average of each profile
        depthdf = theta_cal.groupby('profile_Dist_Bear', as_index=False)[
            'thetweight'].sum()
       
        temp = theta_cal.groupby('profile_Dist_Bear', as_index=False)['WdL'].sum()
        depthdf['Wd_tot'] = temp['WdL']
        depthdf['Profile_SWV_AVG'] = depthdf['thetweight'] / \
            depthdf['Wd_tot']
            
        # Average all the profiles together with arithmetic mean
        CalibTheta = depthdf['Profile_SWV_AVG'].mean()
        
        '''
        # skip horizontal weighting
        # add the correspoding radius and scaled radius to each profile in the table 
        dictprof = dict(zip(theta_cal.profile_Dist_Bear, theta_cal.Distance_m))
        dictprof2 = dict(zip(theta_cal.profile_Dist_Bear, theta_cal.rscaled))
        depthdf.loc[:,'Radius'] = depthdf['profile_Dist_Bear'].map(dictprof)
        depthdf.loc[:,'rscale'] = depthdf['profile_Dist_Bear'].map(dictprof2)  

        # Find the horizontal weights for each profile
        # Need to add value to each row for .loc application

        depthdf['Wr'] = 0.0  # set up column
        # Below three lines applies WrN function based on radius of the measurement
        depthdf.loc[depthdf['Radius'] > 50, 'Wr'] = WrB(
            depthdf.rscale, x, theta_init)
        depthdf.loc[(depthdf['Radius'] > 5) & (depthdf['Radius'] <= 50), 'Wr'] = WrA(
            depthdf.rscale, x, theta_init)
        depthdf.loc[depthdf['Radius'] <= 5, 'Wr'] = WrX(
            depthdf.rscale, x, theta_init)

        depthdf.loc[:,'RadWeight'] = depthdf['Profile_SWV_AVG'] * depthdf['Wr']

        FinalTheta = depthdf.sum()
        
        CalibTheta = FinalTheta['RadWeight'] / FinalTheta['Wr']
        '''
        
        # Compare CalibTheta to theta_init using percent error
        Accuracy = abs((CalibTheta - theta_init) /
                   theta_init)
        #print("Current Accuracy:", Accuracy)
    
    # Find sd and se of the weighted profiles
    
    N = len(depthdf)
    Obs = depthdf['Profile_SWV_AVG']
    #Weights = depthdf['Wr'] # only if horizontal weighting
    
    #numerator = sum(Weights*((Obs-CalibTheta)**2)) #only if horizontal weighting
    numerator = sum((Obs-CalibTheta)**2)
    #denominator = (N-1)*sum(Weights) # only if horizontal weighting
    denominator = N-1
    sd_w = math.sqrt((numerator/denominator)/N)
    se_w = sd_w/math.sqrt(N)
    
    #    loop exited, print final accuracy and calibration theta value
    print(sheet_name)
    print(" Accuracy is now ", Accuracy, "and CalibTheta is now ", CalibTheta, " m^3/m^3.")
    print("Initial average theta was", theta_cal['VWC_m3_m3'].mean(), "m^3/m^3")
    data= {'Site':sheet_name, 'Arithmetic VWC': theta_cal['VWC_m3_m3'].mean(), 
           'Weighted VWC': CalibTheta, 'SD VWC': sd_w, 'SE VWC': se_w, 'Accuracy':Accuracy, 
           'Arithmetic GWC': theta_cal['GWC_g_g'].mean(),
           'SD arithmetic GWC': theta_cal['GWC_g_g'].std(),
           'SE arithmetic GWC': theta_cal['GWC_g_g'].std()/math.sqrt(len(theta_cal)),
           'Weighted GWC': CalibTheta/bda, 'SD GWC': sd_w/bda, 'SE GWC': se_w/bda}
    print(data)
    list_of_dict.append(data)

# The CalibTheta should replace the 0.1455 value in swc1=0.1455/bda
combined_df = pd.DataFrame(list_of_dict)

stamp = datetime.today().strftime('%Y-%m-%d')

output_path_today = f'C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Data\\WeightedVWC{stamp}.xlsx'

combined_df.to_excel(output_path_today, index=False)

### EXPERIMENT OUTSIDE LOOP

