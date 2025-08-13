"""Calculate surface absolute humidity."""
from __future__ import annotations

import numpy as np


def calculate_watervapor(
    tair: np.ndarray, rh: np.ndarray, gama: float
) -> tuple[float, float]:
    """
    Calculate surface absolute humidity.

    This function calculates surface absolute humidity and water vapor integrated at
    pre-defined elevation

    Function by R. Rosolem (2013 J. Hydromet), January 2013, updated June
    2014 by T. Franz

    Input: air temperature (Deg. C), air pressure (mb, hpa), relative
    humidity (%), Temperature lapse-rate (monthly climatology): Source Zweck et al. 2011, use gama=6.7;

    output rhov, [kg/m3], iwv [mm of water]

    For rhovref I use 0, which is a dry atmosphere.

    Comments for use with COSMOS data
    1. The correction factor is: CWV=1.0+0.0054*(rhov-rhovref)*1000;  %%%%correction in g/m3!!!!
    2. Download COSMOS Level 2 data. The Corr data column is corrected counts for scaling and pressure.
    3. Multiply the corresponding CWV values by Corr.
    4. Use the existing calibration dataset on the COSMOS data portal to
    recompute N0 following Desilets 2010 and modified by Bogena 2013. The
    calibration dataset contains the average theta and average soil bulk
    density from the 108 samples averaged over 30 cm.
    theta=(0.0808./(N./N0-.372)-0.115)./bda-tao*bda, where theta is the volumetric water content (cm3/cm3), N is the CWV
    corrected count rate (cph), bda is the site soil bulk density (g/cm3), and tao is the sum of gravimetric soil lattice water and soil organic carbon (g/g). Most COSMOS sites contain estimates LW and SOC.
    5. With the new N0 and CWV corrected counts recompute theta following Bogena 2013.
    """
    # Constants
    R = 8.31432  # Universal gas constant (J mol-1 K-1)
    Mvap = 18.01528  # Molar mass water vapor (g mol-1)
    Rvap = R / (Mvap * 1e-3)  # Gas constant for water vapor (J K-1 kg-1)
    z = 10000  # Top layer for PW calculation (m)

    # Converting some parameters to SI units
    T0 = tair + 273.15  # Kelvin
    RH = rh / 100  # Fraction

    # (1) Calculate saturated vapor pressure at surface es0 based on
    # formulae used by NOAA/NWS
    es0 = 6.112 * np.exp((17.67 * (T0 - 273.15)) / (243.5 + (T0 - 273.15)))  # hPa
    es0 *= 100  # Pa

    # (2) Calculate actual vapor pressure at surface using relative humidity
    e0 = RH * es0  # Pa

    # (3) Absolute humidity
    rho0 = e0 / (Rvap * T0)

    # (4) Calculate water vapor scale height based on Tomasi 1981
    # These are site specific parameters
    apar = 5.1087
    bpar = -1
    cpar = 0.005
    H = 1.0 / ((apar * 1000 * gama / (T0) ** 2) + bpar * (gama / T0) + cpar)

    # Calculated Integrated Water vapor from surface to 335 meters
    # zref = 335;
    zref = 335
    Wcalc = rho0 * (H * 1000) * (1 - np.exp(-(zref) / (H * 1000)))

    # Output variables
    rhov = rho0  # Absolute humidity, in kg/m3
    iwv = Wcalc  # Integrated water vapor up to 335 m above surface, in mm
    return rhov, iwv
