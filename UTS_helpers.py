# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 11:13:36 2025

@author: sbecker14
"""

# function from Daniel Power and Martin Schrön's neptoon project
# Power, D., Erxleben, F., Zacharias, S., Rosolem, R., & Schrön, M. (2025). neptoon (v0.8.2). Helmholtz Zentrum für Umweltforschung. https://doi.org/10.5281/zenodo.15181751
# https://codebase.helmholtz.cloud/cosmos/neptoon

import numpy as np

def convert_soil_moisture_to_neutrons_uts(
    sm, h, n0, off=0.0, bd=1, method="Mar21_uranos_drf", bio=0
):
    """
    Example
    -------
    convert_soil_moisture_to_neutrons_uts(sm=0.0316, n0=3000, h=5)

    """

    # total sm
    smt = sm + off
    smt *= 1.43 / bd
    if smt == 0.0:
        smt = 0.001
    p = []

    if method == "Jan23_uranos":
        p = [
            4.2580,
            0.0212,
            0.206,
            1.776,
            0.241,
            -0.00058,
            -0.02800,
            0.0003200,
            -0.0000000180,
        ]
    elif method == "Jan23_mcnpfull":
        p = [
            7.0000,
            0.0250,
            0.233,
            4.325,
            0.156,
            -0.00066,
            -0.01200,
            0.0004100,
            -0.0000000410,
        ]
    elif method == "Mar12_atmprof":
        p = [
            4.4775,
            0.0230,
            0.217,
            1.540,
            0.213,
            -0.00022,
            -0.03800,
            0.0003100,
            -0.0000000003,
        ]

    elif method == "Mar21_mcnp_drf":
        p = [
            1.0940,
            0.0280,
            0.254,
            3.537,
            0.139,
            -0.00140,
            -0.00880,
            0.0001150,
            0.0000000000,
        ]
    elif method == "Mar21_mcnp_ewin":
        p = [
            1.2650,
            0.0259,
            0.135,
            1.237,
            0.063,
            -0.00021,
            -0.01170,
            0.0001200,
            0.0000000000,
        ]
    elif method == "Mar21_uranos_drf":
        p = [
            1.0240,
            0.0226,
            0.207,
            1.625,
            0.235,
            -0.00290,
            -0.00930,
            0.0000740,
            0.0000000000,
        ]
    elif method == "Mar21_uranos_ewin":
        p = [
            1.2230,
            0.0185,
            0.142,
            2.568,
            0.155,
            -0.00047,
            -0.01190,
            0.0000920,
            0.0000000000,
        ]

    elif method == "Mar22_mcnp_drf_Jan":
        p = [
            1.0820,
            0.0250,
            0.235,
            4.360,
            0.156,
            -0.00071,
            -0.00610,
            0.0000500,
            0.0000000000,
        ]
    elif method == "Mar22_mcnp_ewin_gd":
        p = [
            1.1630,
            0.0244,
            0.182,
            4.358,
            0.118,
            -0.00046,
            -0.00747,
            0.0000580,
            0.0000000000,
        ]
    elif method == "Mar22_uranos_drf_gd":
        p = [
            1.1180,
            0.0221,
            0.173,
            2.300,
            0.184,
            -0.00064,
            -0.01000,
            0.0000810,
            0.0000000000,
        ]
    elif method == "Mar22_uranos_ewin_chi2":
        p = [
            1.0220,
            0.0218,
            0.199,
            1.647,
            0.243,
            -0.00029,
            -0.00960,
            0.0000780,
            0.0000000000,
        ]
    elif method == "Mar22_uranos_drf_h200m":
        p = [
            1.0210,
            0.0222,
            0.203,
            1.600,
            0.244,
            -0.00061,
            -0.00930,
            0.0000740,
            0.0000000000,
        ]

    elif method == "Aug08_mcnp_drf":
        p = [
            1.110773444917129,
            0.034319446894963,
            0.180046592985848,
            1.211393214064259,
            0.093433803170610,
            -1.877788035e-005,
            -0.00698637546803,
            5.0316941885e-005,
            0.0000000000,
        ]
    elif method == "Aug08_mcnp_ewin":
        p = [
            1.271225645585415,
            0.024790265564895,
            0.107603498535911,
            1.243101823658557,
            0.057146624195463,
            -1.93729201894976,
            -0.00866217333051,
            6.198559205414182,
            0.0000000000,
        ]
    elif method == "Aug12_uranos_drf":
        p = [
            1.042588152355816,
            0.024362250648228,
            0.222359434641456,
            1.791314246517330,
            0.197766380530824,
            -0.00053814104957,
            -0.00820189794785,
            6.6412111902e-005,
            0.0000000000,
        ]
    elif method == "Aug12_uranos_ewin":
        p = [
            1.209060105287452,
            0.021546879683024,
            0.129925023764294,
            1.872444149093526,
            0.128883139550384,
            -0.00047134595878,
            -0.01080226893400,
            8.8939419535e-005,
            0.0000000000,
        ]
    elif method == "Aug13_uranos_atmprof":
        p = [
            1.044276170094123,
            0.024099232055379,
            0.227317847739138,
            1.782905159416135,
            0.198949609723093,
            -0.00059182327737,
            -0.00897372356601,
            7.3282344356e-005,
            0.0000000000,
        ]
    elif method == "Aug13_uranos_atmprof2":
        p = [
            4.31237,
            0.020765,
            0.21020,
            1.87120,
            0.16341,
            -0.00052,
            -0.00225,
            0.000308,
            -1.9639e-8,
        ]

    N = (p[1] + p[2] * smt) / (smt + p[1]) * (
        p[0] + p[6] * h + p[7] * h**2 + p[8] * h**3 / smt
    ) + np.exp(-p[3] * smt) * (p[4] + p[5] * (h + bio / 5 * 1000))

    return N * n0

def convert_neutrons_to_soil_moisture_uts(
    neutron_count: float,
    n0: float,
    air_humidity: float,
    bulk_density: float = 1.0,
    lattice_water: float = 0.0,
    water_equiv_soil_organic_carbon: float = 0.0,
    method: str = "Mar21_uranos_drf",
):
    """
    Converts corrected neutrons counts into volumetric soil moisture
    following the method outline in Köhli paper

    https://doi.org/10.3389/frwa.2020.544847

    Example
    -------
    convert_neutrons_to_soil_moisture_uts(
        neutron_count=2000, n0=3000, air_humidity=5
    )


    """
    if np.isnan(neutron_count):
        return np.nan

    # neutron_count = neutron_count/n0

    t0 = 0.0
    t1 = 4.0
    n_i0 = convert_soil_moisture_to_neutrons_uts(
        0.0,
        n0=n0,
        h=air_humidity,
        method=method,
        off=lattice_water + water_equiv_soil_organic_carbon,
        bd=bulk_density,
    )
    n_i1 = convert_soil_moisture_to_neutrons_uts(
        1.0,
        h=air_humidity,
        n0=n0,
        method=method,
        off=lattice_water + water_equiv_soil_organic_carbon,
        bd=bulk_density,
    )
    while t1 - t0 > 0.0001:
        t2 = 0.5 * (t0 + t1)
        n2 = convert_soil_moisture_to_neutrons_uts(
            t2,
            h=air_humidity,
            n0=n0,
            method=method,
            off=lattice_water + water_equiv_soil_organic_carbon,
            bd=bulk_density,
        )
    
        if neutron_count < n2:
            t0 = t2
            n_i0 = n2 
        else:
            t1 = t2
            n_i1 = n2
    t2 = 0.5 * (t0 + t1)

    # if t2 <= 0.0001: t2 = np.nan

    return t2