# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:30:37 2025

@author: sbecker14
"""

import datetime as dt
import re
import warnings
from operator import itemgetter

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser
from urllib3.exceptions import InsecureRequestWarning
from typing import Union


# Functions used in the various processing scripts for raw neutron data

def _long360(long: float) -> float:
    """Return the longitude in the range [0, 360)."""
    return long + 360 if long < 0 else long


def get_correction_parameters(
    pref: float, lat: float, long: float, year: int
) -> tuple[float, float, float]:
    """Get the correction parameters from the CRNSLab website."""
    data = {
        "p": f"{pref:.4f}",
        "lat": f"{lat:.4f}",
        "lon": f"{_long360(long):.4f}",
        "year": f"{year:d}",
    }
    url = "https://crnslab.org/util/intensity.php"
    warnings.filterwarnings("ignore", category=InsecureRequestWarning, module="urllib3")
    resp = requests.post(url, data=data, verify=False)
    resp.raise_for_status()
    html_content = resp.text

    # Extract Fscal
    fscal_pattern = r"f</i><sub>lat</sub> \* <i>f</i><sub>bar</sub> &nbsp=&nbsp ([\d.]+)<br><br>Intensity relative"
    fscal_match = re.search(fscal_pattern, html_content)
    fscal = float(fscal_match.group(1)) if fscal_match else None

    # Extract Rc
    rc_pattern = r"R</i><sub>c</sub>&nbsp=&nbsp([\d.]+)&nbsp GV<br><i>g</i>"
    rc_match = re.search(rc_pattern, html_content)
    rc = float(rc_match.group(1)) if rc_match else None

    # Extract Lpt
    lpt_pattern = r"</sup><br><i>L</i><sub>pt</sub>&nbsp=&nbsp([\d.]+)&nbsp g cm<sup>-2</sup><br><i>f</i><sub>lat</sub>"
    lpt_match = re.search(lpt_pattern, html_content)
    lpt = float(lpt_match.group(1)) if lpt_match else None
    return fscal, rc, lpt


def get_fi_over_time(pref: float, lat: float, long: float) -> pd.DataFrame:
    """Get the solar flux over time from the CRNSLab website."""
    data = {
        "p": f"{pref:.4f}",
        "lat": f"{lat:.4f}",
        "lon": f"{_long360(long):.4f}",
    }
    url = "https://crnslab.org/util/computeSolar.php"
    warnings.filterwarnings("ignore", category=InsecureRequestWarning, module="urllib3")
    resp = requests.post(url, data=data, verify=False)
    resp.raise_for_status()
    html_content = resp.text
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("body").text
    rows = [row.split(",") for row in table.split("\n") if len(row.split(",")) == 2]
    date = list(map(itemgetter(0), rows))
    fsol = list(map(float, map(itemgetter(1), rows)))
    df = pd.DataFrame({"date": pd.to_datetime(date), "fsol": fsol})
    return df

def remove_utc_from_string_date(date_str):
    """Removes 'UTC' from a date string and converts it to a datetime object."""

    # Parse the date string, assuming it's in UTC
    date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S UTC')

    # Convert the datetime object to your desired timezone (or keep it in UTC)
    # Here, we're converting to local timezone
    localized_date = date_obj.replace(tzinfo=dt.timezone.utc).astimezone(None)

    # Format the datetime object back into a string without the timezone information
    formatted_date = localized_date.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_date

def datenum(date: Union[str, dt.date]) -> float:
    """Get equivalent of matlab datenum."""
    if isinstance(date, str):
        date: dt.datetime = parser.parse(date)
    return (
        366
        + date.toordinal()
        + (date - dt.datetime.fromordinal(date.toordinal())).total_seconds()
        / (24 * 60 * 60)
    )

def clean_data_local(df: pd.DataFrame, site_variables: pd.DataFrame) -> pd.DataFrame:
    """Clean the data returned from nearfld.com."""
    df["tdat"] = df.message_date.map(datenum)  # TODO: might need to round to half hour
    df["lati"] = site_variables['lat']  # site location center
    df["longi"] = site_variables['long']  # site location center
    df['elevation']=site_variables['elev']
    df['lattice_water']= site_variables['lw']
    df['soil_organic_carbon'] = site_variables['soc']
    df['bulk_density']= site_variables['bda']
    df['swc']= site_variables['swc_weighted']
    df['caldate']=site_variables['CalDate']
	
    # TODO: these vary by site
    df.loc[np.logical_or(df.t7_c > 80, df.t7_c < -50), "t7_c"] = np.nan
    df.loc[np.logical_or(df.h7_pct > 120, df.h7_pct < 0), "h7_pct"] = np.nan # change upper bound from 109 to 120
    df.loc[np.logical_or(df.n1_persamp > 25000, df.n1_persamp < 100), "n1_persamp"] = (
        np.nan
    )
    df.loc[np.logical_or(df.n2_persamp > 25000, df.n2_persamp < 100), "n2_persamp"] = (
        np.nan
    )
    df.loc[np.logical_or(df.n2_persamp > 25000, df.n2_persamp < 100), "n3_persamp"] = (
        np.nan
    )
    # Note: need to update this normalization if we change the record period
    df["n1_persamp_cph"] = df.n1_persamp / df.e1 * 3600
    df["n2_persamp_cph"] = df.n2_persamp / df.e1 * 3600
    df["n3_persamp_cph"] = df.n3_persamp / df.e1 * 3600
    return df

def half_hour_string(timestamp: pd.Timestamp) -> str:
    """Convert datetime string to its even half hour."""  
    d = timestamp.to_pydatetime()
    minute = (d.minute // 30) * 30
    d = d.replace(minute=minute, second=0, microsecond=0)
    return d.strftime("%Y-%m-%d %H:%M:%S")