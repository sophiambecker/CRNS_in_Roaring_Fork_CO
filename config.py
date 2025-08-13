"""Config class for CRNS API."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import dotenv_values

environment_variables: dict[str, str] = {
    **os.environ,
    **dotenv_values(".env"),
    **dotenv_values(".env.local"),
}


@dataclass
class Config:
    """Values for both configs."""

    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Atmosphere
    trefC: float = 15  # Reference air temperature (Deg. C)
    trefK: float = 273.15 + trefC  # Reference air temperature (Kelvin)
    rhref: float = 0  # Reference air relative humidity (%)
    P0: float = (
        1013.25  # Sea level Pressure (hPa), International Standard Atmosphere, and US Standard Atmosphere, 1976
    )
    L: float = -0.0065  # ISA USSA, lapse rate (Deg. K/m)
    alpha: float = (
        5.25588  # Coefficient from International Standard Atmosphere, and US Standard Atmosphere, 1976
    )
    r0: float = 6356766  # Earth radius (m)
    gama: float = (
        6.7  # Temperature lapse-rate (monthly climatology): Source Zweck et al. 2011
    )
    lambda_: float = 130  # Attenuation rate for Nebraska
    csf: float = 1.0

    mr: float = -4.9506
    N0r: float = 518.34
    crop: str = "mixed"
    window_length: int = 11

    eta: float = 0.01
    BWEi: float = 0

    @classmethod
    def mysql_host(cls) -> str:
        """MySQL host."""
        value = environment_variables.get("MYSQL_HOST")
        if value is None:
            raise ValueError("MYSQL_HOST not set")
        return value

    @classmethod
    def mysql_user(cls) -> str:
        """MySQL user."""
        value = environment_variables.get("MYSQL_USER")
        if value is None:
            raise ValueError("MYSQL_USER not set")
        return value

    @classmethod
    def mysql_password(cls) -> str:
        """MySQL password."""
        value = environment_variables.get("MYSQL_PASSWORD")
        if value is None:
            raise ValueError("MYSQL_PASSWORD not set")
        return value

    @classmethod
    def mysql_database(cls) -> str:
        """MySQL database."""
        value = environment_variables.get("MYSQL_DATABASE")
        if value is None:
            raise ValueError("MYSQL_DATABASE not set")
        return value
