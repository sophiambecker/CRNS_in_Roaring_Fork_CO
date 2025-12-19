# CRNS_in_Roaring_Fork_CO

Python analysis code for the Roaring Fork River Basin cosmic-ray neutron sensing (CRNS) calibration, validation, and soil moisture prediction work.

## Raw Data Source
- (fill in with the location/description of the raw CRNS, meteorology, and TDR data)

## Environments
- `requirements1.txt` (core calibration + processing): pandas/numpy/scipy/matplotlib stack used for count processing, calibration fitting, and most plotting.
- `requirements2.txt` (QA/QC + geospatial/ML extras): includes scikit-learn, geopandas, rasterio, folium/plotly, and widgets; needed for scripts that use LocalOutlierFactor, geospatial layers, or interactive plotting.

## Paths and data expectations
- Scripts assume Windows paths rooted at `C:\Users\sbecker14\Documents\GitHub\CRNS_in_Roaring_Fork_CO` and create dated output folders; update the hard-coded `os.chdir`, `Dir`, and input folder strings if you run elsewhere.
- Common inputs live under `Data/` (e.g., `Data_Relase_2024_b`, `Mock Calibration Summary_*.xlsx`, `DailyPRISM_Ppt.csv`, portable calibration CSVs in `Data/MockPortableData/`, and metadata like `RoaringFork_CRNS_metadata.csv`).
- Outputs are written beside the inputs in time-stamped folders such as `ModCountsProcessing_*`, `CombineDataWithFunction*_output*`, `FilteredSnowFreeData_output*`, and `Calibration_AnalysisWithKGE_output_*`.

## Reproducing the daily workflow (most figures/tables)
1) Process moderated (and optional bare) counts — requirements1
   - `ModCountsProcessing_Des_McJannet20251120.py` and `ModCountsProcessing_UTS_McJannet20251120.py`: apply atmospheric, solar, and biomass corrections; derive half-hour moderated counts and N0 per site for the Desilets and UTS formulations. Inputs: raw `Data_Relase_2024_b/*v1.csv` plus site variables; outputs: `Data/ModCountsProcessing_{Des,UTS}_output<date>/Mod_<Site>.csv`.
   - `BareCountsProcessing.py`: applies the same corrections to bare detectors (optional for later snow checks). Output: `mock_Bare_output<date>/Bare_<Site>.csv`.
2) Merge counts with TDR and precip — requirements1
   - `CombineDataWithFunction.py`: builds daily-averaged site data frames with corrected CRNS soil moisture (Des & UTS), raw moderated counts, depth-weighted TDR, soil temperatures, and PRISM precipitation. Output: `CombineDataWithFunction_output<date>/<site>_CRNS_Site_Data.csv`.
3) QA/QC and snow filtering — requirements2
   - `FilteredSnowFreeDataForEachSite.py`: removes snow-affected periods, outliers (LocalOutlierFactor), and implausible moderated counts; joins daily bare counts. Output: `FilteredSnowFreeData_output<date>/<old>_<new>_FilteredAndSnowFree_<date>.csv`.
4) Calibrate and validate N0 — requirements1
   - `Calibration_AnalysisWithKGE_LOOCV.py`: leave-one-out cross-validation on gravimetric samples; fits N0 (Desilets and UTS), reports KGE/r/RMSE, flags outliers, and saves `Site_Calibration_data_summary.csv`, `Parameter_fit.csv`, `LOOCV_stats.csv`, and per-site predictions.
   - `Calibration_AnalysisWithCorrelation.py`: alternative single-N0 fits with correlation-focused diagnostics and bootstrap CIs; writes residual plots and CSV summaries.
5) Plot calibration results — requirements2
   - `Calibration_Plots2.py`: publication-ready plots and uncertainty propagation using the `Calibration_AnalysisWithKGE_output_*` products.

## Reproducing the half-hour workflow
1) Build half-hour merged data — requirements1
   - `CombineDataWithFunction_halfhour.py`: same as the daily combiner but keeps half-hour resolution; output `CombineDataWithFunction_half_hr_output<date>/<old>_<new>_CRNS_Site_Data_half_hr.csv`.
   - `_no_fs` variant uses alternative processed counts (`ModCountsProcessing_*_no_fs`) and writes to `CombineDataWithFunction_half_hr_output_no_fs_<date>/`.
2) QA/QC and snow filtering — requirements2
   - `FilteredSnowFreeDataForEachSite_half_hour.py`: half-hour version of the snow/outlier filter; outputs `FilteredAndSnowFree_half_hr_*` files.
3) Half-hour N0 fitting and evaluation — requirements1
   - `Half_hour_LOOCV.py`, `Half_hour_LOOCV_20251212.py`, `Half_hour_LOOCV_tweaks.py`, `Half_hour_LOOCV_no_fs.py`: LOOCV and sensitivity analyses on the half-hour data with assorted tweaks/datasets; produce `Calibration_AnalysisWithKGE_output_*` folders.
4) Predict soil moisture and analyze dryness — requirements2
   - `SWC_predictions_for_ea_sitee.py`: generates site-specific half-hour soil moisture predictions (Des & UTS), applies outlier filtering, and saves to `Site_specific_predictions_output_<date>/` along with fit statistics.
   - `AnalyzeDryPeriods.py`: identifies multi-day drought periods using the site-specific prediction outputs and exports period tables and plots to `DryPeriods/output<date>/`.

## Additional analyses and figures
- `PaperTimeSeries_SWC_predictions_20251104.py`, `PaperTimeSeries_N0_and_Veg_1year_20251104.py`: build publication-ready time series for soil moisture, N0, vegetation indices, and drivers.
- `Bare_GDD_GSI_VPD_N0_2024only_20251121.py`: derives growing degree days, greenness indices, VPD, and N0 trajectories for 2024.
- `sample_weighting_loop_with_SD_SE_20251031.py`: helper for depth-weighting profiles with uncertainty (used by the combiner scripts).
- `N0_probe_analysis.py`: computes probe-based N0/ND time series per site with LocalOutlierFactor filtering and saves `*Probe_Based_Parameters.csv` (requirements2).
- `N0_analyses.py` and `N0_analyses_no_fs.py`: seasonal statistics and regressions of probe-derived N0/ND versus BWE and elevation; outputs seasonal summaries and plots.
- `N0_vs_SiteSpecificVariables_20251123.py` and `_no_fs.py`: regress calibrated N0 values against site traits (soil, elevation, cutoff rigidity) and generate summary tables/plots.
- `SWC_predictions_for_ea_sitee.py` (see above) plus `Site_specific_predictions_output_*` are reused by `AnalyzeDryPeriods` and paper figures.
- `PaperTimeSeries_*`, `Calibration_Plots2.py`, and `AnalyzeDryPeriods.py` use matplotlib/seaborn color styling included in requirements2.

## Utilities and shared code
- `helpers.py`: shared corrections (pressure/solar queries, data cleaning, half-hour rounding) used by the processing scripts.
- `UTS_helpers.py`: UTS soil-moisture–neutron forward and inverse functions.
- `watervapor.py`: atmospheric water vapor correction utilities.
- `config.py`: physical constants and defaults (reference T/P, lapse rate, attenuation constants, etc.).

## Running tips
- Activate the correct environment before each stage (`pip install -r requirements1.txt` or `requirements2.txt`).
- Ensure the expected Excel/CSV inputs exist in `Data/` and that output folders referenced in the scripts match your run date or are updated accordingly.
- Most scripts are written to be executed directly (no arguments) from the repository root; adjust hard-coded paths if your data live elsewhere.
