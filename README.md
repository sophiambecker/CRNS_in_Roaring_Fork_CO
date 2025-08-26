# CRNS_in_Roaring_Fork_CO
 Python code accompanying the paper, "Network wide Assessment of Soil Moisture Calibration and Biomass Monitoring Using Cosmic ray Neutron Sensing in the Roaring Fork River Basin, Colorado"

## Contents
The repository consists of mock data, data processing and analysis scripts, some helper functions, and mock output data. Two different python environments were used to run the python code, which are given in the requirements1.txt and requirements2.txt files. 

## Files using requirements1.txt:
* BareCountsProcessing.py
  + Process neutron data from the bare detector. 
* ModCountsProcessing_Des_McJannet.py
  + Process neutron data from moderated detector using the Desilets et al. (2010) method 
* ModCountsProcessing_UTS_McJannet.py
  + Process neutron data from moderated detector using the universal transport simulation (Köhli et al., 2021) method. 
* CombineDataWithFunction.py
  + Combine corrected neutron data, soil moisture estimates, etc. into a single file. Use output from the processing files. 
* CalibrationAnalysisWithKGE.py
  + Find the calibration parameters used in the soil moisture equations and evaluate the Desilets and UTS methods using Kling-Gupta Efficiency. Use the output from 'FilteredSnowFreeDataForEachSite.py
## Files using requirements2.txt:
* FilterdSnowFreeDataForEachSite.py
  + Filter out data from the winter period before doing the calibration analysis. Use the output from 'CombineDataWithFuction.py'.
* Calibration_Plots.py
  + Plot the results of the calibration analysis. Use the output from 'CalibrationAnalysisWithKGE'.
  
## Notes
* 
