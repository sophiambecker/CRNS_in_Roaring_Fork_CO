# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:14:33 2025

@author: sbecker14

Use output from TDR_TemporalResidualsFromSingleCalibration_20250415.py, which
is saved here: C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\TDR_TemporalResidualsFromSingleCalibration\\output20250417

Also incorporate cumulative growing degree day data which is calculated in the script: 
    CalculateGDDatEachSite_20250417.py
    and saved here: 
        C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Data\\GDD\\output20250417
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats

#from sklearn.linear_model import LinearRegression
from dominance_analysis import Dominance # for relative importance of MLR variables

# set up output directory

stamp = datetime.now().strftime("%Y%m%d")
inDir = 'C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis'
outDir = os.path.normpath(inDir + os.sep + 'BARE_GDD_MLR\\2024_only\\output'+stamp) + '\\'   # Set output directory
if not os.path.exists(outDir): os.makedirs(outDir) # Create output directory

directory_path = 'C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\PaperTimeseries\\N0_TimeSeries_output20251205'
file_pattern = f'{directory_path}\\*variables.csv'
file_paths = glob.glob(file_pattern, recursive=True)
df_ls = [pd.read_csv(file_path) for file_path in file_paths]
site_names_new = [os.path.basename(fp).split('_')[0] for fp in file_paths]
print(f'new site names are {site_names_new}')


# make dictionary of dataframes using new site names 
df_dict = dict(zip(site_names_new, df_ls))

# load landcover descriptions: 
RF_veginfo = pd.read_csv("C:\\Users\\sbecker14\\Documents\\CRNS_USGS_Analysis\\Data\RoaringFork_CRNS_metadata.csv")
RF_veginfo.replace('RF 5', 'RF5', inplace=True) 
veg_dict = dict(zip(RF_veginfo['Original_ID'], RF_veginfo['Land Cover ']))

# load forest/non-forest classification:
f_clas = pd.read_excel("C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Data\\VisualSiteClassification.xlsx")
f_clas.replace('RF 5', 'RF5', inplace=True)
f_clas_dict = dict(zip(f_clas['Site Original'], f_clas['Imagery']))

# dictionary to get old site name based on new site name: 

new_to_old_name = {'C2': 'Crys 1', 
          'C1': 'Crys 2',
          'C3': 'Crys 3',
          'C4': 'Crys 6',
          'F2': 'Fry 4',
          'F3': 'Fry 5',
          'F4': 'Fry 6',
          'F1': 'RF 1',
          'R1': 'RF 2',
          'R2': 'RF 3',
          'R4': 'RF5',
          'R5': 'RF 6',
          'R3': 'RF 7',
          'R7': 'RF 8',
          'R6': 'RF 9'
          }


# load calibration day data: 

site_var = pd.read_excel("C:\\Users\\sbecker14\\Documents\\GitHub\\CRNS_in_Roaring_Fork_CO\\Data\\Mock Calibration Summary_20251106.xlsx")
# Extract row 16 and columns 1–15
sample_series = site_var.iloc[17, 1:16]

# Convert values to just the date, keep column names (keys) as strings
sample_date = {
    col: pd.to_datetime(val).date()
    for col, val in sample_series.items()
}

def get_canopy_group(site_new):
    """Map new site name to imagery-based canopy group."""
    old = new_to_old_name.get(site_new, '')
    return f_clas_dict.get(old, np.nan)

def round_sigfig(value, sig=2):
    """Round numeric values to a fixed number of significant figures."""
    if pd.isna(value):
        return value
    try:
        return float(f"{value:.{sig}g}")
    except (ValueError, TypeError):
        return value

def place_text_top_empty_corner(ax, x, y, text, margin=0.02):
    """
    Automatically place annotation text in the emptier of:
      - top-left corner
      - top-right corner

    ax      : matplotlib axis
    x, y    : data arrays
    text    : annotation string
    margin  : padding inside axes (0–1, axes fraction)
    """
    
    # Normalize data to axes coordinates
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Consider only *top* region, say top 30% of plot
    top_mask = y_norm > 0.7

    left_count  = np.sum((x_norm < 0.5) & top_mask)
    right_count = np.sum((x_norm >= 0.5) & top_mask)

    # Choose the emptier corner
    if left_count <= right_count:
        xpos = margin
        halign = 'left'
    else:
        xpos = 1 - margin
        halign = 'right'

    ypos = 1 - margin  # always top

    ax.text(
        xpos, ypos, text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment=halign,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

# plotting variables/pairs:
    
x_var = ['Bare_3_norm','Bare_3_norm', 'Bare_3_norm', 'Bare_3_norm', 
         'cumulative_GSI', 'cumulative_GSI', 'cumulative_GDD','cumulative_GDD',
         'VPD_3', 'VPD_3']    
y_var = ['cumulative_GSI', 'cumulative_GDD', 'N0_3_norm', 'ND_3_norm', 
         'N0_3_norm', 'ND_3_norm', 'N0_3_norm', 'ND_3_norm',
         'N0_3_norm', 'ND_3_norm']

x_lab = [r'$\overline{\mathrm{Bare_{3-day}}}$/$\overline{\mathrm{Bare_{long}}}$',
         r'$\overline{\mathrm{Bare_{3-day}}}$/$\overline{\mathrm{Bare_{long}}}$',
         r'$\overline{\mathrm{Bare_{3-day}}}$/$\overline{\mathrm{Bare_{long}}}$',
         r'$\overline{\mathrm{Bare_{3-day}}}$/$\overline{\mathrm{Bare_{long}}}$',
         'Cum. GSI',
         'Cum. GSI',
         'Cum. GDD',
         'Cum. GDD',
         r'$\overline{\mathrm{VPD_{3-day}}}$/$\overline{\mathrm{VPD_{long}}}$',
         r'$\overline{\mathrm{VPD_{3-day}}}$/$\overline{\mathrm{VPD_{long}}}$']
y_lab = ['Cum. GSI', 
         'Cum. GDD',
         r'$\overline{\mathrm{N_{0 [3-day]}}}$/$\overline{\mathrm{N_{0 [long]}}}$',
         r'$\overline{\mathrm{N_{D [3-day]}}}$/$\overline{\mathrm{N_{D [long]}}}$',
         r'$\overline{\mathrm{N_{0 [3-day]}}}$/$\overline{\mathrm{N_{0 [long]}}}$',
         r'$\overline{\mathrm{N_{D [3-day]}}}$/$\overline{\mathrm{N_{D [long]}}}$',
         r'$\overline{\mathrm{N_{0 [3-day]}}}$/$\overline{\mathrm{N_{0 [long]}}}$',
         r'$\overline{\mathrm{N_{D [3-day]}}}$/$\overline{\mathrm{N_{D [long]}}}$',
         r'$\overline{\mathrm{N_{0 [3-day]}}}$/$\overline{\mathrm{N_{0 [long]}}}$',
         r'$\overline{\mathrm{N_{D [3-day]}}}$/$\overline{\mathrm{N_{D [long]}}}$'
         ]
x_file_name = ['Bare', 'Bare', 'Bare', 'Bare', 'GSI', 'GSI', 'GDD', 'GDD', 'VPD', 'VPD']
y_file_name = ['GSI', 'GDD', 'N0', 'ND', 'N0', 'ND', 'N0', 'ND', 'N0', 'ND']

lists = [x_var, y_var, x_lab, y_lab, x_file_name, y_file_name]
list_names = ['x_var', 'y_var', 'x_lab', 'y_lab', 'x_file_name', 'y_file_name']

# Check lengths
lengths = [len(lst) for lst in lists]
print(dict(zip(list_names, lengths)))

# Assert all lengths are equal
if len(set(lengths)) == 1:
    print("All lists are the same length:", lengths[0])
else:
    print("Lists have different lengths!")
    for name, l in zip(list_names, lengths):
        print(f"{name}: {l}")


#THIS_SITE_new = 'C1'

#### OTHER PLOTS, LIKE GDD VS BARE ##############################################
comb_res = []
linear_all = []
for s in site_names_new:
    THIS_SITE_new = None
    THIS_SITE_new = s
    print(f'Starting loop for {THIS_SITE_new}')
    
    THIS_SITE_old = new_to_old_name[THIS_SITE_new]
    
    df = df_dict[THIS_SITE_new]
    
    '''
    mod_mean_UTS = np.nanmean(df['Mod_cph_UTS_f3'])
    mod_rolling_UTS = df['Mod_cph_UTS_f3'].rolling(window = 3).mean()
    df['Mod_norm_UTS'] = mod_rolling_UTS/mod_mean_UTS
    
    
    mod_mean_Des = np.nanmean(df['Mod_cph_Des_f3'])
    mod_rolling_Des = df['Mod_cph_Des_f3'].rolling(window = 3).mean()
    df['Mod_norm_Des'] = mod_rolling_Des/mod_mean_Des
    mod_rolling_Des7 = df['Mod_cph_Des_f3'].rolling(window = 7).mean()
    df['Mod_norm_Des7'] = mod_rolling_Des7/mod_mean_Des
    '''
    df['date'] = pd.to_datetime(df['date'])  # keep as datetime64
    df['doy'] = df['date'].dt.dayofyear    
    
    
    
    cal_date = pd.to_datetime(sample_date[THIS_SITE_old]).tz_localize(None).normalize()
    
    cal_data = df[df['date']==cal_date]
    
    ## plot gdd vs bare ################
    valid1 = df[['cumulative_GDD', 'Bare', 'year', 'VPD_kPa', 'cumulative_GSI','N0', 'ND','Corrected_Mod_cph_for_Des']].dropna()
    
    year_dict = {}
    for year in valid1['year'].unique():
        # Filter data for the specific year, between May 1 and Dec 1
        year_int = int(year)
        start_date = pd.to_datetime(f'{year_int}-04-01')
        if THIS_SITE_new == 'R6':
            end_date = pd.to_datetime(f'{year_int}-11-01') # R6 doesn't start until mid-october
        else: 
            end_date = pd.to_datetime(f'{year_int}-9-15')
        
        
        seasonal_data = df[
            (df['date'] >= start_date) & 
            (df['date'] <= end_date) & 
            (df['year'] == year)
        ].copy()
        
        seasonal_data['N0_3'] = seasonal_data['N0'].rolling(window = 3).mean()
        seasonal_data['ND_3'] = seasonal_data['ND'].rolling(window = 3).mean()
        
        N0_mean = seasonal_data['N0'].mean()
        seasonal_data['N0_3_norm'] = seasonal_data['N0_3']/N0_mean
        
        ND_mean = seasonal_data['ND'].mean()
        seasonal_data['ND_3_norm'] = seasonal_data['ND_3']/ND_mean
        # % deviation from maximum N0
        N0_max = seasonal_data['N0'].max()
        ND_max = seasonal_data['ND'].max()
        seasonal_data['N0_3_dev'] = 100 - (seasonal_data['ND_3']/N0_max)*100
        seasonal_data['ND_3_dev'] = 100 - (seasonal_data['ND_3']/ND_max)*100
        
        seasonal_data['Bare_3'] = seasonal_data['Bare'].rolling(window = 3).mean()
        Bare_mean = seasonal_data['Bare'].mean()
        seasonal_data['Bare_3_norm'] = seasonal_data['Bare_3']/Bare_mean
        
        seasonal_data['Bare_7'] = seasonal_data['Bare'].rolling(window = 7).mean()
        seasonal_data['Bare_7_norm'] = seasonal_data['Bare_7']/Bare_mean
        
        seasonal_data['VPD_3'] = seasonal_data['VPD_kPa'].rolling(window = 3).mean()
        VPD_mean = seasonal_data['VPD_kPa'].mean()
        seasonal_data['VPD_3_norm'] = seasonal_data['VPD_3']/VPD_mean
        seasonal_data['VPD_3_dev'] = 100 - (seasonal_data['VPD_3']/VPD_mean)*100
        
        seasonal_data['Mod_3'] = seasonal_data['Corrected_Mod_cph_for_Des'].rolling(window = 3).mean()
        Mod_mean = seasonal_data['Corrected_Mod_cph_for_Des'].mean()
        seasonal_data['Mod_3_norm'] = seasonal_data['Mod_3']/Mod_mean
        
        seasonal_data['Mod_7'] = seasonal_data['Corrected_Mod_cph_for_Des'].rolling(window = 7).mean()
        seasonal_data['Mod_7_norm'] = seasonal_data['Mod_7']/Mod_mean
        
        year_dict[year_int] = seasonal_data
        
        
    # combine data back together again for plotting:
    valid = pd.concat(year_dict, ignore_index=False)
       
    #% deviation from long term mean
    
    
    # separate into a df for each year:
    #df_23 = year_dict[2023].dropna(subset = ['cumulative_GDD', 'Bare_3_norm', 'VPD_kPa', 'cumulative_GSI','N0_3', 'ND_3'])
    df_24 = year_dict[2024]
    
    #% deviation from long term mean
    
    # check that lists are the same length: 
    linear_res = []    
    for i in range(len(x_var)):
    
        fig, ax1 = plt.subplots(figsize=(7, 6))
          
        sc = ax1.scatter(x=df_24[x_var[i]], y=df_24[y_var[i]], 
                    c=df_24['doy'],          # color by day of year
                    cmap='viridis',           # choose a colormap
                    s=50,                     # marker size
                    edgecolor='k', )
        # Regression
        
        xy = df_24[[x_var[i], y_var[i]]].dropna()
        x_all_24 = xy[x_var[i]]
        y_all_24 = xy[y_var[i]]
        
        if not xy.empty:
            slope_24, intercept_24, r_value_24, p_value_24, std_err_24 = linregress(x_all_24, y_all_24) #two-sided p-value for the null hypothesis that the slope is zero.
            
            # Plot regression line
            x_range_24 = np.linspace(x_all_24.min(), x_all_24.max(), 100)
            y_fit_24 = slope_24 * x_range_24 + intercept_24
            ax1.plot(x_range_24, y_fit_24, color='darkgray', linestyle='--')
            
            # Annotate full equation
            eq_text = (
                f"y = {slope_24:.2g}x + {intercept_24:.2g}\n"
                f"r = {r_value_24:.2g}, R² = {r_value_24**2:.2g}, p = {p_value_24:.3g}"
            )
        
            place_text_top_empty_corner(ax1, x_all_24, y_all_24, eq_text)
            
            linear_fit = {'x': x_var[i], 'y': y_var[i], 'slope': slope_24, 'intercept': intercept_24, 
                          'r': r_value_24, 'Rsq': r_value_24**2 , 'slope p-value': p_value_24}
            linear_res.append(linear_fit)
        else:
            print(f'Empty dataframe at {THIS_SITE_new} for {x_file_name[i]} and {y_file_name[i]}')
            
        
       
        # Labels and title
        ax1.set_xlabel(x_lab[i], fontsize = 14)
        ax1.set_ylabel(y_lab[i], fontsize = 14)
        
        #Add colorbar
        cbar = plt.colorbar(sc, ax=ax1)
        cbar.set_label('Day of Year')
        
        ax1.tick_params(axis='both', which='major', labelsize=12)
        fig.suptitle(f'{THIS_SITE_new}, {veg_dict[THIS_SITE_old]}', fontsize=12)
        
        plt.tight_layout()
        
        plt.savefig(f'{outDir}\\{THIS_SITE_new}_{x_file_name[i]}_{y_file_name[i]}.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')
 
    linear_df = pd.DataFrame(linear_res)
    linear_df.to_csv(f'{outDir}\\{THIS_SITE_new}_linear_fits.csv')
    linear_df['Site'] = THIS_SITE_new
    linear_df['Land cover group'] = get_canopy_group(THIS_SITE_new)
    linear_all.append(linear_df)
        
    # Initialize a list to store model results
    X_ls = [['Bare_7_norm', 'Mod_7_norm'],
            ['Bare_3_norm', 'Mod_3_norm']]
    X_names_ls = [['Bare_7_norm','Mod_7_norm'],
                  ['Bare_3_norm', 'Mod_3_norm']]
    
    y_ls = ['cumulative_GDD', 'cumulative_GSI']
    y_name_ls = ['Cum_GDD', 'Cum_GSI']
    
    model_results = []
    
    # Loop through all combinations of i and j
    for i in range(len(X_ls)):
        for j in range(len(y_ls)):
            df = df_24  # Assuming df_24 is already defined
            y = df[y_ls[j]]
            X = df[X_ls[i]]
            X = sm.add_constant(X)
    
            # Combine X and y into a DataFrame to drop bad rows together
            model_data = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            
            if not model_data.empty:
                
                # Split back into cleaned X and y
                X_clean = model_data[X.columns]
                y_clean = model_data[y.name]
                
                # Now fit the model
                model = sm.OLS(y_clean, X_clean).fit()
                
                # Store model parameters
                params = model.params
                pvalues = model.pvalues
                overall_pvalue = model.f_pvalue
                rmse = np.sqrt(np.mean(model.resid ** 2))
                r_squared = model.rsquared
                
                # Compute LMG importance
                dominance_regression = Dominance(data=model_data, target=y.name, objective=1)  # objective=1 => regression
                lmg_scores = dominance_regression.incremental_rsquare()
                # this computes the  Overall Average Incremental R-square contribution 
                # of the predictors to the R-square of the complete model, which is equivalent to LMG
             
                # Normalize to percent
                total_lmg = sum(v for k, v in lmg_scores.items() if k != 'const')
                lmg_percent = {
                    f"LMG_{k}": (v / total_lmg) * 100
                    for k, v in lmg_scores.items()
                    if k != 'const'
                }
                            
                # Display result
                print("\nRelative Importance (LMG % of R²):")
                for var, imp in sorted(lmg_percent.items(), key=lambda x: x[1], reverse=True):
                    print(f"{var:>10}: {imp:.2f}%")
                
                x_name = "_".join(X_names_ls[i])
                y_name = y_name_ls[j]
                
                result_entry = {"X": x_name, "Y": y_name, "Overall P-Value": overall_pvalue, "RMSE": rmse, "R-Squared": r_squared}
                for param in params.index:
                    result_entry[f"Coef_{param}"] = params[param]
                    result_entry[f"PValue_{param}"] = pvalues[param]
                    
                # Add LMG scores
                result_entry.update(lmg_percent)
                
                # Save correlation information
                mean_X = df[X_ls[i]].mean()
                var_X = df[X_ls[i]].var()
                print(f"X={x_name}, Y={y_name}, Mean: {mean_X}, Variance: {var_X}")
            
                # Generate and save residual plot
                plt.figure()
                sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, color='r')
                plt.title(f'Residual Plot (X={x_name}, Y={y_name})')
                plt.savefig(f'{outDir}\\residual_plot_{x_name}_{y_name}.png')
                plt.close()
                
                # Q-Q plot for residuals
                plt.figure()
                sm.qqplot(model.resid, line='45')
                plt.title(f'Q-Q Plot (X={x_name}, Y={y_name})')
                plt.savefig(f'{outDir}\\qqplot_{x_name}_{y_name}.png')
                plt.close()
                
                # Q-Q plot against uniform distribution
                plt.figure()
                sm.qqplot(model.resid, dist=stats.uniform, line='45')
                plt.title(f'Q-Q Plot against Uniform Distribution (X={x_name}, Y={y_name})')
                plt.savefig(f'{outDir}\\qqplot_uniform_{x_name}_{y_name}.png')
                plt.close()
                
                # Histogram of residuals
                plt.figure()
                plt.hist(model.resid, bins=30, edgecolor='black')
                plt.title(f'Residual Histogram (X={x_name}, Y={y_name})')
                plt.savefig(f'{outDir}\\residual_hist_{x_name}_{y_name}.png')
                plt.close()
                
            else: 
                result_entry = {"X": x_name, "Y": y_name, "Overall P-Value": 'NA', "RMSE": 'NA', "R-Squared": 'NA'}
               
            model_results.append(result_entry)
    
    # Save model results to a spreadsheet
    df_results = pd.DataFrame(model_results)
    df_results.to_csv(f'{outDir}\\{THIS_SITE_old}_{THIS_SITE_new}_model_results.csv', index=False)
    
    df_results['Site'] = THIS_SITE_new
    comb_res.append(df_results)
        
comb_res_df = pd.concat(comb_res)
comb_res_df.to_csv(f'{outDir}\\AllSites_combined_model_results.csv', index=False)

# Combine simple linear regression results
linear_all_df = pd.concat(linear_all, ignore_index=True)
linear_all_df.to_csv(f'{outDir}\\AllSites_combined_linear_results.csv', index=False)

# Build publication-ready summaries for Bare_3_norm/Mod_3_norm models
def build_publication_table(df, target_y):
    subset = df[(df['X'] == 'Bare_3_norm_Mod_3_norm') & (df['Y'] == target_y)].copy()
    # Map land cover group
    subset['Land cover group'] = subset['Site'].map(get_canopy_group).fillna('Unknown')
    # Select and rename columns
    cols = [
        'Site',
        'Land cover group',
        'RMSE',
        'R-Squared',
        'LMG_Bare_3_norm',
        'LMG_Mod_3_norm',
        'Coef_const',
        'Coef_Bare_3_norm',
        'Coef_Mod_3_norm',
        'PValue_const',
        'PValue_Bare_3_norm'
    ]
    existing_cols = [c for c in cols if c in subset.columns]
    subset = subset[existing_cols]
    subset.rename(columns={
        'R-Squared': 'R2',
        'LMG_Bare_3_norm': 'Bare % R2',
        'LMG_Mod_3_norm': 'Mod % R2',
        'Coef_const': 'a',
        'Coef_Bare_3_norm': 'b',
        'Coef_Mod_3_norm': 'c',
        'PValue_const': 'a p-value',
        'PValue_Bare_3_norm': 'b p-value'
    }, inplace=True)
    # Coerce key numeric fields to numeric so group means work
    numeric_candidates = ['RMSE', 'R2', 'Bare % R2', 'Mod % R2', 'a', 'b', 'c', 'a p-value', 'b p-value']
    for col in numeric_candidates:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')
    # Drop rows where all numeric fields are missing (e.g., no data for this site)
    subset.dropna(subset=[c for c in numeric_candidates if c in subset.columns], how='all', inplace=True)
    # Add summary rows per land cover group (mean of numeric columns)
    numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()
    group_means = (
        subset.groupby('Land cover group')[numeric_cols]
        .mean()
        .reset_index()
    )
    group_means.insert(0, 'Site', 'Group mean')
    combined = pd.concat([subset, group_means], ignore_index=True, sort=False)
    # Round all numeric values to 2 significant figures
    numeric_cols_combined = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols_combined] = combined[numeric_cols_combined].applymap(lambda v: round_sigfig(v, 2))
    # Ensure column order
    final_cols = ['Site', 'Land cover group', 'RMSE', 'R2', 'Bare % R2', 'Mod % R2', 'a', 'b', 'c', 'a p-value', 'b p-value']
    combined = combined[[c for c in final_cols if c in combined.columns]]
    return combined

gsi_table = build_publication_table(comb_res_df, 'Cum_GSI')
gdd_table = build_publication_table(comb_res_df, 'Cum_GDD')

gsi_table.to_csv(f'{outDir}\\publication_summary_GSI_Bare3_Mod3.csv', index=False)
gdd_table.to_csv(f'{outDir}\\publication_summary_GDD_Bare3_Mod3.csv', index=False)

# Build publication-ready summary for simple linear regressions
def build_linear_summary(df):
    subset = df.copy()
    if subset.empty:
        return subset
    # Coerce numeric fields
    numeric_candidates = ['slope', 'intercept', 'r', 'Rsq', 'slope p-value']
    for col in numeric_candidates:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')
    subset.dropna(subset=numeric_candidates, how='all', inplace=True)
    subset['Land cover group'] = subset['Land cover group'].fillna('Unknown')
    # Group means by land cover
    numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()
    group_means = (
        subset.groupby('Land cover group')[numeric_cols]
        .mean()
        .reset_index()
    )
    group_means.insert(0, 'Site', 'Group mean')
    combined = pd.concat([subset, group_means], ignore_index=True, sort=False)
    # Round numerics
    numeric_cols_combined = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols_combined] = combined[numeric_cols_combined].applymap(lambda v: round_sigfig(v, 2))
    final_cols = ['Site', 'Land cover group', 'x', 'y', 'slope', 'intercept', 'r', 'Rsq', 'slope p-value']
    combined = combined[[c for c in final_cols if c in combined.columns]]
    return combined

linear_summary = build_linear_summary(linear_all_df)
linear_summary.to_csv(f'{outDir}\\publication_summary_linear_regressions.csv', index=False)

# Save separate tables for each linear regression x/y combination
for (x_name, y_name), combo_df in linear_all_df.groupby(['x', 'y']):
    combo_summary = build_linear_summary(combo_df)
    if combo_summary.empty:
        continue
    safe_x = str(x_name).replace('/', '_').replace(' ', '')
    safe_y = str(y_name).replace('/', '_').replace(' ', '')
    combo_summary.to_csv(f'{outDir}\\publication_summary_linear_{safe_x}_{safe_y}.csv', index=False)

