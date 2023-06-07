from tsmodule import Preprocessing
from tsmodule import CreateDataframes
from tsmodule import ParameterEstimation
from tsmodule import ModelFitting
from tsmodule import Diagnostics
from tsmodule import CrossValidation
from tsmodule import AutoARIMA
from tsmodule import FileHandling

from itertools import product

# Choose which modules to run, the first three are necessary
run_model_fitting = True
run_diagnostics = True
run_cross_validation = True
run_auto_arima = False

# Set model_type to ARIMA or SARIMA, depending on which needs to be used
model_type = 'ARIMA'

# Set this to true to delete all non archived files in the directory structure
delete_files = True

# Set this to true to archive all result files to the appropriate directory
archive_files = False

file_handling = FileHandling()
if delete_files:
    file_handling.delete_files()

'''
STARTING PARAMETERS
'''
# Start and End dates for the raw data
start_date = '2013-01-01'
end_date = '2022-12-31'
# If the raw data is not comma delimited, set this to the correct delimiter
delimiter = '\t'
# Set the name of the date column here
date_col = 'Date'
# Set this to the time interval between each data point
data_freq = '15min'
# Determine the optimal imputation method and set it here
impute_method = 'ffill'
# Parameters to print the results to the screen and save the results as a pdf or jpg: set each to True or False
show = False
save = True
# Set the name of the data column here
col_name = 'Flow'





'''
PREPROCESSING PARAMETERS
'''
# Process Pound river data
pound_filepath = 'data/raw/pound_river_10yr_cfs_data.txt'
pound_save_filepath = 'data/pound.csv'
pound_list_col_del = ['agency_cd', 'site_no', 'tz_cd', '147720_00060_cd']
pound_dict_col_rename = {"datetime": "Date", "147720_00060": "Flow"}

# Process Russell Fork river data
rf_filepath = 'data/raw/russell_fork_10yr_cfs_data.txt'
rf_save_filepath = 'data/russellfork.csv'
rf_list_col_del = ['agency_cd', 'site_no', 'tz_cd', '147710_00060_cd']
rf_dict_col_rename = {"datetime": "Date", "147710_00060": "Flow"}

# Merge Pound and Russell Fork river data
dict_col = {"Flow_x": "Pound_Flow", "Flow_y": "Rf_Flow"}
new_col_name = 'Flow'
merged_filepath = 'data/rfg.csv'

'''
PREPROCESSING
'''
arima_pre = Preprocessing(start_date, end_date, delimiter, date_col, data_freq, impute_method)
df_pound = arima_pre.preprocessing(pound_filepath, pound_save_filepath, pound_list_col_del, pound_dict_col_rename)
df_rf = arima_pre.preprocessing(rf_filepath, rf_save_filepath, rf_list_col_del, rf_dict_col_rename)
df = arima_pre.merge_dfs(df_pound, df_rf, dict_col, new_col_name, merged_filepath)





'''
CREATE DATAFRAMES PARAMETERS
'''
# Resamples the 15-minute RFG dataframe into dataframes with intervals specified by frequency_list by using
#   the first, mean, and max resample methods for each dataframe, then stores all the dataframes in a dictionary
frequency_list = ['1H', '6H', '12H', '1D']

# To reduce the runtime set row_cap to the maximum number of rows wanted to truncate the dataframes
row_cap = 10000

'''
CREATE DATAFRAMES
'''
arima_create = CreateDataframes(df, frequency_list, date_col)
df_dict = arima_create.create_dfs()
df_dict_trun = arima_create.set_row_cap(row_cap)





'''
PARAMETER ESTIMATION PARAMETERS
'''
# Number of lags in the ACF and PACF plots
lags = 50
# Yearly seasonal trend so match seasonal_period. Under 1 day seasonal trend will be 1 day, >= 1 day yearly trend
seasonal_period = [24, 24, 24, 4, 4, 4, 2, 2, 2, 365, 365, 365]

'''
PARAMETER ESTIMATION
'''
# This calls the ParameterEstimation class which provides the necessary information to estimate good starting
#   parameters for fitting ARIMA and SARIMA models
arima_param_estimation = ParameterEstimation(df_dict_trun)
# This returns a dictionary of dataframes that have been differenced, if not stationary, and left alone is stationary
# Also returns a dataframe that summarizes the results of the adfuller test and whether the df was differenced or not
diff_dict, df_stationarity = arima_param_estimation.stationarity_check(show, save)
# This creates ACF and PACF plots for each dataframe in the diff_dict dictionary
# The plots can be displayed to the screen or saved as jpg files by setting the parameters
arima_param_estimation.autocorr_analysis(col_name, show, save, lags)
# This runs decomposition plots to show trend, seasonal trend, and residuals, along with line plot of the data
arima_param_estimation.plots(seasonal_period, show, save)




if run_model_fitting:
    '''
    MODEL FITTING PARAMETERS
    '''
    testing_set_size = 50
    # List of tuples of ARIMA parameters for each dataframe since SARIMAX passed the parameters as a tuple
    order_list_arima = [(3, 1, 2, 0, 0, 0, 0),  # 'hourly 5 year'
                        (3, 1, 2, 0, 0, 0, 0),  # 'hourly 5 year mean'
                        (3, 1, 2, 0, 0, 0, 0),  # 'hourly 5 year max'
                        (0, 1, 1, 0, 0, 0, 0),  # '6 hour 10 year'
                        (0, 1, 1, 0, 0, 0, 0),  # '6 hour 10 year mean'
                        (0, 1, 1, 0, 0, 0, 0),  # '6 hour 10 year max'
                        (1, 0, 3, 0, 0, 0, 0),  # '12 hour 10 year'
                        (1, 0, 3, 0, 0, 0, 0),  # '12 hour 10 year mean'
                        (1, 0, 3, 0, 0, 0, 0),  # '12 hour 10 year max'
                        (1, 1, 0, 0, 0, 0, 0),  # 'daily 10 year'
                        (1, 1, 0, 0, 0, 0, 0),  # 'daily 10 year mean'
                        (1, 1, 0, 0, 0, 0, 0)]  # 'daily 10 year max'

    order_list_sarima = [(2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year'
                         (2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year mean'
                         (2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year max'
                         (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year'
                         (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year mean'
                         (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year max'
                         (2, 0, 2, 3, 0, 3, 12),  # 'monthly 10 year'
                         (2, 0, 2, 3, 0, 3, 12),  # 'monthly 10 year mean'
                         (2, 0, 2, 3, 0, 3, 12)]  # 'monthly 10 year max'

    '''
    MODEL FITTING
    '''
    # Model Fitting
    if model_type == 'ARIMA':
        arima_model_fitting = ModelFitting(df_dict_trun, testing_set_size, order_list_arima, new_col_name, model_type)
        dict_results_isnr = arima_model_fitting.arima_in_sample_non_rolling(show, save)
        dict_results_isr = arima_model_fitting.arima_in_sample_rolling(show, save)
        dict_results_oosr = arima_model_fitting.arima_out_of_sample_rolling(show, save)
    elif model_type == 'SARIMA':
        sarima_model_fitting = ModelFitting(df_dict_trun, testing_set_size, order_list_sarima, new_col_name, model_type)
        dict_results_isnr = sarima_model_fitting.arima_in_sample_non_rolling(show, save)
        dict_results_isr = sarima_model_fitting.arima_in_sample_rolling(show, save)
        dict_results_oosr = sarima_model_fitting.arima_out_of_sample_rolling(show, save)




if run_diagnostics:
    '''
    DIAGNOSTICS PARAMETERS
    '''
    # None

    '''
    DIAGNOSTICS
    '''
    arima_diag = Diagnostics(df_dict_trun, testing_set_size, new_col_name, dict_results_isnr, dict_results_isr, dict_results_oosr)
    arima_diag.residual_analysis(show, save)




if run_cross_validation:
    '''
    CROSS VALIDATION PARAMETERS
    '''
    # Set the number of cross validation groups
    groups = 5
    # Set this to the size of the in-sample and out_of_sample cross-validation groups
    # Need to ensure there are enough rows in the dataframe, ie. if number of groups is 5 and testing_set_size_cv
    #   is 10 then that would require a minimum of 100 rows of data.
    testing_set_size_cv = 10
    # Select which models to cross validate
    keys = ['1H_mean', '6H_mean', '12H_mean', '1D_mean']
    df_dict_cv = {x:df_dict_trun[x] for x in keys}
    order_list_trun_arima = [(3, 1, 2, 0, 0, 0, 0),  # 1H_mean
                             (0, 1, 1, 0, 0, 0, 0),  # 6H_mean
                             (1, 0, 3, 0, 0, 0, 0),  # 12H_mean
                             (1, 1, 0, 0, 0, 0, 0)]  # 1D_mean

    order_list_trun_sarima = [(2, 1, 2, 1, 0, 2, 52),   # 7D_mean
                              (1, 0, 1, 3, 0, 3, 26),   # 14D_mean
                              (2, 0, 2, 3, 0, 3, 12)]   # 1M_mean

    '''
    CROSS VALIDATION
    '''
    if model_type == 'ARIMA':
        # For cross validation only forecast() will be used, not predict(). Also, will only use rolling forecasts
        arima_cross_validation = CrossValidation(df_dict_cv, new_col_name, dict_results_isr, dict_results_oosr, groups, order_list_trun_arima, testing_set_size_cv, model_type)
        arima_cross_validation.cross_validation(show, save)
    elif model_type == 'SARIMA':
        # For cross validation only forecast() will be used, not predict(). Also, will only use rolling forecasts
        sarima_cross_validation = CrossValidation(df_dict_cv, new_col_name, dict_results_isr, dict_results_oosr, groups, order_list_trun_sarima, testing_set_size_cv, model_type)
        sarima_cross_validation.cross_validation(show, save)



if run_auto_arima:
    '''
    AUTO ARIMA PARAMETERS
    '''
    # Select which time interval to use.
    key = ['7D_mean']
    df_dict_auto = {x:df_dict_trun[x] for x in key}

    # The lower and upper bounds can be set here for the ARIMA or SARIMA parameters
    # To run an ARIMA model set the values of P, D, Q, and m all to 0
    # WARNING: the number of models can become very large so use the results from above to narrow the range of each
    #          parameter as much as possible.

    # ARIMA
    # Non-Seasonal Autoregressive Order (p)
    p_min = 0
    p_max = 3
    # Number of Non-Seasonal Differences (d)
    d_min = 0
    d_max = 2
    # Non-Seasonal Moving Average Order (q)
    q_min = 0
    q_max = 3

    # SARIMA
    # Seasonal Autoregressive Order (P)
    P_min = 0
    P_max = 0
    # Number of Seasonal Differences (D)
    D_min = 0
    D_max = 0
    # Seasonal Moving Average Order (Q)
    Q_min = 0
    Q_max = 0
    # Length of the season
    m = [0]

    '''
    AUTO ARIMA
    '''
    non_seasonal_params = list(product(range(p_min, p_max + 1), range(d_min, d_max + 1), range(q_min, q_max + 1)))
    seasonal_params = list(product(range(P_min, P_max + 1), range(D_min, D_max + 1), range(Q_min, Q_max + 1), m))
    arima_auto = AutoARIMA(df_dict_auto, non_seasonal_params, seasonal_params, new_col_name, model_type)
    arima_auto.auto_arima(show, save)









'''
Refactoring code to include ability to run SARIMA:
    1) Under Model Fitting(this file), include conditional statement to create ARIMA or SARIMA object with either order_params or
       order_params and seasonal_order_params, respectively
    2) In sarimax_modeling, in both modeling and cross validaiton classes, pull the actual model code out, put into function, 
       create conditional to pass either ARIMA or SARIMA params, then run the model there
    3) That should be it
'''
'''
Last thing to do:
    - Add a class that runs auto-sarimax and determines optimal parameters from a range of iterables, pull from SARIMA jupyter notebook
'''