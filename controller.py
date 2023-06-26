from tsmodule import Preprocessing
from tsmodule import CreateDataframes
from tsmodule import ParameterEstimation
from tsmodule import ModelFitting
from tsmodule import Diagnostics
from tsmodule import CrossValidation
from tsmodule import AutoARIMA
from tsmodule import FileHandling

from itertools import product

from config import *

file_handling = FileHandling()
if delete_files:
    file_handling.delete_files()

'''
PREPROCESSING
'''
pound_filepath = 'data/raw/' + raw_filename1
pound_save_filepath = 'data/' + save_filename1
arima_pre = Preprocessing(start_date, end_date, delimiter, date_col, data_freq, impute_method)
df_pound = arima_pre.preprocessing(pound_filepath, pound_save_filepath, list_col_del1, dict_col_rename1)

if process_and_merge_2_raw_files:
    rf_filepath = 'data/raw/' + raw_filename2
    rf_save_filepath = 'data/' + save_filename2
    merged_filepath = 'data/' + merged_filepath
    df_rf = arima_pre.preprocessing(rf_filepath, rf_save_filepath, list_col_del2, dict_col_rename2)
    df = arima_pre.merge_dfs(df_pound, df_rf, dict_col, new_col_name, merged_filepath)
else:
    df = df_pound
'''
CREATE DATAFRAMES
'''
arima_create = CreateDataframes(df, frequency_list, date_col)
df_dict = arima_create.create_dfs()
df_dict_trun = arima_create.set_row_cap(row_cap)

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
    DIAGNOSTICS
    '''
    arima_diag = Diagnostics(df_dict_trun, testing_set_size, new_col_name, dict_results_isnr, dict_results_isr, dict_results_oosr)
    arima_diag.residual_analysis(show, save)

if run_cross_validation:
    df_dict_cv = {x:df_dict_trun[x] for x in keys}

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
    df_dict_auto = {x:df_dict_trun[x] for x in key}

    '''
    AUTO ARIMA
    '''
    non_seasonal_params = list(product(range(p_min, p_max + 1), range(d_min, d_max + 1), range(q_min, q_max + 1)))
    seasonal_params = list(product(range(P_min, P_max + 1), range(D_min, D_max + 1), range(Q_min, Q_max + 1), m))
    arima_auto = AutoARIMA(df_dict_auto, non_seasonal_params, seasonal_params, new_col_name, model_type)
    arima_auto.auto_arima(show, save)
