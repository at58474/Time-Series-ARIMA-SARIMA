from tsmodule import Preprocessing
from tsmodule import CreateDataframes
from tsmodule import ParameterEstimation
from tsmodule import ModelFitting
from tsmodule import Diagnostics
from tsmodule import CrossValidation

# Shared Parameters
start_date = '2013-01-01'
end_date = '2022-12-31'
delimiter = '\t'
date_col = 'Date'
data_freq = '15min'
impute_method = 'ffill'

arima_pre = Preprocessing(start_date, end_date, delimiter, date_col, data_freq, impute_method)

# Process Pound river data
pound_filepath = 'data/raw/pound_river_10yr_cfs_data.txt'
pound_save_filepath = 'data/pound.csv'
pound_list_col_del = ['agency_cd', 'site_no', 'tz_cd', '147720_00060_cd']
pound_dict_col_rename = {"datetime": "Date", "147720_00060": "Flow"}

df_pound = arima_pre.preprocessing(pound_filepath, pound_save_filepath, pound_list_col_del, pound_dict_col_rename)

# Process Russell Fork river data
rf_filepath = 'data/raw/russell_fork_10yr_cfs_data.txt'
rf_save_filepath = 'data/russellfork.csv'
rf_list_col_del = ['agency_cd', 'site_no', 'tz_cd', '147710_00060_cd']
rf_dict_col_rename = {"datetime": "Date", "147710_00060": "Flow"}

df_rf = arima_pre.preprocessing(rf_filepath, rf_save_filepath, rf_list_col_del, rf_dict_col_rename)

# Merge Pound and Russell Fork river data
dict_col = {"Flow_x": "Pound_Flow", "Flow_y": "Rf_Flow"}
new_col_name = 'Flow'
merged_filepath = 'data/rfg.csv'
df = arima_pre.merge_dfs(df_pound, df_rf, dict_col, new_col_name, merged_filepath)

# Resamples the 15-minute RFG dataframe into dataframes with intervals specified by frequency_list by using
#   the first, mean, and max resample methods for each dataframe, then stores all the dataframes in a dictionary
frequency_list = ['15min', '30min', '1H', '6H', '12H', '1D', '7D', '14D', '1M']
arima_create = CreateDataframes(df, frequency_list, date_col)
df_dict = arima_create.create_dfs()

# To reduce the runtime set row_cap to the maximum number of rows wanted to truncate the dataframes
row_cap = 10000
df_dict_trun = arima_create.set_row_cap(row_cap)

# This calls the ParameterEstimation class which provides the necessary information to estimate good starting
#   parameters for fitting ARIMA and SARIMA models
arima_param_estimation = ParameterEstimation(df_dict_trun)

# This returns a dictionary of dataframes that have been differenced, if not stationary, and left alone is stationary
# Also returns a dataframe that summarizes the results of the adfuller test and whether the df was differenced or not
# Parameters to print the results dataframe to the screen and save the dataframe as a pdf can be set to True or False
show_stationarity_df = False
save_stationarity_df = True
diff_dict, df_stationarity = arima_param_estimation.stationarity_check(show_stationarity_df, save_stationarity_df)

# This creates ACF and PACF plots for each dataframe in the diff_dict dictionary
# The plots can be displayed to the screen or saved as jpg files by setting the parameters
col_name = 'Flow'
# Setting this to True can be quite annoying, easier to look at the jpg's
show_acf_pacf_plots = False
save_acf_pacf_plots = True
lags = 50
arima_param_estimation.autocorr_analysis(col_name, show_acf_pacf_plots, save_acf_pacf_plots, lags)

# This runs decomposition plots to show trend, seasonal trend, and residuals, along with line plot of the data
# Performed for each dataframe and can set parameters to print to screen or save to file
show_decomp_plots = False
save_decomp_plots = True
# Yearly seasonal trend so match seasonal_period. Under 1 day seasonal trend will be 1 day, >= 1 day yearly trend
seasonal_period = [96, 48, 48, 48, 24, 24, 24, 4, 4, 4, 2, 2, 2, 365, 365, 365, 52, 52, 52, 26, 26, 26, 12, 12, 12]
#seasonal_period = [24, 24, 24]
arima_param_estimation.plots(seasonal_period, show_decomp_plots, save_decomp_plots)

# Model Fitting
testing_set_size = 10
# List of tuples of ARIMA parameters for each dataframe since SARIMAX passed the parameters as a tuple
order_list = [(5, 0, 0),  # '15min 1 year'
              (3, 0, 0),  # '30min 2 year'
              (3, 0, 0),  # '30 min 2 year mean'
              (3, 0, 0),  # '30 min 2 year max'
              (2, 0, 0),  # 'hourly 5 year'
              (3, 1, 2),  # 'hourly 5 year mean'
              (2, 0, 0),  # 'hourly 5 year max'
              (3, 0, 0),  # '6 hour 10 year'
              (3, 1, 0),  # '6 hour 10 year mean'
              (3, 0, 0),  # '6 hour 10 year max'
              (2, 0, 0),  # '12 hour 10 year'
              (3, 1, 0),  # '12 hour 10 year mean'
              (2, 0, 0),  # '12 hour 10 year max'
              (1, 0, 0),  # 'daily 10 year'
              (1, 1, 0),  # 'daily 10 year mean'
              (3, 0, 0),  # 'daily 10 year max'
              (1, 0, 1),  # '7 day 10 year'
              (1, 1, 3),  # '7 day 10 year mean'
              (1, 0, 2),  # '7 day 10 year max'
              (1, 0, 1),  # '14 day 10 year'
              (1, 1, 3),  # '14 day 10 year mean'
              (1, 0, 1),  # '14 day 10 year max'
              (1, 0, 1),  # 'monthly 10 year'
              (2, 2, 1),  # 'monthly 10 year mean'
              (1, 0, 1)]  # 'monthly 10 year max'

show = False
save = True
arima_model_fitting = ModelFitting(df_dict_trun, testing_set_size, order_list, new_col_name)
dict_results_isnr = arima_model_fitting.arima_in_sample_non_rolling(show, save)
dict_results_isr = arima_model_fitting.arima_in_sample_rolling(show, save)
dict_results_oosr = arima_model_fitting.arima_out_of_sample_rolling(show, save)

arima_diag = Diagnostics(df_dict_trun, testing_set_size, new_col_name, dict_results_isnr, dict_results_isr, dict_results_oosr)
arima_diag.residual_analysis(show, save)

# Set the number of cross validation groups
groups = 5
# Select which models to cross validate
keys = ['1H_mean', '6H_mean', '12H_mean']
df_dict_cv = {x:df_dict_trun[x] for x in keys}
order_list_trun = [(3, 1, 2),  # '6 hour 10 year mean'
                   (3, 1, 0),  # '12 hour 10 year mean'
                   (1, 1, 0)]  # 'daily 10 year mean'

# For cross validation only forecast() will be used, not predict(). Also, will only use rolling forecasts
arima_cross_validation = CrossValidation(df_dict_cv, new_col_name, dict_results_isr, dict_results_oosr, groups, order_list_trun)
arima_cross_validation.cross_validation(show, save)



'''
order_list = [(5, 0, 0),  # '15min 1 year'
              (3, 0, 0),  # '30min 2 year'
              (3, 0, 0),  # '30 min 2 year mean'
              (3, 0, 0),  # '30 min 2 year max'
              (2, 0, 0),  # 'hourly 5 year'
              (3, 1, 2),  # 'hourly 5 year mean'
              (2, 0, 0),  # 'hourly 5 year max'
              (3, 0, 0),  # '6 hour 10 year'
              (3, 1, 0),  # '6 hour 10 year mean'
              (3, 0, 0),  # '6 hour 10 year max'
              (2, 0, 0),  # '12 hour 10 year'
              (3, 1, 0),  # '12 hour 10 year mean'
              (2, 0, 0),  # '12 hour 10 year max'
              (1, 0, 0),  # 'daily 10 year'
              (1, 1, 0),  # 'daily 10 year mean'
              (3, 0, 0),  # 'daily 10 year max'
              (1, 0, 1),  # '7 day 10 year'
              (1, 1, 3),  # '7 day 10 year mean'
              (1, 0, 2),  # '7 day 10 year max'
              (1, 0, 1),  # '14 day 10 year'
              (1, 1, 3),  # '14 day 10 year mean'
              (1, 0, 1),  # '14 day 10 year max'
              (1, 0, 1),  # 'monthly 10 year'
              (2, 2, 1),  # 'monthly 10 year mean'
              (1, 0, 1)]  # 'monthly 10 year max' 
              
              '''
