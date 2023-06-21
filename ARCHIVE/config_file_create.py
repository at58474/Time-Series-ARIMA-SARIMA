from configparser import ConfigParser

config = ConfigParser(allow_no_value=True)

config.add_section('settings')

config.set('settings', '\n ; ||| Choose which modules to run |||', None)
config.set('settings', 'run_model_fitting', 'True')
config.set('settings', 'run_diagnostics', 'True')
config.set('settings', 'run_cross_validation', 'True')
config.set('settings', 'run_auto_arima', 'True')

config.set('settings', '\n ; ||| Set model_type to ARIMA or SARIMA, depending on which needs to be used |||')
config.set('settings', 'model_type', 'True')

config.set('settings', '\n; ||| Set this to true to delete all non archived files in the directory structure |||')
config.set('settings', 'delete_files', 'True')

config.set('settings', '\n; ||| Set this to true to archive all result files to the appropriate directory, NOT IMPLEMENTED |||')
config.set('settings', 'archive_files', 'False')

config.set('settings', '\n; ||| Start and End dates for the raw data |||')
config.set('settings', 'start_date', '2013-01-01')
config.set('settings', 'end_date', '2022-12-31')

config.set('settings', '\n; ||| If the raw data is not comma delimited, set this to the correct delimiter |||')
config.set('settings', 'delimiter', r'\t')

config.set('settings', '\n; ||| Set the name of the date column here |||')
config.set('settings', 'date_col', 'Date')

config.set('settings', '\n; ||| Set this to the time interval between each data point |||')
config.set('settings', 'data_freq', '15min')

config.set('settings', '\n; ||| Determine the optimal imputation method and set it here |||')
config.set('settings', 'impute_method', 'ffill')

config.set('settings', '\n; ||| Parameters to print the results to the screen and save the results as a pdf or jpg: set each to True or False |||')
config.set('settings', 'show', 'False')
config.set('settings', 'save', 'True')

config.set('settings', '\n; ||| Set the name of the data column here |||')
config.set('settings', 'col_name', 'Flow')


config.add_section('preprocessing')

config.set('preprocessing', '\n; ||| Process Pound river data |||')
config.set('preprocessing', 'pound_filepath', 'data/raw/pound_river_10yr_cfs_data.txt')
config.set('preprocessing', 'pound_save_filepath', 'data/pound.csv')
pound_list_col_del = ['agency_cd', 'site_no', 'tz_cd', '147720_00060_cd']
config.set('preprocessing', 'pound_list_col_del', str(pound_list_col_del))
pound_dict_col_rename = {"datetime": "Date", "147720_00060": "Flow"}
config.set('preprocessing', 'pound_dict_col_rename', str(pound_dict_col_rename))

config.set('preprocessing', '\n; ||| Process Russell Fork river data |||')
config.set('preprocessing', 'rf_filepath', 'data/raw/russell_fork_10yr_cfs_data.txt')
config.set('preprocessing', 'rf_save_filepath', 'data/russellfork.csv')
rf_list_col_del = ['agency_cd', 'site_no', 'tz_cd', '147710_00060_cd']
config.set('preprocessing', 'rf_list_col_del', str(rf_list_col_del))
rf_dict_col_rename = {"datetime": "Date", "147710_00060": "Flow"}
config.set('preprocessing', 'rf_dict_col_rename', str(rf_dict_col_rename))

config.set('preprocessing', '\n; ||| Merge Pound and Russell Fork river data |||')
dict_col = {"Flow_x": "Pound_Flow", "Flow_y": "Rf_Flow"}
config.set('preprocessing', 'dict_col', str(dict_col))
config.set('preprocessing', 'new_col_name', 'Flow')
config.set('preprocessing', 'merged_filepath', 'data/rfg.csv')


config.add_section('create_dataframes')

config.set('create_dataframes', '\n; ||| Resamples dataframes with intervals specified here using first, mean, and last |||')
frequency_list = ['1H', '6H', '12H', '1D', '7D', '14D', '1M']
config.set('create_dataframes', 'frequency_list', str(frequency_list))

config.set('create_dataframes', '\n; ||| To reduce the runtime set row_cap to the maximum number of rows wanted to truncate the dataframes |||')
config.set('create_dataframes', 'row_cap', '50000')


config.add_section('parameter_estimation')

config.set('parameter_estimation', '\n; ||| Number of lags in the ACF and PACF plots |||')
config.set('parameter_estimation', 'lags', '50')

config.set('parameter_estimation', '\n; ||| Set the length of the seasonal trend for each dataframe, 3 per interval |||')
seasonal_period = [8760, 8760, 8760, 1460, 1460, 1460, 730, 730, 730, 365, 365, 365, 52, 52, 52, 26, 26, 26, 12, 12, 12]
config.set('parameter_estimation', 'seasonal_period', str(seasonal_period))


config.add_section('model_fitting')

config.set('model_fitting', 'testing_set_size', '50')

config.set('model_fitting', '\n; ||| List of tuples of ARIMA parameters for each dataframe since SARIMAX passed the parameters as a tuple |||')
order_list_arima = [(3, 0, 3, 0, 0, 0, 0),  # 'hourly 5 year'
                    (3, 1, 1, 0, 0, 0, 0),  # 'hourly 5 year mean'
                    (3, 0, 3, 0, 0, 0, 0),  # 'hourly 5 year max'
                    (0, 2, 2, 0, 0, 0, 0),  # '6 hour 10 year'
                    (0, 1, 1, 0, 0, 0, 0),  # '6 hour 10 year mean'
                    (0, 2, 1, 0, 0, 0, 0),  # '6 hour 10 year max'
                    (0, 1, 0, 0, 0, 0, 0),  # '12 hour 10 year'
                    (1, 0, 3, 0, 0, 0, 0),  # '12 hour 10 year mean'
                    (0, 1, 1, 0, 0, 0, 0),  # '12 hour 10 year max'
                    (0, 1, 0, 0, 0, 0, 0),  # 'daily 10 year'
                    (1, 0, 2, 0, 0, 0, 0),  # 'daily 10 year mean'
                    (3, 0, 0, 0, 0, 0, 0)]  # 'daily 10 year max'
config.set('model_fitting', 'order_list_arima', str(order_list_arima))
order_list_sarima = [(2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year'
                     (2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year mean'
                     (2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year max'
                     (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year'
                     (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year mean'
                     (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year max'
                     (2, 0, 2, 3, 0, 3, 12),  # 'monthly 10 year'
                     (2, 0, 2, 3, 0, 3, 12),  # 'monthly 10 year mean'
                     (2, 0, 2, 3, 0, 3, 12)]  # 'monthly 10 year max'
config.set('model_fitting', 'order_list_sarima', str(order_list_sarima))


config.add_section('cross_validation')

config.set('cross_validation', '\n; ||| Set the number of cross validation groups |||')
config.set('cross_validation', 'groups', '5')

config.set('cross_validation', '\n; ||| Set this to the size of the in-sample and out_of_sample cross-validation groups |||')
config.set('cross_validation', '\n; ||| Need to ensure there are enough rows in the dataframe, ie. if number of groups is 5 and testing_set_size_cv |||')
config.set('cross_validation', '\n; ||| is 10 then that would require a minimum of 100 rows of data. |||')
config.set('cross_validation', 'testing_set_size_cv', '50')

config.set('cross_validation', '\n; ||| Select which models to cross validate |||')
keys = ['1H_mean', '6H_mean', '12H_mean', '1D_mean']
config.set('cross_validation', 'keys', str(keys))

order_list_trun_arima = [(3, 1, 2, 0, 0, 0, 0),  # 1H_mean
                         (0, 1, 1, 0, 0, 0, 0),  # 6H_mean
                         (1, 0, 3, 0, 0, 0, 0),  # 12H_mean
                         (1, 1, 0, 0, 0, 0, 0)]  # 1D_mean
config.set('cross_validation', 'order_list_trun_arima', str(order_list_trun_arima))
order_list_trun_sarima = [(2, 1, 2, 1, 0, 2, 52),   # 7D_mean
                          (1, 0, 1, 3, 0, 3, 26),   # 14D_mean
                          (2, 0, 2, 3, 0, 3, 12)]   # 1M_mean
config.set('cross_validation', 'order_list_trun_sarima', str(order_list_trun_sarima))


config.add_section('auto_arima')

config.set('auto_arima', '\n; ||| Select which time interval to use. |||')
key = ['1D_max']
config.set('auto_arima', 'key', str(key))

config.set('auto_arima', '\n; ||| The lower and upper bounds can be set here for the ARIMA or SARIMA parameters |||')
config.set('auto_arima', '\n; ||| To run an ARIMA model set the values of P, D, Q, and m all to 0 |||')
config.set('auto_arima', '\n; ||| WARNING: the number of models can become very large so use the results from above |||')
config.set('auto_arima', '\n; ||| to narrow the range of each parameter as much as possible. |||')

config.set('auto_arima', '\n; ||| ARIMA |||')
config.set('auto_arima', '\n; ||| Non-Seasonal Autoregressive Order (p) |||')
config.set('auto_arima', 'p_min', '0')
config.set('auto_arima', 'p_max', '3')
config.set('auto_arima', '\n; ||| Number of Non-Seasonal Differences (d) |||')
config.set('auto_arima', 'd_min', '0')
config.set('auto_arima', 'd_max', '2')
config.set('auto_arima', '\n; ||| Non-Seasonal Moving Average Order (q) |||')
config.set('auto_arima', 'q_min', '0')
config.set('auto_arima', 'q_max', '3')

config.set('auto_arima', '\n; ||| SARIMA |||')
config.set('auto_arima', '\n; ||| Seasonal Autoregressive Order (P) |||')
config.set('auto_arima', 'P_min', '0')
config.set('auto_arima', 'P_max', '0')
config.set('auto_arima', '\n; ||| Number of Seasonal Differences (D) |||')
config.set('auto_arima', 'D_min', '0')
config.set('auto_arima', 'D_max', '0')
config.set('auto_arima', '\n; ||| Seasonal Moving Average Order (Q) |||')
config.set('auto_arima', 'Q_min', '0')
config.set('auto_arima', 'Q_max', '0')
config.set('auto_arima', '\n; ||| Length of the season |||')
m = [0]
config.set('auto_arima', 'm', str(m))


with open('./config.ini', 'w') as file:
    config.write(file)