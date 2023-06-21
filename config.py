'''
SETTINGS
'''
# Choose which modules to run
run_model_fitting = False
run_diagnostics = False
run_cross_validation = False
run_auto_arima = True

# Set model_type to ARIMA or SARIMA, depending on which needs to be used
model_type = 'ARIMA'

# Set this to true to delete all non archived files in the directory structure
delete_files = False

# Set this to true to archive all result files to the appropriate directory
archive_files = False

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
CREATE DATAFRAMES PARAMETERS
'''
# Resamples the 15-minute RFG dataframe into dataframes with intervals specified by frequency_list by using
#   the first, mean, and max resample methods for each dataframe, then stores all the dataframes in a dictionary
# frequency_list = ['1H', '6H', '12H', '1D', '7D', '14D', '1M']
frequency_list = ['1H', '6H', '12H', '1D']

# To reduce the runtime set row_cap to the maximum number of rows wanted to truncate the dataframes
row_cap = 20000

'''
PARAMETER ESTIMATION PARAMETERS
'''
# Number of lags in the ACF and PACF plots
lags = 50
# Yearly seasonal trend so match seasonal_period. Under 1 day seasonal trend will be 1 day, >= 1 day yearly trend
seasonal_period = [8760, 8760, 8760, 1460, 1460, 1460, 730, 730, 730, 365, 365, 365]
# seasonal_period = [8760, 8760, 8760, 1460, 1460, 1460, 730, 730, 730, 365, 365, 365, 52, 52, 52, 26, 26, 26, 12, 12, 12]
# seasonal_period = [24, 24, 24, 4, 4, 4, 2, 2, 2, 365, 365, 365, 52, 52, 52, 26, 26, 26, 12, 12, 12]

if run_model_fitting:
    '''
    MODEL FITTING PARAMETERS
    '''
    testing_set_size = 5
    # List of tuples of ARIMA parameters for each dataframe since SARIMAX passed the parameters as a tuple
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

    order_list_sarima = [(2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year'
                         (2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year mean'
                         (2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year max'
                         (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year'
                         (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year mean'
                         (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year max'
                         (2, 0, 2, 3, 0, 3, 12),  # 'monthly 10 year'
                         (2, 0, 2, 3, 0, 3, 12),  # 'monthly 10 year mean'
                         (2, 0, 2, 3, 0, 3, 12)]  # 'monthly 10 year max'

if run_cross_validation:
    '''
    CROSS VALIDATION PARAMETERS
    '''
    # Set the number of cross validation groups
    groups = 5
    # Set this to the size of the in-sample and out_of_sample cross-validation groups
    # Need to ensure there are enough rows in the dataframe, ie. if number of groups is 5 and testing_set_size_cv
    #   is 10 then that would require a minimum of 100 rows of data.
    testing_set_size_cv = 5
    # Select which models to cross validate
    keys = ['1H_mean', '6H_mean', '12H_mean', '1D_mean']

    order_list_trun_arima = [(3, 1, 2, 0, 0, 0, 0),  # 1H_mean
                             (0, 1, 1, 0, 0, 0, 0),  # 6H_mean
                             (1, 0, 3, 0, 0, 0, 0),  # 12H_mean
                             (1, 1, 0, 0, 0, 0, 0)]  # 1D_mean

    order_list_trun_sarima = [(2, 1, 2, 1, 0, 2, 52),   # 7D_mean
                              (1, 0, 1, 3, 0, 3, 26),   # 14D_mean
                              (2, 0, 2, 3, 0, 3, 12)]   # 1M_mean

if run_auto_arima:
    '''
    AUTO ARIMA PARAMETERS
    '''
    # Select which time interval to use.
    key = ['1D_max']

    # The lower and upper bounds can be set here for the ARIMA or SARIMA parameters
    # To run an ARIMA model set the values of P, D, Q, and m all to 0
    # WARNING: the number of models can become very large so use the results from above to narrow the range of each
    #          parameter as much as possible.

    # ARIMA
    # Non-Seasonal Autoregressive Order (p)
    p_min = 0
    p_max = 2
    # Number of Non-Seasonal Differences (d)
    d_min = 0
    d_max = 2
    # Non-Seasonal Moving Average Order (q)
    q_min = 0
    q_max = 2

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