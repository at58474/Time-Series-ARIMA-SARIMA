# Time-Series-ARIMA-SARIMA
Module for preprocessing raw data, creating dataframes, generating plots and tables for parameter estimation, fitting the data to the models, producing diagnostic outputs for model analysis, creating forecast and cross validation results plots, running auto-ARIMA and auto-SARIMA methods, and file handling.

<ins><b>Files</b></ins>
- config.py
- controller.py
- tsmodule.py

## config.py
Contains settings and parameters needed for modeling a time series dataset using ARIMA and SARIMA. A description of each parameter is listed below:

Classes that automatically run:
- Preprocessing
- CreateDataframes
- ParameterEstimation

Classes that can be turned on or off
- ModelFitting
- Diagnostics
- CrossValidation
- AutoARIMA
- FileHandling
  
### Settings

##### Parameters for enabling or disabling module functionality
- run_model_fitting: (True, False)
     - If set to True the controller will run the ModelFitting class
     - If set to False ModelFitting will not run
- run_diagnostics: (True, False)
     - True to run Diagnostics class
     - False will disable diagnostics
- run_cross_validation: (True, False)
     - True to enable CrossValidation class
     - False to disable cross validation
- run_auto_arima: (True, False)
     - True to enable AutoARIMA class
     - False to disable running auto ARIMA

##### Model Selection
- model_type: (ARIMA, SARIMA)
     - Set the model_type parameter to the target model. Currently only ARIMA and SARIMA can be used. VAR is being implemented along with several other time series models.

##### File Handling
- delete_files: (True, False)
     - If set to true, all non-archived files will be removed before running the module.
     - If set to false then the non-archived files will remain and risk being overwritten or creating clutter in the results directories
- archive_files: (True, False)
     - Not currently implemented, but will move any non-archived files to their proper archive folder. This currently needs to be done manually to avoid losing results.

##### Raw Data Settings
- start_date: ('YYYY-MM-DD')
     - The desired starting date of the time series, must be a valid date in the raw data file, which has to be located in data/raw/
     - Must be formatted as a string in the proper format described above
     - Can be the date of the first datapoint in the time series, or any subsequent point before the last datapoint
- end_date: ('YYYY-MM-DD')
     - End date for the time series, see start_date for description.

- delimiter: Ex.('\t', ' ', ';', '|')
     - If the raw data is not comma delimited then set the delimiter as a string, ie. between single quotes ''

- date_col: Ex.('Date')
     - Set the desired name of the Date column here as a string
 
- data_freq: Ex.('15min')
     - Set this parameter to the time interval of the raw data, for example if there is a datapoint every 15 minutes, enter '15min'
     - The following have to be used as a string: (min, H, D, M) Examples: 15min, 2H, 7D, 1M

- impute_method: ('ffill', 'bfill', 'linear')
     - Set the desired imputation method here as a string. It should be made certain the data can be safely propegated, if not then additional preprocessing may be required before running this module.

- show: (True, False)
     - If this is set to true then all results will be printed to the screen. This may become very invasive, use with caution or keep set to False.
     - If set to false results will not be printed to the screen.
- save: (True, False)
     - Recommended: if set to True then the results will be saved in the plots directory in the proper location as a pdf or jpg file. It is recommended to keep this set to True.
     - If set to false the results will not be saved into files, not recommended.

- col_name: Ex.('Flow')
     - Set the desired name of the data column here as a string

### Preprocessing Parameters
!!! Refactor to make more user friendly !!!
##### Process dataframe 1
- raw_filename1:  Ex.('pound_river_10yr_cfs_data.txt')
     - file name of the raw data file, which is located in data/raw
- save_filename1:  Ex.('pound.csv')
     - preferred file name for the processed data file, will be saved to data/
- list_col_del1: Ex.(['agency_cd', 'site_no', 'tz_cd', '147720_00060_cd'])
     - A list, [], of strings, '', containing any column names to be removed from the dataframe, leave blank if no columns need to be removed
- dict_col_rename1: Ex.({"datetime": "Date", "147720_00060": "Flow"})
     - A dictionary, {'key':'value'}, of columns to be renamed, passing the original name as the key and the desired name as the value, both being strings, ''
##### Process other dataframes if any
- same parameters as dataframe 1
- leave parameters blank if only 1 raw data file is used
- if more than 2 files need to be merged, either run the first 2, then use that merged file to merge with the 3rd, etc... or the other option is to modify the code to contain a 3rd merge, but could be more time consuming.

##### Merge dataframes
- dict_col: Ex({"Flow_x": "Pound_Flow", "Flow_y": "Rf_Flow"})
     - A dictionary, {'key':'value'}, of columns to be renamed, the keys will be 'col_name'_x and 'col_name'_y, then set the values to the desired column names.
- new_col_name: Ex.('Flow')
     - Set this to the desired name for the combined data dataframe column, this is redundant.
- merged_filepath: Ex.('data/rfg.csv')
     - Filepath for the merged dataframe to be used by the module, leave the location but can edit the name of the file.

### Create Dataframes Parameters
- frequency_list: Ex.(['1H', '6H', '12H', '1D'])
     - A list, [], of time intervals as strings, ''
     - This resamples the original dataset to each of the time intervals included in this list
     - Also resamples each interval by using first(), mean(), and max() resampling methods
 
- row_cap: Ex.(10000)
     - This can be used to truncate the dataframes in order to decrease the runtime of the program, if desired. Set to a number higher than the maximum number of rows in the original dataset to disable.
     - Must be an integer

### Parameter Estimation Parameters
- lags: Ex.(50)
     - Set this to the number of desired lags to be displayed in the ACF and PACF plots, it is an integer

- seasonal_period: Ex.([8760, 8760, 8760, 1460, 1460, 1460, 730, 730, 730, 365, 365, 365])
     - This is a list of integers that represents the seasonal trend, if any, for each time interval.
     - Remember for each time interval, there are 3 resampling methods used so the number of items in this list will be equivalent to frequency_list * 3
     - This is used for the decomposition plots, if the season length is unknown or does not exist any number can be put in here and seasonality can be ignored in the decomp plots
 
### Model Fitting Parameters
- testing_set_size: Ex.(50)
     - This will determine the testing set size for the in-sample rolling and out-of-sample rolling forecasts. This is an integer
- order_list_arima: Ex.<br><br>
[(3, 0, 3, 0, 0, 0, 0),  # 'hourly 5 year'<br>
 (3, 1, 1, 0, 0, 0, 0),  # 'hourly 5 year mean'<br>
 (3, 0, 3, 0, 0, 0, 0),  # 'hourly 5 year max'<br>
 (0, 2, 2, 0, 0, 0, 0),  # '6 hour 10 year'<br>
 (0, 1, 1, 0, 0, 0, 0),  # '6 hour 10 year mean'<br>
 (0, 2, 1, 0, 0, 0, 0),  # '6 hour 10 year max'<br>
 (0, 1, 0, 0, 0, 0, 0),  # '12 hour 10 year'<br>
 (1, 0, 3, 0, 0, 0, 0),  # '12 hour 10 year mean'<br>
 (0, 1, 1, 0, 0, 0, 0),  # '12 hour 10 year max'<br>
 (0, 1, 0, 0, 0, 0, 0),  # 'daily 10 year'<br>
 (1, 0, 2, 0, 0, 0, 0),  # 'daily 10 year mean'<br>
 (3, 0, 0, 0, 0, 0, 0)]  # 'daily 10 year max'<br><br>
     - This is a list of tuples, (), and contain the ARIMA(p,d,q) parameters for each dataset, defined by running the above classes and/or the auto-ARIMA class
     - Since this is for running the ARIMA model the last 4 parameters need to be 0.
     - It is recommended to change the comments to represent the data intervals being used
     - The number of tuples here should be frequency_list * 3

- order_list_sarima: Ex.<br><br>
[(2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year'<br>
 (2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year mean'<br>
 (2, 1, 2, 1, 0, 2, 52),  # '7 day 10 year max'<br>
 (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year'<br>
 (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year mean'<br>
 (1, 0, 1, 3, 0, 3, 26),  # '14 day 10 year max'<br>
 (2, 0, 2, 3, 0, 3, 12),  # 'monthly 10 year'<br>
 (2, 0, 2, 3, 0, 3, 12),  # 'monthly 10 year mean'<br>
 (2, 0, 2, 3, 0, 3, 12)]  # 'monthly 10 year max'<br><br>
     - Same as order_list_arima but this is for running the SARIMA(p,d,q)(P,D,Q,m) model and all 7 parameters need to be filled in
 
### Cross Validation Parameters
- groups: Ex.(5)
     - This is the number of cross-validation groups that will be created for the in-sample rolling and out-of-sample rolling forecasts
     - Important! Make sure there are enough rows in the dataframe, for example if groups is set to 5, and the parameter below, testing_set_size_cv is set to 10 then a minumum of 120 rows are required
     - Code snippet for creating CV groups:<br>
        self.training_start = 0<br>
        self.training_end = int((len(df) / self.groups) * n)<br>
        self.in_sample_start = self.training_end + 1<br>
        self.in_sample_end = self.training_end + self.testing_set_size_cv<br>
        self.out_of_sample_start = self.in_sample_end + 1<br>
        self.out_of_sample_end = self.in_sample_end + self.testing_set_size_cv<br>
- testing_set_size_cv: Ex.(10)
     - Testing set size for each in-sample and out-of-sample cross validation group

- keys: Ex.(['1H_mean', '6H_mean', '12H_mean', '1D_mean'])
     - This is a list, [], of strings, '', that contain the key, from the dictionary df_dict_trun created by the module, for each dataframe that needs to be cross validated.
     - The key is simply the 'time interval' + '_' + 'aggregation method', so for the 1 hour resampled dataset aggragated using the mean() function, the key would be '1H_mean'
 
- order_list_trun_arima: Ex.<br><br>
[(3, 1, 2, 0, 0, 0, 0),  # 1H_mean<br>
(0, 1, 1, 0, 0, 0, 0),  # 6H_mean<br>
(1, 0, 3, 0, 0, 0, 0),  # 12H_mean<br>
(1, 1, 0, 0, 0, 0, 0)]  # 1D_mean<br><br>
     - List of tuples, (), which are the ARIMA(p,d,q) parameters for each of the values listed in the keys list above

- order_list_trun_sarima: Ex.<br><br>
[(2, 1, 2, 1, 0, 2, 52),   # 7D_mean
(1, 0, 1, 3, 0, 3, 26),   # 14D_mean
(2, 0, 2, 3, 0, 3, 12)]   # 1M_mean<br><br>
     - List of tuples, (), corresponsing to the SARIMA(p,d,q)(P,D,Q,m) parameters for values in the keys list
 
### Auto ARIMA Parameters
- key: Ex.(['1D_max'])
     - This is a 1 item list which is the key as described in the keys parameter above. This will run through each of the ARIMA or SARIMA paramater combinations defined by the order ranges below.
     - Will only run 1 time series dataframe at a time

#### ARIMA
##### Non-Seasonal Autoregressive Order (p)
- p_min: Ex.(0)
- p_max: Ex.(3)
##### Number of Non-Seasonal Differences (d)
- d_min: Ex.(0)
- d_max: Ex.(2)
##### Non-Seasonal Moving Average Order (q)
- q_min: Ex.(0)
- q_max: Ex.(2)

#### SARIMA
##### Seasonal Autoregressive Order (P)
- P_min: Ex.(0)
- P_max: Ex.(0)
##### Number of Seasonal Differences (D)
- D_min: Ex.(0)
- D_max: Ex.(0)
##### Seasonal Moving Average Order (Q)
- Q_min: Ex.(0)
- Q_max: Ex.(0)
##### Length of the Season (m)
m = [0]

- These are the lower and upper bounds for the order parameters of the ARIMA and SARIMA models
- To run just ARIMA set the values for P,D,Q, and m to 0, set the values to run SARIMA
- It is recommended to analyze the ACF and PACF plots, along with using the Box-Jenkins methodology and analyzing residual diagnostics to narrow the range here to reduce the time complexity of running the auto_ARIMA class


## Directory Structure
The results from running the tsmodule are stored in pdf and jpg files. The program organizes the files into the directory structure as described below to make them easily accessible. Once the program runs it is recommended to archive any of the needed files as they are subject to being overwriten or deleted. The files can be archived elsewhere, but a suggested folder structure is provided inside each of the folders below.

- ##### plots/acf_pacf<br><br>
ACF and PACF plot results for each time interval, contains 3 files for each interval which are the results from using first(), mean(), and max() resampling methods<br><br>
- ##### plots/adfuller_results<br><br>
Provides a table that summarizes the results from running the ADF test on each dataframe. Gives the p_value, adf_value, critical_value, and the number of times the dataset was differenced with the Pandas diff() method<br><br>
- ##### plots/auto_arima_results<br><br>
Table that provides the top 20 results from running the auto_arima class. Lists which model was used, interval, order (p,d,q)(P,D,Q,m), then AIC, RMSE, and MAPE evaluation metric results. Three files are created which sort the models by the following 3 methods: (AIC,MAPE,RMSE), (MAPE,AIC,RMSE), (RMSE,MAPE,AIC).<br><br>
- ##### plots/cv_plots<br><br>
Contains in-sample 1-step-ahead rolling and out-of-sample rolling forecast plots for each of the cross validation groups, for every time series specified in the configuration above.<br><br>
- ##### plots/cv_results<br><br>
Table that congregates all of the cross validaiton results in one place, gives RMSE and MAPE values for in-sample and out-of-sample results.<br><br>
- ##### plots/decomposition<br><br>
Decomposition plots that show the line plot for the data and plots showing trend, seasonal trend, and a plot of residuals. Plots for each time interval / resample method are created.<br><br>
- ##### plots/diagnostic_plots<br>
     - plots/diagnostic_plots/In-Sample Non-Rolling
     - plots/diagnostic_plots/In-Sample Rolling
     - plots/diagnostic_plots/Out-of-Sample Rolling<br><br>
Creates diagnostic plots for each time interval / resample method for in-sample non-rolling, in-sample rolling, and out-of-sample rolling forecasts, organized into separate folders. Two versions are created: one using the plot_diagnostics() method provided by statsmodel package, and another containing a plot of the residuals, histogram, and a quantile-quantile plot. plot_diagnostics() provides all of these and an autocorrelation plot of the residuals.<br><br>
- ##### plots/forecast_plots<br>
     - plots/forecast_plots/In-Sample Non-Rolling
     - plots/forecast_plots/In-Sample Rolling
     - plots/forecast_plots/Out-of-Sample Rolling<br><br>
Similar to the diagnostic_plots folder, this contains forecast results of each time interval/resample method for each of the forecasting methods.<br><br>
- ##### plots/forecast_results<br><br>
Contains 3 files showing forecast results for each forecasting method. Provides AIC, RMSE, and MAPE values.
