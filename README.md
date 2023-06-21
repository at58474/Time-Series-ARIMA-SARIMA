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
- pound_filepath:  Ex.('data/raw/pound_river_10yr_cfs_data.txt')
     - file path to the raw data file
- pound_save_filepath:  Ex.('data/pound.csv')
     - file path to save the processed data file
- pound_list_col_del: Ex.(['agency_cd', 'site_no', 'tz_cd', '147720_00060_cd'])
     - A list, [], of strings, '', containing any column names to be removed from the dataframe, leave blank if no columns need to be removed
- pound_dict_col_rename: Ex.({"datetime": "Date", "147720_00060": "Flow"})
     - A dictionary, {'key':'value'}, of columns to be renamed, passing the original name as the key and the desired name as the value, both being strings, ''
##### Process other dataframes if any
- same parameters as dataframe 1

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
