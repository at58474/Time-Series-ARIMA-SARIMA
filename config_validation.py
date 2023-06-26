from config import *
import re
import os
import mmap
import warnings
import sys
from pathvalidate import ValidationError, validate_filename


class ValidateInput:

    def __init__(self, parameter_values=None, regex_pattern=None):
        self.parameter_values = parameter_values
        self.regex_pattern = regex_pattern

    def check_parameter_values(self, tf_string):
        return True if tf_string in self.parameter_values else False

    def check_if_date(self, date_string):
        return True if re.match(self.regex_pattern, date_string) else False

    def compare_regex(self, string_to_compare):
        return True if re.match(self.regex_pattern, string_to_compare) else False

    @staticmethod
    def contains_date(date_string, filename):
        # encapsulating open operation in with statement automatically closes the file after block completion
        with open(filename, 'rb', 0) as time_series_file:
            memory_mapped_file = mmap.mmap(time_series_file.fileno(), 0, access=mmap.ACCESS_READ)
            if memory_mapped_file.find(bytes(date_string, 'utf-8')) != -1:
                return True
            else:
                return False

    @staticmethod
    def check_if_filename(filename):
        try:
            validate_filename(filename)
        except ValidationError as error:
            print(f'{error}\n', file=sys.stderr)
            return False
        return True

    @staticmethod
    def check_if_list_of_strings(list_of_strings):
        return isinstance(list_of_strings, list) and all(isinstance(item, str) for item in list_of_strings)

    @staticmethod
    def check_if_dict_of_strings(dict_of_strings):
        return isinstance(dict_of_strings, dict) and all(isinstance(value, str) for value in dict_of_strings.values()) \
            and all(isinstance(dict_key, str) for dict_key in dict_of_strings)

    def check_if_list_of_resamples(self, list_of_resamples):
        return isinstance(list_of_resamples, list) and all(self.compare_regex(item) for item in list_of_resamples)

    @staticmethod
    def check_if_list_of_ints(list_of_ints):
        return isinstance(list_of_ints, list) and all(isinstance(item, int) for item in list_of_ints)

    @staticmethod
    def check_if_list_of_tuples(list_of_tuples):
        return isinstance(list_of_tuples, list) and all(isinstance(item, tuple) for item in list_of_tuples)


# Class Instances
test_true_false = ValidateInput([True, False])
test_model_type = ValidateInput(['ARIMA', 'SARIMA'])
test_date = ValidateInput(None, r'^(19|20)\d\d[- \/.](0[1-9]|1[012])[- \/.](0[1-9]|[12][0-9]|3[01])$')
test_delimiter = ValidateInput([',', '\t', ';', '|', ' '])
test_data_freq = ValidateInput(None, r'\d+(min\b|H\b|D\b|W\b|M\b)')
test_impute_method = ValidateInput(['ffill', 'bfill', 'linear'])
test_filename = ValidateInput()
test_is_list_of_strings = ValidateInput()
test_is_dict_of_strings = ValidateInput()
test_list_of_resamples = ValidateInput(None, r'\d+(min\b|H\b|D\b|W\b|M\b)')
test_is_list_of_ints = ValidateInput()
test_is_list_of_tuples = ValidateInput()

'''
SETTINGS
'''
# run_model_fitting, run_diagnostics, run_cross_validation, run_auto_arima
assert test_true_false.check_parameter_values(run_model_fitting), 'run_model_fitting must be True or False'
assert test_true_false.check_parameter_values(run_diagnostics), 'run_diagnostics must be True or False'
assert test_true_false.check_parameter_values(run_cross_validation), 'run_cross_validation must be True or False'
assert test_true_false.check_parameter_values(run_auto_arima), 'run_auto_arima must be True or False'
# model_type
assert test_model_type.check_parameter_values(model_type), "model_type must be set to 'ARIMA' or 'SARIMA'"
# delete_files
assert test_true_false.check_parameter_values(delete_files), 'delete_files must be True or False'
# archive_files
assert test_true_false.check_parameter_values(archive_files), 'archive_files must be True or False'
# start_date, end_date, pound_filepath, rf_filepath
assert test_date.check_if_date(start_date), 'start_date must be a valid date in the format YYYY-MM-DD'
assert test_date.check_if_date(end_date), 'end_date must be a valid date in the format YYYY-MM-DD'
assert os.path.isfile(f'data/raw/{raw_filename1}'), 'The file specified by pound_filepath does not exist'
assert test_date.contains_date(start_date, f'data/raw/{raw_filename1}'), 'The start date you entered for start_date is not found in the data file, make sure raw data file dates are YYYY-MM-DD'
assert test_date.contains_date(end_date, f'data/raw/{raw_filename1}'), 'The end date you entered for end_date is not found in the data file, make sure raw data file dates are YYYY-MM-DD'
if process_and_merge_2_raw_files:
    assert os.path.isfile(f'data/raw/{raw_filename2}'), 'The file specified by rf_filepath does not exist'
    assert test_date.contains_date(start_date, f'data/raw/{raw_filename2}'), 'The start date you entered for start_date is not found in the data file, make sure raw data file dates are YYYY-MM-DD'
    assert test_date.contains_date(end_date, f'data/raw/{raw_filename2}'), 'The end date you entered for end_date is not found in the data file, make sure raw data file dates are YYYY-MM-DD'
# delimiter
if not test_delimiter.check_parameter_values(delimiter): warnings.warn('The delimiter you entered does not match a standard use case')
# date_col
assert isinstance(date_col, str), 'date_col must be a string'
# data_freq
assert test_data_freq.compare_regex(data_freq), 'data_freq not in correct format, check documentation for proper formatting'
# impute_method
assert test_impute_method.check_parameter_values(impute_method), "impute_method must be set to 'ffill', 'bfill', or 'linear'"
# show, save
assert test_true_false.check_parameter_values(show), 'show must be True or False'
assert test_true_false.check_parameter_values(save), 'save must be True or False'
# col_name
assert isinstance(col_name, str), 'col_name must be a string'

'''
PREPROCESSING PARAMETERS
'''
# process_and_merge_2_raw_files
assert test_true_false.check_parameter_values(process_and_merge_2_raw_files), 'process_and_merge_2_raw_files must be True or False'
# raw_filename1, save_filename1
# this assertion should be redundant since an assertion should already be thrown above
assert test_filename.check_if_filename(raw_filename1), 'raw_filename1 is not a valid file name'
# this also throws an exception with a more specific error description
assert test_filename.check_if_filename(save_filename1), 'save_filename1 is not a valid file name'
# list_col_del1
assert test_is_list_of_strings.check_if_list_of_strings(list_col_del1), 'list_col_del1 is not a list or does not contain only strings'
# dict_col_rename1
assert test_is_dict_of_strings.check_if_dict_of_strings(dict_col_rename1), 'dict_col_rename1 is not a dictionary of keys:values or does not contain only strings'
# raw_filename2, save_filename2
assert test_filename.check_if_filename(raw_filename2), 'raw_filename2 is not a valid file name'
assert test_filename.check_if_filename(save_filename2), 'save_filename2 is not a valid file name'
# list_col_del2
assert test_is_list_of_strings.check_if_list_of_strings(list_col_del2), 'list_col_del2 is not a list or does not contain only strings'
# dict_col_rename2
assert test_is_dict_of_strings.check_if_dict_of_strings(dict_col_rename2), 'dict_col_rename2 is not a dictionary of keys:values or does not contain only strings'
# dict_col
assert test_is_dict_of_strings.check_if_dict_of_strings(dict_col), 'dict_col is not a dictionary of keys:values or does not contain only strings'
# new_col_name
assert isinstance(new_col_name, str), 'new_col_name must be a string'
# merged_filepath
assert test_filename.check_if_filename(merged_filepath), 'merged_filepath is not a valid file name'

'''
CREATE DATAFRAMES PARAMETERS
'''
# frequency_list
assert test_list_of_resamples.check_if_list_of_resamples(frequency_list), 'frequency_list is not a list of resample intervals, see readme for proper format'
# row_cap
assert isinstance(row_cap, int) and row_cap > 0, 'row_cap must be an int greater than 0'
# lags
assert isinstance(lags, int) and lags > 0, 'lags must be an int greater than 0'
# seasonal_period
assert test_is_list_of_ints.check_if_list_of_ints(seasonal_period), 'seasonal_period must be a list of integers'

'''
MODEL FITTING PARAMETERS
'''
if run_model_fitting:
    # testing_set_size
    assert isinstance(testing_set_size, int) and testing_set_size > 0, 'testing_set_size must be an int greater than 0'
    # order_list_arima





# Check if file path exists and/or is in the correct format, for windows, linux, max
# Check if list of strings
# Check if dictionary where key:value pairs are both strings
# Check if list of resample method strings
# Check if int
# Check if list of ints
'''
For seasonal_parameter, have them enter the season and automatically generate the numbers based on the resample time interval
'''
# Check if list of tuples
# Check if list of exactly 1 integer