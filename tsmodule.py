# import pandas library
import pandas as pd
import numpy as np
# import adfuller for ADF test for stationarity
from statsmodels.tsa.stattools import adfuller
# ACF and PACF analyss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
# Fit and Predict with ARIMA
from statsmodels.tsa.arima.model import ARIMA
# Auto ARIMA
import pmdarima as pm
# RMSE
from sklearn.metrics import mean_squared_error
# Jarque-Bera goodness-of-fit test
from statsmodels.stats.stattools import jarque_bera
# For plot
# probplot()
from scipy import stats
import statsmodels.api as sm
# acorr_breusch_godfrey
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
# acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_ljungbox
# shapiro
from scipy.stats import shapiro
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.graphics.gofplots import qqplot
import itertools
import warnings
warnings.filterwarnings("ignore")
import pprint
import random
from dataframe_pdf import *
from itertools import product
import os


class Preprocessing:

    def __init__(self, start_date, end_date, delimiter=',', date_col='Date', data_freq='15min', impute_method='ffill'):
        # Attributes assigned upon initiation of instance
        self.delimiter = delimiter
        self.date_col = date_col
        self.start_date = start_date
        self.end_date = end_date
        self.data_freq = data_freq
        self.impute_method = impute_method

        # Attributes assigned with methods
        self.df = None
        self.interval = None
        self.filepath = None
        self.save_filepath = None
        self.list_col_del = []
        self.dict_col_rename = {}

    def save_df(self, save_filepath):
        self.df.to_csv(save_filepath, index=None)

    # Called from: arima_preprocessing
    def date_to_index(self):
        # Sets the Date column as the index, then drops the original date column, essentially creating a series
        self.df.index = self.df[self.date_col]
        self.df = self.df.drop(self.df.columns[0], axis=1)

    def reindex_impute_df(self):
        # Creates a series of dates in the range and interval that is desired, then reindexes both series/df to insert
        # nan values where the dates are missing so that each series has the same number of rows
        self.interval = pd.date_range(self.start_date, self.end_date, freq=self.data_freq)
        self.df = self.df.reindex(self.interval)
        # imputes data with method provided
        self.df = self.df.fillna(method=self.impute_method, axis=0)

    def index_reset(self):
        # Resets the index and creates new datetime column
        self.df.reset_index(inplace=True)
        self.df = self.df.rename(columns={'index': self.date_col})

    def preprocessing(self, filepath, save_filepath, list_col_del, dict_col_rename):

        self.filepath = filepath
        self.save_filepath = save_filepath
        self.list_col_del = list_col_del
        self.dict_col_rename = dict_col_rename

        # read the text file and store into a dataframe with specified delimiter
        self.df = pd.read_csv(self.filepath, delimiter=self.delimiter)

        # if the file is not a csv then convert it to a csv file and read it to the df
        if self.delimiter != ',':
            # store dataframe into csv file (convert to csv)
            self.save_df(self.save_filepath)
            # read the data and store data in DataFrame
            self.df = pd.read_csv(self.save_filepath)
        else:
            # if read and save filepaths are different then save the file to the new location
            if self.filepath != self.save_filepath:
                # store dataframe into csv file (convert to csv)
                self.save_df(self.save_filepath)

        # Deletes unnecessary columns from provided list
        if self.list_col_del: self.df = self.df.drop(self.list_col_del, axis=1)
        # Renames columns from provided dictionary
        if self.dict_col_rename: self.df = self.df.rename(columns=self.dict_col_rename)

        # Converts Date column from object(string) to datetime
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])

        # Removes any duplicate rows based on the date column, keeps the first instance
        self.df = self.df.drop_duplicates(subset=self.date_col, keep="first")

        # Date column needs to be the index to reindex the dataframe
        self.date_to_index()

        # Corrects any faults such as incorrect timestamps, missing data, then imputes the data
        self.reindex_impute_df()

        # Resets the index
        self.index_reset()

        self.save_df(self.save_filepath)

        return self.df

    def combine_cols(self, new_col_name, col1, col2):
        self.df[new_col_name] = self.df[col1] + self.df[col2]
        self.df = self.df.drop([col1, col2], axis=1)

    def merge_dfs(self, df1, df2, dict_col, new_col_name, merged_filepath):
        # Merges the 2 series into a single dataframe
        self.df = pd.merge_asof(df1, df2, on=self.date_col)
        # Renames the merged columns to something useful
        self.df = self.df.rename(columns=dict_col)

        # After merging the dataframes this adds the values together into a new column
        i = iter(dict_col.values())
        col1, col2 = next(i), next(i)
        self.combine_cols(new_col_name, col1, col2)

        self.save_df(merged_filepath)

        return self.df


class CreateDataframes:

    def __init__(self, df, frequency_list, date_col):
        # Attributes assigned upon initiation of instance
        self.df = df
        self.freq_list = frequency_list
        self.date_col = date_col

        # Attributes assigned with methods
        self.df_dict = {}

    def date_to_index(self):
        # Sets the Date column as the index, then drops the original date column, essentially creating a series
        self.df.index = self.df[self.date_col]
        self.df = self.df.drop(self.df.columns[0], axis=1)

    def resample_df(self, interval):
        # Resample the entire dataframe, original dataframe has datapoints every 15 minutes. This changes the
        #   time interval frequency and returns 3 dataframes based on resample method, first, mean, and max
        # The dictionary key is generated from the interval then the resample method
        self.df_dict[f'{interval}_first'] = self.df.resample(interval).first()
        # If the passed frequency is the same as the original dataframe do not need to resample more than once
        if len(self.df.index) != len(self.df_dict[f'{interval}_first']):
            self.df_dict[f'{interval}_mean'] = self.df.resample(interval).mean()
            self.df_dict[f'{interval}_max'] = self.df.resample(interval).max()

    def create_dfs(self):
        # Loops through the passed list of frequencies and resamples the original dataframe based on intervals
        self.date_to_index()

        for d in self.freq_list:
            self.resample_df(d)

        return self.df_dict

    def set_row_cap(self, row_cap):
        for name, df in self.df_dict.items():
            if len(df.index) > row_cap:
                self.df_dict[name] = df.tail(row_cap)
        return self.df_dict


class ParameterEstimation:
    def __init__(self, df_dict):
        # Attributes assigned upon initiation of instance
        self.df_dict = df_dict

        # Attributes assigned with methods
        self.diff_dict = {}
        self.df_stationarity = pd.DataFrame(columns=['time_series',
                                                     'p_value',
                                                     'adf_value',
                                                     'critical_value',
                                                     'times_differenced'])
        self.adf_value = None
        self.p_value = None
        self.critical_value = None

    def run_adfuller(self, df):
        # Pass the dataframe to the adfuller method provided by statsmodels module
        adf_test = adfuller(df)
        # Store results of adfuller test in variables
        self.adf_value = round(adf_test[0], 2)
        self.p_value = adf_test[1]
        self.critical_value = round(adf_test[4]['5%'], 2)

    # This simply adds a row to the df_stationarity dataframe which just stores the results of the stationarity_check
    #   and lists what the diff_dict contains
    def add_row(self, title, differenced):
        new_row_dict = {'time_series': title,
                        'p_value': self.p_value,
                        'adf_value': self.adf_value,
                        'critical_value': self.critical_value,
                        'times_differenced': differenced}
        new_row_df = pd.DataFrame([new_row_dict])

        self.df_stationarity = pd.concat([self.df_stationarity, new_row_df])

    # This creates a new dictionary containing a dictionary of all the dataframes after differencing has been applied
    #   to the dataframes that were not stationary, if any.
    def run_stationarity(self, df, title):
        # Run adfuller test
        self.run_adfuller(df)

        # This stores the number of times the dataframe is differenced
        differenced = 0

        # If the data is already stationary then add to the stationarity results dataframe
        if (self.p_value < 0.05) and (self.adf_value < self.critical_value):
            self.add_row(title, differenced)

        # If the data is not stationary, then diff() the dataset until it becomes stationary, then add to results df
        else:
            while (self.p_value >= 0.05) and (self.adf_value >= self.critical_value):
                # Passes the dataframe to Pandas diff() method, which returns the differenced dataset. This creates
                #   an empty row which needs to be dropped, lose 1 row.
                df = df.diff().dropna()
                # Increase differenced counter by 1
                differenced += 1
                # Run adfuller test again
                self.run_adfuller(df)
            self.add_row(title, differenced)

        self.diff_dict[title] = df

    def stationarity_check(self, show, save):
        for df in self.df_dict:
            self.run_stationarity(self.df_dict[df], df)

        # Set time_series column as the index
        self.df_stationarity.index = self.df_stationarity['time_series']
        self.df_stationarity = self.df_stationarity.drop(self.df_stationarity.columns[0], axis=1)

        if show: print(self.df_stationarity)
        if save: dataframe_to_pdf(self.df_stationarity, f'plots/adfuller_results/adfuller_results.pdf')

        return self.diff_dict, self.df_stationarity

    @staticmethod
    def plot_acf_pacf(df, title, lags, col_name, show, save):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=80)
        plot_acf(df[col_name], ax=ax1, lags=lags, title=f'Autocorrelation for {title} RFG flow data')
        plot_pacf(df[col_name], ax=ax2, lags=lags, title=f'Partial Autocorrelation for {title} RFG flow data')

        # Decorate
        # lighten the borders
        ax1.spines["top"].set_alpha(.3)
        ax2.spines["top"].set_alpha(.3)
        ax1.spines["bottom"].set_alpha(.3)
        ax2.spines["bottom"].set_alpha(.3)
        ax1.spines["right"].set_alpha(.3)
        ax2.spines["right"].set_alpha(.3)
        ax1.spines["left"].set_alpha(.3)
        ax2.spines["left"].set_alpha(.3)

        # font size of tick labels
        ax1.tick_params(axis='both', labelsize=12)
        ax2.tick_params(axis='both', labelsize=12)
        if show: plt.show()
        if save: plt.savefig(f'plots/acf_pacf/{title}_acf_pacf.jpg')

    def autocorr_analysis(self, col_name, show=False, save=True, lags=50):
        for df in self.diff_dict:
            self.plot_acf_pacf(self.diff_dict[df], df, lags, col_name, show, save)

    @staticmethod
    def season_decomp(df, title, seasonal_period, show, save):
        decomposition = seasonal_decompose(df, model='additive', period=seasonal_period)
        fig = decomposition.plot()

        if show: plt.show()
        if save: plt.savefig(f'plots/decomposition/{title}_decomposition_plots.jpg')

    def plots(self, seasonal_period, show=True, save=True):
        for df, p in zip(self.df_dict, seasonal_period):
            self.season_decomp(self.df_dict[df], df, p, show, save)


class ModelFitting:

    def __init__(self, df_dict, testing_set_size, order_list, col_name, model_type):
        # Initiation
        self.df_dict = df_dict
        self.testing_set_size = testing_set_size
        self.order_list = order_list
        self.col_name = col_name
        self.model_type = model_type

        # Methods
        self.model = None
        self.model_f = None
        self.fit = None
        self.fit_f = None
        self.prediction = None
        self.forecast = None
        self.aic = None
        self.df_arima_results = pd.DataFrame(columns=['Interval',
                                                      'Model',
                                                      'pdq',
                                                      'Size_Testing',
                                                      'AIC',
                                                      'RMSE_Forecast',
                                                      'RMSE_Predict',
                                                      'MAPE_Forecast',
                                                      'MAPE_Predict'])
        self.train = None
        self.test = None
        self.history = None
        self.history_f = None
        self.predictions_p = list()
        self.predictions_f = list()
        self.dict_results_ISNR = {}
        self.dict_results_ISR = {}
        self.dict_results_OOSR = {}
        self.non_seasonal_order = None
        self.seasonal_order = None
        self.hello = 0

    def create_results_dict(self, df, results_list):
        if results_list[0] == 'In_Sample_Non_Rolling':
            self.dict_results_ISNR[df] = results_list

        elif results_list[0] == 'In_Sample_Rolling':
            self.dict_results_ISR[df] = results_list

        elif results_list[0] == 'Out_Of_Sample_Rolling':
            self.dict_results_OOSR[df] = results_list

    def extract_order_params(self, order_params):
        self.non_seasonal_order = order_params[0:3]
        self.seasonal_order = order_params[3:7]

    def sarimax_modeling(self, order_params, df, forecast_type):
        self.hello += 1
        print(f'#{self.hello} dataframa {df} order {order_params} forecast_type {forecast_type}')
        if forecast_type == 'In_Sample_Non_Rolling':
            self.extract_order_params(order_params)

            self.model = sm.tsa.statespace.SARIMAX(self.df_dict[df], order=self.non_seasonal_order, seasonal_order=self.seasonal_order, enforce_stationarity=False)

            self.fit = self.model.fit(disp=0)

            self.prediction = self.fit.predict(type='levels')
            results_list = [forecast_type, order_params, self.model, self.fit, self.prediction]
            self.create_results_dict(df, results_list)

        elif forecast_type == 'In_Sample_Rolling':
            self.extract_order_params(order_params)

            self.model = sm.tsa.statespace.SARIMAX(self.history, order=self.non_seasonal_order, seasonal_order=self.seasonal_order, enforce_stationarity=False)

            self.fit = self.model.fit(disp=0)

            self.prediction = self.fit.predict(start=len(self.history), end=len(self.history))[0]
            self.predictions_p.append(self.prediction)

            self.forecast = self.fit.forecast()[0]
            self.predictions_f.append(self.forecast)

        elif forecast_type == 'Out_Of_Sample_Rolling':
            self.extract_order_params(order_params)

            self.model = sm.tsa.statespace.SARIMAX(self.history, order=self.non_seasonal_order, seasonal_order=self.seasonal_order, enforce_stationarity=False)
            self.model_f = sm.tsa.statespace.SARIMAX(self.history_f, order=self.non_seasonal_order, seasonal_order=self.seasonal_order, enforce_stationarity=False)

            self.fit = self.model.fit(disp=0)
            self.fit_f = self.model_f.fit(disp=0)

            self.prediction = self.fit.predict(start=len(self.history), end=len(self.history))[0]
            self.predictions_p.append(self.prediction)

            self.forecast = self.fit_f.forecast()[0]
            self.predictions_f.append(self.forecast)

    def calc_aic(self):
        self.aic = int(self.fit.aic)

    @staticmethod
    def calc_rmse(df, pred):
        rmse = int(mean_squared_error(df, pred, squared=False))
        return rmse

    @staticmethod
    def calc_mape(df, pred):
        mape = int((mean_absolute_percentage_error(df, pred)) * 100)
        return mape

    def add_row(self, new_row_dict):
        new_row_df = pd.DataFrame([new_row_dict])
        self.df_arima_results = pd.concat([self.df_arima_results, new_row_df])

    def plot_forecast(self, show, save, df, order, forecast_type, metrics_list=None):
        plt.figure(figsize=(16, 8))
        if forecast_type == 'In-Sample Non-Rolling':
            plt.plot(self.df_dict[df], label="Observed")
            plt.plot(self.prediction, label="Predicted")
            plt.title(f'ARIMA({order}) {forecast_type} Forecasts for {df} with {len(self.df_dict[df].index)} rows \n '
                      f'AIC:{self.aic} RMSE:{metrics_list[0]} MAPE:{metrics_list[1]}')
        else:
            predictions_series_p = pd.Series(self.predictions_p, index=self.test.index)
            predictions_series_f = pd.Series(self.predictions_f, index=self.test.index)
            plt.plot(self.test, label='Observations')
            plt.plot(predictions_series_p, color='green', label='Predicted Values')
            plt.plot(predictions_series_f, color='red', label='Forecasted Values')
            plt.title(f'ARIMA({order}) {forecast_type} Forecasts for {df} with {len(self.df_dict[df].index)} rows \n '
                      f'Predict :: AIC:{self.aic} RMSE:{metrics_list[0]} MAPE:{metrics_list[2]} \n'
                      f'Forecast :: AIC:{self.aic} RMSE:{metrics_list[1]} MAPE:{metrics_list[3]}')

        plt.ylabel(self.col_name, fontsize=16)
        plt.legend(loc="upper left")

        if show: plt.show()
        if save: plt.savefig(f'plots/forecast_plots/{forecast_type}/{df}_forecast_plot.jpg')

    def results(self, show, save, forecast_type):
        if show: print(self.df_arima_results)
        if save: dataframe_to_pdf(self.df_arima_results, f'plots/forecast_results/{forecast_type}_results.pdf')

    def train_test_split(self, df):
        size = int(len(df) - self.testing_set_size)
        self.train, self.test = df[self.col_name][0:size], df[self.col_name][size:len(df)]

    def populate_history(self, out_of_sample=False):
        self.history = [x for x in self.train]
        if out_of_sample:
            self.history_f = [x for x in self.train]

    def clear_results_df(self):
        self.df_arima_results = self.df_arima_results[0:0]

    def arima_in_sample_non_rolling(self, show=False, save=True):
        for df, order in zip(self.df_dict, self.order_list):
            self.sarimax_modeling(order, df, 'In_Sample_Non_Rolling')
            self.calc_aic()
            rmse = self.calc_rmse(self.df_dict[df], self.prediction)
            mape = self.calc_mape(self.df_dict[df], self.prediction)
            metrics_list = [rmse, mape]
            new_row_dict = {'Interval': df,
                            'Model': 'In-Sample Non-Rolling',
                            'pdq': order,
                            'Size_Testing': self.testing_set_size,
                            'AIC': self.aic,
                            'RMSE_Forecast': 'N/A',
                            'RMSE_Predict': rmse,
                            'MAPE_Forecast': 'N/A',
                            'MAPE_Predict': mape}
            self.add_row(new_row_dict)
            self.plot_forecast(show, save, df, order, 'In-Sample Non-Rolling', metrics_list)

        self.results(show, save, 'In-Sample Non-Rolling')

        return self.dict_results_ISNR

    def arima_in_sample_rolling(self, show=False, save=True):
        self.clear_results_df()
        for df, order in zip(self.df_dict, self.order_list):
            self.train_test_split(self.df_dict[df])
            self.populate_history()
            self.predictions_p = []
            self.predictions_f = []

            for t in range(len(self.test)):
                self.sarimax_modeling(order, df, 'In_Sample_Rolling')
                self.history.append(self.test[t])

            results_list = ['In_Sample_Rolling', order, self.model, self.fit, self.predictions_p, self.predictions_f]
            self.create_results_dict(df, results_list)

            self.calc_aic()
            rmse_p = self.calc_rmse(self.test, self.predictions_p)
            rmse_f = self.calc_rmse(self.test, self.predictions_f)
            mape_p = self.calc_mape(self.test, self.predictions_p)
            mape_f = self.calc_mape(self.test, self.predictions_f)
            metrics_list = [rmse_p, rmse_f, mape_p, mape_f]

            new_row_dict = {'Interval': df,
                            'Model': 'In-Sample Rolling',
                            'pdq': order,
                            'Size_Testing': self.testing_set_size,
                            'AIC': self.aic,
                            'RMSE_Predict': rmse_p,
                            'RMSE_Forecast': rmse_f,
                            'MAPE_Predict': mape_p,
                            'MAPE_Forecast': mape_f}
            self.add_row(new_row_dict)
            self.plot_forecast(show, save, df, order, 'In-Sample Rolling', metrics_list)

        self.results(show, save, 'In-Sample Rolling')

        return self.dict_results_ISR

    def arima_out_of_sample_rolling(self, show=False, save=True):
        self.clear_results_df()
        for df, order in zip(self.df_dict, self.order_list):
            self.train_test_split(self.df_dict[df])
            self.populate_history(True)
            self.predictions_p = []
            self.predictions_f = []

            for t in range(len(self.test)):
                self.sarimax_modeling(order, df, 'Out_Of_Sample_Rolling')
                # This is the big difference in out-of-sample: here the fitted values are added to history, where in the
                # in-sample values from the test set are added here
                self.history.append(self.prediction)
                self.history_f.append(self.forecast)

            results_list = ['Out_Of_Sample_Rolling', order, self.model, self.model_f, self.fit, self.fit_f, self.predictions_p, self.predictions_f]
            self.create_results_dict(df, results_list)

            self.calc_aic()
            rmse_p = self.calc_rmse(self.test, self.predictions_p)
            rmse_f = self.calc_rmse(self.test, self.predictions_f)
            mape_p = self.calc_mape(self.test, self.predictions_p)
            mape_f = self.calc_mape(self.test, self.predictions_f)
            metrics_list = [rmse_p, rmse_f, mape_p, mape_f]

            new_row_dict = {'Interval': df,
                            'Model': 'Out-of-Sample Rolling',
                            'pdq': order,
                            'Size_Testing': self.testing_set_size,
                            'AIC': self.aic,
                            'RMSE_Predict': rmse_p,
                            'RMSE_Forecast': rmse_f,
                            'MAPE_Predict': mape_p,
                            'MAPE_Forecast': mape_f}
            self.add_row(new_row_dict)
            self.plot_forecast(show, save, df, order, 'Out-of-Sample Rolling', metrics_list)

        self.results(show, save, 'Out-of-Sample Rolling')

        return self.dict_results_OOSR


class Diagnostics:

    def __init__(self, df_dict, testing_set_size, col_name, dict_results_isnr, dict_results_isr, dict_results_oosr):
        # Initiation
        self.df_dict = df_dict
        self.testing_set_size = testing_set_size
        self.col_name = col_name
        self.dict_results_isnr = dict_results_isnr
        self.dict_results_isr = dict_results_isr
        self.dict_results_oosr = dict_results_oosr

        # Methods
        self.train = None
        self.test = None
        self.predictions = None
        self.forecasts = None
        self.order_params = None
        self.fitted_results = None
        self.fitted_results_f = None

    def train_test_split(self, df):
        size = int(len(df) - self.testing_set_size)
        self.train, self.test = df[self.col_name][0:size], df[self.col_name][size:len(df)]

    def plot_diagnostics_custom(self, df, predictions, forecast_type, title, show, save):
        residuals = pd.DataFrame([df[i]-predictions[i] for i in range(len(predictions))])

        fig = plt.figure(figsize=(12, 7))
        layout = (2, 2)
        res_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        his_ax = plt.subplot2grid(layout, (1, 0))
        qq_ax = plt.subplot2grid(layout, (1, 1))

        residuals.plot(ax=res_ax)
        res_ax.set_title(f'ARIMA({self.order_params}) {forecast_type} Residuals for {title}')
        residuals.hist(ax=his_ax)
        his_ax.set_title(f'ARIMA({self.order_params}) {forecast_type} Histogram for {title}')
        qqplot(residuals, ax=qq_ax)
        qq_ax.set_title(f'ARIMA({self.order_params}) {forecast_type} Q-Q Plot for {title}')

        if show: plt.tight_layout()
        if save: plt.savefig(f'plots/diagnostic_plots/{forecast_type}{title}_diagnostic_plots_custom.jpg')

    @staticmethod
    def call_plot_diagnostics(fitted_results, forecast_type, title, show, save):
        fitted_results.plot_diagnostics(figsize=(15, 12))

        if show: plt.show()
        if save: plt.savefig(f'plots/diagnostic_plots/{forecast_type}/plot_diagnostics/{title}_diagnostic_plots.jpg')

    def residual_analysis(self, show=False, save=True):
        for df, res in zip(self.df_dict, self.dict_results_isnr):
            self.predictions = self.dict_results_isnr[res][4].tolist()
            self.order_params = self.dict_results_isnr[res][1]
            self.fitted_results = self.dict_results_isnr[res][3]
            self.plot_diagnostics_custom(self.df_dict[df][self.col_name].tolist(), self.predictions, 'In-Sample Non-Rolling/custom/', res, show, save)
            self.call_plot_diagnostics(self.fitted_results, 'In-Sample Non-Rolling', res, show, save)

        for df, res in zip(self.df_dict, self.dict_results_isr):
            self.predictions = self.dict_results_isr[res][4]
            self.forecasts = self.dict_results_isr[res][5]
            self.order_params = self.dict_results_isr[res][1]
            self.fitted_results = self.dict_results_isr[res][3]
            self.plot_diagnostics_custom(self.df_dict[df][self.col_name].tolist(), self.predictions, 'In-Sample Rolling/custom/predict/', res, show, save)
            self.plot_diagnostics_custom(self.df_dict[df][self.col_name].tolist(), self.forecasts, 'In-Sample Rolling/custom/forecast/', res, show, save)
            self.call_plot_diagnostics(self.fitted_results, 'In-Sample Rolling', res, show, save)

        for df, res in zip(self.df_dict, self.dict_results_oosr):
            self.predictions = self.dict_results_oosr[res][6]
            self.forecasts = self.dict_results_oosr[res][7]
            self.order_params = self.dict_results_oosr[res][1]
            self.fitted_results = self.dict_results_oosr[res][4]
            self.fitted_results_f = self.dict_results_oosr[res][5]
            self.plot_diagnostics_custom(self.df_dict[df][self.col_name].tolist(), self.predictions, 'Out-of-Sample Rolling/custom/predict/', res, show, save)
            self.plot_diagnostics_custom(self.df_dict[df][self.col_name].tolist(), self.forecasts, 'Out-of-Sample Rolling/custom/forecast/', res, show, save)
            self.call_plot_diagnostics(self.fitted_results, 'Out-of-Sample Rolling', f'predict/{res}', show, save)
            self.call_plot_diagnostics(self.fitted_results_f, 'Out-of-Sample Rolling', f'forecast/{res}', show, save)


class CrossValidation:

    def __init__(self, df_dict, col_name, dict_results_isr, dict_results_oosr, groups, order_list, testing_set_size_cv, model_type):
        # Initiation
        self.df_dict = df_dict
        self.col_name = col_name
        self.dict_results_isr = dict_results_isr
        self.dict_results_oosr = dict_results_oosr
        self.groups = groups + 1
        self.order_list = order_list
        self.testing_set_size_cv = testing_set_size_cv
        self.model_type = model_type

        # Methods
        self.cv_group_dict = {}
        self.df_cv_results = pd.DataFrame(columns=['Interval',
                                                   'Group',
                                                   'pdq',
                                                   'Size_Testing',
                                                   'RMSE_In_Sample',
                                                   'MAPE_In_Sample',
                                                   'RMSE_Out_of_Sample',
                                                   'MAPE_Out_of_Sample'])
        self.training_start = None
        self.training_end = None
        self.in_sample_start = None
        self.in_sample_end = None
        self.out_of_sample_start = None
        self.out_of_sample_end = None
        self.model = None
        self.fit = None
        self.forecast = None
        self.predictions_in_sample = []
        self.predictions_out_of_sample = []
        self.history = []
        self.train = None
        self.in_sample = None
        self.out_of_sample = None
        self.observations = None
        self.rmse_in_sample = None
        self.rmse_out_of_sample = None
        self.mape_in_sample = None
        self.mape_out_of_sample = None
        self.non_seasonal_order = None
        self.seasonal_order = None
        self.hello = 0

    def build_cv_groups(self, df, n):
        self.training_start = 0
        self.training_end = int((len(df) / self.groups) * n)
        self.in_sample_start = self.training_end + 1
        self.in_sample_end = self.training_end + self.testing_set_size_cv
        self.out_of_sample_start = self.in_sample_end + 1
        self.out_of_sample_end = self.in_sample_end + self.testing_set_size_cv

    def extract_order_params(self, order_params):
        self.non_seasonal_order = order_params[0:3]
        self.seasonal_order = order_params[3:7]

    def sarimax_modeling(self, forecast_type, order_params, t):
        self.extract_order_params(order_params)
        self.model = sm.tsa.statespace.SARIMAX(self.history, order=self.non_seasonal_order, seasonal_order=self.seasonal_order, enforce_stationarity=False)
        self.fit = self.model.fit(disp=0)
        self.forecast = self.fit.forecast()[0]
        if forecast_type == 'in':
            self.predictions_in_sample.append(self.forecast)
            self.history.append(self.in_sample[t])
        elif forecast_type == 'out':
            self.predictions_out_of_sample.append(self.forecast)
            self.history.append(self.forecast)

    def populate_lists(self, df):
        self.observations = df[self.col_name][self.in_sample_start:self.out_of_sample_end]

        self.train = df[self.col_name][self.training_start:self.training_end]
        self.in_sample = df[self.col_name][self.in_sample_start:self.in_sample_end]
        self.out_of_sample = df[self.col_name][self.out_of_sample_start:self.out_of_sample_end]

        self.history = [x for x in self.train]

    def clear_lists(self):
        self.predictions_in_sample = []
        self.predictions_out_of_sample = []

    def eval_metrics(self):
        self.rmse_in_sample = int(mean_squared_error(self.in_sample, self.predictions_in_sample, squared=False))
        self.rmse_out_of_sample = int(mean_squared_error(self.out_of_sample, self.predictions_out_of_sample, squared=False))
        self.mape_in_sample = int((mean_absolute_percentage_error(self.in_sample, self.predictions_in_sample)) * 100)
        self.mape_out_of_sample = int((mean_absolute_percentage_error(self.out_of_sample, self.predictions_out_of_sample)) * 100)

    def add_row(self, new_row_dict):
        new_row_df = pd.DataFrame([new_row_dict])
        self.df_cv_results = pd.concat([self.df_cv_results, new_row_df])

    def results(self, show=False, save=True):
        if show: print(self.df_cv_results)
        if save: dataframe_to_pdf(self.df_cv_results, f'plots/cv_results/cross_validation_results.pdf')

    def plot_cross_validation(self, title, order_params, n, show, save):
        # Converts the prediction lists into pandas series for plotting
        predictions_series_validate = pd.Series(self.predictions_in_sample, index=self.in_sample.index)
        predictions_series_test = pd.Series(self.predictions_out_of_sample, index=self.out_of_sample.index)

        # Plot
        plt.figure(figsize=(16, 8))
        plt.plot(self.observations, label='Observations')
        plt.plot(predictions_series_validate, color='red', label='In-Sample Forecasts')
        plt.plot(predictions_series_test, color='green', label='Out-of_Sample Forecasts')
        plt.legend(loc="upper left")
        plt.title(f'ARIMA{order_params} Cross Validation for {title} \n'
                  f'In-Sample Results :: RMSE:{self.rmse_in_sample} MAPE:{self.mape_in_sample} \n'
                  f'Out-of_Sample Results :: RMSE:{self.rmse_out_of_sample} MAPE:{self.mape_out_of_sample}')

        if show: plt.show()
        if save: plt.savefig(f'plots/cv_plots/{title}_group{n}_cv_plot.jpg')

    def cv_modeling(self, df, title, order_params, n, show, save):
        self.hello += 1
        print(f'#{self.hello} dataframa {title} order {order_params}')

        self.populate_lists(df)

        self.clear_lists()

        # In-Sample Forecasting
        for t in range(len(self.in_sample)):
            self.sarimax_modeling('in', order_params, t)

        # Out-of-Sample Forecasting
        for t in range(len(self.out_of_sample)):
            self.sarimax_modeling('out', order_params, t)

        # Model Evaluation Metrics RMSE and MAPE
        self.eval_metrics()

        # Create Results Dataframe and create pdf
        new_row_dict = {'Interval': title,
                        'Group': n,
                        'pdq': order_params,
                        'Size_Testing': self.testing_set_size_cv,
                        'RMSE_In_Sample': self.rmse_in_sample,
                        'MAPE_In_Sample': self.mape_in_sample,
                        'RMSE_Out_of_Sample': self.rmse_out_of_sample,
                        'MAPE_Out_of_Sample': self.mape_out_of_sample}
        self.add_row(new_row_dict)

        # Converts predictions to a series then plots cross validation results
        self.plot_cross_validation(title, order_params, n, show, save)

    def cross_validation(self, show=False, save=True):
        for df, order in zip(self.df_dict, self.order_list):
            for n in range(1, self.groups):
                self.build_cv_groups(self.df_dict[df], n)
                self.cv_modeling(self.df_dict[df], df, order, n, show, save)

        self.results(show, save)


class AutoARIMA:

    def __init__(self, df_dict_auto, non_seasonal_params, seasonal_params, col_name, model_type):
        # Initiation
        self.df_dict_auto = df_dict_auto
        self.non_seasonal_params = non_seasonal_params
        self.seasonal_params = seasonal_params
        self.col_name = col_name
        self.model_type = model_type
        # Methods
        self.order_params = ()
        self.aic = None
        self.rmse = None
        self.mape = None
        self.df_auto_results = pd.DataFrame(columns=['Model',
                                                     'Interval',
                                                     'Order',
                                                     'AIC',
                                                     'RMSE',
                                                     'MAPE'])
        self.df_auto_results_mape = pd.DataFrame(columns=['Model',
                                                          'Interval',
                                                          'Order',
                                                          'AIC',
                                                          'RMSE',
                                                          'MAPE'])
        self.df_auto_results_aic = pd.DataFrame(columns=['Model',
                                                         'Interval',
                                                         'Order',
                                                         'AIC',
                                                         'RMSE',
                                                         'MAPE'])
        self.df_auto_results_rmse = pd.DataFrame(columns=['Model',
                                                          'Interval',
                                                          'Order',
                                                          'AIC',
                                                          'RMSE',
                                                          'MAPE'])

    def pack_order_params(self, nsp, sp):
        self.order_params = nsp + sp

    def add_row(self, nsp, sp):
        new_row_dict = {'Model': self.model_type,
                        'Interval': next(iter(self.df_dict_auto)),
                        'Order': f'{nsp}{sp}',
                        'AIC': self.aic,
                        'RMSE': self.rmse,
                        'MAPE': self.mape}
        new_row_df = pd.DataFrame([new_row_dict])
        self.df_auto_results = pd.concat([self.df_auto_results, new_row_df])

    def results(self, show, save):
        self.df_auto_results_mape = self.df_auto_results.sort_values(by=['MAPE', 'RMSE', 'AIC']).head(20)
        self.df_auto_results_aic = self.df_auto_results.sort_values(by=['AIC', 'MAPE', 'RMSE']).head(20)
        self.df_auto_results_rmse = self.df_auto_results.sort_values(by=['RMSE', 'MAPE', 'AIC']).head(20)

        if show:
            print(self.df_auto_results_mape)
            print(self.df_auto_results_aic)
            print(self.df_auto_results_rmse)
        if save:
            dataframe_to_pdf(self.df_auto_results_mape, f'plots/auto_arima_results/auto_{self.model_type}_results_mape_{next(iter(self.df_dict_auto))}.pdf')
            dataframe_to_pdf(self.df_auto_results_aic, f'plots/auto_arima_results/auto_{self.model_type}_results_aic_{next(iter(self.df_dict_auto))}.pdf')
            dataframe_to_pdf(self.df_auto_results_rmse, f'plots/auto_arima_results/auto_{self.model_type}_results_rmse_{next(iter(self.df_dict_auto))}.pdf')

    def auto_arima(self, show, save):
        # Used for printing progress
        total = len(list(product(self.non_seasonal_params, self.seasonal_params)))

        # Similar to a nested for loop iterating through each combo
        for progress, (nsp, sp) in enumerate(product(self.non_seasonal_params, self.seasonal_params), 1):
            print(f'{int((progress / total) * 100)}%', nsp, sp)

            # Converts 2 tuples into 1 tuple since ModelFitting requires parameters be loaded in 1 tuple
            self.pack_order_params(nsp, sp)
            # Creates ModelFitting object and uses its methods and attributes to fit and calculate metrics
            model_fitting_object = ModelFitting(self.df_dict_auto, 0, self.order_params, self.col_name, self.model_type)
            model_fitting_object.sarimax_modeling(self.order_params, next(iter(self.df_dict_auto)), 'In_Sample_Non_Rolling')
            model_fitting_object.calc_aic()
            self.aic = model_fitting_object.aic
            self.rmse = model_fitting_object.calc_rmse(model_fitting_object.df_dict[next(iter(self.df_dict_auto))], model_fitting_object.prediction)
            self.mape = model_fitting_object.calc_mape(model_fitting_object.df_dict[next(iter(self.df_dict_auto))], model_fitting_object.prediction)
            self.add_row(nsp, sp)

        self.results(show, save)


class FileHandling:

    def __init__(self):
        # Initiation

        # Methods
        self.file_list = []
        self.current_dir = None
        self.directory_dict = {}

    def get_file_list(self):
        self.file_list = [file for file in os.listdir(self.current_dir) if os.path.isfile(f'{self.current_dir}/{file}')]
        print(self.file_list)

    def build_dir_structure(self):
        self.directory_dict['cv_plots'] = 'plots/cv_plots'
        self.directory_dict['cv_result'] = 'plots/cv_results'
        self.directory_dict['diag_plots_isnr_cust'] = 'plots/diagnostic_plots/In-Sample Non-Rolling/custom'
        self.directory_dict['diag_plots_isnr'] = 'plots/diagnostic_plots/In-Sample Non-Rolling/plot_diagnostics'
        self.directory_dict['diag_plots_isr_cust_f'] = 'plots/diagnostic_plots/In-Sample Rolling/custom/forecast'
        self.directory_dict['diag_plots_isr_cust_p'] = 'plots/diagnostic_plots/In-Sample Rolling/custom/predict'
        self.directory_dict['diag_plots_isr'] = 'plots/diagnostic_plots/In-Sample Rolling/plot_diagnostics'
        self.directory_dict['diag_plots_oosr_cust_f'] = 'plots/diagnostic_plots/Out-of-Sample Rolling/custom/forecast'
        self.directory_dict['diag_plots_oosr_cust_p'] = 'plots/diagnostic_plots/Out-of-Sample Rolling/custom/predict'
        self.directory_dict['diag_plots_oosr_f'] = 'plots/diagnostic_plots/Out-of-Sample Rolling/plot_diagnostics/forecast'
        self.directory_dict['diag_plots_oosr_p'] = 'plots/diagnostic_plots/Out-of-Sample Rolling/plot_diagnostics/predict'
        self.directory_dict['forecast_plots_isnr'] = 'plots/forecast_plots/In-Sample Non-Rolling'
        self.directory_dict['forecast_plots_isr'] = 'plots/forecast_plots/In-Sample Rolling'
        self.directory_dict['forecast_plots_oosr'] = 'plots/forecast_plots/Out-of-Sample Rolling'
        self.directory_dict['forecast_results'] = 'plots/forecast_results'

    def delete(self):
        for file in self.file_list:
            os.remove(f'{self.current_dir}/{file}')

    def delete_files(self):
        print('!!!!!REMOVING!!!!!')
        self.build_dir_structure()
        for d in self.directory_dict:
            self.current_dir = self.directory_dict[d]
            self.get_file_list()
            self.delete()
