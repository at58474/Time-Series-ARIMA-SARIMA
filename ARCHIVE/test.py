from itertools import product

# The lower and upper bounds can be set here for the ARIMA or SARIMA parameters
# To run an ARIMA model set the values of P, D, Q, and m all to 0
# WARNING: the number of models can become very large so use the results from above to narrow the range of each
#          parameter as much as possible.

# Non-Seasonal Autoregressive Order (p)
p_min = 0
p_max = 3
# Number of Non-Seasonal Differences (d)
d_min = 0
d_max = 2
# Non-Seasonal Moving Average Order (q)
q_min = 0
q_max = 3

non_seasonal_params = list(product(range(p_min, p_max + 1), range(d_min, d_max + 1), range(q_min, q_max + 1)))

# Seasonal Autoregressive Order (P)
P_min = 0
P_max = 2
# Number of Seasonal Differences (D)
D_min = 0
D_max = 1
# Seasonal Moving Average Order (Q)
Q_min = 0
Q_max = 2
# Length of the season
m = [4, 12, 24]

seasonal_params = list(product(range(P_min, P_max + 1), range(D_min, D_max + 1), range(Q_min, Q_max + 1), m))


for nsp in non_seasonal_params:
    for sp in seasonal_params:
        print(f'ARIMA {nsp}{sp}')









self.df_auto_results_mape = self.df_auto_results.sort_values(by=['MAPE', 'RMSE', 'AIC'])
self.df_auto_results = self.df_auto_results.head(20)