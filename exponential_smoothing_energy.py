# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 19:07:19 2025

@author: avina
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error


df_ec = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Time Series Forecasting using Python/energyconsumption-201002-134452/energy consumption.csv')
df_ec.columns
df_ec.dtypes
df_ec.shape
df_ec.head()
df_ec.describe()


df_ec['DATE'] = pd.to_datetime(df_ec['DATE'], format="%m/%Y")
df_ec.set_index('DATE', inplace=True)


forecast_period = 36 # 36 months 
train = df_ec.iloc[:-forecast_period]  # Use all except the last 36 months for training
test = df_ec.iloc[-forecast_period:]

# Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(train['ENERGY_INDEX']).fit(smoothing_level=0.2, optimized=True)
ses_forecast = ses_model.forecast(forecast_period)

# Holt's Linear Method (Double Exponential Smoothing)
holt_model = ExponentialSmoothing(train['ENERGY_INDEX'], trend='add').fit(
    smoothing_level=0.2, smoothing_slope=0.05, optimized=True)
holt_forecast = holt_model.forecast(forecast_period)

# Holt-Winters Method (Triple Exponential Smoothing)
hw_model = ExponentialSmoothing(train['ENERGY_INDEX'], trend='add', seasonal='add', seasonal_periods=12).fit(
    smoothing_level=0.2, smoothing_slope=0.05, optimized=True)
hw_forecast = hw_model.forecast(forecast_period)

# Compute RMSE
ses_rmse = np.sqrt(mean_squared_error(test['ENERGY_INDEX'], ses_forecast))
holt_rmse = np.sqrt(mean_squared_error(test['ENERGY_INDEX'], holt_forecast))
hw_rmse = np.sqrt(mean_squared_error(test['ENERGY_INDEX'], hw_forecast))

# Print RMSE results
print(f"RMSE - Simple Exponential Smoothing: {ses_rmse:.4f}")
print(f"RMSE - Holt's Linear Method: {holt_rmse:.4f}")
print(f"RMSE - Holt-Winters Method: {hw_rmse:.4f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df_ec.index, df_ec['ENERGY_INDEX'], label="Actual Data", color='b')
plt.plot(test.index, ses_forecast, label="Simple Exp Smoothing", linestyle="dashed", color='r')
plt.plot(test.index, holt_forecast, label="Holt's Method", linestyle="dashed", color='g')
plt.plot(test.index, hw_forecast, label="Holt-Winters", linestyle="dashed", color='purple')

plt.xlabel("Year")
plt.ylabel("Energy Index")
plt.title("Energy Consumption Forecast (Next 3 Years)")
plt.legend()
plt.grid()
plt.show()